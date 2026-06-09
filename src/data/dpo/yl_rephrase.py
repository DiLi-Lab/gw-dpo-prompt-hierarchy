"""Rephrase broken y_l / y_w fields using the Claude API.

Provides rephrase_yl() and rephrase_yw() for fixing DPO instances whose
rejected (y_l) or chosen (y_w) fields contain refusals or role-mismatch
language.
"""

import logging
import re
from difflib import SequenceMatcher

from src.data.dpo.response_utils import is_refusal

RESP_START = "<|RESP_START|>"
RESP_END = "<|RESP_END|>"
from src.data.dpo.yl_rephrase_prompts import (
    YL_SYSTEM_PROMPT as _YL_SYSTEM_PROMPT,
    YL_USER_TEMPLATE as _YL_USER_TEMPLATE,
    YW_SYSTEM_PROMPT as _YW_SYSTEM_PROMPT,
    YW_USER_TEMPLATE as _YW_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

REPHRASE_MODEL: str = "claude-sonnet-4-20250514"
REPHRASE_TEMPERATURE: float = 0.7
REPHRASE_MAX_TOKENS: int = 500
MIN_RESPONSE_CHARS: int = 10
MAX_SIMILARITY_RATIO: float = 0.90


def _wrap_response(text: str) -> str:
    """Ensure text is wrapped with response delimiters."""
    if not text.startswith(RESP_START):
        text = "%s%s%s" % (RESP_START, text, RESP_END)
    return text


def _extract_levels(conflict_type: str) -> tuple[int, int]:
    """Return (victim_level, attacker_level) from a conflict_type string.

    E.g. 'L1_vs_L3' -> (1, 3).  Falls back to (1, 3) on parse failure.
    """
    parts = conflict_type.split("_vs_")
    if len(parts) == 2:
        try:
            victim = int(parts[0].lstrip("L"))
            attacker = int(parts[1].lstrip("L"))
            return victim, attacker
        except ValueError:
            pass
    return 1, 3


_META_REASONING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^.{0,20}(?:looking at|analyzing|considering) this scenario.*?(?:here'?s|the rejected|the response)[:\s]*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^.{0,20}(?:I need to|I should|Let me) (?:generate|create|produce).*?(?:here'?s|the rejected|the response)[:\s]*", re.IGNORECASE | re.DOTALL),
    re.compile(r"^.{0,20}(?:as the rejected|for the rejected|the incorrect).*?(?:response|answer)[:\s]*", re.IGNORECASE | re.DOTALL),
]


def _strip_meta_reasoning(text: str) -> str:
    """Remove Claude's chain-of-thought preamble from a generated response."""
    for pattern in _META_REASONING_PATTERNS:
        text = pattern.sub("", text).strip()
    return text


def _is_valid_yl(candidate: str, chosen: str) -> bool:
    """Return True when candidate is an acceptable y_l replacement."""
    if len(candidate) < MIN_RESPONSE_CHARS:
        return False
    if is_refusal(candidate):
        return False
    ratio = SequenceMatcher(None, chosen, candidate).ratio()
    if ratio >= MAX_SIMILARITY_RATIO:
        return False
    return True


def _is_valid_yw(candidate: str) -> bool:
    """Return True when candidate is an acceptable y_w replacement."""
    if len(candidate) < MIN_RESPONSE_CHARS:
        return False
    if is_refusal(candidate):
        return False
    return True


def apply_rephrase(client: object, inst: dict, problems: list[str]) -> tuple[bool, str | None]:
    """Attempt to rephrase the broken field(s) of a single instance.

    Mutates inst in-place on success. Priority: yl_refusal > yl_weak > yw_broken.

    Args:
        client: An AnthropicClient instance.
        inst: A DPO instance dict (mutated in-place on success).
        problems: List of detected problem category strings.

    Returns:
        (fixed, fix_applied) — True and the fix label when successful, else (False, None).
    """
    if "yl_refusal" in problems:
        replacement = rephrase_yl(client, inst)
        if replacement is not None:
            inst["rejected"] = _wrap_response(replacement)
            inst["yl_fallback_used"] = True
            inst["yl_model"] = REPHRASE_MODEL
            inst["fix_applied"] = "yl_refusal_rephrase"
            return True, "yl_refusal_rephrase"
        return False, None

    if "yl_weak" in problems:
        replacement = rephrase_yl(client, inst)
        if replacement is not None:
            inst["rejected"] = _wrap_response(replacement)
            inst["yl_fallback_used"] = True
            inst["yl_model"] = REPHRASE_MODEL
            inst["fix_applied"] = "yl_weak_rephrase"
            return True, "yl_weak_rephrase"
        return False, None

    if "yw_broken" in problems:
        replacement = rephrase_yw(client, inst)
        if replacement is not None:
            inst["chosen"] = _wrap_response(replacement)
            inst["yw_model"] = REPHRASE_MODEL
            inst["fix_applied"] = "yw_rephrase"
            return True, "yw_rephrase"
        return False, None

    return False, None


def rephrase_yl(client: object, instance: dict) -> str | None:
    """Call the Claude API to generate a replacement y_l (rejected) field.

    Args:
        client: An AnthropicClient instance.
        instance: A DPO instance dict with keys: conflict_type, prompt, chosen.

    Returns:
        Replacement text on success, None if both attempts fail validation.
    """
    conflict_type: str = instance.get("conflict_type", "")
    prompt: str = instance.get("prompt", "")
    chosen: str = instance.get("chosen", "")
    victim_level, attacker_level = _extract_levels(conflict_type)

    user_prompt = _YL_USER_TEMPLATE.format(
        conflict_type=conflict_type,
        victim_level=victim_level,
        attacker_level=attacker_level,
        prompt=prompt,
        chosen=chosen,
    )

    for attempt in range(2):
        try:
            candidate = client.generate(  # type: ignore[attr-defined]
                user_prompt=user_prompt,
                system_prompt=_YL_SYSTEM_PROMPT,
                model=REPHRASE_MODEL,
                temperature=REPHRASE_TEMPERATURE,
                max_tokens=REPHRASE_MAX_TOKENS,
            )
            candidate = _strip_meta_reasoning(candidate)
            if _is_valid_yl(candidate, chosen):
                return candidate
            logger.warning(
                "rephrase_yl attempt %d failed validation (len=%d, refusal=%s)",
                attempt + 1,
                len(candidate),
                is_refusal(candidate),
            )
        except Exception:
            logger.exception("rephrase_yl attempt %d raised an exception", attempt + 1)

    return None


def rephrase_yw(client: object, instance: dict) -> str | None:
    """Call the Claude API to generate a replacement y_w (chosen) field.

    Args:
        client: An AnthropicClient instance.
        instance: A DPO instance dict with key: prompt.

    Returns:
        Replacement text on success, None if both attempts fail validation.
    """
    prompt: str = instance.get("prompt", "")
    user_prompt = _YW_USER_TEMPLATE.format(prompt=prompt)

    for attempt in range(2):
        try:
            candidate = client.generate(  # type: ignore[attr-defined]
                user_prompt=user_prompt,
                system_prompt=_YW_SYSTEM_PROMPT,
                model=REPHRASE_MODEL,
                temperature=REPHRASE_TEMPERATURE,
                max_tokens=REPHRASE_MAX_TOKENS,
            )
            candidate = _strip_meta_reasoning(candidate)
            if _is_valid_yw(candidate):
                return candidate
            logger.warning(
                "rephrase_yw attempt %d failed validation (len=%d, refusal=%s)",
                attempt + 1,
                len(candidate),
                is_refusal(candidate),
            )
        except Exception:
            logger.exception("rephrase_yw attempt %d raised an exception", attempt + 1)

    return None
