"""Detection helpers for broken y_l / y_w fields in Phase 3 DPO data.

Provides:
  - extract_level_text(): extract level-delimited content from a prompt
  - compute_content_jaccard(): Jaccard similarity after stopword removal
  - is_yw_role_mismatch(): detect refusal / role-mismatch in y_w
  - load_injection_templates(): load injection template safety index
  - detect_problems(): classify all problems for a single instance
"""

import json
import re
from pathlib import Path

from src.data.dpo.response_utils import STOPWORDS, is_refusal

ROLE_MISMATCH_PATTERNS: list[str] = [
    "outside my role",
    "mismatch between",
    "not within my scope",
    "falls outside what",
    "outside my designated",
]

JACCARD_THRESHOLD: float = 0.02


def extract_level_text(prompt: str, level: int) -> str:
    """Extract text between <|L{level}_START|> and <|L{level}_END|> tokens."""
    pattern = rf"<\|L{level}_START\|>(.*?)<\|L{level}_END\|>"
    match = re.search(pattern, prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def compute_content_jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts after stopword removal."""
    tokens_a = {w for w in text_a.lower().split() if w not in STOPWORDS}
    tokens_b = {w for w in text_b.lower().split() if w not in STOPWORDS}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def is_yw_role_mismatch(text: str) -> bool:
    """Check for refusal or role-mismatch language in y_w (chosen)."""
    text_lower = text.lower()
    return is_refusal(text) or any(p in text_lower for p in ROLE_MISMATCH_PATTERNS)


def load_injection_templates(path: Path) -> dict[str, bool]:
    """Load injection templates and return mapping of id -> targets_safety."""
    with path.open() as f:
        data = json.load(f)
    return {entry["id"]: entry["targets_safety"] for entry in data["position_injections"]}


def detect_problems(
    instance: dict,
    injection_safety: dict[str, bool],
    fix_weak_yl: bool,
    fix_yw: bool,
) -> list[str]:
    """Return list of problem category strings for a single instance."""
    problems: list[str] = []
    rejected: str = instance.get("rejected", "")
    chosen: str = instance.get("chosen", "")
    conflict_type: str = instance.get("conflict_type", "")
    template_id: str | None = instance.get("injection_template_id")
    prompt: str = instance.get("prompt", "")

    if is_refusal(rejected):
        problems.append("yl_refusal")

    if fix_weak_yl:
        l3_text = extract_level_text(prompt, 3)
        l4_text = extract_level_text(prompt, 4)
        reference = f"{l3_text} {l4_text}".strip()
        if compute_content_jaccard(rejected, reference) < JACCARD_THRESHOLD:
            problems.append("yl_weak")

    if conflict_type == "L0_vs_L4":
        if fix_yw and is_yw_role_mismatch(chosen):
            problems.append("yw_broken")

        if template_id is None or not injection_safety.get(template_id, False):
            problems.append("wrong_injection")

    return problems
