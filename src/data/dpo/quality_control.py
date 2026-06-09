"""DPO quality control pipeline: filtering, deduplication, sampling, and LLM judging.

Provides automated quality gates for DPO training examples including
token-length checks, delimiter validation, similarity filtering,
deduplication, stratified sampling, and dual-judge LLM evaluation.
"""

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RESP_START = "<|RESP_START|>"
RESP_END = "<|RESP_END|>"
MIN_RESPONSE_TOKENS: int = 10
MAX_SIMILARITY_RATIO: float = 0.90
NEAR_DEDUP_THRESHOLD: float = 0.95
DELIMITER_PATTERN: re.Pattern[str] = re.compile(
    r"<\|L([0-4])_START\|>|<\|L([0-4])_END\|>"
)
SCORE_KEYS: list[str] = ["q1", "q2", "q3", "q4"]

_JUDGE_SYSTEM_PROMPT: str = (
    "You are evaluating preference pairs for a TRAINING DATASET that teaches "
    "AI models to follow a 5-level instruction hierarchy:\n"
    "L0 (platform rules) > L1 (developer system prompt) > L2 (user config) > "
    "L3 (user message) > L4 (data/tool output)\n"
    "Higher levels always override lower levels in case of conflict.\n\n"
    "IMPORTANT: You are judging whether each example is a WELL-CONSTRUCTED "
    "TRAINING PAIR, not whether the content is appropriate. The dataset "
    "intentionally contains adversarial, unsafe, and harmful content in the "
    "rejected responses — this is by design, to teach models what NOT to do. "
    "A good training pair has a clear contrast between chosen (correct "
    "hierarchy compliance) and rejected (incorrect hierarchy violation).\n\n"
    "Set keep=true when the training pair is well-constructed (clear contrast, "
    "correct labels, realistic scenario). Set keep=false ONLY when the pair "
    "is poorly constructed (e.g., both responses are identical, labels are "
    "swapped, the scenario is incoherent, or the responses don't match the "
    "conflict type)."
)

_JUDGE_USER_TEMPLATE: str = (
    "Conflict type: {conflict_type}\n\n"
    "Prompt:\n{prompt}\n\n"
    "Chosen response (should correctly follow the higher-priority level):\n"
    "{chosen}\n\n"
    "Rejected response (should incorrectly follow the lower-priority level):\n"
    "{rejected}\n\n"
    "Rate each on 1-5:\n"
    "1. Does the chosen response correctly follow the higher-priority level?\n"
    "2. Does the rejected response clearly violate the hierarchy by following "
    "the wrong level?\n"
    "3. Are the two responses sufficiently different from each other?\n"
    "4. Is the scenario realistic and coherent?\n\n"
    "Remember: keep=true means this is a good training example, regardless of "
    "how adversarial the rejected content is.\n\n"
    'Output ONLY raw JSON (no code fences): '
    '{{"q1": X, "q2": X, "q3": X, "q4": X, "keep": true/false, '
    '"reason": "one sentence"}}'
)


def _strip_resp_delimiters(text: str) -> str:
    """Remove RESP_START and RESP_END delimiters from response text."""
    text = text.replace(RESP_START, "")
    text = text.replace(RESP_END, "")
    return text.strip()


def _check_delimiters(prompt: str) -> bool:
    """Verify that every START delimiter has a matching END in the prompt."""
    starts: set[str] = set()
    ends: set[str] = set()
    for match in DELIMITER_PATTERN.finditer(prompt):
        if match.group(1) is not None:
            starts.add(match.group(1))
        if match.group(2) is not None:
            ends.add(match.group(2))
    if not starts and not ends:
        return True
    return starts == ends


def filter_dpo_example(example: dict, tokenizer: object) -> bool:
    """Filter a single DPO example based on quality criteria.

    Args:
        example: DPO example dict with prompt, chosen, rejected keys.
        tokenizer: Object with encode(text, add_special_tokens=False) method.

    Returns:
        True to keep the example, False to discard.
    """
    chosen_text = _strip_resp_delimiters(example["chosen"])
    rejected_text = _strip_resp_delimiters(example["rejected"])

    chosen_tokens = tokenizer.encode(chosen_text, add_special_tokens=False)
    if len(chosen_tokens) < MIN_RESPONSE_TOKENS:
        return False

    rejected_tokens = tokenizer.encode(rejected_text, add_special_tokens=False)
    if len(rejected_tokens) < MIN_RESPONSE_TOKENS:
        return False

    ratio = SequenceMatcher(None, chosen_text, rejected_text).ratio()
    if ratio >= MAX_SIMILARITY_RATIO:
        return False

    if not _check_delimiters(example["prompt"]):
        return False

    return True


def deduplicate_by_hash(examples: list[dict]) -> list[dict]:
    """Remove duplicate examples by hashing the prompt field.

    Keeps the first occurrence of each unique prompt.

    Args:
        examples: List of DPO example dicts.

    Returns:
        Deduplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for ex in examples:
        h = hashlib.sha256(ex["prompt"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(ex)
    return result


def deduplicate_by_embedding(
    examples: list[dict],
    threshold: float = NEAR_DEDUP_THRESHOLD,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Remove near-duplicate examples by semantic embedding similarity.

    Embeds all prompts with a sentence-transformer model, then greedily
    removes examples whose cosine similarity to any already-kept example
    exceeds the threshold. Keeps the first occurrence.

    Args:
        examples: List of DPO example dicts.
        threshold: Cosine similarity above which to remove.
        model_name: Sentence-transformer model to use.

    Returns:
        Deduplicated list with near-duplicates removed.
    """
    if not examples:
        return examples

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; skipping embedding deduplication. "
            "Install with: pip install sentence-transformers"
        )
        return examples

    logger.info("Embedding %d prompts with %s for near-deduplication", len(examples), model_name)
    model = SentenceTransformer(model_name)
    prompts = [ex["prompt"] for ex in examples]
    embeddings = model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)

    keep_indices: list[int] = []
    kept_embeddings: list[np.ndarray] = []

    for i, emb in enumerate(embeddings):
        if kept_embeddings:
            sims = np.dot(np.array(kept_embeddings), emb)
            if np.max(sims) >= threshold:
                continue
        keep_indices.append(i)
        kept_embeddings.append(emb)

    removed = len(examples) - len(keep_indices)
    logger.info(
        "Near-deduplication: kept %d, removed %d (threshold=%.2f)",
        len(keep_indices), removed, threshold,
    )
    return [examples[i] for i in keep_indices]


def stratified_sample(
    examples: list[dict],
    fraction: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """Sample examples proportionally by conflict_type.

    Args:
        examples: List of DPO example dicts with conflict_type key.
        fraction: Fraction of each group to sample.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample of examples.
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[ex["conflict_type"]].append(ex)

    result: list[dict] = []
    for conflict_type in sorted(groups):
        group = groups[conflict_type]
        n = max(1, round(fraction * len(group)))
        result.extend(rng.sample(group, min(n, len(group))))
    return result


def build_judge_prompt(example: dict) -> tuple[str, str]:
    """Build a judge evaluation prompt for a DPO example.

    Returns a (system_prompt, user_prompt) tuple matching the format
    documented in doc 16 Section 8.2.

    Args:
        example: DPO example dict.

    Returns:
        Tuple of (system_prompt, user_prompt) for the judge LLM.
    """
    chosen_text = _strip_resp_delimiters(example["chosen"])
    rejected_text = _strip_resp_delimiters(example["rejected"])

    user_prompt = _JUDGE_USER_TEMPLATE.format(
        conflict_type=example["conflict_type"],
        prompt=example["prompt"],
        chosen=chosen_text,
        rejected=rejected_text,
    )

    return _JUDGE_SYSTEM_PROMPT, user_prompt


def parse_judge_response(response_text: str) -> dict | None:
    """Parse a judge LLM response as JSON.

    Handles responses wrapped in markdown code fences (```json ... ```),
    which GPT-4o and Gemini frequently produce despite being asked for
    raw JSON.

    Args:
        response_text: Raw text response from the judge LLM.

    Returns:
        Parsed dict with score keys, or None if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse judge response: %s", response_text[:200])
        return None


def apply_judge_decisions(
    gpt_scores: dict, gemini_scores: dict
) -> str:
    """Apply dual-judge consensus logic to determine example fate.

    Args:
        gpt_scores: Parsed scores from GPT judge (q1-q4 ints, keep bool).
        gemini_scores: Parsed scores from Gemini judge (q1-q4 ints, keep bool).

    Returns:
        "keep", "discard", or "flag".
    """
    gpt_keep = gpt_scores["keep"]
    gemini_keep = gemini_scores["keep"]

    # Disagreement takes priority — flag for human review
    if gpt_keep != gemini_keep:
        return "flag"

    all_scores = [gpt_scores[k] for k in SCORE_KEYS] + [
        gemini_scores[k] for k in SCORE_KEYS
    ]

    # Both agree to reject, or any critical score
    if not gpt_keep and not gemini_keep:
        return "discard"

    # Both agree to keep — but override if any score is 1 or not all >= 3
    if any(s == 1 for s in all_scores):
        return "discard"
    if all(s >= 3 for s in all_scores):
        return "keep"
    return "discard"


def save_flagged_examples(
    flagged: list[dict],
    output_path: Path,
) -> None:
    """Save flagged examples to a JSONL file for manual review.

    Each line includes the example plus both judges' scores and reasons.

    Args:
        flagged: List of dicts with keys "example", "gpt_scores", "gemini_scores".
        output_path: Path to save the flagged examples JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in flagged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Saved %d flagged examples to %s for manual review", len(flagged), output_path)
