"""L3 user message filtering and sampling from base datasets.

L3 user messages come directly from Alpaca and Dolly instruction fields.
No LLM generation needed. The module applies three filters:
- Remove instructions with fewer than 5 words
- Remove instructions with more than 500 words
- Remove exact duplicates

No persistent library file is created; filtering is applied on demand.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_from_disk

logger = logging.getLogger(__name__)

MIN_WORDS: int = 5
MAX_WORDS: int = 500


@dataclass(frozen=True)
class L3Message:
    """A filtered user message for L3 content.

    Attributes:
        text: The instruction text (stripped of leading/trailing whitespace).
        source: Origin dataset identifier ("alpaca" or "dolly").
    """

    text: str
    source: str


def filter_l3_candidates(
    dataset,
    instruction_field: str = "instruction",
    source: str = "unknown",
) -> list[L3Message]:
    """Filter dataset instructions into valid L3 candidates.

    Applies three filters:
    1. Remove instructions with fewer than MIN_WORDS words.
    2. Remove instructions with more than MAX_WORDS words.
    3. Remove exact duplicates (keeps first occurrence).

    Args:
        dataset: A HuggingFace Dataset with an instruction field.
        instruction_field: Name of the column containing instructions.
        source: Source identifier for provenance tracking.

    Returns:
        List of L3Message instances that pass all filters.
    """
    seen: set[str] = set()
    results: list[L3Message] = []

    for row in dataset:
        text = row[instruction_field].strip()
        word_count = len(text.split())

        if word_count < MIN_WORDS:
            continue
        if word_count > MAX_WORDS:
            continue
        if text in seen:
            continue

        seen.add(text)
        results.append(L3Message(text=text, source=source))

    logger.info(
        "Filtered %d -> %d L3 candidates from %s",
        len(dataset),
        len(results),
        source,
    )
    return results


def load_l3_pool(
    alpaca_path: Path,
    dolly_path: Path,
) -> list[L3Message]:
    """Load and filter L3 messages from both base dataset splits.

    Combines Alpaca and Dolly instruction fields, applies filters,
    and deduplicates across both sources.

    Args:
        alpaca_path: Path to the saved Alpaca train split directory.
        dolly_path: Path to the saved Dolly train split directory.

    Returns:
        Combined, filtered, deduplicated list of L3Message instances.
    """
    alpaca = load_from_disk(str(alpaca_path))
    dolly = load_from_disk(str(dolly_path))

    alpaca_msgs = filter_l3_candidates(
        alpaca, instruction_field="instruction", source="alpaca",
    )
    dolly_msgs = filter_l3_candidates(
        dolly, instruction_field="instruction", source="dolly",
    )

    # Deduplicate across sources (Alpaca takes priority)
    seen: set[str] = {m.text for m in alpaca_msgs}
    combined = list(alpaca_msgs)
    for m in dolly_msgs:
        if m.text not in seen:
            seen.add(m.text)
            combined.append(m)

    logger.info(
        "L3 pool: %d alpaca + %d dolly -> %d combined (after cross-source dedup)",
        len(alpaca_msgs),
        len(dolly_msgs),
        len(combined),
    )
    return combined


def sample_l3_message(
    pool: list[L3Message],
    seed: int | None = None,
) -> L3Message:
    """Sample a single L3 message from the pool.

    Args:
        pool: List of L3Message candidates.
        seed: Random seed for reproducibility.

    Returns:
        A randomly selected L3Message.

    Raises:
        ValueError: If the pool is empty.
    """
    if not pool:
        raise ValueError("Cannot sample from an empty L3 pool")
    rng = random.Random(seed)
    return rng.choice(pool)


def validate_l3_pool(pool: list[L3Message]) -> dict:
    """Compute statistics for an L3 message pool.

    Args:
        pool: List of L3Message instances.

    Returns:
        Dict with total count, per-source counts, and word count statistics.
    """
    source_counts: dict[str, int] = {}
    word_counts: list[int] = []

    for m in pool:
        source_counts[m.source] = source_counts.get(m.source, 0) + 1
        word_counts.append(len(m.text.split()))

    stats: dict = {
        "total": len(pool),
        "source_counts": source_counts,
    }

    if word_counts:
        sorted_wc = sorted(word_counts)
        stats["word_count_stats"] = {
            "min": sorted_wc[0],
            "max": sorted_wc[-1],
            "mean": sum(word_counts) / len(word_counts),
            "median": sorted_wc[len(sorted_wc) // 2],
        }
    else:
        stats["word_count_stats"] = {"min": 0, "max": 0, "mean": 0.0, "median": 0}

    return stats
