"""SFT dataset save/load/stats utilities.

Provides functions for persisting SFT training examples as JSONL files,
loading them back, and computing summary statistics over example collections.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_sft_dataset(examples: list[dict], path: Path) -> None:
    """Write SFT examples to a JSONL file (one JSON object per line).

    Creates parent directories if they do not exist.

    Args:
        examples: List of SFT example dicts to persist.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Saved %d SFT examples to %s", len(examples), path)


def load_sft_dataset(path: Path) -> list[dict]:
    """Load SFT examples from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of SFT example dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError("SFT dataset file not found: %s" % path)

    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(json.loads(stripped))

    logger.info("Loaded %d SFT examples from %s", len(examples), path)
    return examples


def compute_sft_stats(examples: list[dict]) -> dict:
    """Compute summary statistics over a collection of SFT examples.

    Args:
        examples: List of SFT example dicts, each expected to have keys
            ``is_conflict``, ``conflict_type``, and ``levels_present``.

    Returns:
        Dict with keys: total, aligned, conflicting, conflict_types,
        level_configurations, sft_categories, sft_sources, l4_generations.
    """
    total = len(examples)
    aligned = 0
    conflicting = 0
    conflict_types: dict[str, int] = {}
    level_configurations: dict[str, int] = {}
    sft_categories: dict[str | None, int] = {}
    sft_sources: dict[str | None, int] = {}
    l4_generations: dict[str | None, int] = {}

    for ex in examples:
        if ex.get("is_conflict"):
            conflicting += 1
            ct = ex.get("conflict_type")
            if ct is not None:
                conflict_types[ct] = conflict_types.get(ct, 0) + 1
        else:
            aligned += 1

        levels = ex.get("levels_present", [])
        level_key = str(sorted(levels))
        level_configurations[level_key] = level_configurations.get(level_key, 0) + 1

        cat = ex.get("sft_category")
        sft_categories[cat] = sft_categories.get(cat, 0) + 1

        src = ex.get("sft_source")
        sft_sources[src] = sft_sources.get(src, 0) + 1

        gen = ex.get("l4_generation")
        l4_generations[gen] = l4_generations.get(gen, 0) + 1

    return {
        "total": total,
        "aligned": aligned,
        "conflicting": conflicting,
        "conflict_types": conflict_types,
        "level_configurations": level_configurations,
        "sft_categories": sft_categories,
        "sft_sources": sft_sources,
        "l4_generations": l4_generations,
    }
