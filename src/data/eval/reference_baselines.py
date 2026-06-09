"""Reference baseline construction for the eval pipeline (Phase 4).

Strips architectural cue tokens (L0–L4 delimiters, RESP tokens) from conflict
scenario prompts to create flat-text baselines that expose no hierarchy
information to the model being evaluated.
"""

import json
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# All delimiter token patterns to strip
_DELIMITER_PATTERN = re.compile(
    r"<\|(?:L[0-4]_START|L[0-4]_END|RESP_START|RESP_END)\|>"
)

# Collapse 3+ consecutive newlines into double newlines
_EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def strip_delimiters(prompt: str) -> str:
    """Remove all hierarchy delimiter tokens from a prompt string.

    Removes <|L0_START|>, <|L0_END|>, ..., <|L4_START|>, <|L4_END|>,
    <|RESP_START|>, and <|RESP_END|> tokens, then collapses 3+ consecutive
    newlines into double newlines and strips outer whitespace.

    Args:
        prompt: Delimited prompt string containing hierarchy tokens.

    Returns:
        Flat text string with all delimiter tokens removed.
    """
    result = _DELIMITER_PATTERN.sub("", prompt)
    result = _EXCESS_NEWLINES_PATTERN.sub("\n\n", result)
    return result.strip()


def build_reference_baseline(conflict_instance: dict) -> dict:
    """Build a reference baseline from a conflict instance.

    Copies the conflict instance, strips delimiter tokens from the prompt,
    and updates metadata fields to mark this as a reference baseline.

    Args:
        conflict_instance: Eval instance dict with at minimum "id",
            "prompt", and "split" keys.

    Returns:
        New dict with stripped prompt, split="reference",
        source_conflict_id set to the original id, and id prefix
        changed from "eval_" to "ref_".
    """
    baseline = dict(conflict_instance)
    baseline["prompt"] = strip_delimiters(conflict_instance["prompt"])
    baseline["split"] = "reference"
    baseline["source_conflict_id"] = conflict_instance["id"]
    baseline["id"] = conflict_instance["id"].replace("eval_", "ref_", 1)
    return baseline


def sample_for_reference(
    conflict_instances: list[dict],
    per_pair: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Sample conflict instances per conflict type and build reference baselines.

    Groups instances by conflict_type, samples min(per_pair, len(group)) from
    each group using a seeded RNG, then builds a reference baseline for each.

    Args:
        conflict_instances: List of eval instance dicts.
        per_pair: Maximum number of instances to sample per conflict type.
        seed: Random seed for reproducibility.

    Returns:
        List of reference baseline dicts.
    """
    rng = random.Random(seed)

    # Group by conflict_type
    groups: dict[str, list[dict]] = {}
    for instance in conflict_instances:
        ct = instance["conflict_type"]
        groups.setdefault(ct, []).append(instance)

    results: list[dict] = []
    for conflict_type, group in groups.items():
        n = min(per_pair, len(group))
        sampled = rng.sample(group, n)
        logger.debug(
            "Sampled %d/%d instances for conflict type %s",
            n, len(group), conflict_type,
        )
        for instance in sampled:
            results.append(build_reference_baseline(instance))

    return results


def run_phase4(
    *,
    conflict_instances: list[dict],
    output_path: Path,
    per_pair: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Build reference baselines for sampled conflict instances (Phase 4).

    Samples conflict instances per conflict type, builds flat-text reference
    baselines, writes them to a JSONL file, and returns the list.

    Args:
        conflict_instances: List of eval instance dicts from Phase 1+2.
        output_path: Path to output JSONL file.
        per_pair: Maximum number of instances to sample per conflict type.
        seed: Random seed for reproducibility.

    Returns:
        List of reference baseline dicts.
    """
    baselines = sample_for_reference(
        conflict_instances, per_pair=per_pair, seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for baseline in baselines:
            f.write(json.dumps(baseline) + "\n")

    logger.info(
        "Phase 4 complete: generated %d reference baselines, written to %s",
        len(baselines), output_path,
    )
    return baselines
