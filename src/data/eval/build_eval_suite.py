"""Eval suite build orchestrator: Phase 6 final assembly plus utilities.

Provides validation, statistics computation, cache I/O, and the Phase 6
pipeline that deduplicates, validates, and writes all three eval splits.
"""

import json
import logging
import re
from pathlib import Path

from src.data.dpo.quality_control import deduplicate_by_embedding, deduplicate_by_hash

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: list[str] = [
    "conflict_type",
    "level_gap",
    "conflict_description",
    "correct_behaviour",
    "violation_behaviour",
    "evaluation_criteria",
    "gold_response",
    "split",
]

_DELIMITER_PATTERN: re.Pattern[str] = re.compile(r"<\|L[0-4]_(START|END)\|>")


def validate_eval_instance(instance: dict) -> bool:
    """Validate a single eval instance against required field and format rules.

    Args:
        instance: Eval instance dict to validate.

    Returns:
        True if the instance is valid, False otherwise.
    """
    for field in _REQUIRED_FIELDS:
        if field not in instance:
            logger.debug("validate_eval_instance: missing required field '%s'", field)
            return False

    gold = instance["gold_response"]
    if not gold or not isinstance(gold, str):
        logger.debug("validate_eval_instance: gold_response is empty or not a string")
        return False

    criteria = instance["evaluation_criteria"]
    if not isinstance(criteria, list) or len(criteria) == 0:
        logger.debug("validate_eval_instance: evaluation_criteria must be a non-empty list")
        return False

    split = instance["split"]
    prompt = instance.get("prompt", "")
    has_delimiters = bool(_DELIMITER_PATTERN.search(prompt))

    if split == "reference":
        if has_delimiters:
            logger.debug("validate_eval_instance: reference split must not contain delimiter tokens")
            return False
    elif split in ("conflict", "aligned"):
        if not has_delimiters:
            logger.debug(
                "validate_eval_instance: %s split must contain delimiter tokens", split
            )
            return False

    return True


def compute_eval_stats(instances: list[dict]) -> dict:
    """Count eval instances by conflict_type, split, and base_dataset.

    Args:
        instances: List of eval instance dicts.

    Returns:
        Dict with keys: total, by_conflict_type, by_split, by_base_dataset.
    """
    by_conflict_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_base_dataset: dict[str, int] = {}

    for inst in instances:
        ct = inst.get("conflict_type")
        if ct is not None:
            by_conflict_type[ct] = by_conflict_type.get(ct, 0) + 1

        sp = inst.get("split")
        if sp is not None:
            by_split[sp] = by_split.get(sp, 0) + 1

        bd = inst.get("base_dataset")
        if bd is not None:
            by_base_dataset[bd] = by_base_dataset.get(bd, 0) + 1

    return {
        "total": len(instances),
        "by_conflict_type": by_conflict_type,
        "by_split": by_split,
        "by_base_dataset": by_base_dataset,
    }


def save_eval_cache(cache: dict[tuple[str, str, int], dict], path: Path) -> None:
    """Save eval cache to a JSONL file.

    Each line is a JSON object with "key" (list of [str, str, int]) and
    "value" (dict) fields.

    Args:
        cache: Mapping from (conflict_type, base_dataset, base_index) tuples
            to cached result dicts.
        path: Output path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, value in cache.items():
            entry = {"key": list(key), "value": value}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.debug("Saved %d cache entries to %s", len(cache), path)


def load_eval_cache(path: Path) -> dict[tuple[str, str, int], dict]:
    """Load eval cache from a JSONL file.

    Args:
        path: Path to the JSONL cache file.

    Returns:
        Mapping from (conflict_type, base_dataset, base_index) tuples to
        cached result dicts. Returns empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}

    cache: dict[tuple[str, str, int], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            raw_key = entry["key"]
            key: tuple[str, str, int] = (raw_key[0], raw_key[1], raw_key[2])
            cache[key] = entry["value"]

    logger.debug("Loaded %d cache entries from %s", len(cache), path)
    return cache


def run_phase6(
    *,
    conflict_instances: list[dict],
    aligned_instances: list[dict],
    reference_instances: list[dict],
    output_dir: Path,
    near_dedup_threshold: float = 0.85,
    skip_near_dedup: bool = False,
) -> dict:
    """Final assembly: dedup, validate, and write all three eval splits.

    Applies hash-based and (optionally) embedding-based deduplication to
    conflict instances, validates all three splits, writes JSONL files and
    a stats JSON to output_dir.

    Args:
        conflict_instances: Eval instances with split="conflict".
        aligned_instances: Eval instances with split="aligned".
        reference_instances: Eval instances with split="reference".
        output_dir: Directory to write output files.
        near_dedup_threshold: Cosine similarity threshold for near-dedup.
        skip_near_dedup: If True, skip embedding-based deduplication.

    Returns:
        Stats dict from compute_eval_stats over the written instances.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Phase 6: conflict=%d, aligned=%d, reference=%d",
        len(conflict_instances),
        len(aligned_instances),
        len(reference_instances),
    )

    # Dedup conflict instances
    before_hash = len(conflict_instances)
    conflict_instances = deduplicate_by_hash(conflict_instances)
    logger.info(
        "Hash dedup: %d -> %d conflict instances", before_hash, len(conflict_instances)
    )

    if not skip_near_dedup:
        before_near = len(conflict_instances)
        conflict_instances = deduplicate_by_embedding(
            conflict_instances, threshold=near_dedup_threshold
        )
        logger.info(
            "Near-dedup: %d -> %d conflict instances", before_near, len(conflict_instances)
        )

    # Validate all splits
    def _validate_split(instances: list[dict], split_name: str) -> list[dict]:
        valid = [inst for inst in instances if validate_eval_instance(inst)]
        invalid_count = len(instances) - len(valid)
        if invalid_count:
            logger.warning(
                "Phase 6: dropped %d invalid instances from %s split",
                invalid_count,
                split_name,
            )
        return valid

    conflict_instances = _validate_split(conflict_instances, "conflict")
    aligned_instances = _validate_split(aligned_instances, "aligned")
    reference_instances = _validate_split(reference_instances, "reference")

    # Write output files
    def _write_jsonl(instances: list[dict], filename: str) -> None:
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for inst in instances:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
        logger.info("Wrote %d instances to %s", len(instances), out_path)

    _write_jsonl(conflict_instances, "eval_conflicts.jsonl")
    _write_jsonl(aligned_instances, "eval_aligned.jsonl")
    _write_jsonl(reference_instances, "eval_reference.jsonl")

    all_instances = conflict_instances + aligned_instances + reference_instances
    stats = compute_eval_stats(all_instances)

    stats_path = output_dir / "eval_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Wrote eval stats to %s", stats_path)

    return stats
