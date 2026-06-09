#!/usr/bin/env python3
"""Materialise the 3-level DPO dataset for ablation (e).

Reads `data/dpo/{split}/dpo_combined.jsonl`, drops intra-System pairs,
rewrites prompts via `collapse_prompt`, recomputes `level_gap` under the
3-level mapping, adds provenance, and writes the result to
`data/dpo/{split}_3level/dpo_combined.jsonl`. Idempotent — re-running
overwrites the output deterministically.

Usage:
    python bin/build_3level_dpo_dataset.py
    python bin/build_3level_dpo_dataset.py --project-root /tmp/test
    python bin/build_3level_dpo_dataset.py --splits train val
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.three_level import (
    collapse_prompt,
    is_intra_system,
    recompute_3level_gap,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLLAPSE_VERSION = "1.0"


def _hash_record(record: dict) -> str:
    """Stable hash of (prompt, chosen, rejected) for cross-dataset traceability."""
    payload = "|".join(
        record.get(field, "") for field in ("prompt", "chosen", "rejected")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _transform_record(record: dict) -> dict | None:
    """Apply the (e) transform to one DPO record. Returns None if dropped."""
    victim = record["victim_level"]
    attacker = record["attacker_level"]
    if not record.get("is_calibration", False) and is_intra_system(victim, attacker):
        return None

    out = dict(record)  # shallow copy preserves all provenance fields
    out["original_victim_level"] = victim
    out["original_attacker_level"] = attacker
    out["original_conflict_type"] = record.get("conflict_type", "")
    out["source_5level_id"] = _hash_record(record)
    out["collapse_version"] = COLLAPSE_VERSION

    out["prompt"] = collapse_prompt(record["prompt"])

    if record.get("is_calibration", False):
        out["level_gap"] = 0
    else:
        out["level_gap"] = recompute_3level_gap(victim, attacker)

    out["margin"] = float(out["level_gap"])
    return out


def _build_one_split(src: Path, dst: Path) -> tuple[int, int]:
    """Read src JSONL, transform, write to dst JSONL. Returns (kept, dropped)."""
    if not src.exists():
        msg = f"Source file does not exist: {src}"
        raise FileNotFoundError(msg)

    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            transformed = _transform_record(record)
            if transformed is None:
                dropped += 1
                continue
            fout.write(json.dumps(transformed) + "\n")
            kept += 1
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=_PROJECT_ROOT,
        help="Project root containing data/dpo/{split}/.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Splits to process (default: train val).",
    )
    args = parser.parse_args()

    for split in args.splits:
        src = args.project_root / "data" / "dpo" / split / "dpo_combined.jsonl"
        dst = (
            args.project_root / "data" / "dpo" / f"{split}_3level"
            / "dpo_combined.jsonl"
        )
        kept, dropped = _build_one_split(src, dst)
        logger.info(
            "%s: kept=%d, dropped=%d (intra-System) -> %s",
            split, kept, dropped, dst,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
