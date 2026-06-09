#!/usr/bin/env python3
"""Build the HP-select stratified split from the DPO validation set.

Reads ``data/dpo/val/dpo_combined.jsonl`` (or ``--source``) and writes:

- ``models/hp_search/data/hp_select.jsonl`` — held-out set for HP ranking.
- ``models/hp_search/data/val_train.jsonl`` — training-time eval set.
- ``models/hp_search/data/split_manifest.json`` — seed, counts, source
  sha256; enables idempotent re-runs.

Idempotent: if the output directory already contains a manifest whose
``source_sha256`` matches the current source file, the script exits
successfully without rewriting anything.

Usage:
    python bin/build_hp_select_split.py
    python bin/build_hp_select_split.py --target-size 1000 --seed 42
    python bin/build_hp_select_split.py --source path/to/val.jsonl \\
        --out-dir path/to/output
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dpo.hp_split import build_hp_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stratified HP-select split for DPO hyperparameter search.",
    )
    parser.add_argument(
        "--source", type=Path,
        default=_PROJECT_ROOT / "data" / "dpo" / "val" / "dpo_combined.jsonl",
        help="Path to the DPO validation JSONL.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_PROJECT_ROOT / "models" / "hp_search" / "data",
        help="Directory for hp_select.jsonl, val_train.jsonl, split_manifest.json.",
    )
    parser.add_argument("--target-size", type=int, default=1000,
                         help="Number of records for the hp_select cut.")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed for reproducibility.")
    args = parser.parse_args()

    if not args.source.exists():
        logger.error("Source file not found: %s", args.source)
        return 1

    src_sha = sha256_file(args.source)
    hp_path = args.out_dir / "hp_select.jsonl"
    val_path = args.out_dir / "val_train.jsonl"
    manifest_path = args.out_dir / "split_manifest.json"

    # Idempotency check.
    if hp_path.exists() and val_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_sha256") == src_sha
            and manifest.get("seed") == args.seed
            and manifest.get("hp_select_size") == args.target_size
        ):
            logger.info(
                "HP-select split already built at %s (source unchanged). Skipping.",
                args.out_dir,
            )
            return 0

    logger.info("Building HP-select split from %s", args.source)
    records = load_jsonl(args.source)
    logger.info("Loaded %d records", len(records))

    hp_idx, val_idx, bucket_counts = build_hp_split(
        records, target_size=args.target_size, seed=args.seed,
    )

    hp_records = [records[i] for i in hp_idx]
    val_records = [records[i] for i in val_idx]

    write_jsonl(hp_records, hp_path)
    write_jsonl(val_records, val_path)

    manifest = {
        "source_path": str(args.source.relative_to(_PROJECT_ROOT))
            if args.source.is_relative_to(_PROJECT_ROOT) else str(args.source),
        "source_sha256": src_sha,
        "seed": args.seed,
        "hp_select_size": len(hp_records),
        "val_train_size": len(val_records),
        "bucket_counts": {
            f"gap{gap}_cal{int(is_cal)}": count
            for (gap, is_cal), count in sorted(bucket_counts.items())
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Wrote %d hp_select + %d val_train to %s",
        len(hp_records), len(val_records), args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
