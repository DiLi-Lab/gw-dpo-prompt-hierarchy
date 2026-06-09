#!/usr/bin/env python3
"""Validate token lengths for SFT and DPO datasets.

Tokenises all examples, reports length statistics, checks delimiter integrity,
and computes the recommended max_seq_length (smallest power-of-2 >= p100).
Saves results to the split-aware stats directory.

Usage:
    python bin/validate_lengths.py sft
    python bin/validate_lengths.py dpo
    python bin/validate_lengths.py sft --split train
    python bin/validate_lengths.py dpo --split val
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from transformers import AutoTokenizer

from src.config import load_config
from src.data.length_validation import validate_example_lengths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def smallest_power_of_two_gte(n: int) -> int:
    """Return the smallest power of 2 >= n, with a floor of 2048."""
    if n <= 2048:
        return 2048
    return 2 ** math.ceil(math.log2(n))


def load_sft_texts(sft_path: Path) -> list[str]:
    """Load text fields from the SFT JSONL file."""
    if not sft_path.exists():
        logger.error("SFT dataset not found: %s", sft_path)
        sys.exit(1)
    texts = []
    with open(sft_path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    logger.info("Loaded %d SFT examples from %s", len(texts), sft_path)
    return texts


def load_dpo_texts(dpo_path: Path) -> list[str]:
    """Load prompt+chosen and prompt+rejected from DPO JSONL.

    DPO training processes both prompt+chosen and prompt+rejected,
    so we return the longer of the two per example.
    """
    if not dpo_path.exists():
        logger.error("DPO dataset not found: %s", dpo_path)
        sys.exit(1)
    texts = []
    with open(dpo_path) as f:
        for line in f:
            row = json.loads(line)
            prompt = row["prompt"]
            chosen = prompt + row["chosen"]
            rejected = prompt + row["rejected"]
            # Use the longer concatenation as the effective training length
            texts.append(chosen if len(chosen) >= len(rejected) else rejected)
    logger.info("Loaded %d DPO examples from %s", len(texts), dpo_path)
    return texts


def run_validation(
    phase: str,
    texts: list[str],
    tokenizer_path: Path,
    stats_dir: Path,
) -> None:
    """Run validation, print report, and save stats."""
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Use a huge limit so we see the full distribution without false positives
    report = validate_example_lengths(texts, tokenizer, max_seq_length=999_999)

    print(report.summary())
    print()

    max_len = report.stats.max_length
    recommended = smallest_power_of_two_gte(max_len)
    print("Recommended max_seq_length: %d  (smallest power-of-2 >= %d)" % (recommended, max_len))

    if report.issues:
        print("\nDelimiter issues found: %d (see details above)" % len(report.issues))
        for issue in report.issues[:20]:
            print("  - %s" % issue)
        if len(report.issues) > 20:
            print("  ... and %d more" % (len(report.issues) - 20))

    # Save stats
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "phase": phase,
        "count": report.stats.count,
        "min": report.stats.min_length,
        "max": report.stats.max_length,
        "mean": round(report.stats.mean_length, 1),
        "p50": report.stats.p50,
        "p95": report.stats.p95,
        "p99": report.stats.p99,
        "recommended_max_seq_length": recommended,
        "delimiter_issues": len(report.issues),
    }
    stats_file = stats_dir / ("%s_length_stats.json" % phase)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nStats saved to %s" % stats_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate token lengths for SFT or DPO datasets.",
    )
    parser.add_argument(
        "phase",
        choices=["sft", "dpo"],
        help="Which dataset to validate.",
    )
    parser.add_argument(
        "--config",
        default=str(_project_root / "configs" / "base_linear.yaml"),
        help="Path to config YAML (default: configs/base_linear.yaml).",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Which split to validate (default: no split subdirectory).",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("unrecognized arguments: %s" % " ".join(unknown))

    cfg = load_config(Path(args.config))
    if args.split:
        cfg.paths.split = args.split
    paths = cfg.paths
    tokenizer_path = paths.tokenizer_dir
    stats_dir = paths.stats_dir

    if not tokenizer_path.exists():
        logger.error("Tokenizer not found at %s", tokenizer_path)
        sys.exit(1)

    if args.phase == "sft":
        texts = load_sft_texts(paths.sft_combined)
    else:
        dpo_path = paths.dpo_combined
        texts = load_dpo_texts(dpo_path)

    run_validation(args.phase, texts, tokenizer_path, stats_dir)


if __name__ == "__main__":
    main()
