#!/usr/bin/env python3
"""Download and split base datasets (Alpaca Cleaned + Dolly 15K).

Usage:
    python bin/download_base_datasets.py
    python bin/download_base_datasets.py --validate
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import load_config
from src.data.base_datasets import download_and_split, validate_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download, split, and optionally validate base datasets."""
    parser = argparse.ArgumentParser(
        description="Download and split base datasets for hierarchy training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "base_linear.yaml",
        help="Path to YAML config file (default: configs/base_linear.yaml).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check existing splits without downloading.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as section.key=value.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: %s" % " ".join(unknown))

    cfg = load_config(config_path=args.config, overrides=args.override)

    if args.validate:
        logger.info("Validating existing splits...")
        report = validate_splits(cfg.paths)
        if report["all_exist"]:
            logger.info("All splits present:")
            for name, count in report["counts"].items():
                logger.info("  %s: %d examples", name, count)
        else:
            logger.warning("Missing splits: %s", report["missing"])
            if report["counts"]:
                logger.info("Existing splits:")
                for name, count in report["counts"].items():
                    logger.info("  %s: %d examples", name, count)
            sys.exit(1)
    else:
        logger.info("Downloading and splitting base datasets...")
        counts = download_and_split(cfg.paths)
        logger.info("Done. Split sizes:")
        for name, count in counts.items():
            logger.info("  %s: %d examples", name, count)


if __name__ == "__main__":
    main()
