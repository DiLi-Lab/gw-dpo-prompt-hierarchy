#!/usr/bin/env python3
"""Build the TensorTrust hijacking + extraction robustness CSVs.

Fetches the upstream JSONL files from
``HumanCompatibleAI/tensor-trust-data`` (MIT-licensed), pinned to a
specific commit SHA, and writes two CSVs:

  data/external/tensortrust/hijacking_robustness.csv   (~775 rows)
  data/external/tensortrust/extraction_robustness.csv  (~570 rows)

Plus a ``_subsample_manifest.json`` recording the upstream commit and
row counts. The CSVs are committed to the repository and are the
source of truth for runtime evaluation; this script is only re-run
when the upstream pin changes.

Upstream → on-disk schema mapping:

  on-disk column     upstream JSONL field
  ---------------    ------------------------------------------
  id                 row index within the split (1-indexed)
  defense_id         sample_id (stringified)
  pre_prompt         pre_prompt
  attack             attack
  post_prompt        post_prompt
  access_code        access_code
  label              0 (always — these are attack rows)
  source_index       0-based index in the upstream JSONL

No subsampling: the upstream sets are already curated subsets that
papers report on directly. v1 evaluates on the full ~1,345 rows.

Usage:
    python bin/build_tensortrust_subsample.py
"""

import argparse
import csv
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_root / ".env")

from src.config.loader import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Pinned upstream commit on ``HumanCompatibleAI/tensor-trust-data``.
# Verified via the GitHub commits API: this is the most recent (and
# only) commit touching the v1 robustness JSONLs.
_UPSTREAM_REVISION = "01856c36f7e6a70442378d7fe9d5e9de3329040a"
_BASE_URL = (
    f"https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/"
    f"{_UPSTREAM_REVISION}/benchmarks"
)
_HIJACKING_URL = f"{_BASE_URL}/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
_EXTRACTION_URL = f"{_BASE_URL}/extraction-robustness/v1/extraction_robustness_dataset.jsonl"

_REQUIRED_FIELDS = ("sample_id", "pre_prompt", "attack", "post_prompt", "access_code")
_CSV_HEADER = (
    "id", "defense_id", "pre_prompt", "attack", "post_prompt",
    "access_code", "label", "source_index",
)


def _fetch_jsonl(url: str) -> list[dict]:
    """Fetch a JSONL file from ``url`` and return its parsed rows."""
    logger.info("Fetching %s", url)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        msg = f"Failed to fetch {url}: {e}"
        raise RuntimeError(msg) from e
    rows = [json.loads(line) for line in body.splitlines() if line.strip()]
    if not rows:
        msg = f"Upstream {url} returned no rows."
        raise RuntimeError(msg)
    return rows


def _validate_schema(rows: list[dict], split: str) -> None:
    sample = rows[0]
    missing = [f for f in _REQUIRED_FIELDS if f not in sample]
    if missing:
        msg = (
            f"Upstream {split} dataset missing fields {missing}. "
            f"Available: {sorted(sample)}. Update _REQUIRED_FIELDS."
        )
        raise RuntimeError(msg)


def _write_csv_atomically(rows: list[dict], split: str, out_path: Path) -> int:
    """Write ``rows`` to ``out_path`` (atomic tmp+rename). Returns row count."""
    _validate_schema(rows, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(_CSV_HEADER)
        for source_index, r in enumerate(rows):
            writer.writerow([
                source_index + 1,             # 1-indexed runtime id
                str(r["sample_id"]),
                r["pre_prompt"],
                r["attack"],
                r["post_prompt"],
                r["access_code"],
                0,                            # label is always 0
                source_index,
            ])
    tmp.replace(out_path)
    return len(rows)


def build_csvs(
    *,
    hijacking_loader: Callable[[], list[dict]],
    extraction_loader: Callable[[], list[dict]],
    hijacking_csv_path: Path,
    extraction_csv_path: Path,
    manifest_path: Path,
    upstream_revision: str,
) -> None:
    """Build both CSVs + manifest from injected loaders.

    Loaders are injected so unit tests can supply mock rows without
    touching the network.
    """
    hijacking_rows = hijacking_loader()
    extraction_rows = extraction_loader()

    n_hijacking = _write_csv_atomically(
        hijacking_rows, "hijacking", hijacking_csv_path,
    )
    n_extraction = _write_csv_atomically(
        extraction_rows, "extraction", extraction_csv_path,
    )

    manifest = {
        "upstream_revision": upstream_revision,
        "n_hijacking": n_hijacking,
        "n_extraction": n_extraction,
        "n_total": n_hijacking + n_extraction,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(manifest_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tensortrust.yaml")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)
    if cfg.tensortrust is None:
        msg = f"{args.config} did not provide a `tensortrust:` section."
        raise ValueError(msg)

    hijacking_csv_path = cfg.paths.tensortrust_hijacking_csv
    extraction_csv_path = cfg.paths.tensortrust_extraction_csv
    manifest_path = cfg.paths.tensortrust_dir / "_subsample_manifest.json"

    logger.info(
        "Building TensorTrust CSVs:\n  hijacking  -> %s\n  extraction -> %s",
        hijacking_csv_path, extraction_csv_path,
    )

    build_csvs(
        hijacking_loader=lambda: _fetch_jsonl(_HIJACKING_URL),
        extraction_loader=lambda: _fetch_jsonl(_EXTRACTION_URL),
        hijacking_csv_path=hijacking_csv_path,
        extraction_csv_path=extraction_csv_path,
        manifest_path=manifest_path,
        upstream_revision=_UPSTREAM_REVISION,
    )
    logger.info("Done. Manifest -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
