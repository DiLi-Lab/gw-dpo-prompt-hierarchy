#!/usr/bin/env python3
"""Build the 1,500-row stratified SEP subsample CSV.

Reads the upstream SEP dataset, stratifies by the configured field
(default ``domain``), samples deterministically with a fixed seed, and
writes ``data/external/sep/sep_subsample.csv``. The CSV is committed
to the repository and is the source of truth for runtime SEP
evaluation; this script is only re-run when the seed, size, or strata
field changes.

Upstream source — verified during PR 2:
  Repository:  github.com/egozverev/Should-It-Be-Executed-Or-Processed
  Dataset:     SEP_dataset/SEP_dataset.json (9,160 records)
  Pinned commit (used as ``upstream_revision`` in the manifest):
                b2561ee8b631f54d00b08d7db1ebb3b17352f339

The upstream dataset is **not** published on HuggingFace; the
authors distribute it as a single JSON file in the GitHub repo. The
file is fetched at build time via the raw.githubusercontent.com URL
pinned to the commit above. The committed CSV in this repository is
immune to upstream drift; only re-running this script can change it.

Upstream → on-disk schema mapping:

  on-disk column         upstream JSON field
  -------------------    ------------------------------------------
  instruction            system_prompt_clean
  data_with_witness      prompt_instructed   (data slot with the
                                              probe pre-injected)
  witness                witness
  domain                 info.type           (3-class: "Analytical
                                              and Evaluative Tasks",
                                              "Information Processing
                                              and Retrieval",
                                              "Creative and
                                              Generative Tasks")
  probe_type             info.appended_type  ("rr" | "rl" | "ll" |
                                              "lr"; encodes probe
                                              insertion position
                                              relative to the data
                                              slot, per the paper)
  source_index           the row's 0-based index in the upstream
                         JSON list (lets reviewers spot-check
                         post-hoc against the public dataset)

Usage:
    python bin/build_sep_subsample.py
    python bin/build_sep_subsample.py --override sep.subsample_seed=123
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

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_root / ".env")

from src.config.loader import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Upstream loading ----------------------------------------------
#
# Pinned in PR 2 against the upstream GitHub repo. The dataset is a
# single JSON file at the path below; ``_UPSTREAM_REVISION`` is the
# commit SHA of the most recent change to that file (verified via the
# GitHub API). If the upstream renames a JSON field, update
# ``_UPSTREAM_FIELDS`` and re-run the build; the on-disk CSV schema
# (id, domain, instruction, data_with_witness, witness, probe_type,
# source_index) does NOT change.

_UPSTREAM_REVISION = "b2561ee8b631f54d00b08d7db1ebb3b17352f339"
_UPSTREAM_URL = (
    f"https://raw.githubusercontent.com/egozverev/"
    f"Should-It-Be-Executed-Or-Processed/{_UPSTREAM_REVISION}/"
    f"SEP_dataset/SEP_dataset.json"
)

# Upstream → on-disk column map. Keys are our schema; values are the
# upstream field names (top-level or "info.<sub>" for nested) verified
# in step 2.1.
_UPSTREAM_FIELDS = {
    "instruction": "system_prompt_clean",
    "data_with_witness": "prompt_instructed",
    "witness": "witness",
    "domain": "info.type",
    "probe_type": "info.appended_type",
}


def _get_field(record: dict, dotted_path: str) -> object:
    """Read a top-level or ``info.<key>`` field from an upstream record."""
    if "." not in dotted_path:
        return record[dotted_path]
    head, tail = dotted_path.split(".", 1)
    return record[head][tail]


def _default_loader() -> list[dict]:
    """Fetch the upstream SEP dataset and rename columns to our schema.

    Returns:
        List of dicts with the on-disk schema keys plus ``_source_index``.

    Raises:
        RuntimeError: If the upstream JSON does not contain the expected
            top-level or ``info.<sub>`` fields, or if the network fetch
            fails (timeout, HTTP error, or other URL error).
    """
    logger.info("Fetching upstream SEP dataset from %s", _UPSTREAM_URL)
    try:
        with urllib.request.urlopen(_UPSTREAM_URL, timeout=60) as resp:
            upstream = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        msg = f"Failed to fetch SEP dataset from {_UPSTREAM_URL}: {e}"
        raise RuntimeError(msg) from e
    if not isinstance(upstream, list) or not upstream:
        msg = (
            f"Upstream {_UPSTREAM_URL} did not return a non-empty JSON list;"
            f" got {type(upstream).__name__}."
        )
        raise RuntimeError(msg)

    sample = upstream[0]
    available_top = set(sample.keys())
    available_info = set(sample.get("info", {}).keys())
    missing: list[str] = []
    for upstream_field in _UPSTREAM_FIELDS.values():
        if "." in upstream_field:
            head, tail = upstream_field.split(".", 1)
            if head not in available_top or tail not in available_info:
                missing.append(upstream_field)
        elif upstream_field not in available_top:
            missing.append(upstream_field)
    if missing:
        msg = (
            f"Upstream is missing expected fields: {missing}."
            f" Top-level: {sorted(available_top)}."
            f" info.*: {sorted(available_info)}."
            " Update _UPSTREAM_FIELDS in this script."
        )
        raise RuntimeError(msg)

    rows: list[dict] = []
    for i, record in enumerate(upstream):
        rows.append({
            "instruction":       _get_field(record, _UPSTREAM_FIELDS["instruction"]),
            "data_with_witness": _get_field(record, _UPSTREAM_FIELDS["data_with_witness"]),
            "witness":           _get_field(record, _UPSTREAM_FIELDS["witness"]),
            "domain":            _get_field(record, _UPSTREAM_FIELDS["domain"]),
            "probe_type":        _get_field(record, _UPSTREAM_FIELDS["probe_type"]),
            "_source_index":     i,
        })
    logger.info("Loaded %d upstream records", len(rows))
    return rows


# ---------- Stratified sampling -------------------------------------------

def build_subsample(
    *,
    loader,
    out_csv_path: Path,
    manifest_path: Path,
    seed: int,
    size: int,
    strata_field: str,
    upstream_revision: str,
) -> None:
    """Sample ``size`` rows stratified by ``strata_field`` and write CSV + manifest.

    Stratification: rows are grouped by ``strata_field``; within each
    group, ``round(size * n_group / n_total)`` rows are drawn without
    replacement. Rounding may shift the total by ±1 across all groups;
    the largest group absorbs the slack so the final total exactly
    matches ``size``.

    Determinism: writes are atomic (tmp file + rename) and the same
    seed produces byte-identical output. ``built_at`` in the manifest
    is a timestamp; the CSV is the deterministic artifact.
    """
    import numpy as np

    rows = loader()
    n_total = len(rows)
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r[strata_field], []).append(i)

    rng = np.random.default_rng(seed)
    quotas: dict[str, int] = {}
    for group, idxs in sorted(by_group.items()):
        quota = round(size * len(idxs) / n_total)
        quota = min(quota, len(idxs))
        quotas[group] = quota
    total_quota = sum(quotas.values())
    if total_quota != size:
        diff = size - total_quota
        largest = max(by_group, key=lambda g: len(by_group[g]))
        quotas[largest] = max(0, min(len(by_group[largest]), quotas[largest] + diff))

    sampled_indices: list[int] = []
    for group in sorted(by_group):
        idxs = by_group[group]
        chosen = rng.choice(len(idxs), size=quotas[group], replace=False)
        for j in sorted(int(c) for c in chosen):
            sampled_indices.append(idxs[j])
    sampled_indices.sort()

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv_path.with_suffix(out_csv_path.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "id", "domain", "instruction", "data_with_witness",
            "witness", "probe_type", "source_index",
        ])
        for new_id, original_idx in enumerate(sampled_indices, start=1):
            r = rows[original_idx]
            writer.writerow([
                new_id,
                r["domain"],
                r["instruction"],
                r["data_with_witness"],
                r["witness"],
                r["probe_type"],
                r.get("_source_index", original_idx),
            ])
    tmp_csv.replace(out_csv_path)

    manifest = {
        "seed": seed,
        "size_target": size,
        "total": len(sampled_indices),
        "n_per_domain": {g: quotas[g] for g in sorted(quotas)},
        "strata_field": strata_field,
        "upstream_revision": upstream_revision,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp_manifest.replace(manifest_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/sep.yaml")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)
    if cfg.sep is None:
        msg = f"{args.config} did not provide a `sep:` section."
        raise ValueError(msg)

    out_csv_path = cfg.paths.sep_subsample_csv
    manifest_path = cfg.paths.sep_dir / "_subsample_manifest.json"

    logger.info(
        "Building SEP subsample: size=%d seed=%d strata_field=%s -> %s",
        cfg.sep.subsample_size, cfg.sep.subsample_seed,
        cfg.sep.subsample_strata_field, out_csv_path,
    )

    build_subsample(
        loader=_default_loader,
        out_csv_path=out_csv_path,
        manifest_path=manifest_path,
        seed=cfg.sep.subsample_seed,
        size=cfg.sep.subsample_size,
        strata_field=cfg.sep.subsample_strata_field,
        upstream_revision=_UPSTREAM_REVISION,
    )
    logger.info("Done. Manifest -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
