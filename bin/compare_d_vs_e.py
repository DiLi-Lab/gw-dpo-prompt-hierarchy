#!/usr/bin/env python3
"""Side-by-side per-pair comparison of (d) GW-DPO vs (e) 3-level GW-DPO.

Reads metrics.json from both runs, joins on conflict-pair key, and emits
a small JSON document summarising:
  - per-pair PPA delta (d - e)
  - bucketed delta by 3-level pair (intra-System vs representable)
  - macro PPA delta and WHS delta

Usage:
    python bin/compare_d_vs_e.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRA_SYSTEM_PAIRS = {"L0_vs_L1", "L0_vs_L2", "L1_vs_L2"}
REPRESENTABLE_PAIRS = {
    "L0_vs_L3", "L0_vs_L4", "L1_vs_L3", "L1_vs_L4",
    "L2_vs_L3", "L2_vs_L4", "L3_vs_L4",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--d-metrics", type=Path,
        default=PROJECT_ROOT / "evaluation" / "runs" / "dpo_final" / "metrics.json",
    )
    p.add_argument(
        "--e-metrics", type=Path,
        default=PROJECT_ROOT / "evaluation" / "runs" / "3level_gw_dpo_final" / "metrics.json",
    )
    p.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "evaluation" / "runs" / "3level_gw_dpo_final"
        / "d_vs_e_comparison.json",
    )
    args = p.parse_args()

    d = _load_metrics(args.d_metrics)
    e = _load_metrics(args.e_metrics)

    d_per_pair = d["ppa_per_pair"]
    e_per_pair = e["ppa_per_pair"]

    delta_per_pair = {
        pair: round(d_per_pair[pair] - e_per_pair[pair], 4)
        for pair in d_per_pair if pair in e_per_pair
    }
    intra_avg = sum(
        delta_per_pair[p] for p in INTRA_SYSTEM_PAIRS if p in delta_per_pair
    ) / max(1, sum(1 for p in INTRA_SYSTEM_PAIRS if p in delta_per_pair))
    rep_avg = sum(
        delta_per_pair[p] for p in REPRESENTABLE_PAIRS if p in delta_per_pair
    ) / max(1, sum(1 for p in REPRESENTABLE_PAIRS if p in delta_per_pair))

    summary = {
        "delta_per_pair": delta_per_pair,
        "intra_system_avg_delta": round(intra_avg, 4),
        "representable_avg_delta": round(rep_avg, 4),
        "macro_ppa_delta": round(d["ppa_macro"] - e["ppa_macro"], 4),
        "whs_delta": round(d["whs"] - e["whs"], 4),
        "d_metrics_source": str(args.d_metrics),
        "e_metrics_source": str(args.e_metrics),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
