"""MT-Bench metric aggregation.

Consumes per-turn judge records (from scoring.jsonl) and emits the
metrics.json payload documented in the design at
``docs/superpowers/specs/2026-05-06-mt-bench-design.md`` §6.3.

Records with ``score=None`` (parse failures) are excluded from the
numerator and denominator of every mean — never zeroed. The count of
parse failures is preserved as ``n_judge_parse_failures`` plus a
breakdown by category and turn.
"""

from collections import defaultdict
from typing import Iterable


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _diff(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def _score_block(records: list[dict]) -> dict:
    """Compute scored-counts + per-turn / overall means over a list of records."""
    scored = [r for r in records if r["score"] is not None]
    return {
        "n_turns_scored": len(scored),
        "mean_overall": _mean([r["score"] for r in scored]),
        "turn1_mean": _mean([r["score"] for r in scored if r["turn"] == 1]),
        "turn2_mean": _mean([r["score"] for r in scored if r["turn"] == 2]),
    }


def aggregate_mt_bench_metrics(records: Iterable[dict]) -> dict:
    """Aggregate per-turn judge records into headline metrics.

    Each record must carry: ``question_id``, ``turn`` (int 1 or 2),
    ``category`` (str), ``score`` (float | None — None on parse failure),
    ``parse_error`` (bool).
    """
    records = list(records)
    n_turns_total = len(records)
    qids = {r["question_id"] for r in records}

    overall = _score_block(records)
    n_failures = n_turns_total - overall["n_turns_scored"]
    drop = _diff(overall["turn1_mean"], overall["turn2_mean"])

    # Per-category aggregates.
    by_cat_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat_records[r["category"]].append(r)
    by_category = {cat: _score_block(group) for cat, group in by_cat_records.items()}

    # Parse-error breakdown.
    pe_by_cat: dict[str, int] = defaultdict(int)
    pe_by_turn: dict[str, int] = defaultdict(int)
    for r in records:
        if r["score"] is None:
            pe_by_cat[r["category"]] += 1
            pe_by_turn[str(r["turn"])] += 1

    return {
        "n_questions": len(qids),
        "n_turns_total": n_turns_total,
        "n_turns_scored": overall["n_turns_scored"],
        "n_judge_parse_failures": n_failures,

        "overall_mean": overall["mean_overall"],
        "turn1_mean": overall["turn1_mean"],
        "turn2_mean": overall["turn2_mean"],
        "turn1_minus_turn2_drop": drop,

        "by_category": by_category,

        "parse_error_breakdown": {
            "by_category": dict(pe_by_cat),
            "by_turn": dict(pe_by_turn),
        },
    }
