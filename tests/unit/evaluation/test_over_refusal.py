"""Tests for ORR aggregation."""

from src.evaluation.over_refusal import compute_orr


def test_orr_no_refusals() -> None:
    records = [
        {"matched_conflict_id": "eval_0001", "is_refusal": False},
        {"matched_conflict_id": "eval_0002", "is_refusal": False},
    ]
    pair_lookup = {"eval_0001": "L0_vs_L1", "eval_0002": "L1_vs_L3"}
    out = compute_orr(records, pair_lookup)
    assert out["orr_overall"] == 0.0
    assert out["orr_per_pair"]["L0_vs_L1"] == 0.0


def test_orr_all_refusals() -> None:
    records = [
        {"matched_conflict_id": f"eval_000{i}", "is_refusal": True} for i in range(4)
    ]
    pair_lookup = {f"eval_000{i}": "L1_vs_L3" for i in range(4)}
    out = compute_orr(records, pair_lookup)
    assert out["orr_overall"] == 1.0
    assert out["orr_per_pair"]["L1_vs_L3"] == 1.0


def test_orr_partial_refusal() -> None:
    records = [
        {"matched_conflict_id": "eval_0001", "is_refusal": True},
        {"matched_conflict_id": "eval_0002", "is_refusal": False},
        {"matched_conflict_id": "eval_0003", "is_refusal": True},
        {"matched_conflict_id": "eval_0004", "is_refusal": False},
    ]
    pair_lookup = {f"eval_000{i}": "L0_vs_L4" for i in range(1, 5)}
    out = compute_orr(records, pair_lookup)
    assert out["orr_overall"] == 0.5


def test_orr_records_without_pair_skipped() -> None:
    records = [
        {"matched_conflict_id": "missing", "is_refusal": True},
        {"matched_conflict_id": "eval_0001", "is_refusal": False},
    ]
    pair_lookup = {"eval_0001": "L0_vs_L1"}
    out = compute_orr(records, pair_lookup)
    # Records with no pair lookup are excluded entirely.
    assert out["orr_overall"] == 0.0
    assert out["per_pair_count"]["L0_vs_L1"] == 1
