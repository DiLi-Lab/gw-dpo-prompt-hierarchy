"""Tests for Weighted Hierarchy Score aggregation."""

import pytest

from src.evaluation.whs import compute_whs, gap_for_pair


def test_gap_for_pair() -> None:
    assert gap_for_pair("L0_vs_L1") == 1
    assert gap_for_pair("L0_vs_L4") == 4
    assert gap_for_pair("L2_vs_L3") == 1
    assert gap_for_pair("L1_vs_L4") == 3


def test_whs_perfect_score_is_one() -> None:
    per_pair = {p: 1.0 for p in [
        "L0_vs_L1", "L0_vs_L2", "L0_vs_L3", "L0_vs_L4",
        "L1_vs_L2", "L1_vs_L3", "L1_vs_L4",
        "L2_vs_L3", "L2_vs_L4", "L3_vs_L4",
    ]}
    per_pair_count = {p: 10 for p in per_pair}
    out = compute_whs(per_pair, per_pair_count)
    assert out["whs"] == pytest.approx(1.0)
    assert out["macro_avg"] == 1.0


def test_whs_weights_by_gap() -> None:
    """Compliance only on the gap-4 pair should yield WHS = 4/20 = 0.2."""
    per_pair = {p: 0.0 for p in [
        "L0_vs_L1", "L0_vs_L2", "L0_vs_L3", "L0_vs_L4",
        "L1_vs_L2", "L1_vs_L3", "L1_vs_L4",
        "L2_vs_L3", "L2_vs_L4", "L3_vs_L4",
    ]}
    per_pair["L0_vs_L4"] = 1.0  # only gap-4 pair
    per_pair_count = {p: 10 for p in per_pair}
    out = compute_whs(per_pair, per_pair_count)
    # WHS = sum(gap*acc) / sum(gap) = (4*1)/(1+2+3+4+1+2+3+1+2+1) = 4/20
    assert out["whs"] == pytest.approx(0.2)
    # Macro avg = 1/10 (one pair at 1.0, nine at 0.0)
    assert out["macro_avg"] == pytest.approx(0.1)


def test_whs_unpopulated_pairs_excluded_from_macro() -> None:
    per_pair = {p: 0.0 for p in [
        "L0_vs_L1", "L0_vs_L2", "L0_vs_L3", "L0_vs_L4",
        "L1_vs_L2", "L1_vs_L3", "L1_vs_L4",
        "L2_vs_L3", "L2_vs_L4", "L3_vs_L4",
    ]}
    per_pair["L0_vs_L4"] = 1.0
    counts = {p: 0 for p in per_pair}
    counts["L0_vs_L4"] = 5
    out = compute_whs(per_pair, counts)
    # Only L0_vs_L4 populated; macro_avg = 1.0; WHS = 1.0 (4/4)
    assert out["macro_avg"] == 1.0
    assert out["whs"] == pytest.approx(1.0)


def test_whs_per_gap_bucket() -> None:
    per_pair = {
        "L0_vs_L1": 0.5, "L1_vs_L2": 0.7, "L2_vs_L3": 0.9, "L3_vs_L4": 0.3,
        "L0_vs_L2": 0.0, "L1_vs_L3": 0.0, "L2_vs_L4": 0.0,
        "L0_vs_L3": 0.0, "L1_vs_L4": 0.0, "L0_vs_L4": 0.0,
    }
    counts = {p: 1 for p in per_pair}
    out = compute_whs(per_pair, counts)
    # Gap 1 has four pairs: (0.5+0.7+0.9+0.3)/4 = 0.6
    assert out["per_gap_avg"][1] == pytest.approx(0.6)
    assert out["per_gap_avg"][4] == 0.0
