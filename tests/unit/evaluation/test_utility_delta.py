"""Tests for Utility Delta aggregation."""

import pytest

from src.evaluation.utility_delta import compute_utility_delta


def test_no_drop_means_zero_delta() -> None:
    conflict = {"L0_vs_L1": 0.8, "L1_vs_L3": 0.9}
    reference = {"L0_vs_L1": 0.8, "L1_vs_L3": 0.9}
    out = compute_utility_delta(conflict, reference)
    assert out["per_pair_delta"]["L0_vs_L1"] == 0.0
    assert out["mean_delta"] == 0.0
    assert out["mean_abs_delta"] == 0.0


def test_signed_delta_is_conflict_minus_reference() -> None:
    conflict = {"L0_vs_L1": 0.7}
    reference = {"L0_vs_L1": 0.9}
    out = compute_utility_delta(conflict, reference)
    assert out["per_pair_delta"]["L0_vs_L1"] == pytest.approx(-0.2)
    assert out["mean_delta"] == pytest.approx(-0.2)
    assert out["mean_abs_delta"] == pytest.approx(0.2)


def test_pair_missing_from_reference_skipped() -> None:
    conflict = {"L0_vs_L1": 0.5, "L1_vs_L3": 0.5}
    reference = {"L0_vs_L1": 0.7}  # L1_vs_L3 not in reference split
    out = compute_utility_delta(conflict, reference)
    assert "L0_vs_L1" in out["per_pair_delta"]
    assert "L1_vs_L3" not in out["per_pair_delta"]


def test_mean_abs_delta_not_just_abs_of_mean() -> None:
    conflict = {"L0_vs_L1": 0.6, "L1_vs_L3": 0.4}
    reference = {"L0_vs_L1": 0.4, "L1_vs_L3": 0.6}
    out = compute_utility_delta(conflict, reference)
    # Per-pair: +0.2, -0.2 → mean = 0.0 but mean_abs = 0.2
    assert out["mean_delta"] == pytest.approx(0.0)
    assert out["mean_abs_delta"] == pytest.approx(0.2)
