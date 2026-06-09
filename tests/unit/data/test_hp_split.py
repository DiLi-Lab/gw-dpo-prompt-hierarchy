"""Unit tests for the stratified HP-select split."""

import random

import pytest

from src.data.dpo.hp_split import build_hp_split


def _make_records(counts: dict[tuple[int, bool], int]) -> list[dict]:
    """Build synthetic records with given (level_gap, is_calibration) counts."""
    records = []
    for (gap, is_cal), n in counts.items():
        for i in range(n):
            records.append(
                {"level_gap": gap, "is_calibration": is_cal, "payload": f"g{gap}_c{is_cal}_{i}"},
            )
    random.Random(0).shuffle(records)
    return records


def test_total_size_matches_target():
    records = _make_records({(0, True): 1810, (1, False): 219, (2, False): 260,
                              (3, False): 208, (4, False): 76})
    hp_idx, val_idx, counts = build_hp_split(records, target_size=1000, seed=42)
    assert len(hp_idx) == 1000
    assert len(val_idx) == len(records) - 1000


def test_disjoint_and_complete():
    records = _make_records({(0, True): 50, (1, False): 50, (2, False): 50})
    hp_idx, val_idx, _ = build_hp_split(records, target_size=60, seed=42)
    assert set(hp_idx).isdisjoint(set(val_idx))
    assert set(hp_idx) | set(val_idx) == set(range(len(records)))


def test_stratification_proportional():
    records = _make_records({(0, True): 1810, (1, False): 219, (2, False): 260,
                              (3, False): 208, (4, False): 76})
    hp_idx, _, counts = build_hp_split(records, target_size=1000, seed=42)
    for (gap, is_cal), n_chosen in counts.items():
        source_n = {(0, True): 1810, (1, False): 219, (2, False): 260,
                     (3, False): 208, (4, False): 76}[(gap, is_cal)]
        expected = source_n * 1000 / 2573
        assert abs(n_chosen - expected) <= 1, (gap, is_cal, n_chosen, expected)
    assert sum(counts.values()) == 1000


def test_same_seed_same_split():
    records = _make_records({(0, True): 100, (1, False): 50})
    hp1, val1, _ = build_hp_split(records, target_size=30, seed=42)
    hp2, val2, _ = build_hp_split(records, target_size=30, seed=42)
    assert hp1 == hp2
    assert val1 == val2


def test_different_seed_different_split():
    records = _make_records({(0, True): 100, (1, False): 50})
    hp1, _, _ = build_hp_split(records, target_size=30, seed=42)
    hp2, _, _ = build_hp_split(records, target_size=30, seed=43)
    assert hp1 != hp2


def test_bucket_smaller_than_allocation_clamps():
    """If a bucket has fewer records than its proportional allocation, clamp."""
    records = _make_records({(0, True): 100, (4, False): 5})
    hp_idx, val_idx, counts = build_hp_split(records, target_size=95, seed=42)
    assert counts[(4, False)] <= 5
    assert len(hp_idx) + len(val_idx) == len(records)


def test_target_larger_than_source_caps_at_source():
    records = _make_records({(0, True): 10, (1, False): 5})
    hp_idx, val_idx, _ = build_hp_split(records, target_size=100, seed=42)
    assert len(hp_idx) == 15
    assert len(val_idx) == 0
