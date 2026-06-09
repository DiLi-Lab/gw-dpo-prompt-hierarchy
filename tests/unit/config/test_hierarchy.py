"""Tests for hierarchy relationships and conflict pairs."""

from src.config.hierarchy import CONFLICT_PAIRS, get_level_gap, is_valid_conflict


def test_conflict_pairs():
    assert len(CONFLICT_PAIRS) == 10
    for victim, attacker in CONFLICT_PAIRS:
        assert victim < attacker


def test_get_level_gap():
    assert get_level_gap(0, 4) == 4
    assert get_level_gap(1, 3) == 2
    assert get_level_gap(3, 4) == 1


def test_get_level_gap_invalid():
    import pytest
    with pytest.raises(ValueError, match="victim.*must be less"):
        get_level_gap(3, 1)
    with pytest.raises(ValueError, match="victim.*must be less"):
        get_level_gap(2, 2)


def test_is_valid_conflict():
    assert is_valid_conflict(0, 4) is True
    assert is_valid_conflict(1, 3) is True
    assert is_valid_conflict(3, 1) is False
    assert is_valid_conflict(2, 2) is False
    assert is_valid_conflict(-1, 3) is False
    assert is_valid_conflict(0, 5) is False
