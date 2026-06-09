"""Tests for validation pair configurations."""

from src.data.dpo.pair_config import ALL_PAIR_CONFIGS, VAL_PAIR_CONFIGS, get_pair_configs


def test_val_pair_configs_has_same_names():
    """VAL_PAIR_CONFIGS must cover the same pair types as ALL_PAIR_CONFIGS."""
    train_names = {c.name for c in ALL_PAIR_CONFIGS}
    val_names = {c.name for c in VAL_PAIR_CONFIGS}
    assert train_names == val_names


def test_val_pair_configs_total_count():
    """VAL_PAIR_CONFIGS should total ~1000."""
    total = sum(c.target_count for c in VAL_PAIR_CONFIGS)
    assert total == 1000


def test_val_pair_configs_stratification():
    """Check key stratified counts."""
    by_name = {c.name: c.target_count for c in VAL_PAIR_CONFIGS}
    assert by_name["L1_vs_L3"] == 150
    assert by_name["L1_vs_L4"] == 100
    assert by_name["L3_vs_L4"] == 100
    assert by_name["calibration"] == 200
    assert by_name["cascading"] == 100
    for name in ("L0_vs_L1", "L0_vs_L2", "L0_vs_L3", "L0_vs_L4", "L1_vs_L2", "L2_vs_L3", "L2_vs_L4"):
        assert by_name[name] == 50


def test_val_pair_configs_preserve_phases():
    """VAL configs must have the same phase assignments as train configs."""
    train_phases = {c.name: c.phase for c in ALL_PAIR_CONFIGS}
    for c in VAL_PAIR_CONFIGS:
        assert c.phase == train_phases[c.name], f"{c.name} phase mismatch"


def test_get_pair_configs_returns_correct_list():
    """get_pair_configs selects the right config list by split."""
    assert get_pair_configs(split=None) is ALL_PAIR_CONFIGS
    assert get_pair_configs(split="train") is ALL_PAIR_CONFIGS
    assert get_pair_configs(split="val") is VAL_PAIR_CONFIGS
