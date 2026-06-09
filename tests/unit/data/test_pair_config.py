"""Tests for DPO pair configuration definitions."""

import pytest

from src.data.dpo.pair_config import PairConfig, ALL_PAIR_CONFIGS, get_config_by_name


def test_pair_config_fields():
    cfg = PairConfig(
        name="L0_vs_L1",
        victim_level=0,
        attacker_level=1,
        target_count=500,
        category="pairwise",
        yw_strategy="claude_distillation",
        yl_strategy="gpt4o_mini",
        l2_conflict=False,
        l2_conflict_attribute=None,
        injection_method=None,
        injection_target_level=None,
        needs_summarisation_rows=False,
        phase=3,
    )
    assert cfg.level_gap == 1
    assert cfg.margin == 1.0


def test_all_pair_configs_count():
    assert len(ALL_PAIR_CONFIGS) == 12


def test_all_pair_configs_total_target():
    total = sum(c.target_count for c in ALL_PAIR_CONFIGS)
    assert total == 10_000


def test_pair_configs_cover_all_10_pairwise():
    pairwise = [c for c in ALL_PAIR_CONFIGS if c.category == "pairwise"]
    pairs = {(c.victim_level, c.attacker_level) for c in pairwise}
    expected = {(i, j) for i in range(5) for j in range(i + 1, 5)}
    assert pairs == expected


def test_calibration_config():
    cal = get_config_by_name("calibration")
    assert cal.category == "calibration"
    assert cal.target_count == 2000
    assert cal.level_gap == 0
    assert cal.margin == 0.0
    assert cal.yw_strategy == "base_dataset", (
        f"calibration yw_strategy should be 'base_dataset', got '{cal.yw_strategy}'"
    )
    assert cal.yl_strategy == "template", (
        f"calibration yl_strategy should be 'template', got '{cal.yl_strategy}'"
    )


def test_cascading_config():
    casc = get_config_by_name("cascading")
    assert casc.category == "cascading"
    assert casc.target_count == 1000


def test_get_config_by_name():
    cfg = get_config_by_name("L1_vs_L3")
    assert cfg.victim_level == 1
    assert cfg.attacker_level == 3
    assert cfg.target_count == 1500


def test_get_config_by_name_missing():
    with pytest.raises(KeyError):
        get_config_by_name("nonexistent")


def test_phase_assignments():
    phase1 = [c for c in ALL_PAIR_CONFIGS if c.phase == 1]
    phase2 = [c for c in ALL_PAIR_CONFIGS if c.phase == 2]
    phase3 = [c for c in ALL_PAIR_CONFIGS if c.phase == 3]
    assert len(phase1) == 1
    assert len(phase2) == 7
    assert len(phase3) == 4


def test_no_configs_require_summarisation_rows():
    """Summarisation filter was removed — no config should need it."""
    for cfg in ALL_PAIR_CONFIGS:
        assert cfg.needs_summarisation_rows is False, (
            f"{cfg.name} still has needs_summarisation_rows=True"
        )


def test_l0_vs_l1_and_l0_vs_l2_are_scenario_driven():
    l0_l1 = get_config_by_name("L0_vs_L1")
    l0_l2 = get_config_by_name("L0_vs_L2")
    assert l0_l1.scenario_driven is True
    assert l0_l2.scenario_driven is True


def test_calibration_config_levels_match_builder():
    """Calibration is about L3 interpretation, so victim=attacker=3."""
    cal = get_config_by_name("calibration")
    assert cal.victim_level == 3, f"Expected victim_level=3, got {cal.victim_level}"
    assert cal.attacker_level == 3, f"Expected attacker_level=3, got {cal.attacker_level}"


def test_non_scenario_configs_are_not_scenario_driven():
    for cfg in ALL_PAIR_CONFIGS:
        if cfg.name not in ("L0_vs_L1", "L0_vs_L2"):
            assert cfg.scenario_driven is False, f"{cfg.name} should not be scenario_driven"


def test_l1_vs_l4_uses_claude_distillation():
    cfg = get_config_by_name("L1_vs_L4")
    assert cfg.yw_strategy == "claude_distillation"
