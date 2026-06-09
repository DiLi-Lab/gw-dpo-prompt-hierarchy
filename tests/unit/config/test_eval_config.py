"""Tests for EvalConfig dataclass and config loading."""

from pathlib import Path

from src.config.loader import load_config

BASE_YAML = Path("configs/base_linear.yaml")


def test_eval_config_from_yaml():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.eval.count_per_pair == 100
    assert cfg.eval.num_pairs == 10
    assert cfg.eval.reference_per_pair == 30
    assert cfg.eval.near_dedup_threshold == 0.85
    assert cfg.eval.scenario_model == "gpt-4o"
    assert cfg.eval.scenario_temperature == 0.7
    assert cfg.eval.gold_model == "claude-sonnet-4-20250514"
    assert cfg.eval.gold_temperature == 0.3
    assert cfg.eval.judge_min_score == 4
    assert cfg.eval.seed == 42


def test_eval_config_total_counts():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.eval.total_conflicts == 1000
    assert cfg.eval.total_aligned == 1000
    assert cfg.eval.total_reference == 300


def test_eval_config_from_eval_yaml():
    """eval.yaml only has the eval section; other sections need base_linear.yaml.

    Since we removed all defaults, loading eval.yaml alone would fail.
    This test verifies that eval.yaml values are correct by reading them
    directly rather than through the full config loader.
    """
    import yaml

    with open(Path("configs/eval.yaml")) as f:
        data = yaml.safe_load(f)
    assert data["eval"]["count_per_pair"] == 100
    assert data["eval"]["scenario_model"] == "gpt-4o"


def test_missing_field_raises_error():
    import pytest
    from src.config.hyperparameters import EvalConfig

    with pytest.raises(TypeError):
        EvalConfig()
