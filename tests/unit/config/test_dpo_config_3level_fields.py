"""Tests for the three new DPOConfig fields used by ablation (e).

Defaults must preserve the 5-level behaviour of every existing config.
"""

from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config


@pytest.fixture
def base_config_path():
    return Path("configs/base_linear.yaml")


def test_base_yaml_loads_with_new_fields(base_config_path):
    cfg = load_config(base_config_path)
    # Defaults preserve 5-level behaviour.
    assert cfg.dpo.train_split_name == "train"
    assert cfg.dpo.val_split_name == "val"
    assert cfg.dpo.curriculum_min_gap_by_stage is None


def test_3level_yaml_overrides_fields(tmp_path):
    yaml_path = tmp_path / "ablation_e.yaml"
    base = yaml.safe_load(Path("configs/base_linear.yaml").read_text())
    base["dpo"]["train_split_name"] = "train_3level"
    base["dpo"]["val_split_name"] = "val_3level"
    base["dpo"]["curriculum_min_gap_by_stage"] = {1: 2}
    base["dpo"]["num_curriculum_stages"] = 2
    yaml_path.write_text(yaml.safe_dump(base))
    cfg = load_config(yaml_path)
    assert cfg.dpo.train_split_name == "train_3level"
    assert cfg.dpo.val_split_name == "val_3level"
    assert cfg.dpo.curriculum_min_gap_by_stage == {1: 2}
    assert cfg.dpo.num_curriculum_stages == 2
