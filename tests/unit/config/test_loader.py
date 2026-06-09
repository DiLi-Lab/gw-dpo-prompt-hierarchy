"""Tests for config loading from YAML with overrides."""

from pathlib import Path

import yaml

from src.config.loader import load_config

BASE_YAML = Path("configs/base_linear.yaml")


def test_load_from_base_yaml():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.model.torch_dtype == "bfloat16"
    assert cfg.sft.learning_rate == 2e-5
    assert cfg.dpo.beta == 0.1
    assert isinstance(cfg.paths.project_root, Path)


def test_load_no_yaml_raises_error():
    """Loading without YAML and no defaults must fail."""
    import pytest

    with pytest.raises(TypeError):
        load_config()


def test_load_from_yaml_with_partial_overrides(tmp_path):
    """YAML with all required fields + overrides."""
    # Copy base config and modify
    with open(BASE_YAML) as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["model"]["model_name_or_path"] = "my-custom-model"
    yaml_content["dpo"]["gravity_alpha"] = 2.0

    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump(yaml_content))

    cfg = load_config(config_path=yaml_file)
    assert cfg.model.model_name_or_path == "my-custom-model"
    assert cfg.dpo.gravity_alpha == 2.0
    assert cfg.sft.learning_rate == 2e-5


def test_load_with_overrides():
    overrides = ["model.torch_dtype=float32", "sft.num_epochs=5"]
    cfg = load_config(config_path=BASE_YAML, overrides=overrides)
    assert cfg.model.torch_dtype == "float32"
    assert cfg.sft.num_epochs == 5


def test_load_yaml_plus_overrides(tmp_path):
    with open(BASE_YAML) as f:
        yaml_content = yaml.safe_load(f)
    yaml_content["model"]["torch_dtype"] = "float16"

    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump(yaml_content))

    cfg = load_config(
        config_path=yaml_file,
        overrides=["model.torch_dtype=bfloat16"],
    )
    assert cfg.model.torch_dtype == "bfloat16"


def test_load_invalid_override_key():
    import pytest
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(config_path=BASE_YAML, overrides=["nonexistent.key=value"])


def test_load_invalid_yaml_path():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config(config_path=Path("/nonexistent/config.yaml"))


def test_tuple_fields_converted():
    cfg = load_config(config_path=BASE_YAML)
    assert isinstance(cfg.sft.lora_target_modules, tuple)
    assert isinstance(cfg.dpo.lora_target_modules, tuple)
