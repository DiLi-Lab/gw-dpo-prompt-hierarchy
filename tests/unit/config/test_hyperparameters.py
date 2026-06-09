"""Tests for hyperparameter configuration loaded from base_linear.yaml."""

from pathlib import Path

from src.config.loader import load_config

BASE_YAML = Path("configs/base_linear.yaml")


def test_model_config_from_yaml():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.model.torch_dtype == "bfloat16"
    assert cfg.model.num_segments == 6
    assert cfg.model.model_name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.model.token_embedding_init == "mean"
    assert cfg.model.ise_embedding_init == "normal"
    assert cfg.model.ise_init_std == 0.01


def test_model_config_custom():
    from src.config.hyperparameters import ModelConfig

    cfg = ModelConfig(
        model_name_or_path="my-model",
        torch_dtype="float32",
        num_segments=6,
        token_embedding_init="mean",
        ise_embedding_init="normal",
        ise_init_std=0.01,
        use_ise=True,
    )
    assert cfg.model_name_or_path == "my-model"
    assert cfg.torch_dtype == "float32"
    assert cfg.use_ise is True


def test_sft_config_from_yaml():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.sft.learning_rate == 2e-5
    assert cfg.sft.num_epochs == 3
    assert cfg.sft.lora_rank == 64
    assert cfg.sft.lora_alpha == 128
    assert cfg.sft.lora_dropout == 0.1
    assert cfg.sft.max_seq_length == 4096
    assert cfg.sft.per_device_batch_size == 4
    assert cfg.sft.gradient_accumulation_steps == 8
    assert cfg.sft.effective_batch_size == 32
    assert cfg.sft.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert cfg.sft.task_type == "CAUSAL_LM"
    assert cfg.sft.save_steps == 50
    assert cfg.sft.eval_steps == 50
    assert cfg.sft.remove_unused_columns is False


def test_dpo_config_from_yaml():
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.dpo.beta == 0.1
    assert cfg.dpo.gravity_alpha == 1.0
    assert cfg.dpo.learning_rate == 5e-5
    assert cfg.dpo.num_curriculum_stages == 3
    assert cfg.dpo.curriculum_enabled is True
    assert cfg.dpo.max_seq_length == 2048
    assert cfg.dpo.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert cfg.dpo.remove_unused_columns is False


def test_final_stage_index_default():
    """With the curriculum enabled, final_stage_index == num_curriculum_stages."""
    cfg = load_config(config_path=BASE_YAML)
    assert cfg.dpo.curriculum_enabled is True
    assert cfg.dpo.final_stage_index == cfg.dpo.num_curriculum_stages


def test_final_stage_index_disabled():
    """When curriculum is disabled, final_stage_index collapses to 1."""
    cfg = load_config(
        config_path=BASE_YAML,
        overrides=["dpo.curriculum_enabled=false"],
    )
    assert cfg.dpo.curriculum_enabled is False
    assert cfg.dpo.final_stage_index == 1


def test_final_stage_index_disabled_ignores_num_stages():
    """final_stage_index=1 regardless of num_curriculum_stages when disabled."""
    cfg = load_config(
        config_path=BASE_YAML,
        overrides=[
            "dpo.curriculum_enabled=false",
            "dpo.num_curriculum_stages=3",
        ],
    )
    assert cfg.dpo.final_stage_index == 1


def test_missing_field_raises_error():
    """Config instantiation without required fields must fail."""
    import pytest
    from src.config.hyperparameters import ModelConfig

    with pytest.raises(TypeError):
        ModelConfig()
