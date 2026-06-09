"""Tests for LoRA config builder."""

from pathlib import Path

from peft import LoraConfig

from src.config.loader import load_config
from src.training.lora_config import build_lora_config

BASE_YAML = Path("configs/base_linear.yaml")


def test_build_lora_config_from_sft():
    cfg = load_config(config_path=BASE_YAML)
    lora_cfg = build_lora_config(cfg.sft)
    assert isinstance(lora_cfg, LoraConfig)
    assert lora_cfg.r == 64
    assert lora_cfg.lora_alpha == 128
    assert lora_cfg.lora_dropout == 0.1
    assert lora_cfg.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert lora_cfg.modules_to_save is None
    assert lora_cfg.task_type == "CAUSAL_LM"


def test_build_lora_config_from_dpo():
    cfg = load_config(config_path=BASE_YAML)
    lora_cfg = build_lora_config(cfg.dpo)
    assert isinstance(lora_cfg, LoraConfig)
    assert lora_cfg.r == 64
    assert lora_cfg.modules_to_save is None


def test_build_lora_config_with_special_token_ids():
    cfg = load_config(config_path=BASE_YAML)
    token_ids = [128256, 128257, 128258, 128259]
    lora_cfg = build_lora_config(cfg.sft, special_token_ids=token_ids)
    assert lora_cfg.trainable_token_indices == {
        "embed_tokens": token_ids,
        "lm_head": token_ids,
    }
    assert lora_cfg.modules_to_save is None


def test_build_lora_config_with_tied_embeddings():
    """When tie_word_embeddings=True, lm_head must not be in
    trainable_token_indices to avoid double-wrapping in PEFT."""
    cfg = load_config(config_path=BASE_YAML)
    token_ids = [128256, 128257, 128258, 128259]
    lora_cfg = build_lora_config(
        cfg.sft, special_token_ids=token_ids, tie_word_embeddings=True,
    )
    assert lora_cfg.trainable_token_indices == {
        "embed_tokens": token_ids,
    }
    assert "lm_head" not in lora_cfg.trainable_token_indices
