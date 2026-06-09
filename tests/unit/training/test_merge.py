"""Tests for post-training merge logic."""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from src.training.merge import (
    _remap_peft_key_to_plain,
    save_merged_model_with_ise,
    sync_peft_base_weights_to_plain,
)


def test_save_merged_model_with_ise(tmp_path):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    ise_state = {"segment_embedding.weight": torch.randn(6, 16)}
    ise_path = tmp_path / "checkpoint" / "ise_weights.pt"
    ise_path.parent.mkdir(parents=True)
    torch.save(ise_state, ise_path)

    output_dir = tmp_path / "merged"

    save_merged_model_with_ise(
        model=mock_model,
        tokenizer=mock_tokenizer,
        ise_weights_path=ise_path,
        output_dir=output_dir,
    )

    mock_model.save_pretrained.assert_called_once_with(str(output_dir))
    mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
    assert (output_dir / "ise_weights.pt").exists()

    loaded = torch.load(output_dir / "ise_weights.pt", weights_only=True)
    assert "segment_embedding.weight" in loaded


class TestRemapPeftKeyToPlain:
    """Unit tests for the PEFT → plain state-dict key remapper."""

    @pytest.mark.parametrize(
        "peft_key,expected",
        [
            # LoRA-wrapped linear: strip prefix + .base_layer
            (
                "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
            # trainable_token_indices-wrapped embedding: also strip .token_adapter
            (
                "base_model.model.model.embed_tokens.token_adapter.base_layer.weight",
                "model.embed_tokens.weight",
            ),
            (
                "base_model.model.lm_head.token_adapter.base_layer.weight",
                "lm_head.weight",
            ),
            # Non-targeted modules: strip prefix only
            (
                "base_model.model.model.layers.0.input_layernorm.weight",
                "model.layers.0.input_layernorm.weight",
            ),
            (
                "base_model.model.model.layers.0.input_layernorm.bias",
                "model.layers.0.input_layernorm.bias",
            ),
        ],
    )
    def test_base_weight_keys_are_remapped(self, peft_key, expected):
        assert _remap_peft_key_to_plain(peft_key) == expected

    @pytest.mark.parametrize(
        "peft_key",
        [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
            "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta.default",
        ],
    )
    def test_adapter_only_keys_return_none(self, peft_key):
        assert _remap_peft_key_to_plain(peft_key) is None

    def test_non_peft_prefix_returns_none(self):
        assert _remap_peft_key_to_plain("some.other.module.weight") is None


def _tiny_llama() -> LlamaForCausalLM:
    """Build a very small LlamaForCausalLM for fast integration tests."""
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


class TestSyncPeftBaseWeightsToPlain:
    """Integration tests for syncing a PEFT policy into a plain reference."""

    def _build_policy_and_reference(self):
        """Create a PEFT-wrapped policy and a plain reference from the same base."""
        base = _tiny_llama()
        reference = copy.deepcopy(base)
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            trainable_token_indices={
                "embed_tokens": [0, 1, 2],
                "lm_head": [0, 1, 2],
            },
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_cfg)
        return policy, reference

    def _perturb_trainable_params(self, policy):
        """Simulate DPO training by nudging all trainable params."""
        with torch.no_grad():
            for _, p in policy.named_parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.3)

    def test_lora_targeted_weights_are_synced(self):
        policy, reference = self._build_policy_and_reference()
        self._perturb_trainable_params(policy)

        ref_q_before = reference.model.layers[0].self_attn.q_proj.weight.data.clone()

        sync_peft_base_weights_to_plain(policy, reference)

        # q_proj is LoRA-targeted; merging the delta must change the ref weight
        ref_q_after = reference.model.layers[0].self_attn.q_proj.weight.data
        assert not torch.equal(ref_q_after, ref_q_before)

        # And after sync, policy (merged) and reference must match on q_proj
        policy.merge_adapter()
        policy_q = (
            policy.base_model.model.model.layers[0]
            .self_attn.q_proj.base_layer.weight.data
        )
        torch.testing.assert_close(ref_q_after, policy_q)
        policy.unmerge_adapter()

    def test_trainable_token_rows_are_synced(self):
        policy, reference = self._build_policy_and_reference()
        self._perturb_trainable_params(policy)

        ref_emb_before = reference.model.embed_tokens.weight.data.clone()
        ref_lm_before = reference.lm_head.weight.data.clone()

        sync_peft_base_weights_to_plain(policy, reference)

        # Trainable rows (0, 1, 2) should differ; untrained rows should match original
        for row in (0, 1, 2):
            assert not torch.equal(
                reference.model.embed_tokens.weight.data[row], ref_emb_before[row],
            )
            assert not torch.equal(
                reference.lm_head.weight.data[row], ref_lm_before[row],
            )
        for row in (10, 20, 31):
            torch.testing.assert_close(
                reference.model.embed_tokens.weight.data[row], ref_emb_before[row],
            )
            torch.testing.assert_close(
                reference.lm_head.weight.data[row], ref_lm_before[row],
            )

    def test_policy_is_unmerged_after_sync(self):
        """The policy should be usable for training again after the sync."""
        policy, reference = self._build_policy_and_reference()
        self._perturb_trainable_params(policy)

        # Snapshot pre-sync un-merged base_layer weight
        pre_sync = (
            policy.base_model.model.model.layers[0]
            .self_attn.q_proj.base_layer.weight.data.clone()
        )

        sync_peft_base_weights_to_plain(policy, reference)

        post_sync = (
            policy.base_model.model.model.layers[0]
            .self_attn.q_proj.base_layer.weight.data
        )
        # base_layer.weight must be back to its pre-merge value
        torch.testing.assert_close(post_sync, pre_sync)

    def test_non_targeted_modules_stay_unchanged(self):
        """Layernorm weights aren't PEFT-wrapped and shouldn't break the sync."""
        policy, reference = self._build_policy_and_reference()
        self._perturb_trainable_params(policy)

        ref_ln_before = reference.model.layers[0].input_layernorm.weight.data.clone()
        sync_peft_base_weights_to_plain(policy, reference)
        # Layernorms aren't touched by training in this setup, so the policy's
        # value equals the reference's original value — sync must preserve it.
        torch.testing.assert_close(
            reference.model.layers[0].input_layernorm.weight.data, ref_ln_before,
        )

    def test_repeated_sync_keeps_reference_in_step(self):
        """Simulate two stage transitions: each sync must track the latest policy."""
        policy, reference = self._build_policy_and_reference()

        self._perturb_trainable_params(policy)
        sync_peft_base_weights_to_plain(policy, reference)

        # Second round of training
        self._perturb_trainable_params(policy)
        sync_peft_base_weights_to_plain(policy, reference)

        policy.merge_adapter()
        try:
            for (pn, pm_), (rn, rm) in zip(
                policy.base_model.model.named_modules(),
                reference.named_modules(),
            ):
                if hasattr(pm_, "base_layer") and hasattr(rm, "weight"):
                    torch.testing.assert_close(
                        rm.weight.data, pm_.base_layer.weight.data,
                    )
        finally:
            policy.unmerge_adapter()
