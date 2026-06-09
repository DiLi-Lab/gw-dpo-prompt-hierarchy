"""Tests for the DPO hierarchy-aware data collator."""

import pytest
import torch
from transformers import AutoTokenizer

from src.training.dpo_data_collator import (
    DPOHierarchyCollator,
    compute_margin,
)


def _make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer-5level")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _make_dpo_example(tokenizer, prompt_text: str, chosen_text: str, rejected_text: str,
                      margin: float = 2.0,
                      victim_level: int | None = None,
                      attacker_level: int | None = None,
                      is_calibration: bool = False) -> dict:
    """Create a tokenized DPO example in TRL's _prepare_dataset output format."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
    example = {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "margin": margin,
    }
    if victim_level is not None:
        example["victim_level"] = victim_level
    if attacker_level is not None:
        example["attacker_level"] = attacker_level
    if is_calibration:
        example["is_calibration"] = True
    return example


class TestDPOHierarchyCollator:
    """Tests for the DPO collator with segment IDs and margins."""

    def test_returns_segment_ids(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Be safe.<|L0_END|><|L1_START|>System.<|L1_END|>",
            "<|RESP_START|>Good response.<|RESP_END|>",
            "<|RESP_START|>Bad response.<|RESP_END|>",
        )
        batch = collator([example])
        assert "segment_ids" in batch
        assert isinstance(batch["segment_ids"], torch.Tensor)

    def test_segment_ids_shape_matches_input_ids(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>Chosen.<|RESP_END|>",
            "<|RESP_START|>Rejected.<|RESP_END|>",
        )
        batch = collator([example])
        assert batch["segment_ids"].shape == batch["input_ids"].shape

    def test_returns_margin(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>Good.<|RESP_END|>",
            "<|RESP_START|>Bad.<|RESP_END|>",
            margin=3.0,
        )
        batch = collator([example])
        assert "margin" in batch
        assert batch["margin"].shape == (1,)
        assert batch["margin"][0].item() == 3.0

    def test_batch_margin_values(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        e1 = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>A.<|RESP_END|>",
            "<|RESP_START|>B.<|RESP_END|>",
            margin=1.0,
        )
        e2 = _make_dpo_example(
            tokenizer,
            "<|L1_START|>System.<|L1_END|>",
            "<|RESP_START|>C.<|RESP_END|>",
            "<|RESP_START|>D.<|RESP_END|>",
            margin=4.0,
        )
        batch = collator([e1, e2])
        assert batch["margin"].shape == (2,)
        assert batch["margin"][0].item() == 1.0
        assert batch["margin"][1].item() == 4.0

    def test_segment_ids_reflect_hierarchy_levels(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|><|L1_START|>Sys.<|L1_END|>",
            "<|RESP_START|>OK.<|RESP_END|>",
            "<|RESP_START|>Bad.<|RESP_END|>",
        )
        batch = collator([example])
        seg_ids = batch["segment_ids"]
        # First half is prompt+chosen, second half is prompt+rejected
        # Both should contain segment 0 (L0) and segment 1 (L1) in the prompt portion
        chosen_segs = seg_ids[0].tolist()
        rejected_segs = seg_ids[1].tolist()
        assert 0 in chosen_segs, "L0 segment not found in chosen sequence"
        assert 1 in chosen_segs, "L1 segment not found in chosen sequence"
        assert 0 in rejected_segs, "L0 segment not found in rejected sequence"
        assert 1 in rejected_segs, "L1 segment not found in rejected sequence"

    def test_batch_size_doubled_in_output(self):
        """Output batch has 2*N rows: N chosen + N rejected."""
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>Good.<|RESP_END|>",
            "<|RESP_START|>Bad.<|RESP_END|>",
        )
        batch = collator([example, example])
        # 2 examples → 4 rows (2 chosen + 2 rejected)
        assert batch["input_ids"].shape[0] == 4
        assert batch["segment_ids"].shape[0] == 4

    def test_without_margin_field(self):
        """Collator works even when margin is absent from examples."""
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
        example = {
            "prompt_ids": tokenizer.encode("<|L0_START|>Rule.<|L0_END|>",
                                           add_special_tokens=False),
            "chosen_ids": tokenizer.encode("<|RESP_START|>A.<|RESP_END|>",
                                           add_special_tokens=False),
            "rejected_ids": tokenizer.encode("<|RESP_START|>B.<|RESP_END|>",
                                             add_special_tokens=False),
        }
        batch = collator([example])
        assert "segment_ids" in batch
        assert "margin" not in batch


class TestComputeMargin:
    """Pure-function tests for the margin schedule formulas."""

    @pytest.mark.parametrize(
        ("victim", "attacker", "expected"),
        [
            (0, 1, 1.0),  # L0 vs L1, gap 1
            (0, 4, 4.0),  # L0 vs L4, gap 4
            (3, 4, 1.0),  # L3 vs L4, gap 1
            (1, 3, 2.0),  # L1 vs L3, gap 2
        ],
    )
    def test_gap_schedule_returns_level_gap(self, victim, attacker, expected):
        assert compute_margin(victim, attacker, False, "gap") == expected

    @pytest.mark.parametrize(
        ("victim", "attacker", "expected"),
        [
            # δ = (j − i) · (k − 1 − i),  k = 5
            (0, 1, 1 * 4),   # L0 victim: gap 1 × 4 = 4
            (0, 4, 4 * 4),   # L0 victim: gap 4 × 4 = 16
            (1, 2, 1 * 3),   # L1 victim: gap 1 × 3 = 3
            (1, 4, 3 * 3),   # L1 victim: gap 3 × 3 = 9
            (2, 4, 2 * 2),   # L2 victim: gap 2 × 2 = 4
            (3, 4, 1 * 1),   # L3 victim: gap 1 × 1 = 1
        ],
    )
    def test_bilateral_formula(self, victim, attacker, expected):
        result = compute_margin(victim, attacker, False, "bilateral")
        assert result == float(expected)

    def test_calibration_forces_zero_under_gap(self):
        # Even if the data row has non-trivial victim/attacker, calibration
        # rows must collapse to standard DPO (δ=0) so over-refusal calibration
        # works as designed (proposal §4.4.3).
        assert compute_margin(0, 4, True, "gap") == 0.0

    def test_calibration_forces_zero_under_bilateral(self):
        assert compute_margin(0, 4, True, "bilateral") == 0.0

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown margin_schedule"):
            compute_margin(0, 1, False, "exponential")


class TestCollatorMarginSchedule:
    """Collator-level integration tests for margin_schedule selection."""

    def test_default_schedule_is_gap(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
        )
        assert collator.margin_schedule == "gap"

    def test_gap_schedule_reads_persisted_column(self):
        """Backward compat: 'gap' uses whatever the data file says, not the formula.

        The persisted ``margin`` column is honoured even when it disagrees
        with (j−i) — this preserves the calibration-override pattern in
        ``calibration.py`` and the cascading-override in ``cascading.py``.
        """
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
            margin_schedule="gap",
        )
        # Persisted margin = 7.0 (deliberately not equal to attacker-victim);
        # gap-schedule must trust the column.
        ex = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>A.<|RESP_END|>",
            "<|RESP_START|>B.<|RESP_END|>",
            margin=7.0, victim_level=0, attacker_level=1,
        )
        batch = collator([ex])
        assert batch["margin"][0].item() == 7.0

    def test_bilateral_overrides_persisted_column(self):
        """bilateral recomputes from hierarchy columns, ignoring persisted margin."""
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
            margin_schedule="bilateral",
        )
        # Persisted margin = 1.0 (gap), but schedule should yield (1)*(4) = 4.0
        ex = _make_dpo_example(
            tokenizer,
            "<|L0_START|>Rule.<|L0_END|>",
            "<|RESP_START|>A.<|RESP_END|>",
            "<|RESP_START|>B.<|RESP_END|>",
            margin=1.0, victim_level=0, attacker_level=1,
        )
        batch = collator([ex])
        assert batch["margin"][0].item() == 4.0

    def test_bilateral_calibration_zeroed(self):
        tokenizer = _make_tokenizer()
        collator = DPOHierarchyCollator(
            pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
            margin_schedule="bilateral",
        )
        # Calibration row with (victim=0, attacker=4): formula would give 16,
        # but is_calibration=True must force 0.
        ex = _make_dpo_example(
            tokenizer,
            "<|L3_START|>Q.<|L3_END|>",
            "<|RESP_START|>A.<|RESP_END|>",
            "<|RESP_START|>B.<|RESP_END|>",
            margin=0.0, victim_level=0, attacker_level=4,
            is_calibration=True,
        )
        batch = collator([ex])
        assert batch["margin"][0].item() == 0.0

    def test_unknown_schedule_rejected_at_construction(self):
        tokenizer = _make_tokenizer()
        with pytest.raises(ValueError, match="Unknown margin_schedule"):
            DPOHierarchyCollator(
                pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
                margin_schedule="bogus",
            )
