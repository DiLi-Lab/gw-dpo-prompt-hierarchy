"""Hierarchy-aware data collator for DPO training.

Extends TRL's DataCollatorForPreference to compute ISE segment IDs
and pass through gravity-weighted margins for each preference pair.
"""

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from trl.trainer.dpo_trainer import DataCollatorForPreference

from src.model.segment_ids import compute_segment_ids

_HIERARCHY_DEPTH = 5  # L0..L4

_SUPPORTED_SCHEDULES = ("gap", "bilateral")


def compute_margin(
    victim_level: int,
    attacker_level: int,
    is_calibration: bool,
    schedule: str,
) -> float:
    """Derive the per-sample DPO margin δ from hierarchy levels.

    The trainer multiplies the returned value by ``gravity_alpha``.

    Schedules:
      ``gap``        → δ = j − i  (linear margin schedule used by the
                       linear-schedule production run base_linear.yaml).
      ``bilateral``  → δ = (j − i)·(k − 1 − i), k=5  (bilateral /
                       victim-weighted margin schedule used by the
                       bilateral production run base_bilateral.yaml:
                       severity scales with both privilege distance and
                       the intrinsic value of the victim level).

    Calibration rows (``is_calibration=True``) always return 0 so the
    over-refusal calibration objective collapses to standard DPO.
    """
    if is_calibration:
        return 0.0
    gap = attacker_level - victim_level
    if schedule == "gap":
        return float(gap)
    if schedule == "bilateral":
        return float(gap * (_HIERARCHY_DEPTH - 1 - victim_level))
    msg = (
        f"Unknown margin_schedule {schedule!r}; "
        f"supported: {_SUPPORTED_SCHEDULES}"
    )
    raise ValueError(msg)


@dataclass
class DPOHierarchyCollator(DataCollatorForPreference):
    """Data collator that adds segment_ids and margin to DPO batches.

    Extends TRL's ``DataCollatorForPreference`` to:
    1. Compute ``segment_ids`` from delimiter tokens in each sequence.
    2. Pass ``margin`` values through for gravity-weighted loss.

    Args:
        pad_token_id: Token ID used for padding.
        tokenizer: Tokenizer with hierarchy special tokens, used for
            segment ID computation.
        margin_schedule: Schedule used to derive δ from each row's
            ``(victim_level, attacker_level, is_calibration)`` columns.
            ``"gap"`` (linear-schedule production run, default) reads
            the persisted ``margin`` column unchanged for backward
            compatibility with existing data. ``"bilateral"``
            (bilateral / victim-weighted production run) recomputes
            δ = (j − i)·(k − 1 − i) at collation time, ignoring the
            persisted column.
    """

    tokenizer: PreTrainedTokenizerBase | None = None
    margin_schedule: str = "gap"

    def __post_init__(self) -> None:
        super_post_init = getattr(super(), "__post_init__", None)
        if super_post_init is not None:
            super_post_init()
        if self.margin_schedule not in _SUPPORTED_SCHEDULES:
            msg = (
                f"Unknown margin_schedule {self.margin_schedule!r}; "
                f"supported: {_SUPPORTED_SCHEDULES}"
            )
            raise ValueError(msg)

    def torch_call(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # Extract margins before parent processing (parent ignores unknown keys).
        # When margin_schedule == "gap" we read the persisted column for
        # backward compatibility; otherwise we recompute δ from the hierarchy
        # columns so existing dpo_combined.jsonl files work without a rewrite.
        margins: list[float] | None = None
        if examples:
            if self.margin_schedule == "gap":
                if "margin" in examples[0]:
                    margins = [ex["margin"] for ex in examples]
            else:
                margins = [
                    compute_margin(
                        victim_level=int(ex["victim_level"]),
                        attacker_level=int(ex["attacker_level"]),
                        is_calibration=bool(ex.get("is_calibration", False)),
                        schedule=self.margin_schedule,
                    )
                    for ex in examples
                ]

        # Parent handles: concatenation of prompt+chosen / prompt+rejected,
        # padding, attention_mask, completion_mask
        batch = super().torch_call(examples)

        # Compute segment_ids for each sequence in the batch
        input_ids = batch["input_ids"]  # shape: (2*batch_size, seq_len)
        segment_ids = self._compute_batch_segment_ids(input_ids)
        batch["segment_ids"] = segment_ids

        # Pass through margins (one per preference pair, not per sequence)
        if margins is not None:
            batch["margin"] = torch.tensor(margins, dtype=torch.float32)

        return batch

    def _compute_batch_segment_ids(
        self, input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute segment IDs for a batch of token sequences.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len).

        Returns:
            Tensor of segment IDs with the same shape as input_ids.
        """
        batch_segment_ids = []
        for row in input_ids.tolist():
            seg_ids = compute_segment_ids(row, self.tokenizer)
            batch_segment_ids.append(seg_ids)
        return torch.tensor(batch_segment_ids, dtype=torch.long)
