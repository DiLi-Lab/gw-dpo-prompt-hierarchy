"""Hierarchy-aware data collator for SFT training.

Produces batches with segment_ids for ISE and completion-only labels
that mask all prompt tokens (before <|RESP_START|>) with -100.
"""

import torch
from transformers import PreTrainedTokenizerBase

from src.config.constants import RESPONSE_SEGMENT_ID
from src.model.segment_ids import compute_segment_ids


class HierarchyDataCollator:
    """Data collator that computes segment IDs and completion-only labels.

    For each example:
    1. Uses pre-tokenized input_ids (already truncated to max_seq_length).
    2. Computes segment_ids from delimiter structure.
    3. Creates labels by copying input_ids and masking prompt tokens with -100.
    4. Right-pads all tensors to the batch's maximum length.

    Args:
        tokenizer: Tokenizer with hierarchy special tokens.
        max_seq_length: Maximum sequence length (for safety truncation).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.resp_start_id = tokenizer.convert_tokens_to_ids("<|RESP_START|>")

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_segment_ids: list[list[int]] = []

        for feat in features:
            input_ids = feat["input_ids"][:self.max_seq_length]
            attn_mask = feat["attention_mask"][:self.max_seq_length]

            segment_ids = compute_segment_ids(input_ids, self.tokenizer)
            labels = self._make_completion_only_labels(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attn_mask)
            batch_labels.append(labels)
            batch_segment_ids.append(segment_ids)

        return self._pad_batch(
            batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids,
        )

    def _make_completion_only_labels(self, input_ids: list[int]) -> list[int]:
        labels = list(input_ids)
        resp_pos = -1
        for i, tid in enumerate(input_ids):
            if tid == self.resp_start_id:
                resp_pos = i
                break

        mask_end = resp_pos + 1 if resp_pos >= 0 else len(labels)
        for i in range(mask_end):
            labels[i] = -100

        return labels

    def _pad_batch(
        self,
        batch_input_ids: list[list[int]],
        batch_attention_mask: list[list[int]],
        batch_labels: list[list[int]],
        batch_segment_ids: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        padded_segment_ids = []

        for ids, mask, labs, segs in zip(
            batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids,
        ):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_segment_ids.append(segs + [RESPONSE_SEGMENT_ID] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "segment_ids": torch.tensor(padded_segment_ids, dtype=torch.long),
        }
