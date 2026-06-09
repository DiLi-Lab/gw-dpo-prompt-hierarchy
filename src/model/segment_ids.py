"""Compute segment IDs from delimiter token structure.

Each token in an input sequence is assigned a segment ID based on which
hierarchy level it belongs to, determined by the delimiter tokens.
Runs in the data collator during training to produce a segment_ids tensor
with the same shape as input_ids.
"""

import logging

import torch
from transformers import PreTrainedTokenizerBase

from src.config.constants import NUM_LEVELS, RESPONSE_SEGMENT_ID

logger = logging.getLogger(__name__)


def _build_delimiter_maps(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """Build mappings from delimiter token IDs to level indices."""
    start_ids: dict[int, int] = {}
    end_ids: dict[int, int] = {}
    for i in range(NUM_LEVELS):
        start_ids[tokenizer.convert_tokens_to_ids(f"<|L{i}_START|>")] = i
        end_ids[tokenizer.convert_tokens_to_ids(f"<|L{i}_END|>")] = i

    resp_start = tokenizer.convert_tokens_to_ids("<|RESP_START|>")
    resp_end = tokenizer.convert_tokens_to_ids("<|RESP_END|>")

    return start_ids, end_ids, resp_start, resp_end


def compute_segment_ids(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Map each token to its hierarchy level based on delimiters.

    Args:
        token_ids: List of token IDs from the tokenizer.
        tokenizer: Tokenizer with hierarchy special tokens added.

    Returns:
        List of segment IDs, same length as token_ids.
    """
    start_ids, end_ids, resp_start, resp_end = _build_delimiter_maps(tokenizer)

    segment_ids: list[int] = []
    current_level = RESPONSE_SEGMENT_ID

    for tid in token_ids:
        if tid in start_ids:
            current_level = start_ids[tid]
            segment_ids.append(current_level)
        elif tid in end_ids:
            segment_ids.append(current_level)
            current_level = RESPONSE_SEGMENT_ID
        elif tid == resp_start or tid == resp_end:
            current_level = RESPONSE_SEGMENT_ID
            segment_ids.append(RESPONSE_SEGMENT_ID)
        else:
            segment_ids.append(current_level)

    return segment_ids


def compute_segment_ids_batch(
    token_id_lists: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    pad_segment_id: int = RESPONSE_SEGMENT_ID,
) -> torch.Tensor:
    """Compute segment IDs for a batch, padding to uniform length.

    Args:
        token_id_lists: List of token ID lists (variable length).
        tokenizer: Tokenizer with hierarchy special tokens added.
        pad_segment_id: Segment ID to use for padding positions.

    Returns:
        Tensor of shape (batch_size, max_seq_len) with segment IDs.
    """
    segment_id_lists = [
        compute_segment_ids(tids, tokenizer) for tids in token_id_lists
    ]

    max_len = max(len(sids) for sids in segment_id_lists)
    padded = [
        sids + [pad_segment_id] * (max_len - len(sids))
        for sids in segment_id_lists
    ]

    return torch.tensor(padded, dtype=torch.long)
