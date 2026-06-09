"""Add hierarchy-level special tokens to a tokenizer.

Introduces 12 new tokens: start/end delimiters for each of the 5 hierarchy
levels plus response delimiters. These tokens are encoded as single token IDs
that never appeared in pretraining data. A secure front-end strips them from
untrusted input (L3, L4) to prevent spoofing (StruQ; Chen et al., 2025).
"""

import logging

from transformers import PreTrainedTokenizerBase

from src.config.constants import SPECIAL_TOKENS

logger = logging.getLogger(__name__)


def add_hierarchy_tokens(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[PreTrainedTokenizerBase, int]:
    """Add 12 hierarchy delimiter tokens to a tokenizer.

    Args:
        tokenizer: Any HuggingFace tokenizer.

    Returns:
        Tuple of (modified tokenizer, number of tokens actually added).
        If tokens are already present, num_added will be 0.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    logger.info(
        "Added %d special tokens (vocab size: %d)", num_added, len(tokenizer)
    )
    return tokenizer, num_added
