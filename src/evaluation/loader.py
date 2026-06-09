"""Model loader for evaluation: plain HF, PEFT adapter, or merged dir.

Returns ``(model, tokenizer, has_ise)``. When ``ise_weights_path`` is
provided (or ``ise_weights.pt`` exists in the model dir), the model is
wrapped in ``LlamaWithISE`` and segment-aware generation is enabled.

This module deliberately does *not* implement generation; callers (e.g.
``src.evaluation.generation``) build a ``generate_batch_fn`` closure
around the returned model + tokenizer.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.config.constants import SPECIAL_TOKENS
from src.model.ise import InstructionalSegmentEmbedding
from src.model.llama_with_ise import LlamaWithISE

logger = logging.getLogger(__name__)


def load_model_for_eval(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "auto",
    ise_weights_path: str | Path | None = None,
    num_segments: int = 6,
    add_special_tokens: bool = True,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, bool]:
    """Load a model + tokenizer for evaluation.

    Args:
        model_path: HF id or local merged-model directory.
        torch_dtype: dtype for the base model.
        device_map: accelerate device_map ("auto" by default).
        ise_weights_path: Path to ``ise_weights.pt``. If None, attempts to
            load ``{model_path}/ise_weights.pt`` if it exists.
        num_segments: ISE segment count (matches training config).
        add_special_tokens: When True, ensures hierarchy delimiters are
            registered as special tokens in the tokenizer (no-op if the
            merged-model tokenizer already has them).

    Returns:
        Tuple ``(model, tokenizer, has_ise)``. ``model`` is either a
        ``LlamaForCausalLM`` or a ``LlamaWithISE`` wrapping one.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if add_special_tokens:
        # Use get_vocab() rather than the .additional_special_tokens
        # attribute: the latter was removed in transformers 5.x. get_vocab()
        # returns a dict[str, int] of every token in the vocabulary
        # (including added special tokens) and is stable across versions.
        vocab = tokenizer.get_vocab()
        missing = [t for t in SPECIAL_TOKENS if t not in vocab]
        if missing:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": list(missing)},
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models need LEFT padding for batched generation: the
    # generated tokens must directly follow the last real prompt token.
    # The merged DPO model's saved tokenizer config has this set, but
    # base-tokenizer dirs (e.g. models/base-with-tokens/) inherit the
    # Llama default of right-padding and emit a warning per generate()
    # call when batch_size > 1.
    tokenizer.padding_side = "left"

    # ``torch_dtype`` was renamed to ``dtype`` in transformers 5.x; the project
    # pins transformers >=5.3, so we use the new name to avoid the deprecation
    # warning fired on every model load.
    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch_dtype,
        device_map=device_map,
    )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))

    weights = ise_weights_path
    if weights is None:
        candidate = Path(model_path) / "ise_weights.pt"
        if candidate.exists():
            weights = candidate

    if weights is not None:
        ise = InstructionalSegmentEmbedding(
            num_segments=num_segments,
            hidden_size=base.config.hidden_size,
        )
        state = torch.load(str(weights), map_location="cpu")
        ise.load_state_dict(state)
        device = base.device
        ise = ise.to(device=device, dtype=torch_dtype)
        wrapped = LlamaWithISE(base, ise)
        return wrapped, tokenizer, True

    return base, tokenizer, False
