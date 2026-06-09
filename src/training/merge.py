"""Post-training merge: combine LoRA adapter with base weights and save with ISE."""

import logging
import shutil
from pathlib import Path

import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def merge_lora_adapter(
    base_model: PreTrainedModel,
    adapter_path: str | Path,
) -> PreTrainedModel:
    """Load a LoRA adapter and merge it into the base model weights."""
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = peft_model.merge_and_unload()
    logger.info("Merged LoRA adapter from %s", adapter_path)
    return merged  # type: ignore[return-value]


def save_merged_model_with_ise(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    ise_weights_path: str | Path | None,
    output_dir: str | Path,
) -> None:
    """Save a merged model with ISE weights co-located.

    Pass ``ise_weights_path=None`` for the (f) tokens-only ablation, where
    no ISE layer was trained and the eval loader is expected to skip the
    ``LlamaWithISE`` wrap (it does so automatically when
    ``ise_weights.pt`` is absent from the model directory).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if ise_weights_path is None:
        logger.info(
            "Saved merged model to %s (no ISE: tokens-only ablation)",
            output_dir,
        )
        return

    dest = output_dir / "ise_weights.pt"
    shutil.copy2(str(ise_weights_path), str(dest))
    logger.info("Saved merged model with ISE weights to %s", output_dir)


_PEFT_ADAPTER_KEY_FRAGMENTS: tuple[str, ...] = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
    ".trainable_tokens_delta.",
)


def _remap_peft_key_to_plain(key: str) -> str | None:
    """Remap a PeftModel.state_dict() key to its unwrapped-model equivalent.

    PEFT wraps targeted modules so their state-dict keys look like
    ``base_model.model.<path>.base_layer.weight`` (LoRA) or
    ``base_model.model.<path>.token_adapter.base_layer.weight``
    (trainable_token_indices). Non-targeted modules (layernorms, untargeted
    projections) keep a plain ``base_model.model.<path>.weight`` shape.

    Returns None for adapter-only keys (lora_A/B, trainable_tokens_delta),
    whose contribution is already folded into ``base_layer.weight`` once
    :meth:`PeftModel.merge_adapter` has run.
    """
    if not key.startswith("base_model.model."):
        return None
    stripped = key.removeprefix("base_model.model.")
    for frag in _PEFT_ADAPTER_KEY_FRAGMENTS:
        if frag in stripped:
            return None
    for suffix in (".weight", ".bias"):
        marker = ".base_layer" + suffix
        if stripped.endswith(marker):
            stem = stripped.removesuffix(marker)
            if stem.endswith(".token_adapter"):
                stem = stem.removesuffix(".token_adapter")
            return stem + suffix
    return stripped


def sync_peft_base_weights_to_plain(
    peft_model: PeftModel,
    plain_model: nn.Module,
) -> None:
    """Copy a PeftModel's merged base weights into a plain unwrapped model.

    Temporarily calls ``merge_adapter`` so that LoRA deltas and
    ``trainable_token_indices`` deltas are folded into each wrapped
    module's ``base_layer.weight``; then remaps the PEFT state-dict keys
    to the unwrapped form and loads them into ``plain_model``. The PEFT
    model is restored via ``unmerge_adapter`` before returning.

    Use case: sDPO reference-model update, where the reference is a plain
    :class:`transformers.PreTrainedModel` and the policy is a PEFT-wrapped
    version of the same base.
    """
    peft_model.merge_adapter()
    try:
        remapped: dict = {}
        for k, v in peft_model.state_dict().items():
            new_k = _remap_peft_key_to_plain(k)
            if new_k is not None:
                remapped[new_k] = v
        result = plain_model.load_state_dict(remapped, strict=False)
    finally:
        peft_model.unmerge_adapter()

    if result.missing_keys:
        logger.warning(
            "sync_peft_base_weights_to_plain: %d missing keys (first 5: %s)",
            len(result.missing_keys), result.missing_keys[:5],
        )
    if result.unexpected_keys:
        logger.warning(
            "sync_peft_base_weights_to_plain: %d unexpected keys (first 5: %s)",
            len(result.unexpected_keys), result.unexpected_keys[:5],
        )
