"""Build peft LoraConfig from project hyperparameter configs."""

from peft import LoraConfig

from src.config.hyperparameters import DPOConfig, SFTConfig


def build_lora_config(
    cfg: SFTConfig | DPOConfig,
    special_token_ids: list[int] | None = None,
    tie_word_embeddings: bool = False,
) -> LoraConfig:
    """Construct a LoraConfig from an SFTConfig or DPOConfig.

    Args:
        cfg: Training hyperparameter config.
        special_token_ids: Token indices of the 12 hierarchy special tokens.
            When provided, only these token rows in embed_tokens and lm_head
            are made trainable (via trainable_token_indices) instead of making
            the entire modules trainable via modules_to_save.
        tie_word_embeddings: Whether the model ties lm_head weights to
            embed_tokens. When True, PEFT auto-wraps lm_head via its
            tied-weight mechanism, so we must not include it explicitly
            in trainable_token_indices (that causes double-wrapping).
    """
    trainable_token_indices = None
    if special_token_ids is not None:
        trainable_token_indices = {
            "embed_tokens": special_token_ids,
        }
        if not tie_word_embeddings:
            trainable_token_indices["lm_head"] = special_token_ids

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        trainable_token_indices=trainable_token_indices,
        task_type=cfg.task_type,
    )
