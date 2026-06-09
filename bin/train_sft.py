#!/usr/bin/env python3
"""SFT training with LoRA + ISE on the 5-level hierarchy dataset.

Loads the base model with special tokens, wraps it with ISE,
applies LoRA, and trains with completion-only loss on the SFT dataset.
Uses ISETrainer (a Trainer subclass) for checkpoint saving and
BestCheckpointCallback for persistent best-model tracking.

Usage:
    python bin/train_sft.py
    python bin/train_sft.py --config configs/base_linear.yaml
    python bin/train_sft.py --override sft.num_epochs=1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import load_config
from src.config.constants import SPECIAL_TOKENS
from src.model import InstructionalSegmentEmbedding, LlamaWithISE
from src.training import (
    BestCheckpointCallback,
    HierarchyDataCollator,
    ISESaveCallback,
    build_lora_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    """Tokenize the 'text' field of each example."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


class ISETrainer(Trainer):
    """Trainer subclass for LlamaWithISE models.

    Overrides ``_save`` to persist only the three trained components:
    1. LoRA adapter weights (via PEFT's save_pretrained)
    2. Trainable special token embeddings (saved with LoRA by PEFT)
    3. ISE segment embeddings (saved separately as ise_weights.pt)

    Each checkpoint is ~50MB (LoRA + tokens) + ~100KB (ISE) instead of
    ~16GB (full model state dict) that the default Trainer would save.

    Best-checkpoint persistence is handled by BestCheckpointCallback,
    not by this class.
    """

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict | None = None,
    ) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model: LlamaWithISE = self.model
        # Save LoRA adapters + trainable token embeddings via PEFT
        model.model.save_pretrained(output_dir)
        # Save ISE weights separately when present. The (f) tokens-only
        # ablation runs with model.ise=None and writes no ise_weights.pt;
        # the eval loader's auto-detect (`{model_path}/ise_weights.pt` exists?)
        # then correctly skips the LlamaWithISE wrap at inference time.
        if model.ise is not None:
            ise_path = Path(output_dir) / "ise_weights.pt"
            torch.save(model.ise.state_dict(), ise_path)
            logger.info("Saved LoRA adapters + ISE weights to %s", output_dir)
        else:
            logger.info("Saved LoRA adapters to %s (no ISE: tokens-only ablation)", output_dir)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training with LoRA + ISE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "base_linear.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as section.key=value.",
    )
    args = parser.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)

    # --- Resolve paths ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.runs_dir / f"sft_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(args.config) as f:
        config_snapshot = yaml.safe_load(f)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)
    logger.info("Run directory: %s", run_dir)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.paths.tokenizer_dir))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loaded tokenizer from %s", cfg.paths.tokenizer_dir)

    # --- Load and prepare datasets ---
    train_paths = cfg.paths.for_split("train")
    val_paths = cfg.paths.for_split("val")
    train_dataset = tokenize_dataset(
        load_jsonl_dataset(train_paths.sft_combined),
        tokenizer,
        cfg.sft.max_seq_length,
    )
    eval_dataset = tokenize_dataset(
        load_jsonl_dataset(val_paths.sft_combined),
        tokenizer,
        cfg.sft.max_seq_length,
    )
    logger.info("Train: %d examples, Val: %d examples", len(train_dataset), len(eval_dataset))

    # --- Load model ---
    dtype_str = cfg.model.torch_dtype
    if dtype_str not in TORCH_DTYPE_MAP:
        logger.error("Unknown torch_dtype: %s", dtype_str)
        sys.exit(1)
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]

    base_model = AutoModelForCausalLM.from_pretrained(
        str(cfg.paths.models_dir / "base-with-tokens"),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    logger.info("Loaded base model from %s", cfg.paths.models_dir / "base-with-tokens")

    # --- Resolve special token indices for trainable_token_indices ---
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS
    ]
    logger.info(
        "Special token indices for training: %s",
        list(zip(SPECIAL_TOKENS, special_token_ids)),
    )

    # --- Apply LoRA ---
    lora_config = build_lora_config(
        cfg.sft,
        special_token_ids=special_token_ids,
        tie_word_embeddings=getattr(base_model.config, "tie_word_embeddings", False),
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    # --- Wrap with ISE (or skip ISE for the (f) tokens-only ablation) ---
    if cfg.model.use_ise:
        ise = InstructionalSegmentEmbedding(
            num_segments=cfg.model.num_segments,
            hidden_size=base_model.config.hidden_size,
            init_std=cfg.model.ise_init_std,
        )
        ise_init_path = cfg.paths.models_dir / "ise_weights_init.pt"
        if ise_init_path.exists():
            ise.load_state_dict(torch.load(ise_init_path, weights_only=True))
            logger.info("Loaded ISE weights from %s", ise_init_path)
        model = LlamaWithISE(model=peft_model, ise=ise)
    else:
        ise = None
        model = LlamaWithISE(model=peft_model, ise=None)
        logger.info("ISE disabled (use_ise=false): training tokens-only ablation (f)")

    # --- Data collator ---
    collator = HierarchyDataCollator(
        tokenizer=tokenizer,
        max_seq_length=cfg.sft.max_seq_length,
    )

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type=cfg.sft.lr_scheduler,
        warmup_ratio=cfg.sft.warmup_ratio,
        num_train_epochs=cfg.sft.num_epochs,
        per_device_train_batch_size=cfg.sft.per_device_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        weight_decay=cfg.sft.weight_decay,
        bf16=(cfg.sft.precision == "bf16"),
        fp16=(cfg.sft.precision == "fp16"),
        eval_strategy="steps",
        eval_steps=cfg.sft.eval_steps,
        save_strategy="steps",
        save_steps=cfg.sft.save_steps,
        save_total_limit=cfg.sft.save_total_limit,
        load_best_model_at_end=False,
        remove_unused_columns=cfg.sft.remove_unused_columns,
        logging_steps=cfg.sft.logging_steps,
        logging_dir=str(run_dir / "logs"),
        report_to="none",
        gradient_checkpointing=True,
    )

    # --- Callbacks ---
    # ISESaveCallback is a safety net that mirrors what ISETrainer._save
    # already does — only register it when ISE is on. BestCheckpointCallback
    # also handles the no-ISE case (it skips ise_weights.pt when model.ise
    # is None) so it's registered unconditionally.
    callbacks: list = []
    if ise is not None:
        callbacks.append(ISESaveCallback(ise=ise))
    best_callback = BestCheckpointCallback(
        model=model,
        tokenizer=tokenizer,
        run_dir=run_dir,
    )
    callbacks.append(best_callback)

    # --- Create trainer ---
    trainer = ISETrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("Starting SFT training...")
    trainer.train()

    # --- Save full trainer state to run root ---
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    logger.info("Saved trainer_state.json to %s", run_dir)

    # --- Log best checkpoint info ---
    if best_callback.best_eval_loss < float("inf"):
        logger.info(
            "Best checkpoint: %s (eval_loss=%.4f)",
            best_callback.best_dir,
            best_callback.best_eval_loss,
        )
    else:
        logger.warning("No evaluation was run, no best checkpoint saved")

    logger.info("SFT training complete. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
