#!/usr/bin/env python3
"""Gravity-Weighted DPO training with curriculum learning and ISE.

Loads the merged SFT model, applies fresh LoRA adapters, and trains
with gravity-weighted DPO across 3 curriculum stages (easy → hard).
Uses sDPO reference model updates between stages.

Usage:
    python bin/train_dpo.py
    python bin/train_dpo.py --config configs/base_linear.yaml
    python bin/train_dpo.py --sft-checkpoint models/runs/sft_20260417_140013/best-checkpoint
    python bin/train_dpo.py --override dpo.beta=0.05 dpo.gravity_alpha=2.0
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
from transformers import AutoModelForCausalLM, AutoTokenizer

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import load_config
from src.config.constants import SPECIAL_TOKENS
from src.model import InstructionalSegmentEmbedding, LlamaWithISE
from src.training import (
    build_lora_config,
    merge_lora_adapter,
    run_dpo_curriculum,
    save_merged_model_with_ise,
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


def find_sft_best_checkpoint(cfg) -> Path:
    """Find the most recent SFT best-checkpoint in the runs directory."""
    runs_dir = cfg.paths.runs_dir
    sft_runs = sorted(runs_dir.glob("sft_*/best-checkpoint"), reverse=True)
    if not sft_runs:
        logger.error("No SFT best-checkpoint found in %s", runs_dir)
        sys.exit(1)
    return sft_runs[0]


def merge_sft_checkpoint(
    cfg,
    sft_checkpoint: Path,
    tokenizer: AutoTokenizer,
    torch_dtype: torch.dtype,
    merged_dir_override: Path | None = None,
) -> Path:
    """Merge SFT LoRA into base model and save. Idempotent.

    Args:
        merged_dir_override: When provided, write the merged artifact to this
            directory instead of cfg.paths.sft_merged_dir. Required for
            ablations whose merged checkpoints must not collide with the (b)
            artifact at models/llama-3.1-8b-sft-merged/ (e.g. the (f)
            tokens-only run produces an ISE-free merged checkpoint).
    """
    merged_dir = merged_dir_override if merged_dir_override is not None else cfg.paths.sft_merged_dir
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        logger.info("Merged SFT model already exists at %s, skipping merge", merged_dir)
        return merged_dir

    logger.info("Merging SFT checkpoint %s into base model...", sft_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(cfg.paths.models_dir / "base-with-tokens"),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    merged = merge_lora_adapter(base_model, sft_checkpoint)
    # Honour use_ise: tokens-only ablation runs are merged without
    # ise_weights.pt so downstream loaders don't try to wrap with ISE.
    ise_weights_path: Path | None = (
        sft_checkpoint / "ise_weights.pt" if cfg.model.use_ise else None
    )
    save_merged_model_with_ise(merged, tokenizer, ise_weights_path, merged_dir)
    logger.info("Saved merged SFT model to %s", merged_dir)

    # Free memory
    del base_model, merged
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return merged_dir


def create_policy_model(
    cfg,
    merged_dir: Path,
    torch_dtype: torch.dtype,
    special_token_ids: list[int],
) -> LlamaWithISE:
    """Create policy model: merged SFT + fresh LoRA (+ ISE if enabled)."""
    base_model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    lora_config = build_lora_config(
        cfg.dpo,
        special_token_ids=special_token_ids,
        tie_word_embeddings=getattr(base_model.config, "tie_word_embeddings", False),
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    if not cfg.model.use_ise:
        logger.info("ISE disabled (use_ise=false): policy is tokens-only")
        return LlamaWithISE(model=peft_model, ise=None)

    ise = InstructionalSegmentEmbedding(
        num_segments=cfg.model.num_segments,
        hidden_size=base_model.config.hidden_size,
        init_std=cfg.model.ise_init_std,
    )
    ise_path = merged_dir / "ise_weights.pt"
    if ise_path.exists():
        ise.load_state_dict(torch.load(ise_path, weights_only=True))
        logger.info("Loaded ISE weights from %s", ise_path)

    return LlamaWithISE(model=peft_model, ise=ise)


def create_reference_model(
    cfg,
    merged_dir: Path,
    torch_dtype: torch.dtype,
) -> LlamaWithISE:
    """Create frozen reference model: merged SFT (+ ISE if enabled, no LoRA)."""
    ref_base = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if not cfg.model.use_ise:
        ref_model = LlamaWithISE(model=ref_base, ise=None)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        logger.info("Created frozen reference model (no ISE: tokens-only)")
        return ref_model

    ise = InstructionalSegmentEmbedding(
        num_segments=cfg.model.num_segments,
        hidden_size=ref_base.config.hidden_size,
        init_std=cfg.model.ise_init_std,
    )
    ise_path = merged_dir / "ise_weights.pt"
    if ise_path.exists():
        ise.load_state_dict(torch.load(ise_path, weights_only=True))

    ref_model = LlamaWithISE(model=ref_base, ise=ise)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Created frozen reference model")
    return ref_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gravity-Weighted DPO training with curriculum learning.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "base_linear.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="Path to SFT best-checkpoint directory. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as section.key=value.",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="Override the final merged-model output directory "
             "(default: cfg.paths.dpo_final_dir = models/llama-3.1-8b-gw-dpo-final). "
             "Use a distinct path for ablations to avoid overwriting (d).",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Override the SFT-merged-model directory used as the DPO "
             "policy/reference initialisation (default: cfg.paths.sft_merged_dir = "
             "models/llama-3.1-8b-sft-merged). Use a distinct path for ablations "
             "(e.g. (f) tokens-only) so the no-ISE DPO run does not reuse the "
             "existing ISE-on (b) merged checkpoint.",
    )
    args = parser.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)
    final_dir: Path = args.final_dir if args.final_dir is not None else cfg.paths.dpo_final_dir
    sft_merged_override: Path | None = args.merged_dir

    # --- Resolve paths ---
    alpha_str = str(cfg.dpo.gravity_alpha).replace(".", "p")
    beta_str = str(cfg.dpo.beta).replace(".", "p")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.runs_dir / f"dpo_{timestamp}_a{alpha_str}_b{beta_str}"
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

    # --- Resolve dtype ---
    dtype_str = cfg.model.torch_dtype
    if dtype_str not in TORCH_DTYPE_MAP:
        logger.error("Unknown torch_dtype: %s", dtype_str)
        sys.exit(1)
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]

    # --- Pre-DPO: merge SFT checkpoint ---
    sft_checkpoint = args.sft_checkpoint or find_sft_best_checkpoint(cfg)
    logger.info("Using SFT checkpoint: %s", sft_checkpoint)
    merged_dir = merge_sft_checkpoint(
        cfg, sft_checkpoint, tokenizer, torch_dtype,
        merged_dir_override=sft_merged_override,
    )

    # --- Load DPO datasets ---
    train_paths = cfg.paths.for_split(cfg.dpo.train_split_name)
    val_paths = cfg.paths.for_split(cfg.dpo.val_split_name)
    train_dataset = load_jsonl_dataset(train_paths.dpo_combined)
    val_dataset = load_jsonl_dataset(val_paths.dpo_combined)
    logger.info(
        "DPO data: %d train pairs (%s), %d val pairs (%s)",
        len(train_dataset), cfg.dpo.train_split_name,
        len(val_dataset), cfg.dpo.val_split_name,
    )

    # --- Resolve special token indices ---
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS
    ]

    # --- Create policy and reference models ---
    policy_model = create_policy_model(
        cfg, merged_dir, torch_dtype, special_token_ids,
    )
    ref_model = create_reference_model(cfg, merged_dir, torch_dtype)

    # --- Run the DPO curriculum via the shared helper ---
    best_ckpt = run_dpo_curriculum(
        cfg=cfg,
        merged_dir=merged_dir,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        special_token_ids=special_token_ids,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        policy_model=policy_model,
        ref_model=ref_model,
    )
    logger.info("Final stage best-checkpoint: %s", best_ckpt)

    # --- Post-DPO: merge final adapter and save ---
    logger.info("Merging final DPO adapter...")

    # Load a fresh base from the merged SFT dir, apply the final LoRA, merge
    final_base = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    final_merged = merge_lora_adapter(final_base, best_ckpt)

    if not cfg.model.use_ise:
        # (f) tokens-only ablation: no ISE weights to persist.
        ise_weights_path: Path | None = None
    else:
        ise_weights_path = best_ckpt / "ise_weights.pt"
        if not ise_weights_path.exists():
            # Use policy model's current ISE
            final_stage_dir = run_dir / f"stage{cfg.dpo.final_stage_index}"
            ise_weights_path = final_stage_dir / "ise_weights_final.pt"
            torch.save(policy_model.ise.state_dict(), ise_weights_path)

    save_merged_model_with_ise(
        final_merged, tokenizer, ise_weights_path, final_dir,
    )

    # Save final eval results
    final_eval = {
        "run_dir": str(run_dir),
        "alpha": cfg.dpo.gravity_alpha,
        "beta": cfg.dpo.beta,
        "num_stages": cfg.dpo.final_stage_index,
        "curriculum_enabled": cfg.dpo.curriculum_enabled,
        "final_model": str(final_dir),
    }
    with open(run_dir / "final_eval.json", "w") as f:
        json.dump(final_eval, f, indent=2)

    logger.info("DPO training complete. Final model: %s", final_dir)
    logger.info("Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
