#!/usr/bin/env python3
"""Merge the SFT LoRA adapter into base-with-tokens for ablation (b).

Loads the base model with hierarchy tokens, folds the SFT LoRA adapter weights
into it via PeftModel.merge_and_unload(), and writes the merged model plus the
co-located ISE weights to ``models/llama-3.1-8b-sft-merged/`` (the canonical
sft_merged_dir from src/config/paths.py). The output directory is the same one
``bin/train_dpo.py`` writes to between Phase 1 and Phase 2; this script lets
you produce that artifact standalone for evaluation.

Idempotent: if the target directory already exists with a config.json, the
script exits without re-merging.

Usage:
    python bin/merge_sft.py
    python bin/merge_sft.py --sft-checkpoint models/runs/sft_20260417_140013/best-checkpoint
    python bin/merge_sft.py --output-dir models/llama-3.1-8b-sft-merged
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.training import merge_lora_adapter, save_merged_model_with_ise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_sft_checkpoint(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("sft_*/best-checkpoint"), reverse=True)
    if not candidates:
        logger.error("No SFT best-checkpoint found under %s", runs_dir)
        sys.exit(1)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_root / "configs" / "base_linear.yaml",
        help="Project config (default: configs/base_linear.yaml).",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="SFT best-checkpoint directory. If omitted, the latest "
             "sft_*/best-checkpoint under models/runs/ is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the merged model (default: cfg.paths.sft_merged_dir).",
    )
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, overrides=[])

    sft_ckpt = args.sft_checkpoint or find_latest_sft_checkpoint(cfg.paths.runs_dir)
    out_dir = args.output_dir or cfg.paths.sft_merged_dir
    base_dir = cfg.paths.models_dir / "base-with-tokens"

    if out_dir.exists() and (out_dir / "config.json").exists():
        logger.info("Merged SFT model already exists at %s, skipping.", out_dir)
        return

    # Resolve the ISE weights path. The (f) tokens-only ablation produces
    # SFT checkpoints without ise_weights.pt; in that case the merged
    # model is also written without an ISE file. Honour cfg.model.use_ise
    # rather than only inspecting the checkpoint, so a misconfigured
    # checkpoint (ISE on but file missing) still fails loudly.
    ise_weights_path: Path | None = sft_ckpt / "ise_weights.pt"
    if not cfg.model.use_ise:
        ise_weights_path = None
        logger.info("ISE disabled (use_ise=false): merging without ise_weights.pt")
    elif not ise_weights_path.exists():
        logger.error(
            "ise_weights.pt missing in %s (config has use_ise=true)", sft_ckpt,
        )
        sys.exit(1)

    logger.info("Loading base model from %s", base_dir)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    logger.info("Loading tokenizer from %s", sft_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(str(sft_ckpt))

    logger.info("Merging LoRA adapter from %s ...", sft_ckpt)
    merged = merge_lora_adapter(base, sft_ckpt)

    logger.info("Saving merged model to %s", out_dir)
    save_merged_model_with_ise(
        merged, tokenizer,
        ise_weights_path=ise_weights_path,
        output_dir=out_dir,
    )

    logger.info("Done. Merged SFT model at %s", out_dir)


if __name__ == "__main__":
    main()
