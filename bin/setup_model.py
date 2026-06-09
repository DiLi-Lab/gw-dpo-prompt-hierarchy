#!/usr/bin/env python3
"""Set up the model with special tokens and ISE layer.

Loads a base model, adds hierarchy delimiter tokens, resizes embeddings
with mean initialization, attaches the ISE layer, and saves everything.

Usage:
    python bin/setup_model.py
    python bin/setup_model.py --override model.model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.model import (
    InstructionalSegmentEmbedding,
    add_hierarchy_tokens,
    init_new_token_embeddings,
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


def main() -> None:
    """Load model, add hierarchy tokens, init embeddings, save artifacts."""
    parser = argparse.ArgumentParser(
        description="Set up model with special tokens and ISE layer.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root / "configs" / "base_linear.yaml",
        help="Path to YAML config file (default: configs/base_linear.yaml).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as section.key=value.",
    )

    args = parser.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)

    model_path = cfg.model.model_name_or_path
    dtype_str = cfg.model.torch_dtype
    if dtype_str not in TORCH_DTYPE_MAP:
        logger.error("Unknown torch_dtype: %s", dtype_str)
        sys.exit(1)
    torch_dtype = TORCH_DTYPE_MAP[dtype_str]

    logger.info("Loading tokenizer from %s ...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer, num_added = add_hierarchy_tokens(tokenizer)

    tokenizer_dir = cfg.paths.tokenizer_dir
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info("Saved tokenizer to %s", tokenizer_dir)

    logger.info("Loading model from %s (dtype=%s) ...", model_path, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        #device_map="auto",   # we can run model setup on CPU since we're only modifying the embeddings, but this is faster if we have GPU memory available
    )

    model.resize_token_embeddings(len(tokenizer))
    # Initialize new token embeddings to the mean of existing embeddings
    init_new_token_embeddings(model, num_added)

    hidden_size = model.config.hidden_size
    ise = InstructionalSegmentEmbedding(
        num_segments=cfg.model.num_segments,
        hidden_size=hidden_size,
        init_std=cfg.model.ise_init_std,
    )
    ise_path = cfg.paths.models_dir / "ise_weights_init.pt"
    ise_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ise.state_dict(), ise_path)
    logger.info(
        "Saved ISE weights to %s (init=%s, std=%.4f)",
        ise_path, cfg.model.ise_embedding_init, cfg.model.ise_init_std,
    )

    model_save_dir = cfg.paths.models_dir / "base-with-tokens"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    logger.info("Saved model to %s", model_save_dir)

    logger.info("Setup complete.")
    logger.info("  Vocab size: %d (+%d new tokens)", len(tokenizer), num_added)
    logger.info("  ISE segments: %d", cfg.model.num_segments)
    logger.info("  Hidden size: %d", hidden_size)
    logger.info("  ISE parameters: %d", cfg.model.num_segments * hidden_size)


if __name__ == "__main__":
    main()
