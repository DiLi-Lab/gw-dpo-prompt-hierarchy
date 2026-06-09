#!/usr/bin/env python3
"""Run the 5-level evaluation pipeline against a model checkpoint.

Generates responses for the conflict / aligned / reference splits, scores
the conflict + reference responses with a GPT-4o judge, classifies
aligned-control refusals, and writes a single ``metrics.json`` plus
per-stage artifact files (responses, scoring, refusal) under
``--output-dir``.

Usage:
    python bin/run_evaluation.py \\
        --model models/llama-3.1-8b-gw-dpo-final \\
        --output-dir evaluation/runs/dpo_final
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_root / ".env")

from src.api.openai_client import OpenAIClient  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation.loader import load_model_for_eval  # noqa: E402
from src.evaluation.run_eval import run_evaluation_with_callables  # noqa: E402
from src.data.three_level import collapse_prompt  # noqa: E402
from src.evaluation.external.generate import build_generate_fn  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Path or HF id of model to evaluate")
    p.add_argument("--ise-weights", default=None,
                   help="Optional explicit ISE weights path")
    p.add_argument("--output-dir", default=None,
                   help="Where to write artifacts "
                        "(default: evaluation/runs/<timestamp>)")
    p.add_argument("--config", default="configs/base_linear.yaml",
                   help="Project config")
    p.add_argument("--override", nargs="*", default=[],
                   help="key=value overrides for the project config")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on records per split (debug)")
    p.add_argument("--no-special-tokens", action="store_true",
                   help=(
                       "Skip adding the 12 hierarchy delimiters to the tokenizer "
                       "and skip the embedding-table resize. Required for the "
                       "off-the-shelf Llama-3.1-8B-Instruct strict-floor baseline: "
                       "without this flag the loader resizes the embedding matrix "
                       "with random rows for the new tokens, contaminating the "
                       "baseline. With this flag the delimiters in eval prompts "
                       "BPE-shatter (the principled strict-floor behaviour)."
                   ))
    p.add_argument("--collapse-3level", action="store_true",
                   help=(
                       "Apply the 3-level (Wallace System/User/Tool) collapse "
                       "to every conflict and aligned-control prompt before "
                       "tokenisation. Required for evaluating ablation (e). "
                       "Reference split is unaffected (no delimiters)."
                   ))
    return p.parse_args()


def _maybe_collapse_inputs(
    *,
    output_dir: Path,
    conflict_path: Path,
    aligned_path: Path,
    reference_path: Path,
    enable: bool,
) -> tuple[Path, Path, Path]:
    """Write 3-level-collapsed copies of conflict and aligned splits.

    Reference split is passed through unchanged because flat-text prompts
    have no delimiters for collapse_prompt to act on. When ``enable=False``
    the input paths are returned unchanged so that disabled runs are
    byte-identical to upstream.
    """
    if not enable:
        return conflict_path, aligned_path, reference_path

    collapsed_dir = output_dir / "_collapsed_3level"
    collapsed_dir.mkdir(parents=True, exist_ok=True)

    def _write(src: Path, dst: Path) -> None:
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                rec["prompt"] = collapse_prompt(rec["prompt"])
                rec["collapse_3level"] = True
                fout.write(json.dumps(rec) + "\n")

    new_conflict = collapsed_dir / conflict_path.name
    new_aligned = collapsed_dir / aligned_path.name
    _write(conflict_path, new_conflict)
    _write(aligned_path, new_aligned)
    # Reference split: collapse is a no-op (no delimiters), pass through.
    return new_conflict, new_aligned, reference_path


def _maybe_trim_inputs(
    output_dir: Path,
    conflict_path: Path,
    aligned_path: Path,
    reference_path: Path,
    limit: int,
) -> tuple[Path, Path, Path]:
    """Write trimmed copies of each split for debugging runs."""
    trimmed = output_dir / "_trimmed"
    trimmed.mkdir(parents=True, exist_ok=True)
    for src in (conflict_path, aligned_path, reference_path):
        dst = trimmed / src.name
        with src.open() as fin, dst.open("w") as fout:
            for i, line in enumerate(fin):
                if i >= limit:
                    break
                fout.write(line)
    return (
        trimmed / conflict_path.name,
        trimmed / aligned_path.name,
        trimmed / reference_path.name,
    )


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else cfg.paths.evaluation_runs_dir
        / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model)
    model, tokenizer, has_ise = load_model_for_eval(
        args.model,
        ise_weights_path=args.ise_weights,
        num_segments=cfg.model.num_segments,
        add_special_tokens=not args.no_special_tokens,
    )
    model.eval()
    logger.info(
        "Loaded model (has_ise=%s, special_tokens_added=%s)",
        has_ise, not args.no_special_tokens,
    )

    gen_fn = build_generate_fn(
        model, tokenizer, has_ise,
        max_new_tokens=cfg.evaluation.generation_max_new_tokens,
        temperature=cfg.evaluation.generation_temperature,
    )

    openai = OpenAIClient()

    def judge_fn(system_prompt: str, user_prompt: str) -> str:
        return openai.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=cfg.evaluation.judge_model,
            temperature=cfg.evaluation.judge_temperature,
            max_tokens=cfg.evaluation.judge_max_tokens,
            json_mode=True,
        )

    conflict_path = cfg.paths.eval_conflicts
    aligned_path = cfg.paths.eval_aligned
    reference_path = cfg.paths.eval_reference

    conflict_path, aligned_path, reference_path = _maybe_collapse_inputs(
        output_dir=output_dir,
        conflict_path=conflict_path,
        aligned_path=aligned_path,
        reference_path=reference_path,
        enable=args.collapse_3level,
    )

    if args.limit is not None:
        conflict_path, aligned_path, reference_path = _maybe_trim_inputs(
            output_dir, conflict_path, aligned_path, reference_path, args.limit,
        )

    metrics = run_evaluation_with_callables(
        conflict_path=conflict_path,
        aligned_path=aligned_path,
        reference_path=reference_path if reference_path.exists() else None,
        output_dir=output_dir,
        generate_batch_fn=gen_fn,
        judge_fn=judge_fn,
        generation_batch_size=cfg.evaluation.generation_batch_size,
        orr_min_chars=cfg.evaluation.orr_min_response_chars_for_judge,
        run_text_similarity=cfg.evaluation.run_text_similarity,
        run_rewards=cfg.evaluation.run_rewards,
    )
    logger.info(
        "WHS=%.3f  PPA_macro=%.3f  ORR=%.3f  UtilityΔ=%+.3f",
        metrics["whs"], metrics["ppa_macro"],
        metrics["orr_overall"], metrics["utility_delta_mean"],
    )
    logger.info("Metrics written to %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
