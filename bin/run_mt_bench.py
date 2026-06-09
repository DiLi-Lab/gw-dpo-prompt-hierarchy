#!/usr/bin/env python3
"""Run the MT-Bench external benchmark against a registered model checkpoint.

Usage:
    python bin/run_mt_bench.py --model gw_dpo
    python bin/run_mt_bench.py --model base_stock
    python bin/run_mt_bench.py --model gw_dpo --limit 8
    python bin/run_mt_bench.py --model gw_dpo --override mt_bench.generation_batch_size=2
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_root / ".env")

from src.api.openai_client import OpenAIClient  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation.external.generate import build_generate_fn  # noqa: E402
from src.evaluation.external.mt_bench.runner import run_mt_bench_with_callables  # noqa: E402
from src.evaluation.external.registry import resolve_model  # noqa: E402
from src.evaluation.external.resume import (  # noqa: E402
    resolve_output_dir,
    save_run_args,
    validate_run_args,
)
from src.evaluation.loader import load_model_for_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Model nickname (see registry).")
    p.add_argument("--config", default="configs/mt_bench.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to first N questions (debug).")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Reuse the newest run_<ts>/ under the canonical output base; "
             "skip with exit 0 if metrics.json already exists. Use "
             "--no-resume to force a fresh run.",
    )
    return p.parse_args()


def _trim_questions_to_limit(src: Path, dst: Path, limit: int) -> None:
    """Write the first ``limit`` lines of the question JSONL to ``dst``.

    MT-Bench's ``question.jsonl`` has exactly one record per line (no
    embedded newlines), so simple line-based trimming is safe.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)


def main() -> int:  # noqa: PLR0915
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)
    if cfg.mt_bench is None:
        msg = f"{args.config} did not provide an `mt_bench:` section."
        raise ValueError(msg)

    questions_path = Path(cfg.mt_bench.question_jsonl)
    references_path = Path(cfg.mt_bench.reference_answer_jsonl)
    judge_prompts_path = Path(cfg.mt_bench.judge_prompts_jsonl)
    for p in (questions_path, references_path, judge_prompts_path):
        if not p.exists():
            msg = (
                f"Missing MT-Bench data file at {p}. "
                "Vendor MT-Bench data first; see "
                "data/external/mt_bench/NOTICE."
            )
            raise SystemExit(msg)

    base_dir = (
        cfg.paths.external_runs_dir
        / "mt_bench"
        / f"{args.model}__chat_template"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir, resume_mode = resolve_output_dir(
        output_dir_arg=args.output_dir,
        resume=args.resume,
        base_dir=base_dir,
        timestamp=timestamp,
    )
    run_args = {
        "model": args.model,
        "format": "chat_template",
        "limit": args.limit,
    }
    if resume_mode == "resume_complete":
        logger.info(
            "Resume: %s already complete (metrics.json present) — skipping. "
            "Pass --no-resume to force a fresh run.", output_dir,
        )
        return 0
    if resume_mode == "resume_partial":
        logger.info("Resume: continuing partial run at %s", output_dir)
        validate_run_args(output_dir, run_args)
    save_run_args(output_dir, run_args)

    # Resolve the checkpoint only after the resume short-circuit so a
    # missing model dir doesn't block skipping an already-complete run.
    resolved = resolve_model(args.model, cfg.paths)
    logger.info(
        "Loading model %s from %s (special_tokens=%s)",
        args.model, resolved.model_path, resolved.requires_special_tokens,
    )
    model, tokenizer, has_ise = load_model_for_eval(
        resolved.model_path,
        ise_weights_path=resolved.ise_weights_path,
        add_special_tokens=resolved.requires_special_tokens,
    )
    model.eval()

    # chat_template format: ISE is forcibly disabled (no delimiter spans).
    if has_ise:
        logger.warning(
            "Format=chat_template: ISE wrapper present but bypassed for"
            " inference. metrics.json will record ise_active=false.",
        )

    def gen_fn_for_temp(temp: float):
        return build_generate_fn(
            model, tokenizer, has_ise=False,
            max_new_tokens=cfg.mt_bench.generation_max_new_tokens,
            temperature=temp,
        )

    openai = OpenAIClient()

    def judge_fn(*, system_prompt: str, user_prompt: str, temperature: float) -> str:
        return openai.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=cfg.mt_bench.judge_model,
            temperature=temperature,
            max_tokens=cfg.mt_bench.judge_max_tokens,
        )

    if args.limit is not None:
        trimmed = output_dir / "_trimmed_questions.jsonl"
        _trim_questions_to_limit(questions_path, trimmed, args.limit)
        questions_path = trimmed

    metrics = run_mt_bench_with_callables(
        questions_path=questions_path,
        references_path=references_path,
        judge_prompts_path=judge_prompts_path,
        output_dir=output_dir,
        tokenizer=tokenizer,
        generate_batch_fn_for_temperature=gen_fn_for_temp,
        temperature_per_category=cfg.mt_bench.temperature_per_category,
        generation_batch_size=cfg.mt_bench.generation_batch_size,
        judge_fn=judge_fn,
        judge_temperature=cfg.mt_bench.judge_temperature,
        judge_temperature_retry=cfg.mt_bench.judge_temperature_retry,
        run_metadata={
            "model": args.model,
            "format": "chat_template",
            "ise_active": False,
            "judge_model": cfg.mt_bench.judge_model,
            "max_new_tokens": cfg.mt_bench.generation_max_new_tokens,
        },
    )
    overall = metrics["overall_mean"]
    overall_str = f"{overall:.3f}" if overall is not None else "None"
    logger.info(
        "MT-Bench done. overall_mean=%s n_turns_scored=%d parse_failures=%d",
        overall_str,
        metrics["n_turns_scored"],
        metrics["n_judge_parse_failures"],
    )
    logger.info("Metrics → %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
