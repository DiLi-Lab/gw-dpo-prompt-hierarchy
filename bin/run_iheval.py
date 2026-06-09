#!/usr/bin/env python3
"""Run the IHEval external benchmark against a registered model checkpoint.

Usage:
    python bin/run_iheval.py --model gw_dpo --format delimited
    python bin/run_iheval.py --model base_stock --format chat_template \\
        --tasks single-turn,translation --settings aligned,conflict
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

from src.config import load_config  # noqa: E402
from src.evaluation.external.generate import build_generate_fn  # noqa: E402
from src.evaluation.external.iheval.adapters import format_record_for_format  # noqa: E402
from src.evaluation.external.iheval.runner import run_iheval_with_callables  # noqa: E402
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


def _csv_arg(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--format", choices=("delimited", "chat_template"),
                   default="delimited")
    p.add_argument("--config", default="configs/iheval.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--tasks", type=_csv_arg, default=None,
                   help="Comma-separated task names; defaults to config.default_tasks.")
    p.add_argument("--settings", type=_csv_arg, default=None,
                   help="Comma-separated settings; defaults to config.default_settings.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, cap the records per (task, setting, sub) at this many.")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Reuse the newest run_<ts>/ under the canonical output base; "
             "skip with exit 0 if metrics.json already exists. Use "
             "--no-resume to force a fresh run.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)
    if cfg.iheval is None:
        msg = f"{args.config} did not provide an `iheval:` section."
        raise ValueError(msg)

    tasks = args.tasks or cfg.iheval.default_tasks
    settings = args.settings or cfg.iheval.default_settings

    base_dir = (
        cfg.paths.external_runs_dir
        / "iheval"
        / f"{args.model}__{args.format}"
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
        "format": args.format,
        "tasks": sorted(tasks),
        "settings": sorted(settings),
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

    effective_has_ise = has_ise and args.format == "delimited"
    if has_ise and not effective_has_ise:
        logger.warning(
            "Format=chat_template: ISE wrapper present but bypassed for"
            " inference. metrics.json will record ise_active=false.",
        )

    gen_fn = build_generate_fn(
        model, tokenizer, has_ise=effective_has_ise,
        max_new_tokens=cfg.iheval.generation_max_new_tokens,
        temperature=cfg.iheval.generation_temperature,
    )

    def format_record_fn(record):
        return format_record_for_format(
            record, fmt=args.format, tokenizer=tokenizer,
        )

    benchmark_root = Path(cfg.iheval.benchmark_root)

    metrics = run_iheval_with_callables(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        tasks=tasks,
        settings=settings,
        format_record_fn=format_record_fn,
        generate_batch_fn=gen_fn,
        generation_batch_size=cfg.iheval.generation_batch_size,
        run_metadata={
            "model": args.model,
            "format": args.format,
            "ise_active": effective_has_ise,
        },
    )

    logger.info(
        "IHEval done. iheval_score=%.3f, per_setting_macro=%s",
        metrics["iheval_score"], metrics["per_setting_macro"],
    )
    logger.info("Metrics → %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
