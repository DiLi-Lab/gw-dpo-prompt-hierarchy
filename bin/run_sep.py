#!/usr/bin/env python3
"""Run the SEP external benchmark against a registered model checkpoint.

Usage:
    python bin/run_sep.py --model gw_dpo --format delimited
    python bin/run_sep.py --model base_stock --format chat_template
    python bin/run_sep.py --model gw_dpo --format delimited --limit 50
"""

import argparse
import csv
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
from src.evaluation.external.registry import resolve_model  # noqa: E402
from src.evaluation.external.resume import (  # noqa: E402
    resolve_output_dir,
    save_run_args,
    validate_run_args,
)
from src.evaluation.external.sep.prompt import (  # noqa: E402
    build_sep_chat_template,
    build_sep_delimited_mapping_a,
)
from src.evaluation.external.sep.runner import run_sep_with_callables  # noqa: E402
from src.evaluation.loader import load_model_for_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Model nickname (see registry).")
    p.add_argument(
        "--format", choices=("delimited", "chat_template"),
        default="delimited",
        help="Prompt format. delimited = native 5-level (ISE active); "
             "chat_template = tokenizer.apply_chat_template (ISE bypassed).",
    )
    p.add_argument(
        "--mapping", choices=("A", "B"), default="A",
        help="Hierarchy mapping. A = instruction->L1, data->L3 (v1, default). "
             "B = instruction->L3, data->L4 (v2 follow-up; not implemented).",
    )
    p.add_argument("--config", default="configs/sep.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, cap records (debug). Trims a temp CSV.")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Reuse the newest run_<ts>/ under the canonical output base; "
             "skip with exit 0 if metrics.json already exists. Use "
             "--no-resume to force a fresh run.",
    )
    return p.parse_args()


def _trim_csv_to_limit(src_path: Path, dst_path: Path, limit: int) -> None:
    """Re-emit the first ``limit`` data rows of ``src_path`` to ``dst_path``.

    Uses ``csv.reader``/``csv.writer`` so quoted cells with embedded newlines
    stay intact (line-based trimming would split them and corrupt the row).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open(newline="", encoding="utf-8") as fin, \
            dst_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    if args.mapping == "B":
        msg = (
            "Mapping B (instruction->L3, data->L4) is a v2 follow-up; "
            "not implemented yet. See "
            "docs/superpowers/specs/2026-05-05-sep-design.md §5."
        )
        raise SystemExit(msg)

    cfg = load_config(Path(args.config), args.override)
    if cfg.sep is None:
        msg = f"{args.config} did not provide a `sep:` section."
        raise ValueError(msg)

    csv_path = cfg.paths.sep_subsample_csv
    if not csv_path.exists():
        msg = (
            f"Missing SEP subsample CSV at {csv_path}. "
            "Run `python bin/build_sep_subsample.py` first."
        )
        raise SystemExit(msg)

    base_dir = (
        cfg.paths.external_runs_dir
        / "sep"
        / f"{args.model}__{args.format}__mapping{args.mapping}"
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
        "mapping": args.mapping,
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
        max_new_tokens=cfg.sep.generation_max_new_tokens,
        temperature=cfg.sep.generation_temperature,
    )

    if args.format == "delimited":
        def format_record_fn(record):
            return build_sep_delimited_mapping_a(record) + "\n<|RESP_START|>"
    else:
        def format_record_fn(record):
            return build_sep_chat_template(tokenizer, record)

    if args.limit is not None:
        trimmed = output_dir / "_trimmed.csv"
        _trim_csv_to_limit(csv_path, trimmed, args.limit)
        csv_path = trimmed

    metrics = run_sep_with_callables(
        csv_path=csv_path,
        output_dir=output_dir,
        format_record_fn=format_record_fn,
        generate_batch_fn=gen_fn,
        generation_batch_size=cfg.sep.generation_batch_size,
        scoring_min_tokens=cfg.sep.scoring_min_tokens,
        scoring_refusal_patterns=cfg.sep.scoring_refusal_patterns,
        run_metadata={
            "model": args.model,
            "format": args.format,
            "mapping": args.mapping,
            "ise_active": effective_has_ise,
            "subsample_seed": cfg.sep.subsample_seed,
            "subsample_size": cfg.sep.subsample_size,
        },
    )
    logger.info(
        "SEP done. probe_execution_rate_evaluable=%.3f"
        " probe_execution_rate_all=%.3f non_trivial_response_rate=%.3f",
        metrics["probe_execution_rate_evaluable"],
        metrics["probe_execution_rate_all"],
        metrics["non_trivial_response_rate"],
    )
    logger.info("Metrics -> %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
