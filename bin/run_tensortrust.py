#!/usr/bin/env python3
"""Run the TensorTrust external benchmark against a registered model checkpoint.

Usage:
    python bin/run_tensortrust.py --model gw_dpo --format delimited
    python bin/run_tensortrust.py --model base_stock --format chat_template
    python bin/run_tensortrust.py --model gw_dpo --format delimited --limit-hijacking 50
    python bin/run_tensortrust.py --model gw_dpo --splits hijacking
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
from src.evaluation.external.tensortrust.prompt import (  # noqa: E402
    build_tensortrust_chat_template,
    build_tensortrust_delimited,
)
from src.evaluation.external.tensortrust.runner import (  # noqa: E402
    run_tensortrust_with_callables,
)
from src.evaluation.loader import load_model_for_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
    force=True,  # override any prior basicConfig from imported libraries
)
logger = logging.getLogger(__name__)

_VALID_SPLITS = ("hijacking", "extraction")


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
    p.add_argument("--config", default="configs/tensortrust.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit-hijacking", type=int, default=None,
                   help="If set, cap hijacking records (debug). Trims a temp CSV.")
    p.add_argument("--limit-extraction", type=int, default=None,
                   help="If set, cap extraction records (debug). Trims a temp CSV.")
    p.add_argument(
        "--splits", default="hijacking,extraction",
        help=("Comma-separated list of splits to run. "
              f"Valid: {','.join(_VALID_SPLITS)}."),
    )
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


def _parse_splits(raw: str) -> tuple[str, ...]:
    parts = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not parts:
        msg = "--splits cannot be empty"
        raise SystemExit(msg)
    for s in parts:
        if s not in _VALID_SPLITS:
            msg = f"Unknown split={s!r}; valid: {','.join(_VALID_SPLITS)}"
            raise SystemExit(msg)
    return parts


def main() -> int:
    args = _parse_args()
    splits = _parse_splits(args.splits)

    cfg = load_config(Path(args.config), args.override)
    if cfg.tensortrust is None:
        msg = f"{args.config} did not provide a `tensortrust:` section."
        raise ValueError(msg)

    hijacking_csv = cfg.paths.tensortrust_hijacking_csv
    extraction_csv = cfg.paths.tensortrust_extraction_csv
    for split, csv_path in [
        ("hijacking", hijacking_csv),
        ("extraction", extraction_csv),
    ]:
        if split in splits and not csv_path.exists():
            msg = (
                f"Missing TensorTrust {split} CSV at {csv_path}. "
                "Run `python bin/build_tensortrust_subsample.py` first."
            )
            raise SystemExit(msg)

    base_dir = (
        cfg.paths.external_runs_dir
        / "tensortrust"
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
        "splits": list(splits),
        "limit_hijacking": args.limit_hijacking,
        "limit_extraction": args.limit_extraction,
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
        max_new_tokens=cfg.tensortrust.generation_max_new_tokens,
        temperature=cfg.tensortrust.generation_temperature,
    )

    if args.format == "delimited":
        def format_record_fn(record):
            return build_tensortrust_delimited(record) + "\n<|RESP_START|>"
    else:
        def format_record_fn(record):
            return build_tensortrust_chat_template(tokenizer, record)

    if args.limit_hijacking is not None and "hijacking" in splits:
        trimmed = output_dir / "_trimmed_hijacking.csv"
        _trim_csv_to_limit(hijacking_csv, trimmed, args.limit_hijacking)
        hijacking_csv = trimmed
    if args.limit_extraction is not None and "extraction" in splits:
        trimmed = output_dir / "_trimmed_extraction.csv"
        _trim_csv_to_limit(extraction_csv, trimmed, args.limit_extraction)
        extraction_csv = trimmed

    metrics = run_tensortrust_with_callables(
        hijacking_csv=hijacking_csv,
        extraction_csv=extraction_csv,
        splits=splits,
        output_dir=output_dir,
        format_record_fn=format_record_fn,
        generate_batch_fn=gen_fn,
        generation_batch_size=cfg.tensortrust.generation_batch_size,
        rouge_recall_threshold=cfg.tensortrust.rouge_recall_threshold,
        run_metadata={
            "model": args.model,
            "format": args.format,
            "ise_active": effective_has_ise,
        },
    )
    logger.info(
        "TensorTrust done. HRR=%.3f ERR=%.3f macro=%.3f",
        metrics["hijacking_robustness_rate"],
        metrics["extraction_robustness_rate"],
        metrics["tensortrust_macro"],
    )
    logger.info("Metrics -> %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
