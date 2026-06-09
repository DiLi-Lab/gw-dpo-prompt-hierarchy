#!/usr/bin/env python3
"""Run the XSTest external benchmark against a registered model checkpoint.

Usage:
    python bin/run_xstest.py --model gw_dpo --format delimited
    python bin/run_xstest.py --model base_stock --format chat_template
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
from src.evaluation.external.prompt_formats import (  # noqa: E402
    build_chat_template,
    build_delimited,
)
from src.evaluation.external.registry import resolve_model  # noqa: E402
from src.evaluation.external.resume import (  # noqa: E402
    resolve_output_dir,
    save_run_args,
    validate_run_args,
)
from src.evaluation.external.xstest.runner import run_xstest_with_callables  # noqa: E402
from src.evaluation.loader import load_model_for_eval  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model nickname (see registry)")
    p.add_argument(
        "--format", choices=("delimited", "chat_template"),
        default="delimited",
        help="Prompt format. delimited = native 5-level; chat_template = "
             "tokenizer.apply_chat_template (ISE bypassed).",
    )
    p.add_argument("--config", default="configs/xstest.yaml")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on records (debug).")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Reuse the newest run_<ts>/ under the canonical output base; "
             "skip with exit 0 if metrics.json already exists. Use "
             "--no-resume to force a fresh run.",
    )
    args = p.parse_args()
    return args


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config), args.override)
    if cfg.xstest is None:
        msg = f"{args.config} did not provide an `xstest:` section."
        raise ValueError(msg)

    base_dir = (
        cfg.paths.external_runs_dir
        / "xstest"
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

    # ISE is forcibly disabled for chat_template runs (no delimiter spans
    # to derive segment IDs from). Implementation: pass has_ise=False to
    # the generation closure even when the loader produced an ISE wrapper.
    effective_has_ise = has_ise and args.format == "delimited"
    if has_ise and not effective_has_ise:
        logger.warning(
            "Format=chat_template: ISE wrapper present but bypassed for"
            " inference. metrics.json will record ise_active=false.",
        )

    gen_fn = build_generate_fn(
        model, tokenizer, has_ise=effective_has_ise,
        max_new_tokens=cfg.xstest.generation_max_new_tokens,
        temperature=cfg.xstest.generation_temperature,
    )

    if args.format == "delimited":
        def format_prompt_fn(record: dict) -> str:
            return build_delimited(l3=record["prompt"]) + "\n<|RESP_START|>"
    else:
        def format_prompt_fn(record: dict) -> str:
            return build_chat_template(tokenizer, user=record["prompt"])

    openai = OpenAIClient()

    def judge_fn(system_prompt: str, user_prompt: str) -> str:
        return openai.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=cfg.xstest.judge_model,
            temperature=cfg.xstest.judge_temperature,
            max_tokens=cfg.xstest.judge_max_tokens,
        )

    csv_path = Path(cfg.xstest.data_csv)
    if args.limit is not None:
        trimmed = output_dir / "_trimmed.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        with Path(cfg.xstest.data_csv).open() as fin, trimmed.open("w") as fout:
            for i, line in enumerate(fin):
                if i > args.limit:  # +1 to keep the header row
                    break
                fout.write(line)
        csv_path = trimmed

    metrics = run_xstest_with_callables(
        csv_path=csv_path,
        output_dir=output_dir,
        format_prompt_fn=format_prompt_fn,
        generate_batch_fn=gen_fn,
        judge_fn=judge_fn,
        generation_batch_size=cfg.xstest.generation_batch_size,
        run_metadata={
            "model": args.model,
            "format": args.format,
            "ise_active": effective_has_ise,
            "judge_model": cfg.xstest.judge_model,
        },
    )
    logger.info(
        "XSTest done. FRR_safe=%.3f compliance_safe=%.3f refusal_unsafe=%.3f",
        metrics["false_refusal_rate_safe"],
        metrics["compliance_rate_safe"],
        metrics["refusal_rate_unsafe"],
    )
    logger.info("Metrics → %s", output_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
