#!/usr/bin/env python3
"""Detect and fix broken y_l instances in Phase 2 and Phase 3 DPO data.

Detects up to 4 problem categories:
  1. Refusal y_l: rejected field matches is_refusal() patterns
  2. Weak y_l (--fix-weak-yl): low keyword overlap between rejected and L3/L4 content
  3. Broken y_w (--fix-yw, L0_vs_L4 only): chosen field is a refusal or role-mismatch
  4. Wrong injection type (L0_vs_L4 only): injection_template_id targets_safety is False or None

Use --phase 2|3 to select the phase (adjusts default input/output paths).
Use --split train|val to auto-adjust default paths to data/dpo/{split}/.

In dry-run mode: prints summary and writes needs-regeneration list. No API calls.
In live mode: calls Claude API to rephrase broken instances and writes fixed output.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.api.anthropic_client import AnthropicClient
from src.data.dpo.yl_detect import detect_problems, load_injection_templates
from src.data.dpo.yl_rephrase import apply_rephrase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PHASE_DEFAULTS: dict[int, dict[str, str]] = {
    2: {
        "input": "data/dpo/phase2_gpt4o_mini_original.jsonl",
        "output": "data/dpo/phase2_gpt4o_mini.jsonl",
    },
    3: {
        "input": "data/dpo/phase3_claude_original.jsonl",
        "output": "data/dpo/phase3_claude_fixed.jsonl",
    },
}
DEFAULT_TEMPLATES = "data/libraries/injection_templates.json"


def fix_phase_output(
    input_path: Path,
    output_path: Path,
    injection_templates_path: Path,
    dry_run: bool,
    fix_weak_yl: bool,
    fix_yw: bool,
    skip_calibration: bool = False,
    anthropic_client: AnthropicClient | None = None,
    phase_label: str = "DPO",
) -> dict[str, int]:
    """Run detection (and optional rephrase) over all instances.

    In dry-run mode: prints summary and writes needs-regeneration list. No API calls.
    In live mode: calls Claude API to rephrase broken instances and writes fixed output.

    Args:
        input_path: Path to phase JSONL input.
        output_path: Path to write fixed JSONL output.
        injection_templates_path: Path to injection_templates.json.
        dry_run: If True, report only without modifications.
        fix_weak_yl: Also flag weak y_l via Jaccard similarity.
        fix_yw: Also flag broken y_w for L0_vs_L4 instances.
        skip_calibration: If True, skip calibration examples (is_calibration=True).
        anthropic_client: Required when dry_run is False.
        phase_label: Label for log/report output (e.g. "Phase 2", "Phase 3").

    Returns:
        Dict of problem category -> count plus fix/fail tallies.

    Raises:
        ValueError: If dry_run is False but anthropic_client is None.
    """
    if not dry_run and anthropic_client is None:
        raise ValueError("anthropic_client is required when dry_run=False")

    injection_safety = load_injection_templates(injection_templates_path)

    instances: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    logger.info("Loaded %d instances from %s", len(instances), input_path)

    counts: dict[str, int] = {
        "yl_refusal": 0,
        "yl_weak": 0,
        "yw_broken": 0,
        "wrong_injection": 0,
        "total_flagged": 0,
        "skipped_calibration": 0,
        "fixed": 0,
        "failed": 0,
    }
    needs_regen: list[dict] = []
    needs_manual_review: list[dict] = []

    for inst in instances:
        if skip_calibration and inst.get("is_calibration"):
            counts["skipped_calibration"] += 1
            continue

        problems = detect_problems(inst, injection_safety, fix_weak_yl, fix_yw)
        if problems:
            counts["total_flagged"] += 1
            for p in problems:
                counts[p] += 1
            if "wrong_injection" in problems:
                needs_regen.append(inst)
            elif not dry_run:
                fixed, _ = apply_rephrase(anthropic_client, inst, problems)
                if fixed:
                    counts["fixed"] += 1
                else:
                    counts["failed"] += 1
                    needs_manual_review.append(inst)

    if dry_run:
        print(f"\n=== {phase_label} DPO Dry-Run Detection Report ===")
        print(f"Total instances:       {len(instances)}")
        if skip_calibration:
            print(f"Skipped (calibration): {counts['skipped_calibration']}")
        print(f"Total flagged:         {counts['total_flagged']}")
        print(f"  yl_refusal:          {counts['yl_refusal']}")
        print(f"  yl_weak:             {counts['yl_weak']}  (--fix-weak-yl={'on' if fix_weak_yl else 'off'})")
        print(f"  yw_broken:           {counts['yw_broken']}  (--fix-yw={'on' if fix_yw else 'off'})")
        print(f"  wrong_injection:     {counts['wrong_injection']}")

        regen_path = input_path.parent / f"{input_path.stem}_needs_regeneration.jsonl"
        with regen_path.open("w") as f:
            for inst in needs_regen:
                f.write(json.dumps(inst) + "\n")
        print(f"\nWrote {len(needs_regen)} wrong-injection instances to {regen_path}")
    else:
        with output_path.open("w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")
        logger.info("Wrote output to %s", output_path)

        review_path = input_path.parent / f"{input_path.stem}_needs_manual_review.jsonl"
        with review_path.open("w") as f:
            for inst in needs_manual_review:
                f.write(json.dumps(inst) + "\n")

        print(f"\n=== {phase_label} DPO Fix Summary ===")
        print(f"Total instances:       {len(instances)}")
        if skip_calibration:
            print(f"Skipped (calibration): {counts['skipped_calibration']}")
        print(f"Total flagged:         {counts['total_flagged']}")
        print(f"Fixed:                 {counts['fixed']}")
        print(f"Failed rephrase:       {counts['failed']}")
        print(f"Need regeneration:     {len(needs_regen)}")
        if needs_manual_review:
            print(f"\nWrote {len(needs_manual_review)} unresolved instances to {review_path}")

    return counts


# Keep old name as alias for backwards compatibility with any external callers
fix_phase3_output = fix_phase_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and fix broken y_l instances in Phase 2 or Phase 3 DPO data."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--injection-templates", type=Path, default=Path(DEFAULT_TEMPLATES)
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; no modifications.")
    parser.add_argument("--fix-weak-yl", action="store_true", help="Also flag weak y_l via Jaccard.")
    parser.add_argument("--fix-yw", action="store_true", help="Also flag broken y_w for L0_vs_L4.")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[2, 3],
        default=3,
        help="Which phase to fix (adjusts default input/output paths). Default: 3.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Which split to process (adjusts default input/output paths).",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration examples (is_calibration=True). Recommended for Phase 2.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"error: unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    # Determine effective defaults based on phase and split
    defaults = _PHASE_DEFAULTS[args.phase]
    default_input = Path(defaults["input"])
    default_output = Path(defaults["output"])

    if args.split:
        # Insert split directory: data/dpo/phase2_... -> data/dpo/{split}/phase2_...
        default_input = default_input.parent / args.split / default_input.name
        default_output = default_output.parent / args.split / default_output.name

    input_path = args.input if args.input is not None else default_input
    output_path = args.output if args.output is not None else default_output

    # Resolve paths relative to project root so script works from any directory
    root = _project_root
    input_path = input_path if input_path.is_absolute() else root / input_path
    output_path = output_path if output_path.is_absolute() else root / output_path
    templates_path = (
        args.injection_templates
        if args.injection_templates.is_absolute()
        else root / args.injection_templates
    )

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not templates_path.exists():
        print(f"error: injection templates not found: {templates_path}", file=sys.stderr)
        sys.exit(1)

    client: AnthropicClient | None = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
            sys.exit(1)
        client = AnthropicClient()

    fix_phase_output(
        input_path=input_path,
        output_path=output_path,
        injection_templates_path=templates_path,
        dry_run=args.dry_run,
        fix_weak_yl=args.fix_weak_yl,
        fix_yw=args.fix_yw,
        skip_calibration=args.skip_calibration,
        anthropic_client=client,
        phase_label=f"Phase {args.phase}",
    )


if __name__ == "__main__":
    main()
