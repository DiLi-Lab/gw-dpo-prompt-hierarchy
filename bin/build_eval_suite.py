#!/usr/bin/env python3
"""Build the eval suite for prompt-hierarchy evaluation.

Usage:
    python bin/build_eval_suite.py --dry-run
    python bin/build_eval_suite.py
    python bin/build_eval_suite.py --phase 1
    python bin/build_eval_suite.py --phase 3
    python bin/build_eval_suite.py --phase 4
    python bin/build_eval_suite.py --phase 5
    python bin/build_eval_suite.py --phase 6
    python bin/build_eval_suite.py --resume
    python bin/build_eval_suite.py --skip-qc
    python bin/build_eval_suite.py --count 5 --dry-run

Phase map:
    1 = scenario generation + gold responses (requires OPENAI_API_KEY + ANTHROPIC_API_KEY)
    3 = aligned controls
    4 = reference baselines
    5 = dual-judge QC (requires OPENAI_API_KEY + GOOGLE_CLOUD_PROJECT)
    6 = final assembly
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _require_env(var_name: str, phase: int) -> str:
    """Return env var value or exit with a clear error."""
    value = os.environ.get(var_name)
    if not value:
        logger.error(
            "Missing required environment variable %s for phase %d. "
            "Set it in .env or export it.",
            var_name,
            phase,
        )
        sys.exit(1)
    return value


def main() -> None:
    """CLI entry point for building the eval suite."""
    parser = argparse.ArgumentParser(
        description="Build the eval suite for prompt-hierarchy evaluation.",
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
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 3, 4, 5, 6],
        default=None,
        help="Run a single phase (1=scenarios+gold, 3=controls, 4=baselines, "
        "5=QC, 6=assembly). Runs all if omitted.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override count per conflict pair.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load caches and skip completed work.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expected counts and exit.",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip Phase 5 dual-judge QC.",
    )
    parser.add_argument(
        "--skip-near-dedup",
        action="store_true",
        help="Skip embedding-based near-deduplication in Phase 6.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: %s" % " ".join(unknown))

    cfg = load_config(config_path=args.config, overrides=args.override)
    random.seed(args.seed)

    # Determine phases to run
    if args.phase is not None:
        phases_to_run = [args.phase]
    elif args.skip_qc:
        phases_to_run = [1, 3, 4, 6]
    else:
        phases_to_run = [1, 3, 4, 5, 6]

    count_per_pair = args.count if args.count is not None else cfg.eval.count_per_pair

    # ---------------------------------------------------------------
    # Dry run
    # ---------------------------------------------------------------
    if args.dry_run:
        num_pairs = cfg.eval.num_pairs
        total_conflicts = count_per_pair * num_pairs
        total_aligned = total_conflicts
        total_reference = cfg.eval.reference_per_pair * num_pairs

        logger.info("=== Eval Suite Dry Run ===")
        logger.info("  Phases:               %s", phases_to_run)
        logger.info("  Count per pair:       %d", count_per_pair)
        logger.info("  Num conflict pairs:   %d", num_pairs)
        logger.info("  Total conflicts:      %d", total_conflicts)
        logger.info("  Total aligned:        %d", total_aligned)
        logger.info("  Total reference:      %d", total_reference)
        logger.info("  Scenario model:       %s", cfg.eval.scenario_model)
        logger.info("  Gold model:           %s", cfg.eval.gold_model)
        logger.info("  Judge model 1:        %s", cfg.eval.judge_model_1)
        logger.info("  Judge model 2:        %s", cfg.eval.judge_model_2)
        logger.info("  Judge min score:      %d", cfg.eval.judge_min_score)
        logger.info("  Near-dedup threshold: %.2f", cfg.eval.near_dedup_threshold)
        logger.info("  Skip QC:              %s", args.skip_qc)
        logger.info("  Skip near-dedup:      %s", args.skip_near_dedup)
        logger.info("  Resume:               %s", args.resume)
        logger.info("  Seed:                 %d", args.seed)
        logger.info("Dry run complete. Exiting.")
        return

    # ---------------------------------------------------------------
    # Validate prerequisites
    # ---------------------------------------------------------------
    required_paths = {
        "l0_rules": cfg.paths.l0_rules,
        "l1_library": cfg.paths.l1_library,
        "l4_library": cfg.paths.l4_library,
        "injection_templates": cfg.paths.injection_templates,
        "alpaca_eval": cfg.paths.alpaca_eval,
        "dolly_eval": cfg.paths.dolly_eval,
    }
    missing = {name: path for name, path in required_paths.items() if not path.exists()}
    if missing:
        for name, path in missing.items():
            logger.error("Missing prerequisite: %s (%s)", name, path)
        logger.error(
            "Run bin/build_libraries.py and bin/download_base_datasets.py first."
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # Deferred imports
    # ---------------------------------------------------------------
    from datasets import load_from_disk

    from src.data.dpo.build_dpo_dataset import collect_used_base_keys
    from src.data.libraries.injection_templates import load_injection_templates
    from src.data.libraries.l0_rules import load_l0_rules
    from src.data.libraries.l1_prompts import load_l1_library
    from src.data.libraries.l4_tool_outputs import load_l4_library

    # ---------------------------------------------------------------
    # Load libraries and base datasets
    # ---------------------------------------------------------------
    logger.info("Loading libraries...")
    l0_rules = load_l0_rules(cfg.paths.l0_rules)
    l1_library = load_l1_library(cfg.paths.l1_library)
    l4_entries = load_l4_library(cfg.paths.l4_library)
    injection_templates = load_injection_templates(cfg.paths.injection_templates)

    l4_lookup: dict[tuple[str, int], dict[str, str]] = {
        (e.source, e.index): {"l4_content": e.l4_content, "generation": e.generation}
        for e in l4_entries
    }

    logger.info("Loading base datasets...")
    alpaca_eval = load_from_disk(str(cfg.paths.alpaca_eval))
    dolly_eval = load_from_disk(str(cfg.paths.dolly_eval))

    # Tag rows with source and index
    tagged_rows: list[dict] = []
    for i, row in enumerate(alpaca_eval):
        d = dict(row)
        d["_dpo_source"] = "alpaca"
        d["_dpo_index"] = i
        tagged_rows.append(d)
    for i, row in enumerate(dolly_eval):
        d = dict(row)
        d["_dpo_source"] = "dolly"
        d["_dpo_index"] = i
        tagged_rows.append(d)

    # ---------------------------------------------------------------
    # Cross-split exclusion: exclude rows used by val AND train SFT/DPO
    # ---------------------------------------------------------------
    for split in ("train", "val"):
        split_paths = cfg.paths.for_split(split)
        used_keys: set[tuple[str, int]] = set()

        if split_paths.sft_combined.exists():
            used_keys |= collect_used_base_keys(split_paths.sft_combined)

        for dpo_path in [
            split_paths.dpo_phase1,
            split_paths.dpo_phase2,
            split_paths.dpo_phase3_original,
            split_paths.dpo_phase3,
        ]:
            used_keys |= collect_used_base_keys(dpo_path)

        if used_keys:
            before = len(tagged_rows)
            tagged_rows = [
                r for r in tagged_rows
                if (r["_dpo_source"], r["_dpo_index"]) not in used_keys
            ]
            logger.info(
                "Cross-split exclusion (%s): removed %d rows (%d remaining)",
                split, before - len(tagged_rows), len(tagged_rows),
            )

    # Build base_rows_by_key for Phase 3
    base_rows_by_key: dict[tuple[str | None, int | None], dict] = {}
    for row in tagged_rows:
        key = (row.get("_dpo_source"), row.get("_dpo_index"))
        base_rows_by_key[key] = row

    # ---------------------------------------------------------------
    # Phase 1: Scenario generation + gold responses
    # ---------------------------------------------------------------
    conflict_instances: list[dict] = []

    if 1 in phases_to_run:
        logger.info("=== Phase 1: Scenario generation + gold responses ===")
        _require_env("OPENAI_API_KEY", 1)
        _require_env("ANTHROPIC_API_KEY", 1)

        from src.api.anthropic_client import AnthropicClient
        from src.api.openai_client import OpenAIClient
        from src.data.eval.conflict_scenarios import run_phase1_and_2

        openai_client = OpenAIClient()
        anthropic_client = AnthropicClient()

        cache: dict | None = None
        if args.resume and cfg.paths.eval_scenario_cache.exists():
            with open(cfg.paths.eval_scenario_cache, "r", encoding="utf-8") as f:
                cache = {
                    item["key"]: item
                    for line in f
                    for item in [json.loads(line)]
                    if "key" in item
                }
            logger.info("Loaded %d scenario cache entries", len(cache))

        conflict_instances = run_phase1_and_2(
            base_rows=tagged_rows,
            l0_rules=l0_rules,
            l1_library=l1_library,
            injection_templates=injection_templates,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            output_path=cfg.paths.eval_scenarios_raw,
            count_per_pair=count_per_pair,
            seed=args.seed,
            cache=cache,
        )
        logger.info("Phase 1 complete: %d conflict instances", len(conflict_instances))

    else:
        # Load from file if phase 1 was skipped.
        # Prefer QC-kept results over raw scenarios when available,
        # so that Phase 6 only processes instances that passed QC.
        if cfg.paths.eval_qc_results.exists() and 5 not in phases_to_run:
            with open(cfg.paths.eval_qc_results, "r", encoding="utf-8") as f:
                conflict_instances = [json.loads(line) for line in f if line.strip()]
            logger.info(
                "Phase 1 skipped: loaded %d QC-kept instances from %s",
                len(conflict_instances), cfg.paths.eval_qc_results,
            )
        elif cfg.paths.eval_scenarios_raw.exists():
            with open(cfg.paths.eval_scenarios_raw, "r", encoding="utf-8") as f:
                conflict_instances = [json.loads(line) for line in f if line.strip()]
            logger.info(
                "Phase 1 skipped: loaded %d instances from %s",
                len(conflict_instances), cfg.paths.eval_scenarios_raw,
            )
        else:
            logger.warning(
                "Phase 1 skipped and no existing output at %s",
                cfg.paths.eval_scenarios_raw,
            )

    # ---------------------------------------------------------------
    # Phase 3: Aligned controls
    # ---------------------------------------------------------------
    aligned_instances: list[dict] = []

    if 3 in phases_to_run:
        logger.info("=== Phase 3: Aligned controls ===")

        from src.data.eval.aligned_controls import run_phase3

        # Reuse clients from Phase 1 if available, else create new ones
        if "anthropic_client" not in dir():
            _require_env("ANTHROPIC_API_KEY", 3)
            from src.api.anthropic_client import AnthropicClient
            anthropic_client = AnthropicClient()

        openai_client_p3 = openai_client if "openai_client" in dir() else None  # type: ignore[name-defined]

        aligned_instances = run_phase3(
            conflict_instances=conflict_instances,
            base_rows_by_key=base_rows_by_key,
            l1_library=l1_library,
            l4_lookup=l4_lookup,
            anthropic_client=anthropic_client,  # type: ignore[name-defined]
            openai_client=openai_client_p3,
            output_path=cfg.paths.eval_aligned_raw,
            gold_model=cfg.eval.gold_model,
            seed=args.seed,
        )
        logger.info("Phase 3 complete: %d aligned instances", len(aligned_instances))

    else:
        if cfg.paths.eval_aligned_raw.exists():
            with open(cfg.paths.eval_aligned_raw, "r", encoding="utf-8") as f:
                aligned_instances = [json.loads(line) for line in f if line.strip()]
            logger.info(
                "Phase 3 skipped: loaded %d instances from %s",
                len(aligned_instances), cfg.paths.eval_aligned_raw,
            )
        else:
            logger.warning(
                "Phase 3 skipped and no existing output at %s",
                cfg.paths.eval_aligned_raw,
            )

    # ---------------------------------------------------------------
    # Phase 4: Reference baselines
    # ---------------------------------------------------------------
    reference_instances: list[dict] = []

    if 4 in phases_to_run:
        logger.info("=== Phase 4: Reference baselines ===")

        from src.data.eval.reference_baselines import run_phase4

        reference_instances = run_phase4(
            conflict_instances=conflict_instances,
            output_path=cfg.paths.eval_reference,
            per_pair=cfg.eval.reference_per_pair,
            seed=args.seed,
        )
        logger.info("Phase 4 complete: %d reference instances", len(reference_instances))

    else:
        if cfg.paths.eval_reference.exists():
            with open(cfg.paths.eval_reference, "r", encoding="utf-8") as f:
                reference_instances = [json.loads(line) for line in f if line.strip()]
            logger.info(
                "Phase 4 skipped: loaded %d instances from %s",
                len(reference_instances), cfg.paths.eval_reference,
            )
        else:
            logger.warning(
                "Phase 4 skipped and no existing output at %s",
                cfg.paths.eval_reference,
            )

    # ---------------------------------------------------------------
    # Phase 5: Dual-judge QC
    # ---------------------------------------------------------------
    if 5 in phases_to_run:
        logger.info("=== Phase 5: Dual-judge QC ===")
        _require_env("OPENAI_API_KEY", 5)
        _require_env("GOOGLE_CLOUD_PROJECT", 5)

        from src.api.google_client import GoogleClient
        from src.api.openai_client import OpenAIClient
        from src.data.eval.quality_control import run_phase5

        openai_client_p5 = OpenAIClient()
        google_client = GoogleClient()

        qc_results = run_phase5(
            conflict_instances=conflict_instances,
            openai_client=openai_client_p5,
            google_client=google_client,
            output_path=cfg.paths.eval_qc_results,
            flagged_path=cfg.paths.eval_flagged,
            judge_model_1=cfg.eval.judge_model_1,
            judge_model_2=cfg.eval.judge_model_2,
            min_score=cfg.eval.judge_min_score,
            resume=args.resume,
        )
        logger.info(
            "Phase 5 results: %d kept, %d discarded, %d flagged, %d errors, %d total",
            qc_results.get("kept", 0),
            qc_results.get("discarded", 0),
            qc_results.get("flagged", 0),
            qc_results.get("errors", 0),
            qc_results.get("total", 0),
        )

        # Filter instances to QC-kept only
        if cfg.paths.eval_qc_results.exists():
            with open(cfg.paths.eval_qc_results, "r", encoding="utf-8") as f:
                kept_ids: set[str] = {
                    item.get("id", "")
                    for line in f
                    for item in [json.loads(line)]
                    if "id" in item
                }
            conflict_instances = [
                inst for inst in conflict_instances
                if inst.get("id", "") in kept_ids
            ]
            aligned_instances = [
                inst for inst in aligned_instances
                if inst.get("id", "") in kept_ids
            ]
            logger.info(
                "After QC filter: %d conflict, %d aligned instances",
                len(conflict_instances), len(aligned_instances),
            )

    # ---------------------------------------------------------------
    # Phase 6: Final assembly
    # ---------------------------------------------------------------
    if 6 in phases_to_run:
        logger.info("=== Phase 6: Final assembly ===")

        from src.data.eval.build_eval_suite import run_phase6

        stats = run_phase6(
            conflict_instances=conflict_instances,
            aligned_instances=aligned_instances,
            reference_instances=reference_instances,
            output_dir=cfg.paths.eval_dir,
            near_dedup_threshold=cfg.eval.near_dedup_threshold,
            skip_near_dedup=args.skip_near_dedup,
        )
        logger.info("Phase 6 complete. Stats: %s", stats)

    logger.info("=== Eval suite build complete ===")


if __name__ == "__main__":
    main()
