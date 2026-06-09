#!/usr/bin/env python3
"""Build the DPO dataset for Gravity-Weighted DPO training.

Usage:
    python bin/build_dpo_dataset.py --dry-run
    python bin/build_dpo_dataset.py --split train --dry-run
    python bin/build_dpo_dataset.py --split val --dry-run
    python bin/build_dpo_dataset.py --split train
    python bin/build_dpo_dataset.py --split val
    python bin/build_dpo_dataset.py --phase 1
    python bin/build_dpo_dataset.py --phase 2
    python bin/build_dpo_dataset.py --phase 3
    python bin/build_dpo_dataset.py --phase 4
    python bin/build_dpo_dataset.py --phase 5
    python bin/build_dpo_dataset.py
    python bin/build_dpo_dataset.py --resume

When --split is omitted, both train and val are built sequentially.
Val split forces phases 1-4 only (no Phase 5 QC) and uses a different
seed offset (seed + 50000) for independent shuffling.
"""

import argparse
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
from src.data.dpo.pair_config import get_pair_configs

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


def _print_dry_run_table(configs: list | None = None) -> None:
    """Print a summary table of pair types, target counts, and phases.

    Args:
        configs: Override list of PairConfig objects. Uses train configs
            if None.
    """
    if configs is None:
        configs = get_pair_configs(split=None)
    header = f"{'Pair Type':<20} {'Category':<12} {'Phase':>5} {'Target':>7}"
    logger.info("=== DPO Pair Configuration Table ===")
    logger.info(header)
    logger.info("-" * len(header))
    total = 0
    for cfg in configs:
        logger.info(
            "%-20s %-12s %5d %7d",
            cfg.name,
            cfg.category,
            cfg.phase,
            cfg.target_count,
        )
        total += cfg.target_count
    logger.info("-" * len(header))
    logger.info("%-20s %-12s %5s %7d", "TOTAL", "", "", total)


def _build_split(args: argparse.Namespace, split: str) -> None:
    """Build the DPO dataset for a single split.

    Args:
        args: Parsed CLI arguments.
        split: "train" or "val".
    """
    cfg = load_config(config_path=args.config, overrides=args.override)
    cfg.paths.split = split

    seed = args.seed + 50000 if split == "val" else args.seed
    random.seed(seed)

    base_configs = get_pair_configs(split=split)

    # Determine which phases to run
    if args.phase is not None:
        phases_to_run = [args.phase]
    elif args.skip_qc or split == "val":
        # Val split forces phases 1-4 only (no Phase 5 QC)
        phases_to_run = [1, 2, 3, 4]
    else:
        phases_to_run = [1, 2, 3, 4, 5]

    logger.info("=== DPO Dataset Build (split=%s) ===", split)
    logger.info("  Phases:           %s", phases_to_run)
    logger.info("  Pair filter:      %s", args.pair or "(all)")
    logger.info("  Count override:   %s", args.count or "(default)")
    logger.info("  Resume:           %s", args.resume)
    logger.info("  Skip L2 gen:      %s", args.skip_l2_generation)
    logger.info("  Skip QC:          %s", args.skip_qc or split == "val")
    logger.info("  Skip filters:     %s", args.skip_filters)
    logger.info("  Skip near-dedup:  %s", args.skip_near_dedup)
    logger.info("  Near-dedup thresh: %.2f", args.near_dedup_threshold)
    logger.info("  Seed:             %d", seed)

    # ---------------------------------------------------------------
    # Validate prerequisites
    # ---------------------------------------------------------------
    required_paths = {
        "l0_rules": cfg.paths.l0_rules,
        "l1_library": cfg.paths.l1_library,
        "l4_library": cfg.paths.l4_library,
        "injection_templates": cfg.paths.injection_templates,
        "alpaca_train": cfg.paths.alpaca_train,
        "dolly_train": cfg.paths.dolly_train,
    }
    # sft_combined is checked softly — cross-split exclusion handles missing
    missing = {name: path for name, path in required_paths.items() if not path.exists()}
    if missing:
        for name, path in missing.items():
            logger.error("Missing prerequisite: %s (%s)", name, path)
        logger.error(
            "Run bin/build_libraries.py, bin/download_base_datasets.py, "
            "and bin/build_sft_dataset.py first."
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # Deferred imports
    # ---------------------------------------------------------------
    from datasets import load_from_disk

    from src.data.dpo.build_dpo_dataset import (
        collect_used_base_keys,
        combine_phases,
        compute_dpo_stats,
        exclude_prior_phase_rows,
        exclude_sft_rows,
        load_dpo_cache,
        partition_dpo_pool,
        run_phase1,
        run_phase2,
        run_phase3,
        save_dpo_cache,
    )
    from src.data.dpo.pair_config import get_config_by_name
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

    # Load adversarial instruction library (for L0_vs_L3)
    import json

    adv_instructions_path = cfg.paths.l0_adversarial_instructions
    if adv_instructions_path.exists():
        with open(adv_instructions_path, "r", encoding="utf-8") as f:
            l0_adversarial_instructions = json.load(f)
        logger.info("Loaded %d adversarial instructions from %s", len(l0_adversarial_instructions), adv_instructions_path)
    else:
        if 2 in phases_to_run:
            logger.error(
                "Adversarial instruction library required for Phase 2 but not found "
                "at %s. Run bin/generate_l0_adversarial_instructions.py first.",
                adv_instructions_path,
            )
            sys.exit(1)
        l0_adversarial_instructions = None
        logger.info(
            "Adversarial instruction library not found at %s (not needed for phases %s)",
            adv_instructions_path, phases_to_run,
        )

    l4_lookup: dict[tuple[str, int], dict[str, str]] = {
        (e.source, e.index): {"l4_content": e.l4_content, "generation": e.generation}
        for e in l4_entries
    }

    logger.info("Loading base datasets...")

    alpaca_train = load_from_disk(str(cfg.paths.alpaca_train))
    dolly_train = load_from_disk(str(cfg.paths.dolly_train))

    # Tag rows with source and index
    tagged_rows: list[dict] = []
    for i, row in enumerate(alpaca_train):
        d = dict(row)
        d["_dpo_source"] = "alpaca"
        d["_dpo_index"] = i
        tagged_rows.append(d)
    for i, row in enumerate(dolly_train):
        d = dict(row)
        d["_dpo_source"] = "dolly"
        d["_dpo_index"] = i
        tagged_rows.append(d)

    # Build L4 domain index (before SFT exclusion so all l4_lookup keys have instructions)
    from collections import defaultdict

    from src.data.sft.domain_classifier import classify_domain as classify_domain_fn

    base_row_instructions: dict[tuple[str, int], str] = {
        (row["_dpo_source"], row["_dpo_index"]): row.get("instruction", "")
        for row in tagged_rows
    }
    l4_domain_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in l4_lookup:
        instruction = base_row_instructions.get(key, "")
        domain = classify_domain_fn(instruction)
        l4_domain_index[domain].append(key)
    logger.info(
        "Built L4 domain index: %d domains, %d entries",
        len(l4_domain_index),
        sum(len(v) for v in l4_domain_index.values()),
    )

    # Exclude rows used by this split's SFT
    if cfg.paths.sft_combined.exists():
        pool = exclude_sft_rows(tagged_rows, cfg.paths.sft_combined)
    else:
        logger.warning(
            "SFT combined file not found at %s — skipping SFT exclusion",
            cfg.paths.sft_combined,
        )
        pool = list(tagged_rows)

    # Exclude rows used by prior phase outputs (prevents overlap when
    # running phases individually in separate invocations)
    all_phase_paths = {
        1: cfg.paths.dpo_phase1,
        2: cfg.paths.dpo_phase2,
        3: cfg.paths.dpo_phase3_original,
    }
    prior_phase_paths = [
        path for phase, path in all_phase_paths.items()
        if phase not in phases_to_run
    ]
    if prior_phase_paths:
        pool = exclude_prior_phase_rows(pool, prior_phase_paths)

    # ---------------------------------------------------------------
    # Cross-split exclusion: remove rows used by the other split
    # ---------------------------------------------------------------
    other_split = "val" if split == "train" else "train"
    other_paths = cfg.paths.for_split(other_split)

    other_used_keys: set[tuple[str, int]] = set()

    # Other split's SFT
    if other_paths.sft_combined.exists():
        other_used_keys |= collect_used_base_keys(other_paths.sft_combined)

    # Other split's DPO phases
    for dpo_path in [other_paths.dpo_phase1, other_paths.dpo_phase2,
                     other_paths.dpo_phase3_original, other_paths.dpo_phase3]:
        other_used_keys |= collect_used_base_keys(dpo_path)

    if other_used_keys:
        before = len(pool)
        pool = [
            r for r in pool
            if (r["_dpo_source"], r["_dpo_index"]) not in other_used_keys
        ]
        logger.info(
            "Cross-split exclusion: removed %d rows (%d remaining)",
            before - len(pool), len(pool),
        )

    # Determine active configs
    from dataclasses import replace

    if args.pair:
        active_configs = [get_config_by_name(args.pair)]
        if args.count is not None:
            original = active_configs[0]
            active_configs = [replace(original, target_count=args.count)]
    else:
        active_configs = [
            c for c in base_configs if c.phase in phases_to_run
        ]
        if args.count is not None:
            active_configs = [
                replace(c, target_count=args.count) for c in active_configs
            ]

    # Partition pool across pair configs
    pool_slices = partition_dpo_pool(
        rows=pool,
        configs=active_configs,
        l4_lookup=l4_lookup,
        seed=seed,
    )

    # ---------------------------------------------------------------
    # Load caches if resuming
    # ---------------------------------------------------------------
    caches: dict[str, dict] = {}
    if args.resume:
        caches["l2_cache"] = load_dpo_cache(cfg.paths.dpo_l2_cache)
        caches["yw_cache"] = load_dpo_cache(cfg.paths.dpo_yw_cache)
        caches["yl_cache"] = load_dpo_cache(cfg.paths.dpo_yl_cache)
    else:
        caches["l2_cache"] = {}
        caches["yw_cache"] = {}
        caches["yl_cache"] = {}

    # ---------------------------------------------------------------
    # Phase 1: L1-vs-L3 pairs (~$0.08 for L2 generation)
    # ---------------------------------------------------------------
    # Build a lookup from active configs for phase runners
    _active_by_name: dict[str, object] = {c.name: c for c in active_configs}

    if 1 in phases_to_run:
        logger.info("=== Phase 1: L1-vs-L3 pairs ===")
        l2_client = None
        if not args.skip_l2_generation:
            _require_env("OPENAI_API_KEY", 1)
            from src.api.openai_client import OpenAIClient
            l2_client = OpenAIClient()

        _require_env("ANTHROPIC_API_KEY", 1)
        from src.api.anthropic_client import AnthropicClient
        phase1_anthropic_client = AnthropicClient()

        l1_vs_l3_active = _active_by_name.get("L1_vs_L3", get_config_by_name("L1_vs_L3"))
        target = l1_vs_l3_active.target_count
        run_phase1(
            pool_slice=pool_slices.get("L1_vs_L3", []),
            l0_rules=l0_rules,
            l1_library=l1_library,
            injection_templates=injection_templates,
            output_path=cfg.paths.dpo_phase1,
            openai_client=l2_client,
            anthropic_client=phase1_anthropic_client,
            l2_cache=caches.get("l2_cache"),
            l4_lookup=l4_lookup,
            count=target,
            seed=seed,
        )

    # ---------------------------------------------------------------
    # Phase 2: GPT-4o-mini conflict pairs + calibration
    # ---------------------------------------------------------------
    if 2 in phases_to_run:
        logger.info("=== Phase 2: GPT-4o-mini conflict pairs + calibration ===")
        _require_env("OPENAI_API_KEY", 2)
        _require_env("ANTHROPIC_API_KEY", 2)
        from src.api.openai_client import OpenAIClient
        from src.api.anthropic_client import AnthropicClient
        openai_client = OpenAIClient()
        phase2_anthropic_client = AnthropicClient()

        run_phase2(
            pool_slices=pool_slices,
            l0_rules=l0_rules,
            l1_library=l1_library,
            l4_lookup=l4_lookup,
            injection_templates=injection_templates,
            openai_client=openai_client,
            output_path=cfg.paths.dpo_phase2_original,
            caches=caches,
            seed=seed,
            count_override=args.count if not args.pair else None,
            l0_adversarial_instructions=l0_adversarial_instructions,
            anthropic_client=phase2_anthropic_client,
        )

    # ---------------------------------------------------------------
    # Post-Phase 2: Fix y_l refusals
    # ---------------------------------------------------------------
    if 2 in phases_to_run and 4 in phases_to_run:
        logger.info("=== Post-Phase 2: Fix y_l refusals ===")
        import importlib.util
        _fix_spec = importlib.util.spec_from_file_location(
            "fix_yl_refusals", _project_root / "bin" / "fix_yl_refusals.py",
        )
        _fix_mod = importlib.util.module_from_spec(_fix_spec)
        _fix_spec.loader.exec_module(_fix_mod)

        fix_result = _fix_mod.fix_phase_output(
            input_path=cfg.paths.dpo_phase2_original,
            output_path=cfg.paths.dpo_phase2,
            injection_templates_path=cfg.paths.injection_templates,
            dry_run=False,
            fix_weak_yl=True,
            fix_yw=False,
            skip_calibration=True,
            anthropic_client=phase2_anthropic_client,
        )
        logger.info(
            "Phase 2 fix step: %d refusal y_l, %d weak y_l, %d fixed, %d failed",
            fix_result["yl_refusal"],
            fix_result["yl_weak"],
            fix_result["fixed"],
            fix_result["failed"],
        )

    # ---------------------------------------------------------------
    # Phase 3: Claude distillation + cascading
    # ---------------------------------------------------------------
    if 3 in phases_to_run:
        logger.info("=== Phase 3: Claude distillation + cascading ===")
        _require_env("OPENAI_API_KEY", 3)
        _require_env("ANTHROPIC_API_KEY", 3)

        from src.api.anthropic_client import AnthropicClient
        from src.api.openai_client import OpenAIClient
        from src.data.dpo.cascading import SEED_FAMILIES, load_cascading_families
        from src.data.dpo.l0_conflict_builder import load_l0_conflict_scenarios

        openai_client = OpenAIClient()
        anthropic_client = AnthropicClient()

        # Load seed families + any reviewed generated families
        cascading_families = list(SEED_FAMILIES)
        generated_path = cfg.paths.cascading_families_generated
        if generated_path.exists():
            extra = load_cascading_families(generated_path)
            logger.info("Loaded %d additional cascading families from %s", len(extra), generated_path)
            cascading_families.extend(extra)
        else:
            logger.info("No generated cascading families found at %s — using %d seed families only", generated_path, len(cascading_families))

        # Load L0 conflict scenarios for L0-vs-L1 and L0-vs-L2 pairs
        l0_scenarios_path = cfg.paths.l0_conflict_scenarios
        l0_conflict_scenarios = None
        if l0_scenarios_path.exists():
            l0_conflict_scenarios = load_l0_conflict_scenarios(l0_scenarios_path)
            logger.info("Loaded %d L0 conflict scenarios from %s", len(l0_conflict_scenarios), l0_scenarios_path)
        else:
            logger.warning(
                "L0 conflict scenarios not found at %s — L0-vs-L1 and L0-vs-L2 "
                "pairs will be SKIPPED. Run bin/generate_l0_conflict_scenarios.py first.",
                l0_scenarios_path,
            )

        run_phase3(
            pool_slices=pool_slices,
            l0_rules=l0_rules,
            l1_library=l1_library,
            l4_lookup=l4_lookup,
            injection_templates=injection_templates,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            cascading_families=cascading_families,
            output_path=cfg.paths.dpo_phase3_original,
            caches=caches,
            seed=seed,
            l0_conflict_scenarios=l0_conflict_scenarios,
            count_override=args.count if not args.pair else None,
            l4_domain_index=l4_domain_index,
            pair_configs=base_configs,
        )

    # ---------------------------------------------------------------
    # Post-Phase 3: Fix y_l refusals (runs only in full pipeline)
    # ---------------------------------------------------------------
    if 3 in phases_to_run and 4 in phases_to_run:
        logger.info("=== Post-Phase 3: Fix y_l refusals ===")
        import importlib.util
        _fix_spec = importlib.util.spec_from_file_location(
            "fix_yl_refusals", _project_root / "bin" / "fix_yl_refusals.py",
        )
        _fix_mod = importlib.util.module_from_spec(_fix_spec)
        _fix_spec.loader.exec_module(_fix_mod)

        fix_result = _fix_mod.fix_phase_output(
            input_path=cfg.paths.dpo_phase3_original,
            output_path=cfg.paths.dpo_phase3,
            injection_templates_path=cfg.paths.injection_templates,
            dry_run=False,
            fix_weak_yl=True,
            fix_yw=True,
            skip_calibration=False,
            anthropic_client=anthropic_client,
        )
        logger.info(
            "Fix step: %d refusal y_l, %d weak y_l, %d broken y_w, %d fixed, %d failed",
            fix_result["yl_refusal"],
            fix_result["yl_weak"],
            fix_result["yw_broken"],
            fix_result["fixed"],
            fix_result["failed"],
        )

    # ---------------------------------------------------------------
    # Phase 4: Combine + deduplicate (no API calls)
    # ---------------------------------------------------------------
    if 4 in phases_to_run:
        logger.info("=== Phase 4: Combine + deduplicate + filter ===")

        from src.data.dpo.quality_control import (
            deduplicate_by_embedding,
            deduplicate_by_hash,
            filter_dpo_example,
        )

        all_examples = combine_phases(
            phase1_path=cfg.paths.dpo_phase1,
            phase2_path=cfg.paths.dpo_phase2,
            phase3_path=cfg.paths.dpo_phase3,
            output_path=cfg.paths.dpo_combined,
        )

        count_combined = len(all_examples)
        logger.info("Combined %d examples", count_combined)

        # Step 1: Hash deduplication (always runs)
        deduped = deduplicate_by_hash(all_examples)
        count_after_hash = len(deduped)
        logger.info(
            "Hash deduplication: %d -> %d (removed %d exact duplicates)",
            count_combined, count_after_hash, count_combined - count_after_hash,
        )

        # Step 2: Automated quality filters (token length, similarity, delimiters)
        if not args.skip_filters:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(cfg.paths.tokenizer_dir))
            filtered = [ex for ex in deduped if filter_dpo_example(ex, tokenizer)]
            count_after_filter = len(filtered)
            logger.info(
                "Quality filters: %d -> %d (removed %d: short responses, "
                "high chosen/rejected similarity, or broken delimiters)",
                count_after_hash, count_after_filter,
                count_after_hash - count_after_filter,
            )
        else:
            filtered = deduped
            count_after_filter = len(filtered)
            logger.info("Quality filters: skipped (--skip-filters)")

        # Step 3: Near-deduplication by embedding similarity
        if not args.skip_near_dedup:
            count_before_near = len(filtered)
            filtered = deduplicate_by_embedding(
                filtered, threshold=args.near_dedup_threshold,
            )
            count_after_near = len(filtered)
            logger.info(
                "Near-deduplication (threshold=%.2f): %d -> %d (removed %d)",
                args.near_dedup_threshold,
                count_before_near, count_after_near,
                count_before_near - count_after_near,
            )
        else:
            count_after_near = len(filtered)
            logger.info("Near-deduplication: skipped (--skip-near-dedup)")

        # Summary
        logger.info(
            "=== Phase 4 filtering summary ===\n"
            "  Combined:              %d\n"
            "  After hash dedup:      %d (-%d)\n"
            "  After quality filters: %d (-%d)\n"
            "  After near-dedup:      %d (-%d)\n"
            "  Total removed:         %d (%.1f%%)",
            count_combined,
            count_after_hash, count_combined - count_after_hash,
            count_after_filter, count_after_hash - count_after_filter,
            count_after_near, count_after_filter - count_after_near,
            count_combined - count_after_near,
            (count_combined - count_after_near) * 100 / count_combined if count_combined else 0,
        )

        import json

        cfg.paths.dpo_combined.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.paths.dpo_combined, "w", encoding="utf-8") as f:
            for ex in filtered:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info("Phase 4 complete. Output: %s (%d examples)", cfg.paths.dpo_combined, len(filtered))

        stats = compute_dpo_stats(filtered)
        logger.info("=== DPO Dataset Stats ===")
        logger.info("  Total:           %d", stats["total"])
        for key in ("conflict_types", "categories", "yw_sources", "yl_sources"):
            if stats.get(key):
                logger.info("  %s:", key)
                for name, count in sorted(stats[key].items()):
                    logger.info("    %s: %d", name, count)

    # ---------------------------------------------------------------
    # Phase 5: Dual-judge evaluation (train only)
    # ---------------------------------------------------------------
    if 5 in phases_to_run:
        logger.info("=== Phase 5: Dual-judge evaluation ===")
        _require_env("OPENAI_API_KEY", 5)
        _require_env("GOOGLE_CLOUD_PROJECT", 5)

        from src.api.google_client import GoogleClient
        from src.api.openai_client import OpenAIClient
        from src.data.dpo.build_dpo_dataset import run_phase5

        openai_client = OpenAIClient()
        google_client = GoogleClient()

        qc_results = run_phase5(
            combined_path=cfg.paths.dpo_combined,
            openai_client=openai_client,
            google_client=google_client,
            qc_results_path=cfg.paths.dpo_qc_results,
            flagged_path=cfg.paths.dpo_flagged,
            seed=seed,
            resume=args.resume,
        )
        logger.info(
            "Phase 5 results: %d sampled, %d kept, %d discarded, %d flagged, %d skipped",
            qc_results["sampled"], qc_results["kept"],
            qc_results["discarded"], qc_results["flagged"],
            qc_results["skipped"],
        )

    # Save caches
    if caches.get("l2_cache"):
        save_dpo_cache(caches["l2_cache"], cfg.paths.dpo_l2_cache)
    if caches.get("yw_cache"):
        save_dpo_cache(caches["yw_cache"], cfg.paths.dpo_yw_cache)
    if caches.get("yl_cache"):
        save_dpo_cache(caches["yl_cache"], cfg.paths.dpo_yl_cache)

    logger.info("=== DPO dataset build complete (split=%s) ===", split)


def main() -> None:
    """CLI entry point for building the DPO dataset."""
    parser = argparse.ArgumentParser(
        description="Build the DPO dataset for Gravity-Weighted DPO training.",
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
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run a single phase (1-5). Runs all if omitted.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Build a single pair type (e.g. L1_vs_L3).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override target count. With --pair: applies to that pair. "
        "Without --pair: applies to every conflict pair.",
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
        help="Skip Phase 5 dual-judge evaluation.",
    )
    parser.add_argument(
        "--skip-l2-generation",
        action="store_true",
        help="Use template L2 instead of GPT-4o-mini.",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip automated quality filters in Phase 4 (token length, "
        "chosen/rejected similarity, delimiter integrity).",
    )
    parser.add_argument(
        "--skip-near-dedup",
        action="store_true",
        help="Skip embedding-based near-deduplication in Phase 4 "
        "(requires sentence-transformers).",
    )
    parser.add_argument(
        "--near-dedup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for near-deduplication (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Build train or val split. Builds both sequentially if omitted.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: %s" % " ".join(unknown))

    if args.dry_run:
        from dataclasses import replace

        from src.data.dpo.pair_config import get_config_by_name

        splits = [args.split] if args.split else ["train", "val"]
        for split in splits:
            base_configs = get_pair_configs(split=split)

            # Determine phases for this split
            if args.phase is not None:
                phases_to_run = [args.phase]
            elif args.skip_qc or split == "val":
                phases_to_run = [1, 2, 3, 4]
            else:
                phases_to_run = [1, 2, 3, 4, 5]

            if args.pair:
                dry_configs = [get_config_by_name(args.pair)]
                if args.count is not None:
                    dry_configs = [replace(dry_configs[0], target_count=args.count)]
            elif args.count is not None:
                dry_configs = [
                    replace(c, target_count=args.count)
                    for c in base_configs
                    if c.phase in phases_to_run
                ]
            else:
                dry_configs = [
                    c for c in base_configs if c.phase in phases_to_run
                ]

            logger.info("========== Dry-run for split: %s ==========", split)
            _print_dry_run_table(dry_configs)

        logger.info("Dry run complete. Exiting.")
        return

    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        logger.info("========== Building DPO split: %s ==========", split)
        _build_split(args, split)


if __name__ == "__main__":
    main()
