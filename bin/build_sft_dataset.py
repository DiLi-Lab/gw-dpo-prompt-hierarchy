#!/usr/bin/env python3
"""Build the SFT dataset for instruction hierarchy training.

Usage:
    python bin/build_sft_dataset.py --split train
    python bin/build_sft_dataset.py --split val
    python bin/build_sft_dataset.py [--aligned-count N] [--synthesis-count N]
    python bin/build_sft_dataset.py --skip-synthesis
    python bin/build_sft_dataset.py --dry-run
"""

import argparse
import logging
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

# Train defaults (match argparse defaults)
_TRAIN_DEFAULTS = {
    "aligned_count": 7000,
    "synthesis_count": 2000,
    "partial_count": 500,
    "misaligned_count": 250,
}

_VAL_DEFAULTS = {
    "aligned_count": 1050,
    "synthesis_count": 300,
    "partial_count": 125,
    "misaligned_count": 38,
}


def _validate_prerequisites(cfg: object) -> None:
    """Check that all prerequisite library files exist."""
    required = {
        "l0_rules": cfg.paths.l0_rules,
        "l1_library": cfg.paths.l1_library,
        "l4_library": cfg.paths.l4_library,
        "injection_templates": cfg.paths.injection_templates,
        "alpaca_train": cfg.paths.alpaca_train,
        "dolly_train": cfg.paths.dolly_train,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        for name, path in missing.items():
            logger.error("Missing prerequisite: %s (%s)", name, path)
        logger.error(
            "Run bin/build_libraries.py and bin/download_base_datasets.py first."
        )
        sys.exit(1)


def _build_l4_lookup(
    l4_entries: list,
) -> dict[tuple[str, int], dict[str, str]]:
    """Build a lookup dict from L4 entries: (source, index) -> {l4_content, generation}."""
    return {
        (e.source, e.index): {"l4_content": e.l4_content, "generation": e.generation}
        for e in l4_entries
    }


def _build_split(args: argparse.Namespace, split: str) -> None:
    """Build a single SFT split (train or val)."""
    cfg = load_config(config_path=args.config, overrides=args.override)
    cfg.paths.split = split

    # Apply val defaults when user hasn't overridden (value equals train default)
    aligned_count = args.aligned_count
    synthesis_count = args.synthesis_count
    partial_count = args.partial_count
    misaligned_count = args.misaligned_count

    if split == "val":
        if aligned_count == _TRAIN_DEFAULTS["aligned_count"]:
            aligned_count = _VAL_DEFAULTS["aligned_count"]
        if synthesis_count == _TRAIN_DEFAULTS["synthesis_count"]:
            synthesis_count = _VAL_DEFAULTS["synthesis_count"]
        if partial_count == _TRAIN_DEFAULTS["partial_count"]:
            partial_count = _VAL_DEFAULTS["partial_count"]
        if misaligned_count == _TRAIN_DEFAULTS["misaligned_count"]:
            misaligned_count = _VAL_DEFAULTS["misaligned_count"]

    # Use different seed for val to ensure different shuffling
    seed = args.seed + 50000 if split == "val" else args.seed
    random.seed(seed)

    # Compute counts
    if args.skip_synthesis:
        simple_count = aligned_count
        synth_count = 0
    else:
        synth_count = synthesis_count
        simple_count = aligned_count - synth_count

    total_partial = 4 * partial_count
    total_misaligned = 4 * misaligned_count
    total_expected = aligned_count + total_partial + total_misaligned

    logger.info("=== SFT Dataset Build Plan (%s) ===", split)
    logger.info("  Simple aligned:    %d", simple_count)
    logger.info("  Context synthesis: %d%s", synth_count,
                " (skipped)" if args.skip_synthesis else "")
    logger.info("  Partial (4 configs x %d): %d", partial_count, total_partial)
    logger.info("  Misaligned (4 types x %d): %d", misaligned_count, total_misaligned)
    logger.info("  Expected total:    %d", total_expected)
    logger.info("  Output:            %s", cfg.paths.sft_combined)
    logger.info("  Seed:              %d", seed)

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # Validate prerequisites
    _validate_prerequisites(cfg)

    # Deferred imports
    from datasets import load_from_disk

    from src.data.dpo.build_dpo_dataset import collect_used_base_keys
    from src.data.libraries.l0_rules import load_l0_rules
    from src.data.libraries.l1_prompts import load_l1_library
    from src.data.libraries.l4_tool_outputs import load_l4_library
    from src.data.sft.aligned import build_context_synthesis_aligned, build_simple_aligned
    from src.data.sft.build_sft_dataset import compute_sft_stats, load_sft_dataset, save_sft_dataset
    from src.data.sft.misaligned import build_misaligned_examples
    from src.data.sft.partial import build_partial_examples

    # Initialize OpenAI client (needed for synthesis and/or L2 generation)
    openai_client = None
    if not args.skip_synthesis or not args.skip_l2_generation:
        from src.api.openai_client import OpenAIClient
        openai_client = OpenAIClient()

    # L2 cache for response-grounded generation
    l2_cache: dict[tuple[str, int], str] = {}
    l2_cache_path = cfg.paths.sft_combined.parent / "l2_cache.jsonl"
    if args.resume and l2_cache_path.exists():
        import json as _json
        with open(l2_cache_path, encoding="utf-8") as f:
            for line in f:
                entry = _json.loads(line)
                l2_cache[(entry["sft_source"], entry["sft_index"])] = entry["l2_text"]
        logger.info("Loaded %d cached L2 entries", len(l2_cache))

    l2_client = openai_client if not args.skip_l2_generation else None

    def _flush_l2_cache() -> None:
        """Write current L2 cache state to disk for resume support."""
        if not l2_cache:
            return
        import json as _json
        l2_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(l2_cache_path, "w", encoding="utf-8") as f:
            for (src, idx), val in l2_cache.items():
                f.write(_json.dumps({"sft_source": src, "sft_index": idx, "l2_text": val}) + "\n")
        logger.info("Flushed %d L2 cache entries to %s", len(l2_cache), l2_cache_path)

    # Load libraries
    logger.info("Loading libraries...")
    l0_rules = load_l0_rules(cfg.paths.l0_rules)
    l1_library = load_l1_library(cfg.paths.l1_library)
    l4_entries = load_l4_library(cfg.paths.l4_library)
    l4_lookup = _build_l4_lookup(l4_entries)

    alpaca_train = load_from_disk(str(cfg.paths.alpaca_train))
    dolly_train = load_from_disk(str(cfg.paths.dolly_train))

    # Tag each row with _sft_source and _sft_index, then combine
    tagged_rows: list[dict] = []
    for i, row in enumerate(alpaca_train):
        d = dict(row)
        d["_sft_source"] = "alpaca"
        d["_sft_index"] = i
        tagged_rows.append(d)
    for i, row in enumerate(dolly_train):
        d = dict(row)
        d["_sft_source"] = "dolly"
        d["_sft_index"] = i
        tagged_rows.append(d)

    # Cross-split exclusion: remove rows used by the other split or by DPO
    other_split = "val" if split == "train" else "train"
    other_paths = cfg.paths.for_split(other_split)

    excluded_keys: set[tuple[str, int]] = set()

    # Other split's SFT output
    if other_paths.sft_combined.exists():
        excluded_keys |= collect_used_base_keys(other_paths.sft_combined)

    # Other split's DPO phase outputs
    for dpo_path in [other_paths.dpo_phase1, other_paths.dpo_phase2,
                     other_paths.dpo_phase3_original, other_paths.dpo_phase3]:
        excluded_keys |= collect_used_base_keys(dpo_path)

    # Same split's DPO outputs (SFT should not reuse DPO instances)
    for dpo_path in [cfg.paths.dpo_phase1, cfg.paths.dpo_phase2,
                     cfg.paths.dpo_phase3_original, cfg.paths.dpo_phase3]:
        excluded_keys |= collect_used_base_keys(dpo_path)

    if excluded_keys:
        before = len(tagged_rows)
        tagged_rows = [
            r for r in tagged_rows
            if (r["_sft_source"], r["_sft_index"]) not in excluded_keys
        ]
        logger.info(
            "Cross-split exclusion: removed %d rows (%d remaining)",
            before - len(tagged_rows), len(tagged_rows),
        )

    # Split tagged rows by L4 availability
    rng = random.Random(seed)
    rows_with_l4 = [
        r for r in tagged_rows
        if (r["_sft_source"], r["_sft_index"]) in l4_lookup
    ]
    rows_without_l4 = [
        r for r in tagged_rows
        if (r["_sft_source"], r["_sft_index"]) not in l4_lookup
    ]
    rng.shuffle(rows_with_l4)
    rng.shuffle(rows_without_l4)

    needed_l4 = simple_count + synth_count
    if len(rows_with_l4) < needed_l4:
        logger.error(
            "L4 library covers only %d rows, but %d needed (simple=%d + synthesis=%d). "
            "Synthesize more L4 entries with: bin/build_libraries.py l4 --max-synthesis N",
            len(rows_with_l4), needed_l4, simple_count, synth_count,
        )
        sys.exit(1)

    # Simple + synthesis slices MUST have L4 entries
    simple_slice = rows_with_l4[:simple_count]
    synthesis_slice = rows_with_l4[simple_count:needed_l4]
    remaining_l4 = rows_with_l4[needed_l4:]

    # Remaining slices use leftover pool
    remaining_pool = remaining_l4 + rows_without_l4
    rng.shuffle(remaining_pool)

    partial_pool_size = partial_count * 4
    # 10x headroom for summarisation filtering in misaligned builders
    misaligned_pool_size = misaligned_count * 10

    partial_slice = remaining_pool[:partial_pool_size]
    misaligned_slice = remaining_pool[partial_pool_size:partial_pool_size + misaligned_pool_size]

    # Warn if any slice is smaller than requested
    if len(simple_slice) < simple_count:
        logger.warning("Simple slice has %d rows, requested %d", len(simple_slice), simple_count)
    if len(synthesis_slice) < synth_count:
        logger.warning("Synthesis slice has %d rows, requested %d", len(synthesis_slice), synth_count)
    if len(partial_slice) < partial_pool_size:
        logger.warning("Partial slice has %d rows, requested %d", len(partial_slice), partial_pool_size)
    if len(misaligned_slice) < misaligned_pool_size:
        logger.warning("Misaligned slice has %d rows, requested %d", len(misaligned_slice), misaligned_pool_size)

    logger.info(
        "Partitioned %d rows: %d with L4, %d without. "
        "simple=%d, synthesis=%d, partial=%d, misaligned=%d",
        len(tagged_rows), len(rows_with_l4), len(rows_without_l4),
        len(simple_slice), len(synthesis_slice),
        len(partial_slice), len(misaligned_slice),
    )

    all_examples: list[dict] = []

    # Step 1: Simple aligned examples
    logger.info("Building %d simple aligned examples...", simple_count)
    simple_aligned = build_simple_aligned(
        base_rows=simple_slice,
        l0_rules=l0_rules,
        l1_library=l1_library,
        l4_lookup=l4_lookup,
        count=simple_count,
        seed=seed,
        openai_client=l2_client,
        l2_cache=l2_cache,
    )
    all_examples.extend(simple_aligned)
    if l2_client is not None:
        import re as _re
        for ex in simple_aligned:
            m = _re.search(r"<\|L2_START\|>(.*?)<\|L2_END\|>", ex.get("text", ""), _re.DOTALL)
            if m and ex.get("sft_source") and ex.get("sft_index") is not None:
                l2_cache[(ex["sft_source"], ex["sft_index"])] = m.group(1)
        _flush_l2_cache()

    # Step 2: Context synthesis aligned examples
    if not args.skip_synthesis:
        cache_path = cfg.paths.sft_synthesis_cache

        # Build skip set from cache when resuming
        skip_indices: set[tuple[str, int]] | None = None
        cached_examples: list[dict] = []
        if args.resume and cache_path.exists():
            cached_examples = load_sft_dataset(cache_path)
            skip_indices = {
                (ex.get("sft_source") or ex.get("_sft_source"),
                 ex.get("sft_index") if ex.get("sft_index") is not None else ex.get("_sft_index"))
                for ex in cached_examples
                if (ex.get("sft_source") or ex.get("_sft_source")) is not None
            }
            logger.info(
                "Resuming: loaded %d cached synthesis examples (%d unique row keys)",
                len(cached_examples), len(skip_indices),
            )

        logger.info("Building %d context synthesis aligned examples...", synth_count)
        synthesis_aligned = build_context_synthesis_aligned(
            base_rows=synthesis_slice,
            l0_rules=l0_rules,
            client=openai_client,
            count=synth_count,
            seed=seed + 10000,
            flush_path=cache_path,
            flush_every=args.flush_every,
            skip_indices=skip_indices,
            l4_lookup=l4_lookup,
        )

        # Combine cached + newly synthesized
        if cached_examples:
            all_examples.extend(cached_examples)
        all_examples.extend(synthesis_aligned)

    # Step 3: Partial examples
    logger.info("Building %d partial examples...", total_partial)
    partial = build_partial_examples(
        base_rows=partial_slice,
        l0_rules=l0_rules,
        l1_library=l1_library,
        l4_lookup=l4_lookup,
        per_config_count=partial_count,
        seed=seed + 20000,
        openai_client=l2_client,
        l2_cache=l2_cache,
    )
    all_examples.extend(partial)
    if l2_client is not None:
        import re as _re
        for ex in partial:
            m = _re.search(r"<\|L2_START\|>(.*?)<\|L2_END\|>", ex.get("text", ""), _re.DOTALL)
            if m and ex.get("sft_source") and ex.get("sft_index") is not None:
                l2_cache[(ex["sft_source"], ex["sft_index"])] = m.group(1)
        _flush_l2_cache()

    # Step 4: Misaligned examples
    logger.info("Building %d misaligned examples...", total_misaligned)
    misaligned = build_misaligned_examples(
        l0_rules=l0_rules,
        l1_library=l1_library,
        base_rows=misaligned_slice,
        per_type_count=misaligned_count,
        seed=seed + 30000,
        openai_client=l2_client,
        l2_cache=l2_cache,
    )
    all_examples.extend(misaligned)
    if l2_client is not None:
        import re as _re
        for ex in misaligned:
            m = _re.search(r"<\|L2_START\|>(.*?)<\|L2_END\|>", ex.get("text", ""), _re.DOTALL)
            if m and ex.get("sft_source") and ex.get("sft_index") is not None:
                l2_cache[(ex["sft_source"], ex["sft_index"])] = m.group(1)
        _flush_l2_cache()

    # Save
    logger.info("Saving %d SFT examples to %s...", len(all_examples), cfg.paths.sft_combined)
    save_sft_dataset(all_examples, cfg.paths.sft_combined)

    # Stats
    stats = compute_sft_stats(all_examples)
    logger.info("=== SFT Dataset Stats ===")
    logger.info("  Total:       %d", stats["total"])
    logger.info("  Aligned:     %d", stats["aligned"])
    logger.info("  Conflicting: %d", stats["conflicting"])
    if stats["conflict_types"]:
        logger.info("  Conflict types:")
        for ct, count in sorted(stats["conflict_types"].items()):
            logger.info("    %s: %d", ct, count)
    if stats["level_configurations"]:
        logger.info("  Level configurations:")
        for lc, count in sorted(stats["level_configurations"].items()):
            logger.info("    %s: %d", lc, count)
    if stats.get("sft_categories"):
        logger.info("  SFT categories:")
        for cat, count in sorted(stats["sft_categories"].items(), key=lambda x: str(x[0])):
            logger.info("    %s: %d", cat, count)
    if stats.get("sft_sources"):
        logger.info("  Sources:")
        for src, count in sorted(stats["sft_sources"].items(), key=lambda x: str(x[0])):
            logger.info("    %s: %d", src, count)
    if stats.get("l4_generations"):
        logger.info("  L4 generations:")
        for gen, count in sorted(stats["l4_generations"].items(), key=lambda x: str(x[0])):
            logger.info("    %s: %d", gen, count)

    # Post-build validation
    five_level_aligned = sum(
        1 for ex in all_examples
        if ex.get("levels_present") == [0, 1, 2, 3, 4] and not ex.get("is_conflict")
    )
    null_category = sum(1 for ex in all_examples if ex.get("sft_category") is None)

    if null_category > 0:
        logger.error("VALIDATION FAILED: %d examples have sft_category=None", null_category)
        sys.exit(1)

    if five_level_aligned < simple_count:
        logger.warning(
            "Only %d aligned 5-level examples (expected >= %d). "
            "Check L4 library coverage.",
            five_level_aligned, simple_count,
        )

    logger.info("=== SFT dataset build complete (%s) ===", split)


def main() -> None:
    """CLI entry point for building the SFT dataset."""
    parser = argparse.ArgumentParser(
        description="Build the SFT dataset for instruction hierarchy training.",
    )
    parser.add_argument(
        "--config", type=Path, default=_project_root / "configs" / "base_linear.yaml",
        help="Path to YAML config file (default: configs/base_linear.yaml)."
    )
    parser.add_argument(
        "--override", nargs="*", default=[], help="Config overrides as section.key=value."
    )
    parser.add_argument(
        "--aligned-count", type=int, default=7000,
        help="Number of aligned examples (default: 7000).",
    )
    parser.add_argument(
        "--synthesis-count", type=int, default=2000,
        help="Number of context synthesis within aligned (default: 2000).",
    )
    parser.add_argument(
        "--partial-count", type=int, default=500,
        help="Per-config count for partial examples (default: 500).",
    )
    parser.add_argument(
        "--misaligned-count", type=int, default=250,
        help="Per-type count for misaligned examples (default: 250).",
    )
    parser.add_argument(
        "--skip-synthesis", action="store_true",
        help="Skip GPT-4o context synthesis.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cache files, skipping already-processed rows (synthesis + L2).",
    )
    parser.add_argument(
        "--flush-every", type=int, default=100,
        help="Flush synthesis cache every N API calls (default: 100).",
    )
    parser.add_argument(
        "--skip-l2-generation", action="store_true",
        help="Skip GPT-4o-mini L2 generation; use random template L2 instead.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
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

    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        logger.info("========== Building SFT split: %s ==========", split)
        _build_split(args, split)


if __name__ == "__main__":
    main()
