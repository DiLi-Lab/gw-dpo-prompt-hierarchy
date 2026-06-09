#!/usr/bin/env python3
"""Build content libraries for the 5-level instruction hierarchy.

Usage:
    python bin/build_libraries.py l0 [--rules-file PATH]
    python bin/build_libraries.py l1 [--skip-dedup] [--domains DOMAIN,...] [--batches-per-domain N]
    python bin/build_libraries.py l2
    python bin/build_libraries.py l3 [--validate]
    python bin/build_libraries.py l4 [--skip-synthesis] [--max-synthesis N] [--resume]
    python bin/build_libraries.py injection [--templates-file PATH]
    python bin/build_libraries.py all
"""

import argparse
import logging
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


def cmd_l0(args: argparse.Namespace) -> None:
    """Validate L0 rules JSON file, or expand seed rules via LLM."""
    from src.data.libraries.l0_rules import load_l0_rules

    cfg = load_config(config_path=args.config, overrides=args.override)

    if getattr(args, "expand", False):
        from src.api.anthropic_client import AnthropicClient
        from src.data.libraries.l0_expansion import expand_l0_rules

        client = AnthropicClient()
        seed_path = Path(args.seed_file) if args.seed_file else cfg.paths.l0_seed_rules
        output_path = cfg.paths.l0_rules_expanded

        expanded = expand_l0_rules(client, seed_path, output_path)
        logger.info("L0 expansion complete: %d rules saved to %s", len(expanded), output_path)
        return

    if getattr(args, "validate", False):
        rules_file = Path(args.rules_file) if args.rules_file else cfg.paths.l0_rules

        rules = load_l0_rules(rules_file)
        categories: dict[str, list] = {}
        for r in rules:
            categories.setdefault(r.category, []).append(r)

        logger.info("L0 rules validated: %d rules across %d categories", len(rules), len(categories))
        for cat, cat_rules in sorted(categories.items()):
            logger.info("  %s: %d rules", cat, len(cat_rules))

        si_count = len(categories.get("system_integrity", []))
        if si_count == 0:
            logger.error("No system_integrity rules found! At least one is required.")
            sys.exit(1)
        return

    parser_error_msg = "l0 requires either --validate or --expand"
    logger.error(parser_error_msg)
    sys.exit(1)


def cmd_l1(args: argparse.Namespace) -> None:
    """Generate or validate L1 developer system prompts."""
    cfg = load_config(config_path=args.config, overrides=args.override)

    if getattr(args, "validate", False):
        from src.data.libraries.l1_prompts import validate_l1_library

        lib_path = Path(args.library_file) if args.library_file else cfg.paths.l1_library
        try:
            validate_l1_library(lib_path)
        except FileNotFoundError:
            logger.error("L1 library file not found: %s", lib_path)
            sys.exit(1)
        return

    from src.api.anthropic_client import AnthropicClient
    from src.data.libraries.l1_prompts import TASK_DOMAINS, generate_l1_library

    client = AnthropicClient()

    domains = args.domains.split(",") if args.domains else TASK_DOMAINS
    output_path = cfg.paths.l1_library

    prompts = generate_l1_library(
        client=client,
        output_path=output_path,
        domains=domains,
        batches_per_domain=args.batches_per_domain,
        skip_dedup=args.skip_dedup,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    logger.info("L1 library complete: %d prompts saved to %s", len(prompts), output_path)


def cmd_l2(args: argparse.Namespace) -> None:
    """Verify L2 template generation or validate with statistics."""
    from src.data.libraries.l2_templates import generate_l2, generate_l2_batch

    if getattr(args, "validate", False):
        configs, stats = generate_l2_batch(count=args.count, seed=0)
        logger.info(
            "L2 validation: generated %d configs (%d unique)",
            stats["total"],
            stats["unique_texts"],
        )
        logger.info("Attribute distribution:")
        for attr, count in stats["attribute_counts"].items():
            pct = count / stats["total"] * 100
            logger.info("  %s: %d (%.1f%%)", attr, count, pct)
        return

    samples = [generate_l2(seed=i) for i in range(5)]
    logger.info("L2 template generator working. Sample outputs:")
    for i, s in enumerate(samples):
        logger.info("  [%d] %s", i, s)


def cmd_l3(args: argparse.Namespace) -> None:
    """Filter and validate L3 user messages from base datasets."""
    cfg = load_config(config_path=args.config, overrides=args.override)

    if not cfg.paths.alpaca_train.exists() or not cfg.paths.dolly_train.exists():
        logger.error("Base dataset splits not found. Run bin/download_base_datasets.py first.")
        sys.exit(1)

    from src.data.libraries.l3_user_messages import load_l3_pool, validate_l3_pool

    pool = load_l3_pool(cfg.paths.alpaca_train, cfg.paths.dolly_train)

    if getattr(args, "validate", False):
        stats = validate_l3_pool(pool)
        logger.info("L3 pool validated: %d messages", stats["total"])
        logger.info("Source distribution:")
        for source, count in stats["source_counts"].items():
            pct = count / stats["total"] * 100
            logger.info("  %s: %d (%.1f%%)", source, count, pct)
        wc = stats["word_count_stats"]
        logger.info(
            "Word counts: min=%d, max=%d, mean=%.1f, median=%d",
            wc["min"], wc["max"], wc["mean"], wc["median"],
        )
        return

    from src.data.libraries.l3_user_messages import sample_l3_message

    logger.info("L3 pool loaded: %d messages. Sample outputs:", len(pool))
    for i in range(5):
        msg = sample_l3_message(pool, seed=i)
        logger.info("  [%d] (%s) %s", i, msg.source, msg.text[:100])


def cmd_l4(args: argparse.Namespace) -> None:
    """Build L4 tool outputs: wrap existing + synthesise missing."""
    from datasets import load_from_disk

    from src.data.libraries.l4_tool_outputs import (
        build_l4_wrapped,
        load_l4_library,
        save_l4_library,
        synthesize_l4_outputs,
        validate_l4_library,
    )

    cfg = load_config(config_path=args.config, overrides=args.override)

    if not cfg.paths.alpaca_train.exists() or not cfg.paths.dolly_train.exists():
        logger.error("Base dataset splits not found. Run bin/download_base_datasets.py first.")
        sys.exit(1)

    alpaca_train = load_from_disk(str(cfg.paths.alpaca_train))
    dolly_train = load_from_disk(str(cfg.paths.dolly_train))

    # Build skip set from existing library when resuming
    skip_indices: set[tuple[str, int]] | None = None
    existing_synthesized: list = []
    if getattr(args, "resume", False) and cfg.paths.l4_library.exists():
        existing = load_l4_library(cfg.paths.l4_library)
        existing_synthesized = [e for e in existing if e.generation == "synthesized"]
        skip_indices = {(e.source, e.index) for e in existing_synthesized}
        logger.info("Resuming: loaded %d existing synthesised entries to skip", len(skip_indices))

    # Source A: wrap non-empty data fields
    all_entries = build_l4_wrapped(alpaca_train, source="alpaca", data_field="input")
    all_entries += build_l4_wrapped(dolly_train, source="dolly", data_field="context")
    all_entries += existing_synthesized
    save_l4_library(all_entries, cfg.paths.l4_library)

    # Source B: synthesise for empty data fields
    if args.skip_synthesis:
        logger.info("Skipping L4 synthesis (--skip-synthesis)")
    else:
        from src.api.openai_client import OpenAIClient

        client = OpenAIClient()

        # Split max_synthesis budget across both datasets proportionally
        alpaca_empty = sum(1 for i in range(len(alpaca_train)) if not alpaca_train[i]["input"].strip())
        dolly_empty = sum(1 for i in range(len(dolly_train)) if not dolly_train[i]["context"].strip())
        total_empty = alpaca_empty + dolly_empty

        if args.max_synthesis is not None and total_empty > 0:
            alpaca_max = round(args.max_synthesis * alpaca_empty / total_empty)
            dolly_max = args.max_synthesis - alpaca_max
        else:
            alpaca_max = None
            dolly_max = None

        out = cfg.paths.l4_library
        all_entries += synthesize_l4_outputs(
            dataset=alpaca_train, client=client,
            source="alpaca", data_field="input", max_examples=alpaca_max,
            flush_path=out, prior_entries=all_entries,
            skip_indices=skip_indices,
        )
        save_l4_library(all_entries, out)

        all_entries += synthesize_l4_outputs(
            dataset=dolly_train, client=client,
            source="dolly", data_field="context", max_examples=dolly_max,
            flush_path=out, prior_entries=all_entries,
            skip_indices=skip_indices,
        )

    save_l4_library(all_entries, cfg.paths.l4_library)

    stats = validate_l4_library(all_entries)
    logger.info("L4 library complete: %d entries", stats["total"])
    logger.info("Source distribution:")
    for source, count in stats["source_counts"].items():
        pct = count / stats["total"] * 100
        logger.info("  %s: %d (%.1f%%)", source, count, pct)
    logger.info("Generation distribution:")
    for gen, count in stats["generation_counts"].items():
        pct = count / stats["total"] * 100
        logger.info("  %s: %d (%.1f%%)", gen, count, pct)


def cmd_injection(args: argparse.Namespace) -> None:
    """Validate injection templates JSON file."""
    from src.data.libraries.injection_templates import load_injection_templates

    cfg = load_config(config_path=args.config, overrides=args.override)
    templates_file = Path(args.templates_file) if args.templates_file else cfg.paths.injection_templates

    templates = load_injection_templates(templates_file)
    total = len(templates.prefixes) + len(templates.system_overrides) + len(templates.position_injections)
    logger.info("Injection templates validated: %d total", total)
    logger.info("  Prefixes: %d", len(templates.prefixes))
    logger.info("  System overrides: %d", len(templates.system_overrides))
    logger.info("  Position injections: %d", len(templates.position_injections))


def cmd_all(args: argparse.Namespace) -> None:
    """Run all library builders in sequence."""
    logger.info("=== Building all libraries ===")

    logger.info("--- L0: Validating rules ---")
    args.validate = True
    cmd_l0(args)

    logger.info("--- L1: Generating prompts ---")
    cmd_l1(args)

    logger.info("--- L2: Verifying templates ---")
    cmd_l2(args)

    logger.info("--- L3: Filtering user messages ---")
    cmd_l3(args)

    logger.info("--- L4: Building tool outputs ---")
    cmd_l4(args)

    logger.info("--- Injection: Validating templates ---")
    cmd_injection(args)

    logger.info("=== All libraries complete ===")


def main() -> None:
    """CLI entry point for building content libraries."""
    parser = argparse.ArgumentParser(
        description="Build content libraries for the 5-level instruction hierarchy.",
    )
    parser.add_argument("--config", type=Path, default=_project_root / "configs" / "base_linear.yaml",
                        help="Path to YAML config file (default: configs/base_linear.yaml).")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides as section.key=value.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # L0 subcommand
    p_l0 = subparsers.add_parser("l0", help="Validate or expand L0 rules.")
    p_l0.add_argument("--validate", action="store_true", help="Validate existing L0 rules (no generation).")
    p_l0.add_argument("--rules-file", default=None, help="Path to l0_rules.json.")
    p_l0.add_argument("--expand", action="store_true", help="Run LLM expansion on seed rules.")
    p_l0.add_argument("--seed-file", default=None, help="Path to L0_seed_rules.json.")

    # L1 subcommand
    p_l1 = subparsers.add_parser("l1", help="Generate L1 developer system prompts.")
    p_l1.add_argument("--skip-dedup", action="store_true", help="Skip deduplication (for inspection).")
    p_l1.add_argument("--domains", default=None, help="Comma-separated list of domains to generate.")
    p_l1.add_argument("--batches-per-domain", type=int, default=10, help="API calls per domain (default: 10).")
    p_l1.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9).")
    p_l1.add_argument("--max-tokens", type=int, default=4000, help="Max tokens per API response (default: 4000).")
    p_l1.add_argument("--validate", action="store_true", help="Validate existing L1 library (no generation).")
    p_l1.add_argument("--library-file", default=None, help="Path to existing l1_library.json (for --validate).")

    # L2 subcommand
    p_l2 = subparsers.add_parser("l2", help="Verify or validate L2 template generation.")
    p_l2.add_argument("--validate", action="store_true", help="Generate batch and report attribute distribution stats.")
    p_l2.add_argument("--count", type=int, default=500, help="Number of configs to generate for --validate (default: 500).")

    # L3 subcommand
    p_l3 = subparsers.add_parser("l3", help="Filter and validate L3 user messages from base datasets.")
    p_l3.add_argument("--validate", action="store_true", help="Load full pool and report statistics.")

    # L4 subcommand
    p_l4 = subparsers.add_parser("l4", help="Build L4 tool outputs.")
    p_l4.add_argument("--skip-synthesis", action="store_true", help="Skip GPT-4o-mini synthesis.")
    p_l4.add_argument("--max-synthesis", type=int, default=None, help="Max examples to synthesise (default: all).")
    p_l4.add_argument("--resume", action="store_true", help="Resume from existing l4_library.json, skipping already-synthesised entries.")

    # Injection subcommand
    p_inj = subparsers.add_parser("injection", help="Validate injection templates JSON file.")
    p_inj.add_argument("--templates-file", default=None, help="Path to injection_templates.json.")

    # All subcommand
    p_all = subparsers.add_parser("all", help="Run all library builders.")
    p_all.add_argument("--rules-file", default=None, help="Path to l0_rules.json.")
    p_all.add_argument("--expand", action="store_true", help="Run LLM expansion on seed rules.")
    p_all.add_argument("--seed-file", default=None, help="Path to L0_seed_rules.json.")
    p_all.add_argument("--templates-file", default=None, help="Path to injection_templates.json.")
    p_all.add_argument("--library-file", default=None)
    p_all.add_argument("--skip-dedup", action="store_true")
    p_all.add_argument("--domains", default=None)
    p_all.add_argument("--batches-per-domain", type=int, default=10)
    p_all.add_argument("--temperature", type=float, default=0.9)
    p_all.add_argument("--max-tokens", type=int, default=4000)
    p_all.add_argument("--skip-synthesis", action="store_true")
    p_all.add_argument("--max-synthesis", type=int, default=None)
    p_all.add_argument("--resume", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: %s" % " ".join(unknown))

    commands = {
        "l0": cmd_l0,
        "l1": cmd_l1,
        "l2": cmd_l2,
        "l3": cmd_l3,
        "l4": cmd_l4,
        "injection": cmd_injection,
        "all": cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
