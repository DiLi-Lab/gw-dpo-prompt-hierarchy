"""DPO dataset build utilities: cache, stats, SFT exclusion, partitioning, phase orchestration.

Provides functions for saving/loading DPO generation caches, computing summary
statistics over DPO examples, excluding SFT-used rows, partitioning the base
dataset pool across pair configurations, and orchestrating the three build phases.
"""

import json
import logging
import random
from pathlib import Path

from src.data.dpo.calibration import build_calibration_examples
from src.data.dpo.cascading import build_cascading_examples
from src.data.dpo.l0_conflict_builder import build_l0_vs_l1_pair, build_l0_vs_l2_pair
from src.data.dpo.pair_builder import build_conflict_pair
from src.data.dpo.pair_config import ALL_PAIR_CONFIGS, PairConfig, get_config_by_name
from src.data.dpo.zero_cost_pairs import build_l1_vs_l3_pairs

logger = logging.getLogger(__name__)


def compute_dpo_stats(examples: list[dict]) -> dict:
    """Compute summary statistics over a collection of DPO examples.

    Args:
        examples: List of DPO example dicts, each expected to have keys
            conflict_type, category, yw_source, yl_source, yl_fallback_used.

    Returns:
        Dict with keys: total, conflict_types, categories, yw_sources,
        yl_sources, yl_fallbacks -- each mapping to a counter-like dict.
    """
    conflict_types: dict[str, int] = {}
    categories: dict[str, int] = {}
    yw_sources: dict[str, int] = {}
    yl_sources: dict[str, int] = {}
    yl_fallbacks: dict[str | None, int] = {}

    for ex in examples:
        ct = ex.get("conflict_type")
        if ct is not None:
            conflict_types[ct] = conflict_types.get(ct, 0) + 1

        cat = ex.get("category")
        if cat is not None:
            categories[cat] = categories.get(cat, 0) + 1

        yw = ex.get("yw_source")
        if yw is not None:
            yw_sources[yw] = yw_sources.get(yw, 0) + 1

        yl = ex.get("yl_source")
        if yl is not None:
            yl_sources[yl] = yl_sources.get(yl, 0) + 1

        fallback = ex.get("yl_fallback_used")
        yl_fallbacks[fallback] = yl_fallbacks.get(fallback, 0) + 1

    return {
        "total": len(examples),
        "conflict_types": conflict_types,
        "categories": categories,
        "yw_sources": yw_sources,
        "yl_sources": yl_sources,
        "yl_fallbacks": yl_fallbacks,
    }


def save_dpo_cache(cache: dict[tuple[str, str, int], str], path: Path) -> None:
    """Write a DPO generation cache to a JSONL file.

    Each line is a JSON object with "key" (3-element list) and "value" (string).

    Args:
        cache: Dict keyed by (pair_type, source, index) -> cached string.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (pair_type, source, index), value in cache.items():
            line = json.dumps(
                {"key": [pair_type, source, index], "value": value},
                ensure_ascii=False,
            )
            f.write(line + "\n")
    logger.info("Saved %d cache entries to %s", len(cache), path)


def load_dpo_cache(path: Path) -> dict[tuple[str, str, int], str]:
    """Load a DPO generation cache from a JSONL file.

    Args:
        path: Path to the JSONL cache file.

    Returns:
        Dict keyed by (pair_type, source, index) -> cached string.
        Returns empty dict if the file does not exist.
    """
    if not path.exists():
        logger.info("Cache file not found, returning empty cache: %s", path)
        return {}

    cache: dict[tuple[str, str, int], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            key_list = obj["key"]
            key = (key_list[0], key_list[1], key_list[2])
            cache[key] = obj["value"]

    logger.info("Loaded %d cache entries from %s", len(cache), path)
    return cache


def exclude_sft_rows(all_rows: list[dict], sft_path: Path) -> list[dict]:
    """Filter out rows already used in the SFT dataset.

    Loads the SFT combined file and builds a set of (sft_source, sft_index)
    keys. Rows whose (_dpo_source, _dpo_index) match are excluded.

    Args:
        all_rows: Pool of candidate rows with _dpo_source and _dpo_index keys.
        sft_path: Path to sft_combined.jsonl.

    Returns:
        Filtered list of rows not present in the SFT dataset.
    """
    sft_keys: set[tuple[str, int]] = set()
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            sft_keys.add((obj["sft_source"], obj["sft_index"]))

    remaining = [
        row for row in all_rows
        if (row["_dpo_source"], row["_dpo_index"]) not in sft_keys
    ]
    logger.info(
        "Excluded %d SFT rows, %d remaining from %d total",
        len(all_rows) - len(remaining),
        len(remaining),
        len(all_rows),
    )
    return remaining


def exclude_prior_phase_rows(
    pool: list[dict],
    phase_paths: list[Path],
) -> list[dict]:
    """Filter out pool rows already used as base instances in prior phase outputs.

    Scans each existing phase output file for yw_base_dataset/yw_base_index and
    yl_base_dataset/yl_base_index fields, then removes matching pool rows.

    Args:
        pool: Pool of candidate rows with _dpo_source and _dpo_index keys.
        phase_paths: Paths to prior phase JSONL output files to scan.

    Returns:
        Filtered list of rows not used in any prior phase output.
    """
    used_keys: set[tuple[str, int]] = set()
    for path in phase_paths:
        if not path.exists():
            logger.info(
                "Prior phase file %s does not exist, skipping exclusion.",
                path,
            )
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                for prefix in ("yw_base", "yl_base"):
                    ds = obj.get(f"{prefix}_dataset")
                    idx = obj.get(f"{prefix}_index")
                    if ds is not None and idx is not None:
                        used_keys.add((ds, idx))

    remaining = [
        row for row in pool
        if (row["_dpo_source"], row["_dpo_index"]) not in used_keys
    ]
    logger.info(
        "Excluded %d prior-phase rows, %d remaining from %d total",
        len(pool) - len(remaining),
        len(remaining),
        len(pool),
    )
    return remaining


def collect_used_base_keys(path: Path) -> set[tuple[str, int]]:
    """Collect all base dataset instance keys from a JSONL output file.

    Scans for multiple key formats:
    - SFT: (sft_source, sft_index)
    - DPO: (yw_base_dataset, yw_base_index) and (yl_base_dataset, yl_base_index)

    Args:
        path: Path to a JSONL file. Returns empty set if file doesn't exist.

    Returns:
        Set of (dataset_name, row_index) tuples.
    """
    if not path.exists():
        return set()

    keys: set[tuple[str, int]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            # SFT format
            src = obj.get("sft_source")
            idx = obj.get("sft_index")
            if src is not None and idx is not None:
                keys.add((src, idx))
            # DPO format
            for prefix in ("yw_base", "yl_base"):
                ds = obj.get(f"{prefix}_dataset")
                ix = obj.get(f"{prefix}_index")
                if ds is not None and ix is not None:
                    keys.add((ds, ix))
    return keys


COMPATIBLE_L4_RATIO: float = 0.7
"""Target ratio of L4-covered rows for non-L4-conflict configs."""


def _requires_l4(cfg: PairConfig) -> bool:
    """Return True if this config's conflict involves L4."""
    return cfg.injection_target_level == 4


def partition_dpo_pool(
    rows: list[dict],
    configs: list[PairConfig],
    l4_lookup: dict[tuple[str, int], dict],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Partition pool rows into disjoint slices for each pair configuration.

    Shuffles rows deterministically, then allocates slices per config.
    L4-conflict configs only get rows with L4 coverage. Non-L4-conflict
    configs preferentially get L4-covered rows up to COMPATIBLE_L4_RATIO,
    then fill the remainder from any available rows.

    Args:
        rows: Pool of candidate rows with instruction, _dpo_source, _dpo_index.
        configs: List of PairConfig definitions.
        l4_lookup: Dict mapping (source, index) to L4 data for filtering.
        seed: Random seed for deterministic shuffling.

    Returns:
        Dict mapping config name to list of allocated rows.
    """
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)

    used_indices: set[int] = set()
    slices: dict[str, list[dict]] = {}

    def _row_has_l4(row: dict) -> bool:
        return (row["_dpo_source"], row["_dpo_index"]) in l4_lookup

    # Pass 1: allocate L4-required configs first (they MUST have L4)
    for cfg in configs:
        if cfg.scenario_driven:
            slices[cfg.name] = []
            continue
        if not _requires_l4(cfg):
            continue

        target_size = int(cfg.target_count * 1.2)
        allocated: list[dict] = []

        for i, row in enumerate(pool):
            if i in used_indices:
                continue
            if len(allocated) >= target_size:
                break
            if not _row_has_l4(row):
                continue
            allocated.append(row)
            used_indices.add(i)

        if len(allocated) < target_size:
            logger.warning(
                "Pool exhausted for %s: allocated %d / %d requested",
                cfg.name, len(allocated), target_size,
            )
        slices[cfg.name] = allocated

    # Pass 2: allocate non-L4-conflict configs with L4 preference
    for cfg in configs:
        if cfg.scenario_driven:
            slices[cfg.name] = []
            continue
        if _requires_l4(cfg):
            continue

        if cfg.name == "L1_vs_L3":
            target_size = int(cfg.target_count * 2 * 1.2)
        else:
            target_size = int(cfg.target_count * 1.2)

        l4_target = int(target_size * COMPATIBLE_L4_RATIO)

        # Sub-pass A: fill L4-covered rows up to l4_target
        allocated: list[dict] = []
        for i, row in enumerate(pool):
            if i in used_indices:
                continue
            if len(allocated) >= l4_target:
                break
            if not _row_has_l4(row):
                continue
            allocated.append(row)
            used_indices.add(i)

        # Sub-pass B: fill remaining from any rows
        for i, row in enumerate(pool):
            if i in used_indices:
                continue
            if len(allocated) >= target_size:
                break
            allocated.append(row)
            used_indices.add(i)

        if len(allocated) < target_size:
            logger.warning(
                "Pool exhausted for %s: allocated %d / %d requested",
                cfg.name, len(allocated), target_size,
            )
        slices[cfg.name] = allocated

    return slices


# ---------------------------------------------------------------------------
# Phase-level configs (derived from PairConfig.phase)
# ---------------------------------------------------------------------------

_PHASE2_CONFIG_NAMES: list[str] = [
    c.name for c in ALL_PAIR_CONFIGS
    if c.phase == 2 and c.category == "pairwise"
]

_PHASE3_CONFIG_NAMES: list[str] = [
    c.name for c in ALL_PAIR_CONFIGS
    if c.phase == 3 and c.category == "pairwise"
]


def _save_jsonl(examples: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    results: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                results.append(json.loads(stripped))
    return results


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------


def run_phase1(
    pool_slice: list[dict],
    l0_rules: list,
    l1_library: list[dict],
    injection_templates: object,
    output_path: Path,
    openai_client: object | None = None,
    anthropic_client: object | None = None,
    l2_cache: dict | None = None,
    l4_lookup: dict[tuple[str, int], dict] | None = None,
    count: int = 1500,
    seed: int = 42,
) -> list[dict]:
    """Run Phase 1: build L1-vs-L3 pairs.

    Pairs rows pairwise — (rows[0], rows[1]), (rows[2], rows[3]), etc. —
    then delegates to :func:`build_l1_vs_l3_pairs`.

    Args:
        pool_slice: Rows allocated for L1_vs_L3 (need 2 per pair).
        l0_rules: Full list of L0 rules.
        l1_library: List of L1 prompt dicts.
        injection_templates: InjectionTemplate with .prefixes.
        output_path: Where to save phase-1 JSONL output.
        openai_client: Optional OpenAI client for response-grounded L2.
        anthropic_client: Optional Anthropic client for context distillation.
        l2_cache: Optional cache dict for L2 strings.
        count: Maximum number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of DPO example dicts produced by Phase 1.
    """
    row_pairs: list[tuple[dict, dict]] = []
    for i in range(0, len(pool_slice) - 1, 2):
        row_pairs.append((pool_slice[i], pool_slice[i + 1]))

    results = build_l1_vs_l3_pairs(
        row_pairs=row_pairs,
        l0_rules=l0_rules,
        l1_library=l1_library,
        injection_templates=injection_templates,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        l2_cache=l2_cache,
        l4_lookup=l4_lookup,
        count=count,
        seed=seed,
    )

    _save_jsonl(results, output_path)
    logger.info("Phase 1 complete: %d L1_vs_L3 examples saved to %s", len(results), output_path)
    return results


def run_phase2(
    pool_slices: dict[str, list[dict]],
    l0_rules: list,
    l1_library: list[dict],
    l4_lookup: dict,
    injection_templates: object,
    openai_client: object,
    output_path: Path,
    caches: dict | None = None,
    seed: int = 42,
    count_override: int | None = None,
    l0_adversarial_instructions: list[dict] | None = None,
    anthropic_client: object | None = None,
) -> list[dict]:
    """Run Phase 2: pairwise conflict pairs + calibration examples.

    Iterates over the Phase-2 pairwise configs (L0_vs_L3, L1_vs_L2,
    L1_vs_L4, L2_vs_L3, L2_vs_L4, L3_vs_L4), builds conflict pairs via
    :func:`build_conflict_pair`, then appends calibration examples.

    Args:
        pool_slices: Mapping of config name to allocated rows.
        l0_rules: Full list of L0 rules.
        l1_library: List of L1 prompt dicts.
        l4_lookup: Mapping (source, index) -> L4 data.
        injection_templates: InjectionTemplate object.
        openai_client: OpenAI API client.
        output_path: Where to save phase-2 JSONL output.
        caches: Optional dict with keys "l2_cache", "yw_cache", "yl_cache".
        seed: Random seed.
        count_override: If set, use this as target count for calibration
            instead of the config default.
        l0_adversarial_instructions: Optional list of adversarial instruction
            dicts to use as L0_vs_L3 injection payloads.
        anthropic_client: Optional Anthropic client for L1_vs_L4 pairs.

    Returns:
        All Phase-2 DPO examples (pairwise + calibration).
    """
    if caches is None:
        caches = {}

    all_results: list[dict] = []

    for config_name in _PHASE2_CONFIG_NAMES:
        config = get_config_by_name(config_name)
        rows = pool_slices.get(config_name, [])
        phase_results: list[dict] = []

        for i, row in enumerate(rows):
            example = build_conflict_pair(
                config=config,
                base_row=row,
                l0_rules=l0_rules,
                l1_library=l1_library,
                l4_lookup=l4_lookup,
                injection_templates=injection_templates,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                l2_cache=caches.get("l2_cache"),
                yw_cache=caches.get("yw_cache"),
                yl_cache=caches.get("yl_cache"),
                l0_adversarial_instructions=l0_adversarial_instructions,
                seed=seed + i,
            )
            if example is not None:
                phase_results.append(example)

        logger.info(
            "Phase 2 — %s: %d examples from %d rows",
            config_name, len(phase_results), len(rows),
        )
        all_results.extend(phase_results)

    # Calibration examples
    calibration_config = get_config_by_name("calibration")
    calibration_rows = pool_slices.get("calibration", [])
    calibration_count = count_override if count_override is not None else calibration_config.target_count
    calibration_examples = build_calibration_examples(
        base_rows=calibration_rows,
        l0_rules=l0_rules,
        l1_library=l1_library,
        openai_client=openai_client,
        l4_lookup=l4_lookup,
        count=calibration_count,
        seed=seed,
    )
    logger.info("Phase 2 — calibration: %d examples", len(calibration_examples))
    all_results.extend(calibration_examples)

    _save_jsonl(all_results, output_path)
    logger.info("Phase 2 complete: %d total examples saved to %s", len(all_results), output_path)
    return all_results


def run_phase3(
    pool_slices: dict[str, list[dict]],
    l0_rules: list,
    l1_library: list[dict],
    l4_lookup: dict,
    injection_templates: object,
    openai_client: object,
    anthropic_client: object,
    cascading_families: list,
    output_path: Path,
    caches: dict | None = None,
    seed: int = 42,
    l0_conflict_scenarios: list | None = None,
    count_override: int | None = None,
    l4_domain_index: dict[str, list[tuple[str, int]]] | None = None,
    pair_configs: list | None = None,
) -> list[dict]:
    """Run Phase 3: L0-dominant pairwise pairs + cascading examples.

    Iterates over Phase-3 pairwise configs (L0_vs_L1, L0_vs_L2, L0_vs_L4),
    builds conflict pairs, then appends cascading examples.

    Args:
        pool_slices: Mapping of config name to allocated rows.
        l0_rules: Full list of L0 rules.
        l1_library: List of L1 prompt dicts.
        l4_lookup: Mapping (source, index) -> L4 data.
        injection_templates: InjectionTemplate object.
        openai_client: OpenAI API client.
        anthropic_client: Anthropic API client.
        cascading_families: List of CascadingFamily instances.
        output_path: Where to save phase-3 JSONL output.
        caches: Optional dict with keys "l2_cache", "yw_cache", "yl_cache".
        seed: Random seed.
        l0_conflict_scenarios: Optional list of AdversarialScenario instances for
            L0-vs-L1 and L0-vs-L2 scenario-based construction.
        count_override: If set, use this as target count for scenario-driven
            and cascading pairs instead of the config default.
        l4_domain_index: Optional domain -> [(source, index)] index for
            domain-filtered L4 sampling in scenario builders.
        pair_configs: Split-aware pair configs. When provided, scenario-driven
            types use target counts from these configs instead of the global
            ALL_PAIR_CONFIGS defaults. Pass get_pair_configs(split=split).

    Returns:
        All Phase-3 DPO examples (pairwise + cascading).
    """
    if caches is None:
        caches = {}

    all_results: list[dict] = []

    # Build a name->config lookup from split-aware configs if provided
    _config_lookup: dict[str, object] = {}
    if pair_configs:
        _config_lookup = {c.name: c for c in pair_configs}

    def _get_target(name: str) -> int:
        """Return split-aware target count for a config name."""
        if count_override is not None:
            return count_override
        cfg = _config_lookup.get(name) if _config_lookup else None
        if cfg is None:
            cfg = get_config_by_name(name)
        return cfg.target_count

    # L0-vs-L1 and L0-vs-L2 via scenario builders
    if l0_conflict_scenarios:
        offset = 0
        l0_vs_l1_scenarios = [s for s in l0_conflict_scenarios if s.pair_type == "L0_vs_L1"]
        if l0_vs_l1_scenarios:
            l0_vs_l1_target = _get_target("L0_vs_L1")
            per_scenario = max(l0_vs_l1_target // len(l0_vs_l1_scenarios), 1)
            for scenario in l0_vs_l1_scenarios:
                scenario_used_keys: set[tuple[str, int]] = set()
                for i in range(per_scenario):
                    example = build_l0_vs_l1_pair(
                        scenario=scenario, l0_rules=l0_rules, l4_lookup=l4_lookup,
                        openai_client=openai_client, anthropic_client=anthropic_client,
                        seed=seed + offset,
                        l4_domain_index=l4_domain_index,
                        l4_used_keys=scenario_used_keys,
                    )
                    if example is not None:
                        all_results.append(example)
                    offset += 1
            logger.info(
                "Phase 3 — L0_vs_L1: %d examples from %d scenarios",
                sum(1 for r in all_results if r.get("conflict_type") == "L0_vs_L1"),
                len(l0_vs_l1_scenarios),
            )

        l0_vs_l2_scenarios = [s for s in l0_conflict_scenarios if s.pair_type == "L0_vs_L2"]
        if l0_vs_l2_scenarios:
            l0_vs_l2_target = _get_target("L0_vs_L2")
            per_scenario = max(l0_vs_l2_target // len(l0_vs_l2_scenarios), 1)
            for scenario in l0_vs_l2_scenarios:
                scenario_used_keys: set[tuple[str, int]] = set()
                for i in range(per_scenario):
                    example = build_l0_vs_l2_pair(
                        scenario=scenario, l0_rules=l0_rules, l1_library=l1_library,
                        l4_lookup=l4_lookup, openai_client=openai_client,
                        anthropic_client=anthropic_client, seed=seed + offset,
                        l4_domain_index=l4_domain_index,
                        l4_used_keys=scenario_used_keys,
                    )
                    if example is not None:
                        all_results.append(example)
                    offset += 1
            logger.info(
                "Phase 3 — L0_vs_L2: %d examples from %d scenarios",
                sum(1 for r in all_results if r.get("conflict_type") == "L0_vs_L2"),
                len(l0_vs_l2_scenarios),
            )

    # L0-vs-L4 and other non-scenario Phase 3 pairwise configs via generic builder
    for config_name in _PHASE3_CONFIG_NAMES:
        config = _config_lookup.get(config_name) or get_config_by_name(config_name)
        if config.scenario_driven:
            continue  # Already handled above
        rows = pool_slices.get(config_name, [])
        phase_results: list[dict] = []

        for i, row in enumerate(rows):
            example = build_conflict_pair(
                config=config,
                base_row=row,
                l0_rules=l0_rules,
                l1_library=l1_library,
                l4_lookup=l4_lookup,
                injection_templates=injection_templates,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                l2_cache=caches.get("l2_cache"),
                yw_cache=caches.get("yw_cache"),
                yl_cache=caches.get("yl_cache"),
                seed=seed + i,
            )
            if example is not None:
                phase_results.append(example)

        logger.info(
            "Phase 3 — %s: %d examples from %d rows",
            config_name, len(phase_results), len(rows),
        )
        all_results.extend(phase_results)

    # Cascading examples
    cascading_target = _get_target("cascading")
    per_family = max(cascading_target // max(len(cascading_families), 1), 1)
    cascading_examples = build_cascading_examples(
        families=cascading_families,
        anthropic_client=anthropic_client,
        openai_client=openai_client,
        per_family=per_family,
        seed=seed,
    )
    logger.info("Phase 3 — cascading: %d examples", len(cascading_examples))
    all_results.extend(cascading_examples)

    _save_jsonl(all_results, output_path)
    logger.info("Phase 3 complete: %d total examples saved to %s", len(all_results), output_path)
    return all_results


def combine_phases(
    phase1_path: Path,
    phase2_path: Path,
    phase3_path: Path,
    output_path: Path,
) -> list[dict]:
    """Concatenate all phase JSONL outputs into a single file.

    Args:
        phase1_path: Path to Phase 1 JSONL.
        phase2_path: Path to Phase 2 JSONL.
        phase3_path: Path to Phase 3 JSONL.
        output_path: Where to save the combined JSONL.

    Returns:
        Combined list of all DPO examples.
    """
    all_examples: list[dict] = []
    for path in (phase1_path, phase2_path, phase3_path):
        examples = _load_jsonl(path)
        logger.info("Loaded %d examples from %s", len(examples), path)
        all_examples.extend(examples)

    _save_jsonl(all_examples, output_path)
    logger.info(
        "Combined %d examples from 3 phases into %s",
        len(all_examples), output_path,
    )
    return all_examples


PHASE5_FLUSH_INTERVAL: int = 50


def _build_sample_index(sample: list[dict]) -> dict[str, int]:
    """Build a unique index for each sample instance for resume matching.

    Uses a hash of (conflict_type, prompt[:200]) to identify instances
    deterministically across runs with the same seed.
    """
    import hashlib
    index: dict[str, int] = {}
    for i, ex in enumerate(sample):
        key = hashlib.sha256(
            (ex["conflict_type"] + ex["prompt"][:200]).encode()
        ).hexdigest()[:16]
        index[key] = i
    return index


def _load_completed_keys(qc_results_path: Path) -> set[str]:
    """Load keys of already-judged instances from a prior partial run."""
    import hashlib
    completed: set[str] = set()
    if not qc_results_path.exists():
        return completed
    with open(qc_results_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            key = entry.get("_sample_key")
            if key:
                completed.add(key)
    logger.info("Resume: loaded %d completed results from %s", len(completed), qc_results_path)
    return completed


def run_phase5(
    combined_path: Path,
    openai_client: object,
    google_client: object,
    qc_results_path: Path,
    flagged_path: Path,
    fraction: float = 0.15,
    seed: int = 42,
    resume: bool = False,
) -> dict[str, int]:
    """Run dual-judge evaluation on a stratified sample of combined DPO examples.

    Samples a fraction of the combined dataset stratified by conflict_type,
    sends each sampled example to both GPT and Gemini judges, and applies
    consensus logic to classify examples as keep/discard/flag.

    Results are flushed to disk every PHASE5_FLUSH_INTERVAL instances so that
    progress is preserved on crash. When resume=True, already-judged instances
    (identified by sample key) are skipped.

    Args:
        combined_path: Path to the combined DPO JSONL (output of Phase 4).
        openai_client: OpenAI API client with a generate() method.
        google_client: Google API client with a generate() method.
        qc_results_path: Path to save per-example judge results.
        flagged_path: Path to save flagged (disagreement) examples.
        fraction: Fraction of each conflict_type group to sample.
        seed: Random seed for stratified sampling.
        resume: If True, skip instances already present in qc_results_path.

    Returns:
        Dict with keys: sampled, kept, discarded, flagged, skipped.
    """
    import hashlib

    from src.data.dpo.quality_control import (
        apply_judge_decisions,
        build_judge_prompt,
        parse_judge_response,
        save_flagged_examples,
        stratified_sample,
    )

    all_examples = _load_jsonl(combined_path)

    # Exclude calibration pairs from dual-judge QC — their structure
    # (y_w=helpful, y_l=unnecessary refusal, level_gap=0) confuses judges
    # that expect a clear hierarchy conflict with swapped compliance.
    judgeable = [ex for ex in all_examples if ex.get("category") != "calibration"]
    calibration_excluded = len(all_examples) - len(judgeable)
    if calibration_excluded:
        logger.info(
            "Phase 5: excluded %d calibration pairs from judging (%d judgeable)",
            calibration_excluded, len(judgeable),
        )

    sample = stratified_sample(judgeable, fraction=fraction, seed=seed)
    logger.info("Phase 5 — dual-judge: sampled %d / %d judgeable examples", len(sample), len(judgeable))

    # Build deterministic keys for each sample instance
    sample_keys: list[str] = []
    for ex in sample:
        key = hashlib.sha256(
            (ex["conflict_type"] + ex["prompt"][:200]).encode()
        ).hexdigest()[:16]
        sample_keys.append(key)

    # Resume: load already-completed keys
    completed_keys: set[str] = set()
    if resume:
        completed_keys = _load_completed_keys(qc_results_path)

    # If resuming, load existing results; otherwise start fresh
    if resume and qc_results_path.exists():
        qc_results = _load_jsonl(qc_results_path)
        open_mode = "a"
    else:
        qc_results = []
        open_mode = "w"

    kept = sum(1 for r in qc_results if r.get("decision") == "keep")
    discarded = sum(1 for r in qc_results if r.get("decision") == "discard")
    flagged_list: list[dict] = []
    skipped = sum(1 for r in qc_results if r.get("decision") == "skipped")
    pending_results: list[str] = []

    # Open file for incremental flushing
    qc_results_path.parent.mkdir(parents=True, exist_ok=True)
    flush_fh = open(qc_results_path, open_mode, encoding="utf-8")

    try:
        remaining = sum(1 for k in sample_keys if k not in completed_keys)
        processed_this_run = 0
        logger.info(
            "Phase 5: %d remaining (%d already completed)",
            remaining, len(completed_keys),
        )

        for i, (ex, key) in enumerate(zip(sample, sample_keys)):
            if key in completed_keys:
                continue

            system_prompt, user_prompt = build_judge_prompt(ex)

            try:
                gpt_raw = openai_client.generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception:
                logger.warning("GPT judge failed for %s, skipping", ex["conflict_type"])
                gpt_raw = None

            try:
                gemini_raw = google_client.generate(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception:
                logger.warning(
                    "Gemini judge failed for %s (likely PROHIBITED_CONTENT), skipping",
                    ex["conflict_type"],
                )
                gemini_raw = None

            gpt_scores = parse_judge_response(gpt_raw) if gpt_raw else None
            gemini_scores = parse_judge_response(gemini_raw) if gemini_raw else None

            if gpt_scores is None or gemini_scores is None:
                skipped += 1
                result_entry = {
                    "_sample_key": key,
                    "conflict_type": ex["conflict_type"],
                    "decision": "skipped",
                    "gpt_raw": gpt_raw[:200] if gpt_raw and gpt_scores is None else None,
                    "gemini_raw": gemini_raw[:200] if gemini_raw and gemini_scores is None else None,
                }
            else:
                decision = apply_judge_decisions(gpt_scores, gemini_scores)
                result_entry = {
                    "_sample_key": key,
                    "conflict_type": ex["conflict_type"],
                    "decision": decision,
                    "gpt_scores": gpt_scores,
                    "gemini_scores": gemini_scores,
                }
                if decision == "keep":
                    kept += 1
                elif decision == "discard":
                    discarded += 1
                elif decision == "flag":
                    flagged_list.append({
                        "example": ex,
                        "gpt_scores": gpt_scores,
                        "gemini_scores": gemini_scores,
                    })

            pending_results.append(json.dumps(result_entry, ensure_ascii=False))
            processed_this_run += 1

            # Flush periodically
            if len(pending_results) >= PHASE5_FLUSH_INTERVAL:
                flush_fh.write("\n".join(pending_results) + "\n")
                flush_fh.flush()
                logger.info(
                    "Phase 5 progress: %d/%d (kept=%d, discarded=%d, flagged=%d, skipped=%d)",
                    processed_this_run + len(completed_keys),
                    len(sample),
                    kept, discarded, len(flagged_list), skipped,
                )
                pending_results.clear()

        # Final flush
        if pending_results:
            flush_fh.write("\n".join(pending_results) + "\n")
            flush_fh.flush()

    finally:
        flush_fh.close()

    if flagged_list:
        save_flagged_examples(flagged_list, flagged_path)

    logger.info(
        "Phase 5 complete: %d sampled, %d kept, %d discarded, %d flagged, %d skipped",
        len(sample), kept, discarded, len(flagged_list), skipped,
    )

    return {
        "sampled": len(sample),
        "kept": kept,
        "discarded": discarded,
        "flagged": len(flagged_list),
        "skipped": skipped,
    }
