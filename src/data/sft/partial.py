"""Partial-level SFT examples builder.

Constructs 2,000 examples where only a subset of hierarchy levels are
present (2-3 levels per example). This teaches the model that not all
levels need be populated. Four configurations of 500 examples each:

1. L1+L3 only       -- system prompt + user message
2. L0+L1+L3         -- platform rules + system prompt + user message
3. L1+L3+L4         -- system prompt + user message + tool output
4. L0+L1+L2+L3      -- everything except tool output

Zero LLM cost: uses base dataset outputs as responses.
"""

import logging
import random

from src.data.libraries.l0_rules import L0Rule, sample_l0_rules
from src.data.libraries.l2_generator import generate_l2_from_response
from src.data.libraries.l2_templates import generate_l2
from src.data.sft.assembly import assemble_sft_example
from src.data.sft.domain_classifier import classify_domain, select_matched_l1
from src.data.sft.row_utils import get_output

logger = logging.getLogger(__name__)

PARTIAL_CONFIGS: list[dict] = [
    {"name": "L1+L3", "levels": [1, 3]},
    {"name": "L0+L1+L3", "levels": [0, 1, 3]},
    {"name": "L1+L3+L4", "levels": [1, 3, 4]},
    {"name": "L0+L1+L2+L3", "levels": [0, 1, 2, 3]},
]


def build_partial_examples(
    base_rows: list[dict],
    l0_rules: list[L0Rule],
    l1_library: list[dict],
    l4_lookup: dict[tuple[str, int], dict[str, str]],
    per_config_count: int = 500,
    seed: int = 42,
    openai_client: object | None = None,
    l2_cache: dict[tuple[str, int], str] | None = None,
) -> list[dict]:
    """Build partial-level SFT examples across all 4 configurations.

    For each configuration, samples base_rows and assembles examples
    with only the specified hierarchy levels present.

    Args:
        base_rows: List of tagged row dicts (``_sft_source``, ``_sft_index``).
        l0_rules: Full list of L0Rule objects to sample from.
        l1_library: List of L1 prompt dicts with domain key.
        l4_lookup: Dict mapping (source, index) to ``{"l4_content": str, "generation": str}``.
        per_config_count: Number of examples per configuration.
        seed: Random seed for reproducibility.

    Returns:
        List of 4 * per_config_count assembled SFT example dicts.
    """
    examples: list[dict] = []

    for config_idx, config in enumerate(PARTIAL_CONFIGS):
        config_seed = seed + config_idx * per_config_count
        rng = random.Random(config_seed)

        levels = config["levels"]

        # For configs that include L4, filter to rows that have L4 entries
        # to avoid silently degrading to a different config
        if 4 in levels:
            candidate_indices = [
                idx for idx in range(len(base_rows))
                if l4_lookup.get((base_rows[idx]["_sft_source"], base_rows[idx]["_sft_index"])) is not None
            ]
            if len(candidate_indices) < per_config_count:
                logger.warning(
                    "Config '%s': only %d rows have L4 content, requested %d",
                    config["name"], len(candidate_indices), per_config_count,
                )
        else:
            candidate_indices = list(range(len(base_rows)))

        rng.shuffle(candidate_indices)
        selected = candidate_indices[:per_config_count]

        for i, row_idx in enumerate(selected):
            row = base_rows[row_idx]
            row_seed = config_seed + i

            l4_key = (row["_sft_source"], row["_sft_index"])
            l4_entry = l4_lookup.get(l4_key) if 4 in levels else None
            l4_content = l4_entry["l4_content"] if l4_entry else None
            l4_gen = l4_entry["generation"] if l4_entry else None
            effective_levels = list(levels)

            # Build level content only for included levels
            l0_sampled = None
            if 0 in effective_levels:
                sampled = sample_l0_rules(l0_rules, seed=row_seed)
                l0_sampled = [r.rule for r in sampled]

            l1_text = None
            if 1 in effective_levels:
                instruction = row["instruction"]
                domain = classify_domain(instruction)
                l1 = select_matched_l1(l1_library, domain, seed=row_seed)
                l1_text = l1["full_prompt"]

            l2_text = None
            if 2 in effective_levels:
                row_key = (row["_sft_source"], row["_sft_index"])
                if openai_client is not None:
                    cached_l2 = (l2_cache or {}).get(row_key)
                    if cached_l2 is not None:
                        l2_text = cached_l2
                    else:
                        l2_text = generate_l2_from_response(
                            openai_client,
                            l1_prompt=l1_text or "",
                            l3_message=row["instruction"],
                            response=get_output(row),
                        )
                else:
                    l2_text = generate_l2(seed=row_seed)

            l3_text = None
            if 3 in effective_levels:
                l3_text = row["instruction"]

            example = assemble_sft_example(
                response=get_output(row),
                levels_present=sorted(effective_levels),
                is_conflict=False,
                conflict_type=None,
                l0_rules=l0_sampled,
                l1_prompt=l1_text,
                l2_config=l2_text,
                l3_message=l3_text,
                l4_data=l4_content,
                include_levels=effective_levels,
                sft_source=row["_sft_source"],
                sft_index=row["_sft_index"],
                sft_category="partial_%s" % config["name"],
                l4_generation=l4_gen,
            )
            examples.append(example)

        logger.info(
            "Built %d partial examples for config '%s' (levels=%s)",
            len(selected), config["name"], levels,
        )

    logger.info("Built %d total partial-level examples", len(examples))
    return examples
