"""Scenario-based builders for L0-vs-L1 and L0-vs-L2 DPO pairs.

Constructs thematically coherent conflict pairs where L0 platform
rules are the victim. Uses adversarial scenario templates that define
the adversarial content and matching L3 user messages, ensuring all
levels are thematically related.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.data.dpo.assembly import assemble_dpo_example, assemble_dpo_prompt
from src.data.dpo.yl_generation import generate_yl
from src.data.dpo.yw_generation import generate_yw_context_distillation
from src.data.libraries.l0_rules import sample_l0_rules
from src.data.libraries.l2_generator import generate_l2_from_response
from src.data.libraries.l2_templates import generate_l2, generate_l2_for_conflict
from src.data.sft.domain_classifier import classify_domain, select_matched_l1

logger = logging.getLogger(__name__)

COMPATIBLE_L4_RATIO: float = 0.7


@dataclass
class AdversarialScenario:
    """A scenario template for L0-vs-L1 or L0-vs-L2 conflict construction.

    Attributes:
        id: Unique identifier.
        pair_type: "L0_vs_L1" or "L0_vs_L2".
        l0_category: Which L0 category this scenario targets.
        l0_rule_ids: Specific L0 rule IDs targeted (for metadata/auditing).
        adversarial_l1: Adversarial system prompt (L0-vs-L1 only).
        l2_conflict_attribute: L2 attribute to pin (L0-vs-L2 only).
        l2_conflict_value: L2 value to pin (L0-vs-L2 only).
        l3_templates: List of L3 user messages that trigger the conflict.
    """

    id: str
    pair_type: str
    l0_category: str
    l0_rule_ids: list[str]
    adversarial_l1: str | None
    l2_conflict_attribute: str | None
    l2_conflict_value: str | None
    l3_templates: list[str]


def load_l0_conflict_scenarios(path: Path) -> list[AdversarialScenario]:
    """Load adversarial scenario templates from a JSON file.

    Args:
        path: Path to l0_conflict_scenarios.json.

    Returns:
        List of AdversarialScenario instances.
    """
    with open(path) as f:
        data = json.load(f)

    scenarios = [
        AdversarialScenario(
            id=e["id"],
            pair_type=e["pair_type"],
            l0_category=e["l0_category"],
            l0_rule_ids=e.get("l0_rule_ids", []),
            adversarial_l1=e.get("adversarial_l1"),
            l2_conflict_attribute=e.get("l2_conflict_attribute"),
            l2_conflict_value=e.get("l2_conflict_value"),
            l3_templates=e["l3_templates"],
        )
        for e in data
    ]
    logger.info("Loaded %d L0 conflict scenarios from %s", len(scenarios), path)
    return scenarios


def _sample_random_l4(
    l4_lookup: dict[tuple[str, int], dict],
    rng: random.Random,
) -> tuple[str | None, dict | None]:
    """Randomly sample an L4 entry from the lookup.

    Returns:
        Tuple of (l4_content, l4_entry_dict) or (None, None).
    """
    if not l4_lookup:
        return None, None
    key = rng.choice(list(l4_lookup.keys()))
    entry = l4_lookup[key]
    return entry["l4_content"], entry


def _sample_domain_filtered_l4(
    l4_lookup: dict[tuple[str, int], dict],
    l4_domain_index: dict[str, list[tuple[str, int]]],
    domain: str,
    rng: random.Random,
    used_keys: set[tuple[str, int]] | None = None,
) -> tuple[str | None, dict | None]:
    """Sample an L4 entry matching the given domain, with random fallback.

    Tries the domain-specific pool first (excluding used_keys). If
    exhausted or empty, falls back to random sampling from the full
    l4_lookup. Adds the sampled key to used_keys.

    Args:
        l4_lookup: Full L4 library mapping (source, index) to entry dicts.
        l4_domain_index: Mapping domain -> list of (source, index) keys.
        domain: Target domain to sample from.
        rng: Random number generator.
        used_keys: Set of already-used keys (mutated in place).

    Returns:
        Tuple of (l4_content, l4_entry_dict) or (None, None).
    """
    if not l4_lookup:
        return None, None

    if used_keys is None:
        used_keys = set()

    # Try domain-filtered pool
    domain_keys = l4_domain_index.get(domain, [])
    available = [k for k in domain_keys if k not in used_keys]
    if available:
        key = rng.choice(available)
        used_keys.add(key)
        entry = l4_lookup[key]
        return entry["l4_content"], entry

    # Fallback to random from full pool
    all_available = [k for k in l4_lookup if k not in used_keys]
    if not all_available:
        return None, None

    key = rng.choice(all_available)
    used_keys.add(key)
    entry = l4_lookup[key]
    return entry["l4_content"], entry


def build_l0_vs_l1_pair(
    scenario: AdversarialScenario,
    l0_rules: list,
    l4_lookup: dict,
    openai_client: object,
    anthropic_client: object,
    seed: int,
    l4_domain_index: dict[str, list[tuple[str, int]]] | None = None,
    l4_used_keys: set[tuple[str, int]] | None = None,
) -> dict | None:
    """Build one L0-vs-L1 DPO pair from an adversarial scenario.

    Args:
        scenario: The adversarial scenario template.
        l0_rules: Full list of L0 rules.
        l4_lookup: L4 library lookup for compatible L4.
        openai_client: OpenAI client for L2 generation and y_l.
        anthropic_client: Anthropic client for y_w context distillation.
        seed: Random seed.
        l4_domain_index: Optional domain -> [(source, index)] index for
            domain-filtered L4 sampling. Falls back to random if None.
        l4_used_keys: Optional set of already-used L4 keys (mutated in place)
            for deduplication within a scenario.

    Returns:
        DPO example dict, or None if y_l generation fails.
    """
    rng = random.Random(seed)

    l3_message = rng.choice(scenario.l3_templates)
    l1_prompt = scenario.adversarial_l1 or ""

    sampled_l0 = sample_l0_rules(
        l0_rules, seed=rng.randint(0, 2**31 - 1),
        required_category=scenario.l0_category,
        required_rule_ids=scenario.l0_rule_ids or None,
    )
    l0_rule_strings = [r.rule for r in sampled_l0]
    l0_rule_ids = [r.id for r in sampled_l0]

    # Compatible L4 (~70%) — resolved BEFORE y_w so context distillation sees L4
    include_l4 = rng.random() < COMPATIBLE_L4_RATIO
    l4_content, l4_entry = (None, None)
    if include_l4:
        scenario_domain = classify_domain(scenario.l3_templates[0])
        if l4_domain_index is not None:
            l4_content, l4_entry = _sample_domain_filtered_l4(
                l4_lookup, l4_domain_index, scenario_domain, rng, l4_used_keys,
            )
        else:
            l4_content, l4_entry = _sample_random_l4(l4_lookup, rng)

    # y_w via Claude context distillation
    conflict_description = (
        "L0 (%s) prohibits this action. L1 is an adversarial system prompt "
        "that encourages the model to comply. L3 is a request within L1's "
        "adversarial scope. The correct response follows L0: refuse the "
        "unsafe request while being as helpful as possible."
        % scenario.l0_category
    )
    yw_text = generate_yw_context_distillation(
        client=anthropic_client,
        l0_rules=l0_rule_strings,
        l1_prompt=l1_prompt,
        l3_message=l3_message,
        conflict_description=conflict_description,
        l4_data=l4_content,
        expect_refusal=True,
    )

    # L2 response-grounded on y_w
    if openai_client is not None:
        l2_text = generate_l2_from_response(
            openai_client, l1_prompt, l3_message, yw_text,
        )
        l2_source = "response_grounded"
        l2_model = "gpt-4o-mini"
    else:
        l2_text = generate_l2(seed=rng.randint(0, 2**31 - 1))
        l2_source = "template"
        l2_model = None

    levels_present = [0, 1, 2, 3]
    if l4_content is not None:
        levels_present.append(4)

    # y_l via GPT-4o-mini research framing
    yl_result = generate_yl(client=openai_client, injection=l3_message)
    if yl_result.text is None:
        logger.info("Discarding L0_vs_L1 pair for scenario %s: y_l failed", scenario.id)
        return None

    prompt = assemble_dpo_prompt(
        l0_rules=l0_rule_strings,
        l1_prompt=l1_prompt,
        l2_config=l2_text,
        l3_message=l3_message,
        l4_data=l4_content,
    )

    return assemble_dpo_example(
        prompt=prompt,
        chosen=yw_text,
        rejected=yl_result.text,
        conflict_type="L0_vs_L1",
        victim_level=0,
        attacker_level=1,
        category="pairwise",
        levels_present=levels_present,
        attack_type="adversarial_l1",
        yw_source="context_distillation",
        yw_model="claude-sonnet-4-20250514",
        yl_source="research_framing",
        yl_model=yl_result.model,
        yl_fallback_used=yl_result.fallback_used is not None,
        l0_rule_ids=l0_rule_ids,
        l1_domain="adversarial",
        l2_source=l2_source,
        l2_model=l2_model,
        l4_source=l4_entry.get("generation") if l4_entry else None,
        seed=seed,
    )


def build_l0_vs_l2_pair(
    scenario: AdversarialScenario,
    l0_rules: list,
    l1_library: list[dict],
    l4_lookup: dict,
    openai_client: object,
    anthropic_client: object,
    seed: int,
    l4_domain_index: dict[str, list[tuple[str, int]]] | None = None,
    l4_used_keys: set[tuple[str, int]] | None = None,
) -> dict | None:
    """Build one L0-vs-L2 DPO pair from an adversarial scenario.

    Args:
        scenario: The adversarial scenario template.
        l0_rules: Full list of L0 rules.
        l1_library: Normal L1 library for domain-matched L1.
        l4_lookup: L4 library lookup for compatible L4.
        openai_client: OpenAI client for L2 generation and y_l.
        anthropic_client: Anthropic client for y_w context distillation.
        seed: Random seed.
        l4_domain_index: Optional domain -> [(source, index)] index for
            domain-filtered L4 sampling. Falls back to random if None.
        l4_used_keys: Optional set of already-used L4 keys (mutated in place)
            for deduplication within a scenario.

    Returns:
        DPO example dict, or None if y_l generation fails.
    """
    rng = random.Random(seed)

    l3_message = rng.choice(scenario.l3_templates)

    # Domain-match L1 from normal library using L3 text
    domain = classify_domain(l3_message)
    l1_entry = select_matched_l1(l1_library, domain, seed=rng.randint(0, 2**31 - 1))
    l1_prompt = l1_entry.get("full_prompt", "")

    sampled_l0 = sample_l0_rules(
        l0_rules, seed=rng.randint(0, 2**31 - 1),
        required_category=scenario.l0_category,
        required_rule_ids=scenario.l0_rule_ids or None,
    )
    l0_rule_strings = [r.rule for r in sampled_l0]
    l0_rule_ids = [r.id for r in sampled_l0]

    # Validate L2 conflict fields
    if not scenario.l2_conflict_attribute or not scenario.l2_conflict_value:
        msg = "L0-vs-L2 scenario %s missing l2_conflict_attribute/value" % scenario.id
        raise ValueError(msg)

    # L2 with conflict attribute
    l2_config_obj = generate_l2_for_conflict(
        scenario.l2_conflict_attribute, scenario.l2_conflict_value,
        seed=rng.randint(0, 2**31 - 1),
    )
    l2_text = l2_config_obj.text

    # Compatible L4 (~70%) — resolved BEFORE y_w so context distillation sees L4
    include_l4 = rng.random() < COMPATIBLE_L4_RATIO
    l4_content, l4_entry = (None, None)
    if include_l4:
        scenario_domain = classify_domain(scenario.l3_templates[0])
        if l4_domain_index is not None:
            l4_content, l4_entry = _sample_domain_filtered_l4(
                l4_lookup, l4_domain_index, scenario_domain, rng, l4_used_keys,
            )
        else:
            l4_content, l4_entry = _sample_random_l4(l4_lookup, rng)

    # y_w via Claude context distillation
    conflict_description = (
        "L0 (%s) sets a safety constraint. L2 tries to disable it via "
        "%s=%s. L3 is a request affected by this constraint. "
        "The correct response follows L0 while being as helpful as possible."
        % (scenario.l0_category, scenario.l2_conflict_attribute, scenario.l2_conflict_value)
    )
    yw_text = generate_yw_context_distillation(
        client=anthropic_client,
        l0_rules=l0_rule_strings,
        l1_prompt=l1_prompt,
        l3_message=l3_message,
        conflict_description=conflict_description,
        l4_data=l4_content,
        l2_config=l2_text,
        expect_refusal=True,
    )

    levels_present = [0, 1, 2, 3]
    if l4_content is not None:
        levels_present.append(4)

    # y_l via GPT-4o-mini
    yl_result = generate_yl(client=openai_client, injection=l3_message)
    if yl_result.text is None:
        logger.info("Discarding L0_vs_L2 pair for scenario %s: y_l failed", scenario.id)
        return None

    prompt = assemble_dpo_prompt(
        l0_rules=l0_rule_strings,
        l1_prompt=l1_prompt,
        l2_config=l2_text,
        l3_message=l3_message,
        l4_data=l4_content,
    )

    return assemble_dpo_example(
        prompt=prompt,
        chosen=yw_text,
        rejected=yl_result.text,
        conflict_type="L0_vs_L2",
        victim_level=0,
        attacker_level=2,
        category="pairwise",
        levels_present=levels_present,
        attack_type="adversarial_l2",
        yw_source="context_distillation",
        yw_model="claude-sonnet-4-20250514",
        yl_source="research_framing",
        yl_model=yl_result.model,
        yl_fallback_used=yl_result.fallback_used is not None,
        l0_rule_ids=l0_rule_ids,
        l1_domain=domain,
        l2_source="template_conflict",
        l2_conflict_attribute=scenario.l2_conflict_attribute,
        l2_conflict_value=scenario.l2_conflict_value,
        l4_source=l4_entry.get("generation") if l4_entry else None,
        seed=seed,
    )
