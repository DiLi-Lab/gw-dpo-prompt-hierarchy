"""L0 platform governance rules loader and sampler.

Loads handcrafted rules from l0_rules.json and provides sampling
with a guarantee that at least one system_integrity rule is always
included, since it is directly relevant to hierarchy enforcement.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_INTEGRITY_CATEGORY: str = "system_integrity"


@dataclass(frozen=True)
class L0Rule:
    """A single L0 platform governance rule."""

    category: str
    rule: str
    id: str


def load_l0_rules(path: Path) -> list[L0Rule]:
    """Load L0 rules from a JSON file.

    Args:
        path: Path to l0_rules.json.

    Returns:
        List of L0Rule instances.

    Raises:
        FileNotFoundError: If the rules file does not exist.
    """
    if not path.exists():
        msg = f"L0 rules file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw = json.load(f)

    rules = [L0Rule(category=r["category"], rule=r["rule"], id=r["id"]) for r in raw]
    logger.info("Loaded %d L0 rules from %s", len(rules), path)
    return rules


def sample_l0_rules(
    rules: list[L0Rule],
    min_rules: int = 3,
    max_rules: int = 6,
    seed: int | None = None,
    required_category: str | None = None,
    required_rule_ids: list[str] | None = None,
) -> list[L0Rule]:
    """Sample L0 rules with optional category and specific rule guarantees.

    When required_category is None, defaults to guaranteeing one
    system_integrity rule (existing behavior). When required_rule_ids
    is provided, those specific rules are always included.

    Args:
        rules: Full list of L0 rules to sample from.
        min_rules: Minimum number of rules to sample.
        max_rules: Maximum number of rules to sample.
        seed: Random seed for reproducibility.
        required_category: Category to guarantee. Defaults to system_integrity.
        required_rule_ids: Specific rule IDs that must be included.

    Returns:
        List of sampled L0Rule instances.
    """
    rng = random.Random(seed)

    # Start with specifically required rules
    pinned: list[L0Rule] = []
    pinned_ids: set[str] = set()
    if required_rule_ids:
        rules_by_id = {r.id: r for r in rules}
        for rid in required_rule_ids:
            if rid in rules_by_id:
                pinned.append(rules_by_id[rid])
                pinned_ids.add(rid)

    # Guarantee at least one rule from the required category
    target_category = required_category or SYSTEM_INTEGRITY_CATEGORY
    category_rules = [r for r in rules if r.category == target_category and r.id not in pinned_ids]

    if not category_rules and not any(r.category == target_category for r in pinned):
        msg = "No rules found for category %r" % target_category
        raise ValueError(msg)

    if not any(r.category == target_category for r in pinned) and category_rules:
        pinned.append(rng.choice(category_rules))
        pinned_ids.add(pinned[-1].id)

    num_rules = rng.randint(min_rules, max(max_rules, len(pinned)))
    remaining_count = max(0, num_rules - len(pinned))

    pool = [r for r in rules if r.id not in pinned_ids]
    chosen_others = rng.sample(pool, min(remaining_count, len(pool)))

    result = pinned + chosen_others
    rng.shuffle(result)
    return result
