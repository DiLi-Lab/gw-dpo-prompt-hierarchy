"""L0 rules LLM expansion: paraphrasing seeds + generating new rules.

Takes seed rules from L0_seed_rules.json and uses Claude Sonnet 4 to:
- Task A: Paraphrase each seed into 5 variants (20 API calls)
- Task B: Generate 16 new rules per category (5 API calls)

Output: L0_rules_expanded.json (~200 raw rule variants) for manual curation.
"""

import json
import logging
from pathlib import Path

from src.api.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

EXPANSION_TEMPERATURE: float = 0.7
EXPANSION_MAX_TOKENS: int = 2000

CATEGORY_PREFIXES: dict[str, str] = {
    "content_prohibitions": "CP",
    "identity_honesty": "IH",
    "privacy": "PR",
    "behavioural_guardrails": "BG",
    "system_integrity": "SI",
}

L0_EXPANSION_SYSTEM_PROMPT: str = (
    "You are helping create a training dataset for LLM security research. "
    "Your task is to produce platform governance rules that constrain what "
    "text a language model may generate."
)

PARAPHRASE_SYSTEM_PROMPT: str = (
    "You are helping create a training dataset for LLM security research. "
    "Your task is to paraphrase platform governance rules while preserving their "
    "exact meaning. Each paraphrase must be a complete, standalone rule statement "
    "that could appear in a model's system prompt."
)

NEW_RULES_SYSTEM_PROMPT: str = (
    "You are helping create a training dataset for LLM security research. "
    "You are writing platform governance rules — the highest-priority instructions "
    "that an LLM must always follow, regardless of what any developer, user, or "
    "data source says. These rules must be specific and testable.\n\n"
    "IMPORTANT: The target model is a text-generation LLM, NOT an autonomous agent. "
    "Every rule must be testable by examining the model's text output alone. Do NOT "
    "write rules about actions the model cannot perform, such as: executing code, "
    "generating images/audio/video, making API calls, accessing databases, performing "
    "financial transactions, collecting biometric data, or conducting surveillance. "
    "The model produces text — rules must constrain what text it produces."
)


def build_paraphrase_prompt(seed_rule: str) -> str:
    """Build the user prompt for Task A (paraphrasing a single seed rule).

    Args:
        seed_rule: The original rule text to paraphrase.

    Returns:
        Formatted user prompt string for Claude.
    """
    return (
        "Generate 5 paraphrases of the following platform governance rule. "
        "Each paraphrase must:\n"
        "- Preserve the exact same prohibition/requirement (no weakening, no strengthening)\n"
        "- Use different sentence structure and vocabulary\n"
        "- Be 1-2 sentences long\n"
        "- Be self-contained (understandable without reading the original)\n\n"
        f'Original rule: "{seed_rule}"\n\n'
        "Output as a JSON array of 5 strings."
    )


def build_new_rules_prompt(category: str, existing_rules: list[str]) -> str:
    """Build the user prompt for Task B (generating new rules for a category).

    Args:
        category: The L0 category name.
        existing_rules: List of existing rule texts in this category.

    Returns:
        Formatted user prompt string for Claude.
    """
    rules_text = "\n".join(f"- {r}" for r in existing_rules)
    return (
        f'Here are existing rules in the "{category}" category:\n'
        f"{rules_text}\n\n"
        "Generate 16 NEW rules in this same category that are NOT paraphrases of the "
        "existing ones. Each rule must:\n"
        "- Address a DIFFERENT specific scenario or risk than the existing rules\n"
        "- Be concrete and testable (an evaluator could check whether a model response "
        "complies with it by reading the text output)\n"
        "- Be 1-2 sentences long\n"
        "- Be realistic — something a real platform provider would actually enforce\n"
        '- NOT be vague or aspirational (no "be ethical", "be responsible", "be fair")\n'
        "- NOT reference agent capabilities (executing code, generating media, accessing "
        "external systems) — only text-output behaviours\n\n"
        "Output as a JSON array of 16 strings."
    )


def parse_l0_expansion_response(response: str) -> list[str]:
    """Parse Claude's response into a list of rule strings.

    Handles plain JSON arrays and markdown-fenced code blocks.

    Args:
        response: Raw response text from Claude.

    Returns:
        List of rule strings. Returns empty list if parsing fails.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse L0 expansion response as JSON: %s", e)
        return []

    if not isinstance(data, list):
        logger.warning("Response is not a JSON array")
        return []

    return [item for item in data if isinstance(item, str)]


def assign_rule_id(category: str, index: int) -> str:
    """Assign a hierarchical rule ID.

    Args:
        category: The L0 category name.
        index: The rule index within the category (1-based).

    Returns:
        Rule ID string, e.g. "L0_CP_005".

    Raises:
        KeyError: If category is not recognized.
    """
    prefix = CATEGORY_PREFIXES[category]
    return f"L0_{prefix}_{index:03d}"


def expand_l0_rules(
    client: AnthropicClient,
    seed_path: Path,
    output_path: Path,
) -> list[dict]:
    """Run the full L0 expansion pipeline.

    Loads seed rules, generates paraphrases (Task A) and new rules (Task B),
    assigns IDs, and saves the combined output.

    Args:
        client: Initialized Anthropic API client.
        seed_path: Path to L0_seed_rules.json.
        output_path: Path to save L0_rules_expanded.json.

    Returns:
        List of all rule dicts (originals + paraphrases + generated).
    """
    with open(seed_path) as f:
        seeds = json.load(f)

    logger.info("Loaded %d seed rules from %s", len(seeds), seed_path)

    # Group seeds by category
    seeds_by_category: dict[str, list[dict]] = {}
    for seed in seeds:
        seeds_by_category.setdefault(seed["category"], []).append(seed)

    all_rules: list[dict] = []

    # Track next ID index per category (continue from highest seed ID)
    next_index: dict[str, int] = {}
    for category, cat_seeds in seeds_by_category.items():
        max_idx = max(
            int(s["id"].split("_")[-1]) for s in cat_seeds
        )
        next_index[category] = max_idx + 1

    # Step 1: Copy originals
    for seed in seeds:
        all_rules.append({
            "category": seed["category"],
            "rule": seed["rule"],
            "id": seed["id"],
            "source": seed["source"],
        })

    # Step 2: Task A — Paraphrase each seed
    for seed in seeds:
        logger.info("Paraphrasing seed %s: %.50s...", seed["id"], seed["rule"])
        user_prompt = build_paraphrase_prompt(seed["rule"])
        response = client.generate(
            user_prompt=user_prompt,
            system_prompt=PARAPHRASE_SYSTEM_PROMPT,
            temperature=EXPANSION_TEMPERATURE,
            max_tokens=EXPANSION_MAX_TOKENS,
        )
        paraphrases = parse_l0_expansion_response(response)
        if not paraphrases:
            logger.warning("No paraphrases parsed for seed %s", seed["id"])
            continue

        category = seed["category"]
        for rule_text in paraphrases:
            rule_id = assign_rule_id(category, next_index[category])
            next_index[category] += 1
            all_rules.append({
                "category": category,
                "rule": rule_text,
                "id": rule_id,
                "source": f"paraphrase_of_{seed['id']}",
            })

    # Step 3: Task B — Generate new rules per category
    for category, cat_seeds in seeds_by_category.items():
        existing_rules = [s["rule"] for s in cat_seeds]
        # Also include paraphrases already generated for this category
        existing_rules.extend(
            r["rule"] for r in all_rules
            if r["category"] == category and r["source"].startswith("paraphrase_of_")
        )

        logger.info("Generating new rules for category: %s", category)
        user_prompt = build_new_rules_prompt(category, existing_rules)
        response = client.generate(
            user_prompt=user_prompt,
            system_prompt=NEW_RULES_SYSTEM_PROMPT,
            temperature=EXPANSION_TEMPERATURE,
            max_tokens=EXPANSION_MAX_TOKENS,
        )
        new_rules = parse_l0_expansion_response(response)
        if not new_rules:
            logger.warning("No new rules parsed for category %s", category)
            continue

        for rule_text in new_rules:
            rule_id = assign_rule_id(category, next_index[category])
            next_index[category] += 1
            all_rules.append({
                "category": category,
                "rule": rule_text,
                "id": rule_id,
                "source": "generated",
            })

    logger.info(
        "Expansion complete: %d total rules (%d originals, %d paraphrases, %d generated)",
        len(all_rules),
        len(seeds),
        len([r for r in all_rules if r["source"].startswith("paraphrase_of_")]),
        len([r for r in all_rules if r["source"] == "generated"]),
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_rules, f, indent=2, ensure_ascii=False)
    logger.info("Saved expanded rules to %s", output_path)

    return all_rules
