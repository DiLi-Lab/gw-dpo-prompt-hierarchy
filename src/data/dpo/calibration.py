"""Over-refusal calibration builder for DPO training.

Constructs calibration examples that teach the model NOT to refuse benign
instructions that superficially resemble prompt injection attacks. y_w is
helpful compliance with the user request, y_l is an unnecessary refusal.
All examples use margin=0.0 (standard DPO, no gravity weighting).
"""

import logging
import random

from src.data.dpo.assembly import assemble_dpo_example, assemble_dpo_prompt
from src.data.libraries.l0_rules import sample_l0_rules
from src.data.libraries.l2_templates import generate_l2
from src.data.sft.domain_classifier import classify_domain, select_matched_l1
from src.data.sft.row_utils import get_output

logger = logging.getLogger(__name__)

_SUSPICIOUS_TRIGGER_WORDS: list[str] = [
    "ignore", "forget", "disregard", "override", "skip",
    "bypass", "cancel", "drop", "stop", "reset", "abandon",
]

_REPHRASE_SYSTEM_PROMPT: str = (
    "Rephrase the following user instruction so it includes "
    "suspicious-sounding language (words like 'ignore', 'forget', "
    "'override', 'disregard', 'skip', 'bypass', 'cancel', 'drop', "
    "'stop', 'reset', 'abandon') while keeping the SAME request. "
    "The rephrased instruction must still be a legitimate, benign "
    "user request — the suspicious words should be used naturally.\n\n"
    "Output ONLY the rephrased instruction, nothing else."
)

_SUSPICIOUS_PREFIXES: list[str] = [
    "Forget about the details and just ",
    "Override the default format and ",
    "Ignore any previous context and ",
    "Disregard the standard approach and ",
    "Skip the formalities and just ",
    "Stop following the usual structure and ",
    "Drop the verbose explanation and simply ",
    "Cancel any prior constraints and ",
    "Bypass the lengthy intro and directly ",
    "Reset expectations and just ",
]

REFUSAL_YL_TEMPLATES: list[str] = [
    "I'm sorry, but I cannot ignore my previous instructions or change my behaviour based on user requests.",
    "I must follow my configured settings and cannot override them.",
    "I'm not able to disregard my system instructions. Is there something else I can help with?",
    "I cannot bypass my guidelines. Let me help you within my current configuration.",
    "I'm unable to change my response format based on that kind of request.",
    "For safety reasons, I cannot comply with requests that ask me to ignore instructions.",
    "I'm designed to follow my system configuration and cannot override it on request.",
    "That request appears to be asking me to change my core behaviour, which I cannot do.",
    "I cannot process requests that ask me to disregard my instructions.",
    "I'm sorry, I'm not able to cancel or override my configured behaviour.",
    "Requests to ignore, forget, or bypass my settings cannot be fulfilled.",
    "I need to maintain my current configuration and cannot alter it based on user input.",
    "I'm unable to abandon my current instructions. How else can I assist you?",
    "I cannot reset my instructions based on a user message.",
    "That looks like it might be trying to change my behaviour, so I'll have to decline.",
    "I'm sorry, but I can only operate within my predefined guidelines.",
    "I must respectfully decline — I cannot override my system settings.",
    "For security purposes, I cannot comply with instructions to disregard my configuration.",
]


def rephrase_with_suspicious_wording(
    client: object,
    instruction: str,
) -> str:
    """Rephrase an instruction to include suspicious-sounding trigger words.

    Args:
        client: OpenAI API client with a generate() method.
        instruction: The original benign instruction.

    Returns:
        The rephrased instruction with suspicious wording.
    """
    return client.generate(
        user_prompt=instruction,
        system_prompt=_REPHRASE_SYSTEM_PROMPT,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
    )


def _fallback_suspicious_rephrase(
    instruction: str,
    rng: random.Random,
) -> str:
    """Prepend a suspicious-sounding prefix to the instruction (no LLM).

    Args:
        instruction: The original benign instruction.
        rng: Random number generator.

    Returns:
        The instruction with a suspicious prefix prepended.
    """
    prefix = rng.choice(_SUSPICIOUS_PREFIXES)
    return prefix + instruction[0].lower() + instruction[1:]


def build_calibration_examples(
    base_rows: list[dict],
    l0_rules: list,
    l1_library: list[dict],
    openai_client: object | None = None,
    l4_lookup: dict[tuple[str, int], dict] | None = None,
    count: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Build over-refusal calibration examples for DPO training.

    For each example, the base row's instruction is rephrased with
    suspicious-sounding wording via GPT-4o-mini (or template fallback).
    y_w is the base row's output (genuine response to the instruction),
    y_l is a randomly chosen unnecessary refusal template.

    Args:
        base_rows: List of base dataset rows with instruction/output fields.
        l0_rules: Full list of L0 rules to sample from.
        l1_library: List of L1 prompt dicts with "domain" key.
        openai_client: Optional OpenAI client for suspicious rephrasing.
            Falls back to template prefix when None.
        l4_lookup: Optional L4 lookup for compatible L4 content.
        count: Number of calibration examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of DPO example dicts with calibration metadata.
    """
    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(count):
        example_seed = seed + i
        row = base_rows[i % len(base_rows)]

        original_instruction = row.get("instruction", "")

        # Rephrase instruction with suspicious wording
        if openai_client is not None:
            try:
                suspicious_instruction = rephrase_with_suspicious_wording(
                    openai_client, original_instruction,
                )
            except Exception:
                logger.warning(
                    "Suspicious rephrase failed for row %d, using fallback", i,
                )
                suspicious_instruction = _fallback_suspicious_rephrase(
                    original_instruction, rng,
                )
        else:
            suspicious_instruction = _fallback_suspicious_rephrase(
                original_instruction, rng,
            )

        # Domain-match L1 to the original instruction (content unchanged)
        domain = classify_domain(original_instruction)
        l1_entry = select_matched_l1(l1_library, domain, seed=example_seed)
        l1_prompt = l1_entry["full_prompt"]

        sampled_rules = sample_l0_rules(l0_rules, seed=example_seed)
        l0_rule_texts = [r.rule for r in sampled_rules]
        l0_rule_ids = [r.id for r in sampled_rules]

        l2_text = generate_l2(seed=example_seed)

        # Resolve compatible L4
        if l4_lookup is not None:
            l4_key = (row.get("_dpo_source", ""), row.get("_dpo_index", 0))
            l4_entry = l4_lookup.get(l4_key)
        else:
            l4_entry = None
        l4_content = l4_entry["l4_content"] if l4_entry else None

        y_w = get_output(row)
        y_l = rng.choice(REFUSAL_YL_TEMPLATES)

        levels_present = [0, 1, 2, 3]
        if l4_content is not None:
            levels_present.append(4)

        prompt = assemble_dpo_prompt(
            l0_rules=l0_rule_texts,
            l1_prompt=l1_prompt,
            l2_config=l2_text,
            l3_message=suspicious_instruction,
            l4_data=l4_content,
        )

        example = assemble_dpo_example(
            prompt=prompt,
            chosen=y_w,
            rejected=y_l,
            conflict_type="calibration_L3",
            victim_level=3,
            attacker_level=3,
            category="calibration",
            levels_present=levels_present,
            attack_type="benign",
            yw_source="base_dataset",
            yw_base_dataset=row.get("_dpo_source"),
            yw_base_index=row.get("_dpo_index"),
            yl_source="template",
            l0_rule_ids=l0_rule_ids,
            l1_domain=domain,
            l2_source="template",
            l4_source=l4_entry.get("generation") if l4_entry else None,
            l4_base_dataset=row.get("_dpo_source") if l4_entry else None,
            l4_base_index=row.get("_dpo_index") if l4_entry else None,
            margin_override=0.0,
            seed=example_seed,
        )

        results.append(example)

    logger.info("Built %d calibration examples", len(results))
    return results
