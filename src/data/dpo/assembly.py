"""DPO prompt assembly and example creation with delimiter token wrapping.

Constructs DPO training examples by wrapping each hierarchy level's content
in its delimiter tokens (<|L0_START|>...<|L0_END|>, etc.) and splitting
the prompt from chosen/rejected responses wrapped in RESP delimiters.
"""

import datetime
import logging
import random

logger = logging.getLogger(__name__)


def assemble_dpo_prompt(
    l0_rules: list[str] | None = None,
    l1_prompt: str | None = None,
    l2_config: str | None = None,
    l3_message: str | None = None,
    l4_data: str | None = None,
) -> str:
    """Build an L0-L4 prompt with delimiter tokens, without response section.

    Args:
        l0_rules: Platform governance rules, joined with newline. Skipped if
            None or empty.
        l1_prompt: Developer system prompt. Skipped if None.
        l2_config: Per-user configuration. Skipped if None.
        l3_message: User message. Skipped if None.
        l4_data: Data or tool output. Skipped if None.

    Returns:
        Prompt string with each level wrapped in its delimiters, joined
        by newlines. Does NOT include RESP delimiters.
    """
    contents: dict[int, str] = {}

    if l0_rules:
        contents[0] = "\n".join(l0_rules)
    if l1_prompt is not None:
        contents[1] = l1_prompt
    if l2_config is not None:
        contents[2] = l2_config
    if l3_message is not None:
        contents[3] = l3_message
    if l4_data is not None:
        contents[4] = l4_data

    parts: list[str] = []
    for level in sorted(contents):
        parts.append(
            "<|L%d_START|>%s<|L%d_END|>" % (level, contents[level], level)
        )

    return "\n".join(parts)


def assemble_dpo_example(
    prompt: str,
    chosen: str,
    rejected: str,
    conflict_type: str,
    victim_level: int,
    attacker_level: int,
    category: str,
    levels_present: list[int],
    *,
    attack_type: str | None = None,
    yw_source: str | None = None,
    yw_model: str | None = None,
    yw_base_dataset: str | None = None,
    yw_base_index: int | None = None,
    yl_source: str | None = None,
    yl_model: str | None = None,
    yl_base_dataset: str | None = None,
    yl_base_index: int | None = None,
    yl_fallback_used: bool | None = None,
    l0_rule_ids: list[str] | None = None,
    l1_domain: str | None = None,
    l1_index: int | None = None,
    l2_source: str | None = None,
    l2_model: str | None = None,
    l4_source: str | None = None,
    l4_base_dataset: str | None = None,
    l4_base_index: int | None = None,
    injection_template_id: str | None = None,
    injection_position: str | None = None,
    l2_conflict_attribute: str | None = None,
    l2_conflict_value: str | None = None,
    cascading_chain: str | None = None,
    cascading_resolution: str | None = None,
    margin_override: float | None = None,
    embedded_injection: bool | None = None,
    seed: int | None = None,
) -> dict:
    """Build a complete DPO training example with prompt, chosen/rejected, and metadata.

    Wraps chosen and rejected responses in RESP delimiters, computes the
    level gap and margin, and returns a dict containing all fields.

    Args:
        prompt: Pre-assembled prompt string with L0-L4 delimiters.
        chosen: The preferred (winning) response text.
        rejected: The dispreferred (losing) response text.
        conflict_type: Type of conflict (e.g. "L1_vs_L3", "calibration_L3").
        victim_level: The hierarchy level that should win.
        attacker_level: The hierarchy level that attempts to override.
        category: Example category ("pairwise", "calibration", "cascading").
        levels_present: Which hierarchy levels are present in the prompt.
        attack_type: Type of attack (e.g. "naive", "benign").
        yw_source: Source of the chosen response.
        yw_model: Model used to generate chosen response.
        yw_base_dataset: Base dataset for chosen response.
        yw_base_index: Index in base dataset for chosen response.
        yl_source: Source of the rejected response.
        yl_model: Model used to generate rejected response.
        yl_base_dataset: Base dataset for rejected response.
        yl_base_index: Index in base dataset for rejected response.
        yl_fallback_used: Whether a fallback was used for rejected response.
        l0_rule_ids: IDs of L0 rules used.
        l1_domain: Domain of the L1 system prompt.
        l1_index: Index of the L1 system prompt.
        l2_source: Source of L2 configuration.
        l2_model: Model used for L2 generation.
        l4_source: Source of L4 data.
        l4_base_dataset: Base dataset for L4 data.
        l4_base_index: Index in base dataset for L4 data.
        injection_template_id: ID of the injection template used.
        injection_position: Position of the injection in the prompt.
        l2_conflict_attribute: Conflicting attribute in L2 config.
        l2_conflict_value: Conflicting value in L2 config.
        cascading_chain: Chain of cascading conflicts (e.g. "L0>L1>L2>L3").
        cascading_resolution: Resolution strategy for cascading conflicts.
        margin_override: Override for the computed margin value.
        embedded_injection: Whether instruction_a was embedded in the L3 message.
        seed: Random seed for reproducibility. Auto-generated if None.

    Returns:
        Dict with all DPO example fields including computed metadata.
    """
    if not (0 <= victim_level <= 4):
        msg = "victim_level must be in [0, 4], got %d" % victim_level
        raise ValueError(msg)
    if not (0 <= attacker_level <= 4):
        msg = "attacker_level must be in [0, 4], got %d" % attacker_level
        raise ValueError(msg)
    if victim_level > attacker_level:
        msg = (
            "victim_level (%d) must be <= attacker_level (%d) "
            "for conflict_type %r"
            % (victim_level, attacker_level, conflict_type)
        )
        raise ValueError(msg)

    level_gap = attacker_level - victim_level

    if margin_override is not None:
        margin = margin_override
    else:
        margin = float(level_gap)

    is_calibration = category == "calibration"

    build_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    return {
        "prompt": prompt,
        "chosen": "<|RESP_START|>%s<|RESP_END|>" % chosen,
        "rejected": "<|RESP_START|>%s<|RESP_END|>" % rejected,
        "conflict_type": conflict_type,
        "level_gap": level_gap,
        "margin": margin,
        "category": category,
        "is_calibration": is_calibration,
        "attack_type": attack_type,
        "levels_present": levels_present,
        "victim_level": victim_level,
        "attacker_level": attacker_level,
        "yw_source": yw_source,
        "yw_model": yw_model,
        "yw_base_dataset": yw_base_dataset,
        "yw_base_index": yw_base_index,
        "yl_source": yl_source,
        "yl_model": yl_model,
        "yl_base_dataset": yl_base_dataset,
        "yl_base_index": yl_base_index,
        "yl_fallback_used": yl_fallback_used,
        "l0_rule_ids": l0_rule_ids,
        "l1_domain": l1_domain,
        "l1_index": l1_index,
        "l2_source": l2_source,
        "l2_model": l2_model,
        "l4_source": l4_source,
        "l4_base_dataset": l4_base_dataset,
        "l4_base_index": l4_base_index,
        "injection_template_id": injection_template_id,
        "injection_position": injection_position,
        "l2_conflict_attribute": l2_conflict_attribute,
        "l2_conflict_value": l2_conflict_value,
        "cascading_chain": cascading_chain,
        "cascading_resolution": cascading_resolution,
        "embedded_injection": embedded_injection,
        "seed": seed,
        "build_timestamp": build_timestamp,
    }
