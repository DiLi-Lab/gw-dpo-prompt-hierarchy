"""SFT instance assembly with delimiter token wrapping.

Constructs training examples by wrapping each hierarchy level's content
in its delimiter tokens (<|L0_START|>...<|L0_END|>, etc.) and combining
them into a single prompt string with an optional response block.
"""

import logging

logger = logging.getLogger(__name__)

_LEVEL_ARGS = ("l0_rules", "l1_prompt", "l2_config", "l3_message", "l4_data")


def assemble_instance(
    l0_rules: list[str] | None = None,
    l1_prompt: str | None = None,
    l2_config: str | None = None,
    l3_message: str | None = None,
    l4_data: str | None = None,
    include_levels: list[int] | None = None,
) -> str:
    """Wrap hierarchy-level content in delimiter tokens and join into a prompt.

    Args:
        l0_rules: Platform governance rules, joined with newline.
        l1_prompt: Developer system prompt.
        l2_config: Per-user configuration.
        l3_message: User message.
        l4_data: Data or tool output.
        include_levels: Which levels (0-4) to include. Defaults to all
            non-None levels.

    Returns:
        Prompt string with each level wrapped in its delimiters, joined
        by newlines. Empty string if no levels are present.
    """
    contents: dict[int, str] = {}

    if l0_rules is not None:
        contents[0] = "\n".join(l0_rules)
    if l1_prompt is not None:
        contents[1] = l1_prompt
    if l2_config is not None:
        contents[2] = l2_config
    if l3_message is not None:
        contents[3] = l3_message
    if l4_data is not None and l4_data.strip():
        contents[4] = l4_data

    if include_levels is not None:
        contents = {k: v for k, v in contents.items() if k in include_levels}

    parts: list[str] = []
    for level in sorted(contents):
        parts.append(
            "<|L%d_START|>%s<|L%d_END|>" % (level, contents[level], level)
        )

    return "\n".join(parts)


def assemble_sft_example(
    response: str,
    levels_present: list[int],
    is_conflict: bool,
    conflict_type: str | None = None,
    l0_rules: list[str] | None = None,
    l1_prompt: str | None = None,
    l2_config: str | None = None,
    l3_message: str | None = None,
    l4_data: str | None = None,
    include_levels: list[int] | None = None,
    sft_source: str | None = None,
    sft_index: int | None = None,
    sft_category: str | None = None,
    l4_generation: str | None = None,
) -> dict[str, str | int | list[int] | bool | None]:
    """Build a complete SFT training example with prompt, response, and metadata.

    Args:
        response: The target response text.
        levels_present: Which hierarchy levels are present in this example.
        is_conflict: Whether this example contains a hierarchy conflict.
        conflict_type: Type of conflict (e.g. "l0_vs_l1"), or None.
        l0_rules: Platform governance rules.
        l1_prompt: Developer system prompt.
        l2_config: Per-user configuration.
        l3_message: User message.
        l4_data: Data or tool output.
        include_levels: Which levels to include in the prompt.
        sft_source: Origin dataset name (e.g. "alpaca", "dolly").
        sft_index: Row index in the source dataset.
        sft_category: Builder category (e.g. "simple_aligned").
        l4_generation: L4 generation method (e.g. "wrapped", "synthesized").

    Returns:
        Dict with keys: text, levels_present, is_conflict, conflict_type,
        sft_source, sft_index, sft_category, l4_generation.
    """
    prompt = assemble_instance(
        l0_rules=l0_rules,
        l1_prompt=l1_prompt,
        l2_config=l2_config,
        l3_message=l3_message,
        l4_data=l4_data,
        include_levels=include_levels,
    )

    resp_block = "<|RESP_START|>%s<|RESP_END|>" % response
    text = "%s\n%s" % (prompt, resp_block) if prompt else resp_block

    return {
        "text": text,
        "levels_present": levels_present,
        "is_conflict": is_conflict,
        "conflict_type": conflict_type,
        "sft_source": sft_source,
        "sft_index": sft_index,
        "sft_category": sft_category,
        "l4_generation": l4_generation,
    }
