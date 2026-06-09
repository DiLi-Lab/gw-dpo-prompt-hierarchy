"""Injection placement logic for DPO pair construction.

Provides functions to place adversarial injections into prompts and content
at various positions, used to construct rejected samples in DPO training pairs.
"""


def apply_prefix_injection(
    instruction: str,
    prefix: str,
    legitimate_instruction: str | None = None,
) -> str:
    """Prepend a prefix to a stripped instruction, optionally embedding a legitimate instruction.

    Args:
        instruction: The adversarial instruction text (whitespace will be stripped).
        prefix: The adversarial prefix to prepend.
        legitimate_instruction: If provided, prepended before the prefix so the
            legitimate task appears in L3 alongside the injection.

    Returns:
        The assembled injection string.
    """
    if legitimate_instruction is not None:
        return legitimate_instruction.strip() + " " + prefix + instruction.strip()
    return prefix + instruction.strip()


def apply_position_injection(content: str, injection: str, position: str) -> str:
    """Insert an injection at a specified position within content.

    Args:
        content: The original content text.
        injection: The text to inject.
        position: Where to place the injection — "start", "middle", or "end".

    Returns:
        The content with the injection inserted at the specified position.

    Raises:
        ValueError: If position is not one of "start", "middle", or "end".
    """
    if position == "start":
        return injection + "\n\n" + content
    if position == "end":
        return content + "\n\n" + injection
    if position == "middle":
        mid = len(content) // 2
        return content[:mid] + "\n\n" + injection + "\n\n" + content[mid:]
    raise ValueError(
        f"Invalid position '{position}': must be 'start', 'middle', or 'end'"
    )


def apply_system_override(template: str, payload: str) -> str:
    """Format a system override template with a payload.

    Args:
        template: A template string containing ``{payload}`` as placeholder.
        payload: The adversarial payload to insert.

    Returns:
        The formatted template with the payload substituted.
    """
    return template.format(payload=payload)
