"""Two prompt-format builders used by the XSTest and IHEval runners.

- :func:`build_delimited` wraps level-tagged content in the project's
  ``<|Lx_START|>...<|Lx_END|>`` tokens via
  :func:`src.data.sft.assembly.assemble_instance`. ISE-aware checkpoints
  consume this format natively.
- :func:`build_chat_template` produces a ``tokenizer.apply_chat_template``
  string. ISE is bypassed for this format because there are no delimiter
  spans for segment-id derivation.
"""

from typing import Any

from src.data.sft.assembly import assemble_instance


def build_delimited(
    *,
    l1: str | None = None,
    l3: str,
    l4: str | None = None,
) -> str:
    """Wrap level-tagged content in the project's hierarchy delimiters.

    L0 and L2 are intentionally absent for these benchmarks: XSTest has
    no platform/per-user content, and IHEval's hierarchy maps cleanly
    onto L1/L3/L4 only.

    Args:
        l1: Optional system / developer content.
        l3: User instruction (required).
        l4: Optional data/history block.

    Returns:
        A string containing the level blocks joined by newlines. The
        caller is responsible for appending the response delimiter
        (``<|RESP_START|>``) if needed.
    """
    return assemble_instance(
        l1_prompt=l1,
        l3_message=l3,
        l4_data=l4,
    )


def build_chat_template(
    tokenizer: Any,
    *,
    system: str | None = None,
    user: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Render an OpenAI-style message list via the tokenizer's chat template.

    Args:
        tokenizer: Any object exposing
            ``apply_chat_template(messages, add_generation_prompt)``.
        system: Optional system message content.
        user: Final user message (required).
        history: Optional list of ``(role, content)`` tuples appended after
            the system message and before the final user message. ``role``
            must be ``"user"`` or ``"assistant"``.

    Returns:
        The string returned by ``tokenizer.apply_chat_template`` with
        ``add_generation_prompt=True``.
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if history:
        for role, content in history:
            if role not in ("user", "assistant"):
                msg = f"history role must be 'user' or 'assistant', got {role!r}"
                raise ValueError(msg)
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    # tokenize=False is critical: the default returns token IDs (list[int]) in
    # current transformers, which breaks the downstream `tokenizer(prompts, ...)`
    # call inside the generate-fn closure (it expects list[str]). The unit tests
    # use stub tokenizers that always return strings, so the bug only surfaces
    # under a real tokenizer at run time.
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
