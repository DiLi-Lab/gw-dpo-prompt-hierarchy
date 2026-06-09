"""Adapters for the IHEval rule-following tasks (single-turn, multi-turn)."""

from typing import Any

from src.evaluation.external.iheval.data import IHEvalRecord
from src.evaluation.external.prompt_formats import (
    build_chat_template,
    build_delimited,
)


def _history_pairs(history: list[str] | None) -> list[tuple[str, str]]:
    """Convert IHEval's flat conversation_history list into (role, content) pairs."""
    if not history:
        return []
    pairs: list[tuple[str, str]] = []
    for i, content in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        pairs.append((role, content))
    return pairs


def _history_as_text(history: list[str] | None) -> str | None:
    """Render conversation_history as a plain-text USER/ASSISTANT block."""
    if not history:
        return None
    lines = []
    for i, content in enumerate(history):
        role = "USER" if i % 2 == 0 else "ASSISTANT"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_single_turn(
    record: IHEvalRecord, *, fmt: str, tokenizer: Any,
) -> str:
    if fmt == "delimited":
        return build_delimited(l1=record.system, l3=record.instruction) + "\n<|RESP_START|>"
    return build_chat_template(
        tokenizer, system=record.system, user=record.instruction,
    )


def format_multi_turn(
    record: IHEvalRecord, *, fmt: str, tokenizer: Any,
) -> str:
    if fmt == "delimited":
        return build_delimited(
            l1=record.system,
            l3=record.instruction,
            l4=_history_as_text(record.conversation_history),
        ) + "\n<|RESP_START|>"
    return build_chat_template(
        tokenizer,
        system=record.system,
        history=_history_pairs(record.conversation_history),
        user=record.instruction,
    )
