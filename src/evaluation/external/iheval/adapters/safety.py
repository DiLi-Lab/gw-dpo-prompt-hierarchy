"""Adapters for the IHEval safety tasks (TensorTrust-derived)."""

from typing import Any

from src.evaluation.external.iheval.data import IHEvalRecord
from src.evaluation.external.prompt_formats import (
    build_chat_template,
    build_delimited,
)


def format_safety(
    record: IHEvalRecord, *, fmt: str, tokenizer: Any,
) -> str:
    """user-prompt-hijack / system-prompt-extract: system + instruction."""
    if fmt == "delimited":
        return build_delimited(l1=record.system, l3=record.instruction) + "\n<|RESP_START|>"
    return build_chat_template(
        tokenizer, system=record.system, user=record.instruction,
    )
