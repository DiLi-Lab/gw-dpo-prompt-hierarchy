"""Adapters for the IHEval task-execution tasks."""

from typing import Any

from src.evaluation.external.iheval.data import IHEvalRecord
from src.evaluation.external.prompt_formats import (
    build_chat_template,
    build_delimited,
)


def format_task_execution(
    record: IHEvalRecord, *, fmt: str, tokenizer: Any,
) -> str:
    """verb-extract / translation / lang-detect: system + instruction only."""
    if fmt == "delimited":
        return build_delimited(l1=record.system, l3=record.instruction) + "\n<|RESP_START|>"
    return build_chat_template(
        tokenizer, system=record.system, user=record.instruction,
    )
