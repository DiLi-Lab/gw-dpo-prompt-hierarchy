"""IHEval per-task prompt adapters.

Exposes :func:`format_record_for_format` which turns an
:class:`~src.evaluation.external.iheval.data.IHEvalRecord` into the
prompt string for the chosen format. Per-task dispatch lives in
sibling modules to keep each task family independently testable.
"""

from typing import Any

from src.evaluation.external.iheval.adapters.rule_following import (
    format_multi_turn,
    format_single_turn,
)
from src.evaluation.external.iheval.adapters.safety import format_safety
from src.evaluation.external.iheval.adapters.task_execution import format_task_execution
from src.evaluation.external.iheval.data import IHEvalRecord

_DISPATCH = {
    "single-turn":           format_single_turn,
    "multi-turn":            format_multi_turn,
    "verb-extract":          format_task_execution,
    "translation":           format_task_execution,
    "lang-detect":           format_task_execution,
    "user-prompt-hijack":    format_safety,
    "system-prompt-extract": format_safety,
}


def format_record_for_format(
    record: IHEvalRecord, *, fmt: str, tokenizer: Any,
) -> str:
    """Format one IHEvalRecord into the prompt string for ``fmt``.

    Args:
        record: The record to format.
        fmt: ``"delimited"`` or ``"chat_template"``.
        tokenizer: Required when ``fmt == "chat_template"``; otherwise
            unused (pass ``None``).
    """
    if record.task not in _DISPATCH:
        msg = f"No adapter wired for task={record.task!r}."
        raise KeyError(msg)
    return _DISPATCH[record.task](record, fmt=fmt, tokenizer=tokenizer)
