"""MT-Bench multi-turn chat-template prompt builders.

Mirrors the upstream FastChat protocol: no system prompt, just a
user message for turn 1, then user/assistant/user for turn 2. Reuses
:func:`src.evaluation.external.prompt_formats.build_chat_template`,
which threads ``add_generation_prompt=True`` and ``tokenize=False``
through ``apply_chat_template``.
"""

from typing import Any

from src.evaluation.external.mt_bench.data import MTBenchQuestion
from src.evaluation.external.prompt_formats import build_chat_template


def build_turn1_prompt(tokenizer: Any, question: MTBenchQuestion) -> str:
    """Render turn 1 as a user-only chat template (no system prompt)."""
    return build_chat_template(tokenizer, user=question.turns[0])


def build_turn2_prompt(
    tokenizer: Any,
    question: MTBenchQuestion,
    turn1_response: str,
) -> str:
    """Render turn 2 with turn 1 user/assistant in the history.

    Layout: user(q.turns[0]) -> assistant(turn1_response) -> user(q.turns[1]).
    No system prompt, matching upstream MT-Bench.
    """
    return build_chat_template(
        tokenizer,
        user=question.turns[1],
        history=[
            ("user",      question.turns[0]),
            ("assistant", turn1_response),
        ],
    )
