"""MT-Bench judge prompt builder + score parser.

The judge is called twice per question (turn 1, turn 2). Math /
coding / reasoning use the math-variant template with the gpt-4
reference answer threaded through the ``{ref_answer_*}`` placeholder.
Score parsing mirrors upstream FastChat: ``[[N]]`` primary,
``[N]`` fallback, integers and decimals both accepted, out-of-range
treated as parse failure.
"""

import re
from dataclasses import dataclass
from typing import Literal

from src.evaluation.external.mt_bench.data import (
    MATH_CATEGORIES,
    JudgePromptTemplate,
    MTBenchQuestion,
    ReferenceAnswer,
)

_PRIMARY_RE = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")
_FALLBACK_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")


@dataclass(frozen=True)
class JudgeCall:
    """A fully-rendered judge call ready for the OpenAI client."""

    template_name: str
    system_prompt: str
    user_prompt: str


def parse_score(judge_text: str) -> float | None:
    """Parse a 1-10 score from a judge response. None on parse failure.

    Tries ``[[N]]`` first, then the looser ``[N]`` fallback. Out-of-range
    values (``[[0]]``, ``[[12]]``) are treated as parse failures, not
    silently clipped — better that a downstream aggregator counts the
    parse failure than that a clipped 1.0 / 10.0 distort the mean.
    """
    m = _PRIMARY_RE.search(judge_text) or _FALLBACK_RE.search(judge_text)
    if not m:
        return None
    score = float(m.group(1))
    return score if 1.0 <= score <= 10.0 else None


def _select_template(category: str, turn: int) -> str:
    is_math = category in MATH_CATEGORIES
    if turn == 1:
        return "single-math-v1" if is_math else "single-v1"
    if turn == 2:
        return "single-math-v1-multi-turn" if is_math else "single-v1-multi-turn"
    msg = f"Unknown turn={turn}; expected 1 or 2"
    raise ValueError(msg)


def build_judge_prompt(
    *,
    templates: dict[str, JudgePromptTemplate],
    question: MTBenchQuestion,
    reference: ReferenceAnswer | None,
    turn: Literal[1, 2],
    responses: tuple[str, str],
) -> JudgeCall:
    """Construct the judge call for one (question, turn).

    Args:
        templates: The 4 upstream judge templates loaded by
            :func:`load_judge_prompts`.
        question: The MT-Bench question.
        reference: gpt-4 reference for math / coding / reasoning. Must
            be present when the category is in MATH_CATEGORIES.
        turn: 1 or 2. Determines whether the multi-turn template variant
            and the second reference turn are used.
        responses: ``(turn1_response, turn2_response)``. Turn 2 entry
            is unused when ``turn == 1`` but still required so callers
            cannot accidentally drop it.

    Returns:
        Rendered :class:`JudgeCall` ready to pass to the OpenAI client.
    """
    template_name = _select_template(question.category, turn)
    template = templates[template_name]
    is_math = question.category in MATH_CATEGORIES
    if is_math and reference is None:
        msg = (
            f"question_id={question.question_id} category={question.category} "
            "requires a reference answer; got None."
        )
        raise ValueError(msg)

    fmt: dict[str, str] = {}
    if turn == 1:
        fmt["question"] = question.turns[0]
        fmt["answer"] = responses[0]
        if is_math:
            fmt["ref_answer_1"] = reference.turns[0]  # type: ignore[union-attr]
    else:
        fmt["question_1"] = question.turns[0]
        fmt["question_2"] = question.turns[1]
        fmt["answer_1"] = responses[0]
        fmt["answer_2"] = responses[1]
        if is_math:
            fmt["ref_answer_1"] = reference.turns[0]  # type: ignore[union-attr]
            fmt["ref_answer_2"] = reference.turns[1]  # type: ignore[union-attr]

    user_prompt = template.prompt_template.format(**fmt)
    return JudgeCall(
        template_name=template_name,
        system_prompt=template.system_prompt,
        user_prompt=user_prompt,
    )
