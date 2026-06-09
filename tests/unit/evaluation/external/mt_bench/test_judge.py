"""Unit tests for the MT-Bench judge prompt builder + score parser."""

import pytest

from src.evaluation.external.mt_bench.data import (
    JudgePromptTemplate,
    MTBenchQuestion,
    ReferenceAnswer,
)
from src.evaluation.external.mt_bench.judge import (
    build_judge_prompt,
    parse_score,
)


# --- parse_score ---------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Some reasoning... [[7]]", 7.0),
    ("Reasoning. Score: [[5]]", 5.0),
    ("Decimal scores: [[7.5]]", 7.5),
    ("Fallback bracket [3]", 3.0),
    ("No score here at all.", None),
    ("Out of range [[12]]", None),
    ("Out of range [[0]]", None),
    ("Both formats [[8]] and [3]", 8.0),  # primary wins
])
def test_parse_score_table(text: str, expected: float | None) -> None:
    assert parse_score(text) == expected


# --- build_judge_prompt --------------------------------------------------


def _writing_q() -> MTBenchQuestion:
    return MTBenchQuestion(
        question_id=81, category="writing",
        turns=("Q1 user msg", "Q2 user msg"),
    )


def _math_q() -> MTBenchQuestion:
    return MTBenchQuestion(
        question_id=111, category="math",
        turns=("Math Q1", "Math Q2"),
    )


def _math_ref() -> ReferenceAnswer:
    return ReferenceAnswer(
        question_id=111, turns=("Ref answer 1", "Ref answer 2"),
    )


def _templates() -> dict[str, JudgePromptTemplate]:
    # Use placeholder-bearing templates that match upstream's substitution
    # surface. The real templates carry more text but the placeholders
    # are the contract.
    return {
        "single-v1": JudgePromptTemplate(
            name="single-v1",
            system_prompt="judge sys v1",
            prompt_template=(
                "[Question]\n{question}\n\n"
                "[Assistant Answer]\n{answer}\n"
            ),
        ),
        "single-math-v1": JudgePromptTemplate(
            name="single-math-v1",
            system_prompt="judge sys math v1",
            prompt_template=(
                "[Reference]\n{ref_answer_1}\n\n"
                "[Question]\n{question}\n\n"
                "[Assistant Answer]\n{answer}\n"
            ),
        ),
        "single-v1-multi-turn": JudgePromptTemplate(
            name="single-v1-multi-turn",
            system_prompt="judge sys mt",
            prompt_template=(
                "<|First user|>\n{question_1}\n"
                "<|First assistant|>\n{answer_1}\n"
                "<|Second user|>\n{question_2}\n"
                "<|Second assistant|>\n{answer_2}\n"
            ),
        ),
        "single-math-v1-multi-turn": JudgePromptTemplate(
            name="single-math-v1-multi-turn",
            system_prompt="judge sys math mt",
            prompt_template=(
                "<|Ref 1|>\n{ref_answer_1}\n"
                "<|Ref 2|>\n{ref_answer_2}\n"
                "<|First user|>\n{question_1}\n"
                "<|First assistant|>\n{answer_1}\n"
                "<|Second user|>\n{question_2}\n"
                "<|Second assistant|>\n{answer_2}\n"
            ),
        ),
    }


def test_turn1_writing_picks_single_v1() -> None:
    out = build_judge_prompt(
        templates=_templates(), question=_writing_q(),
        reference=None, turn=1,
        responses=("ans 1", "ans 2"),
    )
    assert out.template_name == "single-v1"
    assert "Q1 user msg" in out.user_prompt
    assert "ans 1" in out.user_prompt
    assert "ans 2" not in out.user_prompt
    assert out.system_prompt == "judge sys v1"


def test_turn2_writing_picks_single_v1_multi_turn() -> None:
    out = build_judge_prompt(
        templates=_templates(), question=_writing_q(),
        reference=None, turn=2,
        responses=("ans 1", "ans 2"),
    )
    assert out.template_name == "single-v1-multi-turn"
    assert "Q1 user msg" in out.user_prompt
    assert "Q2 user msg" in out.user_prompt
    assert "ans 1" in out.user_prompt
    assert "ans 2" in out.user_prompt


def test_turn1_math_picks_single_math_v1_with_ref() -> None:
    out = build_judge_prompt(
        templates=_templates(), question=_math_q(),
        reference=_math_ref(), turn=1,
        responses=("ans 1", "ans 2"),
    )
    assert out.template_name == "single-math-v1"
    assert "Ref answer 1" in out.user_prompt
    assert "Ref answer 2" not in out.user_prompt  # turn 1 only uses ref 1
    assert "Math Q1" in out.user_prompt


def test_turn2_math_picks_single_math_v1_multi_turn_with_both_refs() -> None:
    out = build_judge_prompt(
        templates=_templates(), question=_math_q(),
        reference=_math_ref(), turn=2,
        responses=("ans 1", "ans 2"),
    )
    assert out.template_name == "single-math-v1-multi-turn"
    assert "Ref answer 1" in out.user_prompt
    assert "Ref answer 2" in out.user_prompt
    assert "Math Q1" in out.user_prompt
    assert "Math Q2" in out.user_prompt


def test_math_question_without_reference_raises() -> None:
    with pytest.raises(ValueError, match="reference"):
        build_judge_prompt(
            templates=_templates(), question=_math_q(),
            reference=None, turn=1,
            responses=("ans 1", "ans 2"),
        )


def test_unknown_turn_raises() -> None:
    with pytest.raises(ValueError, match="turn"):
        build_judge_prompt(
            templates=_templates(), question=_writing_q(),
            reference=None, turn=3,
            responses=("ans 1", "ans 2"),
        )
