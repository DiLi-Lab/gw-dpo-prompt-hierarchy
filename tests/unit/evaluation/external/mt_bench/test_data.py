"""Unit tests for MT-Bench data loaders."""

import json
from pathlib import Path

import pytest

from src.evaluation.external.mt_bench.data import (
    MATH_CATEGORIES,
    JudgePromptTemplate,
    MTBenchQuestion,
    ReferenceAnswer,
    load_judge_prompts,
    load_questions,
    load_reference_answers,
)

_FIXTURE_DIR = Path("tests/fixtures/mt_bench")


def test_load_questions_fixture_returns_four_records() -> None:
    qs = load_questions(_FIXTURE_DIR / "question_subset.jsonl", expect_count=None)
    assert len(qs) == 4
    assert all(isinstance(q, MTBenchQuestion) for q in qs)
    assert all(len(q.turns) == 2 for q in qs)
    cats = {q.category for q in qs}
    assert cats == {"writing", "roleplay", "math", "coding"}


def test_load_questions_real_file_has_80_records() -> None:
    qs = load_questions(Path("data/external/mt_bench/question.jsonl"))
    assert len(qs) == 80
    cats: dict[str, int] = {}
    for q in qs:
        cats[q.category] = cats.get(q.category, 0) + 1
    assert cats == {
        "writing": 10, "roleplay": 10, "extraction": 10, "reasoning": 10,
        "math": 10, "coding": 10, "stem": 10, "humanities": 10,
    }


def test_load_reference_answers_fixture_covers_math_and_coding() -> None:
    refs = load_reference_answers(_FIXTURE_DIR / "reference_answer_subset.jsonl")
    assert set(refs.keys()) == {111, 121}
    for r in refs.values():
        assert isinstance(r, ReferenceAnswer)
        assert len(r.turns) == 2


def test_load_reference_answers_real_file_covers_30_questions() -> None:
    refs = load_reference_answers(
        Path("data/external/mt_bench/reference_answer_gpt4.jsonl"),
    )
    assert len(refs) == 30
    qs = load_questions(Path("data/external/mt_bench/question.jsonl"))
    math_qids = {q.question_id for q in qs if q.category in MATH_CATEGORIES}
    assert math_qids == set(refs.keys())


def test_load_judge_prompts_fixture_has_all_four_templates() -> None:
    tpls = load_judge_prompts(_FIXTURE_DIR / "judge_prompts_subset.jsonl")
    assert set(tpls.keys()) == {
        "single-v1",
        "single-math-v1",
        "single-v1-multi-turn",
        "single-math-v1-multi-turn",
    }
    for t in tpls.values():
        assert isinstance(t, JudgePromptTemplate)
        assert t.system_prompt
        assert t.prompt_template


def test_load_judge_prompts_drops_pair_mode_rows() -> None:
    """The fixture mirrors upstream: 4 pair-mode + 4 single-mode rows.
    The loader must filter to single-mode only.
    """
    tpls = load_judge_prompts(_FIXTURE_DIR / "judge_prompts_subset.jsonl")
    assert all(not name.startswith("pair") for name in tpls)
    assert len(tpls) == 4


def test_load_questions_raises_on_wrong_record_count(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(
        {"question_id": 81, "category": "writing",
         "turns": ["t1", "t2"]},
    ) + "\n")
    with pytest.raises(ValueError, match="expected 80"):
        load_questions(bad)


def test_load_judge_prompts_raises_on_missing_template(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({
        "name": "single-v1", "type": "single",
        "system_prompt": "x", "prompt_template": "y",
    }) + "\n")
    with pytest.raises(ValueError, match="missing.*templates"):
        load_judge_prompts(bad)
