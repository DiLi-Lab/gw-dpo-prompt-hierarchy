"""Unit tests for the MT-Bench per-question scorer."""

from src.evaluation.external.mt_bench.scorer import (
    MTBenchQuestionScore,
    build_question_score,
)


def test_both_turns_scored_produces_mean() -> None:
    s = build_question_score(
        question_id=42, category="writing",
        turn1_score=8.0, turn2_score=6.0,
    )
    assert s == MTBenchQuestionScore(
        question_id=42, category="writing",
        turn1_score=8.0, turn2_score=6.0, mean_score=7.0,
    )


def test_turn1_parse_failure_yields_none_mean() -> None:
    s = build_question_score(
        question_id=43, category="math",
        turn1_score=None, turn2_score=5.0,
    )
    assert s.mean_score is None
    assert s.turn1_score is None
    assert s.turn2_score == 5.0


def test_turn2_parse_failure_yields_none_mean() -> None:
    s = build_question_score(
        question_id=44, category="coding",
        turn1_score=7.0, turn2_score=None,
    )
    assert s.mean_score is None


def test_both_turns_failed_yields_none_mean_and_scores() -> None:
    s = build_question_score(
        question_id=45, category="reasoning",
        turn1_score=None, turn2_score=None,
    )
    assert s.mean_score is None
    assert s.turn1_score is None
    assert s.turn2_score is None
