"""Unit tests for the MT-Bench metric aggregator."""

import math

from src.evaluation.external.mt_bench.metrics import aggregate_mt_bench_metrics


def _row(qid: int, turn: int, category: str, score: float | None) -> dict:
    return {
        "question_id": qid, "turn": turn, "category": category,
        "score": score,
        "parse_error": score is None,
    }


def test_two_questions_two_categories_all_scored() -> None:
    rows = [
        _row(81, 1, "writing", 8.0),
        _row(81, 2, "writing", 6.0),
        _row(111, 1, "math", 5.0),
        _row(111, 2, "math", 4.0),
    ]
    m = aggregate_mt_bench_metrics(rows)

    assert m["n_questions"] == 2
    assert m["n_turns_total"] == 4
    assert m["n_turns_scored"] == 4
    assert m["n_judge_parse_failures"] == 0

    assert m["overall_mean"] == (8.0 + 6.0 + 5.0 + 4.0) / 4
    assert m["turn1_mean"] == (8.0 + 5.0) / 2
    assert m["turn2_mean"] == (6.0 + 4.0) / 2
    assert math.isclose(
        m["turn1_minus_turn2_drop"],
        m["turn1_mean"] - m["turn2_mean"],
    )

    assert m["by_category"]["writing"]["n_turns_scored"] == 2
    assert m["by_category"]["writing"]["mean_overall"] == 7.0
    assert m["by_category"]["math"]["mean_overall"] == 4.5


def test_parse_failures_excluded_from_numerator_and_denominator() -> None:
    rows = [
        _row(81, 1, "writing", 8.0),
        _row(81, 2, "writing", None),     # parse-failed
        _row(82, 1, "writing", 6.0),
        _row(82, 2, "writing", 4.0),
    ]
    m = aggregate_mt_bench_metrics(rows)

    assert m["n_turns_scored"] == 3
    assert m["n_judge_parse_failures"] == 1
    assert m["overall_mean"] == (8.0 + 6.0 + 4.0) / 3
    assert m["turn1_mean"] == (8.0 + 6.0) / 2
    assert m["turn2_mean"] == 4.0
    assert m["parse_error_breakdown"]["by_category"] == {"writing": 1}
    assert m["parse_error_breakdown"]["by_turn"] == {"2": 1}


def test_empty_input_returns_none_means() -> None:
    m = aggregate_mt_bench_metrics([])
    assert m["n_turns_scored"] == 0
    assert m["overall_mean"] is None
    assert m["turn1_mean"] is None
    assert m["turn2_mean"] is None
    assert m["turn1_minus_turn2_drop"] is None
    assert m["by_category"] == {}


def test_refuse_everywhere_returns_none_means() -> None:
    rows = [_row(qid, t, "writing", None) for qid in (81, 82) for t in (1, 2)]
    m = aggregate_mt_bench_metrics(rows)
    assert m["n_turns_total"] == 4
    assert m["n_turns_scored"] == 0
    assert m["n_judge_parse_failures"] == 4
    assert m["overall_mean"] is None
    assert m["by_category"]["writing"]["n_turns_scored"] == 0
    assert m["by_category"]["writing"]["mean_overall"] is None


def test_single_turn_only_in_category_uses_that_turn_for_mean() -> None:
    rows = [_row(81, 1, "writing", 7.0)]
    m = aggregate_mt_bench_metrics(rows)
    assert m["turn1_mean"] == 7.0
    assert m["turn2_mean"] is None
    assert m["turn1_minus_turn2_drop"] is None  # asymmetric → undefined
    assert m["overall_mean"] == 7.0
