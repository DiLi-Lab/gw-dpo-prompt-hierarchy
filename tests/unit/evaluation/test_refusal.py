"""Tests for the two-stage refusal classifier."""

from src.evaluation.refusal import (
    classify_refusal,
    needs_judge_confirmation,
    regex_refusal_match,
)


def test_short_clean_refusal_classified_as_refusal() -> None:
    """Short response with refusal phrase is unambiguously a refusal."""
    response = "I'm sorry, but I cannot help with that."
    assert regex_refusal_match(response) is True
    assert needs_judge_confirmation(response, min_chars=200) is False
    out = classify_refusal(response, min_chars=200, judge_fn=None)
    assert out["is_refusal"] is True
    assert out["judge_used"] is False


def test_compliance_without_refusal_phrase_classified_as_not_refusal() -> None:
    response = "Here is the answer to your question. The capital of France is Paris."
    assert regex_refusal_match(response) is False
    out = classify_refusal(response, min_chars=200, judge_fn=None)
    assert out["is_refusal"] is False
    assert out["judge_used"] is False


def test_long_response_with_refusal_phrase_triggers_judge() -> None:
    response = (
        "I'm sorry, but I can't help directly. "
        + "However, here's some general information that may be useful: " * 10
    )
    assert regex_refusal_match(response) is True
    assert needs_judge_confirmation(response, min_chars=200) is True

    judge_calls: list[str] = []

    def fake_judge(text: str) -> bool:
        judge_calls.append(text)
        return False  # judge says: complies after the hedge

    out = classify_refusal(response, min_chars=200, judge_fn=fake_judge)
    assert out["is_refusal"] is False
    assert out["judge_used"] is True
    assert len(judge_calls) == 1


def test_judge_can_confirm_refusal() -> None:
    response = (
        "I'm sorry, but I cannot assist with that request. "
        + "I would advise you to consult a professional. " * 10
    )

    def fake_judge(text: str) -> bool:
        return True  # judge confirms refusal

    out = classify_refusal(response, min_chars=200, judge_fn=fake_judge)
    assert out["is_refusal"] is True
    assert out["judge_used"] is True


def test_judge_only_called_when_ambiguous() -> None:
    """Judge must NOT be called for short clean refusals or non-matches."""

    def explode(_text: str) -> bool:
        raise AssertionError("judge should not be invoked")

    classify_refusal("I'm sorry, but I cannot help.", min_chars=200, judge_fn=explode)
    classify_refusal("Sure! Here's the answer.", min_chars=200, judge_fn=explode)
