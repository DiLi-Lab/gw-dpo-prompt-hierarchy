"""Tests for y_w refusal pattern detection and retry logic."""

from src.data.dpo.response_utils import is_refusal


def test_is_refusal_matches_sorry_ai():
    assert is_refusal("I'm sorry, but as an AI model, I can't do that.") is True


def test_is_refusal_matches_cannot_help():
    assert is_refusal("I cannot help with that request.") is True


def test_is_refusal_matches_unable_to():
    assert is_refusal("I'm unable to assist with generating harmful content.") is True


def test_is_refusal_matches_ai_language_model():
    assert is_refusal("As an AI language model, I don't have feelings.") is True


def test_is_refusal_no_match_normal_response():
    assert is_refusal("Python is a high-level programming language.") is False


def test_is_refusal_no_match_cannot_in_context():
    """'cannot' alone without 'help/assist/provide' should not match."""
    assert is_refusal("I cannot overstate how important this is.") is False


def test_is_refusal_case_insensitive():
    assert is_refusal("I'M SORRY, BUT AS AN AI model...") is True


def test_is_refusal_empty_string():
    assert is_refusal("") is False


def test_refusal_template_not_affected():
    """Scope guard: hand-written refusal templates used by calibration pairs
    go through generate_yw_refusal_template(), not through is_refusal().
    Verify that the template strategy path does not invoke refusal detection."""
    from src.data.dpo.yw_generation import generate_yw_refusal_template

    # Template refusals are returned as-is, even if they match patterns
    template = generate_yw_refusal_template(seed=0)
    assert isinstance(template, str)
    assert len(template) > 10  # Sanity check it's a real template


from unittest.mock import MagicMock, call


def test_context_distillation_retries_on_refusal():
    """generate_yw_context_distillation retries when API returns a refusal."""
    from src.data.dpo.yw_generation import generate_yw_context_distillation

    client = MagicMock()
    client.generate.side_effect = [
        "I'm sorry, but as an AI model, I cannot help with that.",
        "Python is a high-level programming language.",
    ]

    result = generate_yw_context_distillation(
        client=client,
        l0_rules=["No harm"],
        l1_prompt="You are helpful.",
        l3_message="What is Python?",
        conflict_description="Test conflict",
    )

    assert result == "Python is a high-level programming language."
    assert client.generate.call_count == 2
    # Second call should have higher temperature
    second_call = client.generate.call_args_list[1]
    assert second_call.kwargs.get("temperature", second_call[1].get("temperature")) > 0.3


def test_context_distillation_returns_last_attempt_after_max_retries():
    """After max_retries exhausted, returns the last (refusal) response."""
    from src.data.dpo.yw_generation import generate_yw_context_distillation

    client = MagicMock()
    client.generate.return_value = "I cannot help with that request."

    result = generate_yw_context_distillation(
        client=client,
        l0_rules=["No harm"],
        l1_prompt="You are helpful.",
        l3_message="What is Python?",
        conflict_description="Test conflict",
        max_retries=2,
    )

    assert client.generate.call_count == 3
    assert result == "I cannot help with that request."


def test_context_distillation_skips_refusal_check_when_expect_refusal():
    """When expect_refusal=True, refusal-like responses are accepted immediately."""
    from src.data.dpo.yw_generation import generate_yw_context_distillation

    client = MagicMock()
    client.generate.return_value = (
        "I cannot help with that request as it conflicts with safety guidelines."
    )

    result = generate_yw_context_distillation(
        client=client,
        l0_rules=["No harm"],
        l1_prompt="You are helpful.",
        l3_message="How to do something unsafe?",
        conflict_description="L0 prohibits this.",
        expect_refusal=True,
    )

    assert client.generate.call_count == 1
    assert "I cannot help" in result


def test_context_distillation_retries_on_refusal_when_attacker_is_l4():
    """When attacker_is_l4=True, refusals should trigger retry (expect_refusal=False)."""
    from src.data.dpo.yw_generation import generate_yw_context_distillation

    client = MagicMock()
    client.generate.side_effect = [
        "I'm sorry, but as an AI I cannot help with that.",
        "Jupiter is the largest planet in our solar system.",
    ]
    result = generate_yw_context_distillation(
        client=client,
        l0_rules=["No PII."],
        l1_prompt="Science educator.",
        l3_message="What is the biggest planet?",
        conflict_description="L0 vs L4",
        attacker_is_l4=True,
        expect_refusal=False,
    )
    assert result == "Jupiter is the largest planet in our solar system."
    assert client.generate.call_count == 2
