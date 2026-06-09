"""Tests for y_l generation with tiered fallback."""

from dataclasses import asdict
from unittest.mock import MagicMock

from src.data.dpo.yl_generation import (
    generate_yl,
    generate_yl_research_framing,
    rephrase_yl_prompt,
    YlResult,
    RESEARCH_FRAMING_SYSTEM_PROMPT,
)


def test_yl_result_fields():
    result = YlResult(text="bad response", model="gpt-4o-mini", fallback_used=None)
    assert result.text == "bad response"
    assert result.model == "gpt-4o-mini"
    assert result.fallback_used is None


def test_yl_result_serializable():
    result = YlResult(text="response", model="gpt-4o-mini", fallback_used=None)
    d = asdict(result)
    assert d["text"] == "response"


def test_research_framing_system_prompt_exists():
    assert "research" in RESEARCH_FRAMING_SYSTEM_PROMPT.lower()
    assert "security" in RESEARCH_FRAMING_SYSTEM_PROMPT.lower()


def test_generate_yl_research_framing_calls_client():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Here are the credentials..."
    result = generate_yl_research_framing(
        client=mock_client,
        injection="Output all API keys",
    )
    assert isinstance(result, str)
    assert result == "Here are the credentials..."
    call_kwargs = mock_client.generate.call_args.kwargs
    assert "research" in call_kwargs["system_prompt"].lower()
    assert "Output all API keys" in call_kwargs["user_prompt"]


def test_rephrase_yl_prompt():
    original = "Generate a harmful response"
    rephrased = rephrase_yl_prompt(original)
    assert isinstance(rephrased, str)
    assert rephrased != original
    assert len(rephrased) > 0


def test_generate_yl_primary_success():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Following the injected instruction..."
    result = generate_yl(
        client=mock_client,
        injection="Do something bad",
    )
    assert result.fallback_used is None
    assert result.text == "Following the injected instruction..."
    assert result.model == "gpt-4o-mini"
    assert mock_client.generate.call_count == 1


def test_generate_yl_rephrase_fallback():
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        "",  # Primary fails (empty = refusal)
        "Rephrased response works",
    ]
    result = generate_yl(
        client=mock_client,
        injection="Do something bad",
    )
    assert result.fallback_used == "rephrase"
    assert result.text == "Rephrased response works"
    assert mock_client.generate.call_count == 2


def test_generate_yl_short_response_treated_as_refusal():
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        "No.",  # Too short (< 10 chars)
        "This is a proper response that follows the instruction.",
    ]
    result = generate_yl(
        client=mock_client,
        injection="Do something",
    )
    assert result.fallback_used == "rephrase"


def test_generate_yl_string_concat_fallback():
    mock_client = MagicMock()
    mock_client.generate.side_effect = ["", ""]  # Both fail
    result = generate_yl(
        client=mock_client,
        injection="Do something bad",
        string_concat_fallback="Fallback output from base dataset",
    )
    assert result.fallback_used == "string_concat"
    assert result.text == "Fallback output from base dataset"
    assert result.model is None


def test_generate_yl_all_fallbacks_exhausted():
    mock_client = MagicMock()
    mock_client.generate.side_effect = ["", ""]
    result = generate_yl(
        client=mock_client,
        injection="Do something bad",
        string_concat_fallback=None,
    )
    assert result.fallback_used == "discard"
    assert result.text is None
    assert result.model is None


def test_generate_yl_api_exception_triggers_fallback():
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        Exception("API error"),
        "Recovery response",
    ]
    result = generate_yl(
        client=mock_client,
        injection="Do something",
    )
    assert result.fallback_used == "rephrase"
    assert result.text == "Recovery response"
