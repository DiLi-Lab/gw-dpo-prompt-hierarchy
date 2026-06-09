"""Tests for response-aware L2 generation."""

from unittest.mock import MagicMock

import pytest

from src.data.libraries.l2_generator import (
    FALLBACK_L2,
    L2_SYSTEM_PROMPT,
    MISALIGNED_L2_REDIRECT,
    MISALIGNED_L2_REFUSAL,
    generate_l2_from_response,
    validate_l2,
)


class TestConstants:
    """Tests for L2 generator constants."""

    def test_refusal_contains_english(self) -> None:
        assert "English" in MISALIGNED_L2_REFUSAL

    def test_redirect_contains_english(self) -> None:
        assert "English" in MISALIGNED_L2_REDIRECT

    def test_fallback_contains_english(self) -> None:
        assert "English" in FALLBACK_L2

    def test_fallback_does_not_contain_default(self) -> None:
        assert "default" not in FALLBACK_L2.lower()

    def test_refusal_is_single_line(self) -> None:
        assert "\n" not in MISALIGNED_L2_REFUSAL

    def test_redirect_is_single_line(self) -> None:
        assert "\n" not in MISALIGNED_L2_REDIRECT

    def test_system_prompt_mentions_english(self) -> None:
        assert "English" in L2_SYSTEM_PROMPT


class TestValidateL2:
    """Tests for L2 output validation."""

    def test_valid_string(self) -> None:
        assert validate_l2("Session config: Respond in English. Tone: casual.") is True

    def test_rejects_multiline(self) -> None:
        assert validate_l2("Line 1\nLine 2") is False

    def test_rejects_too_long(self) -> None:
        assert validate_l2("x" * 121) is False

    def test_accepts_without_english_keyword(self) -> None:
        assert validate_l2("Session config: Tone: casual. Length: brief.") is True

    def test_rejects_non_english_language(self) -> None:
        assert validate_l2("Session config: Respond in French. Tone: casual.") is False

    def test_accepts_120_chars(self) -> None:
        base = "Session config: Respond in English. Tone: casual."
        s = base + "x" * (120 - len(base))
        assert len(s) == 120
        assert validate_l2(s) is True


class TestGenerateL2FromResponse:
    """Tests for generate_l2_from_response with mocked client."""

    def _mock_client(self, responses: list[str]) -> MagicMock:
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)
        return client

    def test_returns_valid_l2(self) -> None:
        client = self._mock_client(
            ["Session config: Respond in English. Tone: casual. Length: brief."]
        )
        result = generate_l2_from_response(
            client, l1_prompt="Be helpful.", l3_message="Hi", response="Hello!",
        )
        assert "English" in result
        assert "\n" not in result

    def test_calls_client_generate(self) -> None:
        client = self._mock_client(
            ["Preferences: Language=English, Format=plain text."]
        )
        generate_l2_from_response(
            client, l1_prompt="Sys", l3_message="Msg", response="Resp",
        )
        client.generate.assert_called_once()
        call_kwargs = client.generate.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 60

    def test_truncates_response_to_500_chars(self) -> None:
        long_response = "x" * 1000
        client = self._mock_client(
            ["Session config: Respond in English. Tone: professional."]
        )
        generate_l2_from_response(
            client, l1_prompt="Sys", l3_message="Msg", response=long_response,
        )
        user_prompt = client.generate.call_args.kwargs["user_prompt"]
        assert "x" * 1000 not in user_prompt
        assert "x" * 500 in user_prompt

    def test_retries_once_on_invalid_output(self) -> None:
        client = self._mock_client([
            "This is invalid\nhas newlines",
            "Session config: Respond in English. Tone: casual.",
        ])
        result = generate_l2_from_response(
            client, l1_prompt="Sys", l3_message="Msg", response="Resp",
        )
        assert client.generate.call_count == 2
        assert "English" in result

    def test_falls_back_after_two_failures(self) -> None:
        client = self._mock_client([
            "invalid\nmultiline",
            "still invalid\nmore lines",
        ])
        result = generate_l2_from_response(
            client, l1_prompt="Sys", l3_message="Msg", response="Resp",
        )
        assert result == FALLBACK_L2

    def test_falls_back_on_exception(self) -> None:
        client = MagicMock()
        client.generate = MagicMock(side_effect=Exception("API error"))
        result = generate_l2_from_response(
            client, l1_prompt="Sys", l3_message="Msg", response="Resp",
        )
        assert result == FALLBACK_L2
        assert client.generate.call_count == 2  # retries once before fallback
