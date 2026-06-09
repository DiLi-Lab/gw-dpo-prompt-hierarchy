"""Tests for Anthropic API client wrapper."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.api.anthropic_client import AnthropicClient


def test_init_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicClient()


def test_init_from_env():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
            client = AnthropicClient()
            mock_cls.assert_called_once_with(api_key="test-key")


def test_init_explicit_key():
    with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
        client = AnthropicClient(api_key="explicit-key")
        mock_cls.assert_called_once_with(api_key="explicit-key")


def test_generate_returns_text():
    with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated text")]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(api_key="test")
        result = client.generate(
            user_prompt="Hello",
            system_prompt="Be helpful",
        )
        assert result == "Generated text"


def test_generate_retries_on_rate_limit():
    with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        import anthropic as anthropic_mod
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Success after retry")]
        mock_client.messages.create.side_effect = [
            anthropic_mod.RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None),
            mock_response,
        ]

        with patch("src.api.anthropic_client.time.sleep"):
            client = AnthropicClient(api_key="test")
            result = client.generate(user_prompt="Hello")
            assert result == "Success after retry"
            assert mock_client.messages.create.call_count == 2


def test_generate_raises_after_max_retries():
    with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        import anthropic as anthropic_mod
        mock_client.messages.create.side_effect = anthropic_mod.RateLimitError(
            message="rate limited", response=MagicMock(status_code=429), body=None,
        )

        with patch("src.api.anthropic_client.time.sleep"):
            client = AnthropicClient(api_key="test")
            with pytest.raises(anthropic_mod.RateLimitError):
                client.generate(user_prompt="Hello")
            assert mock_client.messages.create.call_count == 3


def test_generate_batch_returns_list():
    with patch("src.api.anthropic_client.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(api_key="test")
        results = client.generate_batch(
            user_prompts=["q1", "q2", "q3"],
            system_prompt="Be helpful",
        )
        assert len(results) == 3
        assert all(r == "Response" for r in results)
