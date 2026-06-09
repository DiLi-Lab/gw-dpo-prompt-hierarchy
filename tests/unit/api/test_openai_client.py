"""Tests for OpenAI API client wrapper."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.api.openai_client import OpenAIClient


def test_init_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIClient()


def test_init_from_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
            client = OpenAIClient()
            mock_cls.assert_called_once_with(api_key="test-key")


def test_generate_returns_text():
    with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="test")
        result = client.generate(
            user_prompt="Hello",
            system_prompt="Be helpful",
        )
        assert result == "Generated text"


def test_generate_json_mode():
    with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="test")
        result = client.generate(
            user_prompt="Hello",
            system_prompt="Return JSON",
            json_mode=True,
        )
        assert result == '{"key": "value"}'
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"


def test_generate_retries_on_rate_limit():
    with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        import openai as openai_mod
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success after retry"))]
        mock_client.chat.completions.create.side_effect = [
            openai_mod.RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None),
            mock_response,
        ]

        with patch("src.api.openai_client.time.sleep"):
            client = OpenAIClient(api_key="test")
            result = client.generate(user_prompt="Hello")
            assert result == "Success after retry"
            assert mock_client.chat.completions.create.call_count == 2


def test_generate_raises_after_max_retries():
    with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        import openai as openai_mod
        mock_client.chat.completions.create.side_effect = openai_mod.RateLimitError(
            message="rate limited", response=MagicMock(status_code=429), body=None,
        )

        with patch("src.api.openai_client.time.sleep"):
            client = OpenAIClient(api_key="test")
            with pytest.raises(openai_mod.RateLimitError):
                client.generate(user_prompt="Hello")
            assert mock_client.chat.completions.create.call_count == 3


def test_generate_batch_returns_list():
    with patch("src.api.openai_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="test")
        results = client.generate_batch(
            user_prompts=["q1", "q2"],
            system_prompt="Be helpful",
        )
        assert len(results) == 2
