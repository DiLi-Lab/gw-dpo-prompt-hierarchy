"""Tests for Google/Gemini API client (Vertex AI)."""

import pytest
from unittest.mock import MagicMock, patch

from src.api.google_client import GoogleClient


def test_init_requires_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        GoogleClient()


def test_init_with_valid_project(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch("src.api.google_client.vertexai"):
        client = GoogleClient()
        assert client is not None


def test_init_calls_vertexai_init(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch("src.api.google_client.vertexai") as mock_vertexai:
        GoogleClient()
        mock_vertexai.init.assert_called_once_with(
            project="test-project", location="europe-west1"
        )


def test_generate_returns_string(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with (
        patch("src.api.google_client.vertexai"),
        patch("src.api.google_client.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        client = GoogleClient()
        result = client.generate(
            user_prompt="What is Python?",
            system_prompt="You are helpful.",
        )
        assert result == "Generated text"


def test_generate_default_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with (
        patch("src.api.google_client.vertexai"),
        patch("src.api.google_client.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        client = GoogleClient()
        client.generate(user_prompt="test")
        mock_model_cls.assert_called_once()
        call_args = mock_model_cls.call_args
        assert "gemini" in call_args.args[0].lower()


def test_generate_with_custom_params(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with (
        patch("src.api.google_client.vertexai"),
        patch("src.api.google_client.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        client = GoogleClient()
        client.generate(
            user_prompt="test",
            model="gemini-2.5-pro",
            temperature=0.0,
            max_tokens=500,
        )
        gen_config = mock_model.generate_content.call_args.kwargs.get(
            "generation_config", None
        )
        assert gen_config is not None


def test_generate_retries_on_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with (
        patch("src.api.google_client.vertexai"),
        patch("src.api.google_client.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            Exception("API error"),
            MagicMock(text="recovered"),
        ]
        mock_model_cls.return_value = mock_model

        client = GoogleClient()
        with patch("time.sleep"):
            result = client.generate(user_prompt="test")
        assert result == "recovered"
