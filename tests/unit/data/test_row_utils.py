"""Tests for shared row accessor helpers (get_output, get_input)."""

import pytest

from src.data.sft.row_utils import get_input, get_output


class TestGetOutput:
    """Tests for get_output()."""

    def test_alpaca_schema(self) -> None:
        row = {"output": "alpaca answer", "input": "some input"}
        assert get_output(row) == "alpaca answer"

    def test_dolly_schema(self) -> None:
        row = {"response": "dolly answer", "context": "some context"}
        assert get_output(row) == "dolly answer"

    def test_alpaca_preferred_over_dolly(self) -> None:
        row = {"output": "alpaca answer", "response": "dolly answer"}
        assert get_output(row) == "alpaca answer"

    def test_empty_output_falls_back_to_response(self) -> None:
        row = {"output": "", "response": "dolly answer"}
        assert get_output(row) == "dolly answer"

    def test_neither_field_present(self) -> None:
        row = {"instruction": "do something"}
        assert get_output(row) == ""


class TestGetInput:
    """Tests for get_input()."""

    def test_alpaca_schema(self) -> None:
        row = {"input": "alpaca input", "output": "answer"}
        assert get_input(row) == "alpaca input"

    def test_dolly_schema(self) -> None:
        row = {"context": "dolly context", "response": "answer"}
        assert get_input(row) == "dolly context"

    def test_alpaca_preferred_over_dolly(self) -> None:
        row = {"input": "alpaca input", "context": "dolly context"}
        assert get_input(row) == "alpaca input"

    def test_empty_input_falls_back_to_context(self) -> None:
        row = {"input": "", "context": "dolly context"}
        assert get_input(row) == "dolly context"

    def test_neither_field_present(self) -> None:
        row = {"instruction": "do something"}
        assert get_input(row) == ""
