"""Tests for response delimiter wrapping in yl_rephrase.apply_rephrase()."""

from unittest.mock import MagicMock, patch

import pytest

from src.data.dpo.yl_rephrase import (
    RESP_END,
    RESP_START,
    _wrap_response,
    apply_rephrase,
)


class TestWrapResponse:
    """Tests for _wrap_response()."""

    def test_wraps_bare_text(self) -> None:
        result = _wrap_response("Hello world")
        assert result == f"{RESP_START}Hello world{RESP_END}"

    def test_idempotent_on_already_wrapped(self) -> None:
        already = f"{RESP_START}Hello world{RESP_END}"
        assert _wrap_response(already) == already

    def test_empty_string(self) -> None:
        result = _wrap_response("")
        assert result == f"{RESP_START}{RESP_END}"


class TestApplyRephraseWrapping:
    """Verify apply_rephrase wraps replacement text with response delimiters."""

    def _make_inst(self, rejected: str = "bad response") -> dict:
        return {
            "conflict_type": "L1_vs_L3",
            "prompt": "<|L1_START|>sys<|L1_END|>\n<|L3_START|>msg<|L3_END|>",
            "chosen": f"{RESP_START}good{RESP_END}",
            "rejected": f"{RESP_START}{rejected}{RESP_END}",
        }

    @patch("src.data.dpo.yl_rephrase.rephrase_yl")
    def test_yl_refusal_replacement_is_wrapped(
        self, mock_rephrase: MagicMock
    ) -> None:
        mock_rephrase.return_value = "fixed response without delimiters"
        inst = self._make_inst()
        client = MagicMock()

        fixed, label = apply_rephrase(client, inst, ["yl_refusal"])

        assert fixed is True
        assert label == "yl_refusal_rephrase"
        assert inst["rejected"].startswith(RESP_START)
        assert inst["rejected"].endswith(RESP_END)
        assert "fixed response without delimiters" in inst["rejected"]

    @patch("src.data.dpo.yl_rephrase.rephrase_yl")
    def test_yl_weak_replacement_is_wrapped(
        self, mock_rephrase: MagicMock
    ) -> None:
        mock_rephrase.return_value = "strengthened response"
        inst = self._make_inst()
        client = MagicMock()

        fixed, label = apply_rephrase(client, inst, ["yl_weak"])

        assert fixed is True
        assert inst["rejected"].startswith(RESP_START)
        assert inst["rejected"].endswith(RESP_END)

    @patch("src.data.dpo.yl_rephrase.rephrase_yw")
    def test_yw_broken_replacement_is_wrapped(
        self, mock_rephrase: MagicMock
    ) -> None:
        mock_rephrase.return_value = "fixed chosen"
        inst = self._make_inst()
        client = MagicMock()

        fixed, label = apply_rephrase(client, inst, ["yw_broken"])

        assert fixed is True
        assert inst["chosen"].startswith(RESP_START)
        assert inst["chosen"].endswith(RESP_END)

    @patch("src.data.dpo.yl_rephrase.rephrase_yl")
    def test_no_double_wrapping(self, mock_rephrase: MagicMock) -> None:
        """If rephrase already returns wrapped text, don't double-wrap."""
        mock_rephrase.return_value = f"{RESP_START}already wrapped{RESP_END}"
        inst = self._make_inst()
        client = MagicMock()

        apply_rephrase(client, inst, ["yl_refusal"])

        assert inst["rejected"].count(RESP_START) == 1
        assert inst["rejected"].count(RESP_END) == 1
