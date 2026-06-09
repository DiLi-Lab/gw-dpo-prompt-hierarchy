"""Tests for L4-aware partitioning and expanded L4 lookup in the SFT CLI."""

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from bin.build_sft_dataset import _build_l4_lookup


class _FakeL4Entry:
    """Minimal stand-in for L4Entry dataclass."""
    def __init__(self, source: str, index: int, l4_content: str, generation: str):
        self.source = source
        self.index = index
        self.l4_content = l4_content
        self.generation = generation


class TestBuildL4Lookup:
    """Tests for the expanded _build_l4_lookup."""

    def test_returns_dict_with_content_and_generation(self) -> None:
        entries = [_FakeL4Entry("alpaca", 0, "content0", "wrapped")]
        result = _build_l4_lookup(entries)
        assert ("alpaca", 0) in result
        assert result[("alpaca", 0)]["l4_content"] == "content0"
        assert result[("alpaca", 0)]["generation"] == "wrapped"

    def test_multiple_entries(self) -> None:
        entries = [
            _FakeL4Entry("alpaca", 0, "c0", "wrapped"),
            _FakeL4Entry("dolly", 5, "c5", "synthesized"),
        ]
        result = _build_l4_lookup(entries)
        assert len(result) == 2
        assert result[("dolly", 5)]["generation"] == "synthesized"
