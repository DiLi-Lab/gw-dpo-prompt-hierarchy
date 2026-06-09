"""Tests for SFT dataset save/load/stats utilities."""

from pathlib import Path

import pytest

from src.data.sft.build_sft_dataset import (
    compute_sft_stats,
    load_sft_dataset,
    save_sft_dataset,
)


def _make_examples() -> list[dict]:
    """Create a small set of SFT examples for testing."""
    return [
        {
            "text": "prompt1\nresponse1",
            "levels_present": [0, 1, 2, 3],
            "is_conflict": False,
            "conflict_type": None,
            "sft_source": "alpaca",
            "sft_index": 0,
            "sft_category": "simple_aligned",
            "l4_generation": None,
        },
        {
            "text": "prompt2\nresponse2",
            "levels_present": [0, 1, 2, 3, 4],
            "is_conflict": False,
            "conflict_type": None,
            "sft_source": "dolly",
            "sft_index": 1,
            "sft_category": "simple_aligned",
            "l4_generation": "wrapped",
        },
        {
            "text": "prompt3\nresponse3",
            "levels_present": [0, 1, 3],
            "is_conflict": True,
            "conflict_type": "l0_vs_l1",
            "sft_source": None,
            "sft_index": None,
            "sft_category": "misaligned_L0_vs_L3",
            "l4_generation": None,
        },
        {
            "text": "prompt4\nresponse4",
            "levels_present": [0, 1, 3],
            "is_conflict": True,
            "conflict_type": "l0_vs_l1",
            "sft_source": None,
            "sft_index": None,
            "sft_category": "misaligned_L0_vs_L3",
            "l4_generation": None,
        },
        {
            "text": "prompt5\nresponse5",
            "levels_present": [1, 2, 3],
            "is_conflict": True,
            "conflict_type": "l1_vs_l3",
            "sft_source": "alpaca",
            "sft_index": 5,
            "sft_category": "misaligned_L1_vs_L3",
            "l4_generation": None,
        },
    ]


class TestSaveSftDataset:
    """Tests for save_sft_dataset."""

    def test_creates_jsonl_with_correct_line_count(self, tmp_path: Path) -> None:
        examples = _make_examples()
        out = tmp_path / "dataset.jsonl"
        save_sft_dataset(examples, out)

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(examples)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "dataset.jsonl"
        save_sft_dataset([{"x": 1}], out)

        assert out.exists()
        assert out.parent.is_dir()


class TestLoadSftDataset:
    """Tests for load_sft_dataset."""

    def test_round_trip(self, tmp_path: Path) -> None:
        examples = _make_examples()
        out = tmp_path / "dataset.jsonl"
        save_sft_dataset(examples, out)

        loaded = load_sft_dataset(out)
        assert loaded == examples

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            load_sft_dataset(missing)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "dataset.jsonl"
        out.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")

        loaded = load_sft_dataset(out)
        assert len(loaded) == 2
        assert loaded[0] == {"a": 1}
        assert loaded[1] == {"b": 2}


class TestComputeSftStats:
    """Tests for compute_sft_stats."""

    def test_correct_totals(self) -> None:
        examples = _make_examples()
        stats = compute_sft_stats(examples)

        assert stats["total"] == 5
        assert stats["aligned"] == 2
        assert stats["conflicting"] == 3

    def test_conflict_type_breakdown(self) -> None:
        examples = _make_examples()
        stats = compute_sft_stats(examples)

        assert stats["conflict_types"] == {
            "l0_vs_l1": 2,
            "l1_vs_l3": 1,
        }

    def test_level_configurations(self) -> None:
        examples = _make_examples()
        stats = compute_sft_stats(examples)

        assert stats["level_configurations"] == {
            str([0, 1, 2, 3]): 1,
            str([0, 1, 2, 3, 4]): 1,
            str([0, 1, 3]): 2,
            str([1, 2, 3]): 1,
        }

    def test_new_metadata_distributions(self) -> None:
        examples = _make_examples()
        stats = compute_sft_stats(examples)

        assert "sft_categories" in stats
        assert stats["sft_categories"]["simple_aligned"] == 2
        assert stats["sft_categories"]["misaligned_L0_vs_L3"] == 2
        assert stats["sft_categories"]["misaligned_L1_vs_L3"] == 1

        assert "sft_sources" in stats
        assert stats["sft_sources"]["alpaca"] == 2
        assert stats["sft_sources"]["dolly"] == 1
        assert stats["sft_sources"][None] == 2

        assert "l4_generations" in stats
        assert stats["l4_generations"]["wrapped"] == 1
        assert stats["l4_generations"][None] == 4

    def test_empty_examples(self) -> None:
        stats = compute_sft_stats([])
        assert stats["total"] == 0
        assert stats["aligned"] == 0
        assert stats["conflicting"] == 0
        assert stats["conflict_types"] == {}
        assert stats["level_configurations"] == {}
        assert stats["sft_categories"] == {}
        assert stats["sft_sources"] == {}
        assert stats["l4_generations"] == {}
