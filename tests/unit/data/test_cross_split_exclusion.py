"""Tests for cross-split instance exclusion."""

import json
from pathlib import Path

from src.data.dpo.build_dpo_dataset import collect_used_base_keys


def test_collect_used_base_keys_from_sft(tmp_path: Path):
    """Collects (sft_source, sft_index) keys from SFT output."""
    sft_file = tmp_path / "sft_combined.jsonl"
    sft_file.write_text(
        json.dumps({"sft_source": "alpaca", "sft_index": 0, "text": "..."}) + "\n"
        + json.dumps({"sft_source": "dolly", "sft_index": 5, "text": "..."}) + "\n"
    )
    keys = collect_used_base_keys(sft_file)
    assert keys == {("alpaca", 0), ("dolly", 5)}


def test_collect_used_base_keys_from_dpo(tmp_path: Path):
    """Collects (yw_base_dataset, yw_base_index) and (yl_base_*) from DPO output."""
    dpo_file = tmp_path / "phase1.jsonl"
    dpo_file.write_text(
        json.dumps({
            "yw_base_dataset": "alpaca", "yw_base_index": 10,
            "yl_base_dataset": "dolly", "yl_base_index": 20,
        }) + "\n"
    )
    keys = collect_used_base_keys(dpo_file)
    assert ("alpaca", 10) in keys
    assert ("dolly", 20) in keys


def test_collect_used_base_keys_missing_file(tmp_path: Path):
    """Returns empty set for non-existent file."""
    keys = collect_used_base_keys(tmp_path / "nonexistent.jsonl")
    assert keys == set()


def test_collect_used_base_keys_mixed(tmp_path: Path):
    """Handles file with both sft_source and yw_base_dataset fields."""
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        json.dumps({"sft_source": "alpaca", "sft_index": 1}) + "\n"
        + json.dumps({"yw_base_dataset": "dolly", "yw_base_index": 2}) + "\n"
    )
    keys = collect_used_base_keys(f)
    assert ("alpaca", 1) in keys
    assert ("dolly", 2) in keys
