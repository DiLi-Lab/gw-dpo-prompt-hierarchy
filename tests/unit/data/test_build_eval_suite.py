"""Tests for eval suite build orchestrator: validation, stats, cache I/O."""

import json
import tempfile
from pathlib import Path

from src.data.eval.build_eval_suite import (
    compute_eval_stats,
    load_eval_cache,
    save_eval_cache,
    validate_eval_instance,
)


def _make_instance(split: str = "conflict", victim: int = 1) -> dict:
    prompt = "<|L0_START|>r<|L0_END|>\n<|L1_START|>s<|L1_END|>\n<|L3_START|>m<|L3_END|>"
    if split == "reference":
        prompt = "r\ns\nm"  # no delimiters
    return {
        "id": "eval_L1_vs_L3_001",
        "prompt": prompt,
        "conflict_type": "L1_vs_L3",
        "victim_level": victim,
        "attacker_level": 3,
        "level_gap": 2,
        "conflict_description": "desc",
        "correct_behaviour": "correct",
        "violation_behaviour": "wrong",
        "evaluation_criteria": ["c1", "c2"],
        "gold_response": "gold",
        "split": split,
        "base_dataset": "alpaca",
        "base_index": 1,
    }


# ---------------------------------------------------------------------------
# validate_eval_instance — valid cases
# ---------------------------------------------------------------------------


def test_validate_accepts_valid_conflict_instance():
    assert validate_eval_instance(_make_instance(split="conflict")) is True


def test_validate_accepts_valid_aligned_instance():
    instance = _make_instance(split="aligned")
    assert validate_eval_instance(instance) is True


def test_validate_accepts_valid_reference_instance():
    assert validate_eval_instance(_make_instance(split="reference")) is True


# ---------------------------------------------------------------------------
# validate_eval_instance — rejection cases
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_required_field():
    instance = _make_instance()
    del instance["conflict_type"]
    assert validate_eval_instance(instance) is False


def test_validate_rejects_empty_gold_response():
    instance = _make_instance()
    instance["gold_response"] = ""
    assert validate_eval_instance(instance) is False


def test_validate_rejects_none_gold_response():
    instance = _make_instance()
    instance["gold_response"] = None
    assert validate_eval_instance(instance) is False


def test_validate_rejects_empty_evaluation_criteria():
    instance = _make_instance()
    instance["evaluation_criteria"] = []
    assert validate_eval_instance(instance) is False


def test_validate_rejects_non_list_evaluation_criteria():
    instance = _make_instance()
    instance["evaluation_criteria"] = "single string"
    assert validate_eval_instance(instance) is False


def test_validate_rejects_reference_with_delimiters():
    instance = _make_instance(split="reference")
    # Insert delimiter tokens into the prompt — should fail
    instance["prompt"] = "<|L0_START|>r<|L0_END|>\ns\nm"
    assert validate_eval_instance(instance) is False


def test_validate_rejects_conflict_without_delimiters():
    instance = _make_instance(split="conflict")
    instance["prompt"] = "plain prompt without any delimiters"
    assert validate_eval_instance(instance) is False


def test_validate_rejects_aligned_without_delimiters():
    instance = _make_instance(split="aligned")
    instance["prompt"] = "plain prompt without any delimiters"
    assert validate_eval_instance(instance) is False


def test_validate_rejects_all_required_fields():
    required = [
        "conflict_type",
        "level_gap",
        "conflict_description",
        "correct_behaviour",
        "violation_behaviour",
        "evaluation_criteria",
        "gold_response",
        "split",
    ]
    for field in required:
        instance = _make_instance()
        del instance[field]
        assert validate_eval_instance(instance) is False, f"should reject missing {field}"


# ---------------------------------------------------------------------------
# compute_eval_stats
# ---------------------------------------------------------------------------


def test_compute_eval_stats_total_count():
    instances = [_make_instance() for _ in range(5)]
    stats = compute_eval_stats(instances)
    assert stats["total"] == 5


def test_compute_eval_stats_by_conflict_type():
    instances = [
        {**_make_instance(), "conflict_type": "L1_vs_L3"},
        {**_make_instance(), "conflict_type": "L1_vs_L3"},
        {**_make_instance(), "conflict_type": "L0_vs_L4"},
    ]
    stats = compute_eval_stats(instances)
    assert stats["by_conflict_type"]["L1_vs_L3"] == 2
    assert stats["by_conflict_type"]["L0_vs_L4"] == 1


def test_compute_eval_stats_by_split():
    instances = [
        _make_instance(split="conflict"),
        _make_instance(split="conflict"),
        _make_instance(split="aligned"),
        _make_instance(split="reference"),
    ]
    stats = compute_eval_stats(instances)
    assert stats["by_split"]["conflict"] == 2
    assert stats["by_split"]["aligned"] == 1
    assert stats["by_split"]["reference"] == 1


def test_compute_eval_stats_by_base_dataset():
    instances = [
        {**_make_instance(), "base_dataset": "alpaca"},
        {**_make_instance(), "base_dataset": "alpaca"},
        {**_make_instance(), "base_dataset": "dolly"},
    ]
    stats = compute_eval_stats(instances)
    assert stats["by_base_dataset"]["alpaca"] == 2
    assert stats["by_base_dataset"]["dolly"] == 1


def test_compute_eval_stats_empty_list():
    stats = compute_eval_stats([])
    assert stats["total"] == 0
    assert stats["by_conflict_type"] == {}
    assert stats["by_split"] == {}
    assert stats["by_base_dataset"] == {}


def test_compute_eval_stats_returns_all_keys():
    stats = compute_eval_stats([_make_instance()])
    assert "total" in stats
    assert "by_conflict_type" in stats
    assert "by_split" in stats
    assert "by_base_dataset" in stats


# ---------------------------------------------------------------------------
# save_eval_cache / load_eval_cache round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip_empty(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    save_eval_cache({}, path)
    loaded = load_eval_cache(path)
    assert loaded == {}


def test_cache_roundtrip_single_entry(tmp_path: Path):
    key = ("L1_vs_L3", "alpaca", 42)
    value = {"gold_response": "some text", "score": 4}
    cache = {key: value}
    path = tmp_path / "cache.jsonl"
    save_eval_cache(cache, path)
    loaded = load_eval_cache(path)
    assert tuple(loaded.keys()) == (key,)
    assert loaded[key] == value


def test_cache_roundtrip_multiple_entries(tmp_path: Path):
    cache = {
        ("L0_vs_L1", "alpaca", 1): {"gold": "a"},
        ("L1_vs_L3", "dolly", 99): {"gold": "b"},
        ("L2_vs_L4", "flan", 7): {"gold": "c"},
    }
    path = tmp_path / "subdir" / "cache.jsonl"
    save_eval_cache(cache, path)
    loaded = load_eval_cache(path)
    assert len(loaded) == 3
    for key, val in cache.items():
        assert loaded[key] == val


def test_load_eval_cache_missing_file_returns_empty(tmp_path: Path):
    path = tmp_path / "nonexistent.jsonl"
    result = load_eval_cache(path)
    assert result == {}


def test_save_eval_cache_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "deep" / "nested" / "dir" / "cache.jsonl"
    save_eval_cache({("L1_vs_L3", "alpaca", 0): {"x": 1}}, path)
    assert path.exists()


def test_cache_file_is_valid_jsonl(tmp_path: Path):
    cache = {
        ("L1_vs_L3", "alpaca", 1): {"gold": "hello"},
    }
    path = tmp_path / "cache.jsonl"
    save_eval_cache(cache, path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "key" in entry
    assert "value" in entry
    assert entry["key"] == ["L1_vs_L3", "alpaca", 1]
    assert entry["value"] == {"gold": "hello"}
