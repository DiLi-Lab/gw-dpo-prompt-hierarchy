"""Tests for L3 user message filtering and sampling."""

import pytest
from datasets import Dataset

from src.data.libraries.l3_user_messages import (
    L3Message,
    filter_l3_candidates,
    load_l3_pool,
    sample_l3_message,
    validate_l3_pool,
)


def test_filter_removes_short_instructions():
    ds = Dataset.from_dict({
        "instruction": ["Hi", "What is the capital of France?", "Go"],
    })
    results = filter_l3_candidates(ds, instruction_field="instruction")
    texts = [r.text for r in results]
    assert "What is the capital of France?" in texts
    assert "Hi" not in texts
    assert "Go" not in texts


def test_filter_removes_long_instructions():
    short = "Explain gravity in simple clear terms."
    long = " ".join(["word"] * 501)
    ds = Dataset.from_dict({"instruction": [short, long]})
    results = filter_l3_candidates(ds, instruction_field="instruction")
    assert len(results) == 1
    assert results[0].text == short


def test_filter_removes_exact_duplicates():
    ds = Dataset.from_dict({
        "instruction": [
            "Explain the concept of gravity clearly.",
            "Explain the concept of gravity clearly.",
            "What is the process of photosynthesis?",
        ],
    })
    results = filter_l3_candidates(ds, instruction_field="instruction")
    texts = [r.text for r in results]
    assert texts.count("Explain the concept of gravity clearly.") == 1
    assert "What is the process of photosynthesis?" in texts


def test_filter_preserves_source_metadata():
    ds = Dataset.from_dict({"instruction": ["Explain the concept of gravity in detail."]})
    results = filter_l3_candidates(ds, instruction_field="instruction", source="alpaca")
    assert results[0].source == "alpaca"


def test_filter_returns_l3message_dataclass():
    ds = Dataset.from_dict({"instruction": ["Explain the concept of gravity in detail."]})
    results = filter_l3_candidates(ds, instruction_field="instruction")
    assert isinstance(results[0], L3Message)
    assert isinstance(results[0].text, str)
    assert isinstance(results[0].source, str)


def test_filter_boundary_five_words():
    exactly_five = "one two three four five"
    four_words = "one two three four"
    ds = Dataset.from_dict({"instruction": [exactly_five, four_words]})
    results = filter_l3_candidates(ds, instruction_field="instruction")
    texts = [r.text for r in results]
    assert exactly_five in texts
    assert four_words not in texts


def test_filter_boundary_500_words():
    exactly_500 = " ".join(["word"] * 500)
    ds = Dataset.from_dict({"instruction": [exactly_500]})
    results = filter_l3_candidates(ds, instruction_field="instruction")
    assert len(results) == 1


def test_filter_strips_whitespace_before_counting():
    padded = "   What is the capital of France?   "
    ds = Dataset.from_dict({"instruction": [padded]})
    results = filter_l3_candidates(ds, instruction_field="instruction")
    assert len(results) == 1
    assert results[0].text == padded.strip()


def test_load_l3_pool_combines_sources(tmp_path):
    alpaca_ds = Dataset.from_dict({
        "instruction": ["Explain gravity in simple terms."],
        "input": [""],
        "output": ["Gravity is..."],
    })
    dolly_ds = Dataset.from_dict({
        "instruction": ["What is machine learning used for?"],
        "context": [""],
        "response": ["ML is..."],
        "category": ["open_qa"],
    })
    alpaca_path = tmp_path / "alpaca_train"
    dolly_path = tmp_path / "dolly_train"
    alpaca_ds.save_to_disk(str(alpaca_path))
    dolly_ds.save_to_disk(str(dolly_path))

    pool = load_l3_pool(alpaca_path, dolly_path)
    assert len(pool) == 2
    sources = {m.source for m in pool}
    assert sources == {"alpaca", "dolly"}


def test_load_l3_pool_deduplicates_across_sources(tmp_path):
    shared_instruction = "Explain the theory of relativity in detail."
    alpaca_ds = Dataset.from_dict({
        "instruction": [shared_instruction],
        "input": [""],
        "output": ["..."],
    })
    dolly_ds = Dataset.from_dict({
        "instruction": [shared_instruction],
        "context": [""],
        "response": ["..."],
        "category": ["open_qa"],
    })
    alpaca_path = tmp_path / "alpaca_train"
    dolly_path = tmp_path / "dolly_train"
    alpaca_ds.save_to_disk(str(alpaca_path))
    dolly_ds.save_to_disk(str(dolly_path))

    pool = load_l3_pool(alpaca_path, dolly_path)
    texts = [m.text for m in pool]
    assert texts.count(shared_instruction) == 1


def test_sample_l3_message_returns_l3message():
    pool = [L3Message(text="Explain gravity.", source="alpaca")]
    result = sample_l3_message(pool, seed=42)
    assert isinstance(result, L3Message)


def test_sample_l3_message_deterministic():
    pool = [
        L3Message(text=f"Question {i}?", source="alpaca")
        for i in range(100)
    ]
    r1 = sample_l3_message(pool, seed=42)
    r2 = sample_l3_message(pool, seed=42)
    assert r1.text == r2.text


def test_sample_l3_message_varies():
    pool = [
        L3Message(text=f"Question {i}?", source="alpaca")
        for i in range(100)
    ]
    results = {sample_l3_message(pool, seed=i).text for i in range(50)}
    assert len(results) > 10


def test_sample_l3_message_empty_pool_raises():
    with pytest.raises(ValueError, match="empty"):
        sample_l3_message([], seed=42)


def test_validate_l3_pool_returns_stats():
    pool = [
        L3Message(text="Question 1?", source="alpaca"),
        L3Message(text="Question 2?", source="alpaca"),
        L3Message(text="Question 3?", source="dolly"),
    ]
    stats = validate_l3_pool(pool)
    assert stats["total"] == 3
    assert stats["source_counts"]["alpaca"] == 2
    assert stats["source_counts"]["dolly"] == 1
    assert "word_count_stats" in stats


def test_validate_l3_pool_word_count_stats():
    pool = [
        L3Message(text="one two three four five", source="alpaca"),
        L3Message(text="one two three four five six seven eight nine ten", source="dolly"),
    ]
    stats = validate_l3_pool(pool)
    assert stats["word_count_stats"]["min"] == 5
    assert stats["word_count_stats"]["max"] == 10
