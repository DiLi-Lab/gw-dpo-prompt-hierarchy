"""Tests for the generation cache I/O. Model invocation is mocked."""

import json
from pathlib import Path

from src.evaluation.generation import generate_responses


def test_generate_responses_resumes_from_cache(tmp_path: Path) -> None:
    eval_records = [
        {"id": "eval_0001", "prompt": "p1"},
        {"id": "eval_0002", "prompt": "p2"},
    ]
    cache_path = tmp_path / "gen_cache.jsonl"

    calls: list[str] = []

    def fake_generate_batch(prompts: list[str]) -> list[str]:
        calls.extend(prompts)
        return [f"answer to {p}" for p in prompts]

    out1 = generate_responses(
        eval_records, cache_path, fake_generate_batch, batch_size=2,
    )
    assert len(out1) == 2
    assert calls == ["p1", "p2"]

    # Second pass: cache hits, no new generation
    out2 = generate_responses(
        eval_records, cache_path, fake_generate_batch, batch_size=2,
    )
    assert len(out2) == 2
    assert calls == ["p1", "p2"]


def test_generate_responses_partial_cache(tmp_path: Path) -> None:
    eval_records = [
        {"id": "eval_0001", "prompt": "p1"},
        {"id": "eval_0002", "prompt": "p2"},
    ]
    cache_path = tmp_path / "gen_cache.jsonl"
    cache_path.write_text(json.dumps({"id": "eval_0001", "response": "cached"}) + "\n")

    new_calls: list[str] = []

    def fake_generate_batch(prompts: list[str]) -> list[str]:
        new_calls.extend(prompts)
        return [f"new {p}" for p in prompts]

    out = generate_responses(
        eval_records, cache_path, fake_generate_batch, batch_size=2,
    )
    by_id = {r["id"]: r["response"] for r in out}
    assert by_id["eval_0001"] == "cached"
    assert by_id["eval_0002"] == "new p2"
    assert new_calls == ["p2"]


def test_generate_responses_writes_in_order_of_input(tmp_path: Path) -> None:
    eval_records = [{"id": f"eval_{i:04d}", "prompt": f"p{i}"} for i in range(5)]
    cache_path = tmp_path / "gen.jsonl"

    def fake_batch(prompts: list[str]) -> list[str]:
        return [f"ans-{p}" for p in prompts]

    out = generate_responses(eval_records, cache_path, fake_batch, batch_size=2)
    assert [r["id"] for r in out] == [f"eval_{i:04d}" for i in range(5)]


def test_generate_responses_validates_response_count(tmp_path: Path) -> None:
    eval_records = [{"id": "eval_0001", "prompt": "p1"}]
    cache_path = tmp_path / "gen.jsonl"

    def broken_batch(_prompts: list[str]) -> list[str]:
        return []  # wrong length

    import pytest

    with pytest.raises(RuntimeError, match="returned 0"):
        generate_responses(eval_records, cache_path, broken_batch, batch_size=2)
