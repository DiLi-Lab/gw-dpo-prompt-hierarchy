"""Tests for L1 developer system prompt generation and deduplication."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data.libraries.l1_prompts import (
    TASK_DOMAINS,
    build_l1_generation_prompt,
    compute_domain_stats,
    deduplicate_prompts,
    filter_by_length,
    generate_l1_library,
    load_l1_library,
    parse_l1_response,
    save_l1_library,
    validate_l1_library,
)


def test_task_domains_count():
    assert len(TASK_DOMAINS) == 15


def test_build_l1_generation_prompt():
    prompt = build_l1_generation_prompt("coding")
    assert "coding" in prompt
    assert "10" in prompt
    assert "JSON" in prompt


def test_parse_l1_response_valid_json():
    response = json.dumps([
        {
            "persona": "Python expert",
            "constraints": ["only Python", "no SQL"],
            "full_prompt": "You are a Python expert. Only discuss Python. Never suggest SQL.",
        },
    ])
    prompts = parse_l1_response(response)
    assert len(prompts) == 1
    assert prompts[0]["full_prompt"].startswith("You are")


def test_parse_l1_response_invalid_json():
    prompts = parse_l1_response("not json at all")
    assert prompts == []


def test_parse_l1_response_missing_field():
    response = json.dumps([{"persona": "test"}])
    prompts = parse_l1_response(response)
    assert prompts == []


def test_parse_l1_response_code_block():
    response = '```json\n[{"persona": "Coder", "constraints": ["be concise"], "full_prompt": "You are a coder."}]\n```'
    prompts = parse_l1_response(response)
    assert len(prompts) == 1


def test_filter_by_length():
    prompts = [
        {"full_prompt": "Short."},
        {"full_prompt": "This is a valid prompt with enough words to pass the minimum length filter for testing purposes. " * 2},
        {"full_prompt": "word " * 301},
    ]
    filtered = filter_by_length(prompts, min_words=30, max_words=300)
    assert len(filtered) == 1


def test_save_and_load_l1_library(tmp_path):
    prompts = [
        {"persona": "Coder", "constraints": ["be concise"], "full_prompt": "You are a coder."},
        {"persona": "Writer", "constraints": ["be creative"], "full_prompt": "You are a writer."},
    ]
    path = tmp_path / "l1_library.json"
    save_l1_library(prompts, path)
    loaded = load_l1_library(path)
    assert len(loaded) == 2
    assert loaded[0]["persona"] == "Coder"


def test_load_l1_library_missing_file():
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        load_l1_library(Path("/nonexistent/l1_library.json"))


def test_generate_l1_library_with_mock(tmp_path):
    mock_client = MagicMock()
    mock_client.generate.return_value = json.dumps([
        {
            "persona": "Python tutor",
            "constraints": ["only Python", "beginner-friendly"],
            "full_prompt": "You are a Python tutor for beginners. Only discuss Python programming. Keep explanations simple and use plenty of examples. Never assume prior programming knowledge. Always provide step by step instructions with clear comments explaining each line of code you write for the student.",
        },
    ])

    output_path = tmp_path / "l1_library.json"
    result = generate_l1_library(
        client=mock_client,
        output_path=output_path,
        domains=["coding"],
        batches_per_domain=1,
        skip_dedup=True,
    )

    assert len(result) == 1
    assert result[0]["domain"] == "coding"
    assert output_path.exists()

    loaded = load_l1_library(output_path)
    assert len(loaded) == 1


def test_parse_l1_response_validates_constraints_field():
    """Reject items where constraints is not a list of strings."""
    response = json.dumps([
        {
            "persona": "Coder",
            "constraints": "not a list",
            "full_prompt": "You are a coder.",
        },
        {
            "persona": "Writer",
            "constraints": ["valid constraint"],
            "full_prompt": "You are a writer.",
        },
        {
            "persona": "Analyst",
            "constraints": [123, 456],
            "full_prompt": "You are an analyst.",
        },
    ])
    prompts = parse_l1_response(response)
    assert len(prompts) == 1
    assert prompts[0]["persona"] == "Writer"


def test_deduplicate_prompts_removes_similar():
    prompts = [
        {"full_prompt": "You are a helpful Python coding assistant. Write clean code."},
        {"full_prompt": "You are a helpful Python coding assistant. Write clean code."},
        {"full_prompt": "You are a creative writing coach. Help with storytelling."},
    ]
    deduped = deduplicate_prompts(prompts, threshold=0.85)
    assert len(deduped) < len(prompts)
    assert len(deduped) >= 2


def test_deduplicate_prompts_empty_input():
    assert deduplicate_prompts([]) == []


def test_generate_l1_library_passes_api_params(tmp_path):
    """Verify generate_l1_library passes explicit temperature and max_tokens."""
    mock_client = MagicMock()
    mock_client.generate.return_value = json.dumps([
        {
            "persona": "Python tutor",
            "constraints": ["only Python", "beginner-friendly"],
            "full_prompt": "You are a Python tutor for beginners. Only discuss Python programming. "
            "Keep explanations simple and use plenty of examples. Never assume prior "
            "programming knowledge. Always provide step by step instructions with clear "
            "comments explaining each line of code you write for the student.",
        },
    ])

    output_path = tmp_path / "l1_library.json"
    generate_l1_library(
        client=mock_client,
        output_path=output_path,
        domains=["coding"],
        batches_per_domain=1,
        skip_dedup=True,
        temperature=0.8,
        max_tokens=2000,
    )

    call_kwargs = mock_client.generate.call_args
    assert call_kwargs.kwargs["temperature"] == 0.8
    assert call_kwargs.kwargs["max_tokens"] == 2000


def test_compute_domain_stats():
    prompts = [
        {"domain": "coding", "full_prompt": "x"},
        {"domain": "coding", "full_prompt": "y"},
        {"domain": "legal", "full_prompt": "z"},
    ]
    stats = compute_domain_stats(prompts)
    assert stats["coding"] == 2
    assert stats["legal"] == 1
    assert stats["total"] == 3


def test_validate_l1_library(tmp_path):
    prompts = [
        {
            "persona": "Coder",
            "constraints": ["be concise"],
            "full_prompt": "You are a concise coder who writes clean Python. Always prefer readability over cleverness.",
            "domain": "coding",
            "batch_idx": 0,
        },
        {
            "persona": "Writer",
            "constraints": ["be creative"],
            "full_prompt": "You are a creative writing coach. Help users improve their storytelling skills and narrative structure.",
            "domain": "creative writing",
            "batch_idx": 0,
        },
    ]
    path = tmp_path / "l1_library.json"
    save_l1_library(prompts, path)

    stats = validate_l1_library(path)
    assert stats["total"] == 2
    assert stats["coding"] == 1
    assert stats["creative writing"] == 1


def test_validate_l1_library_missing_file():
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        validate_l1_library(Path("/nonexistent/l1_library.json"))
