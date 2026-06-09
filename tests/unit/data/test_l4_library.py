"""Tests for L4 library: wrapping, synthesis, and unified persistence."""

import json
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from src.data.libraries.l4_tool_outputs import (
    L4Entry,
    MIN_CONTENT_CHARS,
    PLACEHOLDER_PATTERNS,
    build_l4_wrapped,
    is_placeholder,
    load_l4_library,
    save_l4_library,
    synthesize_l4_outputs,
    validate_l4_library,
)


# ---------------------------------------------------------------------------
# L4Entry dataclass
# ---------------------------------------------------------------------------


def test_l4entry_fields():
    entry = L4Entry(source="alpaca", index=42, l4_content="<tool>data</tool>", generation="wrapped")
    assert entry.source == "alpaca"
    assert entry.index == 42
    assert entry.l4_content == "<tool>data</tool>"
    assert entry.generation == "wrapped"


def test_l4entry_to_dict():
    entry = L4Entry(source="dolly", index=7, l4_content="content", generation="synthesized")
    d = entry.to_dict()
    assert d == {
        "source": "dolly",
        "index": 7,
        "l4_content": "content",
        "generation": "synthesized",
    }


def test_l4entry_from_dict():
    d = {"source": "alpaca", "index": 3, "l4_content": "data", "generation": "wrapped"}
    entry = L4Entry.from_dict(d)
    assert entry.source == "alpaca"
    assert entry.index == 3


# ---------------------------------------------------------------------------
# build_l4_wrapped — Source A
# ---------------------------------------------------------------------------


# Long enough content that passes the 200-char minimum
_LONG_ALPACA_INPUT = (
    "Global warming is the term used to describe a gradual increase in the "
    "average temperature of the Earth's atmosphere and its oceans, a change "
    "that is believed to be permanently changing the Earth's climate forever."
)
_LONG_DOLLY_CONTEXT = (
    "Machine learning is a subset of artificial intelligence that provides "
    "systems the ability to automatically learn and improve from experience "
    "without being explicitly programmed. It focuses on the development of "
    "computer programs that can access data and use it to learn for themselves."
)


def test_build_l4_wrapped_alpaca():
    ds = Dataset.from_dict({
        "instruction": ["Summarise this article.", "Explain gravity."],
        "input": [_LONG_ALPACA_INPUT, ""],
        "output": ["Summary...", "Gravity is..."],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 1
    assert entries[0].source == "alpaca"
    assert entries[0].index == 0
    assert entries[0].generation == "wrapped"
    assert "Global warming" in entries[0].l4_content


def test_build_l4_wrapped_dolly():
    ds = Dataset.from_dict({
        "instruction": ["What is ML?", "Define AI."],
        "context": [_LONG_DOLLY_CONTEXT, ""],
        "response": ["ML is...", "AI is..."],
        "category": ["open_qa", "open_qa"],
    })
    entries = build_l4_wrapped(ds, source="dolly", data_field="context")
    assert len(entries) == 1
    assert entries[0].source == "dolly"
    assert entries[0].index == 0
    assert "Machine learning" in entries[0].l4_content


def test_build_l4_wrapped_skips_empty():
    ds = Dataset.from_dict({
        "instruction": ["Q1", "Q2", "Q3"],
        "input": ["", _LONG_ALPACA_INPUT, "   "],
        "output": ["A1", "A2", "A3"],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 1
    assert entries[0].index == 1


def test_build_l4_wrapped_uses_instruction_as_query():
    ds = Dataset.from_dict({
        "instruction": ["What is Python?"],
        "input": [_LONG_ALPACA_INPUT],
        "output": ["..."],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 1


def test_build_l4_wrapped_deterministic():
    long2 = _LONG_ALPACA_INPUT.replace("Global", "Regional")
    ds = Dataset.from_dict({
        "instruction": ["Q1", "Q2"],
        "input": [_LONG_ALPACA_INPUT, long2],
        "output": ["A1", "A2"],
    })
    e1 = build_l4_wrapped(ds, source="alpaca", data_field="input")
    e2 = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert e1[0].l4_content == e2[0].l4_content
    assert e1[1].l4_content == e2[1].l4_content


# ---------------------------------------------------------------------------
# Quality filters: minimum length and placeholder detection
# ---------------------------------------------------------------------------


def test_min_content_chars_is_200():
    assert MIN_CONTENT_CHARS == 200


def test_is_placeholder_detects_insert_patterns():
    assert is_placeholder("[Insert article]") is True
    assert is_placeholder("[insert poem]") is True
    assert is_placeholder("[Insert Photo Here]") is True
    assert is_placeholder("(Story)") is True


def test_is_placeholder_passes_real_content():
    assert is_placeholder(_LONG_ALPACA_INPUT) is False
    assert is_placeholder("A normal sentence about inserting data.") is False


def test_build_l4_wrapped_filters_short_content():
    short_input = "Toronto"  # 7 chars, well below 200
    ds = Dataset.from_dict({
        "instruction": ["Identify the city.", "Summarise this."],
        "input": [short_input, _LONG_ALPACA_INPUT],
        "output": ["...", "..."],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 1
    assert entries[0].index == 1


def test_build_l4_wrapped_filters_placeholders():
    ds = Dataset.from_dict({
        "instruction": ["Summarise the article.", "Describe the image."],
        "input": ["[Insert article]", "[Insert Photo Here]"],
        "output": ["...", "..."],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 0


def test_build_l4_wrapped_filters_borderline_length():
    exactly_199 = "x" * 199
    exactly_200 = "x" * 200
    ds = Dataset.from_dict({
        "instruction": ["Q1", "Q2"],
        "input": [exactly_199, exactly_200],
        "output": ["A1", "A2"],
    })
    entries = build_l4_wrapped(ds, source="alpaca", data_field="input")
    assert len(entries) == 1
    assert entries[0].index == 1


# ---------------------------------------------------------------------------
# synthesize_l4_outputs — Source B (updated signature)
# ---------------------------------------------------------------------------


def test_synthesize_returns_l4entries():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized tool output"

    ds = Dataset.from_dict({
        "instruction": ["What is AI?", "Explain gravity."],
        "input": ["", ""],
        "output": ["AI is...", "Gravity is..."],
    })

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        max_examples=2,
    )
    assert len(results) == 2
    assert all(isinstance(r, L4Entry) for r in results)
    assert all(r.generation == "synthesized" for r in results)
    assert all(r.source == "alpaca" for r in results)


def test_synthesize_dolly_uses_context_field():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized output"

    ds = Dataset.from_dict({
        "instruction": ["Q1", "Q2"],
        "context": ["has context", ""],
        "response": ["A1", "A2"],
        "category": ["open_qa", "open_qa"],
    })

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="dolly",
        data_field="context",
    )
    # Only Q2 has empty context
    assert len(results) == 1
    assert results[0].source == "dolly"
    assert results[0].index == 1


def test_synthesize_skips_nonempty_data():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": ["Q1", "Q2", "Q3"],
        "input": ["has data", "", "also has data"],
        "output": ["A1", "A2", "A3"],
    })

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
    )
    assert len(results) == 1
    assert results[0].index == 1


def test_synthesize_respects_max_examples():
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": [f"Q{i}" for i in range(10)],
        "input": [""] * 10,
        "output": [f"A{i}" for i in range(10)],
    })

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        max_examples=3,
    )
    assert len(results) == 3


def test_synthesize_flushes_intermediate_results(tmp_path):
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": [f"Q{i}" for i in range(5)],
        "input": [""] * 5,
        "output": [f"A{i}" for i in range(5)],
    })

    prior = [L4Entry(source="alpaca", index=99, l4_content="prior", generation="wrapped")]
    out = tmp_path / "l4_library.json"

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        flush_path=out,
        prior_entries=prior,
        flush_every=2,
    )

    assert len(results) == 5
    # File should exist from intermediate flush
    assert out.exists()
    loaded = load_l4_library(out)
    # Last flush at idx==3 (4th item), so file has prior(1) + results(4) = 5
    assert len(loaded) == 5
    assert loaded[0].source == "alpaca"
    assert loaded[0].index == 99


# ---------------------------------------------------------------------------
# save_l4_library / load_l4_library — unified persistence
# ---------------------------------------------------------------------------


def test_save_and_load_l4_library(tmp_path):
    entries = [
        L4Entry(source="alpaca", index=0, l4_content="wrapped data", generation="wrapped"),
        L4Entry(source="dolly", index=5, l4_content="synth data", generation="synthesized"),
    ]
    path = tmp_path / "l4_library.json"
    save_l4_library(entries, path)
    loaded = load_l4_library(path)

    assert len(loaded) == 2
    assert loaded[0].source == "alpaca"
    assert loaded[0].generation == "wrapped"
    assert loaded[1].source == "dolly"
    assert loaded[1].index == 5


def test_load_l4_library_missing_file():
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        load_l4_library(Path("/nonexistent/l4_library.json"))


def test_save_l4_library_creates_parent_dirs(tmp_path):
    entries = [L4Entry(source="alpaca", index=0, l4_content="data", generation="wrapped")]
    path = tmp_path / "nested" / "dir" / "l4_library.json"
    save_l4_library(entries, path)
    assert path.exists()


def test_save_l4_library_roundtrip_preserves_content(tmp_path):
    entries = [
        L4Entry(
            source="alpaca",
            index=42,
            l4_content='<tool_output source="web_search" query="test">\ndata\n</tool_output>',
            generation="wrapped",
        ),
    ]
    path = tmp_path / "l4_library.json"
    save_l4_library(entries, path)
    loaded = load_l4_library(path)
    assert loaded[0].l4_content == entries[0].l4_content


# ---------------------------------------------------------------------------
# validate_l4_library
# ---------------------------------------------------------------------------


def test_validate_l4_library_returns_stats():
    entries = [
        L4Entry(source="alpaca", index=0, l4_content="data", generation="wrapped"),
        L4Entry(source="alpaca", index=1, l4_content="data", generation="synthesized"),
        L4Entry(source="dolly", index=0, l4_content="data", generation="wrapped"),
    ]
    stats = validate_l4_library(entries)
    assert stats["total"] == 3
    assert stats["source_counts"]["alpaca"] == 2
    assert stats["source_counts"]["dolly"] == 1
    assert stats["generation_counts"]["wrapped"] == 2
    assert stats["generation_counts"]["synthesized"] == 1


# ---------------------------------------------------------------------------
# synthesize_l4_outputs — skip_indices (resume support)
# ---------------------------------------------------------------------------


def test_synthesize_skips_indices_in_skip_set():
    """When skip_indices is provided, those (source, index) pairs are not synthesised."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": ["Q0", "Q1", "Q2", "Q3"],
        "input": ["", "", "", ""],
        "output": ["A0", "A1", "A2", "A3"],
    })

    skip = {("alpaca", 0), ("alpaca", 2)}
    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        skip_indices=skip,
    )

    assert len(results) == 2
    result_indices = {r.index for r in results}
    assert result_indices == {1, 3}
    assert mock_client.generate.call_count == 2


def test_synthesize_skip_indices_none_skips_nothing():
    """When skip_indices is None (default), all empty rows are synthesised."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": ["Q0", "Q1"],
        "input": ["", ""],
        "output": ["A0", "A1"],
    })

    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
    )

    assert len(results) == 2


def test_synthesize_skip_indices_wrong_source_not_skipped():
    """skip_indices for a different source should not affect this source."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": ["Q0", "Q1"],
        "input": ["", ""],
        "output": ["A0", "A1"],
    })

    # Skip indices are for dolly, not alpaca
    skip = {("dolly", 0), ("dolly", 1)}
    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        skip_indices=skip,
    )

    assert len(results) == 2


def test_synthesize_skip_indices_combined_with_max_examples():
    """max_examples should apply after filtering out skip_indices."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Synthesized"

    ds = Dataset.from_dict({
        "instruction": [f"Q{i}" for i in range(10)],
        "input": [""] * 10,
        "output": [f"A{i}" for i in range(10)],
    })

    skip = {("alpaca", 0), ("alpaca", 1)}
    results = synthesize_l4_outputs(
        dataset=ds,
        client=mock_client,
        source="alpaca",
        data_field="input",
        max_examples=3,
        skip_indices=skip,
    )

    assert len(results) == 3
    # Skipped indices should not appear
    result_indices = {r.index for r in results}
    assert 0 not in result_indices
    assert 1 not in result_indices
