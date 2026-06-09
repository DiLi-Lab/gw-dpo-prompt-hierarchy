"""Tests for aligned SFT examples builder."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.libraries.l0_rules import L0Rule
from src.data.sft.aligned import (
    build_context_synthesis_prompt,
    build_context_synthesis_aligned,
    build_simple_aligned,
    parse_context_synthesis_response,
)
from src.data.sft.build_sft_dataset import load_sft_dataset


def _make_l0_rules() -> list[L0Rule]:
    """Create minimal L0 rules for testing (needs at least one system_integrity)."""
    return [
        L0Rule(category="system_integrity", rule="Follow hierarchy.", id="si_1"),
        L0Rule(category="safety", rule="No harmful content.", id="s_1"),
        L0Rule(category="safety", rule="No PII.", id="s_2"),
        L0Rule(category="privacy", rule="Protect user data.", id="p_1"),
        L0Rule(category="compliance", rule="Follow regulations.", id="c_1"),
        L0Rule(category="fairness", rule="Be unbiased.", id="f_1"),
    ]


def _make_base_rows(n: int = 10) -> list[dict]:
    """Create mixed Alpaca/Dolly base rows with _sft_source and _sft_index tags."""
    rows: list[dict] = []
    for i in range(n):
        if i % 2 == 0:
            rows.append({
                "instruction": f"Write a Python function that does task {i}",
                "input": f"data for task {i}",
                "output": f"Here is the solution for task {i}.",
                "_sft_source": "alpaca",
                "_sft_index": i // 2,
            })
        else:
            rows.append({
                "instruction": f"Explain concept {i} in simple terms",
                "context": f"background for concept {i}",
                "response": f"Concept {i} is about fundamentals.",
                "_sft_source": "dolly",
                "_sft_index": i // 2,
            })
    return rows


def _make_l1_library() -> list[dict]:
    """Create a minimal L1 library with coding and general knowledge domains."""
    return [
        {
            "domain": "coding",
            "persona": "senior developer",
            "constraints": ["Write clean code"],
            "full_prompt": "You are a senior developer. Write clean, efficient code.",
        },
        {
            "domain": "general knowledge",
            "persona": "helpful assistant",
            "constraints": ["Be accurate"],
            "full_prompt": "You are a helpful assistant. Be accurate and clear.",
        },
    ]


def _make_l4_lookup_for_rows(rows: list[dict]) -> dict[tuple[str, int], dict[str, str]]:
    """Create an L4 lookup covering every row in the list."""
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for i, r in enumerate(rows):
        key = (r["_sft_source"], r["_sft_index"])
        lookup[key] = {
            "l4_content": f'<tool_output source="web_search">Data for {key[0]} {key[1]}</tool_output>',
            "generation": "wrapped" if i % 2 == 0 else "synthesized",
        }
    return lookup


def _make_l4_lookup_sparse(n: int = 10) -> dict[tuple[str, int], dict[str, str]]:
    """Create an L4 lookup with entries for only even indices."""
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for i in range(0, n // 2, 2):
        lookup[("alpaca", i)] = {
            "l4_content": f'<tool_output source="web_search">Data for alpaca {i}</tool_output>',
            "generation": "wrapped",
        }
        lookup[("dolly", i)] = {
            "l4_content": f'<tool_output source="web_search">Data for dolly {i}</tool_output>',
            "generation": "synthesized",
        }
    return lookup


class TestBuildSimpleAligned:
    """Tests for build_simple_aligned()."""

    def test_produces_correct_count(self) -> None:
        """Should produce exactly the requested number of examples."""
        rows = _make_base_rows(20)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=5,
            seed=42,
        )
        assert len(result) == 5

    def test_correct_schema(self) -> None:
        """Each example should have text, levels_present, is_conflict=False."""
        rows = _make_base_rows(10)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=3,
            seed=42,
        )
        for ex in result:
            assert "text" in ex
            assert "levels_present" in ex
            assert "is_conflict" in ex
            assert "conflict_type" in ex
            assert ex["is_conflict"] is False
            assert ex["conflict_type"] is None
            assert isinstance(ex["text"], str)
            assert isinstance(ex["levels_present"], list)

    def test_includes_delimiter_tokens(self) -> None:
        """Assembled text should include L0, L1, L3, and RESP tokens."""
        rows = _make_base_rows(10)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=3,
            seed=42,
        )
        for ex in result:
            text = ex["text"]
            assert "<|L0_START|>" in text
            assert "<|L0_END|>" in text
            assert "<|L1_START|>" in text
            assert "<|L1_END|>" in text
            assert "<|L3_START|>" in text
            assert "<|L3_END|>" in text
            assert "<|RESP_START|>" in text
            assert "<|RESP_END|>" in text

    def test_levels_present_with_l4(self) -> None:
        """When L4 is available, levels_present should be [0,1,2,3,4]."""
        # All rows have L4 entries keyed by their tags
        rows = _make_base_rows(3)
        lookup = {(r["_sft_source"], r["_sft_index"]): {"l4_content": f"tool data {i}", "generation": "wrapped"} for i, r in enumerate(rows)}
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=lookup,
            count=3,
            seed=42,
        )
        for ex in result:
            assert ex["levels_present"] == [0, 1, 2, 3, 4]

    def test_levels_present_without_l4(self) -> None:
        """When no L4 is available, should raise ValueError (all rows must have L4)."""
        rows = _make_base_rows(3)
        with pytest.raises(ValueError, match="no L4 entry"):
            build_simple_aligned(
                base_rows=rows,
                l0_rules=_make_l0_rules(),
                l1_library=_make_l1_library(),
                l4_lookup={},
                count=3,
                seed=42,
            )

    def test_count_capped_by_available_rows(self) -> None:
        """If count exceeds available rows, should produce len(rows) examples."""
        rows = _make_base_rows(3)
        lookup = _make_l4_lookup_for_rows(rows)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=lookup,
            count=100,
            seed=42,
        )
        assert len(result) == 3

    def test_dolly_schema_rows_produce_valid_output(self) -> None:
        """Dolly-format rows (response/context) should produce valid output text."""
        rows = [
            {
                "instruction": "Explain quantum computing",
                "context": "background info on qubits",
                "response": "Quantum computing uses qubits to perform calculations.",
                "_sft_source": "dolly",
                "_sft_index": 0,
            },
            {
                "instruction": "What is machine learning?",
                "context": "",
                "response": "Machine learning is a subset of AI.",
                "_sft_source": "dolly",
                "_sft_index": 1,
            },
        ]
        lookup = _make_l4_lookup_for_rows(rows)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=lookup,
            count=2,
            seed=42,
        )
        assert len(result) == 2
        responses = [r["response"] for r in rows]
        for ex in result:
            text = ex["text"]
            assert any(resp in text for resp in responses), (
                f"Expected one of the Dolly responses in output text, got: {text[:200]}"
            )


class TestSimpleAlignedMetadata:
    """Tests for metadata fields in build_simple_aligned."""

    def test_all_have_sft_source(self) -> None:
        rows = _make_base_rows(10)
        lookup = _make_l4_lookup_for_rows(rows)
        result = build_simple_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=lookup, count=5, seed=42,
        )
        for ex in result:
            assert ex["sft_source"] in ("alpaca", "dolly")
            assert isinstance(ex["sft_index"], int)

    def test_sft_category_is_simple_aligned(self) -> None:
        rows = _make_base_rows(10)
        lookup = _make_l4_lookup_for_rows(rows)
        result = build_simple_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=lookup, count=5, seed=42,
        )
        for ex in result:
            assert ex["sft_category"] == "simple_aligned"

    def test_l4_generation_from_lookup(self) -> None:
        rows = _make_base_rows(10)
        lookup = {(r["_sft_source"], r["_sft_index"]): {"l4_content": f"data {i}", "generation": "synthesized"} for i, r in enumerate(rows)}
        result = build_simple_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=lookup, count=5, seed=42,
        )
        for ex in result:
            assert ex["l4_generation"] == "synthesized"

    def test_raises_on_missing_l4(self) -> None:
        rows = _make_base_rows(3)
        with pytest.raises(ValueError, match="no L4 entry"):
            build_simple_aligned(
                base_rows=rows, l0_rules=_make_l0_rules(),
                l1_library=_make_l1_library(), l4_lookup={}, count=3, seed=42,
            )


class TestBuildContextSynthesisPrompt:
    """Tests for build_context_synthesis_prompt()."""

    def test_includes_instruction(self) -> None:
        """Prompt should contain the original instruction."""
        prompt = build_context_synthesis_prompt("Write a poem about cats")
        assert "Write a poem about cats" in prompt

    def test_includes_data(self) -> None:
        """Prompt should contain the data field when provided."""
        prompt = build_context_synthesis_prompt("Summarize this", "Some article text")
        assert "Some article text" in prompt

    def test_includes_level_names(self) -> None:
        """Prompt should reference all hierarchy level names."""
        prompt = build_context_synthesis_prompt("Test instruction")
        assert "L1" in prompt
        assert "L2" in prompt
        assert "L3" in prompt
        assert "L4" in prompt
        assert "developer system prompt" in prompt
        assert "user configuration" in prompt
        assert "user message" in prompt

    def test_empty_data_default(self) -> None:
        """With no data arg, prompt should still be valid."""
        prompt = build_context_synthesis_prompt("Some instruction")
        assert 'Original input/data: ""' in prompt


class TestParseContextSynthesisResponse:
    """Tests for parse_context_synthesis_response()."""

    def test_valid_json(self) -> None:
        """Should parse valid JSON with required keys."""
        response = '{"l1": "sys prompt", "l2": "config", "l3": "user msg"}'
        result = parse_context_synthesis_response(response)
        assert result is not None
        assert result["l1"] == "sys prompt"
        assert result["l2"] == "config"
        assert result["l3"] == "user msg"

    def test_valid_json_with_l4(self) -> None:
        """Should parse JSON with optional l4 key."""
        response = '{"l1": "a", "l2": "b", "l3": "c", "l4": "data"}'
        result = parse_context_synthesis_response(response)
        assert result is not None
        assert result["l4"] == "data"

    def test_invalid_json_returns_none(self) -> None:
        """Should return None for malformed JSON."""
        assert parse_context_synthesis_response("not json at all") is None

    def test_missing_required_keys_returns_none(self) -> None:
        """Should return None when required keys are missing."""
        assert parse_context_synthesis_response('{"l1": "a"}') is None

    def test_non_dict_returns_none(self) -> None:
        """Should return None when response is a JSON array."""
        assert parse_context_synthesis_response("[1, 2, 3]") is None

    def test_handles_markdown_code_fences(self) -> None:
        """Should strip markdown code fences and parse JSON inside."""
        response = '```json\n{"l1": "a", "l2": "b", "l3": "c"}\n```'
        result = parse_context_synthesis_response(response)
        assert result is not None
        assert result["l1"] == "a"

    def test_handles_plain_code_fences(self) -> None:
        """Should strip ``` fences without language tag."""
        response = '```\n{"l1": "x", "l2": "y", "l3": "z"}\n```'
        result = parse_context_synthesis_response(response)
        assert result is not None
        assert result["l3"] == "z"

    def test_empty_string_returns_none(self) -> None:
        """Should return None for empty input."""
        assert parse_context_synthesis_response("") is None


class TestBuildContextSynthesisAligned:
    """Tests for build_context_synthesis_aligned() with mocked client."""

    def _make_mock_client(self, responses: list[str]) -> MagicMock:
        """Create a mock client returning the given responses in sequence."""
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)
        return client

    def test_produces_correct_count(self) -> None:
        """Should produce examples matching successful API calls."""
        rows = _make_base_rows(5)
        responses = [
            '{"l1": "sys", "l2": "cfg", "l3": "msg"}' for _ in range(5)
        ]
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=5,
            seed=42,
        )
        assert len(result) == 5

    def test_skips_failures(self) -> None:
        """Failed parses should be skipped, reducing the count."""
        rows = _make_base_rows(4)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}',
            "invalid json",
            '{"l1": "d", "l2": "e", "l3": "f"}',
            '{"l1": "g", "l2": "h", "l3": "i"}',
        ]
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=4,
            seed=42,
        )
        assert len(result) == 3

    def test_correct_schema(self) -> None:
        """Each example should have the correct SFT schema."""
        rows = _make_base_rows(2)
        responses = [
            '{"l1": "sys", "l2": "cfg", "l3": "msg", "l4": "data"}',
            '{"l1": "sys2", "l2": "cfg2", "l3": "msg2"}',
        ]
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=2,
            seed=42,
        )
        for ex in result:
            assert ex["is_conflict"] is False
            assert ex["conflict_type"] is None
            assert "<|RESP_START|>" in ex["text"]

    def test_levels_present_depends_on_l4(self) -> None:
        """levels_present should include 4 only when l4 is in the response."""
        rows = _make_base_rows(2)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c", "l4": "d"}',
            '{"l1": "a", "l2": "b", "l3": "c"}',
        ]
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=2,
            seed=42,
        )
        assert result[0]["levels_present"] == [0, 1, 2, 3, 4]
        assert result[1]["levels_present"] == [0, 1, 2, 3]

    def test_client_called_with_correct_args(self) -> None:
        """Client.generate should be called with proper system prompt and model."""
        rows = _make_base_rows(1)
        responses = ['{"l1": "a", "l2": "b", "l3": "c"}']
        client = self._make_mock_client(responses)
        build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=1,
            seed=42,
        )
        call_kwargs = client.generate.call_args
        assert "system_prompt" in call_kwargs.kwargs
        assert "gpt-4o" == call_kwargs.kwargs["model"]
        assert call_kwargs.kwargs["temperature"] == 0.3


class TestContextSynthesisIncrementalSave:
    """Tests for incremental saving in build_context_synthesis_aligned()."""

    def _make_mock_client(self, responses: list[str]) -> MagicMock:
        """Create a mock client returning the given responses in sequence."""
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)
        return client

    def test_flush_writes_intermediate_results(self, tmp_path: Path) -> None:
        """With flush_every=2, cache file should be written after every 2 items."""
        rows = _make_base_rows(6)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(6)
        ]
        client = self._make_mock_client(responses)
        flush_path = tmp_path / "synthesis_cache.jsonl"

        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=6,
            seed=42,
            flush_path=flush_path,
            flush_every=2,
        )

        assert len(result) == 6
        assert flush_path.exists()
        cached = load_sft_dataset(flush_path)
        assert len(cached) == 6

    def test_flush_not_written_without_flush_path(self, tmp_path: Path) -> None:
        """Without flush_path, no cache file should be created."""
        rows = _make_base_rows(3)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(3)
        ]
        client = self._make_mock_client(responses)

        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=3,
            seed=42,
        )

        assert len(result) == 3
        # No file should exist anywhere in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_flush_includes_only_successful_results(self, tmp_path: Path) -> None:
        """Failed API calls should not appear in the cache file."""
        rows = _make_base_rows(4)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}',
            "invalid json",
            '{"l1": "d", "l2": "e", "l3": "f"}',
            '{"l1": "g", "l2": "h", "l3": "i"}',
        ]
        client = self._make_mock_client(responses)
        flush_path = tmp_path / "cache.jsonl"

        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=4,
            seed=42,
            flush_path=flush_path,
            flush_every=2,
        )

        assert len(result) == 3
        cached = load_sft_dataset(flush_path)
        assert len(cached) == 3

    def test_skip_indices_skips_already_processed_rows(self) -> None:
        """Rows in skip_indices should not trigger API calls."""
        rows = _make_base_rows(4)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(4)
        ]
        client = self._make_mock_client(responses)

        # Skip the first two row indices (seed=42 shuffles indices)
        # We need to know which rows are selected; easier to skip all and check 0 calls
        all_indices = {(r["_sft_source"], r["_sft_index"]) for r in rows}

        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=4,
            seed=42,
            skip_indices=all_indices,
        )

        assert len(result) == 0
        assert client.generate.call_count == 0

    def test_skip_indices_partial_skip(self) -> None:
        """Only rows NOT in skip_indices should trigger API calls."""
        rows = _make_base_rows(4)
        # Skip only some rows
        skip = {(rows[0]["_sft_source"], rows[0]["_sft_index"])}

        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(4)
        ]
        client = self._make_mock_client(responses)

        result = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=4,
            seed=42,
            skip_indices=skip,
        )

        # Should have fewer API calls than total count
        assert client.generate.call_count < 4
        # And fewer results (some skipped, some synthesized)
        assert len(result) == client.generate.call_count

    def test_cached_examples_contain_source_tags(self, tmp_path: Path) -> None:
        """Cached examples should include sft_source and sft_index for resume."""
        rows = _make_base_rows(3)
        responses = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(3)
        ]
        client = self._make_mock_client(responses)
        flush_path = tmp_path / "cache.jsonl"

        build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client,
            count=3,
            seed=42,
            flush_path=flush_path,
            flush_every=1,
        )

        cached = load_sft_dataset(flush_path)
        for ex in cached:
            assert "sft_source" in ex
            assert "sft_index" in ex
            assert ex["sft_source"] in ("alpaca", "dolly")
            assert isinstance(ex["sft_index"], int)

    def test_resume_round_trip_from_cache(self, tmp_path: Path) -> None:
        """Full round-trip: first run saves cache, second run loads and skips."""
        rows = _make_base_rows(4)
        flush_path = tmp_path / "cache.jsonl"

        # First run: synthesize all 4
        responses_1 = [
            '{"l1": "a", "l2": "b", "l3": "c"}' for _ in range(4)
        ]
        client_1 = self._make_mock_client(responses_1)
        result_1 = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client_1,
            count=4,
            seed=42,
            flush_path=flush_path,
            flush_every=2,
        )
        assert len(result_1) == 4
        assert client_1.generate.call_count == 4

        # Load cache and build skip set (simulating what bin/build_sft_dataset.py does)
        cached = load_sft_dataset(flush_path)
        skip_indices = {
            (ex.get("sft_source") or ex.get("_sft_source"),
             ex.get("sft_index") if ex.get("sft_index") is not None else ex.get("_sft_index"))
            for ex in cached
        }

        # Second run: all rows should be skipped via cache-derived skip_indices
        responses_2 = [
            '{"l1": "x", "l2": "y", "l3": "z"}' for _ in range(4)
        ]
        client_2 = self._make_mock_client(responses_2)
        result_2 = build_context_synthesis_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            client=client_2,
            count=4,
            seed=42,
            skip_indices=skip_indices,
        )
        assert len(result_2) == 0
        assert client_2.generate.call_count == 0


class TestContextSynthesisMetadata:
    """Tests for metadata and L4 fallback in build_context_synthesis_aligned."""

    def _make_mock_client(self, responses: list[str]) -> MagicMock:
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)
        return client

    def test_sft_category_is_context_synthesis(self) -> None:
        rows = _make_base_rows(2)
        responses = ['{"l1": "a", "l2": "b", "l3": "c", "l4": "d"}'] * 2
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=2, seed=42,
        )
        for ex in result:
            assert ex["sft_category"] == "context_synthesis"

    def test_provenance_set(self) -> None:
        rows = _make_base_rows(2)
        responses = ['{"l1": "a", "l2": "b", "l3": "c", "l4": "d"}'] * 2
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=2, seed=42,
        )
        for ex in result:
            assert ex["sft_source"] in ("alpaca", "dolly")
            assert isinstance(ex["sft_index"], int)

    def test_l4_generation_context_synthesis_when_gpt4o_provides_l4(self) -> None:
        rows = _make_base_rows(1)
        responses = ['{"l1": "a", "l2": "b", "l3": "c", "l4": "data"}']
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=1, seed=42,
        )
        assert result[0]["l4_generation"] == "context_synthesis"

    def test_l4_fallback_to_library(self) -> None:
        """When GPT-4o omits L4, should fall back to L4 library."""
        rows = _make_base_rows(1)
        responses = ['{"l1": "a", "l2": "b", "l3": "c"}']
        client = self._make_mock_client(responses)
        lookup = {(rows[0]["_sft_source"], rows[0]["_sft_index"]): {"l4_content": "fallback data", "generation": "wrapped"}}
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=1, seed=42, l4_lookup=lookup,
        )
        assert len(result) == 1
        assert result[0]["levels_present"] == [0, 1, 2, 3, 4]
        assert "fallback data" in result[0]["text"]
        assert result[0]["l4_generation"] == "wrapped"

    def test_l4_fallback_on_empty_string_l4(self) -> None:
        """When GPT-4o returns l4 as empty string, should fall back to library."""
        rows = _make_base_rows(1)
        responses = ['{"l1": "a", "l2": "b", "l3": "c", "l4": ""}']
        client = self._make_mock_client(responses)
        lookup = {(rows[0]["_sft_source"], rows[0]["_sft_index"]): {"l4_content": "fallback", "generation": "synthesized"}}
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=1, seed=42, l4_lookup=lookup,
        )
        assert result[0]["levels_present"] == [0, 1, 2, 3, 4]
        assert result[0]["l4_generation"] == "synthesized"

    def test_l4_fallback_failure_allows_4_levels(self) -> None:
        """When both GPT-4o and library miss L4, allow 4 levels."""
        rows = _make_base_rows(1)
        responses = ['{"l1": "a", "l2": "b", "l3": "c"}']
        client = self._make_mock_client(responses)
        result = build_context_synthesis_aligned(
            base_rows=rows, l0_rules=_make_l0_rules(), client=client,
            count=1, seed=42, l4_lookup={},
        )
        assert len(result) == 1
        assert result[0]["levels_present"] == [0, 1, 2, 3]
        assert result[0]["l4_generation"] is None


class TestSimpleAlignedWithL2Client:
    """Tests for build_simple_aligned() with L2 generation client."""

    def _mock_l2_client(self, l2_text: str = "Session config: Respond in English. Tone: casual.") -> MagicMock:
        client = MagicMock()
        client.generate = MagicMock(return_value=l2_text)
        return client

    def test_uses_client_for_l2_when_provided(self) -> None:
        rows = _make_base_rows(5)
        client = self._mock_l2_client()
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=3,
            seed=42,
            openai_client=client,
        )
        assert len(result) == 3
        for ex in result:
            assert "Respond in English" in ex["text"]
        assert client.generate.call_count == 3

    def test_uses_cache_before_api(self) -> None:
        rows = _make_base_rows(10)
        client = self._mock_l2_client()
        cache = {
            (r["_sft_source"], r["_sft_index"]): "Cached L2: English, casual."
            for r in rows
        }
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=5,
            seed=42,
            openai_client=client,
            l2_cache=cache,
        )
        assert len(result) == 5
        assert client.generate.call_count == 0

    def test_falls_back_to_template_without_client(self) -> None:
        rows = _make_base_rows(5)
        result = build_simple_aligned(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup_for_rows(rows),
            count=3,
            seed=42,
            openai_client=None,
        )
        assert len(result) == 3
        for ex in result:
            assert "<|L2_START|>" in ex["text"]
