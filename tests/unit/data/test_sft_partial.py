"""Tests for partial-level SFT examples builder."""

import pytest

from src.data.libraries.l0_rules import L0Rule
from src.data.sft.partial import PARTIAL_CONFIGS, build_partial_examples


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


def _make_base_rows(n: int = 100) -> list[dict]:
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


def _make_l4_lookup(n: int = 100) -> dict[tuple[str, int], dict[str, str]]:
    """Create an L4 lookup with entries for a subset of rows (even indices)."""
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


class TestPartialConfigs:
    """Tests for the PARTIAL_CONFIGS constant."""

    def test_has_exactly_four_entries(self) -> None:
        """PARTIAL_CONFIGS should contain exactly 4 configuration dicts."""
        assert len(PARTIAL_CONFIGS) == 4

    def test_each_config_has_name_and_levels(self) -> None:
        """Each config dict must have 'name' and 'levels' keys."""
        for config in PARTIAL_CONFIGS:
            assert "name" in config, f"Config missing 'name': {config}"
            assert "levels" in config, f"Config missing 'levels': {config}"

    def test_all_level_indices_valid(self) -> None:
        """All level indices in every config must be in range 0-4."""
        for config in PARTIAL_CONFIGS:
            for level in config["levels"]:
                assert 0 <= level <= 4, (
                    "Level %d out of range in config %r" % (level, config["name"])
                )

    def test_each_config_has_two_to_four_levels(self) -> None:
        """Each config should specify 2-4 levels (partial, not full)."""
        for config in PARTIAL_CONFIGS:
            n_levels = len(config["levels"])
            assert 2 <= n_levels <= 4, (
                "Config %r has %d levels, expected 2-4"
                % (config["name"], n_levels)
            )

    def test_all_configs_include_l1_and_l3(self) -> None:
        """L1 (system prompt) and L3 (user message) should be in every config."""
        for config in PARTIAL_CONFIGS:
            assert 1 in config["levels"], (
                "Config %r missing L1" % config["name"]
            )
            assert 3 in config["levels"], (
                "Config %r missing L3" % config["name"]
            )


class TestBuildPartialExamples:
    """Tests for build_partial_examples()."""

    def test_produces_correct_total_count(self) -> None:
        """Should produce exactly 4 * per_config_count examples."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=10,
            seed=42,
        )
        assert len(result) == 4 * 10

    def test_levels_present_matches_config(self) -> None:
        """Each example's levels_present should match its config's levels."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        # Collect all distinct levels_present tuples
        all_level_sets = {tuple(ex["levels_present"]) for ex in result}
        # Every levels_present must be a subset of some config's levels
        config_level_sets = {tuple(sorted(c["levels"])) for c in PARTIAL_CONFIGS}
        for lp in all_level_sets:
            assert tuple(sorted(lp)) in config_level_sets or (
                # L4 might be dropped if no l4_content available
                any(
                    set(lp).issubset(set(c["levels"]))
                    for c in PARTIAL_CONFIGS
                )
            )

    def test_all_examples_not_conflict(self) -> None:
        """All partial examples must have is_conflict=False."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        for ex in result:
            assert ex["is_conflict"] is False
            assert ex["conflict_type"] is None

    def test_l0_absent_when_not_in_levels(self) -> None:
        """L0 delimiter tokens should not appear when 0 is not in config levels."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        # Find the L1+L3 only config (no L0)
        l1_l3_config = next(
            c for c in PARTIAL_CONFIGS
            if 0 not in c["levels"] and 4 not in c["levels"]
        )
        l1_l3_levels = sorted(l1_l3_config["levels"])

        for ex in result:
            if sorted(ex["levels_present"]) == l1_l3_levels:
                assert "<|L0_START|>" not in ex["text"]
                assert "<|L0_END|>" not in ex["text"]

    def test_l2_absent_when_not_in_levels(self) -> None:
        """L2 delimiter tokens should not appear when 2 is not in config levels."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        # Check examples where L2 is absent
        for ex in result:
            if 2 not in ex["levels_present"]:
                assert "<|L2_START|>" not in ex["text"]
                assert "<|L2_END|>" not in ex["text"]

    def test_l4_absent_when_not_in_levels(self) -> None:
        """L4 delimiter tokens should not appear when 4 is not in config levels."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        for ex in result:
            if 4 not in ex["levels_present"]:
                assert "<|L4_START|>" not in ex["text"]
                assert "<|L4_END|>" not in ex["text"]

    def test_l1_and_l3_always_present(self) -> None:
        """L1 and L3 delimiter tokens should always appear in every example."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        for ex in result:
            assert "<|L1_START|>" in ex["text"]
            assert "<|L1_END|>" in ex["text"]
            assert "<|L3_START|>" in ex["text"]
            assert "<|L3_END|>" in ex["text"]

    def test_response_block_always_present(self) -> None:
        """Every example should have RESP_START and RESP_END tokens."""
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
        )
        for ex in result:
            assert "<|RESP_START|>" in ex["text"]
            assert "<|RESP_END|>" in ex["text"]

    def test_l4_config_skipped_when_no_content_available(self) -> None:
        """If L4 is in config levels but no l4_content exists, produce 0 examples for that config."""
        rows = _make_base_rows(200)
        # Empty L4 lookup -- no L4 content for any row
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup={},
            per_config_count=5,
            seed=42,
        )
        # 3 configs produce 5 each; L1+L3+L4 produces 0 (no L4 rows)
        assert len(result) == 3 * 5
        for ex in result:
            assert 4 not in ex["levels_present"]
            assert "<|L4_START|>" not in ex["text"]

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical results."""
        rows = _make_base_rows(200)
        lookup = _make_l4_lookup(n=200)
        r1 = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=lookup,
            per_config_count=5,
            seed=99,
        )
        r2 = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=lookup,
            per_config_count=5,
            seed=99,
        )
        assert r1 == r2

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
        # No L4 lookup -- L1+L3+L4 config will produce 0 examples
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup={},
            per_config_count=2,
            seed=42,
        )
        # 3 configs produce 2 each; L1+L3+L4 produces 0 (no L4 rows)
        assert len(result) == 3 * 2
        responses = [r["response"] for r in rows]
        # At least some examples should contain the Dolly response text
        found = False
        for ex in result:
            if any(resp in ex["text"] for resp in responses):
                found = True
                break
        assert found, (
            "Expected at least one example to contain a Dolly response"
        )


class TestPartialMetadata:
    """Tests for metadata in partial examples."""

    def test_sft_category_set(self) -> None:
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5, seed=42,
        )
        categories = {ex["sft_category"] for ex in result}
        assert "partial_L1+L3" in categories
        assert "partial_L0+L1+L3" in categories
        assert "partial_L0+L1+L2+L3" in categories

    def test_provenance_set(self) -> None:
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5, seed=42,
        )
        for ex in result:
            assert ex["sft_source"] in ("alpaca", "dolly")
            assert isinstance(ex["sft_index"], int)

    def test_l4_generation_set_for_l4_config(self) -> None:
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows, l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(), l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5, seed=42,
        )
        for ex in result:
            if 4 in ex["levels_present"]:
                assert ex["l4_generation"] in ("wrapped", "synthesized")
            else:
                assert ex["l4_generation"] is None


class TestPartialWithL2Client:
    """Tests for build_partial_examples() with L2 generation client."""

    def test_l2_client_used_for_l2_config(self) -> None:
        """L0+L1+L2+L3 config should use client for L2 generation."""
        from unittest.mock import MagicMock
        rows = _make_base_rows(200)
        client = MagicMock()
        client.generate = MagicMock(
            return_value="Session config: Respond in English. Tone: casual."
        )
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
            openai_client=client,
        )
        # Only L0+L1+L2+L3 config has L2 (5 examples)
        assert client.generate.call_count == 5
        # Total should still be 4 * 5 = 20
        assert len(result) == 20

    def test_no_client_falls_back_to_template(self) -> None:
        rows = _make_base_rows(200)
        result = build_partial_examples(
            base_rows=rows,
            l0_rules=_make_l0_rules(),
            l1_library=_make_l1_library(),
            l4_lookup=_make_l4_lookup(n=200),
            per_config_count=5,
            seed=42,
            openai_client=None,
        )
        assert len(result) == 20
