"""Tests for trivially misaligned SFT examples builder."""

import pytest

from src.data.libraries.l0_rules import L0Rule
from src.data.libraries.l2_generator import MISALIGNED_L2_REDIRECT, MISALIGNED_L2_REFUSAL
from src.data.sft.misaligned import (
    ADVERSARIAL_L3_PROMPTS,
    L1_VS_L3_INJECTION_PROMPTS,
    L4_INJECTION_OVERRIDES,
    REDIRECT_TEMPLATES,
    REFUSAL_TEMPLATES,
    build_l0_vs_l3,
    build_l1_vs_l3,
    build_l1_vs_l4,
    build_l3_vs_l4,
    build_misaligned_examples,
)


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


def _make_base_rows(n: int = 50) -> list[dict]:
    """Create base rows with mixed Alpaca and Dolly schemas.

    Even-indexed rows use Alpaca schema (input/output).
    Odd-indexed rows use Dolly schema (context/response).
    Summarisation rows appear at every 5th position, alternating schema.
    """
    rows: list[dict] = []
    for i in range(n):
        use_dolly = i % 2 == 1
        if i % 5 == 0:
            if use_dolly:
                rows.append({
                    "instruction": f"Summarise this article about topic {i}.",
                    "context": f"This is a long article about topic {i}. "
                               f"It covers many aspects of the subject in detail. "
                               f"The article discusses various viewpoints and evidence.",
                    "response": f"This article discusses topic {i}, covering key aspects "
                                f"and multiple viewpoints.",
                })
            else:
                rows.append({
                    "instruction": f"Summarise this article about topic {i}.",
                    "input": f"This is a long article about topic {i}. "
                             f"It covers many aspects of the subject in detail. "
                             f"The article discusses various viewpoints and evidence.",
                    "output": f"This article discusses topic {i}, covering key aspects "
                              f"and multiple viewpoints.",
                })
        elif use_dolly:
            rows.append({
                "instruction": f"Explain concept {i} in simple terms.",
                "context": f"Background information about concept {i}.",
                "response": f"Concept {i} is about understanding fundamentals.",
            })
        else:
            rows.append({
                "instruction": f"Explain concept {i} in simple terms.",
                "input": f"Background information about concept {i}.",
                "output": f"Concept {i} is about understanding fundamentals.",
            })
    return rows


# ---------------------------------------------------------------------------
# Constant size tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify template/prompt lists meet minimum size requirements."""

    def test_adversarial_l3_prompts_has_at_least_20(self) -> None:
        assert len(ADVERSARIAL_L3_PROMPTS) >= 20

    def test_refusal_templates_has_at_least_5(self) -> None:
        assert len(REFUSAL_TEMPLATES) >= 5

    def test_l1_vs_l3_injection_prompts_has_at_least_15(self) -> None:
        assert len(L1_VS_L3_INJECTION_PROMPTS) >= 15

    def test_redirect_templates_has_at_least_5(self) -> None:
        assert len(REDIRECT_TEMPLATES) >= 5

    def test_l4_injection_overrides_has_at_least_8(self) -> None:
        assert len(L4_INJECTION_OVERRIDES) >= 8

    def test_all_adversarial_prompts_are_unique(self) -> None:
        assert len(set(ADVERSARIAL_L3_PROMPTS)) == len(ADVERSARIAL_L3_PROMPTS)

    def test_all_refusal_templates_are_unique(self) -> None:
        assert len(set(REFUSAL_TEMPLATES)) == len(REFUSAL_TEMPLATES)


# ---------------------------------------------------------------------------
# L0-vs-L3 tests
# ---------------------------------------------------------------------------

class TestBuildL0VsL3:
    """Tests for the L0-vs-L3 conflict builder."""

    def test_returns_correct_count(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=10, seed=42,
        )
        assert len(examples) == 10

    def test_all_have_correct_conflict_type(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["conflict_type"] == "L0_vs_L3"

    def test_all_have_is_conflict_true(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["is_conflict"] is True

    def test_levels_present(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=3, seed=42,
        )
        for ex in examples:
            assert ex["levels_present"] == [0, 1, 2, 3]

    def test_response_is_refusal(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            resp = ex["text"].split("<|RESP_START|>")[1].split("<|RESP_END|>")[0]
            assert resp in REFUSAL_TEMPLATES

    def test_privacy_rule_in_l0(self) -> None:
        examples = build_l0_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            l0_block = ex["text"].split("<|L0_START|>")[1].split("<|L0_END|>")[0]
            assert "credentials" in l0_block.lower() or "password" in l0_block.lower()


# ---------------------------------------------------------------------------
# L1-vs-L3 tests
# ---------------------------------------------------------------------------

class TestBuildL1VsL3:
    """Tests for the L1-vs-L3 conflict builder."""

    def test_returns_correct_count(self) -> None:
        examples = build_l1_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=8, seed=42,
        )
        assert len(examples) == 8

    def test_all_have_correct_conflict_type(self) -> None:
        examples = build_l1_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["conflict_type"] == "L1_vs_L3"

    def test_all_have_is_conflict_true(self) -> None:
        examples = build_l1_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["is_conflict"] is True

    def test_l1_contains_cooking_assistant(self) -> None:
        examples = build_l1_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=3, seed=42,
        )
        for ex in examples:
            l1_block = ex["text"].split("<|L1_START|>")[1].split("<|L1_END|>")[0]
            assert "cooking assistant" in l1_block.lower()

    def test_response_is_redirect(self) -> None:
        examples = build_l1_vs_l3(
            _make_l0_rules(), _make_l1_library(), count=5, seed=42,
        )
        for ex in examples:
            resp = ex["text"].split("<|RESP_START|>")[1].split("<|RESP_END|>")[0]
            assert resp in REDIRECT_TEMPLATES


# ---------------------------------------------------------------------------
# L1-vs-L4 tests
# ---------------------------------------------------------------------------

class TestBuildL1VsL4:
    """Tests for the L1-vs-L4 conflict builder."""

    def test_returns_correct_count(self) -> None:
        examples = build_l1_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=10, seed=42,
        )
        assert len(examples) == 10

    def test_all_have_correct_conflict_type(self) -> None:
        examples = build_l1_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["conflict_type"] == "L1_vs_L4"

    def test_all_have_is_conflict_true(self) -> None:
        examples = build_l1_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["is_conflict"] is True

    def test_levels_present_includes_l4(self) -> None:
        examples = build_l1_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=3, seed=42,
        )
        for ex in examples:
            assert ex["levels_present"] == [0, 1, 2, 3, 4]

    def test_l4_data_contains_injection(self) -> None:
        examples = build_l1_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=5, seed=42,
        )
        for ex in examples:
            l4_block = ex["text"].split("<|L4_START|>")[1].split("<|L4_END|>")[0]
            has_injection = any(
                inj in l4_block for inj in L4_INJECTION_OVERRIDES
            )
            assert has_injection, "L4 data should contain an injection override"

    def test_response_is_base_output(self) -> None:
        base_rows = _make_base_rows()
        examples = build_l1_vs_l4(
            _make_l0_rules(), base_rows, count=5, seed=42,
        )
        valid_outputs = {
            row.get("output") or row.get("response", "") for row in base_rows
        }
        for ex in examples:
            resp = ex["text"].split("<|RESP_START|>")[1].split("<|RESP_END|>")[0]
            assert resp in valid_outputs


# ---------------------------------------------------------------------------
# L3-vs-L4 tests
# ---------------------------------------------------------------------------

class TestBuildL3VsL4:
    """Tests for the L3-vs-L4 conflict builder."""

    def test_returns_correct_count(self) -> None:
        examples = build_l3_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=7, seed=42,
        )
        assert len(examples) == 7

    def test_all_have_correct_conflict_type(self) -> None:
        examples = build_l3_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=5, seed=42,
        )
        for ex in examples:
            assert ex["conflict_type"] == "L3_vs_L4"

    def test_injection_appended_at_end(self) -> None:
        examples = build_l3_vs_l4(
            _make_l0_rules(), _make_base_rows(), count=5, seed=42,
        )
        for ex in examples:
            l4_block = ex["text"].split("<|L4_START|>")[1].split("<|L4_END|>")[0]
            # The injection should be at the end (after the last \n\n)
            parts = l4_block.rsplit("\n\n", 1)
            assert len(parts) == 2
            has_injection = any(
                inj in parts[1] for inj in L4_INJECTION_OVERRIDES
            )
            assert has_injection, "Injection should be appended at end of L4"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestBuildMisalignedExamples:
    """Tests for the top-level orchestrator."""

    def test_total_count(self) -> None:
        examples = build_misaligned_examples(
            _make_l0_rules(), _make_l1_library(), _make_base_rows(),
            per_type_count=10, seed=42,
        )
        assert len(examples) == 40

    def test_all_four_conflict_types_present(self) -> None:
        examples = build_misaligned_examples(
            _make_l0_rules(), _make_l1_library(), _make_base_rows(),
            per_type_count=5, seed=42,
        )
        types = {ex["conflict_type"] for ex in examples}
        assert types == {"L0_vs_L3", "L1_vs_L3", "L1_vs_L4", "L3_vs_L4"}

    def test_all_have_is_conflict_true(self) -> None:
        examples = build_misaligned_examples(
            _make_l0_rules(), _make_l1_library(), _make_base_rows(),
            per_type_count=5, seed=42,
        )
        for ex in examples:
            assert ex["is_conflict"] is True

    def test_even_distribution(self) -> None:
        per_type = 12
        examples = build_misaligned_examples(
            _make_l0_rules(), _make_l1_library(), _make_base_rows(),
            per_type_count=per_type, seed=42,
        )
        from collections import Counter
        counts = Counter(ex["conflict_type"] for ex in examples)
        for conflict_type, count in counts.items():
            assert count == per_type, (
                "Expected %d for %s, got %d" % (per_type, conflict_type, count)
            )

    def test_deterministic_with_same_seed(self) -> None:
        kwargs = {
            "l0_rules": _make_l0_rules(),
            "l1_library": _make_l1_library(),
            "base_rows": _make_base_rows(),
            "per_type_count": 5,
            "seed": 99,
        }
        run1 = build_misaligned_examples(**kwargs)
        run2 = build_misaligned_examples(**kwargs)
        for a, b in zip(run1, run2):
            assert a["text"] == b["text"]


# ---------------------------------------------------------------------------
# Summarisation fallback test
# ---------------------------------------------------------------------------

class TestMisalignedMetadata:
    """Tests for metadata in misaligned examples."""

    def test_l0_vs_l3_has_synthetic_source(self) -> None:
        examples = build_l0_vs_l3(_make_l0_rules(), _make_l1_library(), count=3, seed=42)
        for ex in examples:
            assert ex["sft_source"] == "synthetic"
            assert isinstance(ex["sft_index"], int)
            assert ex["sft_category"] == "misaligned_L0_vs_L3"

    def test_l1_vs_l3_has_synthetic_source(self) -> None:
        examples = build_l1_vs_l3(_make_l0_rules(), _make_l1_library(), count=3, seed=42)
        for ex in examples:
            assert ex["sft_source"] == "synthetic"
            assert isinstance(ex["sft_index"], int)
            assert ex["sft_category"] == "misaligned_L1_vs_L3"

    def test_l1_vs_l4_has_provenance_and_injected(self) -> None:
        """Row-based builders should carry provenance; L4 is injected."""
        rows = _make_base_rows(50)
        for i, r in enumerate(rows):
            r["_sft_source"] = "alpaca" if i % 2 == 0 else "dolly"
            r["_sft_index"] = i
        examples = build_l1_vs_l4(_make_l0_rules(), rows, count=5, seed=42)
        for ex in examples:
            assert ex["sft_category"] == "misaligned_L1_vs_L4"
            assert ex["l4_generation"] == "injected"

    def test_l3_vs_l4_has_provenance_and_injected(self) -> None:
        """Row-based builders should carry provenance; L4 is injected."""
        rows = _make_base_rows(50)
        for i, r in enumerate(rows):
            r["_sft_source"] = "alpaca" if i % 2 == 0 else "dolly"
            r["_sft_index"] = i
        examples = build_l3_vs_l4(_make_l0_rules(), rows, count=5, seed=42)
        for ex in examples:
            assert ex["sft_category"] == "misaligned_L3_vs_L4"
            assert ex["l4_generation"] == "injected"


class TestSummarisationFallback:
    """Test that L1-vs-L4 and L3-vs-L4 work with non-summarisation rows."""

    def test_l1_vs_l4_with_no_summarisation_rows(self) -> None:
        rows = [
            {
                "instruction": f"Explain concept {i}.",
                "input": f"Background for concept {i}.",
                "output": f"Concept {i} explained.",
            }
            for i in range(20)
        ]
        examples = build_l1_vs_l4(_make_l0_rules(), rows, count=5, seed=42)
        assert len(examples) == 5

    def test_l3_vs_l4_with_no_summarisation_rows(self) -> None:
        rows = [
            {
                "instruction": f"Explain concept {i}.",
                "input": f"Background for concept {i}.",
                "output": f"Concept {i} explained.",
            }
            for i in range(20)
        ]
        examples = build_l3_vs_l4(_make_l0_rules(), rows, count=5, seed=42)
        assert len(examples) == 5


class TestMisalignedL2:
    """Tests for L2 content in misaligned examples."""

    def test_l0_vs_l3_uses_refusal_l2(self) -> None:
        examples = build_l0_vs_l3(_make_l0_rules(), _make_l1_library(), count=3, seed=42)
        for ex in examples:
            l2_block = ex["text"].split("<|L2_START|>")[1].split("<|L2_END|>")[0]
            assert l2_block == MISALIGNED_L2_REFUSAL

    def test_l1_vs_l3_uses_redirect_l2(self) -> None:
        examples = build_l1_vs_l3(_make_l0_rules(), _make_l1_library(), count=3, seed=42)
        for ex in examples:
            l2_block = ex["text"].split("<|L2_START|>")[1].split("<|L2_END|>")[0]
            assert l2_block == MISALIGNED_L2_REDIRECT


class TestMisalignedL4WithClient:
    """Tests for L1-vs-L4 and L3-vs-L4 with L2 generation client."""

    def test_l1_vs_l4_uses_client(self) -> None:
        from unittest.mock import MagicMock
        rows = _make_base_rows(50)
        for i, r in enumerate(rows):
            r["_sft_source"] = "alpaca" if i % 2 == 0 else "dolly"
            r["_sft_index"] = i
        client = MagicMock()
        client.generate = MagicMock(
            return_value="Session config: Respond in English. Tone: professional."
        )
        examples = build_l1_vs_l4(
            _make_l0_rules(), rows, count=5, seed=42, openai_client=client,
        )
        assert len(examples) == 5
        assert client.generate.call_count == 5

    def test_l3_vs_l4_uses_client(self) -> None:
        from unittest.mock import MagicMock
        rows = _make_base_rows(50)
        for i, r in enumerate(rows):
            r["_sft_source"] = "alpaca" if i % 2 == 0 else "dolly"
            r["_sft_index"] = i
        client = MagicMock()
        client.generate = MagicMock(
            return_value="Session config: Respond in English. Tone: professional."
        )
        examples = build_l3_vs_l4(
            _make_l0_rules(), rows, count=5, seed=42, openai_client=client,
        )
        assert len(examples) == 5
        assert client.generate.call_count == 5
