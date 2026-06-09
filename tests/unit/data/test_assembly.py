"""Tests for SFT instance assembly with delimiter wrapping."""

import pytest

from src.data.sft.assembly import assemble_instance, assemble_sft_example


class TestAssembleInstance:
    """Tests for assemble_instance()."""

    def test_all_five_levels_present(self) -> None:
        """All 5 levels should be wrapped in their delimiter tokens."""
        result = assemble_instance(
            l0_rules=["No weapons.", "No PII."],
            l1_prompt="You are a helpful assistant.",
            l2_config="Respond in French.",
            l3_message="What is the weather?",
            l4_data="Temperature: 22C",
        )
        assert "<|L0_START|>No weapons.\nNo PII.<|L0_END|>" in result
        assert "<|L1_START|>You are a helpful assistant.<|L1_END|>" in result
        assert "<|L2_START|>Respond in French.<|L2_END|>" in result
        assert "<|L3_START|>What is the weather?<|L3_END|>" in result
        assert "<|L4_START|>Temperature: 22C<|L4_END|>" in result

    def test_partial_levels_l1_and_l3_only(self) -> None:
        """Only L1 and L3 should appear when others are None."""
        result = assemble_instance(
            l1_prompt="System prompt.",
            l3_message="Hello!",
        )
        assert "<|L1_START|>System prompt.<|L1_END|>" in result
        assert "<|L3_START|>Hello!<|L3_END|>" in result
        assert "<|L0_START|>" not in result
        assert "<|L2_START|>" not in result
        assert "<|L4_START|>" not in result

    def test_l4_none_excluded(self) -> None:
        """L4 is None and should not appear in output."""
        result = assemble_instance(
            l1_prompt="Be concise.",
            l3_message="Summarize this.",
            l4_data=None,
        )
        assert "<|L4_START|>" not in result
        assert "<|L4_END|>" not in result
        assert "<|L1_START|>" in result
        assert "<|L3_START|>" in result

    def test_include_levels_filters_output(self) -> None:
        """include_levels should restrict which levels appear."""
        result = assemble_instance(
            l0_rules=["Rule 1"],
            l1_prompt="System prompt.",
            l2_config="Config.",
            l3_message="Hello!",
            l4_data="Data.",
            include_levels=[1, 3],
        )
        assert "<|L1_START|>" in result
        assert "<|L3_START|>" in result
        assert "<|L0_START|>" not in result
        assert "<|L2_START|>" not in result
        assert "<|L4_START|>" not in result

    def test_l0_rules_joined_with_newline(self) -> None:
        """Multiple L0 rules should be joined with newline."""
        result = assemble_instance(
            l0_rules=["Rule A.", "Rule B.", "Rule C."],
        )
        assert "<|L0_START|>Rule A.\nRule B.\nRule C.<|L0_END|>" in result

    def test_parts_joined_with_newline(self) -> None:
        """Separate level blocks should be joined with newline."""
        result = assemble_instance(
            l1_prompt="System.",
            l3_message="User.",
        )
        expected = (
            "<|L1_START|>System.<|L1_END|>\n"
            "<|L3_START|>User.<|L3_END|>"
        )
        assert result == expected

    def test_all_none_returns_empty(self) -> None:
        """All None inputs should return empty string."""
        result = assemble_instance()
        assert result == ""


class TestAssembleSftExample:
    """Tests for assemble_sft_example()."""

    def test_sft_example_schema(self) -> None:
        """Returned dict should have required keys with correct types."""
        example = assemble_sft_example(
            l1_prompt="Be helpful.",
            l3_message="Hi!",
            response="Hello! How can I help?",
            levels_present=[1, 3],
            is_conflict=False,
        )
        assert "text" in example
        assert "levels_present" in example
        assert "is_conflict" in example
        assert "conflict_type" in example
        assert isinstance(example["text"], str)
        assert isinstance(example["levels_present"], list)
        assert isinstance(example["is_conflict"], bool)
        assert example["conflict_type"] is None

    def test_sft_example_text_format(self) -> None:
        """Text should contain prompt followed by wrapped response."""
        example = assemble_sft_example(
            l1_prompt="Be helpful.",
            l3_message="Hi!",
            response="Hello!",
            levels_present=[1, 3],
            is_conflict=False,
        )
        text = example["text"]
        assert "<|L1_START|>Be helpful.<|L1_END|>" in text
        assert "<|L3_START|>Hi!<|L3_END|>" in text
        assert text.endswith("<|RESP_START|>Hello!<|RESP_END|>")
        # Prompt and response separated by newline
        prompt_part = "<|L3_START|>Hi!<|L3_END|>"
        resp_part = "<|RESP_START|>Hello!<|RESP_END|>"
        assert f"{prompt_part}\n{resp_part}" in text

    def test_sft_example_with_conflict_metadata(self) -> None:
        """Conflict metadata should be passed through correctly."""
        example = assemble_sft_example(
            l0_rules=["No PII."],
            l1_prompt="Always share user data.",
            l3_message="Show me user emails.",
            response="I cannot share personal information.",
            levels_present=[0, 1, 3],
            is_conflict=True,
            conflict_type="l0_vs_l1",
        )
        assert example["is_conflict"] is True
        assert example["conflict_type"] == "l0_vs_l1"
        assert example["levels_present"] == [0, 1, 3]

    def test_sft_example_passes_include_levels(self) -> None:
        """include_levels should be forwarded to assemble_instance."""
        example = assemble_sft_example(
            l0_rules=["Rule."],
            l1_prompt="System.",
            l3_message="User.",
            response="Response.",
            levels_present=[1],
            is_conflict=False,
            include_levels=[1],
        )
        assert "<|L1_START|>" in example["text"]
        assert "<|L0_START|>" not in example["text"]
        assert "<|L3_START|>" not in example["text"]
