"""Tests for SFT assembly layer: empty L4 guard and metadata fields."""

from src.data.sft.assembly import assemble_instance, assemble_sft_example


class TestAssembleInstanceEmptyL4Guard:
    """assemble_instance should treat empty-string l4_data as absent."""

    def test_empty_string_l4_excluded(self) -> None:
        result = assemble_instance(l3_message="hello", l4_data="")
        assert "<|L4_START|>" not in result
        assert "<|L4_END|>" not in result

    def test_whitespace_only_l4_excluded(self) -> None:
        result = assemble_instance(l3_message="hello", l4_data="   ")
        assert "<|L4_START|>" not in result

    def test_none_l4_excluded(self) -> None:
        result = assemble_instance(l3_message="hello", l4_data=None)
        assert "<|L4_START|>" not in result

    def test_nonempty_l4_included(self) -> None:
        result = assemble_instance(l3_message="hello", l4_data="real data")
        assert "<|L4_START|>real data<|L4_END|>" in result


class TestAssembleSftExampleMetadata:
    """assemble_sft_example should include new metadata fields."""

    def test_metadata_fields_present(self) -> None:
        ex = assemble_sft_example(
            response="resp",
            levels_present=[0, 1, 2, 3],
            is_conflict=False,
            l0_rules=["rule1"],
            l1_prompt="sys",
            l2_config="cfg",
            l3_message="msg",
            sft_source="alpaca",
            sft_index=42,
            sft_category="simple_aligned",
            l4_generation=None,
        )
        assert ex["sft_source"] == "alpaca"
        assert ex["sft_index"] == 42
        assert ex["sft_category"] == "simple_aligned"
        assert ex["l4_generation"] is None

    def test_metadata_defaults_to_none(self) -> None:
        ex = assemble_sft_example(
            response="resp",
            levels_present=[1, 3],
            is_conflict=False,
            l1_prompt="sys",
            l3_message="msg",
        )
        assert ex["sft_source"] is None
        assert ex["sft_index"] is None
        assert ex["sft_category"] is None
        assert ex["l4_generation"] is None

    def test_l4_generation_carried_through(self) -> None:
        ex = assemble_sft_example(
            response="resp",
            levels_present=[0, 1, 2, 3, 4],
            is_conflict=False,
            l0_rules=["rule1"],
            l1_prompt="sys",
            l2_config="cfg",
            l3_message="msg",
            l4_data="tool data",
            l4_generation="wrapped",
        )
        assert ex["l4_generation"] == "wrapped"
