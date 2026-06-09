"""Tests for DPO prompt assembly and example creation."""

import json

import pytest

from src.data.dpo.assembly import assemble_dpo_prompt, assemble_dpo_example


def test_assemble_dpo_prompt_all_levels():
    prompt = assemble_dpo_prompt(
        l0_rules=["Rule 1", "Rule 2"],
        l1_prompt="You are a helpful assistant.",
        l2_config="Tone: professional.",
        l3_message="What is Python?",
        l4_data="<tool_output>Python is a language.</tool_output>",
    )
    assert "<|L0_START|>" in prompt
    assert "<|L0_END|>" in prompt
    assert "Rule 1\nRule 2" in prompt
    assert "<|L1_START|>You are a helpful assistant.<|L1_END|>" in prompt
    assert "<|L2_START|>Tone: professional.<|L2_END|>" in prompt
    assert "<|L3_START|>What is Python?<|L3_END|>" in prompt
    assert "<|L4_START|>" in prompt
    assert "<|RESP_START|>" not in prompt
    assert "<|RESP_END|>" not in prompt


def test_assemble_dpo_prompt_missing_levels():
    prompt = assemble_dpo_prompt(
        l0_rules=["Rule"],
        l1_prompt="Assistant.",
        l2_config=None,
        l3_message="Hello",
        l4_data=None,
    )
    assert "<|L2_START|>" not in prompt
    assert "<|L4_START|>" not in prompt
    assert "<|L0_START|>" in prompt
    assert "<|L1_START|>" in prompt
    assert "<|L3_START|>" in prompt


def test_assemble_dpo_prompt_empty_rules():
    prompt = assemble_dpo_prompt(
        l0_rules=[],
        l1_prompt="Assistant.",
        l2_config=None,
        l3_message="Hello",
        l4_data=None,
    )
    assert "<|L0_START|>" not in prompt


def test_assemble_dpo_example_schema():
    example = assemble_dpo_example(
        prompt="<|L0_START|>rule<|L0_END|>\n<|L1_START|>sys<|L1_END|>\n<|L3_START|>hello<|L3_END|>",
        chosen="Good answer",
        rejected="Bad answer",
        conflict_type="L1_vs_L3",
        victim_level=1,
        attacker_level=3,
        category="pairwise",
        levels_present=[0, 1, 3],
        attack_type="naive",
        yw_source="base_dataset",
        yl_source="gpt4o_mini",
    )
    assert example["chosen"] == "<|RESP_START|>Good answer<|RESP_END|>"
    assert example["rejected"] == "<|RESP_START|>Bad answer<|RESP_END|>"
    assert example["level_gap"] == 2
    assert example["margin"] == 2.0
    assert example["is_calibration"] is False
    assert "build_timestamp" in example
    assert "seed" in example
    # Check all schema fields exist
    for key in ["prompt", "chosen", "rejected", "conflict_type", "level_gap",
                 "margin", "category", "is_calibration", "attack_type",
                 "levels_present", "victim_level", "attacker_level",
                 "yw_source", "yw_model", "yw_base_dataset", "yw_base_index",
                 "yl_source", "yl_model", "yl_base_dataset", "yl_base_index",
                 "yl_fallback_used", "l0_rule_ids", "l1_domain", "l1_index",
                 "l2_source", "l2_model", "l4_source", "l4_base_dataset",
                 "l4_base_index", "injection_template_id", "injection_position",
                 "l2_conflict_attribute", "l2_conflict_value",
                 "cascading_chain", "cascading_resolution",
                 "seed", "build_timestamp"]:
        assert key in example, f"Missing key: {key}"


def test_assemble_dpo_example_calibration():
    example = assemble_dpo_example(
        prompt="...",
        chosen="Helpful response",
        rejected="I cannot do that.",
        conflict_type="calibration_L3",
        victim_level=3,
        attacker_level=3,
        category="calibration",
        levels_present=[0, 1, 2, 3],
        attack_type="benign",
        yw_source="gpt4o_mini",
        yl_source="template",
    )
    assert example["margin"] == 0.0
    assert example["is_calibration"] is True
    assert example["level_gap"] == 0


def test_assemble_dpo_example_cascading_margin_override():
    example = assemble_dpo_example(
        prompt="...",
        chosen="Redacted response",
        rejected="Full PII response",
        conflict_type="cascading_L0_L1_L2_L3",
        victim_level=0,
        attacker_level=3,
        category="cascading",
        levels_present=[0, 1, 2, 3],
        cascading_chain="L0>L1>L2>L3",
        margin_override=3.0,
    )
    assert example["margin"] == 3.0
    assert example["level_gap"] == 3
    assert example["cascading_chain"] == "L0>L1>L2>L3"


def test_assemble_dpo_example_cascading_partial_chain():
    example = assemble_dpo_example(
        prompt="...",
        chosen="Markdown response",
        rejected="JSON response",
        conflict_type="cascading_L1_L2_L3",
        victim_level=1,
        attacker_level=3,
        category="cascading",
        levels_present=[0, 1, 2, 3],
        cascading_chain="L1>L2>L3",
        margin_override=2.0,
    )
    assert example["margin"] == 2.0
    assert example["level_gap"] == 2


def test_assemble_dpo_example_serializable():
    example = assemble_dpo_example(
        prompt="...",
        chosen="answer",
        rejected="wrong",
        conflict_type="L0_vs_L1",
        victim_level=0,
        attacker_level=1,
        category="pairwise",
        levels_present=[0, 1, 2, 3, 4],
    )
    serialized = json.dumps(example)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["margin"] == 1.0


def test_assemble_dpo_example_rejects_negative_victim():
    with pytest.raises(ValueError, match="victim_level"):
        assemble_dpo_example(
            prompt="...", chosen="a", rejected="b",
            conflict_type="test", victim_level=-1, attacker_level=3,
            category="pairwise", levels_present=[0, 1, 3],
        )


def test_assemble_dpo_example_rejects_out_of_range_attacker():
    with pytest.raises(ValueError, match="attacker_level"):
        assemble_dpo_example(
            prompt="...", chosen="a", rejected="b",
            conflict_type="test", victim_level=0, attacker_level=5,
            category="pairwise", levels_present=[0, 1, 3],
        )


def test_assemble_dpo_example_rejects_victim_above_attacker_non_calibration():
    with pytest.raises(ValueError, match="victim_level.*must.*<=.*attacker_level"):
        assemble_dpo_example(
            prompt="...", chosen="a", rejected="b",
            conflict_type="test", victim_level=3, attacker_level=1,
            category="pairwise", levels_present=[0, 1, 3],
        )
