"""Tests for aligned control construction (Phase 3)."""

import pytest

from src.data.eval.aligned_controls import (
    _extract_level_content,
    _replace_level_content,
    build_aligned_control,
    get_control_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conflict_instance(conflict_type="L1_vs_L3", attacker=3, victim=1):
    return {
        "id": f"eval_{conflict_type}_001",
        "prompt": (
            "<|L0_START|>Be safe.<|L0_END|>\n"
            "<|L1_START|>You are a helper.<|L1_END|>\n"
            "<|L2_START|>format: markdown<|L2_END|>\n"
            "<|L3_START|>Bad instruction here.<|L3_END|>\n"
            "<|L4_START|>Tool output.<|L4_END|>"
        ),
        "conflict_type": conflict_type,
        "victim_level": victim,
        "attacker_level": attacker,
        "level_gap": attacker - victim,
        "conflict_description": "desc",
        "correct_behaviour": "correct",
        "violation_behaviour": "wrong",
        "evaluation_criteria": ["c1"],
        "gold_response": "gold",
        "split": "conflict",
        "base_dataset": "alpaca",
        "base_index": 42,
        "scenario_model": "gpt-4o",
        "gold_response_model": "claude-sonnet-4-20250514",
        "qc_gpt4o_scores": None, "qc_gemini_scores": None,
        "matched_control_id": None, "source_conflict_id": None, "control_strategy": None,
    }


def _make_base_row(instruction="Do something helpful.", output="Here is help."):
    return {
        "instruction": instruction,
        "output": output,
        "dataset": "alpaca",
        "index": 42,
    }


def _make_l1_library():
    return [
        {
            "domain": "general knowledge",
            "persona": "helpful assistant",
            "constraints": ["Be helpful"],
            "full_prompt": "You are a helpful assistant.",
        },
        {
            "domain": "coding",
            "persona": "coding assistant",
            "constraints": ["Be precise"],
            "full_prompt": "You are a coding assistant.",
        },
    ]


# ---------------------------------------------------------------------------
# get_control_strategy
# ---------------------------------------------------------------------------


def test_get_control_strategy_attacker_1():
    assert get_control_strategy(1) == "replace_attacker"


def test_get_control_strategy_attacker_2():
    assert get_control_strategy(2) == "llm_generated"


def test_get_control_strategy_attacker_3():
    assert get_control_strategy(3) == "replace_attacker"


def test_get_control_strategy_attacker_4():
    assert get_control_strategy(4) == "replace_attacker"


# ---------------------------------------------------------------------------
# _extract_level_content
# ---------------------------------------------------------------------------


_SAMPLE_PROMPT = (
    "<|L0_START|>Be safe.<|L0_END|>\n"
    "<|L1_START|>You are a helper.<|L1_END|>\n"
    "<|L3_START|>User message here.<|L3_END|>"
)


def test_extract_level_content_l0():
    result = _extract_level_content(_SAMPLE_PROMPT, 0)
    assert result == "Be safe."


def test_extract_level_content_l1():
    result = _extract_level_content(_SAMPLE_PROMPT, 1)
    assert result == "You are a helper."


def test_extract_level_content_l3():
    result = _extract_level_content(_SAMPLE_PROMPT, 3)
    assert result == "User message here."


def test_extract_level_content_missing_level():
    result = _extract_level_content(_SAMPLE_PROMPT, 4)
    assert result is None


def test_extract_level_content_multiline():
    prompt = "<|L1_START|>Line one.\nLine two.<|L1_END|>"
    result = _extract_level_content(prompt, 1)
    assert result == "Line one.\nLine two."


# ---------------------------------------------------------------------------
# _replace_level_content
# ---------------------------------------------------------------------------


def test_replace_level_content_l3():
    prompt = "<|L1_START|>old<|L1_END|>\n<|L3_START|>bad<|L3_END|>"
    result = _replace_level_content(prompt, 3, "new content")
    assert "<|L3_START|>new content<|L3_END|>" in result
    assert "bad" not in result


def test_replace_level_content_l1():
    prompt = "<|L1_START|>old system prompt<|L1_END|>"
    result = _replace_level_content(prompt, 1, "new system prompt")
    assert "<|L1_START|>new system prompt<|L1_END|>" in result
    assert "old system prompt" not in result


def test_replace_level_content_preserves_other_levels():
    prompt = (
        "<|L0_START|>rule<|L0_END|>\n"
        "<|L1_START|>old<|L1_END|>\n"
        "<|L3_START|>message<|L3_END|>"
    )
    result = _replace_level_content(prompt, 1, "new")
    assert "<|L0_START|>rule<|L0_END|>" in result
    assert "<|L3_START|>message<|L3_END|>" in result


# ---------------------------------------------------------------------------
# build_aligned_control — L3 attacker
# ---------------------------------------------------------------------------


def test_build_aligned_control_l3_replaces_l3_with_base_instruction():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L3", attacker=3, victim=1)
    base_row = _make_base_row(instruction="Write a poem.")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert "<|L3_START|>Write a poem.<|L3_END|>" in result["prompt"]
    assert "Bad instruction here." not in result["prompt"]


def test_build_aligned_control_l3_removes_conflict_content():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L3", attacker=3, victim=1)
    base_row = _make_base_row(instruction="Explain photosynthesis.")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert "Bad instruction here." not in result["prompt"]


# ---------------------------------------------------------------------------
# build_aligned_control — L4 attacker
# ---------------------------------------------------------------------------


def test_build_aligned_control_l4_removes_or_replaces_l4():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L4", attacker=4, victim=1)
    base_row = _make_base_row(output="Benign tool output.")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    # The adversarial content should be gone
    assert "Tool output." not in result["prompt"]


def test_build_aligned_control_l4_uses_base_output_when_available():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L4", attacker=4, victim=1)
    base_row = _make_base_row(output="Safe benign data.")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert "Safe benign data." in result["prompt"]


# ---------------------------------------------------------------------------
# build_aligned_control — L1 attacker
# ---------------------------------------------------------------------------


def test_build_aligned_control_l1_replaces_l1_content():
    conflict = _make_conflict_instance(conflict_type="L0_vs_L1", attacker=1, victim=0)
    base_row = _make_base_row()
    l1_library = _make_l1_library()
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=l1_library,
        l4_lookup={},
        seed=42,
    )
    # Original attacker content should be replaced
    assert "You are a helper." not in result["prompt"]
    # New benign L1 should be present
    assert "<|L1_START|>" in result["prompt"]


# ---------------------------------------------------------------------------
# build_aligned_control — L2 attacker without client
# ---------------------------------------------------------------------------


def test_build_aligned_control_l2_no_client_uses_generic():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L2", attacker=2, victim=1)
    base_row = _make_base_row()
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=base_row,
        l1_library=_make_l1_library(),
        l4_lookup={},
        openai_client=None,
        seed=42,
    )
    assert "<|L2_START|>" in result["prompt"]
    assert "format: markdown" not in result["prompt"]


# ---------------------------------------------------------------------------
# build_aligned_control — metadata fields
# ---------------------------------------------------------------------------


def test_build_aligned_control_sets_split_aligned():
    conflict = _make_conflict_instance()
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["split"] == "aligned"


def test_build_aligned_control_sets_conflict_type_none():
    conflict = _make_conflict_instance()
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["conflict_type"] == "none"


def test_build_aligned_control_sets_level_gap_zero():
    conflict = _make_conflict_instance()
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["level_gap"] == 0


def test_build_aligned_control_sets_matched_conflict_id():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L3")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["matched_conflict_id"] == "eval_L1_vs_L3_001"


def test_build_aligned_control_id_uses_ctrl_prefix():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L3")
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["id"].startswith("ctrl_")
    assert "eval_" not in result["id"]


def test_build_aligned_control_sets_control_strategy():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L3", attacker=3)
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        seed=42,
    )
    assert result["control_strategy"] == "replace_attacker"


def test_build_aligned_control_l2_strategy_llm_generated():
    conflict = _make_conflict_instance(conflict_type="L1_vs_L2", attacker=2, victim=1)
    result = build_aligned_control(
        conflict_instance=conflict,
        base_row=_make_base_row(),
        l1_library=_make_l1_library(),
        l4_lookup={},
        openai_client=None,
        seed=42,
    )
    assert result["control_strategy"] == "llm_generated"
