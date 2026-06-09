"""Tests for reference baseline construction (Phase 4)."""

from src.data.eval.reference_baselines import (
    build_reference_baseline,
    sample_for_reference,
    strip_delimiters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DELIMITED_PROMPT = (
    "<|L0_START|>Be safe.<|L0_END|>\n"
    "<|L1_START|>You are a helper.<|L1_END|>\n"
    "<|L2_START|>format: markdown<|L2_END|>\n"
    "<|L3_START|>Write a poem.<|L3_END|>\n"
    "<|L4_START|>Tool output here.<|L4_END|>"
)


def _make_conflict(conflict_type="L1_vs_L3", idx=0):
    return {
        "id": f"eval_{conflict_type}_{idx:03d}",
        "prompt": (
            "<|L0_START|>Be safe.<|L0_END|>\n"
            "<|L1_START|>You are a helper.<|L1_END|>\n"
            "<|L2_START|>format: markdown<|L2_END|>\n"
            "<|L3_START|>Write a poem.<|L3_END|>\n"
            "<|L4_START|>Tool output here.<|L4_END|>"
        ),
        "conflict_type": conflict_type,
        "victim_level": 1, "attacker_level": 3, "level_gap": 2,
        "conflict_description": "desc", "correct_behaviour": "correct",
        "violation_behaviour": "wrong", "evaluation_criteria": ["c1"],
        "gold_response": "gold", "split": "conflict",
        "base_dataset": "alpaca", "base_index": idx,
    }


# ---------------------------------------------------------------------------
# strip_delimiters
# ---------------------------------------------------------------------------


def test_strip_delimiters_removes_all_l0_tokens():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "<|L0_START|>" not in result
    assert "<|L0_END|>" not in result


def test_strip_delimiters_removes_all_l1_tokens():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "<|L1_START|>" not in result
    assert "<|L1_END|>" not in result


def test_strip_delimiters_removes_all_l2_tokens():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "<|L2_START|>" not in result
    assert "<|L2_END|>" not in result


def test_strip_delimiters_removes_all_l3_tokens():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "<|L3_START|>" not in result
    assert "<|L3_END|>" not in result


def test_strip_delimiters_removes_all_l4_tokens():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "<|L4_START|>" not in result
    assert "<|L4_END|>" not in result


def test_strip_delimiters_removes_resp_tokens():
    prompt = "<|RESP_START|>Some response.<|RESP_END|>"
    result = strip_delimiters(prompt)
    assert "<|RESP_START|>" not in result
    assert "<|RESP_END|>" not in result


def test_strip_delimiters_preserves_content():
    result = strip_delimiters(_DELIMITED_PROMPT)
    assert "Be safe." in result
    assert "You are a helper." in result
    assert "format: markdown" in result
    assert "Write a poem." in result
    assert "Tool output here." in result


def test_strip_delimiters_no_delimiter_tokens_remain():
    result = strip_delimiters(_DELIMITED_PROMPT)
    import re
    tokens = re.findall(r"<\|[A-Z0-9_]+\|>", result)
    assert tokens == [], "Unexpected tokens remaining: %s" % tokens


def test_strip_delimiters_collapses_triple_newlines():
    prompt = "Line one.\n\n\nLine two."
    result = strip_delimiters(prompt)
    assert "\n\n\n" not in result
    assert "Line one." in result
    assert "Line two." in result


def test_strip_delimiters_strips_outer_whitespace():
    prompt = "   <|L1_START|>Hello<|L1_END|>   "
    result = strip_delimiters(prompt)
    assert result == result.strip()


def test_strip_delimiters_plain_text_unchanged():
    plain = "No delimiters here."
    result = strip_delimiters(plain)
    assert result == plain


# ---------------------------------------------------------------------------
# build_reference_baseline
# ---------------------------------------------------------------------------


def test_build_reference_baseline_sets_split_reference():
    conflict = _make_conflict()
    result = build_reference_baseline(conflict)
    assert result["split"] == "reference"


def test_build_reference_baseline_sets_source_conflict_id():
    conflict = _make_conflict(conflict_type="L1_vs_L3", idx=5)
    result = build_reference_baseline(conflict)
    assert result["source_conflict_id"] == "eval_L1_vs_L3_005"


def test_build_reference_baseline_id_uses_ref_prefix():
    conflict = _make_conflict(conflict_type="L1_vs_L3", idx=0)
    result = build_reference_baseline(conflict)
    assert result["id"].startswith("ref_")
    assert "eval_" not in result["id"]


def test_build_reference_baseline_id_replaces_only_first_eval():
    conflict = _make_conflict(conflict_type="L1_vs_L3", idx=0)
    conflict["id"] = "eval_L1_vs_L3_000"
    result = build_reference_baseline(conflict)
    assert result["id"] == "ref_L1_vs_L3_000"


def test_build_reference_baseline_strips_delimiters_from_prompt():
    conflict = _make_conflict()
    result = build_reference_baseline(conflict)
    assert "<|L0_START|>" not in result["prompt"]
    assert "<|L3_END|>" not in result["prompt"]


def test_build_reference_baseline_prompt_preserves_content():
    conflict = _make_conflict()
    result = build_reference_baseline(conflict)
    assert "Be safe." in result["prompt"]
    assert "Write a poem." in result["prompt"]


def test_build_reference_baseline_does_not_mutate_original():
    conflict = _make_conflict()
    original_prompt = conflict["prompt"]
    original_split = conflict["split"]
    build_reference_baseline(conflict)
    assert conflict["prompt"] == original_prompt
    assert conflict["split"] == original_split


def test_build_reference_baseline_preserves_other_fields():
    conflict = _make_conflict()
    result = build_reference_baseline(conflict)
    assert result["conflict_type"] == conflict["conflict_type"]
    assert result["victim_level"] == conflict["victim_level"]
    assert result["evaluation_criteria"] == conflict["evaluation_criteria"]


# ---------------------------------------------------------------------------
# sample_for_reference
# ---------------------------------------------------------------------------


def test_sample_for_reference_correct_total_count():
    instances = (
        [_make_conflict("L1_vs_L3", i) for i in range(50)]
        + [_make_conflict("L0_vs_L2", i) for i in range(50)]
    )
    results = sample_for_reference(instances, per_pair=30, seed=42)
    assert len(results) == 60


def test_sample_for_reference_correct_count_per_type():
    instances = (
        [_make_conflict("L1_vs_L3", i) for i in range(50)]
        + [_make_conflict("L0_vs_L2", i) for i in range(50)]
    )
    results = sample_for_reference(instances, per_pair=30, seed=42)
    types = [r["conflict_type"] for r in results]
    assert types.count("L1_vs_L3") == 30
    assert types.count("L0_vs_L2") == 30


def test_sample_for_reference_takes_all_when_fewer_than_per_pair():
    instances = [_make_conflict("L1_vs_L3", i) for i in range(10)]
    results = sample_for_reference(instances, per_pair=30, seed=42)
    assert len(results) == 10


def test_sample_for_reference_builds_reference_baselines():
    instances = [_make_conflict("L1_vs_L3", i) for i in range(5)]
    results = sample_for_reference(instances, per_pair=5, seed=42)
    for r in results:
        assert r["split"] == "reference"
        assert r["id"].startswith("ref_")
        assert "<|L0_START|>" not in r["prompt"]


def test_sample_for_reference_reproducible_with_same_seed():
    instances = [_make_conflict("L1_vs_L3", i) for i in range(100)]
    results_a = sample_for_reference(instances, per_pair=30, seed=99)
    results_b = sample_for_reference(instances, per_pair=30, seed=99)
    assert [r["id"] for r in results_a] == [r["id"] for r in results_b]


def test_sample_for_reference_different_seeds_give_different_results():
    instances = [_make_conflict("L1_vs_L3", i) for i in range(100)]
    results_a = sample_for_reference(instances, per_pair=30, seed=1)
    results_b = sample_for_reference(instances, per_pair=30, seed=2)
    assert [r["id"] for r in results_a] != [r["id"] for r in results_b]


def test_sample_for_reference_empty_input_returns_empty():
    results = sample_for_reference([], per_pair=30, seed=42)
    assert results == []
