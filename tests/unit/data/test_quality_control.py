"""Tests for DPO quality control pipeline."""

import json
from unittest.mock import MagicMock, patch

from src.data.dpo.quality_control import (
    filter_dpo_example,
    deduplicate_by_hash,
    deduplicate_by_embedding,
    stratified_sample,
    build_judge_prompt,
    parse_judge_response,
    apply_judge_decisions,
    save_flagged_examples,
)


def _make_example(prompt: str, chosen: str, rejected: str,
                  conflict_type: str = "L1_vs_L3") -> dict:
    return {
        "prompt": prompt,
        "chosen": f"<|RESP_START|>{chosen}<|RESP_END|>",
        "rejected": f"<|RESP_START|>{rejected}<|RESP_END|>",
        "conflict_type": conflict_type,
        "level_gap": 2,
        "margin": 2.0,
        "category": "pairwise",
        "is_calibration": False,
    }


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # word-level tokenization


def test_filter_accepts_valid_example():
    tok = FakeTokenizer()
    ex = _make_example(
        "<|L0_START|>rule<|L0_END|>\n<|L1_START|>sys<|L1_END|>\n<|L3_START|>hello<|L3_END|>",
        "A " * 50,
        "B " * 50,
    )
    assert filter_dpo_example(ex, tok) is True


def test_filter_rejects_short_chosen():
    tok = FakeTokenizer()
    ex = _make_example("prompt", "short", "A " * 20)
    assert filter_dpo_example(ex, tok) is False


def test_filter_rejects_short_rejected():
    tok = FakeTokenizer()
    ex = _make_example("prompt", "A " * 20, "tiny")
    assert filter_dpo_example(ex, tok) is False


def test_filter_rejects_too_similar():
    tok = FakeTokenizer()
    text = "This is a response about Python programming language features."
    ex = _make_example("prompt", text, text)
    assert filter_dpo_example(ex, tok) is False


def test_filter_rejects_broken_delimiters():
    tok = FakeTokenizer()
    ex = _make_example(
        "<|L0_START|>rule",  # Missing L0_END
        "A " * 20,
        "B " * 20,
    )
    assert filter_dpo_example(ex, tok) is False


def test_deduplicate_by_hash():
    examples = [
        _make_example("prompt A", "chosen A", "rejected A"),
        _make_example("prompt A", "chosen B", "rejected B"),  # duplicate prompt
        _make_example("prompt C", "chosen C", "rejected C"),
    ]
    deduped = deduplicate_by_hash(examples)
    assert len(deduped) == 2
    prompts = {ex["prompt"] for ex in deduped}
    assert len(prompts) == 2


def test_deduplicate_by_embedding_without_sentence_transformers():
    """When sentence-transformers is not installed, returns examples unchanged."""
    examples = [_make_example(f"prompt {i}", f"c{i}", f"r{i}") for i in range(5)]
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        result = deduplicate_by_embedding(examples)
    assert len(result) == 5


def test_deduplicate_by_embedding_empty():
    assert deduplicate_by_embedding([]) == []


def test_stratified_sample_proportional():
    examples = (
        [_make_example(f"p{i}", f"c{i}", f"r{i}", "L1_vs_L3") for i in range(100)]
        + [_make_example(f"q{i}", f"c{i}", f"r{i}", "L0_vs_L4") for i in range(50)]
    )
    sample = stratified_sample(examples, fraction=0.2)
    # ~20% of 150 = 30
    assert 25 <= len(sample) <= 35
    # Both types should be represented
    types = {ex["conflict_type"] for ex in sample}
    assert "L1_vs_L3" in types
    assert "L0_vs_L4" in types


def test_stratified_sample_seed_reproducible():
    examples = [_make_example(f"p{i}", f"c{i}", f"r{i}") for i in range(100)]
    s1 = stratified_sample(examples, fraction=0.1, seed=42)
    s2 = stratified_sample(examples, fraction=0.1, seed=42)
    assert [e["prompt"] for e in s1] == [e["prompt"] for e in s2]


def test_build_judge_prompt_returns_system_and_user():
    ex = _make_example("prompt", "good response", "bad response", "L1_vs_L3")
    system, user = build_judge_prompt(ex)
    assert "hierarchy" in system.lower()
    assert "L0" in system
    assert "L1_vs_L3" in user
    assert "good response" in user
    assert "bad response" in user
    assert "q1" in user.lower() or "1." in user


def test_build_judge_prompt_matches_doc_format():
    """The judge prompt should match the format documented in doc 16 Section 8.2."""
    ex = _make_example("prompt text", "chosen text", "rejected text", "L0_vs_L1")
    system, user = build_judge_prompt(ex)
    # System prompt should explain the hierarchy
    assert "L0 (platform rules)" in system
    assert "higher levels always override" in system.lower()
    # User prompt should have the 4 evaluation criteria
    assert "correctly follow the higher-priority level" in user
    assert "violate the hierarchy" in user
    assert "sufficiently different" in user
    assert "realistic" in user
    # Should request JSON output
    assert '"keep"' in user


def test_parse_judge_response_valid():
    response = json.dumps({"q1": 5, "q2": 4, "q3": 5, "q4": 4, "keep": True, "reason": "Good pair"})
    result = parse_judge_response(response)
    assert result["keep"] is True
    assert result["q1"] == 5


def test_parse_judge_response_invalid_json():
    result = parse_judge_response("not json")
    assert result is None


def test_apply_judge_decisions_both_keep():
    gpt_scores = {"q1": 5, "q2": 4, "q3": 5, "q4": 4, "keep": True}
    gemini_scores = {"q1": 4, "q2": 5, "q3": 4, "q4": 3, "keep": True}
    decision = apply_judge_decisions(gpt_scores, gemini_scores)
    assert decision == "keep"


def test_apply_judge_decisions_both_reject():
    gpt_scores = {"q1": 1, "q2": 2, "q3": 1, "q4": 1, "keep": False}
    gemini_scores = {"q1": 2, "q2": 1, "q3": 2, "q4": 1, "keep": False}
    decision = apply_judge_decisions(gpt_scores, gemini_scores)
    assert decision == "discard"


def test_apply_judge_decisions_disagree():
    gpt_scores = {"q1": 5, "q2": 4, "q3": 5, "q4": 4, "keep": True}
    gemini_scores = {"q1": 2, "q2": 1, "q3": 2, "q4": 2, "keep": False}
    decision = apply_judge_decisions(gpt_scores, gemini_scores)
    assert decision == "flag"


def test_apply_judge_decisions_low_score_overrides_keep():
    gpt_scores = {"q1": 1, "q2": 4, "q3": 5, "q4": 4, "keep": True}  # q1=1
    gemini_scores = {"q1": 4, "q2": 4, "q3": 4, "q4": 4, "keep": True}
    decision = apply_judge_decisions(gpt_scores, gemini_scores)
    assert decision == "discard"  # Any score=1 -> discard


def test_save_flagged_examples(tmp_path):
    flagged = [
        {
            "example": _make_example("p1", "c1", "r1"),
            "gpt_scores": {"q1": 4, "q2": 3, "q3": 4, "q4": 3, "keep": True},
            "gemini_scores": {"q1": 2, "q2": 2, "q3": 3, "q4": 2, "keep": False},
        },
    ]
    output = tmp_path / "flagged.jsonl"
    save_flagged_examples(flagged, output)
    assert output.exists()
    with open(output) as f:
        lines = f.readlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert "example" in parsed
    assert "gpt_scores" in parsed
    assert "gemini_scores" in parsed
