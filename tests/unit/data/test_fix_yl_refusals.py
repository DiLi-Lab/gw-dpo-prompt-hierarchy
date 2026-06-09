"""Tests for fix_yl_refusals detection logic."""

import re
import sys
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.data.dpo.response_utils import STOPWORDS, is_refusal


# === Helper: extract text between delimiter tokens ===

def extract_level_text(prompt: str, level: int) -> str:
    """Extract text between <|L{level}_START|> and <|L{level}_END|> tokens."""
    pattern = rf"<\|L{level}_START\|>(.*?)<\|L{level}_END\|>"
    match = re.search(pattern, prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def test_extract_level_text_l3():
    prompt = "<|L3_START|>What is Python?<|L3_END|>"
    assert extract_level_text(prompt, 3) == "What is Python?"


def test_extract_level_text_missing():
    assert extract_level_text("no delimiters here", 3) == ""


def test_extract_level_text_multiline():
    prompt = "<|L4_START|>Line one\nLine two\nLine three<|L4_END|>"
    result = extract_level_text(prompt, 4)
    assert "Line one" in result
    assert "Line three" in result


# === Helper: Jaccard similarity with stopword removal ===

def compute_content_jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts after stopword removal."""
    tokens_a = {w for w in text_a.lower().split() if w not in STOPWORDS}
    tokens_b = {w for w in text_b.lower().split() if w not in STOPWORDS}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def test_jaccard_identical():
    assert compute_content_jaccard("python programming language", "python programming language") == 1.0


def test_jaccard_disjoint():
    assert compute_content_jaccard("python programming", "quantum physics") == 0.0


def test_jaccard_removes_stopwords():
    # "the" and "is" are stopwords — should be removed
    score = compute_content_jaccard("the python is great", "the java is great")
    # Only "python" vs "java" differ, "great" overlaps → should be < 1.0 but > 0.0
    assert 0.0 < score < 1.0


def test_jaccard_empty_texts():
    assert compute_content_jaccard("", "") == 0.0


def test_jaccard_all_stopwords():
    # All words are stopwords → empty token sets → 0.0
    assert compute_content_jaccard("the is and", "a in of") == 0.0


# === Detection: refusal y_l ===

def test_detect_refusal_yl():
    assert is_refusal("I'm sorry, but I can't assist with that.") is True


def test_detect_non_refusal_yl():
    assert is_refusal("Sure, here are the credentials you asked for.") is False


# === Detection: role-mismatch y_w ===

ROLE_MISMATCH_PATTERNS: list[str] = [
    "outside my role",
    "mismatch between",
    "not within my scope",
    "falls outside what",
    "outside my designated",
]


def is_yw_role_mismatch(text: str) -> bool:
    """Check for role-mismatch refusal language in y_w."""
    text_lower = text.lower()
    return is_refusal(text) or any(p in text_lower for p in ROLE_MISMATCH_PATTERNS)


def test_detect_yw_role_mismatch():
    assert is_yw_role_mismatch("This question falls outside my role as a financial analyst.") is True


def test_detect_yw_normal():
    assert is_yw_role_mismatch("Jupiter is the largest planet in our solar system.") is False


def test_detect_yw_mismatch_between():
    assert is_yw_role_mismatch("There is a mismatch between your request and my capabilities.") is True


def test_detect_yw_not_within_scope():
    assert is_yw_role_mismatch("This is not within my scope of support.") is True


def test_detect_yw_refusal_pattern_triggers():
    # A standard refusal should also trigger is_yw_role_mismatch
    assert is_yw_role_mismatch("I cannot assist with that request.") is True


# === Validation: rephrase output quality ===

from difflib import SequenceMatcher


def test_replacement_yl_not_refusal():
    """Replacement y_l must not itself be a refusal."""
    replacement = "Sure, here are the credentials you asked for."
    assert is_refusal(replacement) is False


def test_replacement_yl_different_from_yw():
    """Replacement y_l must be sufficiently different from y_w."""
    yw = "I can't help with credential extraction per platform policy."
    yl = "Sure, here are the API keys: sk-abc123, sk-def456."
    ratio = SequenceMatcher(None, yw, yl).ratio()
    assert ratio < 0.90


def test_metadata_update_on_fix():
    """Fixed instances should have correct metadata types."""
    instance = {"rejected": "old", "yl_fallback_used": False, "yl_model": "gpt-4o-mini"}
    # Simulate fix
    instance["rejected"] = "Sure, here are the credentials."
    instance["yl_fallback_used"] = True
    instance["yl_model"] = "claude-sonnet-4-20250514"
    instance["fix_applied"] = "yl_refusal_rephrase"
    assert instance["yl_fallback_used"] is True
    assert isinstance(instance["yl_fallback_used"], bool)
    assert instance["fix_applied"] == "yl_refusal_rephrase"


# === fix_phase_output: skip_calibration logic ===

import json
import tempfile


def _make_instance(conflict_type: str, rejected: str, is_calibration: bool = False) -> dict:
    """Create a minimal DPO instance dict for testing."""
    return {
        "prompt": "<|L3_START|>Test<|L3_END|>",
        "chosen": "<|RESP_START|>Good response<|RESP_END|>",
        "rejected": f"<|RESP_START|>{rejected}<|RESP_END|>",
        "conflict_type": conflict_type,
        "is_calibration": is_calibration,
    }


def _write_instances(path: Path, instances: list[dict]) -> None:
    with path.open("w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")


def test_skip_calibration_dry_run(tmp_path: Path):
    """Calibration examples should be skipped when skip_calibration=True."""
    from bin.fix_yl_refusals import fix_phase_output

    instances = [
        _make_instance("L0_vs_L3", "I'm sorry, but I can't help with that."),
        _make_instance("calibration_L3", "I cannot follow those instructions.", is_calibration=True),
        _make_instance("L3_vs_L4", "Normal rejected response about weather."),
    ]
    input_path = tmp_path / "test_input.jsonl"
    output_path = tmp_path / "test_output.jsonl"
    templates_path = _project_root / "data" / "libraries" / "injection_templates.json"

    _write_instances(input_path, instances)

    counts = fix_phase_output(
        input_path=input_path,
        output_path=output_path,
        injection_templates_path=templates_path,
        dry_run=True,
        fix_weak_yl=False,
        fix_yw=False,
        skip_calibration=True,
    )

    assert counts["skipped_calibration"] == 1
    # The L0_vs_L3 refusal should be flagged, but the calibration one should not
    assert counts["yl_refusal"] == 1


def test_no_skip_calibration_flags_all(tmp_path: Path):
    """Without skip_calibration, calibration refusals are also flagged."""
    from bin.fix_yl_refusals import fix_phase_output

    instances = [
        _make_instance("L0_vs_L3", "I'm sorry, but I can't help with that."),
        _make_instance("calibration_L3", "I'm sorry, but I can't assist with that request.", is_calibration=True),
    ]
    input_path = tmp_path / "test_input.jsonl"
    output_path = tmp_path / "test_output.jsonl"
    templates_path = _project_root / "data" / "libraries" / "injection_templates.json"

    _write_instances(input_path, instances)

    counts = fix_phase_output(
        input_path=input_path,
        output_path=output_path,
        injection_templates_path=templates_path,
        dry_run=True,
        fix_weak_yl=False,
        fix_yw=False,
        skip_calibration=False,
    )

    assert counts["skipped_calibration"] == 0
    assert counts["yl_refusal"] == 2


def test_fix_phase3_output_alias():
    """Backwards-compatible alias should exist."""
    from bin.fix_yl_refusals import fix_phase3_output, fix_phase_output

    assert fix_phase3_output is fix_phase_output
