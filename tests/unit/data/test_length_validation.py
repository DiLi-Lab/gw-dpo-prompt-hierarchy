"""Tests for token length validation utility."""

from transformers import AutoTokenizer

from src.config.constants import SPECIAL_TOKENS
from src.data.length_validation import (
    LengthReport,
    check_delimiter_integrity,
    compute_length_stats,
    validate_example_lengths,
)
from src.model.special_tokens import add_hierarchy_tokens


def _make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    add_hierarchy_tokens(tokenizer)
    return tokenizer


def _make_example(l0="rule", l1="system", l2="config", l3="query", l4="data", resp="answer"):
    """Build a canonical 5-level prompt string."""
    return (
        f"<|L0_START|>{l0}<|L0_END|>"
        f"<|L1_START|>{l1}<|L1_END|>"
        f"<|L2_START|>{l2}<|L2_END|>"
        f"<|L3_START|>{l3}<|L3_END|>"
        f"<|L4_START|>{l4}<|L4_END|>"
        f"<|RESP_START|>{resp}<|RESP_END|>"
    )


def test_compute_length_stats_basic():
    tokenizer = _make_tokenizer()
    examples = [_make_example(), _make_example(l3="a longer query with more words")]
    stats = compute_length_stats(examples, tokenizer)
    assert stats.count == 2
    assert stats.min_length > 0
    assert stats.max_length >= stats.min_length
    assert stats.mean_length > 0
    assert stats.p50 > 0
    assert stats.p95 >= stats.p50
    assert stats.p99 >= stats.p95


def test_compute_length_stats_over_limit():
    tokenizer = _make_tokenizer()
    short = _make_example()
    long_content = "word " * 5000  # Very long content
    long_example = _make_example(l3=long_content)
    examples = [short, long_example]

    stats = compute_length_stats(examples, tokenizer, max_seq_length=50)
    assert stats.num_over_limit > 0
    assert stats.over_limit_fraction > 0
    assert len(stats.over_limit_indices) > 0


def test_validate_example_lengths_all_ok():
    tokenizer = _make_tokenizer()
    examples = [_make_example() for _ in range(5)]
    report = validate_example_lengths(examples, tokenizer, max_seq_length=4096)
    assert report.all_valid is True
    assert len(report.issues) == 0


def test_validate_detects_truncated_response():
    tokenizer = _make_tokenizer()
    # Create example where response would be cut off at a very low limit
    examples = [_make_example(resp="a response that should not be cut")]
    report = validate_example_lengths(examples, tokenizer, max_seq_length=20)
    assert report.all_valid is False
    assert any("response" in issue.lower() or "truncat" in issue.lower() for issue in report.issues)


def test_check_delimiter_integrity_valid():
    tokenizer = _make_tokenizer()
    text = _make_example()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    issues = check_delimiter_integrity(token_ids, tokenizer)
    assert len(issues) == 0


def test_check_delimiter_integrity_missing_end():
    tokenizer = _make_tokenizer()
    # Simulate truncation: L0 start but no L0 end
    text = "<|L0_START|>some rule content"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    issues = check_delimiter_integrity(token_ids, tokenizer)
    assert len(issues) > 0
    assert any("L0" in issue for issue in issues)


def test_check_delimiter_integrity_missing_resp_end():
    tokenizer = _make_tokenizer()
    text = (
        "<|L0_START|>rule<|L0_END|>"
        "<|RESP_START|>answer without closing"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    issues = check_delimiter_integrity(token_ids, tokenizer)
    assert len(issues) > 0
    assert any("RESP" in issue for issue in issues)


def test_length_report_summary():
    tokenizer = _make_tokenizer()
    examples = [_make_example() for _ in range(3)]
    report = validate_example_lengths(examples, tokenizer, max_seq_length=4096)
    summary = report.summary()
    assert isinstance(summary, str)
    assert "examples" in summary.lower()
