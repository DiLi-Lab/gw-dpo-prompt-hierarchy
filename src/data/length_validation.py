"""Token length validation for constructed training examples.

Validates that training examples fit within max_seq_length without
truncation, checks delimiter integrity, and reports length statistics.
Used during dataset construction (Steps 2-3) to catch over-length
examples before they enter the training pipeline.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from transformers import PreTrainedTokenizerBase

from src.config.constants import NUM_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class LengthStats:
    """Token length statistics for a set of examples."""

    count: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    p50: int = 0
    p95: int = 0
    p99: int = 0
    num_over_limit: int = 0
    over_limit_fraction: float = 0.0
    over_limit_indices: list[int] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """A single validation issue for an example."""

    index: int
    token_length: int
    issue: str


@dataclass
class LengthReport:
    """Full validation report for a set of examples."""

    stats: LengthStats
    issues: list[str] = field(default_factory=list)
    issue_details: list[ValidationIssue] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        lines = [
            f"Length validation: {self.stats.count} examples",
            f"  Token lengths: min={self.stats.min_length}, max={self.stats.max_length}, "
            f"mean={self.stats.mean_length:.0f}, p50={self.stats.p50}, "
            f"p95={self.stats.p95}, p99={self.stats.p99}",
            f"  Over limit: {self.stats.num_over_limit}/{self.stats.count} "
            f"({self.stats.over_limit_fraction:.1%})",
        ]
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues[:10]:
                lines.append(f"    - {issue}")
            if len(self.issues) > 10:
                lines.append(f"    ... and {len(self.issues) - 10} more")
        else:
            lines.append("  No issues found.")
        return "\n".join(lines)


def compute_length_stats(
    examples: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 4096,
) -> LengthStats:
    """Compute token length statistics for a list of text examples.

    Args:
        examples: List of full prompt strings (with delimiters).
        tokenizer: Tokenizer with hierarchy special tokens added.
        max_seq_length: Maximum sequence length threshold.

    Returns:
        LengthStats with distribution info and over-limit counts.
    """
    lengths = []
    over_limit_indices = []

    for i, text in enumerate(examples):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        length = len(token_ids)
        lengths.append(length)
        if length > max_seq_length:
            over_limit_indices.append(i)

    if not lengths:
        return LengthStats()

    arr = np.array(lengths)
    return LengthStats(
        count=len(lengths),
        min_length=int(arr.min()),
        max_length=int(arr.max()),
        mean_length=float(arr.mean()),
        p50=int(np.percentile(arr, 50)),
        p95=int(np.percentile(arr, 95)),
        p99=int(np.percentile(arr, 99)),
        num_over_limit=len(over_limit_indices),
        over_limit_fraction=len(over_limit_indices) / len(lengths),
        over_limit_indices=over_limit_indices,
    )


def check_delimiter_integrity(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Check that all opened delimiters have matching closers.

    Detects truncation damage: if a START token appears without its
    corresponding END token, the segment ID computation will be corrupted.

    Args:
        token_ids: Token IDs of a single example.
        tokenizer: Tokenizer with hierarchy special tokens added.

    Returns:
        List of issue descriptions. Empty list means no problems.
    """
    issues = []

    for i in range(NUM_LEVELS):
        start_id = tokenizer.convert_tokens_to_ids(f"<|L{i}_START|>")
        end_id = tokenizer.convert_tokens_to_ids(f"<|L{i}_END|>")

        has_start = start_id in token_ids
        has_end = end_id in token_ids

        if has_start and not has_end:
            issues.append(
                f"L{i}: START token found but END token missing "
                f"(likely truncated)"
            )
        elif has_end and not has_start:
            issues.append(
                f"L{i}: END token found but START token missing"
            )

    resp_start_id = tokenizer.convert_tokens_to_ids("<|RESP_START|>")
    resp_end_id = tokenizer.convert_tokens_to_ids("<|RESP_END|>")

    has_resp_start = resp_start_id in token_ids
    has_resp_end = resp_end_id in token_ids

    if has_resp_start and not has_resp_end:
        issues.append(
            "RESP: START token found but END token missing "
            "(response truncated)"
        )
    elif has_resp_end and not has_resp_start:
        issues.append("RESP: END token found but START token missing")

    return issues


def validate_example_lengths(
    examples: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 4096,
) -> LengthReport:
    """Validate token lengths and delimiter integrity for all examples.

    Args:
        examples: List of full prompt strings (with delimiters).
        tokenizer: Tokenizer with hierarchy special tokens added.
        max_seq_length: Maximum allowed sequence length.

    Returns:
        LengthReport with statistics, issues, and a human-readable summary.
    """
    stats = compute_length_stats(examples, tokenizer, max_seq_length)
    issues: list[str] = []
    issue_details: list[ValidationIssue] = []

    for i, text in enumerate(examples):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        length = len(token_ids)

        if length > max_seq_length:
            msg = (
                f"Example {i}: {length} tokens exceeds limit of "
                f"{max_seq_length} (will be truncated)"
            )
            issues.append(msg)
            issue_details.append(ValidationIssue(i, length, msg))

            # Check delimiter integrity at the truncation point
            truncated_ids = token_ids[:max_seq_length]
            delim_issues = check_delimiter_integrity(truncated_ids, tokenizer)
        else:
            delim_issues = check_delimiter_integrity(token_ids, tokenizer)

        for delim_issue in delim_issues:
            msg = f"Example {i}: {delim_issue}"
            issues.append(msg)
            issue_details.append(ValidationIssue(i, length, msg))

    return LengthReport(stats=stats, issues=issues, issue_details=issue_details)
