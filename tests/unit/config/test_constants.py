"""Tests for hierarchy constants."""

from src.config.constants import (
    HIERARCHY_LEVELS,
    NUM_LEVELS,
    NUM_SEGMENTS,
    RESPONSE_SEGMENT_ID,
    SPECIAL_TOKENS,
)


def test_hierarchy_levels():
    assert HIERARCHY_LEVELS == ("L0", "L1", "L2", "L3", "L4")
    assert NUM_LEVELS == 5


def test_special_tokens():
    assert len(SPECIAL_TOKENS) == 12
    for i in range(5):
        assert f"<|L{i}_START|>" in SPECIAL_TOKENS
        assert f"<|L{i}_END|>" in SPECIAL_TOKENS
    assert "<|RESP_START|>" in SPECIAL_TOKENS
    assert "<|RESP_END|>" in SPECIAL_TOKENS


def test_segments():
    assert NUM_SEGMENTS == 6
    assert RESPONSE_SEGMENT_ID == 5
