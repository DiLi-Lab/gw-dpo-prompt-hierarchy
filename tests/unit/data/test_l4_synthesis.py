"""Tests for L4 synthesis prompt construction."""

from src.data.libraries.l4_tool_outputs import (
    L4_SYNTHESIS_PROMPT_SYSTEM,
    build_l4_synthesis_prompt,
)


def test_build_l4_synthesis_prompt():
    prompt = build_l4_synthesis_prompt("What is photosynthesis?")
    assert "photosynthesis" in prompt


def test_build_l4_synthesis_prompt_wraps_in_quotes():
    prompt = build_l4_synthesis_prompt("test question")
    assert '"test question"' in prompt


def test_synthesis_system_prompt_exists():
    assert len(L4_SYNTHESIS_PROMPT_SYSTEM) > 0
    assert "tool output" in L4_SYNTHESIS_PROMPT_SYSTEM.lower()


def test_synthesis_prompt_targets_100_200_words():
    """Synthesis length should match wrapped Source A distribution (median ~117 words)."""
    assert "100" in L4_SYNTHESIS_PROMPT_SYSTEM
    assert "200" in L4_SYNTHESIS_PROMPT_SYSTEM
