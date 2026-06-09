"""Unit tests for the 3-level collapse primitives.

Mapping under (e):
    L0 + L1 + L2 -> System (3-level role 0)
    L3 -> User (3-level role 1)
    L4 -> Tool (3-level role 2)

System content is rendered inside the existing <|L0_START|>...<|L0_END|>
wrapper; L1/L2 wrappers are never produced.
"""

import pytest

from src.data.three_level import (
    collapse_prompt,
    is_intra_system,
    map_pair_to_3level,
    recompute_3level_gap,
)


def _make_5level_prompt(
    *,
    l0: str | None = "rule",
    l1: str | None = "persona",
    l2: str | None = "config",
    l3: str | None = "user-msg",
    l4: str | None = "tool-out",
) -> str:
    parts: list[str] = []
    if l0 is not None:
        parts.append(f"<|L0_START|>{l0}<|L0_END|>")
    if l1 is not None:
        parts.append(f"<|L1_START|>{l1}<|L1_END|>")
    if l2 is not None:
        parts.append(f"<|L2_START|>{l2}<|L2_END|>")
    if l3 is not None:
        parts.append(f"<|L3_START|>{l3}<|L3_END|>")
    if l4 is not None:
        parts.append(f"<|L4_START|>{l4}<|L4_END|>")
    return "\n".join(parts)


class TestIsIntraSystem:
    def test_intra_system_pairs_return_true(self):
        for victim, attacker in [(0, 1), (0, 2), (1, 2)]:
            assert is_intra_system(victim, attacker) is True
            assert is_intra_system(attacker, victim) is True  # order-agnostic

    def test_cross_block_pairs_return_false(self):
        for victim, attacker in [
            (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        ]:
            assert is_intra_system(victim, attacker) is False

    def test_calibration_returns_false(self):
        # Calibration pairs encode the L3 axis only (victim=attacker=3 in the
        # synthetic data, or both None). Either way, not intra-system.
        assert is_intra_system(3, 3) is False


class TestMapPairTo3Level:
    @pytest.mark.parametrize(
        ("victim", "attacker", "expected"),
        [
            (0, 3, ("system", "user")),
            (1, 3, ("system", "user")),
            (2, 3, ("system", "user")),
            (3, 4, ("user", "tool")),
            (0, 4, ("system", "tool")),
            (1, 4, ("system", "tool")),
            (2, 4, ("system", "tool")),
        ],
    )
    def test_cross_block_pairs(self, victim, attacker, expected):
        assert map_pair_to_3level(victim, attacker) == expected

    def test_intra_system_raises(self):
        for victim, attacker in [(0, 1), (0, 2), (1, 2)]:
            with pytest.raises(ValueError, match="intra-System"):
                map_pair_to_3level(victim, attacker)


class TestRecompute3LevelGap:
    @pytest.mark.parametrize(
        ("victim", "attacker", "expected"),
        [
            (0, 3, 1),  # System vs User
            (1, 3, 1),
            (2, 3, 1),
            (3, 4, 1),  # User vs Tool
            (0, 4, 2),  # System vs Tool
            (1, 4, 2),
            (2, 4, 2),
        ],
    )
    def test_cross_block_gaps(self, victim, attacker, expected):
        assert recompute_3level_gap(victim, attacker) == expected

    def test_calibration_gap_zero(self):
        # By convention: calibration pairs keep gap=0 regardless of victim/attacker.
        assert recompute_3level_gap(3, 3) == 0


class TestCollapsePrompt:
    def test_full_5level_prompt(self):
        out = collapse_prompt(_make_5level_prompt())
        assert "<|L0_START|>rule\n\npersona\n\nconfig<|L0_END|>" in out
        assert "<|L1_START|>" not in out
        assert "<|L2_START|>" not in out
        assert "<|L3_START|>user-msg<|L3_END|>" in out
        assert "<|L4_START|>tool-out<|L4_END|>" in out

    def test_no_l4(self):
        prompt = _make_5level_prompt(l4=None)
        out = collapse_prompt(prompt)
        assert "<|L4_START|>" not in out
        assert "<|L0_START|>rule\n\npersona\n\nconfig<|L0_END|>" in out

    def test_only_l3_and_l4(self):
        prompt = _make_5level_prompt(l0=None, l1=None, l2=None)
        out = collapse_prompt(prompt)
        assert "<|L0_START|>" not in out
        assert "<|L3_START|>user-msg<|L3_END|>" in out
        assert "<|L4_START|>tool-out<|L4_END|>" in out

    def test_partial_system_block(self):
        # Missing L1 — System block should contain only L0 and L2 joined by \n\n.
        prompt = _make_5level_prompt(l1=None)
        out = collapse_prompt(prompt)
        assert "<|L0_START|>rule\n\nconfig<|L0_END|>" in out

    def test_idempotent(self):
        prompt = _make_5level_prompt()
        once = collapse_prompt(prompt)
        twice = collapse_prompt(once)
        assert once == twice

    def test_no_delimiters_passthrough(self):
        # Reference split has flat-text prompts with no delimiters.
        flat = "Just a plain instruction."
        assert collapse_prompt(flat) == flat

    def test_preserves_content_with_special_chars(self):
        weird = (
            "<|L0_START|>rule with **markdown** and \"quotes\"<|L0_END|>\n"
            "<|L3_START|>user msg<|L3_END|>"
        )
        out = collapse_prompt(weird)
        assert 'rule with **markdown** and "quotes"' in out
