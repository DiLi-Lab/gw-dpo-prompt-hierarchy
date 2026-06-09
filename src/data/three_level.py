"""3-level collapse primitives for ablation (e).

Maps the 5-level hierarchy {L0, L1, L2, L3, L4} to a 3-level Wallace-style
hierarchy {System, User, Tool}: L0+L1+L2 -> System (3-level role 0),
L3 -> User (1), L4 -> Tool (2). System content is rendered inside the
existing <|L0_START|>...<|L0_END|> wrapper; L1/L2 wrappers are never
produced under this assembly.

These functions are pure, deterministic, and have no project-config
dependencies. They run in both data-prep and eval-time pipelines.
"""

from __future__ import annotations

import re

_LEVEL_SPAN_RE = re.compile(
    r"<\|L(?P<level>[0-4])_START\|>(?P<text>.*?)<\|L(?P=level)_END\|>",
    flags=re.DOTALL,
)

_SYSTEM_LEVELS: frozenset[int] = frozenset({0, 1, 2})


def is_intra_system(victim_level: int, attacker_level: int) -> bool:
    """True iff both levels collapse into the System block.

    Intra-System pairs (L0_vs_L1, L0_vs_L2, L1_vs_L2 and their reverses)
    cannot be expressed under the 3-level hierarchy; the build script
    drops them from the (e) train/val datasets.
    """
    return victim_level in _SYSTEM_LEVELS and attacker_level in _SYSTEM_LEVELS


def map_pair_to_3level(victim_level: int, attacker_level: int) -> tuple[str, str]:
    """Return the (victim_role, attacker_role) tuple under the 3-level mapping.

    Roles: 'system' (L0/L1/L2), 'user' (L3), 'tool' (L4).

    Raises:
        ValueError: if the pair is intra-System (cannot be represented).
    """
    if is_intra_system(victim_level, attacker_level):
        msg = (
            "intra-System pair (%d, %d) has no 3-level representation; "
            "filter via is_intra_system before calling this function"
            % (victim_level, attacker_level)
        )
        raise ValueError(msg)
    return _to_role(victim_level), _to_role(attacker_level)


def recompute_3level_gap(victim_level: int, attacker_level: int) -> int:
    """Return the new level gap under the 3-level mapping.

    Gaps: System=0, User=1, Tool=2. Calibration (victim==attacker==3) keeps
    gap=0. Intra-System pairs return 0 too — the caller is responsible for
    filtering them out before calling this function for production data.
    """
    if is_intra_system(victim_level, attacker_level):
        return 0
    if victim_level == attacker_level:
        return 0
    return abs(_to_role_index(attacker_level) - _to_role_index(victim_level))


def collapse_prompt(prompt_5level: str) -> str:
    """Rewrite a delimited 5-level prompt as a delimited 3-level prompt.

    L0/L1/L2 spans (in that order) are concatenated with '\\n\\n' inside a
    single <|L0_START|>...<|L0_END|> wrapper. L3 and L4 wrappers are
    preserved untouched. Missing components are skipped (no empty '\\n\\n'
    artefacts). If all of L0/L1/L2 are absent the System wrapper is
    omitted. Prompts with no delimiters at all (the reference split) are
    returned unchanged.

    The function is idempotent: ``collapse_prompt(collapse_prompt(p))``
    equals ``collapse_prompt(p)``.
    """
    spans: dict[int, str] = {}
    for match in _LEVEL_SPAN_RE.finditer(prompt_5level):
        level = int(match.group("level"))
        spans.setdefault(level, match.group("text"))

    if not spans:
        return prompt_5level

    out_parts: list[str] = []
    sys_parts = [spans[lvl] for lvl in (0, 1, 2) if lvl in spans]
    if sys_parts:
        sys_text = "\n\n".join(sys_parts)
        out_parts.append(f"<|L0_START|>{sys_text}<|L0_END|>")
    if 3 in spans:
        out_parts.append(f"<|L3_START|>{spans[3]}<|L3_END|>")
    if 4 in spans:
        out_parts.append(f"<|L4_START|>{spans[4]}<|L4_END|>")
    return "\n".join(out_parts)


def _to_role(level: int) -> str:
    if level in _SYSTEM_LEVELS:
        return "system"
    if level == 3:
        return "user"
    if level == 4:
        return "tool"
    msg = "unknown level: %d" % level
    raise ValueError(msg)


def _to_role_index(level: int) -> int:
    """Map a 5-level index to its 3-level role index (System=0, User=1, Tool=2)."""
    if level in _SYSTEM_LEVELS:
        return 0
    if level == 3:
        return 1
    if level == 4:
        return 2
    msg = "unknown level: %d" % level
    raise ValueError(msg)
