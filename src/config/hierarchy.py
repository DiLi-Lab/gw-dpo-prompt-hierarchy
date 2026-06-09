"""Hierarchy relationships and conflict pair definitions.

Defines all 10 pairwise conflict relationships for the 5-level hierarchy
and provides utilities for computing level gaps and validating conflicts.
"""

from src.config.constants import NUM_LEVELS

CONFLICT_PAIRS: list[tuple[int, int]] = [
    (i, j)
    for i in range(NUM_LEVELS)
    for j in range(i + 1, NUM_LEVELS)
]

CONFLICT_LABELS: dict[tuple[int, int], str] = {
    (i, j): f"L{i}_vs_L{j}"
    for i, j in CONFLICT_PAIRS
}


def get_level_gap(victim_level: int, attacker_level: int) -> int:
    """Compute hierarchy distance between two conflicting levels.

    Args:
        victim_level: Higher-privilege level index (0-4).
        attacker_level: Lower-privilege level index (0-4).

    Returns:
        The positive integer gap (attacker - victim).

    Raises:
        ValueError: If victim_level >= attacker_level.
    """
    if victim_level >= attacker_level:
        msg = (
            f"victim level ({victim_level}) must be less than "
            f"attacker level ({attacker_level})"
        )
        raise ValueError(msg)
    return attacker_level - victim_level


def is_valid_conflict(victim_level: int, attacker_level: int) -> bool:
    """Check whether a conflict pair is valid."""
    return (
        0 <= victim_level < NUM_LEVELS
        and 0 <= attacker_level < NUM_LEVELS
        and victim_level < attacker_level
    )
