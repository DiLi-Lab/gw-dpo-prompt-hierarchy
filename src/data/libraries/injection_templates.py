"""Injection templates loader and utilities.

Loads handcrafted injection patterns from injection_templates.json
and provides functions for sampling prefixes and injecting content
at different positions within text.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InjectionEntry:
    """A single injection template entry.

    Attributes:
        text: The injection text.
        id: Unique identifier.
        targets_safety: Whether this injection involves L0-level safety
            concerns (data exfiltration, credential theft, jailbreaking).
            Used to filter position_injections for L3-vs-L4 pairs where
            only benign task redirections should be used.
        l0_category: The L0 category targeted by this injection
            (system_integrity, privacy, content_prohibitions,
            identity_honesty, behavioural_guardrails). Only set when
            targets_safety is True.
    """

    text: str
    id: str
    targets_safety: bool = False
    l0_category: str | None = None
    causes_refusal: bool = False


@dataclass
class InjectionTemplate:
    """Collection of injection templates by category."""

    prefixes: list[InjectionEntry] = field(default_factory=list)
    system_overrides: list[InjectionEntry] = field(default_factory=list)
    position_injections: list[InjectionEntry] = field(default_factory=list)


def load_injection_templates(path: Path) -> InjectionTemplate:
    """Load injection templates from a JSON file.

    Args:
        path: Path to injection_templates.json.

    Returns:
        InjectionTemplate with all categories populated.

    Raises:
        FileNotFoundError: If the templates file does not exist.
    """
    if not path.exists():
        msg = "Injection templates file not found: %s" % path
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw = json.load(f)

    templates = InjectionTemplate(
        prefixes=[InjectionEntry(text=e["text"], id=e["id"]) for e in raw.get("prefixes", [])],
        system_overrides=[InjectionEntry(text=e["text"], id=e["id"]) for e in raw.get("system_overrides", [])],
        position_injections=[
            InjectionEntry(
                text=e["text"],
                id=e["id"],
                targets_safety=e.get("targets_safety", False),
                l0_category=e.get("l0_category"),
                causes_refusal=e.get("causes_refusal", False),
            )
            for e in raw.get("position_injections", [])
        ],
    )
    total = len(templates.prefixes) + len(templates.system_overrides) + len(templates.position_injections)
    logger.info("Loaded %d injection templates from %s", total, path)
    return templates


def sample_injection_prefix(
    templates: InjectionTemplate,
    seed: int | None = None,
) -> str:
    """Sample a random injection prefix string.

    Args:
        templates: Loaded injection templates.
        seed: Random seed for reproducibility.

    Returns:
        A prefix string for prepending to injected instructions.

    Raises:
        ValueError: If no prefixes are available.
    """
    if not templates.prefixes:
        msg = "No injection prefixes available"
        raise ValueError(msg)

    rng = random.Random(seed)
    return rng.choice(templates.prefixes).text


def inject_into_content(
    content: str,
    injection: str,
    position: str = "middle",
) -> str:
    """Insert an injection string into content at the specified position.

    Args:
        content: The legitimate content to inject into.
        injection: The injected instruction text.
        position: Where to place the injection: "start", "middle", or "end".

    Returns:
        Content with the injection inserted.

    Raises:
        ValueError: If position is not one of start/middle/end.
    """
    if position == "start":
        return injection + "\n\n" + content
    if position == "end":
        return content + "\n\n" + injection
    if position == "middle":
        mid = len(content) // 2
        return content[:mid] + "\n" + injection + "\n" + content[mid:]

    msg = "Invalid position '%s': must be 'start', 'middle', or 'end'" % position
    raise ValueError(msg)
