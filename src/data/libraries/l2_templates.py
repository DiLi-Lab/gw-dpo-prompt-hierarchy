"""L2 per-user configuration template generator.

Generates on-the-fly L2 content from a set of attribute categories
and template strings. No LLM calls or persistent files needed.
"""

import random
import re
from dataclasses import dataclass

ATTRIBUTES: dict[str, list[str]] = {
    "language": [
        "English",
    ],
    "tone": ["casual", "professional", "academic", "friendly", "concise"],
    "format": ["markdown", "plain text", "JSON", "bullet points", "numbered lists"],
    "length": ["brief (1-2 sentences)", "moderate (1-2 paragraphs)", "detailed"],
    "content_filter": [
        "family-friendly", "allow-mature", "no-politics", "educational-only",
    ],
    "access_tier": ["basic", "premium", "admin"],
    "expertise": ["beginner", "intermediate", "expert", "technical"],
    "data_access": ["restricted", "standard", "unrestricted"],
    "system_access": ["sandboxed", "standard", "elevated"],
    "identity_mode": ["transparent", "standard", "flexible"],
    "compliance_mode": ["strict", "standard", "relaxed"],
}

# Mapping from L0 category to the L2 attribute and value that conflicts with it.
L0_CATEGORY_TO_L2_CONFLICT: dict[str, tuple[str, str]] = {
    "content_prohibitions": ("content_filter", "allow-mature"),
    "privacy": ("data_access", "unrestricted"),
    "system_integrity": ("system_access", "elevated"),
    "identity_honesty": ("identity_mode", "flexible"),
    "behavioural_guardrails": ("compliance_mode", "relaxed"),
}

TEMPLATES: list[str] = [
    "User preferences — Language: {language}. Tone: {tone}. Format: {format}.",
    "Session config: Respond in {language}. Tone: {tone}. Length: {length}. Access: {access_tier}.",
    "User profile: expertise level is {expertise}. Output: {format}. Filter: {content_filter}.",
    "Configuration: {tone} tone, {language} language, {length} responses.",
    "Settings: {format} output, {expertise} level, {content_filter} content policy.",
    "Preferences: Language={language}, Format={format}, Expertise={expertise}.",
    "Security config: Data access={data_access}. System access={system_access}. Compliance={compliance_mode}.",
    "Identity settings: Mode={identity_mode}. Tone: {tone}. Filter: {content_filter}.",
]


@dataclass(frozen=True)
class L2Config:
    """A generated L2 per-user configuration with structured metadata.

    Attributes:
        text: The rendered configuration string (for insertion into prompts).
        attributes: The attribute key-value pairs used to fill the template.
    """

    text: str
    attributes: dict[str, str]


def _generate_l2_parts(
    seed: int | None = None,
    required_attributes: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Generate L2 text and the chosen attributes dict.

    Args:
        seed: Random seed for reproducibility.
        required_attributes: Attributes that must appear with specific values.
            Remaining slots are filled randomly.

    Returns:
        Tuple of (filled template string, chosen attributes dict).
    """
    rng = random.Random(seed)

    required = dict(required_attributes) if required_attributes else {}
    remaining_keys = [k for k in ATTRIBUTES if k not in required]

    min_extra = max(0, 2 - len(required))
    max_extra = max(min_extra, 5 - len(required))
    num_extra = rng.randint(min_extra, min(max_extra, len(remaining_keys)))
    extra_keys = rng.sample(remaining_keys, num_extra)

    values = {k: rng.choice(ATTRIBUTES[k]) for k in extra_keys}
    values.update(required)

    # Prefer templates containing placeholders for required attributes
    if required:
        matching = [
            t for t in TEMPLATES
            if all("{" + k + "}" in t for k in required)
        ]
        template = rng.choice(matching if matching else TEMPLATES)
    else:
        template = rng.choice(TEMPLATES)
    filled = template
    for k, v in values.items():
        filled = filled.replace("{" + k + "}", v)

    filled = re.sub(r'\{[^}]+\}', 'default', filled)
    return filled, values


def generate_l2_config(
    seed: int | None = None,
    required_attributes: dict[str, str] | None = None,
) -> L2Config:
    """Generate a structured L2 per-user configuration.

    Args:
        seed: Random seed for reproducibility.
        required_attributes: Attributes that must appear with specific values.

    Returns:
        L2Config with the rendered text and chosen attributes.
    """
    text, attributes = _generate_l2_parts(seed, required_attributes)
    return L2Config(text=text, attributes=attributes)


def generate_l2_for_conflict(
    attribute: str,
    value: str,
    seed: int | None = None,
) -> L2Config:
    """Generate an L2 config with a specific attribute pinned to a given value.

    Used by DPO pair construction to create L2 content that conflicts
    with a specific higher or lower level.

    Args:
        attribute: The attribute key to pin (e.g., "language", "format").
        value: The value to set for that attribute.
        seed: Random seed for reproducibility.

    Returns:
        L2Config with the pinned attribute guaranteed present.

    Raises:
        ValueError: If attribute or value is not in the attribute space.
    """
    if attribute not in ATTRIBUTES:
        raise ValueError(
            "Unknown attribute %r. Valid: %s" % (attribute, list(ATTRIBUTES.keys()))
        )
    if value not in ATTRIBUTES[attribute]:
        raise ValueError(
            "Invalid value %r for attribute %r. Valid: %s"
            % (value, attribute, ATTRIBUTES[attribute])
        )
    return generate_l2_config(seed=seed, required_attributes={attribute: value})


def generate_l2_batch(
    count: int,
    seed: int = 0,
) -> tuple[list[L2Config], dict[str, object]]:
    """Generate a batch of L2 configs and compute diversity statistics.

    Args:
        count: Number of configs to generate.
        seed: Base seed; each config uses seed + i.

    Returns:
        Tuple of (list of L2Config, stats dict with attribute_counts,
        template_counts, total, and unique_texts).
    """
    configs: list[L2Config] = []
    attribute_counts: dict[str, int] = {k: 0 for k in ATTRIBUTES}
    template_counts: dict[str, int] = {t: 0 for t in TEMPLATES}

    for i in range(count):
        config = generate_l2_config(seed=seed + i)
        configs.append(config)
        for attr in config.attributes:
            attribute_counts[attr] += 1
        for t in TEMPLATES:
            static_parts = re.split(r'\{[^}]+\}', t)
            if all(part in config.text for part in static_parts if part.strip()):
                template_counts[t] += 1
                break

    stats: dict[str, object] = {
        "total": count,
        "unique_texts": len({c.text for c in configs}),
        "attribute_counts": attribute_counts,
        "template_counts": template_counts,
    }
    return configs, stats


def generate_l2(seed: int | None = None) -> str:
    """Generate a random L2 per-user configuration string.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A filled L2 configuration string with no unfilled placeholders.
    """
    text, _ = _generate_l2_parts(seed)
    return text
