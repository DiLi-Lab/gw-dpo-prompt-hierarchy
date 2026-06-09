"""Tests for L2 per-user configuration template generator."""

import re

import pytest

from src.data.libraries.l2_templates import (
    ATTRIBUTES,
    TEMPLATES,
    L2Config,
    generate_l2,
    generate_l2_batch,
    generate_l2_config,
    generate_l2_for_conflict,
)


def test_generate_l2_returns_string():
    result = generate_l2(seed=42)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_l2_no_unfilled_placeholders():
    for seed in range(100):
        result = generate_l2(seed=seed)
        assert not re.search(r'\{[^}]+\}', result), f"seed={seed}: unfilled placeholder in '{result}'"


def test_generate_l2_deterministic():
    r1 = generate_l2(seed=42)
    r2 = generate_l2(seed=42)
    assert r1 == r2


def test_generate_l2_varies_across_seeds():
    results = {generate_l2(seed=i) for i in range(50)}
    assert len(results) > 10, "L2 generation should produce diverse outputs"


def test_attributes_coverage():
    assert len(ATTRIBUTES) == 11
    assert "tone" in ATTRIBUTES, "Expected 'tone' key (renamed from 'formality')"
    assert "formality" not in ATTRIBUTES, "'formality' should have been renamed to 'tone'"
    assert ATTRIBUTES["language"] == ["English"], "Only English should be supported"
    for key, values in ATTRIBUTES.items():
        if key == "language":
            continue
        assert len(values) >= 3, f"Attribute '{key}' has too few values"


def test_templates_exist():
    assert len(TEMPLATES) >= 4
    for t in TEMPLATES:
        assert "{" in t, f"Template has no placeholders: {t}"


def test_l2_config_has_text_and_attributes():
    config = generate_l2_config(seed=42)
    assert isinstance(config, L2Config)
    assert isinstance(config.text, str)
    assert len(config.text) > 0
    assert isinstance(config.attributes, dict)
    assert len(config.attributes) >= 2


def test_l2_config_attributes_are_valid():
    config = generate_l2_config(seed=42)
    for key, value in config.attributes.items():
        assert key in ATTRIBUTES, f"Unknown attribute key: {key}"
        assert value in ATTRIBUTES[key], f"Invalid value '{value}' for attribute '{key}'"


def test_l2_config_text_matches_generate_l2():
    """generate_l2_config and generate_l2 produce the same text for the same seed."""
    config = generate_l2_config(seed=42)
    text = generate_l2(seed=42)
    assert config.text == text


def test_l2_config_deterministic():
    c1 = generate_l2_config(seed=99)
    c2 = generate_l2_config(seed=99)
    assert c1.text == c2.text
    assert c1.attributes == c2.attributes


# --- Task 2: generate_l2_for_conflict tests ---


def test_generate_l2_for_conflict_pins_language():
    config = generate_l2_for_conflict(attribute="language", value="English", seed=42)
    assert isinstance(config, L2Config)
    assert config.attributes["language"] == "English"
    assert "English" in config.text


def test_generate_l2_for_conflict_rejects_non_english():
    with pytest.raises(ValueError, match="Invalid value"):
        generate_l2_for_conflict(attribute="language", value="Spanish")


def test_generate_l2_for_conflict_pins_format():
    config = generate_l2_for_conflict(attribute="format", value="JSON", seed=10)
    assert config.attributes["format"] == "JSON"
    assert "JSON" in config.text


def test_generate_l2_for_conflict_pins_content_filter():
    config = generate_l2_for_conflict(attribute="content_filter", value="allow-mature", seed=5)
    assert config.attributes["content_filter"] == "allow-mature"


def test_generate_l2_for_conflict_pins_access_tier():
    config = generate_l2_for_conflict(attribute="access_tier", value="admin", seed=7)
    assert config.attributes["access_tier"] == "admin"


def test_generate_l2_for_conflict_invalid_attribute():
    with pytest.raises(ValueError, match="Unknown attribute"):
        generate_l2_for_conflict(attribute="nonexistent", value="foo")


def test_generate_l2_for_conflict_invalid_value():
    with pytest.raises(ValueError, match="Invalid value"):
        generate_l2_for_conflict(attribute="language", value="Klingon")


def test_generate_l2_for_conflict_deterministic():
    c1 = generate_l2_for_conflict(attribute="language", value="English", seed=42)
    c2 = generate_l2_for_conflict(attribute="language", value="English", seed=42)
    assert c1.text == c2.text
    assert c1.attributes == c2.attributes


# --- Task 3: generate_l2_batch tests ---


def test_generate_l2_batch_returns_correct_count():
    configs, stats = generate_l2_batch(count=100, seed=42)
    assert len(configs) == 100
    assert all(isinstance(c, L2Config) for c in configs)


def test_generate_l2_batch_stats_has_all_attributes():
    _, stats = generate_l2_batch(count=200, seed=42)
    assert "attribute_counts" in stats
    assert "template_counts" in stats
    assert "total" in stats
    assert stats["total"] == 200
    for attr in ATTRIBUTES:
        assert attr in stats["attribute_counts"]


def test_generate_l2_batch_deterministic():
    configs1, _ = generate_l2_batch(count=50, seed=42)
    configs2, _ = generate_l2_batch(count=50, seed=42)
    assert [c.text for c in configs1] == [c.text for c in configs2]


def test_generate_l2_batch_diverse():
    configs, stats = generate_l2_batch(count=500, seed=42)
    unique_texts = {c.text for c in configs}
    assert len(unique_texts) > 100, "Batch should produce diverse outputs"
    for attr, count in stats["attribute_counts"].items():
        assert count > 0, f"Attribute '{attr}' never appeared in 500 samples"
