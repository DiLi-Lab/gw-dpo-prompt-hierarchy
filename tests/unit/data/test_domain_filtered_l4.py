"""Tests for domain-filtered L4 sampling in scenario builders."""

import random

from src.data.dpo.l0_conflict_builder import _sample_domain_filtered_l4


def test_samples_from_matching_domain():
    l4_lookup = {
        ("alpaca", 0): {"l4_content": "Python tutorial", "generation": "wrapped"},
        ("alpaca", 1): {"l4_content": "Recipe for cake", "generation": "wrapped"},
        ("dolly", 0): {"l4_content": "SQL query guide", "generation": "synthesized"},
    }
    l4_domain_index = {
        "coding": [("alpaca", 0), ("dolly", 0)],
        "creative writing": [("alpaca", 1)],
    }
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4(l4_lookup, l4_domain_index, "coding", rng)
    assert content is not None
    assert content in ("Python tutorial", "SQL query guide")


def test_falls_back_to_random_on_empty_domain():
    l4_lookup = {
        ("alpaca", 0): {"l4_content": "Python tutorial", "generation": "wrapped"},
    }
    l4_domain_index = {
        "coding": [("alpaca", 0)],
    }
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4(l4_lookup, l4_domain_index, "medical", rng)
    assert content == "Python tutorial"


def test_respects_used_keys():
    l4_lookup = {
        ("alpaca", 0): {"l4_content": "Tutorial A", "generation": "wrapped"},
        ("alpaca", 1): {"l4_content": "Tutorial B", "generation": "wrapped"},
    }
    l4_domain_index = {
        "coding": [("alpaca", 0), ("alpaca", 1)],
    }
    used = {("alpaca", 0)}
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4(l4_lookup, l4_domain_index, "coding", rng, used_keys=used)
    assert content == "Tutorial B"
    assert ("alpaca", 1) in used


def test_falls_back_when_domain_exhausted():
    l4_lookup = {
        ("alpaca", 0): {"l4_content": "Code thing", "generation": "wrapped"},
        ("dolly", 0): {"l4_content": "Recipe thing", "generation": "wrapped"},
    }
    l4_domain_index = {
        "coding": [("alpaca", 0)],
        "creative writing": [("dolly", 0)],
    }
    used = {("alpaca", 0)}
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4(l4_lookup, l4_domain_index, "coding", rng, used_keys=used)
    assert content == "Recipe thing"


def test_returns_none_when_all_exhausted():
    l4_lookup = {
        ("alpaca", 0): {"l4_content": "Only entry", "generation": "wrapped"},
    }
    l4_domain_index = {"coding": [("alpaca", 0)]}
    used = {("alpaca", 0)}
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4(l4_lookup, l4_domain_index, "coding", rng, used_keys=used)
    assert content is None
    assert entry is None


def test_empty_lookup_returns_none():
    rng = random.Random(42)
    content, entry = _sample_domain_filtered_l4({}, {}, "coding", rng)
    assert content is None
    assert entry is None
