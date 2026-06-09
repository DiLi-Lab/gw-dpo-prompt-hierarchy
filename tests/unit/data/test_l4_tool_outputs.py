"""Tests for L4 tool output wrapping and synthesis."""

from src.data.libraries.l4_tool_outputs import TOOL_TEMPLATES, wrap_as_l4


def test_wrap_as_l4_returns_string():
    result = wrap_as_l4("some content", seed=42)
    assert isinstance(result, str)
    assert "some content" in result


def test_wrap_as_l4_deterministic():
    r1 = wrap_as_l4("content", seed=42)
    r2 = wrap_as_l4("content", seed=42)
    assert r1 == r2


def test_wrap_as_l4_varies():
    results = {wrap_as_l4("content", seed=i) for i in range(50)}
    assert len(results) > 1, "Should use different templates"


def test_wrap_as_l4_with_query():
    result = wrap_as_l4("search result", query="python tutorial", seed=0)
    assert "search result" in result


def test_tool_templates_count():
    assert len(TOOL_TEMPLATES) == 6


def test_wrap_as_l4_plain_template():
    found_plain = False
    for seed in range(100):
        result = wrap_as_l4("exact content", seed=seed)
        if result == "exact content":
            found_plain = True
            break
    assert found_plain, "Plain template should be reachable"
