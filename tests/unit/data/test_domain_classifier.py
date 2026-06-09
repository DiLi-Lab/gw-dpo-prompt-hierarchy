"""Tests for keyword-based domain classifier for L1-L3 matching."""

import pytest

from src.data.sft.domain_classifier import (
    DOMAIN_KEYWORDS,
    GENERIC_L1_PROMPT,
    classify_domain,
    select_matched_l1,
)


class TestClassifyDomain:
    """Tests for classify_domain function."""

    def test_returns_coding_for_python_text(self):
        instruction = "Write a Python function that sorts a list of integers"
        assert classify_domain(instruction) == "coding"

    def test_returns_coding_for_debug_text(self):
        instruction = "Debug this JavaScript code that throws an error"
        assert classify_domain(instruction) == "coding"

    def test_returns_translation_for_translate_text(self):
        instruction = "Translate the following paragraph from English to French"
        assert classify_domain(instruction) == "translation"

    def test_returns_math_for_equation_text(self):
        instruction = "Solve this equation: 2x + 5 = 15"
        assert classify_domain(instruction) == "math/reasoning"

    def test_returns_math_for_calculate_text(self):
        instruction = "Calculate the derivative of f(x) = 3x^2 + 2x"
        assert classify_domain(instruction) == "math/reasoning"

    def test_returns_creative_writing_for_story(self):
        instruction = "Write a short story about a dragon who learns to fly"
        assert classify_domain(instruction) == "creative writing"

    def test_returns_summarisation_for_summarize(self):
        instruction = "Summarize the following article in three sentences"
        assert classify_domain(instruction) == "summarisation"

    def test_returns_medical_for_health_text(self):
        instruction = "What are the symptoms and treatment options for diabetes?"
        assert classify_domain(instruction) == "medical"

    def test_returns_legal_for_contract_text(self):
        instruction = "Review this contract clause for potential legal issues"
        assert classify_domain(instruction) == "legal"

    def test_fallback_to_general_knowledge(self):
        instruction = "Tell me something interesting"
        assert classify_domain(instruction) == "general knowledge"

    def test_case_insensitive(self):
        instruction = "WRITE A PYTHON FUNCTION"
        assert classify_domain(instruction) == "coding"

    def test_highest_score_wins(self):
        # Multiple coding keywords should beat a single match elsewhere
        instruction = "Write Python code to implement an algorithm using functions"
        assert classify_domain(instruction) == "coding"


class TestDomainKeywords:
    """Tests for DOMAIN_KEYWORDS coverage."""

    def test_all_task_domains_have_keywords(self):
        from src.data.libraries.l1_prompts import TASK_DOMAINS

        for domain in TASK_DOMAINS:
            assert domain in DOMAIN_KEYWORDS, f"Missing keywords for domain: {domain}"

    def test_each_domain_has_sufficient_keywords(self):
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert len(keywords) >= 5, (
                f"Domain '{domain}' has only {len(keywords)} keywords, expected >= 5"
            )


class TestSelectMatchedL1:
    """Tests for select_matched_l1 function."""

    def test_returns_matching_domain(self):
        library = [
            {"domain": "coding", "full_prompt": "You are a coder.", "persona": "coder", "constraints": ["code only"]},
            {"domain": "medical", "full_prompt": "You are a doctor.", "persona": "doctor", "constraints": ["medical only"]},
        ]
        result = select_matched_l1(library, "coding", seed=42)
        assert result["domain"] == "coding"

    def test_returns_one_of_matching_entries(self):
        library = [
            {"domain": "coding", "full_prompt": "Prompt A", "persona": "a", "constraints": []},
            {"domain": "coding", "full_prompt": "Prompt B", "persona": "b", "constraints": []},
            {"domain": "medical", "full_prompt": "Prompt C", "persona": "c", "constraints": []},
        ]
        result = select_matched_l1(library, "coding", seed=42)
        assert result["domain"] == "coding"
        assert result["full_prompt"] in ("Prompt A", "Prompt B")

    def test_fallback_to_generic_when_domain_not_found(self):
        library = [
            {"domain": "coding", "full_prompt": "You are a coder.", "persona": "coder", "constraints": ["code only"]},
        ]
        result = select_matched_l1(library, "medical")
        assert result == GENERIC_L1_PROMPT

    def test_fallback_to_generic_for_empty_library(self):
        result = select_matched_l1([], "coding")
        assert result == GENERIC_L1_PROMPT

    def test_deterministic_with_seed(self):
        library = [
            {"domain": "coding", "full_prompt": f"Prompt {i}", "persona": f"p{i}", "constraints": []}
            for i in range(10)
        ]
        result1 = select_matched_l1(library, "coding", seed=123)
        result2 = select_matched_l1(library, "coding", seed=123)
        assert result1 == result2

    def test_generic_l1_prompt_structure(self):
        assert "domain" in GENERIC_L1_PROMPT
        assert "persona" in GENERIC_L1_PROMPT
        assert "constraints" in GENERIC_L1_PROMPT
        assert "full_prompt" in GENERIC_L1_PROMPT
        assert GENERIC_L1_PROMPT["domain"] == "general knowledge"


class TestSelectMatchedL1PreferBroad:
    """Tests for prefer_broad parameter in select_matched_l1."""

    def test_prefer_broad_selects_broad_entries(self):
        library = [
            {"domain": "factual QA", "full_prompt": "Narrow specialist", "scope": "narrow"},
            {"domain": "factual QA", "full_prompt": "Broad generalist", "scope": "broad"},
        ]
        result = select_matched_l1(library, "factual QA", seed=42, prefer_broad=True)
        assert result["scope"] == "broad"

    def test_prefer_broad_false_allows_any(self):
        library = [
            {"domain": "factual QA", "full_prompt": "Narrow specialist", "scope": "narrow"},
        ]
        result = select_matched_l1(library, "factual QA", seed=42, prefer_broad=False)
        assert result["full_prompt"] == "Narrow specialist"

    def test_prefer_broad_falls_back_to_narrow_when_no_broad(self):
        library = [
            {"domain": "factual QA", "full_prompt": "Only narrow", "scope": "narrow"},
        ]
        result = select_matched_l1(library, "factual QA", seed=42, prefer_broad=True)
        assert result["full_prompt"] == "Only narrow"

    def test_prefer_broad_ignores_missing_scope(self):
        """Entries without scope field should be treated as non-broad."""
        library = [
            {"domain": "coding", "full_prompt": "No scope field"},
            {"domain": "coding", "full_prompt": "Broad entry", "scope": "broad"},
        ]
        result = select_matched_l1(library, "coding", seed=42, prefer_broad=True)
        assert result["scope"] == "broad"

    def test_default_prefer_broad_is_false(self):
        """Default behaviour unchanged — no filtering by scope."""
        library = [
            {"domain": "coding", "full_prompt": "Narrow", "scope": "narrow"},
        ]
        result = select_matched_l1(library, "coding", seed=42)
        assert result["full_prompt"] == "Narrow"
