"""Tests for shared response utilities."""

from src.data.dpo.response_utils import REFUSAL_PATTERNS, STOPWORDS, is_refusal


class TestIsRefusal:
    def test_matches_sorry_ai(self):
        assert is_refusal("I'm sorry, but as an AI model, I can't do that.") is True

    def test_matches_cannot_help(self):
        assert is_refusal("I cannot help with that request.") is True

    def test_matches_unable_to(self):
        assert is_refusal("I'm unable to assist with generating harmful content.") is True

    def test_no_match_normal_response(self):
        assert is_refusal("Python is a high-level programming language.") is False

    def test_case_insensitive(self):
        assert is_refusal("I'M SORRY, BUT AS AN AI model...") is True

    def test_empty_string(self):
        assert is_refusal("") is False


class TestConstants:
    def test_refusal_patterns_nonempty(self):
        assert len(REFUSAL_PATTERNS) >= 10

    def test_stopwords_nonempty(self):
        assert len(STOPWORDS) >= 100

    def test_stopwords_contains_common_words(self):
        for word in ("the", "is", "and", "of", "to", "a", "in"):
            assert word in STOPWORDS
