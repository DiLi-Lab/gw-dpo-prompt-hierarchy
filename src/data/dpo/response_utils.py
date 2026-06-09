"""Shared response analysis utilities for DPO y_w and y_l generation.

Provides refusal pattern detection and stopwords for content overlap analysis.
Extracted from yw_generation.py to avoid circular imports between yw and yl modules.
"""

REFUSAL_PATTERNS: list[str] = [
    "i'm sorry, but as an ai",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "as an ai language model",
    "i cannot help",
    "i cannot assist",
    "i cannot provide",
    "i cannot disclose",
    "i cannot and will not",
    "i can't assist",
    "i can't help",
    "i can't provide",
    "i can't disclose",
    "i can't process",
    "i'm not able to",
    "i don't have the ability",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i will not provide",
    "i will not assist",
]


def is_refusal(text: str) -> bool:
    """Check whether a response matches known AI refusal patterns.

    Args:
        text: Response text to check.

    Returns:
        True if any refusal pattern is found.
    """
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in REFUSAL_PATTERNS)


STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "like", "ll", "me", "might",
    "more", "most", "must", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "she", "should",
    "shouldn't", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve",
    "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with",
    "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
    "yourselves",
})
