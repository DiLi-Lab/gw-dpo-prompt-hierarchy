"""Keyword-based domain classifier for matching L3 user messages to L1 system prompts.

Classifies user instructions into one of 15 task domains using keyword
frequency scoring, then selects a domain-matched L1 developer system prompt
from the library.
"""

import logging
import random

from src.data.libraries.l1_prompts import TASK_DOMAINS

logger = logging.getLogger(__name__)

GENERIC_L1_PROMPT: dict = {
    "domain": "general knowledge",
    "persona": "knowledgeable AI assistant",
    "constraints": ["Provide accurate and clear responses"],
    "full_prompt": (
        "You are a knowledgeable AI assistant. "
        "Provide accurate, clear, and helpful responses."
    ),
}

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "program", "function", "class", "variable", "debug",
        "compile", "script", "algorithm", "python", "javascript", "java",
        "api", "bug", "syntax", "refactor", "implement", "library",
        "framework", "database", "sql", "html", "css", "git",
    ],
    "creative writing": [
        "story", "poem", "novel", "character", "narrative", "fiction",
        "plot", "dialogue", "creative", "write a story", "chapter",
        "protagonist", "setting", "imagery", "metaphor", "prose",
    ],
    "summarisation": [
        "summarize", "summarise", "summary", "condense", "brief",
        "overview", "key points", "tldr", "recap", "shorten",
        "main ideas", "abstract", "gist", "outline", "digest",
    ],
    "factual QA": [
        "who is", "what is", "when did", "where is", "how many",
        "fact", "history", "define", "explain what", "tell me about",
        "information about", "describe", "who was", "what was",
    ],
    "classification": [
        "classify", "categorize", "categorise", "label", "sort into",
        "group", "identify the type", "sentiment", "detect", "tag",
        "positive or negative", "spam", "class", "bucket",
    ],
    "translation": [
        "translate", "translation", "convert to", "in french",
        "in spanish", "in german", "in japanese", "in chinese",
        "language", "multilingual", "localize", "interpret",
        "from english", "to english",
    ],
    "math/reasoning": [
        "calculate", "solve", "equation", "math", "proof", "theorem",
        "algebra", "geometry", "calculus", "derivative", "integral",
        "probability", "statistics", "formula", "compute", "arithmetic",
        "logic", "reasoning", "infer",
    ],
    "data analysis": [
        "data", "dataset", "analyze", "analyse", "chart", "graph",
        "trend", "correlation", "visualization", "csv", "spreadsheet",
        "metrics", "insight", "dashboard", "report", "pivot",
    ],
    "email/letter writing": [
        "email", "letter", "compose", "dear", "regards", "subject line",
        "formal letter", "cover letter", "memo", "correspondence",
        "reply to", "draft an email", "professional tone",
    ],
    "education/explanation": [
        "explain", "teach", "lesson", "tutorial", "concept", "understand",
        "learn", "education", "student", "course", "curriculum",
        "example", "step by step", "beginner", "introduction",
    ],
    "brainstorming": [
        "brainstorm", "ideas", "suggest", "creative ideas", "generate ideas",
        "list of", "come up with", "alternatives", "options",
        "possibilities", "inspiration", "innovate", "think of",
    ],
    "conversation/roleplay": [
        "roleplay", "role play", "pretend", "act as", "character",
        "conversation", "chat", "dialogue", "scenario", "simulate",
        "persona", "impersonate", "interactive", "play",
    ],
    "legal": [
        "legal", "law", "contract", "clause", "regulation", "statute",
        "court", "attorney", "lawyer", "liability", "compliance",
        "terms of service", "intellectual property", "litigation",
    ],
    "medical": [
        "medical", "health", "symptom", "diagnosis", "treatment",
        "disease", "patient", "clinical", "medicine", "doctor",
        "prescription", "therapy", "surgery", "condition", "hospital",
        "diabetes", "blood pressure",
    ],
    "general knowledge": [
        "general", "knowledge", "trivia", "miscellaneous", "random",
        "interesting", "curious", "wonder", "question",
    ],
}


def classify_domain(instruction: str) -> str:
    """Classify a user instruction into one of the 15 task domains.

    Uses case-insensitive keyword matching. Each domain's score is the
    count of its keywords found in the instruction text. Returns the
    domain with the highest score, or "general knowledge" as fallback.

    Args:
        instruction: The user instruction text to classify.

    Returns:
        One of the 15 task domain strings from TASK_DOMAINS.
    """
    text_lower = instruction.lower()
    best_domain = "general knowledge"
    best_score = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_domain = domain

    logger.debug(
        "Classified instruction as '%s' (score=%d): %.60s...",
        best_domain, best_score, instruction,
    )
    return best_domain


def select_matched_l1(
    l1_library: list[dict],
    domain: str,
    seed: int | None = None,
    prefer_broad: bool = False,
) -> dict:
    """Select a domain-matched L1 system prompt from the library.

    Filters the library for entries matching the given domain, then
    randomly selects one. Falls back to GENERIC_L1_PROMPT if no
    matching entries exist.

    Args:
        l1_library: List of L1 prompt dicts with "domain" key.
        domain: Target domain to match against.
        seed: Optional random seed for reproducibility.
        prefer_broad: If True, prefer entries with scope="broad" when available.

    Returns:
        A single L1 prompt dict matching the domain, or GENERIC_L1_PROMPT.
    """
    matches = [entry for entry in l1_library if entry.get("domain") == domain]

    if prefer_broad:
        broad = [e for e in matches if e.get("scope") == "broad"]
        if broad:
            matches = broad

    if not matches:
        logger.debug("No L1 match for domain '%s', using generic fallback", domain)
        return GENERIC_L1_PROMPT

    rng = random.Random(seed)
    return rng.choice(matches)
