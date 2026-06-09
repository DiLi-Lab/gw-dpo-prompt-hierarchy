"""Response-aware L2 per-user configuration generator.

Generates L2 strings via GPT-4o-mini conditioned on the actual response,
L1 system prompt, and L3 user message. Used by SFT builders to produce
L2 content that accurately describes the response characteristics.
"""

import logging

logger = logging.getLogger(__name__)

NON_ENGLISH_LANGUAGES: list[str] = [
    "French", "Spanish", "Chinese", "Japanese", "Korean",
    "Portuguese", "Arabic", "Hindi", "Polish", "Italian",
    "German", "Russian", "Dutch", "Swedish", "Turkish",
]

MISALIGNED_L2_REFUSAL: str = (
    "Session config: Respond in English. Tone: professional. Length: brief. Access: basic."
)

MISALIGNED_L2_REDIRECT: str = (
    "Session config: Respond in English. Tone: friendly. Length: brief. Access: basic."
)

FALLBACK_L2: str = (
    "Session config: Respond in English. Tone: professional. Length: moderate. Access: basic."
)

_MAX_L2_LENGTH: int = 120
_RESPONSE_TRUNCATION: int = 500
_MODEL: str = "gpt-4o-mini"
_TEMPERATURE: float = 0.7
_MAX_TOKENS: int = 60

L2_SYSTEM_PROMPT: str = (
    "You generate user-session configuration strings for an AI assistant.\n"
    "The configuration describes the user's preferences and must be consistent\n"
    "with how the assistant actually responds.\n"
    "\n"
    "Use this style — short, structured, key-value-like:\n"
    '- "Session config: Respond in English. Tone: professional. Length: detailed. Access: premium."\n'
    '- "User preferences — Language: English. Tone: friendly. Format: bullet points."\n'
    '- "Configuration: casual tone, English language, brief responses."\n'
    '- "Settings: plain text output, intermediate level, family-friendly content policy."\n'
    '- "Preferences: Language=English, Format=markdown, Expertise=expert."\n'
    '- "User profile: expertise level is beginner. Output: numbered lists. Filter: educational-only."\n'
    "\n"
    "Rules:\n"
    "- Language is always English.\n"
    "- Keep it to ONE line, under 120 characters.\n"
    "- Use 2-5 attributes from: tone, format, length, expertise, access_tier, content_filter.\n"
    '- The attributes MUST match the response. If the response is casual, don\'t say "professional".\n'
    "- Vary the template style across calls. Don't always use the same format.\n"
    "- Return ONLY the configuration string, nothing else."
)


def validate_l2(text: str) -> bool:
    """Check whether an L2 string meets format requirements.

    Args:
        text: The L2 string to validate.

    Returns:
        True if valid, False otherwise.
    """
    if "\n" in text:
        return False
    if len(text) > _MAX_L2_LENGTH:
        return False
    text_lower = text.lower()
    for lang in NON_ENGLISH_LANGUAGES:
        if lang.lower() in text_lower:
            return False
    return True


def generate_l2_from_response(
    client: object,
    l1_prompt: str,
    l3_message: str,
    response: str,
) -> str:
    """Generate a response-grounded L2 string via GPT-4o-mini.

    Calls client.generate() with the L2 system prompt and a user prompt
    containing l1_prompt, l3_message, and response (truncated to 500 chars).
    Validates the output and retries once on failure.

    Args:
        client: OpenAIClient instance with .generate() method.
        l1_prompt: The developer system prompt for this instance.
        l3_message: The user message for this instance.
        response: The target response text.

    Returns:
        L2 configuration string, or FALLBACK_L2 on persistent failure.
    """
    truncated = response[:_RESPONSE_TRUNCATION]
    user_prompt = (
        "System prompt: %s\n"
        "User message: %s\n"
        "Response: %s"
    ) % (l1_prompt, l3_message, truncated)

    for attempt in range(2):
        try:
            result = client.generate(
                user_prompt=user_prompt,
                system_prompt=L2_SYSTEM_PROMPT,
                model=_MODEL,
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
            )
            result = result.strip()
            if validate_l2(result):
                return result
            logger.warning(
                "L2 validation failed (attempt %d): %.80s...", attempt + 1, result,
            )
        except Exception:
            logger.warning(
                "L2 generation API error (attempt %d)", attempt + 1, exc_info=True,
            )

    return FALLBACK_L2
