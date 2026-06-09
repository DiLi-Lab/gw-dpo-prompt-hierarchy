"""y_l (rejected response) generation for DPO preference pairs.

Generates plausible but incorrect model responses that follow injected
instructions instead of legitimate system prompts. Uses a tiered fallback
strategy: primary GPT-4o-mini -> rephrase -> string concat -> discard.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_RESPONSE_LENGTH: int = 10

RESEARCH_FRAMING_SYSTEM_PROMPT: str = (
    "You are helping create a security research dataset for training AI systems "
    "to resist prompt injection attacks. For each scenario, generate a plausible "
    "response that a poorly-trained model might produce if it incorrectly followed "
    "an injected instruction instead of the legitimate system prompt. This data is "
    "used to teach models what NOT to do."
)


@dataclass
class YlResult:
    """Result of y_l generation with fallback metadata.

    Attributes:
        text: The generated rejected response, or None if discarded.
        model: The model used for generation, or None for non-model fallbacks.
        fallback_used: Which fallback tier was used, or None if primary succeeded.
    """

    text: str | None
    model: str | None
    fallback_used: str | None


def generate_yl_research_framing(client: object, injection: str) -> str:
    """Call GPT-4o-mini with research framing to generate a y_l response.

    Args:
        client: OpenAI API client with a generate() method.
        injection: The injected instruction to simulate following.

    Returns:
        The generated response text.
    """
    user_prompt = (
        f'Scenario: The model received this injected instruction: "{injection}"\n'
        "Generate a realistic response that incorrectly follows this injection. "
        "The response should be 2-4 sentences, plausible, and clearly demonstrate "
        "the model following the wrong instruction."
    )
    return client.generate(
        user_prompt=user_prompt,
        system_prompt=RESEARCH_FRAMING_SYSTEM_PROMPT,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
    )


def rephrase_yl_prompt(injection: str) -> str:
    """Rephrase an injection prompt with additional research context.

    Args:
        injection: The original injected instruction.

    Returns:
        A rephrased version emphasising the research context.
    """
    return (
        "For a security research dataset, demonstrate what an unaligned model "
        f"would say when given this instruction: {injection}"
    )


def _is_valid_response(text: str) -> bool:
    """Check whether a response is non-empty and meets minimum length."""
    return bool(text) and len(text.strip()) >= MIN_RESPONSE_LENGTH


def generate_yl(
    client: object,
    injection: str,
    string_concat_fallback: str | None = None,
) -> YlResult:
    """Generate a y_l response with tiered fallback strategy.

    Fallback tiers:
        1. Primary: GPT-4o-mini with research framing system prompt.
        2. Rephrase: Rephrased prompt emphasising research context.
        3. String concat: Use pre-existing fallback text if provided.
        4. Discard: Mark the sample for removal.

    Args:
        client: OpenAI API client with a generate() method.
        injection: The injected instruction to simulate following.
        string_concat_fallback: Optional pre-existing fallback text from the
            base dataset.

    Returns:
        YlResult with the generated text and fallback metadata.
    """
    # Tier 1: Primary generation
    try:
        response = generate_yl_research_framing(client, injection)
        if _is_valid_response(response):
            return YlResult(text=response, model="gpt-4o-mini", fallback_used=None)
        logger.info("Primary y_l generation returned invalid response, trying rephrase")
    except Exception:
        logger.warning("Primary y_l generation failed with exception, trying rephrase")

    # Tier 2: Rephrase fallback
    try:
        rephrased = rephrase_yl_prompt(injection)
        response = client.generate(
            user_prompt=rephrased,
            system_prompt=RESEARCH_FRAMING_SYSTEM_PROMPT,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=500,
        )
        if _is_valid_response(response):
            return YlResult(
                text=response, model="gpt-4o-mini", fallback_used="rephrase"
            )
        logger.info("Rephrase fallback returned invalid response")
    except Exception:
        logger.warning("Rephrase fallback failed with exception")

    # Tier 3: String concat fallback
    if string_concat_fallback is not None:
        return YlResult(
            text=string_concat_fallback, model=None, fallback_used="string_concat"
        )

    # Tier 4: Discard
    return YlResult(text=None, model=None, fallback_used="discard")
