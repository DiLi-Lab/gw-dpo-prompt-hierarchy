"""Anthropic API client wrapper for Claude model calls.

Provides a thin wrapper around the Anthropic SDK with retry logic
and rate limiting for batch operations.
"""

import logging
import os
import time

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS: int = 4000
DEFAULT_TEMPERATURE: float = 0.9
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BASE_DELAY: float = 2.0
BATCH_DELAY: float = 0.5


class AnthropicClient:
    """Wrapper around the Anthropic SDK for Claude API calls.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Raises:
        ValueError: If no API key is provided or found in environment.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            msg = "ANTHROPIC_API_KEY not set. Provide api_key or set the environment variable."
            raise ValueError(msg)
        self._client = anthropic.Anthropic(api_key=resolved_key)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate a single response from Claude.

        Args:
            user_prompt: The user message content.
            system_prompt: Optional system message.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, RETRY_MAX_ATTEMPTS,
                )
                time.sleep(delay)
            except anthropic.APIError as e:
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("API error: %s. Retrying in %.1fs", e, delay)
                time.sleep(delay)

        msg = "Max retries exceeded"
        raise RuntimeError(msg)

    def generate_batch(
        self,
        user_prompts: list[str],
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[str]:
        """Generate responses for multiple prompts sequentially.

        Args:
            user_prompts: List of user messages.
            system_prompt: Shared system message for all calls.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per response.

        Returns:
            List of generated text responses, one per prompt.
        """
        results: list[str] = []
        for i, prompt in enumerate(user_prompts):
            logger.info("Generating %d/%d", i + 1, len(user_prompts))
            result = self.generate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
            if i < len(user_prompts) - 1:
                time.sleep(BATCH_DELAY)
        return results
