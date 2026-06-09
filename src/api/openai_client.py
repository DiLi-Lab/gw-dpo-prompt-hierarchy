"""OpenAI API client wrapper for GPT model calls.

Provides a thin wrapper around the OpenAI SDK with retry logic
and rate limiting for batch operations.
"""

import logging
import os
import time

import openai

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_MAX_TOKENS: int = 200
DEFAULT_TEMPERATURE: float = 0.7
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BASE_DELAY: float = 2.0
BATCH_DELAY: float = 0.2


class OpenAIClient:
    """Wrapper around the OpenAI SDK for GPT API calls.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

    Raises:
        ValueError: If no API key is provided or found in environment.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            msg = "OPENAI_API_KEY not set. Provide api_key or set the environment variable."
            raise ValueError(msg)
        self._client = openai.OpenAI(api_key=resolved_key)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        json_mode: bool = False,
    ) -> str:
        """Generate a single response from a GPT model.

        Args:
            user_prompt: The user message content.
            system_prompt: Optional system message.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            json_mode: If True, forces JSON output format.

        Returns:
            The generated text response.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except openai.RateLimitError:
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, RETRY_MAX_ATTEMPTS,
                )
                time.sleep(delay)
            except openai.APIError as e:
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
        json_mode: bool = False,
    ) -> list[str]:
        """Generate responses for multiple prompts sequentially.

        Args:
            user_prompts: List of user messages.
            system_prompt: Shared system message for all calls.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per response.
            json_mode: If True, forces JSON output format.

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
                json_mode=json_mode,
            )
            results.append(result)
            if i < len(user_prompts) - 1:
                time.sleep(BATCH_DELAY)
        return results
