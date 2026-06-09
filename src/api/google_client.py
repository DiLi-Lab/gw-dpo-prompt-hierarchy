"""Google/Gemini API client wrapper for Gemini model calls via Vertex AI.

Provides a thin wrapper around the Vertex AI SDK with retry logic
and rate limiting for batch operations. Uses Application Default
Credentials (ADC) for authentication.
"""

import logging
import os
import time

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gemini-2.5-pro"
DEFAULT_MAX_TOKENS: int = 4000
DEFAULT_TEMPERATURE: float = 0.7
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BASE_DELAY: float = 2.0
BATCH_DELAY: float = 0.5
VERTEX_LOCATION: str = "europe-west1"


class GoogleClient:
    """Wrapper around the Vertex AI SDK for Gemini API calls.

    Uses Application Default Credentials (ADC) and reads the GCP project
    from the GOOGLE_CLOUD_PROJECT environment variable.

    Raises:
        ValueError: If GOOGLE_CLOUD_PROJECT is not set.
    """

    def __init__(self) -> None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            msg = "GOOGLE_CLOUD_PROJECT environment variable is not set."
            raise ValueError(msg)
        vertexai.init(project=project, location=VERTEX_LOCATION)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate a single response from a Gemini model.

        Args:
            user_prompt: The user message content.
            system_prompt: Optional system instruction.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """
        if system_prompt:
            gen_model = GenerativeModel(model, system_instruction=system_prompt)
        else:
            gen_model = GenerativeModel(model)

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                response = gen_model.generate_content(
                    user_prompt,
                    generation_config=generation_config,
                )
                return response.text
            except Exception as e:
                if attempt == RETRY_MAX_ATTEMPTS - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API error: %s. Retrying in %.1fs (attempt %d/%d)",
                    e, delay, attempt + 1, RETRY_MAX_ATTEMPTS,
                )
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
            system_prompt: Shared system instruction for all calls.
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
