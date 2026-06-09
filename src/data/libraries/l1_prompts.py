"""L1 developer system prompt generation and deduplication.

Generates diverse developer system prompts across 15 task domains
using Claude Sonnet 4, then deduplicates via sentence-transformer
embeddings with cosine similarity thresholding.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.api.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

TASK_DOMAINS: list[str] = [
    "coding",
    "creative writing",
    "summarisation",
    "factual QA",
    "classification",
    "translation",
    "math/reasoning",
    "data analysis",
    "email/letter writing",
    "education/explanation",
    "brainstorming",
    "conversation/roleplay",
    "legal",
    "medical",
    "general knowledge",
]

L1_SYSTEM_PROMPT: str = (
    "You are a dataset engineer creating realistic developer system prompts "
    "for training a language model. Generate diverse, specific system prompts that "
    "a real application developer would write."
)

DEDUP_MODEL_NAME: str = "all-MiniLM-L6-v2"

L1_TEMPERATURE: float = 0.9
L1_MAX_TOKENS: int = 4000


def build_l1_generation_prompt(domain: str) -> str:
    """Build the user prompt for L1 generation.

    Args:
        domain: The task domain to generate prompts for.

    Returns:
        Formatted user prompt string for Claude.
    """
    return (
        f'Generate 10 developer system prompts for the "{domain}" domain. Each must:\n'
        '- Define a specific persona with a role (not just "helpful assistant")\n'
        "- Include 2-4 behavioural constraints (things the assistant must or must not do)\n"
        "- Optionally specify an output format preference\n"
        "- Be 50-150 words long\n"
        "- Be meaningfully distinct from each other\n"
        "\n"
        "Output as a JSON array where each element has:\n"
        '- "persona": brief role description\n'
        '- "constraints": list of 2-4 constraint strings\n'
        '- "full_prompt": the complete system prompt text as it would appear in production\n'
        "\n"
        'Do NOT include generic prompts like "You are a helpful assistant." Every prompt '
        "should be specific enough that you can imagine a real product using it.\n"
        "Avoid repetitive roles. Constraints should be specific and testable."
    )


def parse_l1_response(response: str) -> list[dict]:
    """Parse Claude's JSON response into a list of prompt dicts.

    Args:
        response: Raw response text from Claude.

    Returns:
        List of dicts with persona/constraints/full_prompt keys.
        Returns empty list if parsing fails.
    """
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]

        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning("Response is not a JSON array")
            return []

        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "full_prompt" not in item or "persona" not in item:
                logger.warning("Skipping item missing required fields: %s", list(item.keys()))
                continue
            constraints = item.get("constraints")
            if not isinstance(constraints, list) or not all(
                isinstance(c, str) for c in constraints
            ):
                logger.warning(
                    "Skipping item with invalid constraints for persona: %s",
                    item["persona"],
                )
                continue
            valid.append(item)

        return valid

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse L1 response as JSON: %s", e)
        return []


def filter_by_length(
    prompts: list[dict],
    min_words: int = 30,
    max_words: int = 300,
) -> list[dict]:
    """Filter prompts by word count of their full_prompt field.

    Args:
        prompts: List of prompt dicts with full_prompt key.
        min_words: Minimum word count (inclusive).
        max_words: Maximum word count (inclusive).

    Returns:
        Filtered list of prompt dicts.
    """
    filtered = []
    for p in prompts:
        word_count = len(p["full_prompt"].split())
        if min_words <= word_count <= max_words:
            filtered.append(p)
    removed = len(prompts) - len(filtered)
    if removed > 0:
        logger.info("Filtered %d prompts by length (%d-%d words)", removed, min_words, max_words)
    return filtered


def deduplicate_prompts(
    prompts: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicate prompts via cosine similarity.

    Args:
        prompts: List of prompt dicts with full_prompt key.
        threshold: Cosine similarity threshold above which to remove.

    Returns:
        Deduplicated list of prompt dicts.
    """
    if len(prompts) <= 1:
        return prompts

    model = SentenceTransformer(DEDUP_MODEL_NAME)
    texts = [p["full_prompt"] for p in prompts]
    embeddings = model.encode(texts, normalize_embeddings=True)

    keep_mask = [True] * len(prompts)
    for i in range(len(prompts)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(prompts)):
            if not keep_mask[j]:
                continue
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            if similarity > threshold:
                keep_mask[j] = False

    deduped = [p for p, keep in zip(prompts, keep_mask) if keep]
    removed = len(prompts) - len(deduped)
    if removed > 0:
        logger.info("Removed %d near-duplicate prompts (threshold=%.2f)", removed, threshold)
    return deduped


def compute_domain_stats(prompts: list[dict]) -> dict[str, int]:
    """Compute per-domain counts and total for the L1 library.

    Args:
        prompts: List of prompt dicts with domain key.

    Returns:
        Dict mapping domain names to counts, plus "total" key.
    """
    stats: dict[str, int] = {}
    for p in prompts:
        domain = p.get("domain", "unknown")
        stats[domain] = stats.get(domain, 0) + 1
    stats["total"] = len(prompts)
    return stats


def save_l1_library(prompts: list[dict], path: Path) -> None:
    """Save L1 prompts to a JSON file.

    Args:
        prompts: List of prompt dicts to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d L1 prompts to %s", len(prompts), path)


def load_l1_library(path: Path) -> list[dict]:
    """Load L1 prompts from a JSON file.

    Args:
        path: Path to l1_library.json.

    Returns:
        List of prompt dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        msg = f"L1 library file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        prompts = json.load(f)
    logger.info("Loaded %d L1 prompts from %s", len(prompts), path)
    return prompts


def validate_l1_library(path: Path) -> dict[str, int]:
    """Load and validate an existing L1 library, returning domain stats.

    Args:
        path: Path to l1_library.json.

    Returns:
        Domain distribution stats dict (same as compute_domain_stats).

    Raises:
        FileNotFoundError: If the library file does not exist.
    """
    prompts = load_l1_library(path)
    stats = compute_domain_stats(prompts)

    logger.info("L1 library validated: %d prompts across %d domains", stats["total"], len(stats) - 1)
    for domain, count in sorted(stats.items()):
        if domain != "total":
            logger.info("  %s: %d prompts", domain, count)
    return stats


def generate_l1_library(
    client: AnthropicClient,
    output_path: Path,
    domains: list[str] | None = None,
    batches_per_domain: int = 10,
    skip_dedup: bool = False,
    temperature: float = L1_TEMPERATURE,
    max_tokens: int = L1_MAX_TOKENS,
) -> list[dict]:
    """Generate the full L1 library via Claude Sonnet 4.

    Args:
        client: Initialized Anthropic API client.
        output_path: Path to save the library JSON.
        domains: Task domains to generate for. Defaults to all 15.
        batches_per_domain: Number of API calls per domain (10 prompts each).
        skip_dedup: If True, skip deduplication (useful for inspection).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens per API response.

    Returns:
        List of generated prompt dicts.
    """
    domains = domains or TASK_DOMAINS

    all_prompts: list[dict] = []
    for domain in domains:
        logger.info("Generating L1 prompts for domain: %s", domain)
        for batch_idx in range(batches_per_domain):
            user_prompt = build_l1_generation_prompt(domain)
            response = client.generate(
                user_prompt=user_prompt,
                system_prompt=L1_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = parse_l1_response(response)
            for p in parsed:
                p["domain"] = domain
                p["batch_idx"] = batch_idx
            all_prompts.extend(parsed)
            logger.info(
                "  Batch %d/%d: parsed %d prompts",
                batch_idx + 1, batches_per_domain, len(parsed),
            )

    logger.info("Total raw prompts: %d", len(all_prompts))

    all_prompts = filter_by_length(all_prompts)

    if not skip_dedup:
        all_prompts = deduplicate_prompts(all_prompts)

    stats = compute_domain_stats(all_prompts)
    for domain, count in sorted(stats.items()):
        if domain != "total":
            logger.info("  %s: %d prompts", domain, count)
    logger.info("Final L1 library size: %d", stats["total"])
    save_l1_library(all_prompts, output_path)
    return all_prompts
