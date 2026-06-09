"""Aligned SFT examples builder.

Constructs 7,000 fully aligned training examples where all hierarchy levels
are present and consistent (no conflicts):

- 5,000 via simple assembly: instruction->L3, input->L4, domain-matched L1,
  templated L2, sampled L0 rules.
- 2,000 via GPT-4o context synthesis: decomposing complex instructions across
  L1/L2/L3/L4 hierarchy levels.
"""

import json
import logging
import random
from pathlib import Path

from src.data.libraries.l0_rules import L0Rule, sample_l0_rules
from src.data.libraries.l2_generator import generate_l2_from_response
from src.data.libraries.l2_templates import generate_l2
from src.data.sft.assembly import assemble_sft_example
from src.data.sft.domain_classifier import classify_domain, select_matched_l1
from src.data.sft.row_utils import get_input, get_output

logger = logging.getLogger(__name__)

CONTEXT_SYNTHESIS_SYSTEM_PROMPT: str = (
    "You are a dataset engineer. Decompose the following instruction into "
    "components belonging to different hierarchy levels of an LLM system."
)

CONTEXT_SYNTHESIS_MODEL: str = "gpt-4o"
CONTEXT_SYNTHESIS_TEMPERATURE: float = 0.3
CONTEXT_SYNTHESIS_MAX_TOKENS: int = 500


def build_context_synthesis_prompt(instruction: str, data: str = "") -> str:
    """Build the GPT-4o user prompt for decomposing an instruction across levels.

    Args:
        instruction: The original user instruction to decompose.
        data: Optional input/data field from the base example.

    Returns:
        The user message string for the context synthesis API call.
    """
    return (
        'Original instruction: "%s"\n'
        'Original input/data: "%s"\n'
        "\n"
        "Decompose into:\n"
        "- L1 (developer system prompt): Persistent persona and behavioural constraints\n"
        "- L2 (user configuration): Preferences like language, format, length\n"
        "- L3 (user message): The core request or question\n"
        "- L4 (data): Reference data or context\n"
        "\n"
        'Output as JSON: {"l1": "...", "l2": "...", "l3": "...", "l4": "..."}\n'
        "The combination of all components must be semantically equivalent to the original.\n"
        "Do not add new requirements not present in the original."
    ) % (instruction, data)


def parse_context_synthesis_response(response: str) -> dict | None:
    """Parse a JSON response from the context synthesis API call.

    Handles raw JSON and markdown code fences (```json ... ```).
    Returns None on any parsing failure.

    Args:
        response: The raw response string from the API.

    Returns:
        Dict with keys l1, l2, l3, and optionally l4, or None on failure.
    """
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse context synthesis response: %.80s...", text)
        return None

    if not isinstance(parsed, dict):
        logger.warning("Context synthesis response is not a dict")
        return None

    required_keys = {"l1", "l2", "l3"}
    if not required_keys.issubset(parsed.keys()):
        logger.warning(
            "Context synthesis response missing keys: %s",
            required_keys - parsed.keys(),
        )
        return None

    # LLM occasionally returns nested dicts/lists instead of strings;
    # coerce all values to plain strings so downstream .strip() calls work.
    for key in list(parsed.keys()):
        if not isinstance(parsed[key], str):
            parsed[key] = json.dumps(parsed[key], ensure_ascii=False)

    return parsed


def build_simple_aligned(
    base_rows: list[dict],
    l0_rules: list[L0Rule],
    l1_library: list[dict],
    l4_lookup: dict[tuple[str, int], dict[str, str]],
    count: int = 5000,
    seed: int = 42,
    openai_client: object | None = None,
    l2_cache: dict[tuple[str, int], str] | None = None,
) -> list[dict]:
    """Build aligned SFT examples via simple assembly.

    For each row: classify domain, match L1, generate L2, sample L0 rules,
    look up L4, and wrap with assemble_sft_example.

    Args:
        base_rows: List of tagged dicts with instruction and output/response
            keys, plus ``_sft_source`` and ``_sft_index`` tags.
        l0_rules: Full list of L0Rule objects to sample from.
        l1_library: List of L1 prompt dicts with domain key.
        l4_lookup: Dict mapping (source, index) to {"l4_content": str, "generation": str}.
        count: Number of examples to produce.
        seed: Random seed for reproducibility.

    Returns:
        List of assembled SFT example dicts.
    """
    rng = random.Random(seed)
    indices = list(range(len(base_rows)))
    rng.shuffle(indices)
    selected = indices[:count]

    examples: list[dict] = []
    for i, row_idx in enumerate(selected):
        row = base_rows[row_idx]
        row_seed = seed + i

        instruction = row["instruction"]
        domain = classify_domain(instruction)
        l1 = select_matched_l1(l1_library, domain, seed=row_seed)
        sampled_l0 = sample_l0_rules(l0_rules, seed=row_seed)

        l4_key = (row["_sft_source"], row["_sft_index"])
        if openai_client is not None:
            cached_l2 = (l2_cache or {}).get(l4_key)
            if cached_l2 is not None:
                l2_text = cached_l2
            else:
                l2_text = generate_l2_from_response(
                    openai_client,
                    l1_prompt=l1["full_prompt"],
                    l3_message=instruction,
                    response=get_output(row),
                )
        else:
            l2_text = generate_l2(seed=row_seed)

        l4_entry = l4_lookup.get(l4_key)
        if l4_entry is None:
            raise ValueError(
                "Simple aligned row %s has no L4 entry — ensure base_rows "
                "are filtered to rows with L4 library coverage" % (l4_key,)
            )
        l4_content = l4_entry["l4_content"]
        l4_generation = l4_entry["generation"]

        example = assemble_sft_example(
            response=get_output(row),
            levels_present=[0, 1, 2, 3, 4],
            is_conflict=False,
            conflict_type=None,
            l0_rules=[r.rule for r in sampled_l0],
            l1_prompt=l1["full_prompt"],
            l2_config=l2_text,
            l3_message=instruction,
            l4_data=l4_content,
            sft_source=row["_sft_source"],
            sft_index=row["_sft_index"],
            sft_category="simple_aligned",
            l4_generation=l4_generation,
        )
        examples.append(example)

    logger.info("Built %d simple aligned examples", len(examples))
    return examples


def _flush_synthesis_cache(examples: list[dict], path: Path) -> None:
    """Write synthesis examples to a JSONL cache file.

    Args:
        examples: List of SFT example dicts to persist.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Flushed %d synthesis examples to %s", len(examples), path)


def build_context_synthesis_aligned(
    base_rows: list[dict],
    l0_rules: list[L0Rule],
    client: object,
    count: int = 2000,
    seed: int = 42,
    model: str = CONTEXT_SYNTHESIS_MODEL,
    temperature: float = CONTEXT_SYNTHESIS_TEMPERATURE,
    max_tokens: int = CONTEXT_SYNTHESIS_MAX_TOKENS,
    flush_path: Path | None = None,
    flush_every: int = 100,
    skip_indices: set[tuple[str, int]] | None = None,
    l4_lookup: dict[tuple[str, int], dict[str, str]] | None = None,
) -> list[dict]:
    """Build aligned SFT examples via GPT-4o context synthesis.

    Decomposes complex instructions across hierarchy levels using the API,
    then assembles each decomposed result with sampled L0 rules.

    Supports incremental checkpointing via ``flush_path`` and resumption
    via ``skip_indices``.

    Args:
        base_rows: List of tagged dicts with instruction and output/response keys.
        l0_rules: Full list of L0Rule objects to sample from.
        client: API client with .generate(user_prompt, system_prompt,
            model, temperature, max_tokens) method.
        count: Number of examples to produce.
        seed: Random seed for reproducibility.
        model: Model name for the API call.
        temperature: Temperature for the API call.
        max_tokens: Max tokens for the API call.
        flush_path: If provided, write intermediate results to this JSONL file
            every ``flush_every`` API calls.
        flush_every: How often to flush results to disk (default: 100).
        skip_indices: Set of (source, index) tuples to skip. Used for resuming
            from a previous run.

    Returns:
        List of assembled SFT example dicts.
    """
    rng = random.Random(seed)
    indices = list(range(len(base_rows)))
    rng.shuffle(indices)
    selected = indices[:count]

    examples: list[dict] = []
    failures = 0
    api_calls = 0

    for i, row_idx in enumerate(selected):
        row = base_rows[row_idx]
        row_seed = seed + i

        # Skip rows that were already processed in a previous run
        row_key = (row["_sft_source"], row["_sft_index"])
        if skip_indices and row_key in skip_indices:
            continue

        instruction = row["instruction"]
        data = get_input(row)
        user_prompt = build_context_synthesis_prompt(instruction, data)

        raw_response = client.generate(
            user_prompt=user_prompt,
            system_prompt=CONTEXT_SYNTHESIS_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        api_calls += 1

        parsed = parse_context_synthesis_response(raw_response)
        if parsed is None:
            failures += 1
            continue

        sampled_l0 = sample_l0_rules(l0_rules, seed=row_seed)

        l4_content = parsed.get("l4")
        l4_gen: str | None = "context_synthesis"

        if not l4_content or not l4_content.strip():
            # Fallback to L4 library
            l4_key = (row["_sft_source"], row["_sft_index"])
            l4_entry = (l4_lookup or {}).get(l4_key)
            if l4_entry is not None:
                l4_content = l4_entry["l4_content"]
                l4_gen = l4_entry["generation"]
            else:
                l4_content = None
                l4_gen = None
                logger.warning(
                    "Context synthesis row %s: no L4 from GPT-4o or library",
                    l4_key,
                )

        levels_present = [0, 1, 2, 3, 4] if l4_content else [0, 1, 2, 3]

        example = assemble_sft_example(
            response=get_output(row),
            levels_present=levels_present,
            is_conflict=False,
            conflict_type=None,
            l0_rules=[r.rule for r in sampled_l0],
            l1_prompt=parsed["l1"],
            l2_config=parsed["l2"],
            l3_message=parsed["l3"],
            l4_data=l4_content,
            sft_source=row["_sft_source"],
            sft_index=row["_sft_index"],
            sft_category="context_synthesis",
            l4_generation=l4_gen,
        )
        examples.append(example)

        if api_calls % flush_every == 0:
            logger.info(
                "Context synthesis progress: %d/%d (failures: %d)",
                i + 1, len(selected), failures,
            )
            if flush_path is not None:
                _flush_synthesis_cache(examples, flush_path)

    # Final flush to ensure all results are saved
    if flush_path is not None and examples:
        _flush_synthesis_cache(examples, flush_path)

    logger.info(
        "Built %d context synthesis aligned examples (%d failures, %d API calls)",
        len(examples), failures, api_calls,
    )
    return examples
