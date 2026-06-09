"""L4 tool output wrapping and synthesis.

Source A (build_l4_wrapped): Wraps existing dataset fields (Alpaca input,
Dolly context) in realistic tool-output templates. No LLM needed.

Source B (synthesize_l4_outputs): Synthesises tool outputs via GPT-4o-mini
for examples with empty data fields. Requires OpenAI API access.

Both sources produce L4Entry objects with pointer-based provenance
(source dataset + row index), persisted as a unified l4_library.json.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Dataset

from src.api.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

MIN_CONTENT_CHARS: int = 200
"""Minimum character count for Source A content to be realistic as a tool output."""

PLACEHOLDER_PATTERNS: list[str] = [
    r"^\[insert\b.*\]$",
    r"^\[.*\bhere\]$",
    r"^\(.*\)$",
]
"""Regex patterns (case-insensitive) that match placeholder/template content."""


def is_placeholder(text: str) -> bool:
    """Check whether text matches a known placeholder pattern.

    Args:
        text: The content string to check.

    Returns:
        True if the text looks like a placeholder rather than real content.
    """
    import re

    stripped = text.strip()
    return any(re.match(p, stripped, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS)


TOOL_TEMPLATES: list[str] = [
    '<tool_output source="web_search" query="{query}">\n{content}\n</tool_output>',
    '<tool_output source="database" table="knowledge_base">\n{content}\n</tool_output>',
    '<tool_output source="api" endpoint="/v1/lookup">\n{{"status": "ok", "data": "{content}"}}\n</tool_output>',
    '<tool_output source="document_retrieval" doc_id="DOC_{doc_id}">\n{content}\n</tool_output>',
    '<tool_output source="code_execution" language="python">\nOutput:\n{content}\n</tool_output>',
    '{content}',
]

L4_SYNTHESIS_PROMPT_SYSTEM: str = (
    "Generate a realistic tool output that could serve as reference "
    "data for answering the following question. Format it as a search result, "
    "database entry, or API response. Aim for 100 to 200 words."
)


@dataclass
class L4Entry:
    """A single L4 tool-output entry with pointer-based provenance.

    Attributes:
        source: Origin dataset, either "alpaca" or "dolly".
        index: Row index in the source dataset's train split.
        l4_content: The wrapped or synthesised tool-output string.
        generation: How this entry was produced: "wrapped" or "synthesized".
    """

    source: str
    index: int
    l4_content: str
    generation: str

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "L4Entry":
        """Deserialise from a plain dict."""
        return cls(**d)


def wrap_as_l4(
    content: str,
    query: str = "",
    doc_id: str = "001",
    seed: int | None = None,
) -> str:
    """Wrap content in a randomly chosen tool-output template.

    Args:
        content: The data content to wrap.
        query: Optional query string for search templates.
        doc_id: Optional document ID for retrieval templates.
        seed: Random seed for template selection.

    Returns:
        Content wrapped in a tool-output template.
    """
    rng = random.Random(seed)
    template = rng.choice(TOOL_TEMPLATES)
    return template.format(content=content, query=query, doc_id=doc_id)


def build_l4_wrapped(
    dataset: Dataset,
    source: str,
    data_field: str,
) -> list[L4Entry]:
    """Wrap non-empty data fields as L4 tool outputs (Source A).

    Args:
        dataset: Base dataset split (Alpaca or Dolly train).
        source: Origin label, "alpaca" or "dolly".
        data_field: Column name containing the data to wrap
                    ("input" for Alpaca, "context" for Dolly).

    Returns:
        List of L4Entry objects for rows with non-empty data fields.
    """
    entries: list[L4Entry] = []
    skipped = 0
    for i in range(len(dataset)):
        content = dataset[i][data_field].strip()
        if not content:
            continue
        if len(content) < MIN_CONTENT_CHARS or is_placeholder(content):
            skipped += 1
            continue
        instruction = dataset[i]["instruction"]
        wrapped = wrap_as_l4(content, query=instruction, doc_id=str(i), seed=i)
        entries.append(L4Entry(
            source=source,
            index=i,
            l4_content=wrapped,
            generation="wrapped",
        ))

    if skipped:
        logger.info(
            "Source A (%s): filtered %d entries (short/placeholder)", source, skipped,
        )
    logger.info(
        "Source A (%s): wrapped %d non-empty %s fields as L4",
        source, len(entries), data_field,
    )
    return entries


def build_l4_synthesis_prompt(instruction: str) -> str:
    """Build the user prompt for GPT-4o-mini L4 synthesis.

    Args:
        instruction: The instruction to synthesise a tool output for.

    Returns:
        The user message string for the API call.
    """
    return f'Question: "{instruction}"'


def synthesize_l4_outputs(
    dataset: Dataset,
    client: OpenAIClient,
    source: str,
    data_field: str,
    max_examples: int | None = None,
    flush_path: Path | None = None,
    prior_entries: list[L4Entry] | None = None,
    flush_every: int = 100,
    skip_indices: set[tuple[str, int]] | None = None,
) -> list[L4Entry]:
    """Synthesise L4 tool outputs for examples with empty data fields (Source B).

    Args:
        dataset: Base dataset split (Alpaca or Dolly train).
        client: Initialised OpenAI API client.
        source: Origin label, "alpaca" or "dolly".
        data_field: Column name to check for emptiness
                    ("input" for Alpaca, "context" for Dolly).
        max_examples: Maximum number to synthesise. None means all empty rows.
        flush_path: If provided, intermediate results are saved to this path
                    every ``flush_every`` entries.
        prior_entries: Entries already accumulated (written alongside new
                       results during intermediate flushes).
        flush_every: How often to flush intermediate results (default 100).
        skip_indices: Set of (source, index) pairs to skip (already synthesised).
                      Used for resuming interrupted generation runs.

    Returns:
        List of L4Entry objects for synthesised tool outputs.
    """
    empty_indices = [
        i for i in range(len(dataset)) if not dataset[i][data_field].strip()
    ]

    if skip_indices:
        before = len(empty_indices)
        empty_indices = [i for i in empty_indices if (source, i) not in skip_indices]
        skipped = before - len(empty_indices)
        if skipped:
            logger.info(
                "Source B (%s): resuming — skipped %d already-synthesised entries",
                source, skipped,
            )

    if max_examples is not None:
        empty_indices = empty_indices[:max_examples]

    logger.info(
        "Source B (%s): synthesising L4 outputs for %d empty-%s examples",
        source, len(empty_indices), data_field,
    )

    prior = prior_entries or []
    results: list[L4Entry] = []
    for idx, i in enumerate(empty_indices):
        instruction = dataset[i]["instruction"]
        user_prompt = build_l4_synthesis_prompt(instruction)
        l4_content = client.generate(
            user_prompt=user_prompt,
            system_prompt=L4_SYNTHESIS_PROMPT_SYSTEM,
        )
        wrapped = wrap_as_l4(l4_content, query=instruction, seed=idx)
        results.append(L4Entry(
            source=source,
            index=i,
            l4_content=wrapped,
            generation="synthesized",
        ))
        if (idx + 1) % flush_every == 0:
            logger.info("  Synthesised %d/%d", idx + 1, len(empty_indices))
            if flush_path is not None:
                save_l4_library(prior + results, flush_path)

    return results


def save_l4_library(entries: list[L4Entry], path: Path) -> None:
    """Save L4 library entries to a JSON file.

    Args:
        entries: List of L4Entry objects (wrapped + synthesised).
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in entries], f, indent=2, ensure_ascii=False)
    logger.info("Saved %d L4 entries to %s", len(entries), path)


def load_l4_library(path: Path) -> list[L4Entry]:
    """Load L4 library entries from a JSON file.

    Args:
        path: Path to l4_library.json.

    Returns:
        List of L4Entry objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        msg = f"L4 library file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw = json.load(f)
    entries = [L4Entry.from_dict(d) for d in raw]
    logger.info("Loaded %d L4 entries from %s", len(entries), path)
    return entries


def validate_l4_library(entries: list[L4Entry]) -> dict:
    """Compute statistics for an L4 library.

    Args:
        entries: List of L4Entry objects.

    Returns:
        Dict with total count, per-source counts, and per-generation counts.
    """
    source_counts: dict[str, int] = {}
    generation_counts: dict[str, int] = {}
    for e in entries:
        source_counts[e.source] = source_counts.get(e.source, 0) + 1
        generation_counts[e.generation] = generation_counts.get(e.generation, 0) + 1

    return {
        "total": len(entries),
        "source_counts": source_counts,
        "generation_counts": generation_counts,
    }
