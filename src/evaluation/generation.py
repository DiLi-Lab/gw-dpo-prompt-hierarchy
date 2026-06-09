"""Batched generation runner with resumable JSONL cache.

This module is decoupled from the model: callers pass a
``generate_batch_fn`` that takes a list of prompt strings and returns
a list of response strings of equal length. The runner handles batching,
caching, and ordering so the model loader can swap freely.
"""

import json
import logging
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

GenerateBatchFn = Callable[[list[str]], list[str]]


def _load_cached_responses(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    cached: dict[str, str] = {}
    with cache_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec and "response" in rec:
                cached[rec["id"]] = rec["response"]
    return cached


def _append(cache_path: Path, rec: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def generate_responses(
    eval_records: list[dict],
    cache_path: Path,
    generate_batch_fn: GenerateBatchFn,
    batch_size: int,
) -> list[dict]:
    """Generate one response per eval record; cache to JSONL.

    Args:
        eval_records: Records with at least ``id`` and ``prompt``.
        cache_path: JSONL path; existing entries are loaded and skipped.
        generate_batch_fn: Callable taking a list of prompts and returning
            a list of responses (same length).
        batch_size: Number of prompts per ``generate_batch_fn`` call.

    Returns:
        List of ``{"id": ..., "response": ...}`` in the order of
        ``eval_records``.
    """
    cached = _load_cached_responses(cache_path)
    pending: list[dict] = [r for r in eval_records if r["id"] not in cached]
    n_cached = len(eval_records) - len(pending)

    # tqdm-based progress is the source of truth for visibility. The
    # earlier logger.info(...) per-batch was unreliable: imported
    # libraries (transformers / torch / datasets) frequently call
    # logging.basicConfig() before our CLI does, which makes any
    # later basicConfig in bin/run_*.py a no-op and suppresses our
    # INFO output entirely. tqdm writes directly to stderr so it
    # bypasses the logging machinery.
    pbar = tqdm(
        total=len(eval_records),
        desc="Generating",
        unit="prompt",
        initial=n_cached,
        dynamic_ncols=True,
    )
    try:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            prompts = [r["prompt"] for r in chunk]
            responses = generate_batch_fn(prompts)
            if len(responses) != len(prompts):
                msg = (
                    f"generate_batch_fn returned {len(responses)} for "
                    f"{len(prompts)} prompts"
                )
                raise RuntimeError(msg)
            for r, resp in zip(chunk, responses):
                rec = {"id": r["id"], "response": resp}
                cached[r["id"]] = resp
                _append(cache_path, rec)
            pbar.update(len(chunk))
    finally:
        pbar.close()

    return [{"id": r["id"], "response": cached[r["id"]]} for r in eval_records]
