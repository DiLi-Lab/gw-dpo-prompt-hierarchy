"""End-to-end SEP runner.

Loads the subsample CSV → formats prompts via an injected callable →
generates responses (resumable JSONL cache) → scores each response with
the witness scorer + non-trivial gate → aggregates metrics.

External dependencies are injected so unit tests can mock generation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.evaluation.external.sep.data import SEPRecord, load_sep_csv
from src.evaluation.external.sep.metrics import aggregate_sep_metrics
from src.evaluation.external.sep.scorer import (
    is_non_trivial,
    is_witness_present,
    skip_reason,
)
from src.evaluation.generation import GenerateBatchFn, generate_responses

logger = logging.getLogger(__name__)

FormatRecordFn = Callable[[SEPRecord], str]


def _load_jsonl_index(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    index: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec:
                index[int(rec["id"])] = rec
    return index


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_sep_with_callables(
    *,
    csv_path: Path,
    output_dir: Path,
    format_record_fn: FormatRecordFn,
    generate_batch_fn: GenerateBatchFn,
    generation_batch_size: int,
    scoring_min_tokens: int,
    scoring_refusal_patterns: tuple[str, ...],
    run_metadata: dict[str, Any],
) -> dict:
    """Run SEP end-to-end and return the aggregated metrics dict.

    Side effects:
        Writes ``responses.jsonl``, ``scoring.jsonl``, and
        ``metrics.json`` under ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_sep_csv(csv_path)
    # Keep id as int so responses.jsonl and scoring.jsonl agree on type.
    eval_records = [
        {"id": r.id, "prompt": format_record_fn(r)} for r in records
    ]

    # --- Generation (resumable) ----------------------------------------
    responses = generate_responses(
        eval_records,
        output_dir / "responses.jsonl",
        generate_batch_fn,
        generation_batch_size,
    )
    response_by_id = {int(r["id"]): r["response"] for r in responses}

    # --- Scoring (resumable via per-id cache) --------------------------
    scoring_path = output_dir / "scoring.jsonl"
    cached_scores = _load_jsonl_index(scoring_path)
    scored_records: list[dict] = []
    for r in records:
        if r.id in cached_scores:
            scored_records.append(cached_scores[r.id])
            continue
        response = response_by_id.get(r.id, "")
        non_trivial = is_non_trivial(
            response,
            min_tokens=scoring_min_tokens,
            refusal_patterns=scoring_refusal_patterns,
        )
        rec = {
            "id": r.id,
            "domain": r.domain,
            "witness": r.witness,
            "response": response,
            "witness_match": is_witness_present(response, r.witness),
            "non_trivial": non_trivial,
            "skip_reason": (
                None if non_trivial
                else skip_reason(
                    response,
                    min_tokens=scoring_min_tokens,
                    refusal_patterns=scoring_refusal_patterns,
                )
            ),
        }
        _append_jsonl(scoring_path, rec)
        scored_records.append(rec)

    # --- Aggregate -----------------------------------------------------
    metrics = aggregate_sep_metrics(scored_records)
    metrics["run_metadata"] = {
        **run_metadata,
        "min_tokens": scoring_min_tokens,
        "scoring_method": "exact_substring",
        "refusal_patterns": list(scoring_refusal_patterns),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
