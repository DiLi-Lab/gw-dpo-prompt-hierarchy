"""End-to-end XSTest runner.

Glues the data loader, prompt formatter, response generator (cached
JSONL), 3-class judge, and metrics aggregator into a single callable.
External dependencies are injected so unit tests can mock them.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.evaluation.external.xstest.data import load_xstest_csv
from src.evaluation.external.xstest.metrics import aggregate_xstest_metrics
from src.evaluation.external.xstest.scorer import (
    XSTEST_JUDGE_SYSTEM,
    build_judge_user_prompt,
    parse_judge_label,
)
from src.evaluation.generation import GenerateBatchFn, generate_responses

logger = logging.getLogger(__name__)

FormatPromptFn = Callable[[dict], str]
JudgeFn = Callable[[str, str], str]


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


def run_xstest_with_callables(
    *,
    csv_path: Path,
    output_dir: Path,
    format_prompt_fn: FormatPromptFn,
    generate_batch_fn: GenerateBatchFn,
    judge_fn: JudgeFn,
    generation_batch_size: int,
    run_metadata: dict[str, Any],
) -> dict:
    """Run XSTest end-to-end and return the aggregated metrics dict.

    Side effects:
        Writes ``responses.jsonl``, ``scoring.jsonl``, and
        ``metrics.json`` under ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_xstest_csv(csv_path)
    eval_records = [
        {"id": str(r.id), "prompt": format_prompt_fn(r.__dict__)}
        for r in records
    ]

    # --- Generation (resumable) -----------------------------------------
    responses = generate_responses(
        eval_records,
        output_dir / "responses.jsonl",
        generate_batch_fn,
        generation_batch_size,
    )
    response_by_id = {int(r["id"]): r["response"] for r in responses}

    # --- Scoring (resumable via per-id cache) ---------------------------
    scoring_path = output_dir / "scoring.jsonl"
    cached_scores = _load_jsonl_index(scoring_path)
    scored_records: list[dict] = []
    for r in records:
        if r.id in cached_scores:
            scored_records.append(cached_scores[r.id])
            continue
        response = response_by_id.get(r.id, "")
        try:
            raw = judge_fn(
                XSTEST_JUDGE_SYSTEM,
                build_judge_user_prompt(r.prompt, response),
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft logging on judge errors
            logger.error("XSTest judge call failed for id=%s: %s", r.id, exc)
            raw = ""
        label = parse_judge_label(raw)
        rec = {
            "id": r.id,
            "prompt": r.prompt,
            "label": r.label,
            "type": r.type,
            "response": response,
            "judge_label": label,
            "judge_raw": raw,
        }
        _append_jsonl(scoring_path, rec)
        scored_records.append(rec)

    # --- Aggregate ------------------------------------------------------
    metrics = aggregate_xstest_metrics(scored_records)
    metrics["run_metadata"] = {
        **run_metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
