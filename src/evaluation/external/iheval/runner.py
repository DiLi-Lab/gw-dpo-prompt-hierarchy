"""End-to-end IHEval runner.

Walks the benchmark tree → formats prompts via an injected callable →
generates responses (resumable via the shared generation cache) →
scores each response with the upstream wrapper → aggregates metrics.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.evaluation.external.iheval.data import IHEvalRecord, iter_iheval_records
from src.evaluation.external.iheval.metrics import aggregate_iheval_metrics
from src.evaluation.external.iheval.scorers import score
from src.evaluation.generation import GenerateBatchFn, generate_responses

logger = logging.getLogger(__name__)

FormatRecordFn = Callable[[IHEvalRecord], str]


def _file_safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _load_scoring_index(path: Path) -> dict[str, dict]:
    """Index a per-group scoring JSONL by ``uid`` for resume."""
    if not path.exists():
        return {}
    index: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "uid" in rec:
                index[rec["uid"]] = rec
    return index


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_iheval_with_callables(
    *,
    benchmark_root: Path,
    output_dir: Path,
    tasks: Sequence[str],
    settings: Sequence[str],
    format_record_fn: FormatRecordFn,
    generate_batch_fn: GenerateBatchFn,
    generation_batch_size: int,
    run_metadata: dict[str, Any],
) -> dict:
    """Run IHEval end-to-end and return the aggregated metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = list(iter_iheval_records(
        benchmark_root, tuple(tasks), tuple(settings),
    ))

    # Group by (task, setting, sub) so each grouping gets its own
    # responses/scoring JSONL pair, mirroring the upstream layout.
    grouped: dict[tuple[str, str, str], list[IHEvalRecord]] = {}
    for r in all_records:
        grouped.setdefault((r.task, r.setting, r.sub), []).append(r)

    scored_records: list[dict] = []
    for (task, setting, sub), records in grouped.items():
        key_safe = _file_safe(f"{task}_{setting}__{sub}")
        responses_path = output_dir / f"responses.{key_safe}.jsonl"
        scoring_path = output_dir / f"scoring.{key_safe}.jsonl"

        eval_records = [
            {"id": r.uid, "prompt": format_record_fn(r)} for r in records
        ]
        responses = generate_responses(
            eval_records, responses_path,
            generate_batch_fn, generation_batch_size,
        )
        response_by_uid = {r["id"]: r["response"] for r in responses}

        # Score with per-uid append cache so a crash mid-group can resume.
        cached = _load_scoring_index(scoring_path)
        for rec in records:
            if rec.uid in cached:
                scored_records.append(cached[rec.uid])
                continue
            response = response_by_uid.get(rec.uid, "")
            try:
                out = score(rec.task, rec.answer, response)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "IHEval scorer failed task=%s uid=%s: %s",
                    rec.task, rec.uid, exc,
                )
                out = {"score": 0.0, "details": {"error": str(exc)}}
            line = {
                "uid": rec.uid,
                "task": rec.task, "setting": rec.setting, "sub": rec.sub,
                "id": rec.id,
                "score": out["score"],
                "details": out["details"],
            }
            _append_jsonl(scoring_path, line)
            scored_records.append(line)

    metrics = aggregate_iheval_metrics(
        scored_records, tasks_run=tuple(tasks),
    )
    metrics["run_metadata"] = {
        **run_metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_records": len(all_records),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
