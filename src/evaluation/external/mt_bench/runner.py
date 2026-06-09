"""End-to-end MT-Bench runner.

Loads vendored data → generates turn 1 then turn 2 per category at the
category's temperature → judges each turn → aggregates metrics.
External dependencies (model loading, OpenAI calls) are injected so
unit tests can mock generation and judging.

On-disk layout (under output_dir):
    responses.jsonl   keyed by id="q{qid}_t{turn}"   (resumable)
    scoring.jsonl     keyed by (question_id, turn)   (resumable)
    metrics.json      written once at successful completion
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.evaluation.external.mt_bench.data import (
    MTBenchQuestion,
    load_judge_prompts,
    load_questions,
    load_reference_answers,
)
from src.evaluation.external.mt_bench.judge import (
    build_judge_prompt,
    parse_score,
)
from src.evaluation.external.mt_bench.metrics import aggregate_mt_bench_metrics
from src.evaluation.external.mt_bench.prompt import (
    build_turn1_prompt,
    build_turn2_prompt,
)
from src.evaluation.generation import GenerateBatchFn, generate_responses

logger = logging.getLogger(__name__)


JudgeCallFn = Callable[..., str]
"""Callable signature: judge_fn(*, system_prompt, user_prompt, temperature) -> str."""


_RUNTIME_METADATA_KEYS: frozenset[str] = frozenset({
    "judge_temperature",
    "judge_temperature_retry",
    "temperature_per_category",
    "timestamp",
})


def _turn_key(qid: int, turn: int) -> str:
    return f"q{qid}_t{turn}"


def _scoring_record_key(rec: dict) -> tuple[int, int]:
    return (int(rec["question_id"]), int(rec["turn"]))


def _load_scoring_index(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    out: dict[tuple[int, int], dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[_scoring_record_key(rec)] = rec
    return out


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _judge_one(
    *,
    judge_fn: JudgeCallFn,
    system_prompt: str,
    user_prompt: str,
    judge_temperature: float,
    judge_temperature_retry: float,
) -> tuple[float | None, str, int]:
    """Issue judge call with one retry on parse failure.

    Returns (score, raw_judge_text, retry_count).
    """
    text = judge_fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=judge_temperature,
    )
    score = parse_score(text)
    if score is not None:
        return score, text, 0
    text_retry = judge_fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=judge_temperature_retry,
    )
    score_retry = parse_score(text_retry)
    return score_retry, text_retry, 1


def run_mt_bench_with_callables(
    *,
    questions_path: Path,
    references_path: Path,
    judge_prompts_path: Path,
    output_dir: Path,
    tokenizer: Any,
    generate_batch_fn_for_temperature: Callable[[float], GenerateBatchFn],
    temperature_per_category: dict[str, float],
    generation_batch_size: int,
    judge_fn: JudgeCallFn,
    judge_temperature: float,
    judge_temperature_retry: float,
    run_metadata: dict[str, Any],
) -> dict:
    """Run MT-Bench end-to-end and return the aggregated metrics dict.

    Side effects:
        Writes ``responses.jsonl``, ``scoring.jsonl``, ``metrics.json``
        under ``output_dir``.
    """
    collisions = _RUNTIME_METADATA_KEYS & set(run_metadata.keys())
    if collisions:
        msg = (
            f"run_metadata cannot contain runtime-reserved keys "
            f"{sorted(collisions)}; the runner sets these from its own state."
        )
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path, expect_count=None)
    references = load_reference_answers(references_path)
    templates = load_judge_prompts(judge_prompts_path)

    responses_path = output_dir / "responses.jsonl"
    scoring_path = output_dir / "scoring.jsonl"

    # --- Generation: per category, turn 1 then turn 2 ------------------
    grouped_by_cat: dict[str, list[MTBenchQuestion]] = {}
    for q in questions:
        grouped_by_cat.setdefault(q.category, []).append(q)

    for category, qs in grouped_by_cat.items():
        if category not in temperature_per_category:
            msg = f"category {category!r} missing from temperature_per_category"
            raise ValueError(msg)
        temp = temperature_per_category[category]
        gen_fn = generate_batch_fn_for_temperature(temp)

        # Turn 1: independent of any prior generation.
        t1_records = [
            {"id": _turn_key(q.question_id, 1),
             "prompt": build_turn1_prompt(tokenizer, q)}
            for q in qs
        ]
        t1_responses = generate_responses(
            t1_records, responses_path, gen_fn, generation_batch_size,
        )
        t1_resp_by_qid = {
            q.question_id: r["response"] for q, r in zip(qs, t1_responses)
        }

        # Turn 2: prompt depends on the model's turn-1 response.
        t2_records = [
            {"id": _turn_key(q.question_id, 2),
             "prompt": build_turn2_prompt(
                 tokenizer, q, t1_resp_by_qid[q.question_id],
             )}
            for q in qs
        ]
        generate_responses(
            t2_records, responses_path, gen_fn, generation_batch_size,
        )

    # --- Index responses for the judging pass --------------------------
    response_by_key: dict[str, str] = {}
    if responses_path.exists():
        with responses_path.open() as f:
            for line in f:
                rec = json.loads(line)
                response_by_key[rec["id"]] = rec["response"]

    # --- Judging: 2 calls per question, cached per (qid, turn) --------
    cached_scores = _load_scoring_index(scoring_path)
    scored_records: list[dict] = list(cached_scores.values())

    for q in questions:
        r1 = response_by_key.get(_turn_key(q.question_id, 1), "")
        r2 = response_by_key.get(_turn_key(q.question_id, 2), "")
        ref = references.get(q.question_id)
        for turn in (1, 2):
            key = (q.question_id, turn)
            if key in cached_scores:
                continue
            try:
                call = build_judge_prompt(
                    templates=templates, question=q, reference=ref,
                    turn=turn, responses=(r1, r2),
                )
            except ValueError as exc:
                logger.error(
                    "Skipping judge for q=%d turn=%d: %s",
                    q.question_id, turn, exc,
                )
                rec = {
                    "question_id": q.question_id, "turn": turn,
                    "category": q.category,
                    "judge_template": None,
                    "score": None, "judge_response": None,
                    "parse_error": True, "retry_count": 0,
                    "error": str(exc),
                }
                _append_jsonl(scoring_path, rec)
                scored_records.append(rec)
                continue

            score, text, retry_count = _judge_one(
                judge_fn=judge_fn,
                system_prompt=call.system_prompt,
                user_prompt=call.user_prompt,
                judge_temperature=judge_temperature,
                judge_temperature_retry=judge_temperature_retry,
            )
            rec = {
                "question_id": q.question_id, "turn": turn,
                "category": q.category,
                "judge_template": call.template_name,
                "score": score,
                "judge_response": text,
                "parse_error": score is None,
                "retry_count": retry_count,
            }
            _append_jsonl(scoring_path, rec)
            scored_records.append(rec)

    # --- Aggregate ----------------------------------------------------
    metrics = aggregate_mt_bench_metrics(scored_records)
    metrics["run_metadata"] = {
        **run_metadata,
        "judge_temperature": judge_temperature,
        "judge_temperature_retry": judge_temperature_retry,
        "temperature_per_category": dict(temperature_per_category),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
