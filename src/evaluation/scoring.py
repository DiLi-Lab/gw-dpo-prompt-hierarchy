"""Scoring driver: run the PPA judge over generated responses with caching.

Reads (eval_records, response_records). For each pair, builds the judge
prompt, calls the judge, parses the response, and persists the parsed
record to a JSONL cache keyed by ``id``. Resumable: a second pass with
the same cache file skips already-judged ids without invoking the judge.
"""

import json
import logging
from pathlib import Path
from typing import Callable

from src.evaluation.judges import (
    PPA_JUDGE_SYSTEM_PROMPT,
    build_ppa_judge_prompt,
    parse_ppa_judgement,
)

logger = logging.getLogger(__name__)

JudgeFn = Callable[[str, str], str]
"""Callable taking (system_prompt, user_prompt) and returning judge output."""


def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cached: dict[str, dict] = {}
    with cache_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                cached[rec["id"]] = rec
    return cached


def _append_cache(cache_path: Path, record: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def score_responses(
    eval_records: list[dict],
    response_records: list[dict],
    cache_path: Path,
    judge_fn: JudgeFn,
) -> list[dict]:
    """Judge each response against its eval record; cache results.

    Args:
        eval_records: The conflict / aligned / reference records (must
            include ``id`` and the fields needed by ``build_ppa_judge_prompt``).
        response_records: List of ``{"id": ..., "response": ...}``.
        cache_path: JSONL path; results are appended.
        judge_fn: Callable taking (system_prompt, user_prompt) returning
            the judge's raw text.

    Returns:
        Judged records (one per eval record that had a response), with
        the parsed judgement fields plus ``id`` and ``conflict_type``.
    """
    response_by_id: dict[str, str] = {
        r["id"]: r["response"] for r in response_records if "response" in r
    }
    cached = _load_cache(cache_path)
    out: list[dict] = []

    for er in eval_records:
        eid = er["id"]
        if eid not in response_by_id:
            continue

        if eid in cached:
            out.append(cached[eid])
            continue

        criteria = er.get("evaluation_criteria", [])
        user_prompt = build_ppa_judge_prompt(
            prompt_text=er["prompt"],
            response=response_by_id[eid],
            conflict_type=er["conflict_type"],
            victim_level=er["victim_level"],
            attacker_level=er["attacker_level"],
            conflict_description=er.get("conflict_description", ""),
            correct_behaviour=er.get("correct_behaviour", ""),
            violation_behaviour=er.get("violation_behaviour", ""),
            evaluation_criteria=criteria,
        )
        try:
            raw = judge_fn(PPA_JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("Judge call failed for %s: %s", eid, exc)
            raw = ""

        parsed = parse_ppa_judgement(raw, num_criteria=len(criteria))
        record = {"id": eid, "conflict_type": er["conflict_type"], **parsed}
        _append_cache(cache_path, record)
        out.append(record)

    return out
