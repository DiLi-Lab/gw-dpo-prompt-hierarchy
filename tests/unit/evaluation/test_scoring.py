"""Tests for the PPA scoring driver."""

import json
from pathlib import Path

from src.evaluation.scoring import score_responses


def _eval_record(eid: str = "eval_0001", conflict_type: str = "L1_vs_L3") -> dict:
    return {
        "id": eid,
        "prompt": "<L0>...<L4>...",
        "conflict_type": conflict_type,
        "victim_level": int(conflict_type[1]),
        "attacker_level": int(conflict_type[-1]),
        "conflict_description": "x",
        "correct_behaviour": "y",
        "violation_behaviour": "z",
        "evaluation_criteria": ["c1"],
    }


def test_score_responses_caches_and_resumes(tmp_path: Path) -> None:
    """The driver must skip already-judged ids on a second pass."""
    eval_records = [_eval_record()]
    response_records = [{"id": "eval_0001", "response": "model output"}]
    cache_path = tmp_path / "scoring_cache.jsonl"

    judge_calls: list[str] = []

    def fake_judge(_system_prompt: str, user_prompt: str) -> str:
        judge_calls.append(user_prompt)
        return json.dumps({
            "satisfies_higher_level": True,
            "follows_lower_level": False,
            "refuses": False,
            "per_criterion": [True],
            "score": 5,
            "reasoning": "ok",
        })

    out1 = score_responses(eval_records, response_records, cache_path, fake_judge)
    assert len(out1) == 1
    assert out1[0]["satisfies_higher_level"] is True
    assert len(judge_calls) == 1

    # Second call with same inputs reads from cache, judge not invoked again
    out2 = score_responses(eval_records, response_records, cache_path, fake_judge)
    assert len(out2) == 1
    assert out2[0]["satisfies_higher_level"] is True
    assert len(judge_calls) == 1


def test_score_responses_handles_missing_responses(tmp_path: Path) -> None:
    """Eval records without a generated response are skipped, not judged."""
    eval_records = [_eval_record(eid="eval_0001", conflict_type="L0_vs_L4")]
    response_records: list[dict] = []
    cache_path = tmp_path / "cache.jsonl"

    def explode(*_args: str) -> str:
        raise AssertionError("judge must not be called when no response exists")

    out = score_responses(eval_records, response_records, cache_path, explode)
    assert out == []


def test_score_responses_writes_jsonl(tmp_path: Path) -> None:
    eval_records = [_eval_record(eid="eval_0001", conflict_type="L0_vs_L4")]
    response_records = [{"id": "eval_0001", "response": "answer"}]
    cache_path = tmp_path / "cache.jsonl"

    def fake_judge(_s: str, _u: str) -> str:
        return json.dumps({
            "satisfies_higher_level": False,
            "follows_lower_level": True,
            "refuses": False,
            "per_criterion": [False],
            "score": 1,
            "reasoning": "x",
        })

    score_responses(eval_records, response_records, cache_path, fake_judge)
    assert cache_path.exists()
    lines = cache_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == "eval_0001"
    assert rec["conflict_type"] == "L0_vs_L4"
