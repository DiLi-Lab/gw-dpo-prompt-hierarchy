"""Driver test: full pipeline with mocked generation and judge."""

import json
from pathlib import Path

from src.evaluation.run_eval import run_evaluation_with_callables


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_run_evaluation_end_to_end(tmp_path: Path) -> None:
    conflict_path = tmp_path / "conflicts.jsonl"
    aligned_path = tmp_path / "aligned.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    out_dir = tmp_path / "results"

    conflicts = [
        {"id": "eval_0001", "prompt": "p1", "conflict_type": "L0_vs_L4",
         "victim_level": 0, "attacker_level": 4, "level_gap": 4,
         "conflict_description": "x", "correct_behaviour": "y",
         "violation_behaviour": "z", "evaluation_criteria": ["c"]},
        {"id": "eval_0002", "prompt": "p2", "conflict_type": "L1_vs_L3",
         "victim_level": 1, "attacker_level": 3, "level_gap": 2,
         "conflict_description": "x", "correct_behaviour": "y",
         "violation_behaviour": "z", "evaluation_criteria": ["c"]},
    ]
    aligned = [
        {"id": "ctrl_0001", "matched_conflict_id": "eval_0001", "prompt": "p1c"},
        {"id": "ctrl_0002", "matched_conflict_id": "eval_0002", "prompt": "p2c"},
    ]
    reference = [
        {"id": "ref_0001", "source_conflict_id": "eval_0001", "prompt": "p1f",
         "conflict_type": "L0_vs_L4", "victim_level": 0, "attacker_level": 4,
         "level_gap": 4, "conflict_description": "x", "correct_behaviour": "y",
         "violation_behaviour": "z", "evaluation_criteria": ["c"]},
    ]
    _write_jsonl(conflict_path, conflicts)
    _write_jsonl(aligned_path, aligned)
    _write_jsonl(reference_path, reference)

    def fake_gen(prompts: list[str]) -> list[str]:
        return ["model output complies"] * len(prompts)

    def fake_judge(_sys: str, _user: str) -> str:
        return json.dumps({
            "satisfies_higher_level": True,
            "follows_lower_level": False,
            "refuses": False,
            "per_criterion": [True],
            "score": 5,
            "reasoning": "ok",
        })

    metrics = run_evaluation_with_callables(
        conflict_path=conflict_path,
        aligned_path=aligned_path,
        reference_path=reference_path,
        output_dir=out_dir,
        generate_batch_fn=fake_gen,
        judge_fn=fake_judge,
        generation_batch_size=2,
        orr_min_chars=200,
        run_text_similarity=False,
        run_rewards=False,
    )

    assert metrics["ppa_macro"] == 1.0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "responses_conflicts.jsonl").exists()
    assert (out_dir / "responses_aligned.jsonl").exists()
    assert (out_dir / "scoring_conflicts.jsonl").exists()
    # ORR: zero refusals when fake_gen never matches a refusal pattern
    assert metrics["orr_overall"] == 0.0


def test_run_evaluation_resumes_from_caches(tmp_path: Path) -> None:
    """A second invocation hits all caches and never re-invokes generation/judge."""
    conflict_path = tmp_path / "conflicts.jsonl"
    aligned_path = tmp_path / "aligned.jsonl"
    out_dir = tmp_path / "results"

    conflicts = [
        {"id": "eval_0001", "prompt": "p", "conflict_type": "L0_vs_L4",
         "victim_level": 0, "attacker_level": 4, "level_gap": 4,
         "conflict_description": "x", "correct_behaviour": "y",
         "violation_behaviour": "z", "evaluation_criteria": ["c"]},
    ]
    aligned = [
        {"id": "ctrl_0001", "matched_conflict_id": "eval_0001", "prompt": "p"},
    ]
    _write_jsonl(conflict_path, conflicts)
    _write_jsonl(aligned_path, aligned)

    gen_calls = [0]
    judge_calls = [0]

    def gen(prompts: list[str]) -> list[str]:
        gen_calls[0] += 1
        return ["resp"] * len(prompts)

    def judge(_s: str, _u: str) -> str:
        judge_calls[0] += 1
        return json.dumps({
            "satisfies_higher_level": True,
            "follows_lower_level": False,
            "refuses": False,
            "per_criterion": [True],
            "score": 5,
            "reasoning": "",
        })

    run_evaluation_with_callables(
        conflict_path=conflict_path,
        aligned_path=aligned_path,
        reference_path=None,
        output_dir=out_dir,
        generate_batch_fn=gen,
        judge_fn=judge,
        generation_batch_size=2,
        orr_min_chars=200,
    )
    first_gen, first_judge = gen_calls[0], judge_calls[0]

    run_evaluation_with_callables(
        conflict_path=conflict_path,
        aligned_path=aligned_path,
        reference_path=None,
        output_dir=out_dir,
        generate_batch_fn=gen,
        judge_fn=judge,
        generation_batch_size=2,
        orr_min_chars=200,
    )
    # Second run should hit caches: generation not called again; judge not
    # called again for PPA scoring.
    assert gen_calls[0] == first_gen
    assert judge_calls[0] == first_judge
