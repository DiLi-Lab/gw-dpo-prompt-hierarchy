"""IHEval runner smoke test with mocked model + real upstream scorers."""

import json
from pathlib import Path

import pytest

from src.evaluation.external.iheval.runner import run_iheval_with_callables


def _write_min_tree(root: Path) -> Path:
    base = root / "benchmark" / "rule-following" / "single-turn"
    aligned = base / "aligned" / "default"
    conflict = base / "conflict" / "default"
    reference = base / "reference" / "default"
    for d in (aligned, conflict, reference):
        d.mkdir(parents=True)
    # Use a real, satisfiable IFEval rule so the upstream scorer returns 1.0
    # for our fake responses.
    payload_template = lambda id_: [
        {"id": id_, "system": "S", "instruction": "I",
         "answer": {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}},
    ]
    (aligned / "input_data.json").write_text(json.dumps(payload_template(1)))
    (conflict / "input_data.json").write_text(json.dumps(payload_template(2)))
    (reference / "input_data.json").write_text(json.dumps(payload_template(3)))
    return root / "benchmark"


def test_runner_smoke_writes_metrics(tmp_path: Path) -> None:
    bench_root = _write_min_tree(tmp_path)

    def fake_format(record):
        return f"PROMPT::{record.task}::{record.setting}::{record.id}"

    def fake_generate(prompts):
        # Plain comma-free responses → punctuation:no_comma rule passes.
        return [f"plain response for {i}" for i, _ in enumerate(prompts)]

    out_dir = tmp_path / "run"
    metrics = run_iheval_with_callables(
        benchmark_root=bench_root,
        output_dir=out_dir,
        tasks=("single-turn",),
        settings=("aligned", "conflict", "reference"),
        format_record_fn=fake_format,
        generate_batch_fn=fake_generate,
        generation_batch_size=2,
        run_metadata={"model": "fake", "format": "delimited", "ise_active": True},
    )

    assert (out_dir / "metrics.json").exists()
    on_disk = json.loads((out_dir / "metrics.json").read_text())
    assert on_disk["by_setting"]["aligned"]["single-turn"] == 1.0
    assert on_disk["by_setting"]["conflict"]["single-turn"] == 1.0
    assert on_disk["iheval_score"] == 1.0
    assert on_disk["ih_following"]["single-turn"] == 1.0
    assert on_disk["run_metadata"]["model"] == "fake"


def test_runner_resumes_scoring_from_cached_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running with same output_dir hits both response and scoring caches."""
    bench_root = _write_min_tree(tmp_path)

    from src.evaluation.external.iheval import runner as iheval_runner
    score_calls: list[tuple[str, str]] = []
    real_score = iheval_runner.score

    def counting_score(task, answer, response):
        score_calls.append((task, response))
        return real_score(task, answer, response)

    monkeypatch.setattr(iheval_runner, "score", counting_score)

    def fake_format(record):
        return f"PROMPT::{record.task}::{record.setting}::{record.id}"

    gen_calls: list[list[str]] = []

    def counting_generate(prompts):
        gen_calls.append(list(prompts))
        return [f"plain response for {i}" for i, _ in enumerate(prompts)]

    out_dir = tmp_path / "run"
    common = dict(
        benchmark_root=bench_root,
        output_dir=out_dir,
        tasks=("single-turn",),
        settings=("aligned", "conflict", "reference"),
        format_record_fn=fake_format,
        generate_batch_fn=counting_generate,
        generation_batch_size=2,
        run_metadata={"model": "fake", "format": "delimited", "ise_active": True},
    )

    run_iheval_with_callables(**common)
    first_gen = sum(len(c) for c in gen_calls)
    first_score = len(score_calls)

    gen_calls.clear()
    score_calls.clear()
    run_iheval_with_callables(**common)

    assert first_gen > 0
    assert first_score > 0
    # On the resumed run, neither generation nor scoring should re-fire.
    assert sum(len(c) for c in gen_calls) == 0, "responses were re-generated"
    assert score_calls == [], "scorer was re-invoked despite cache"
