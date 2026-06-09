"""End-to-end smoke test for the MT-Bench runner with mocked I/O."""

import json
from pathlib import Path

import pytest

from src.evaluation.external.mt_bench.runner import (
    JudgeCallFn,
    run_mt_bench_with_callables,
)

_FIXTURE_DIR = Path("tests/fixtures/mt_bench")
# 4-question fixture: writing, roleplay, math, coding
# question ids: 81 (writing), 91 (roleplay), 111 (math), 121 (coding)


def _make_generate_fn(canned_responses: dict[str, str]):
    """Returns a function returning the per-prompt response by lookup key."""

    def gen(prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            for key, resp in canned_responses.items():
                if key in p:
                    out.append(resp)
                    break
            else:
                out.append("DEFAULT_RESPONSE_TEXT_LONG_ENOUGH_TO_AVOID_GATE")
        return out

    return gen


def _stub_judge_returning(score: int | None) -> JudgeCallFn:
    text = f"Brief reasoning. [[{score}]]" if score is not None else "no score here"

    def judge_fn(*, system_prompt: str, user_prompt: str, temperature: float) -> str:
        return text

    return judge_fn


class _StubTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        rendered = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if kwargs.get("add_generation_prompt"):
            rendered += "<assistant>"
        return rendered


def test_smoke_writes_all_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    metrics = run_mt_bench_with_callables(
        questions_path=_FIXTURE_DIR / "question_subset.jsonl",
        references_path=_FIXTURE_DIR / "reference_answer_subset.jsonl",
        judge_prompts_path=_FIXTURE_DIR / "judge_prompts_subset.jsonl",
        output_dir=out_dir,
        tokenizer=_StubTokenizer(),
        generate_batch_fn_for_temperature=lambda t: _make_generate_fn({}),
        temperature_per_category={
            "writing": 0.7, "roleplay": 0.7, "math": 0.0, "coding": 0.0,
        },
        generation_batch_size=2,
        judge_fn=_stub_judge_returning(7),
        judge_temperature=0.0,
        judge_temperature_retry=0.2,
        run_metadata={"model": "fake", "format": "chat_template",
                      "ise_active": False, "judge_model": "fake-judge"},
    )

    assert (out_dir / "responses.jsonl").exists()
    assert (out_dir / "scoring.jsonl").exists()
    assert (out_dir / "metrics.json").exists()

    on_disk = json.loads((out_dir / "metrics.json").read_text())
    # 4 questions × 2 turns = 8 turns, all judged.
    assert on_disk["n_questions"] == 4
    assert on_disk["n_turns_total"] == 8
    assert on_disk["n_turns_scored"] == 8
    assert on_disk["overall_mean"] == 7.0
    assert on_disk["run_metadata"]["model"] == "fake"
    assert on_disk["run_metadata"]["judge_temperature"] == 0.0


def test_smoke_resumes_skipping_cached(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    gen_calls: list[int] = []
    judge_calls: list[int] = []

    def counting_gen_factory(temperature: float):
        def gen(prompts: list[str]) -> list[str]:
            gen_calls.append(len(prompts))
            return ["LONG ENOUGH RESPONSE TEXT FOR GATING " * 3 for _ in prompts]
        return gen

    def counting_judge(*, system_prompt: str, user_prompt: str, temperature: float) -> str:
        judge_calls.append(1)
        return "ok [[6]]"

    common_kwargs = dict(
        questions_path=_FIXTURE_DIR / "question_subset.jsonl",
        references_path=_FIXTURE_DIR / "reference_answer_subset.jsonl",
        judge_prompts_path=_FIXTURE_DIR / "judge_prompts_subset.jsonl",
        output_dir=out_dir,
        tokenizer=_StubTokenizer(),
        generate_batch_fn_for_temperature=counting_gen_factory,
        temperature_per_category={
            "writing": 0.7, "roleplay": 0.7, "math": 0.0, "coding": 0.0,
        },
        generation_batch_size=2,
        judge_fn=counting_judge,
        judge_temperature=0.0,
        judge_temperature_retry=0.2,
        run_metadata={"model": "fake", "format": "chat_template",
                      "ise_active": False, "judge_model": "fake-judge"},
    )

    run_mt_bench_with_callables(**common_kwargs)
    first_gen = sum(gen_calls)
    first_judge = sum(judge_calls)
    gen_calls.clear()
    judge_calls.clear()

    run_mt_bench_with_callables(**common_kwargs)
    # Second run hits both caches: generate and judge are not called.
    assert sum(gen_calls) == 0
    assert sum(judge_calls) == 0
    assert first_gen == 8       # 4 questions × 2 turns
    assert first_judge == 8     # 4 questions × 2 turns


def test_smoke_judge_retries_on_parse_failure(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    judge_calls: list[float] = []

    def flaky_judge(*, system_prompt: str, user_prompt: str, temperature: float) -> str:
        judge_calls.append(temperature)
        # First call (temp=0.0) returns unparseable text; retry (temp=0.2) returns score.
        if temperature == 0.0:
            return "no parseable score in here"
        return "second attempt [[5]]"

    metrics = run_mt_bench_with_callables(
        questions_path=_FIXTURE_DIR / "question_subset.jsonl",
        references_path=_FIXTURE_DIR / "reference_answer_subset.jsonl",
        judge_prompts_path=_FIXTURE_DIR / "judge_prompts_subset.jsonl",
        output_dir=out_dir,
        tokenizer=_StubTokenizer(),
        generate_batch_fn_for_temperature=lambda t: _make_generate_fn({}),
        temperature_per_category={
            "writing": 0.7, "roleplay": 0.7, "math": 0.0, "coding": 0.0,
        },
        generation_batch_size=2,
        judge_fn=flaky_judge,
        judge_temperature=0.0,
        judge_temperature_retry=0.2,
        run_metadata={"model": "fake", "format": "chat_template",
                      "ise_active": False, "judge_model": "fake-judge"},
    )
    # Each turn-judgment makes one initial call (parse-fail) + one retry → 2 calls.
    # 8 turns total → 16 judge calls.
    assert len(judge_calls) == 16
    # All retries return [[5]] so n_turns_scored should be 8, not 0.
    assert metrics["n_turns_scored"] == 8
    assert metrics["overall_mean"] == 5.0


def test_smoke_judge_retry_exhaustion_yields_parse_error(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"

    def always_fail(*, system_prompt: str, user_prompt: str, temperature: float) -> str:
        return "no score whatsoever"

    metrics = run_mt_bench_with_callables(
        questions_path=_FIXTURE_DIR / "question_subset.jsonl",
        references_path=_FIXTURE_DIR / "reference_answer_subset.jsonl",
        judge_prompts_path=_FIXTURE_DIR / "judge_prompts_subset.jsonl",
        output_dir=out_dir,
        tokenizer=_StubTokenizer(),
        generate_batch_fn_for_temperature=lambda t: _make_generate_fn({}),
        temperature_per_category={
            "writing": 0.7, "roleplay": 0.7, "math": 0.0, "coding": 0.0,
        },
        generation_batch_size=2,
        judge_fn=always_fail,
        judge_temperature=0.0,
        judge_temperature_retry=0.2,
        run_metadata={"model": "fake", "format": "chat_template",
                      "ise_active": False, "judge_model": "fake-judge"},
    )
    assert metrics["n_turns_scored"] == 0
    assert metrics["n_judge_parse_failures"] == 8
    assert metrics["overall_mean"] is None


def test_smoke_rejects_run_metadata_collisions(tmp_path: Path) -> None:
    """Caller cannot smuggle runtime-reserved keys via run_metadata."""
    out_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="runtime-reserved"):
        run_mt_bench_with_callables(
            questions_path=_FIXTURE_DIR / "question_subset.jsonl",
            references_path=_FIXTURE_DIR / "reference_answer_subset.jsonl",
            judge_prompts_path=_FIXTURE_DIR / "judge_prompts_subset.jsonl",
            output_dir=out_dir,
            tokenizer=_StubTokenizer(),
            generate_batch_fn_for_temperature=lambda t: _make_generate_fn({}),
            temperature_per_category={
                "writing": 0.7, "roleplay": 0.7, "math": 0.0, "coding": 0.0,
            },
            generation_batch_size=2,
            judge_fn=_stub_judge_returning(7),
            judge_temperature=0.0,
            judge_temperature_retry=0.2,
            run_metadata={
                "model": "fake", "format": "chat_template",
                "ise_active": False, "judge_model": "fake-judge",
                "timestamp": "ATTEMPT-TO-OVERRIDE",  # collision
            },
        )
