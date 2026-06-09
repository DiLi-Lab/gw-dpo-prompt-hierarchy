"""XSTest / IHEval / SEP / MT-Bench / TensorTrust external-benchmark config-surface tests."""

from pathlib import Path

import pytest

from src.config.hyperparameters import IHEvalConfig, TensorTrustConfig, XSTestConfig
from src.config.loader import load_config


def test_xstest_config_has_required_fields() -> None:
    cfg = XSTestConfig(
        data_csv="data/external/xstest/xstest_prompts.csv",
        judge_model="gpt-4o",
        judge_temperature=0.0,
        judge_max_tokens=16,
        generation_max_new_tokens=512,
        generation_temperature=0.0,
        generation_batch_size=4,
    )
    assert cfg.judge_model == "gpt-4o"
    assert cfg.generation_batch_size == 4


def test_xstest_config_rejects_unknown_field() -> None:
    with pytest.raises(TypeError):
        XSTestConfig(
            data_csv="x", judge_model="gpt-4o", judge_temperature=0.0,
            judge_max_tokens=16, generation_max_new_tokens=512,
            generation_temperature=0.0, generation_batch_size=4,
            unknown_field="boom",
        )


def test_iheval_config_has_required_fields() -> None:
    cfg = IHEvalConfig(
        benchmark_root="vendor/iheval/benchmark",
        scorer_src_root="vendor/iheval",
        generation_max_new_tokens=2048,
        generation_temperature=0.0,
        generation_batch_size=2,
        default_tasks=("single-turn", "multi-turn"),
        default_settings=("aligned", "conflict", "reference"),
    )
    assert cfg.generation_batch_size == 2
    assert "single-turn" in cfg.default_tasks
    assert cfg.default_settings == ("aligned", "conflict", "reference")


def test_load_sep_config_from_yaml() -> None:
    cfg = load_config(Path("configs/sep.yaml"))
    assert cfg.sep is not None
    assert cfg.sep.data_csv == "data/external/sep/sep_subsample.csv"
    assert cfg.sep.subsample_seed == 42
    assert cfg.sep.subsample_strata_field == "domain"
    assert cfg.sep.subsample_size == 1500
    assert cfg.sep.scoring_min_tokens == 10
    assert isinstance(cfg.sep.scoring_refusal_patterns, tuple)
    assert "i can't" in cfg.sep.scoring_refusal_patterns


def test_load_sep_config_override() -> None:
    cfg = load_config(
        Path("configs/sep.yaml"),
        overrides=["sep.scoring_min_tokens=20", "sep.subsample_seed=123"],
    )
    assert cfg.sep is not None
    assert cfg.sep.scoring_min_tokens == 20
    assert cfg.sep.subsample_seed == 123


def test_load_xstest_config_does_not_carry_sep() -> None:
    cfg = load_config(Path("configs/xstest.yaml"))
    assert cfg.sep is None


def test_load_mt_bench_config_from_yaml() -> None:
    cfg = load_config(Path("configs/mt_bench.yaml"))
    assert cfg.mt_bench is not None
    assert cfg.mt_bench.question_jsonl == "data/external/mt_bench/question.jsonl"
    assert cfg.mt_bench.judge_model == "gpt-4o"
    assert cfg.mt_bench.judge_temperature == 0.0
    assert cfg.mt_bench.judge_temperature_retry == 0.2
    assert cfg.mt_bench.generation_max_new_tokens == 1024
    assert cfg.mt_bench.generation_batch_size == 4
    assert isinstance(cfg.mt_bench.temperature_per_category, dict)
    assert cfg.mt_bench.temperature_per_category["writing"] == 0.7
    assert cfg.mt_bench.temperature_per_category["math"] == 0.0


def test_load_mt_bench_config_override() -> None:
    cfg = load_config(
        Path("configs/mt_bench.yaml"),
        overrides=[
            "mt_bench.judge_temperature=0.5",
            "mt_bench.generation_batch_size=2",
        ],
    )
    assert cfg.mt_bench is not None
    assert cfg.mt_bench.judge_temperature == 0.5
    assert cfg.mt_bench.generation_batch_size == 2


def test_load_xstest_config_does_not_carry_mt_bench() -> None:
    cfg = load_config(Path("configs/xstest.yaml"))
    assert cfg.mt_bench is None


def test_tensortrust_config_has_required_fields() -> None:
    cfg = TensorTrustConfig(
        hijacking_csv="data/external/tensortrust/hijacking_robustness.csv",
        extraction_csv="data/external/tensortrust/extraction_robustness.csv",
        generation_max_new_tokens=128,
        generation_temperature=0.0,
        generation_batch_size=4,
        rouge_recall_threshold=0.5,
    )
    assert cfg.generation_batch_size == 4
    assert cfg.rouge_recall_threshold == 0.5


def test_tensortrust_config_rejects_unknown_field() -> None:
    with pytest.raises(TypeError):
        TensorTrustConfig(
            hijacking_csv="x",
            extraction_csv="y",
            generation_max_new_tokens=128,
            generation_temperature=0.0,
            generation_batch_size=4,
            rouge_recall_threshold=0.5,
            unknown_field="boom",
        )


def test_load_tensortrust_config_from_yaml() -> None:
    cfg = load_config(Path("configs/tensortrust.yaml"))
    assert cfg.tensortrust is not None
    assert cfg.tensortrust.hijacking_csv == "data/external/tensortrust/hijacking_robustness.csv"
    assert cfg.tensortrust.extraction_csv == "data/external/tensortrust/extraction_robustness.csv"
    assert cfg.tensortrust.generation_max_new_tokens == 128
    assert cfg.tensortrust.generation_batch_size == 4
    assert cfg.tensortrust.rouge_recall_threshold == 0.5


def test_load_tensortrust_config_override() -> None:
    cfg = load_config(
        Path("configs/tensortrust.yaml"),
        overrides=[
            "tensortrust.rouge_recall_threshold=0.6",
            "tensortrust.generation_batch_size=2",
        ],
    )
    assert cfg.tensortrust is not None
    assert cfg.tensortrust.rouge_recall_threshold == 0.6
    assert cfg.tensortrust.generation_batch_size == 2


def test_load_xstest_config_does_not_carry_tensortrust() -> None:
    cfg = load_config(Path("configs/xstest.yaml"))
    assert cfg.tensortrust is None
