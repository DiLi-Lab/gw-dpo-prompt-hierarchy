"""Tests for eval-specific path properties on PathsConfig."""

from pathlib import Path

from src.config.paths import PathsConfig


def test_eval_scenarios_raw():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_scenarios_raw == Path("/root/data/eval/eval_scenarios_raw.jsonl")


def test_eval_conflicts():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_conflicts == Path("/root/data/eval/eval_conflicts.jsonl")


def test_eval_aligned():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_aligned == Path("/root/data/eval/eval_aligned.jsonl")


def test_eval_aligned_raw():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_aligned_raw == Path("/root/data/eval/eval_aligned_raw.jsonl")


def test_eval_reference():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_reference == Path("/root/data/eval/eval_reference.jsonl")


def test_eval_qc_results():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_qc_results == Path("/root/data/eval/eval_qc_results.jsonl")


def test_eval_flagged():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_flagged == Path("/root/data/eval/eval_flagged.jsonl")


def test_eval_stats():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_stats == Path("/root/data/eval/eval_stats.json")


def test_eval_scenario_cache():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_scenario_cache == Path("/root/data/eval/scenario_cache.jsonl")


def test_eval_gold_cache():
    cfg = PathsConfig(project_root=Path("/root"))
    assert cfg.eval_gold_cache == Path("/root/data/eval/gold_cache.jsonl")


def test_all_eval_paths_rooted_under_eval_dir():
    cfg = PathsConfig(project_root=Path("/root"))
    eval_dir = cfg.eval_dir
    paths = [
        cfg.eval_scenarios_raw,
        cfg.eval_conflicts,
        cfg.eval_aligned,
        cfg.eval_aligned_raw,
        cfg.eval_reference,
        cfg.eval_qc_results,
        cfg.eval_flagged,
        cfg.eval_stats,
        cfg.eval_scenario_cache,
        cfg.eval_gold_cache,
    ]
    for p in paths:
        assert p.parent == eval_dir, f"{p} is not directly under eval_dir"
