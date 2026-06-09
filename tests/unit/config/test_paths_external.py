"""External-benchmark paths derived from project_root."""

from pathlib import Path

from src.config.paths import PathsConfig


def test_external_dir_under_data() -> None:
    p = PathsConfig(project_root=Path("/tmp/pp"))
    assert p.external_dir == Path("/tmp/pp/data/external")


def test_xstest_csv_path() -> None:
    p = PathsConfig(project_root=Path("/tmp/pp"))
    assert p.xstest_csv == Path("/tmp/pp/data/external/xstest/xstest_prompts.csv")


def test_iheval_benchmark_root_under_vendor() -> None:
    p = PathsConfig(project_root=Path("/tmp/pp"))
    assert p.iheval_benchmark_root == Path("/tmp/pp/vendor/iheval/benchmark")


def test_iheval_scorer_src_root() -> None:
    p = PathsConfig(project_root=Path("/tmp/pp"))
    assert p.iheval_scorer_src_root == Path("/tmp/pp/vendor/iheval")


def test_external_runs_dir_under_evaluation() -> None:
    p = PathsConfig(project_root=Path("/tmp/pp"))
    assert p.external_runs_dir == Path("/tmp/pp/evaluation/external")
