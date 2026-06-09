"""Loading configs/xstest.yaml and configs/iheval.yaml via load_config."""

import textwrap
from pathlib import Path

from src.config.loader import load_config


def test_loads_xstest_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "xstest.yaml"
    cfg_path.write_text(textwrap.dedent("""
        xstest:
          data_csv: "data/external/xstest/xstest_prompts.csv"
          judge_model: "gpt-4o"
          judge_temperature: 0.0
          judge_max_tokens: 16
          generation_max_new_tokens: 512
          generation_temperature: 0.0
          generation_batch_size: 4
    """))
    cfg = load_config(cfg_path)
    assert cfg.xstest.judge_model == "gpt-4o"
    assert cfg.xstest.generation_batch_size == 4


def test_loads_iheval_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "iheval.yaml"
    cfg_path.write_text(textwrap.dedent("""
        iheval:
          benchmark_root: "vendor/iheval/benchmark"
          scorer_src_root: "vendor/iheval"
          generation_max_new_tokens: 2048
          generation_temperature: 0.0
          generation_batch_size: 2
          default_tasks:
            - "single-turn"
            - "multi-turn"
          default_settings:
            - "aligned"
            - "conflict"
            - "reference"
    """))
    cfg = load_config(cfg_path)
    assert cfg.iheval.generation_batch_size == 2
    assert cfg.iheval.default_tasks == ("single-turn", "multi-turn")
    assert cfg.iheval.default_settings == ("aligned", "conflict", "reference")


def test_xstest_override_via_cli(tmp_path: Path) -> None:
    cfg_path = tmp_path / "xstest.yaml"
    cfg_path.write_text(textwrap.dedent("""
        xstest:
          data_csv: "x"
          judge_model: "gpt-4o"
          judge_temperature: 0.0
          judge_max_tokens: 16
          generation_max_new_tokens: 512
          generation_temperature: 0.0
          generation_batch_size: 4
    """))
    cfg = load_config(cfg_path, overrides=["xstest.generation_batch_size=2"])
    assert cfg.xstest.generation_batch_size == 2
