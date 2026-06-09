"""Tests for EvaluationConfig wiring."""

import textwrap
from pathlib import Path

from src.config.loader import load_config
from src.config.paths import PathsConfig


def test_evaluation_config_loads_from_yaml(tmp_path: Path) -> None:
    cfg_text = textwrap.dedent(
        """
        model:
          model_name_or_path: "test"
          torch_dtype: "bfloat16"
          num_segments: 6
          token_embedding_init: "mean"
          ise_embedding_init: "normal"
          ise_init_std: 0.01
          use_ise: true
        sft:
          learning_rate: 2.0e-5
          lr_scheduler: cosine
          warmup_ratio: 0.03
          num_epochs: 3
          per_device_batch_size: 4
          gradient_accumulation_steps: 8
          max_seq_length: 4096
          weight_decay: 0.01
          precision: bf16
          lora_rank: 64
          lora_alpha: 128
          lora_dropout: 0.1
          lora_target_modules: ["q_proj"]
          task_type: CAUSAL_LM
          save_steps: 50
          eval_steps: 50
          save_total_limit: 5
          metric_for_best_model: eval_loss
          greater_is_better: false
          remove_unused_columns: false
          logging_steps: 1
        dpo:
          beta: 0.1
          gravity_alpha: 0.3
          margin_schedule: "gap"
          learning_rate: 5.0e-5
          lr_scheduler: cosine
          warmup_ratio: 0.03
          num_curriculum_stages: 3
          curriculum_enabled: true
          epochs_per_stage: 1
          per_device_batch_size: 1
          gradient_accumulation_steps: 32
          max_seq_length: 2048
          weight_decay: 0.01
          precision: bf16
          lora_rank: 64
          lora_alpha: 128
          lora_dropout: 0.1
          lora_target_modules: ["q_proj"]
          task_type: CAUSAL_LM
          save_steps: 30
          eval_steps: 30
          save_total_limit: 2
          metric_for_best_model: eval_loss
          greater_is_better: false
          remove_unused_columns: false
          logging_steps: 1
        eval:
          count_per_pair: 100
          num_pairs: 10
          reference_per_pair: 30
          near_dedup_threshold: 0.85
          scenario_model: gpt-4o
          scenario_temperature: 0.7
          scenario_max_tokens: 2000
          gold_model: claude-sonnet-4-20250514
          gold_temperature: 0.3
          control_model: gpt-4o-mini
          judge_model_1: gpt-4o
          judge_model_2: gemini-2.5-pro
          judge_min_score: 4
          max_retries: 2
          seed: 42
        evaluation:
          generation_max_new_tokens: 1024
          generation_temperature: 0.0
          generation_batch_size: 4
          judge_model: gpt-4o
          judge_temperature: 0.0
          judge_max_tokens: 800
          orr_min_response_chars_for_judge: 200
          reward_batch_size: 2
          bertscore_model: roberta-large
          run_text_similarity: true
          run_rewards: false
        paths:
          project_root: "."
        """
    )
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(config_path=cfg_path)
    assert cfg.evaluation.generation_max_new_tokens == 1024
    assert cfg.evaluation.judge_model == "gpt-4o"
    assert cfg.evaluation.run_rewards is False
    assert cfg.evaluation.bertscore_model == "roberta-large"


def test_evaluation_paths_properties() -> None:
    paths = PathsConfig(project_root=Path("/tmp/proj"))
    assert paths.evaluation_dir == Path("/tmp/proj/evaluation")
    assert paths.evaluation_runs_dir == Path("/tmp/proj/evaluation/runs")
