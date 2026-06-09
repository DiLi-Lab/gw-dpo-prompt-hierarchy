"""Path configuration for all project artifacts.

All paths are derived from a single project_root, ensuring consistency
across local development and remote cluster execution.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    """Canonical paths for data, models, and checkpoints."""

    project_root: Path = field(default_factory=lambda: Path("."))
    split: str | None = None

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def libraries_dir(self) -> Path:
        return self.data_dir / "libraries"

    @property
    def l0_rules(self) -> Path:
        return self.libraries_dir / "L0_rules.json"

    @property
    def l0_seed_rules(self) -> Path:
        return self.libraries_dir / "L0_seed_rules.json"

    @property
    def l0_rules_expanded(self) -> Path:
        return self.libraries_dir / "L0_rules_expanded.json"

    @property
    def l1_library(self) -> Path:
        return self.libraries_dir / "l1_library.json"

    @property
    def l4_library(self) -> Path:
        return self.libraries_dir / "l4_library.json"

    @property
    def l4_synthesized(self) -> Path:
        return self.libraries_dir / "l4_synthesized.json"

    @property
    def injection_templates(self) -> Path:
        return self.libraries_dir / "injection_templates.json"

    @property
    def l0_conflict_scenarios(self) -> Path:
        return self.libraries_dir / "l0_conflict_scenarios.json"

    @property
    def cascading_families_generated(self) -> Path:
        return self.libraries_dir / "cascading_families_generated.json"

    @property
    def l0_adversarial_instructions(self) -> Path:
        return self.libraries_dir / "l0_adversarial_instructions.json"

    @property
    def stats_dir(self) -> Path:
        base = self.data_dir / "stats"
        return base / self.split if self.split else base

    @property
    def sft_dir(self) -> Path:
        base = self.data_dir / "sft"
        return base / self.split if self.split else base

    @property
    def sft_combined(self) -> Path:
        return self.sft_dir / "sft_combined.jsonl"

    @property
    def sft_synthesis_cache(self) -> Path:
        return self.sft_dir / "synthesis_cache.jsonl"

    @property
    def dpo_dir(self) -> Path:
        base = self.data_dir / "dpo"
        return base / self.split if self.split else base

    @property
    def dpo_combined(self) -> Path:
        return self.dpo_dir / "dpo_combined.jsonl"

    @property
    def dpo_yw_cache(self) -> Path:
        return self.dpo_dir / "yw_cache.jsonl"

    @property
    def dpo_yl_cache(self) -> Path:
        return self.dpo_dir / "yl_cache.jsonl"

    @property
    def dpo_l2_cache(self) -> Path:
        return self.dpo_dir / "l2_dpo_cache.jsonl"

    @property
    def dpo_phase1(self) -> Path:
        return self.dpo_dir / "phase1_l1_vs_l3.jsonl"

    @property
    def dpo_phase2_original(self) -> Path:
        return self.dpo_dir / "phase2_gpt4o_mini_original.jsonl"

    @property
    def dpo_phase2(self) -> Path:
        return self.dpo_dir / "phase2_gpt4o_mini.jsonl"

    @property
    def dpo_phase3_original(self) -> Path:
        return self.dpo_dir / "phase3_claude_original.jsonl"

    @property
    def dpo_phase3(self) -> Path:
        return self.dpo_dir / "phase3_claude_fixed.jsonl"

    @property
    def dpo_qc_results(self) -> Path:
        return self.dpo_dir / "qc_judge_results.jsonl"

    @property
    def dpo_flagged(self) -> Path:
        return self.dpo_dir / "qc_flagged.jsonl"

    @property
    def dpo_stats(self) -> Path:
        return self.dpo_dir / "dpo_stats.json"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def eval_scenarios_raw(self) -> Path:
        return self.eval_dir / "eval_scenarios_raw.jsonl"

    @property
    def eval_conflicts(self) -> Path:
        return self.eval_dir / "eval_conflicts.jsonl"

    @property
    def eval_aligned(self) -> Path:
        return self.eval_dir / "eval_aligned.jsonl"

    @property
    def eval_aligned_raw(self) -> Path:
        return self.eval_dir / "eval_aligned_raw.jsonl"

    @property
    def eval_reference(self) -> Path:
        return self.eval_dir / "eval_reference.jsonl"

    @property
    def eval_qc_results(self) -> Path:
        return self.eval_dir / "eval_qc_results.jsonl"

    @property
    def eval_flagged(self) -> Path:
        return self.eval_dir / "eval_flagged.jsonl"

    @property
    def eval_stats(self) -> Path:
        return self.eval_dir / "eval_stats.json"

    @property
    def eval_scenario_cache(self) -> Path:
        return self.eval_dir / "scenario_cache.jsonl"

    @property
    def eval_gold_cache(self) -> Path:
        return self.eval_dir / "gold_cache.jsonl"

    @property
    def evaluation_dir(self) -> Path:
        return self.project_root / "evaluation"

    @property
    def evaluation_runs_dir(self) -> Path:
        return self.evaluation_dir / "runs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def tokenizer_dir(self) -> Path:
        return self.models_dir / "tokenizer-5level"

    @property
    def checkpoints_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def runs_dir(self) -> Path:
        return self.models_dir / "runs"

    @property
    def sft_merged_dir(self) -> Path:
        return self.models_dir / "llama-3.1-8b-sft-merged"

    @property
    def dpo_final_dir(self) -> Path:
        return self.models_dir / "llama-3.1-8b-gw-dpo-final"

    @property
    def hp_search_dir(self) -> Path:
        """Root directory for all DPO hyperparameter-search artifacts."""
        return self.models_dir / "hp_search"

    @property
    def hp_search_data_dir(self) -> Path:
        return self.hp_search_dir / "data"

    @property
    def hp_search_runs_dir(self) -> Path:
        return self.hp_search_dir / "runs"

    @property
    def alpaca_train(self) -> Path:
        return self.splits_dir / "alpaca_train"

    @property
    def alpaca_eval(self) -> Path:
        return self.splits_dir / "alpaca_eval"

    @property
    def dolly_train(self) -> Path:
        return self.splits_dir / "dolly_train"

    @property
    def dolly_eval(self) -> Path:
        return self.splits_dir / "dolly_eval"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def xstest_csv(self) -> Path:
        return self.external_dir / "xstest" / "xstest_prompts.csv"

    @property
    def iheval_benchmark_root(self) -> Path:
        return self.project_root / "vendor" / "iheval" / "benchmark"

    @property
    def iheval_scorer_src_root(self) -> Path:
        return self.project_root / "vendor" / "iheval"

    @property
    def sep_dir(self) -> Path:
        return self.external_dir / "sep"

    @property
    def sep_subsample_csv(self) -> Path:
        return self.sep_dir / "sep_subsample.csv"

    @property
    def tensortrust_dir(self) -> Path:
        return self.external_dir / "tensortrust"

    @property
    def tensortrust_hijacking_csv(self) -> Path:
        return self.tensortrust_dir / "hijacking_robustness.csv"

    @property
    def tensortrust_extraction_csv(self) -> Path:
        return self.tensortrust_dir / "extraction_robustness.csv"

    @property
    def mt_bench_dir(self) -> Path:
        return self.external_dir / "mt_bench"

    @property
    def mt_bench_question_jsonl(self) -> Path:
        return self.mt_bench_dir / "question.jsonl"

    @property
    def mt_bench_reference_answer_jsonl(self) -> Path:
        return self.mt_bench_dir / "reference_answer_gpt4.jsonl"

    @property
    def mt_bench_judge_prompts_jsonl(self) -> Path:
        return self.mt_bench_dir / "judge_prompts.jsonl"

    @property
    def external_runs_dir(self) -> Path:
        return self.evaluation_dir / "external"

    def for_split(self, split: str) -> "PathsConfig":
        """Return a new PathsConfig with the requested split."""
        return PathsConfig(project_root=self.project_root, split=split)
