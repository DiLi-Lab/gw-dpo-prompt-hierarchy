"""Training hyperparameter configurations.

All values are read from configs/base_linear.yaml. No defaults are provided here —
missing values in the YAML will raise an error at instantiation time.

Sources: ISE (Wu et al., ICLR 2025), SecAlign (Chen et al., CCS 2025).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model and architecture configuration."""

    model_name_or_path: str
    torch_dtype: str
    num_segments: int
    token_embedding_init: str   # "mean" for special tokens (StruQ)
    ise_embedding_init: str     # "normal" for ISE segment layer (Wu et al.)
    ise_init_std: float         # std for ISE normal initialization
    use_ise: bool               # False = (f) tokens-only ablation (no ISE layer)


@dataclass
class SFTConfig:
    """SFT training hyperparameters (from ISE paper)."""

    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    num_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    weight_decay: float
    precision: str
    # LoRA
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    task_type: str
    # Checkpointing and evaluation
    save_steps: int
    eval_steps: int
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool
    remove_unused_columns: bool
    logging_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass
class DPOConfig:
    """DPO training hyperparameters (from SecAlign + ODPO)."""

    beta: float
    gravity_alpha: float
    margin_schedule: str
    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    num_curriculum_stages: int
    curriculum_enabled: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    weight_decay: float
    precision: str
    # LoRA
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    task_type: str
    epochs_per_stage: int
    # Checkpointing and evaluation
    save_steps: int
    eval_steps: int
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool
    remove_unused_columns: bool
    logging_steps: int
    # Data routing (additive — defaults preserve 5-level behaviour)
    train_split_name: str = "train"
    val_split_name: str = "val"
    # Optional override for the curriculum's per-stage min-gap schedule.
    # None preserves the 5-level default {1: 3, 2: 2}; ablation (e) sets {1: 2}.
    curriculum_min_gap_by_stage: dict[int, int] | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def final_stage_index(self) -> int:
        """1-indexed directory index of the final curriculum stage.

        When ``curriculum_enabled=False`` only one stage is ever trained
        (regardless of ``num_curriculum_stages``) and it is written to
        ``stageN/`` where N=1. Callers resolving the final-stage directory
        should consult this property rather than ``num_curriculum_stages``
        directly, so resume-checks and post-training reads stay aligned
        with what ``run_dpo_curriculum`` actually produced on disk.
        """
        return 1 if not self.curriculum_enabled else self.num_curriculum_stages


EVAL_CONFLICT_PAIRS: list[str] = [
    "L0_vs_L1", "L0_vs_L2", "L0_vs_L3", "L0_vs_L4",
    "L1_vs_L2", "L1_vs_L3", "L1_vs_L4",
    "L2_vs_L3", "L2_vs_L4",
    "L3_vs_L4",
]


@dataclass
class EvalConfig:
    """Configuration for evaluation suite construction."""

    count_per_pair: int
    num_pairs: int
    reference_per_pair: int
    near_dedup_threshold: float
    scenario_model: str
    scenario_temperature: float
    scenario_max_tokens: int
    gold_model: str
    gold_temperature: float
    control_model: str
    judge_model_1: str
    judge_model_2: str
    judge_min_score: int
    max_retries: int
    seed: int

    @property
    def total_conflicts(self) -> int:
        return self.count_per_pair * self.num_pairs

    @property
    def total_aligned(self) -> int:
        return self.total_conflicts

    @property
    def total_reference(self) -> int:
        return self.reference_per_pair * self.num_pairs


@dataclass
class EvaluationConfig:
    """Runtime configuration for the model evaluation pipeline.

    Distinct from ``EvalConfig`` which controls eval-suite *construction*.
    This dataclass controls how trained / baseline models are *evaluated*
    on the constructed suite.
    """

    generation_max_new_tokens: int
    generation_temperature: float
    generation_batch_size: int
    judge_model: str
    judge_temperature: float
    judge_max_tokens: int
    orr_min_response_chars_for_judge: int
    reward_batch_size: int
    bertscore_model: str
    run_text_similarity: bool
    run_rewards: bool


@dataclass
class XSTestConfig:
    """XSTest external-benchmark configuration."""

    data_csv: str
    judge_model: str
    judge_temperature: float
    judge_max_tokens: int
    generation_max_new_tokens: int
    generation_temperature: float
    generation_batch_size: int


@dataclass
class IHEvalConfig:
    """IHEval external-benchmark configuration."""

    benchmark_root: str
    scorer_src_root: str
    generation_max_new_tokens: int
    generation_temperature: float
    generation_batch_size: int
    default_tasks: tuple[str, ...]
    default_settings: tuple[str, ...]


@dataclass
class SEPConfig:
    """SEP external-benchmark configuration."""

    data_csv: str
    subsample_seed: int
    subsample_strata_field: str
    subsample_size: int
    generation_max_new_tokens: int
    generation_temperature: float
    generation_batch_size: int
    scoring_min_tokens: int
    scoring_refusal_patterns: tuple[str, ...]


@dataclass
class MTBenchConfig:
    """MT-Bench external-benchmark configuration."""

    question_jsonl: str
    reference_answer_jsonl: str
    judge_prompts_jsonl: str
    judge_model: str
    judge_temperature: float
    judge_temperature_retry: float
    judge_max_tokens: int
    generation_max_new_tokens: int
    generation_batch_size: int
    temperature_per_category: dict[str, float]


@dataclass
class TensorTrustConfig:
    """TensorTrust external-benchmark configuration."""

    hijacking_csv: str
    extraction_csv: str
    generation_max_new_tokens: int
    generation_temperature: float
    generation_batch_size: int
    rouge_recall_threshold: float
