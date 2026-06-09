"""Training pipeline for the 5-level instruction hierarchy.

Provides data collation with segment IDs, LoRA configuration,
ISE checkpoint saving, DPO training with gravity-weighted margins,
curriculum learning, and post-training model merging.
"""

from src.training.callbacks import BestCheckpointCallback, ISESaveCallback
from src.training.curriculum import build_curriculum_stages, filter_by_curriculum_stage
from src.training.curriculum_training import (
    STAGE_COMPLETE,
    STAGE_EMPTY,
    STAGE_PARTIAL,
    probe_stage_state,
    run_dpo_curriculum,
)
from src.training.data_collator import HierarchyDataCollator
from src.training.dpo_data_collator import DPOHierarchyCollator
from src.training.gw_dpo_trainer import GravityDPOTrainer
from src.training.hp_search_eval import (
    compute_reward_accuracy_metrics,
    evaluate_reward_accuracies,
)
from src.training.lora_config import build_lora_config
from src.training.merge import (
    merge_lora_adapter,
    save_merged_model_with_ise,
    sync_peft_base_weights_to_plain,
)

__all__ = [
    "BestCheckpointCallback",
    "DPOHierarchyCollator",
    "GravityDPOTrainer",
    "HierarchyDataCollator",
    "ISESaveCallback",
    "STAGE_COMPLETE",
    "STAGE_EMPTY",
    "STAGE_PARTIAL",
    "build_curriculum_stages",
    "build_lora_config",
    "compute_reward_accuracy_metrics",
    "evaluate_reward_accuracies",
    "filter_by_curriculum_stage",
    "merge_lora_adapter",
    "probe_stage_state",
    "run_dpo_curriculum",
    "save_merged_model_with_ise",
    "sync_peft_base_weights_to_plain",
]
