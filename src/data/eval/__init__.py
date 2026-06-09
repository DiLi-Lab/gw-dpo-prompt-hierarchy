"""Evaluation suite construction for 5-level instruction hierarchy."""

from src.data.eval.aligned_controls import build_aligned_control, run_phase3
from src.data.eval.build_eval_suite import (
    compute_eval_stats,
    load_eval_cache,
    run_phase6,
    save_eval_cache,
    validate_eval_instance,
)
from src.data.eval.conflict_scenarios import (
    CONFLICT_PAIRS,
    assemble_eval_instance,
    generate_conflict_scenario,
    generate_gold_response,
    run_phase1_and_2,
)
from src.data.eval.quality_control import (
    apply_eval_judge_decisions,
    build_eval_judge_prompt,
    run_phase5,
)
from src.data.eval.reference_baselines import (
    build_reference_baseline,
    run_phase4,
    strip_delimiters,
)

__all__ = [
    "CONFLICT_PAIRS",
    "apply_eval_judge_decisions",
    "assemble_eval_instance",
    "build_aligned_control",
    "build_eval_judge_prompt",
    "build_reference_baseline",
    "compute_eval_stats",
    "generate_conflict_scenario",
    "generate_gold_response",
    "load_eval_cache",
    "run_phase1_and_2",
    "run_phase3",
    "run_phase4",
    "run_phase5",
    "run_phase6",
    "save_eval_cache",
    "strip_delimiters",
    "validate_eval_instance",
]
