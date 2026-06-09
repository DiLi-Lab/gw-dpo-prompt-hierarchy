"""Curriculum learning for DPO training.

Filters DPO datasets by hierarchy distance to implement easy-to-hard
curriculum following Curry-DPO (Pattnaik et al., 2024) and
Curriculum-DPO++ (Croitoru et al., CVPR 2025).

Default schedule (5-level hierarchy):
  1 (easy):   level_gap >= 3  (L0-vs-L3, L0-vs-L4, L1-vs-L4)
  2 (medium): level_gap >= 2  (adds L0-vs-L2, L1-vs-L3, L2-vs-L4)
  3 (hard):   all pairs       (adds all adjacent-level conflicts)

Ablation (e) overrides the schedule with ``min_gap_by_stage={1: 2}``
(2-stage 3-level curriculum: Sys-vs-Tool first, then add adjacent
pairs). Calibration examples (``is_calibration=True``) are included in
every stage to prevent over-refusal drift.
"""

import logging

from datasets import Dataset

logger = logging.getLogger(__name__)

DEFAULT_MIN_GAP_BY_STAGE: dict[int, int] = {
    1: 3,
    2: 2,
}


def filter_by_curriculum_stage(
    dataset: Dataset,
    stage: int,
    min_gap_by_stage: dict[int, int] | None = None,
) -> Dataset:
    """Filter a DPO dataset to include only examples for the given stage.

    Args:
        dataset: HuggingFace Dataset with ``level_gap`` and ``is_calibration`` columns.
        stage: Curriculum stage (1-indexed).
        min_gap_by_stage: Optional override of the per-stage minimum-gap
            threshold. Stages beyond the largest key fall through to "keep
            all" (the final-stage behaviour). When None, the 5-level default
            ``{1: 3, 2: 2}`` is used.

    Returns:
        Filtered dataset containing examples appropriate for this stage.

    Raises:
        ValueError: if ``stage < 1``.
    """
    if stage < 1:
        msg = "stage must be >= 1, got %d" % stage
        raise ValueError(msg)

    schedule = (
        DEFAULT_MIN_GAP_BY_STAGE if min_gap_by_stage is None else min_gap_by_stage
    )
    if stage not in schedule:
        # Stages beyond the schedule's largest key -> final-stage behaviour
        # (keep everything).
        return dataset

    min_gap = schedule[stage]

    def keep(example: dict) -> bool:
        return example["level_gap"] >= min_gap or example["is_calibration"]

    filtered = dataset.filter(keep)
    logger.info(
        "Curriculum stage %d: %d/%d examples (min_gap=%d)",
        stage, len(filtered), len(dataset), min_gap,
    )
    return filtered


def build_curriculum_stages(
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_stages: int = 3,
    enabled: bool = True,
    min_gap_by_stage: dict[int, int] | None = None,
) -> list[dict[str, Dataset]]:
    """Build curriculum stage splits for DPO training.

    Training data is filtered by stage; validation data is always unfiltered
    so that evaluation detects catastrophic forgetting of earlier stages.

    Args:
        train_dataset: Full DPO training dataset.
        val_dataset: Full DPO validation dataset.
        num_stages: Number of curriculum stages.
        enabled: When False, return a single unfiltered stage on the full
            training set (``num_stages`` is ignored).
        min_gap_by_stage: Optional override of the per-stage minimum-gap
            threshold. None preserves the 5-level default.

    Returns:
        List of dicts with ``train`` and ``val`` keys, one per stage. Length
        is 1 when ``enabled=False``, else ``num_stages``.
    """
    if not enabled:
        logger.info(
            "Curriculum disabled: single stage on full train set "
            "(%d train, %d val examples); num_stages=%d ignored.",
            len(train_dataset), len(val_dataset), num_stages,
        )
        return [{"train": train_dataset, "val": val_dataset}]

    stages = []
    for stage in range(1, num_stages + 1):
        train_filtered = filter_by_curriculum_stage(
            train_dataset, stage=stage, min_gap_by_stage=min_gap_by_stage,
        )
        stages.append({"train": train_filtered, "val": val_dataset})
        logger.info(
            "Stage %d: %d train examples, %d val examples",
            stage, len(train_filtered), len(val_dataset),
        )
    return stages
