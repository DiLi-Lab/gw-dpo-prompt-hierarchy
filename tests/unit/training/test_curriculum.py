"""Tests for DPO curriculum stage filtering."""

import pytest
from datasets import Dataset

from src.training.curriculum import build_curriculum_stages, filter_by_curriculum_stage


def _make_dpo_dataset(rows: list[dict]) -> Dataset:
    """Create a synthetic DPO dataset from row dicts."""
    defaults = {
        "prompt": "<|L0_START|>rule<|L0_END|>",
        "chosen": "<|RESP_START|>good<|RESP_END|>",
        "rejected": "<|RESP_START|>bad<|RESP_END|>",
        "conflict_type": "L0_vs_L4",
        "margin": 0.0,
        "category": "pairwise",
        "is_calibration": False,
        "victim_level": 0,
        "attacker_level": 4,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return Dataset.from_list(full_rows)


def _dataset_with_all_gaps() -> Dataset:
    """Dataset with examples at every gap (1-4), plus calibration."""
    rows = [
        {"level_gap": 4, "conflict_type": "L0_vs_L4"},
        {"level_gap": 3, "conflict_type": "L0_vs_L3"},
        {"level_gap": 3, "conflict_type": "L1_vs_L4"},
        {"level_gap": 2, "conflict_type": "L0_vs_L2"},
        {"level_gap": 2, "conflict_type": "L1_vs_L3"},
        {"level_gap": 2, "conflict_type": "L2_vs_L4"},
        {"level_gap": 1, "conflict_type": "L0_vs_L1"},
        {"level_gap": 1, "conflict_type": "L1_vs_L2"},
        {"level_gap": 1, "conflict_type": "L2_vs_L3"},
        {"level_gap": 1, "conflict_type": "L3_vs_L4"},
        {"level_gap": 0, "is_calibration": True, "category": "calibration",
         "conflict_type": "calibration"},
        {"level_gap": 0, "is_calibration": True, "category": "calibration",
         "conflict_type": "calibration"},
    ]
    return _make_dpo_dataset(rows)


class TestFilterByCurriculumStage:
    """Tests for filter_by_curriculum_stage."""

    def test_stage1_keeps_gap_ge_3_and_calibration(self):
        ds = _dataset_with_all_gaps()
        filtered = filter_by_curriculum_stage(ds, stage=1)
        gaps = filtered["level_gap"]
        # Should keep: gap 4 (1), gap 3 (2), calibration (2) = 5 examples
        assert len(filtered) == 5
        for gap, is_cal in zip(gaps, filtered["is_calibration"]):
            assert gap >= 3 or is_cal

    def test_stage2_keeps_gap_ge_2_and_calibration(self):
        ds = _dataset_with_all_gaps()
        filtered = filter_by_curriculum_stage(ds, stage=2)
        gaps = filtered["level_gap"]
        # gap 4 (1) + gap 3 (2) + gap 2 (3) + calibration (2) = 8
        assert len(filtered) == 8
        for gap, is_cal in zip(gaps, filtered["is_calibration"]):
            assert gap >= 2 or is_cal

    def test_stage3_keeps_all(self):
        ds = _dataset_with_all_gaps()
        filtered = filter_by_curriculum_stage(ds, stage=3)
        assert len(filtered) == len(ds)

    def test_calibration_included_in_all_stages(self):
        ds = _dataset_with_all_gaps()
        for stage in [1, 2, 3]:
            filtered = filter_by_curriculum_stage(ds, stage=stage)
            cal_count = sum(1 for c in filtered["is_calibration"] if c)
            assert cal_count == 2, (
                "Expected 2 calibration examples in stage %d, got %d" % (stage, cal_count)
            )

    def test_stages_are_cumulative(self):
        ds = _dataset_with_all_gaps()
        s1 = filter_by_curriculum_stage(ds, stage=1)
        s2 = filter_by_curriculum_stage(ds, stage=2)
        s3 = filter_by_curriculum_stage(ds, stage=3)
        assert len(s1) <= len(s2) <= len(s3)

    def test_invalid_stage_raises(self):
        ds = _dataset_with_all_gaps()
        with pytest.raises(ValueError):
            filter_by_curriculum_stage(ds, stage=0)
        with pytest.raises(ValueError):
            filter_by_curriculum_stage(ds, stage=-1)

    def test_stage_above_default_schedule_keeps_all(self):
        """Stages beyond the default schedule fall through to 'keep all'."""
        ds = _dataset_with_all_gaps()
        # Default schedule has entries for stages 1 and 2 only; stage>=3 -> keep all.
        s4 = filter_by_curriculum_stage(ds, stage=4)
        assert len(s4) == len(ds)

    def test_empty_dataset(self):
        ds = _make_dpo_dataset([])
        filtered = filter_by_curriculum_stage(ds, stage=1)
        assert len(filtered) == 0


class TestBuildCurriculumStages:
    """Tests for build_curriculum_stages."""

    def test_returns_correct_number_of_stages(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        assert len(stages) == 3

    def test_val_includes_all_types(self):
        """Validation set is unfiltered to detect catastrophic forgetting."""
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        for stage_data in stages:
            assert len(stage_data["val"]) == len(ds)

    def test_train_subsets_grow_with_stages(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        train_sizes = [len(s["train"]) for s in stages]
        assert train_sizes[0] <= train_sizes[1] <= train_sizes[2]

    def test_stage_dicts_have_train_and_val(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        for s in stages:
            assert "train" in s
            assert "val" in s


class TestCurriculumDisabled:
    """Tests for build_curriculum_stages(enabled=False).

    When the curriculum is disabled, HP search and any ablation runs should
    train a single stage on the full unfiltered train set. This bypasses the
    easy→hard filtering entirely while preserving the train/val dict shape
    expected by run_dpo_curriculum.
    """

    def test_returns_single_stage(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3, enabled=False)
        assert len(stages) == 1

    def test_train_is_unfiltered_full_dataset(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3, enabled=False)
        assert len(stages[0]["train"]) == len(ds)
        # Every level_gap bucket (including hard gap=1) is present.
        train_gaps = set(stages[0]["train"]["level_gap"])
        assert train_gaps == {0, 1, 2, 3, 4}

    def test_val_is_unfiltered(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3, enabled=False)
        assert len(stages[0]["val"]) == len(ds)

    def test_single_stage_ignores_num_stages_value(self):
        """num_stages is irrelevant when enabled=False."""
        ds = _dataset_with_all_gaps()
        for num_stages in (1, 2, 3):
            stages = build_curriculum_stages(
                ds, ds, num_stages=num_stages, enabled=False,
            )
            assert len(stages) == 1

    def test_stage_dict_has_train_and_val(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3, enabled=False)
        assert "train" in stages[0]
        assert "val" in stages[0]

    def test_enabled_default_true_preserves_curriculum(self):
        """Omitting enabled must yield the 3-stage curriculum (backcompat)."""
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        assert len(stages) == 3


class TestConfigurableMinGap:
    """Tests for the optional min_gap_by_stage parameter (ablation (e))."""

    def test_default_preserves_5level_schedule(self):
        ds = _dataset_with_all_gaps()
        s1 = filter_by_curriculum_stage(ds, stage=1)
        # Default is {1: 3, 2: 2}: stage 1 keeps gap>=3 + calibration -> 5
        assert len(s1) == 5

    def test_3level_schedule_via_kwarg(self):
        """min_gap_by_stage={1: 2} for 2-stage 3-level curriculum."""
        ds = _dataset_with_all_gaps()
        # Stage 1: gap>=2 + calibration. From the fixture: gap 4 (1) + gap 3 (2)
        # + gap 2 (3) + calibration (2) = 8.
        s1 = filter_by_curriculum_stage(ds, stage=1, min_gap_by_stage={1: 2})
        assert len(s1) == 8
        # Stage 2: max stage in the schedule -> all kept.
        s2 = filter_by_curriculum_stage(ds, stage=2, min_gap_by_stage={1: 2})
        assert len(s2) == len(ds)

    def test_build_stages_2stage(self):
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(
            ds, ds, num_stages=2, min_gap_by_stage={1: 2},
        )
        assert len(stages) == 2
        assert len(stages[0]["train"]) <= len(stages[1]["train"])

    def test_build_stages_default_unchanged(self):
        """Omitting min_gap_by_stage reproduces the existing 3-stage behaviour."""
        ds = _dataset_with_all_gaps()
        stages = build_curriculum_stages(ds, ds, num_stages=3)
        assert len(stages) == 3
        # Stage 1 is the {1: 3} default: gap>=3 + calibration = 5.
        assert len(stages[0]["train"]) == 5

    def test_stage_above_schedule_keeps_all(self):
        """Stages beyond the largest configured stage keep the full dataset."""
        ds = _dataset_with_all_gaps()
        # Schedule has stage 1 only -> stage 2 falls through to "keep all".
        s2 = filter_by_curriculum_stage(ds, stage=2, min_gap_by_stage={1: 2})
        assert len(s2) == len(ds)
