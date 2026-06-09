"""Tests for path configuration."""

from pathlib import Path

from src.config.paths import PathsConfig


def test_default_paths():
    paths = PathsConfig()
    assert paths.project_root == Path(".")
    assert paths.data_dir == Path("data")
    assert paths.splits_dir == Path("data/splits")
    assert paths.libraries_dir == Path("data/libraries")
    assert paths.models_dir == Path("models")
    assert paths.tokenizer_dir == Path("models/tokenizer-5level")
    assert paths.checkpoints_dir == Path("models/checkpoints")


def test_custom_project_root():
    paths = PathsConfig(project_root=Path("/tmp/my-project"))
    assert paths.splits_dir == Path("/tmp/my-project/data/splits")
    assert paths.tokenizer_dir == Path("/tmp/my-project/models/tokenizer-5level")


def test_split_paths():
    paths = PathsConfig()
    assert paths.alpaca_train == Path("data/splits/alpaca_train")
    assert paths.alpaca_eval == Path("data/splits/alpaca_eval")
    assert paths.dolly_train == Path("data/splits/dolly_train")
    assert paths.dolly_eval == Path("data/splits/dolly_eval")


def test_l0_rules_path():
    paths = PathsConfig()
    assert paths.l0_rules == paths.libraries_dir / "L0_rules.json"


def test_l1_library_path():
    paths = PathsConfig()
    assert paths.l1_library == paths.libraries_dir / "l1_library.json"


def test_l4_library_path():
    paths = PathsConfig()
    assert paths.l4_library == paths.libraries_dir / "l4_library.json"


def test_l4_synthesized_path():
    paths = PathsConfig()
    assert paths.l4_synthesized == paths.libraries_dir / "l4_synthesized.json"


def test_injection_templates_path():
    paths = PathsConfig()
    assert paths.injection_templates == paths.libraries_dir / "injection_templates.json"


def test_sft_combined_path():
    paths = PathsConfig()
    assert paths.sft_combined.name == "sft_combined.jsonl"
    assert paths.sft_combined.parent.name == "sft"
    assert paths.sft_combined == paths.sft_dir / "sft_combined.jsonl"


def test_sft_synthesis_cache_path():
    paths = PathsConfig()
    assert paths.sft_synthesis_cache.name == "synthesis_cache.jsonl"
    assert paths.sft_synthesis_cache.parent.name == "sft"
    assert paths.sft_synthesis_cache == paths.sft_dir / "synthesis_cache.jsonl"


def test_dpo_combined_path():
    paths = PathsConfig()
    assert paths.dpo_combined == paths.dpo_dir / "dpo_combined.jsonl"


def test_dpo_cache_paths():
    paths = PathsConfig()
    assert paths.dpo_yw_cache == paths.dpo_dir / "yw_cache.jsonl"
    assert paths.dpo_yl_cache == paths.dpo_dir / "yl_cache.jsonl"
    assert paths.dpo_l2_cache == paths.dpo_dir / "l2_dpo_cache.jsonl"
    assert paths.dpo_phase1 == paths.dpo_dir / "phase1_l1_vs_l3.jsonl"
    assert paths.dpo_phase2 == paths.dpo_dir / "phase2_gpt4o_mini.jsonl"
    assert paths.dpo_phase3_original == paths.dpo_dir / "phase3_claude_original.jsonl"
    assert paths.dpo_phase3 == paths.dpo_dir / "phase3_claude_fixed.jsonl"
    assert paths.dpo_qc_results == paths.dpo_dir / "qc_judge_results.jsonl"
    assert paths.dpo_flagged == paths.dpo_dir / "qc_flagged.jsonl"
    assert paths.dpo_stats == paths.dpo_dir / "dpo_stats.json"


def test_split_none_backward_compat():
    """With split=None, dirs resolve without subdirectory (backward compat)."""
    paths = PathsConfig()
    assert paths.sft_dir == Path("data/sft")
    assert paths.dpo_dir == Path("data/dpo")
    assert paths.stats_dir == Path("data/stats")


def test_split_train_dirs():
    """With split='train', dirs include train/ subdirectory."""
    paths = PathsConfig(split="train")
    assert paths.sft_dir == Path("data/sft/train")
    assert paths.dpo_dir == Path("data/dpo/train")
    assert paths.stats_dir == Path("data/stats/train")
    assert paths.sft_combined == Path("data/sft/train/sft_combined.jsonl")
    assert paths.dpo_combined == Path("data/dpo/train/dpo_combined.jsonl")
    assert paths.dpo_phase1 == Path("data/dpo/train/phase1_l1_vs_l3.jsonl")


def test_split_val_dirs():
    """With split='val', dirs include val/ subdirectory."""
    paths = PathsConfig(split="val")
    assert paths.sft_dir == Path("data/sft/val")
    assert paths.dpo_dir == Path("data/dpo/val")
    assert paths.stats_dir == Path("data/stats/val")
    assert paths.sft_combined == Path("data/sft/val/sft_combined.jsonl")
    assert paths.dpo_combined == Path("data/dpo/val/dpo_combined.jsonl")


def test_for_split():
    """for_split() returns a new PathsConfig with the requested split."""
    paths = PathsConfig(project_root=Path("/tmp/proj"), split="train")
    val_paths = paths.for_split("val")
    assert val_paths.split == "val"
    assert val_paths.project_root == Path("/tmp/proj")
    assert val_paths.sft_dir == Path("/tmp/proj/data/sft/val")
    # Original unchanged
    assert paths.split == "train"


def test_split_does_not_affect_libraries():
    """Split field must not affect library paths (they're shared)."""
    paths = PathsConfig(split="val")
    assert paths.libraries_dir == Path("data/libraries")
    assert paths.l0_rules == Path("data/libraries/L0_rules.json")
    assert paths.l1_library == Path("data/libraries/l1_library.json")


def test_split_does_not_affect_splits():
    """Split field must not affect base dataset split paths."""
    paths = PathsConfig(split="val")
    assert paths.splits_dir == Path("data/splits")
    assert paths.alpaca_train == Path("data/splits/alpaca_train")


def test_runs_dir():
    cfg = PathsConfig()
    assert cfg.runs_dir == Path("models/runs")


def test_sft_merged_dir():
    cfg = PathsConfig()
    assert cfg.sft_merged_dir == Path("models/llama-3.1-8b-sft-merged")
