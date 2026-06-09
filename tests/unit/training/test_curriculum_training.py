"""Contract tests for run_dpo_curriculum.

These do NOT train a real model (that is covered by test_dpo_pipeline.py).
They check signature, argument contract, and the return tuple shape using
a stub-level assertion approach.
"""

import inspect
import json

from src.training.curriculum_training import (
    STAGE_COMPLETE,
    STAGE_EMPTY,
    STAGE_PARTIAL,
    probe_stage_state,
    run_dpo_curriculum,
)


def test_signature_is_stable():
    sig = inspect.signature(run_dpo_curriculum)
    params = list(sig.parameters.keys())
    for expected in [
        "cfg", "merged_dir", "tokenizer", "torch_dtype",
        "special_token_ids", "train_dataset", "val_dataset",
        "run_dir", "policy_model", "ref_model",
    ]:
        assert expected in params, expected


# ---------- probe_stage_state ----------

def _write_complete_stage(stage_dir, max_steps=10):
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "trainer_state.json").write_text(json.dumps({
        "global_step": max_steps, "max_steps": max_steps,
    }))
    best = stage_dir / "best-checkpoint"
    best.mkdir(parents=True, exist_ok=True)
    (best / "ise_weights.pt").write_bytes(b"x")
    return best


def _write_partial_checkpoint(stage_dir, step):
    cdir = stage_dir / f"checkpoint-{step}"
    cdir.mkdir(parents=True, exist_ok=True)
    for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "ise_weights.pt"):
        (cdir / f).write_bytes(b"x")
    return cdir


def test_probe_empty_when_dir_missing(tmp_path):
    state, p = probe_stage_state(tmp_path / "stage1")
    assert state == STAGE_EMPTY
    assert p is None


def test_probe_empty_when_dir_has_nothing(tmp_path):
    (tmp_path / "stage1").mkdir()
    state, p = probe_stage_state(tmp_path / "stage1")
    assert state == STAGE_EMPTY
    assert p is None


def test_probe_complete_requires_max_steps_reached(tmp_path):
    stage = tmp_path / "stage1"
    best = _write_complete_stage(stage, max_steps=10)
    state, p = probe_stage_state(stage)
    assert state == STAGE_COMPLETE
    assert p == best


def test_probe_partial_when_global_step_lt_max_steps(tmp_path):
    stage = tmp_path / "stage1"
    stage.mkdir()
    # Stage-level trainer_state.json claims partial training (90/273).
    (stage / "trainer_state.json").write_text(json.dumps({
        "global_step": 90, "max_steps": 273,
    }))
    cdir = _write_partial_checkpoint(stage, step=90)
    # Even an existing best-checkpoint must NOT count as complete here
    # (this is exactly the cfg02 silent-undertrain trap).
    best = stage / "best-checkpoint"
    best.mkdir()
    (best / "ise_weights.pt").write_bytes(b"x")
    state, p = probe_stage_state(stage)
    assert state == STAGE_PARTIAL
    assert p == cdir


def test_probe_partial_picks_latest_checkpoint(tmp_path):
    stage = tmp_path / "stage1"
    stage.mkdir()
    _write_partial_checkpoint(stage, step=30)
    latest = _write_partial_checkpoint(stage, step=120)
    _write_partial_checkpoint(stage, step=60)
    state, p = probe_stage_state(stage)
    assert state == STAGE_PARTIAL
    assert p == latest


def test_probe_partial_skips_checkpoints_missing_optimizer(tmp_path):
    stage = tmp_path / "stage1"
    stage.mkdir()
    cdir = stage / "checkpoint-30"
    cdir.mkdir()
    # Only adapter weights, no optimizer.pt → not resumable.
    (cdir / "ise_weights.pt").write_bytes(b"x")
    state, p = probe_stage_state(stage)
    assert state == STAGE_EMPTY
    assert p is None


def test_probe_complete_requires_ise_weights_in_best_checkpoint(tmp_path):
    stage = tmp_path / "stage1"
    stage.mkdir()
    (stage / "trainer_state.json").write_text(json.dumps({
        "global_step": 10, "max_steps": 10,
    }))
    best = stage / "best-checkpoint"
    best.mkdir()
    # Missing ise_weights.pt → not safely resumable as COMPLETE.
    state, _ = probe_stage_state(stage)
    assert state == STAGE_EMPTY


def test_probe_handles_corrupt_trainer_state(tmp_path):
    stage = tmp_path / "stage1"
    stage.mkdir()
    (stage / "trainer_state.json").write_text("not json {")
    cdir = _write_partial_checkpoint(stage, step=30)
    state, p = probe_stage_state(stage)
    # Falls back to partial via periodic checkpoint scan rather than crashing.
    assert state == STAGE_PARTIAL
    assert p == cdir
