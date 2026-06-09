"""Tests for training callbacks."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.model.ise import InstructionalSegmentEmbedding
from src.training.callbacks import BestCheckpointCallback, ISESaveCallback


def test_ise_save_callback_saves_weights(tmp_path):
    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=16)

    callback = ISESaveCallback(ise=ise)

    args = MagicMock()
    args.output_dir = str(tmp_path / "run")
    state = MagicMock()
    state.global_step = 100
    control = MagicMock()

    checkpoint_dir = tmp_path / "run" / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)

    callback.on_save(args, state, control, output_dir=str(checkpoint_dir))

    ise_path = checkpoint_dir / "ise_weights.pt"
    assert ise_path.exists()

    loaded = torch.load(ise_path, weights_only=True)
    assert "segment_embedding.weight" in loaded
    assert loaded["segment_embedding.weight"].shape == (6, 16)


def _make_best_checkpoint_callback(tmp_path):
    """Create a BestCheckpointCallback with a minimal mock model."""
    from unittest.mock import MagicMock, PropertyMock

    import torch.nn as nn

    from src.model.ise import InstructionalSegmentEmbedding

    ise = InstructionalSegmentEmbedding(num_segments=6, hidden_size=16)

    # Mock peft_model with save_pretrained
    peft_model = MagicMock()
    peft_model.save_pretrained = MagicMock()

    # Mock LlamaWithISE: model.model is the peft_model, model.ise is the ISE
    model = MagicMock()
    model.model = peft_model
    model.ise = ise

    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    callback = BestCheckpointCallback(
        model=model,
        tokenizer=tokenizer,
        run_dir=run_dir,
    )
    return callback, run_dir, peft_model, tokenizer


def test_best_checkpoint_saves_on_improvement(tmp_path):
    callback, run_dir, peft_model, tokenizer = _make_best_checkpoint_callback(tmp_path)

    args = MagicMock()
    state = MagicMock()
    state.global_step = 50
    state.epoch = 0.5
    control = MagicMock()
    metrics = {"eval_loss": 1.5}

    callback.on_evaluate(args, state, control, metrics=metrics)

    best_dir = run_dir / "best-checkpoint"
    assert best_dir.exists()
    assert (best_dir / "ise_weights.pt").exists()
    assert (best_dir / "best_info.json").exists()
    peft_model.save_pretrained.assert_called_once_with(str(best_dir))
    tokenizer.save_pretrained.assert_called_once_with(str(best_dir))

    info = json.loads((best_dir / "best_info.json").read_text())
    assert info["step"] == 50
    assert info["eval_loss"] == 1.5


def test_best_checkpoint_skips_when_not_improved(tmp_path):
    callback, run_dir, peft_model, tokenizer = _make_best_checkpoint_callback(tmp_path)

    args = MagicMock()
    state = MagicMock()
    control = MagicMock()

    # First eval: saves
    state.global_step = 50
    state.epoch = 0.5
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 1.5})
    assert peft_model.save_pretrained.call_count == 1

    # Second eval: worse loss, does NOT save
    state.global_step = 100
    state.epoch = 1.0
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 1.8})
    assert peft_model.save_pretrained.call_count == 1  # still 1


def test_best_checkpoint_overwrites_on_further_improvement(tmp_path):
    callback, run_dir, peft_model, tokenizer = _make_best_checkpoint_callback(tmp_path)

    args = MagicMock()
    state = MagicMock()
    control = MagicMock()

    # First eval
    state.global_step = 50
    state.epoch = 0.5
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 1.5})

    # Second eval: better
    state.global_step = 100
    state.epoch = 1.0
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 1.2})
    assert peft_model.save_pretrained.call_count == 2

    info = json.loads((run_dir / "best-checkpoint" / "best_info.json").read_text())
    assert info["step"] == 100
    assert info["eval_loss"] == 1.2
