"""Training callbacks for hierarchy-aware training."""

import json
import logging
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.model.ise import InstructionalSegmentEmbedding

logger = logging.getLogger(__name__)


class ISESaveCallback(TrainerCallback):
    """Save ISE weights alongside each PEFT checkpoint.

    The Trainer's on_save does not pass output_dir in kwargs, so we
    compute the checkpoint path from args.output_dir and state.global_step.

    Note: When using ISETrainer, ISE weights are already saved in _save().
    This callback serves as a safety net for other Trainer subclasses.

    Args:
        ise: The InstructionalSegmentEmbedding module to save.
    """

    def __init__(self, ise: InstructionalSegmentEmbedding) -> None:
        self.ise = ise

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        ise_path = checkpoint_dir / "ise_weights.pt"
        if not ise_path.exists():
            torch.save(self.ise.state_dict(), ise_path)
            logger.info("Saved ISE weights to %s", ise_path)


class BestCheckpointCallback(TrainerCallback):
    """Persist the best checkpoint (by eval_loss) to a fixed directory.

    Saves to ``<run_dir>/best-checkpoint/`` whenever eval_loss improves.
    The directory contains: LoRA adapter, trainable token embeddings,
    ISE weights, tokenizer, trainer state, and a ``best_info.json``
    for quick inspection.

    Args:
        model: The LlamaWithISE model (model.model is the PEFT model,
            model.ise is the ISE module).
        tokenizer: The tokenizer to save alongside the checkpoint.
        run_dir: The run output directory (e.g. ``runs/sft_20260414_172825``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        run_dir: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.best_dir = run_dir / "best-checkpoint"
        self.best_eval_loss = float("inf")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss >= self.best_eval_loss:
            return

        self.best_eval_loss = eval_loss
        self.best_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters + trainable token embeddings via PEFT
        self.model.model.save_pretrained(str(self.best_dir))
        # Save ISE weights only when present. The (f) tokens-only ablation
        # runs with model.ise=None — leaving ise_weights.pt absent here
        # propagates correctly through the eval loader, which treats a
        # missing ise_weights.pt as "no ISE wrap at inference time".
        if getattr(self.model, "ise", None) is not None:
            torch.save(
                self.model.ise.state_dict(),
                self.best_dir / "ise_weights.pt",
            )
        # Save tokenizer
        self.tokenizer.save_pretrained(str(self.best_dir))
        # Save trainer state (contains log_history up to this point)
        state.save_to_json(str(self.best_dir / "trainer_state.json"))
        # Save quick-lookup info
        best_info = {
            "step": state.global_step,
            "epoch": state.epoch,
            "eval_loss": eval_loss,
        }
        with open(self.best_dir / "best_info.json", "w") as f:
            json.dump(best_info, f, indent=2)

        logger.info(
            "New best eval_loss=%.4f at step %d, saved to %s",
            eval_loss,
            state.global_step,
            self.best_dir,
        )
