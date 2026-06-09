"""Shared DPO curriculum training loop.

Encapsulates the 3-stage curriculum with sDPO reference updates used by
both ``bin/train_dpo.py`` (production) and ``bin/train_dpo_hp_search.py``
(HP sweep). The caller supplies the configuration, data, pre-built policy
and reference models, and an output directory; this module runs all stages,
handles best-checkpoint reloads between stages, performs sDPO reference
updates, and returns the final best-checkpoint path for the caller to merge.

This module deliberately does NOT perform the final LoRA merge — each
caller controls where the merged artifact lives.
"""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer
from trl import DPOConfig as TRLDPOConfig

from src.model import LlamaWithISE
from src.training.callbacks import BestCheckpointCallback, ISESaveCallback
from src.training.curriculum import build_curriculum_stages
from src.training.dpo_data_collator import DPOHierarchyCollator
from src.training.gw_dpo_trainer import GravityDPOTrainer
from src.training.merge import sync_peft_base_weights_to_plain

logger = logging.getLogger(__name__)


# Tri-state stage-status values returned by ``probe_stage_state``. Kept as
# bare strings rather than an Enum so callers in tests and other scripts
# don't need to import a class to compare results.
STAGE_COMPLETE = "complete"
STAGE_PARTIAL = "partial"
STAGE_EMPTY = "empty"


def probe_stage_state(
    stage_dir: Path,
    *,
    requires_ise: bool = True,
) -> tuple[str, Path | None]:
    """Classify a curriculum stage directory as complete / partial / empty.

    The HP-search and production drivers both need to decide whether a
    stage's ``best-checkpoint/`` is safe to skip-resume from, whether a
    crashed run can resume training mid-stage from the latest periodic
    checkpoint, or whether the stage must restart from scratch. Centralising
    that decision here keeps the policy in one place.

    Args:
        stage_dir: The ``stageN/`` directory to probe.
        requires_ise: Whether ``ise_weights.pt`` must be present for a
            checkpoint to count as resumable. Pass ``False`` for the (f)
            tokens-only ablation, which trains without ISE and therefore
            never writes that file. Defaults to ``True`` for backward
            compatibility with all existing ISE-on configurations.

    Returns one of:
    - ``("complete", <best-checkpoint path>)`` — ``trainer.train()`` ran to
      completion (the stage-level ``trainer_state.json`` was written, which
      only happens after ``trainer.train()`` and the end-of-stage
      ``trainer.evaluate()`` both return), and ``best-checkpoint/`` carries
      the artifacts needed to reload weights (LoRA adapter + ISE iff
      ``requires_ise``). The caller can skip training and reuse the best
      weights.
    - ``("partial", <checkpoint-N path>)`` — training did not finish, but
      the latest periodic ``checkpoint-N/`` carries a complete optimizer +
      scheduler + RNG state, so HF Trainer can resume from it. The caller
      should pass that path as ``resume_from_checkpoint`` to ``trainer.train``.
      Any ``best-checkpoint/`` in the stage may be from a partial run and
      its eval_loss must be used to seed ``BestCheckpointCallback`` so the
      callback does not regress to a worse weight set later in training.
    - ``("empty", None)`` — nothing reusable; train this stage from scratch.

    A "partial best-checkpoint without periodic resume state" case is
    treated as ``empty`` on purpose: the partial best-checkpoint is from
    an undertrained run and the script-level caller will delete the stage
    dir before retraining, otherwise that undertrained best-checkpoint
    would silently win against fresh checkpoints later.
    """
    if not stage_dir.exists() or not stage_dir.is_dir():
        return (STAGE_EMPTY, None)

    state_file = stage_dir / "trainer_state.json"
    best_dir = stage_dir / "best-checkpoint"

    def _has_ise(p: Path) -> bool:
        return (not requires_ise) or (p / "ise_weights.pt").exists()

    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
        max_steps = int(state.get("max_steps", 0) or 0)
        global_step = int(state.get("global_step", 0) or 0)
        if (
            max_steps > 0
            and global_step >= max_steps
            and best_dir.exists()
            and _has_ise(best_dir)
        ):
            return (STAGE_COMPLETE, best_dir)

    candidates: list[tuple[int, Path]] = []
    for cdir in stage_dir.glob("checkpoint-*"):
        if not cdir.is_dir():
            continue
        try:
            step = int(cdir.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (
            (cdir / "optimizer.pt").exists()
            and (cdir / "scheduler.pt").exists()
            and (cdir / "rng_state.pth").exists()
            and _has_ise(cdir)
        ):
            candidates.append((step, cdir))

    if candidates:
        candidates.sort(reverse=True)
        return (STAGE_PARTIAL, candidates[0][1])

    return (STAGE_EMPTY, None)


def _seed_best_callback_from_disk(
    callback: BestCheckpointCallback, stage_dir: Path,
) -> None:
    """Restore ``best_eval_loss`` from a prior run's ``best_info.json``.

    Without this, a partial-resume training run starts the callback at
    ``inf`` and the next eval (which may be worse than the prior best)
    would overwrite the existing best-checkpoint. Seeding from disk
    preserves the prior best so the callback only updates on genuine
    improvement.
    """
    info_path = stage_dir / "best-checkpoint" / "best_info.json"
    if not info_path.exists():
        return
    try:
        info = json.loads(info_path.read_text())
        prior = float(info["eval_loss"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return
    callback.best_eval_loss = prior
    logger.info(
        "Seeded BestCheckpointCallback with prior eval_loss=%.4f from %s",
        prior, info_path,
    )


def _reload_best_into_policy(
    policy_model: LlamaWithISE, best_dir: Path,
) -> None:
    """Reload best-of-stage LoRA + ISE weights into the policy model in place.

    Between curriculum stages, the policy typically reaches its lowest eval
    loss well before end-of-epoch and then overfits. Without this reload,
    stage N+1 would continue from the overfit end-of-stage weights and the
    sDPO reference update would propagate those weights too. Reloading
    from ``best-checkpoint/`` restores the lowest-eval-loss state so that
    both the policy continuation and the subsequent sDPO reference update
    start from the best-of-stage snapshot.

    PEFT's ``load_peft_weights`` / ``set_peft_model_state_dict`` handle
    the LoRA adapters (including ``trainable_token_indices``) correctly.
    ISE weights live outside the PEFT wrapper and must be reloaded
    explicitly from ``ise_weights.pt``.
    """
    adapter_state = load_peft_weights(str(best_dir))
    set_peft_model_state_dict(policy_model.model, adapter_state)

    if policy_model.ise is None:
        # Tokens-only ablation: no ISE weights to reload.
        logger.info(
            "Reloaded best-checkpoint LoRA from %s into policy (no ISE)",
            best_dir,
        )
        return

    ise_path = best_dir / "ise_weights.pt"
    if not ise_path.exists():
        msg = f"ISE weights missing at {ise_path}; cannot reload best checkpoint"
        raise FileNotFoundError(msg)
    policy_model.ise.load_state_dict(torch.load(ise_path, weights_only=True))
    logger.info("Reloaded best-checkpoint weights from %s into policy", best_dir)


def _sdpo_update(ref_model: LlamaWithISE, policy_model: LlamaWithISE) -> None:
    """sDPO: update reference model weights from policy (Kim et al., 2024).

    The policy is a PEFT-wrapped model whose state-dict keys are nested
    under ``base_model.model....base_layer.weight``; the reference is a
    plain transformers model whose keys are ``...weight``. A naive
    ``load_state_dict(..., strict=False)`` silently copies nothing.
    ``sync_peft_base_weights_to_plain`` merges the policy's adapters
    into its base-layer weights, remaps the keys, and loads them into
    the reference. ISE weights are then copied directly (matching keys).
    """
    sync_peft_base_weights_to_plain(policy_model.model, ref_model.model)
    if policy_model.ise is not None and ref_model.ise is not None:
        ref_model.ise.load_state_dict(policy_model.ise.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Updated reference model from policy (sDPO)")


def _resolve_final_best_checkpoint(final_stage_dir: Path) -> Path:
    """Resolve final-stage best-checkpoint with fallback to periodic checkpoint."""
    best_ckpt = final_stage_dir / "best-checkpoint"
    if best_ckpt.exists():
        return best_ckpt
    periodic = sorted(
        final_stage_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    if not periodic:
        msg = (
            f"No best-checkpoint or periodic checkpoints in {final_stage_dir}. "
            "This typically means the final stage had fewer optimizer steps "
            "than save_steps/eval_steps."
        )
        raise FileNotFoundError(msg)
    logger.warning(
        "No best-checkpoint in final stage; falling back to %s", periodic[0],
    )
    return periodic[0]


def run_dpo_curriculum(
    cfg,
    merged_dir: Path,
    tokenizer: AutoTokenizer,
    torch_dtype: torch.dtype,
    special_token_ids: list[int],
    train_dataset: Dataset,
    val_dataset: Dataset,
    run_dir: Path,
    policy_model: LlamaWithISE,
    ref_model: LlamaWithISE,
) -> Path:
    """Run the full DPO curriculum + sDPO against the provided datasets.

    Args:
        cfg: Resolved top-level config (provides ``cfg.dpo`` and
            ``cfg.model``).
        merged_dir: Directory of the merged SFT model (used by the caller;
            passed through in case future pieces need it).
        tokenizer: Tokenizer with hierarchy special tokens.
        torch_dtype: Torch dtype for training.
        special_token_ids: Special-token IDs for LoRA ``trainable_token_indices``.
        train_dataset: Training DPO pairs (typically full ``data/dpo/train``).
        val_dataset: Evaluation DPO pairs used for training-time ``eval_loss``
            and best-checkpoint selection.
        run_dir: Directory for per-stage subdirectories and checkpoints.
        policy_model: Policy model created by the caller (merged SFT + fresh
            LoRA + ISE).
        ref_model: Frozen reference model created by the caller.

    Returns:
        Path to the final-stage best-checkpoint (or periodic fallback) that
        the caller should use for the final merge.
    """
    del merged_dir, special_token_ids  # reserved for future hooks
    requires_ise = bool(getattr(cfg.model, "use_ise", True))
    stages = build_curriculum_stages(
        train_dataset,
        val_dataset,
        num_stages=cfg.dpo.num_curriculum_stages,
        enabled=cfg.dpo.curriculum_enabled,
        min_gap_by_stage=cfg.dpo.curriculum_min_gap_by_stage,
    )

    collator = DPOHierarchyCollator(
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        margin_schedule=cfg.dpo.margin_schedule,
    )

    for stage_idx, stage_data in enumerate(stages, 1):
        stage_dir = run_dir / f"stage{stage_idx}"

        # Resume policy: probe disk first so a stage that already finished
        # in a prior run is skipped, and a stage that crashed mid-training
        # resumes from its latest periodic checkpoint instead of restarting.
        # See ``probe_stage_state`` for the classification rules.
        stage_state, resume_path = probe_stage_state(
            stage_dir, requires_ise=requires_ise,
        )
        if stage_state == STAGE_COMPLETE:
            logger.info(
                "=== Curriculum Stage %d/%d: COMPLETE on disk, "
                "skipping training (resuming policy/ref state from %s) ===",
                stage_idx, len(stages), resume_path,
            )
            _reload_best_into_policy(policy_model, resume_path)
            if stage_idx < len(stages):
                logger.info(
                    "Updating reference model (sDPO) for stage %d...",
                    stage_idx + 1,
                )
                _sdpo_update(ref_model, policy_model)
            continue

        stage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "=== Curriculum Stage %d/%d: %d train, %d val ===",
            stage_idx, len(stages),
            len(stage_data["train"]), len(stage_data["val"]),
        )

        training_args = TRLDPOConfig(
            output_dir=str(stage_dir),
            beta=cfg.dpo.beta,
            loss_type=["sigmoid"],
            max_length=cfg.dpo.max_seq_length,
            learning_rate=cfg.dpo.learning_rate,
            lr_scheduler_type=cfg.dpo.lr_scheduler,
            warmup_ratio=cfg.dpo.warmup_ratio,
            num_train_epochs=cfg.dpo.epochs_per_stage,
            per_device_train_batch_size=cfg.dpo.per_device_batch_size,
            per_device_eval_batch_size=cfg.dpo.per_device_batch_size,
            gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
            weight_decay=cfg.dpo.weight_decay,
            bf16=(cfg.dpo.precision == "bf16"),
            fp16=(cfg.dpo.precision == "fp16"),
            eval_strategy="steps",
            eval_steps=cfg.dpo.eval_steps,
            save_strategy="steps",
            save_steps=cfg.dpo.save_steps,
            save_total_limit=cfg.dpo.save_total_limit,
            load_best_model_at_end=False,
            remove_unused_columns=cfg.dpo.remove_unused_columns,
            logging_steps=cfg.dpo.logging_steps,
            logging_dir=str(stage_dir / "logs"),
            report_to="none",
            gradient_checkpointing=True,
        )

        # ISESaveCallback only makes sense when ISE is enabled; the (f)
        # tokens-only ablation runs with policy_model.ise=None and skips it.
        callbacks: list = []
        if policy_model.ise is not None:
            callbacks.append(ISESaveCallback(ise=policy_model.ise))
        best_callback = BestCheckpointCallback(
            model=policy_model, tokenizer=tokenizer, run_dir=stage_dir,
        )
        callbacks.append(best_callback)

        if stage_state == STAGE_PARTIAL:
            # Reload ISE manually: HF Trainer's resume_from_checkpoint
            # restores PEFT adapters, optimizer, scheduler, and RNG state
            # from the checkpoint, but knows nothing about the ISE module
            # which lives on the LlamaWithISE wrapper outside PEFT. Skip
            # this for the tokens-only ablation since there's no ISE.
            if policy_model.ise is not None:
                ise_path = resume_path / "ise_weights.pt"
                policy_model.ise.load_state_dict(
                    torch.load(ise_path, weights_only=True),
                )
                logger.info(
                    "Reloaded ISE from %s for mid-training resume", ise_path,
                )
            # Preserve any prior best-checkpoint's eval_loss so a worse
            # post-resume eval doesn't overwrite it.
            _seed_best_callback_from_disk(best_callback, stage_dir)

        trainer = GravityDPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            data_collator=collator,
            train_dataset=stage_data["train"],
            eval_dataset=stage_data["val"],
            processing_class=tokenizer,
            gravity_alpha=cfg.dpo.gravity_alpha,
            callbacks=callbacks,
        )

        if stage_state == STAGE_PARTIAL:
            logger.info(
                "Resuming stage %d training from %s",
                stage_idx, resume_path,
            )
            trainer.train(resume_from_checkpoint=str(resume_path))
        else:
            logger.info("Starting stage %d training...", stage_idx)
            trainer.train()

        # End-of-stage eval: guarantees one eval on the final weights even
        # when max_steps does not land on an eval_steps boundary, so
        # BestCheckpointCallback can consider the final checkpoint.
        logger.info("Stage %d end-of-training evaluation...", stage_idx)
        trainer.evaluate()

        trainer.state.save_to_json(str(stage_dir / "trainer_state.json"))

        has_best = best_callback.best_eval_loss < float("inf")
        if has_best:
            logger.info(
                "Stage %d best: %s (eval_loss=%.4f)",
                stage_idx, best_callback.best_dir, best_callback.best_eval_loss,
            )
            _reload_best_into_policy(policy_model, best_callback.best_dir)
        else:
            logger.warning(
                "Stage %d produced no best-checkpoint (no eval fired). "
                "Continuing from end-of-stage policy weights.",
                stage_idx,
            )

        if stage_idx < len(stages):
            logger.info("Updating reference model (sDPO) for stage %d...", stage_idx + 1)
            _sdpo_update(ref_model, policy_model)

    final_stage_dir = run_dir / f"stage{len(stages)}"
    return _resolve_final_best_checkpoint(final_stage_dir)
