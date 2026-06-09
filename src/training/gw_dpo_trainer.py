"""Gravity-Weighted DPO trainer for instruction hierarchy enforcement.

Subclasses TRL's DPOTrainer to add:
1. Per-sample gravity-weighted margins in the DPO loss.
2. ISE segment_ids passed to both policy and reference model forward calls.
3. ISE weight saving alongside LoRA checkpoints.

The loss is: L = -logsigmoid(beta * delta_score - alpha * margin)
where margin = gravity_alpha * level_gap for each sample.

Designed for TRL 0.29.0. The _compute_loss override is version-specific.
Note: loss_type is hardcoded to sigmoid; the TRLDPOConfig loss_type
parameter is ignored by this override.
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import selective_log_softmax

from src.model.llama_with_ise import LlamaWithISE

logger = logging.getLogger(__name__)


def compute_gw_dpo_loss(
    beta: float,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    margins: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Gravity-Weighted DPO loss.

    Args:
        beta: DPO inverse temperature.
        chosen_logps: Log-probs of chosen responses under policy.
        rejected_logps: Log-probs of rejected responses under policy.
        ref_chosen_logps: Log-probs of chosen responses under reference.
        ref_rejected_logps: Log-probs of rejected responses under reference.
        margins: Per-sample margins (already scaled by gravity_alpha).
            None or all-zeros gives standard DPO.

    Returns:
        Per-sample loss tensor (not yet averaged).
    """
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)

    if margins is not None:
        logits = logits - margins.to(logits.device)

    return -F.logsigmoid(logits)


class GravityDPOTrainer(DPOTrainer):
    """DPO trainer with gravity-weighted per-sample margins and ISE support.

    Extends TRL's DPOTrainer to:
    - Subtract ``gravity_alpha * margin`` from the DPO logits before
      the sigmoid loss, enforcing larger reward gaps for larger hierarchy
      distance violations.
    - Pass ``segment_ids`` through to model forward calls so that both
      the policy and reference model receive ISE segment embeddings.
    - Save ISE weights (``ise_weights.pt``) alongside LoRA checkpoints.

    Note: This override hardcodes the sigmoid loss type. The ``loss_type``
    parameter in TRLDPOConfig is ignored. The ``ld_alpha`` length-differential
    normalization is also not supported.

    Args:
        gravity_alpha: Scaling coefficient for the hierarchy-distance margin.
            Each sample's raw margin (level_gap) is multiplied by this value.
        All other args are passed to TRL's DPOTrainer.
    """

    def __init__(self, *args, gravity_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gravity_alpha = gravity_alpha

    def _compute_loss(self, model, inputs, return_outputs):
        """Compute GW-DPO loss with segment_ids and margin support.

        Overrides TRL 0.29.0's _compute_loss to:
        1. Inject segment_ids into model_kwargs for ISE.
        2. Apply gravity-weighted margin to delta_score before loss.
        3. Log DPO-specific metrics (rewards, logps, accuracy).
        """
        mode = "train" if model.training else "eval"

        # --- Extract custom fields from inputs ---
        margins = None
        if "margin" in inputs:
            margins = inputs.pop("margin").float() * self.gravity_alpha

        segment_ids = inputs.pop("segment_ids", None)

        # --- Standard TRL processing: extract and truncate ---
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(
            input_ids, attention_mask, completion_mask,
        )

        # Truncate segment_ids to match
        if segment_ids is not None:
            segment_ids = segment_ids[:, :input_ids.shape[1]]

        # --- Build model_kwargs with segment_ids ---
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if segment_ids is not None:
            model_kwargs["segment_ids"] = segment_ids

        # Pass through multimodal keys if present
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw",
                     "image_sizes", "token_type_ids"):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        # --- Policy model forward ---
        outputs = model(**model_kwargs)

        # --- Compute per-token log probabilities ---
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0

        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # --- Reference model log probabilities ---
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(
                ref_shift_logits, shift_labels,
            )
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # --- Gravity-Weighted DPO loss ---
        per_sample_loss = compute_gw_dpo_loss(
            beta=self.beta,
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            margins=margins,
        )
        loss = per_sample_loss.mean()

        # --- Compute reward metrics ---
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # --- Log metrics (matching TRL's _metrics pattern) ---
        agg_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        agg_reward_accuracies = self.accelerator.gather_for_metrics(reward_accuracies)

        self._metrics[mode]["rewards/chosen"].append(
            agg_chosen_rewards.mean().item()
        )
        self._metrics[mode]["rewards/rejected"].append(
            agg_rejected_rewards.mean().item()
        )
        self._metrics[mode]["rewards/accuracies"].append(
            agg_reward_accuracies.mean().item()
        )
        self._metrics[mode]["rewards/margins"].append(
            (agg_chosen_rewards - agg_rejected_rewards).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather(chosen_logps).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather(rejected_logps).mean().item()
        )

        if return_outputs:
            return loss, {
                "chosen_logps": chosen_logps.detach(),
                "rejected_logps": rejected_logps.detach(),
            }
        return loss

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict | None = None,
    ) -> None:
        """Save LoRA adapters + ISE weights alongside each checkpoint."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapters + trainable token embeddings via PEFT
        model = self.model
        if isinstance(model, LlamaWithISE):
            model.model.save_pretrained(output_dir)
            # The (f) tokens-only ablation runs with model.ise=None; in that
            # case skip the ISE save so probe_stage_state(requires_ise=False)
            # treats the resulting checkpoint as resumable.
            if model.ise is not None:
                ise_path = Path(output_dir) / "ise_weights.pt"
                torch.save(model.ise.state_dict(), ise_path)
                logger.info("Saved LoRA + ISE weights to %s", output_dir)
            else:
                logger.info(
                    "Saved LoRA to %s (no ISE: tokens-only ablation)",
                    output_dir,
                )
        else:
            # Fallback for non-ISE models
            model.save_pretrained(output_dir)
            logger.info("Saved model to %s", output_dir)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
