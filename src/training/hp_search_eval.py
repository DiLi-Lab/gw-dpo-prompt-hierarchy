"""Post-training evaluation helpers for the DPO HP search.

``compute_reward_accuracy_metrics`` is a pure function operating on already
computed log-probabilities — it has no dependency on any model and can be
unit-tested in isolation.

``evaluate_reward_accuracies`` (added below, Task 4) runs the policy and
reference forwards over an HP-select dataset and feeds the log-probs to
the pure function.
"""

import contextlib

import torch
from datasets import Dataset
from trl.trainer.utils import selective_log_softmax


def compute_reward_accuracy_metrics(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    level_gaps: list[int],
    beta: float,
) -> dict:
    """Compute per-gap reward accuracies and aggregate metrics.

    Args:
        chosen_logps: Policy log-probs of the chosen response, one per pair.
        rejected_logps: Policy log-probs of the rejected response.
        ref_chosen_logps: Reference log-probs of the chosen response.
        ref_rejected_logps: Reference log-probs of the rejected response.
        level_gaps: Hierarchy distance per pair (same length as logp tensors).
        beta: DPO inverse temperature, used only for computing reward
            margins (the sign of the reward margin is invariant to beta
            so accuracy numbers do not depend on it).

    Returns:
        Dict with keys:
        - ``per_gap_accuracy``: dict ``{gap: accuracy}`` over gaps 0..4.
        - ``per_gap_count``: dict ``{gap: count}`` over gaps 0..4.
        - ``macro_avg_accuracy``: equal-weighted mean over populated gaps.
        - ``gap_weighted_accuracy``: sum(gap*correct) / sum(gap*count) over
          populated gaps. Zero when all populated gaps are 0.
        - ``mean_reward_margin``: mean of ``beta*(r_chosen - r_rejected)``.
    """
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("chosen/rejected logps must have matching shape")
    if len(level_gaps) != chosen_logps.shape[0]:
        raise ValueError(
            f"level_gaps ({len(level_gaps)}) must match logps batch "
            f"({chosen_logps.shape[0]})",
        )

    r_chosen = beta * (chosen_logps - ref_chosen_logps)
    r_rejected = beta * (rejected_logps - ref_rejected_logps)
    correct = (r_chosen > r_rejected).cpu().tolist()
    margins = (r_chosen - r_rejected).cpu().tolist()

    per_gap_correct: dict[int, int] = {g: 0 for g in range(5)}
    per_gap_count: dict[int, int] = {g: 0 for g in range(5)}
    for is_correct, gap in zip(correct, level_gaps):
        per_gap_correct[gap] += int(is_correct)
        per_gap_count[gap] += 1

    per_gap_accuracy = {
        g: (per_gap_correct[g] / per_gap_count[g]) if per_gap_count[g] > 0 else 0.0
        for g in range(5)
    }

    populated = [g for g in range(5) if per_gap_count[g] > 0]
    macro_avg = (
        sum(per_gap_accuracy[g] for g in populated) / len(populated)
        if populated else 0.0
    )

    total_weight = sum(g * per_gap_count[g] for g in populated)
    gap_weighted = (
        sum(g * per_gap_correct[g] for g in populated) / total_weight
        if total_weight > 0 else 0.0
    )

    mean_margin = sum(margins) / len(margins) if margins else 0.0

    return {
        "per_gap_accuracy": per_gap_accuracy,
        "per_gap_count": per_gap_count,
        "macro_avg_accuracy": macro_avg,
        "gap_weighted_accuracy": gap_weighted,
        "mean_reward_margin": mean_margin,
    }


def evaluate_reward_accuracies(
    policy_model,
    ref_model,
    dataset: Dataset,
    beta: float,
    collator,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict:
    """Run policy+reference forwards over ``dataset`` and compute metrics.

    Iterates the dataset in fixed-size slices (not via HF DataLoader) so
    that ``level_gap`` can be read from each raw record — the DPO collator
    does not pass it through. For each slice, builds the collated batch,
    runs policy and reference forwards under ``torch.no_grad``, computes
    per-pair log-probabilities (with masking over the completion tokens),
    and accumulates chosen/rejected logps and level_gaps for the batch.

    After all slices complete, delegates to ``compute_reward_accuracy_metrics``.

    Args:
        policy_model: Trained DPO policy.
        ref_model: Frozen reference model (SFT-merged).
        dataset: HuggingFace Dataset of DPO preference records, including
            ``level_gap``.
        beta: Policy's DPO beta; used for reward-margin reporting.
        collator: ``DPOHierarchyCollator`` instance.
        batch_size: Number of preference pairs per batched forward.
        device: Torch device; inferred from policy parameters if None.

    Returns:
        Dict from ``compute_reward_accuracy_metrics``.
    """
    if device is None:
        device = next(policy_model.parameters()).device
    policy_model.eval()
    ref_model.eval()

    # TRL's DataCollatorForPreference expects pre-tokenized prompt_ids /
    # chosen_ids / rejected_ids on each example. Training datasets get
    # tokenized by DPOTrainer._prepare_dataset; the hp_select split is
    # consumed directly here, so we mirror that tokenization before
    # delegating to the collator.
    tokenizer = getattr(collator, "tokenizer", None)
    if tokenizer is not None and "prompt_ids" not in dataset.column_names:
        eos_token = tokenizer.eos_token

        def _tokenize(example):
            chosen = example["chosen"]
            rejected = example["rejected"]
            if not chosen.endswith(eos_token):
                chosen = chosen + eos_token
            if not rejected.endswith(eos_token):
                rejected = rejected + eos_token
            prompt_ids = tokenizer(text=example["prompt"])["input_ids"]
            prompt_chosen_ids = tokenizer(
                text=example["prompt"] + chosen,
            )["input_ids"]
            prompt_rejected_ids = tokenizer(
                text=example["prompt"] + rejected,
            )["input_ids"]
            return {
                "prompt_ids": prompt_ids,
                "chosen_ids": prompt_chosen_ids[len(prompt_ids):],
                "rejected_ids": prompt_rejected_ids[len(prompt_ids):],
            }

        dataset = dataset.map(_tokenize)

    all_chosen_p: list[torch.Tensor] = []
    all_rejected_p: list[torch.Tensor] = []
    all_chosen_r: list[torch.Tensor] = []
    all_rejected_r: list[torch.Tensor] = []
    all_gaps: list[int] = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        raw = [dataset[i] for i in range(start, end)]
        gaps = [int(r["level_gap"]) for r in raw]
        batch = collator(raw)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        completion_mask = batch["completion_mask"].to(device)
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            model_kwargs["segment_ids"] = segment_ids.to(device)

        # ISE parameters are fp32 (HF Trainer keeps params in fp32 even
        # with bf16=True; only forwards are autocast), so without autocast
        # here the fp32 segment embeddings promote inputs_embeds to fp32
        # and the subsequent bf16 linear layers crash. Matches the
        # autocast wrapping that TRL applies during training.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            p_out = policy_model(**model_kwargs)
            r_out = ref_model(**model_kwargs)

        shift_labels = input_ids[..., 1:].contiguous()
        shift_cm = completion_mask[..., 1:].contiguous()

        def _logps(logits):
            shifted = logits[..., :-1, :].contiguous()
            lp = selective_log_softmax(shifted, shift_labels)
            lp = torch.where(shift_cm == 0, torch.zeros_like(lp), lp)
            return lp.sum(dim=1)

        p_logps = _logps(p_out.logits)
        r_logps = _logps(r_out.logits)

        chosen_p, rejected_p = p_logps.chunk(2, dim=0)
        chosen_r, rejected_r = r_logps.chunk(2, dim=0)

        all_chosen_p.append(chosen_p)
        all_rejected_p.append(rejected_p)
        all_chosen_r.append(chosen_r)
        all_rejected_r.append(rejected_r)
        all_gaps.extend(gaps)

    chosen_p = torch.cat(all_chosen_p, dim=0)
    rejected_p = torch.cat(all_rejected_p, dim=0)
    chosen_r = torch.cat(all_chosen_r, dim=0)
    rejected_r = torch.cat(all_rejected_r, dim=0)

    return compute_reward_accuracy_metrics(
        chosen_p, rejected_p, chosen_r, rejected_r, all_gaps, beta=beta,
    )
