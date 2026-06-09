#!/usr/bin/env python3
"""DPO hyperparameter search across ρ × β configurations.

Runs the full 3-stage curriculum with sDPO for each (ρ, β) configuration
in the sweep grid, then evaluates on the held-out hp_select cut and
writes ranked results.

All outputs live under ``models/hp_search/`` and are isolated from the
production training artifacts.

Usage:
    python bin/train_dpo_hp_search.py                      # all 12 configs
    python bin/train_dpo_hp_search.py --configs "1-6"     # subset (parallel)
    python bin/train_dpo_hp_search.py --configs "3,7,11"  # specific configs
    python bin/train_dpo_hp_search.py --config configs/test.yaml --configs 1
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import os
# Route HF + pip caches to fast NVMe on Thunder instances.
_EPHEMERAL = Path("/ephemeral")
if _EPHEMERAL.exists():
    os.environ.setdefault("HF_HOME", str(_EPHEMERAL / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_EPHEMERAL / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_EPHEMERAL / "hf_cache" / "datasets"))
    (_EPHEMERAL / "hf_cache" / "datasets").mkdir(parents=True, exist_ok=True)

import torch
import yaml
from peft import load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config
from src.config.constants import SPECIAL_TOKENS
from src.training import (
    STAGE_COMPLETE,
    DPOHierarchyCollator,
    evaluate_reward_accuracies,
    probe_stage_state,
    run_dpo_curriculum,
)

# Reuse helpers from the production script.
sys.path.insert(0, str(_PROJECT_ROOT / "bin"))
from train_dpo import (  # noqa: E402
    TORCH_DTYPE_MAP,
    create_policy_model,
    create_reference_model,
    find_sft_best_checkpoint,
    load_jsonl_dataset,
    merge_sft_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# -------- Sweep grid --------

RHO_VALUES = [0.5, 1.0, 2.0, 3.0]
BETA_VALUES = [0.05, 0.1, 0.2]


def build_sweep_grid() -> list[dict]:
    """Return the 12-config grid as a list of dicts with stable numbering."""
    configs = []
    idx = 1
    for rho in RHO_VALUES:
        for beta in BETA_VALUES:
            alpha = rho * beta
            rho_str = f"{rho}".replace(".", "p")
            beta_str = f"{beta}".replace(".", "p")
            configs.append({
                "config_id": f"cfg{idx:02d}_r{rho_str}_b{beta_str}",
                "index": idx, "rho": rho, "beta": beta, "alpha": alpha,
            })
            idx += 1
    return configs


def parse_configs_arg(arg: str, max_index: int) -> list[int]:
    """Parse --configs like ``1-6`` or ``3,7,11`` into a list of indices."""
    result: set[int] = set()
    for piece in arg.split(","):
        piece = piece.strip()
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                result.add(i)
        else:
            result.add(int(piece))
    out = sorted(i for i in result if 1 <= i <= max_index)
    if not out:
        raise ValueError(f"No valid configs in {arg!r} (max={max_index})")
    return out


# -------- Results logging --------

def read_final_losses(final_stage_dir: Path) -> dict:
    """Extract the most recent train loss and eval loss from trainer_state.json.

    Returns a dict with ``final_train_loss`` and ``final_eval_loss_val_train``
    keys, each either a float or None if the corresponding loss never fired.
    """
    state_path = final_stage_dir / "trainer_state.json"
    if not state_path.exists():
        return {"final_train_loss": None, "final_eval_loss_val_train": None}
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {"final_train_loss": None, "final_eval_loss_val_train": None}
    train_loss: float | None = None
    eval_loss: float | None = None
    for entry in state.get("log_history", []):
        if "loss" in entry and "eval_loss" not in entry:
            train_loss = float(entry["loss"])
        if "eval_loss" in entry:
            eval_loss = float(entry["eval_loss"])
    return {"final_train_loss": train_loss, "final_eval_loss_val_train": eval_loss}


def append_result(results_path: Path, row: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _dedupe_by_config_id(rows: list[dict]) -> list[dict]:
    """Keep the most recent row per config_id (by completed_at).

    A failed attempt followed by a successful re-run leaves two rows in
    results.jsonl; the summary should reflect only the latest outcome.
    """
    latest: dict[str, dict] = {}
    for row in rows:
        cid = row.get("config_id")
        if cid is None:
            continue
        prev = latest.get(cid)
        if prev is None or row.get("completed_at", "") >= prev.get("completed_at", ""):
            latest[cid] = row
    return list(latest.values())


def rewrite_summary(hp_root: Path) -> None:
    """Regenerate results_summary.md from results.jsonl."""
    results_path = hp_root / "results.jsonl"
    if not results_path.exists():
        return
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    rows = _dedupe_by_config_id(rows)
    completed = [r for r in rows if r.get("status", "completed") == "completed"]
    failed = [r for r in rows if r.get("status") == "failed"]
    completed.sort(
        key=lambda r: (
            -r["hp_select"]["macro_avg_accuracy"],
            -r["hp_select"]["gap_weighted_accuracy"],
        ),
    )

    lines = ["# DPO HP Search Results", ""]
    if completed:
        winner = completed[0]
        lines.append(f"**Winner:** `{winner['config_id']}` — "
                       f"macro-avg={winner['hp_select']['macro_avg_accuracy']:.4f}, "
                       f"ρ={winner['rho']}, β={winner['beta']}, α={winner['alpha']}")
        lines.append("")
        lines.append("| Rank | Config | ρ | β | α | macro-avg | gap-wtd | "
                       "gap0 | gap1 | gap2 | gap3 | gap4 | wall (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for rank, r in enumerate(completed, 1):
            acc = r["hp_select"]["per_gap_accuracy"]
            lines.append(
                f"| {rank} | `{r['config_id']}` | {r['rho']} | {r['beta']} | "
                f"{r['alpha']:.3f} | "
                f"{r['hp_select']['macro_avg_accuracy']:.4f} | "
                f"{r['hp_select']['gap_weighted_accuracy']:.4f} | "
                f"{acc['0']:.3f} | {acc['1']:.3f} | {acc['2']:.3f} | "
                f"{acc['3']:.3f} | {acc['4']:.3f} | "
                f"{int(r['wall_time_seconds'])} |",
            )
    if failed:
        lines.append("")
        lines.append("## Failed configs")
        for r in failed:
            lines.append(f"- `{r['config_id']}`: {r.get('error', 'unknown')}")
    (hp_root / "results_summary.md").write_text("\n".join(lines) + "\n")


def write_best_config(hp_root: Path) -> None:
    results_path = hp_root / "results.jsonl"
    if not results_path.exists():
        return
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    rows = _dedupe_by_config_id(rows)
    completed = [r for r in rows if r.get("status", "completed") == "completed"]
    if not completed:
        return
    completed.sort(
        key=lambda r: (
            -r["hp_select"]["macro_avg_accuracy"],
            -r["hp_select"]["gap_weighted_accuracy"],
        ),
    )
    w = completed[0]
    best = {
        "config_id": w["config_id"],
        "rho": w["rho"], "beta": w["beta"], "alpha": w["alpha"],
        "macro_avg_accuracy": w["hp_select"]["macro_avg_accuracy"],
        "gap_weighted_accuracy": w["hp_select"]["gap_weighted_accuracy"],
        "best_checkpoint_path": w["best_checkpoint_path"],
        "recommendation": (
            f"Set configs/base_linear.yaml dpo.beta={w['beta']} and "
            f"dpo.gravity_alpha={w['alpha']}, then run bin/train_dpo.py "
            "on the full val set."
        ),
    }
    (hp_root / "best_config.json").write_text(json.dumps(best, indent=2))


# -------- Main --------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DPO hyperparameter search across ρ × β.",
    )
    parser.add_argument("--config", type=Path,
                         default=_PROJECT_ROOT / "configs" / "base_linear.yaml")
    parser.add_argument("--sft-checkpoint", type=Path, default=None)
    parser.add_argument("--configs", type=str, default="1-12",
                         help='Config indices to run, e.g. "1-12" or "3,7,11".')
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    base_cfg = load_config(config_path=args.config, overrides=args.override)
    grid = build_sweep_grid()
    selected_indices = parse_configs_arg(args.configs, max_index=len(grid))

    hp_root = base_cfg.paths.hp_search_dir
    hp_data_dir = base_cfg.paths.hp_search_data_dir
    hp_runs_dir = base_cfg.paths.hp_search_runs_dir
    hp_root.mkdir(parents=True, exist_ok=True)

    # --- Ensure HP-select split exists (idempotent) ---
    split_script = _PROJECT_ROOT / "bin" / "build_hp_select_split.py"
    val_source = base_cfg.paths.for_split("val").dpo_combined
    subprocess.run(
        [sys.executable, str(split_script),
          "--source", str(val_source),
          "--out-dir", str(hp_data_dir),
          "--target-size", "1000", "--seed", "42"],
        cwd=str(_PROJECT_ROOT), check=True,
    )

    hp_select_path = hp_data_dir / "hp_select.jsonl"
    val_train_path = hp_data_dir / "val_train.jsonl"

    # --- Load datasets once ---
    train_path = base_cfg.paths.for_split("train").dpo_combined
    train_dataset = load_jsonl_dataset(train_path)
    val_train_dataset = load_jsonl_dataset(val_train_path)
    hp_select_dataset = load_jsonl_dataset(hp_select_path)
    logger.info(
        "Datasets: train=%d, val_train=%d, hp_select=%d",
        len(train_dataset), len(val_train_dataset), len(hp_select_dataset),
    )

    # --- Tokenizer + dtype ---
    tokenizer = AutoTokenizer.from_pretrained(str(base_cfg.paths.tokenizer_dir))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = TORCH_DTYPE_MAP[base_cfg.model.torch_dtype]
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS
    ]

    # --- Merge SFT once (shared across all configs) ---
    sft_checkpoint = args.sft_checkpoint or find_sft_best_checkpoint(base_cfg)
    logger.info("Using SFT checkpoint: %s", sft_checkpoint)
    merged_dir = merge_sft_checkpoint(base_cfg, sft_checkpoint, tokenizer, torch_dtype)

    # --- Per-config loop ---
    results_path = hp_root / "results.jsonl"
    for idx in selected_indices:
        entry = grid[idx - 1]
        config_id = entry["config_id"]
        run_dir = hp_runs_dir / config_id
        hp_eval_path = run_dir / "hp_eval.json"

        if hp_eval_path.exists():
            logger.info("[%s] Already completed, skipping.", config_id)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(
                {"rho": entry["rho"], "beta": entry["beta"],
                  "alpha": entry["alpha"],
                  "curriculum_enabled": base_cfg.dpo.curriculum_enabled,
                  "num_curriculum_stages": base_cfg.dpo.num_curriculum_stages,
                  "base_config": str(args.config)},
                f,
            )

        logger.info(
            "=== [%s] ρ=%s, β=%s, α=%s ===",
            config_id, entry["rho"], entry["beta"], entry["alpha"],
        )

        # Build a per-config override for α and β.
        per_cfg = load_config(
            config_path=args.config,
            overrides=args.override + [
                f"dpo.gravity_alpha={entry['alpha']}",
                f"dpo.beta={entry['beta']}",
            ],
        )

        start = time.time()
        policy_model = None
        ref_model = None
        eval_policy = None
        eval_ref = None
        try:
            # Resume policy: skip both training and model creation only when
            # every stage is COMPLETE on disk (probe-verified — i.e., the
            # stage-level trainer_state.json shows global_step==max_steps and
            # best-checkpoint/ has its ISE weights). Any other state means
            # run_dpo_curriculum needs the policy/ref models so it can either
            # mid-stage resume or train from scratch as appropriate.
            final_stage = per_cfg.dpo.final_stage_index
            stage_states = [
                probe_stage_state(run_dir / f"stage{i}")
                for i in range(1, final_stage + 1)
            ]
            all_complete = all(s[0] == STAGE_COMPLETE for s in stage_states)
            if all_complete:
                best_ckpt = stage_states[-1][1]
                logger.info(
                    "[%s] All %d stage(s) complete on disk; skipping training.",
                    config_id, final_stage,
                )
            else:
                policy_model = create_policy_model(
                    per_cfg, merged_dir, torch_dtype, special_token_ids,
                )
                ref_model = create_reference_model(per_cfg, merged_dir, torch_dtype)

                best_ckpt = run_dpo_curriculum(
                    cfg=per_cfg, merged_dir=merged_dir, tokenizer=tokenizer,
                    torch_dtype=torch_dtype, special_token_ids=special_token_ids,
                    train_dataset=train_dataset, val_dataset=val_train_dataset,
                    run_dir=run_dir, policy_model=policy_model, ref_model=ref_model,
                )

            # LoRA + ISE at best_ckpt is the portable per-config artifact;
            # skip the full 16 GB merge to stay within the persistent disk
            # budget. Only the winner is merged post-sweep.
            ise_weights_path = best_ckpt / "ise_weights.pt"
            if not ise_weights_path.exists():
                ise_weights_path = run_dir / "ise_weights_final.pt"
                if policy_model is not None:
                    torch.save(policy_model.ise.state_dict(), ise_weights_path)
                else:
                    raise FileNotFoundError(
                        f"Missing ISE weights at {best_ckpt}; cannot eval.",
                    )

            # Release the training models BEFORE creating eval models. On an
            # 80 GB GPU, two 8B models in bf16 (~16 GB each) plus LoRA/ISE
            # already sit at ~35 GB; loading two more fresh copies for eval
            # (another ~32 GB) leaves <15 GB for activations + 128K-vocab
            # logits and OOMs inside hp_select forward. Free them here so
            # only the eval pair lives on GPU during evaluate_reward_accuracies.
            del policy_model, ref_model
            policy_model = None
            ref_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Evaluate on hp_select using the merged model as policy ---
            # Move to GPU and unify dtypes: create_policy_model leaves models
            # on CPU with fp32 ISE (and fp32 LoRA adapters), which during
            # training the HF Trainer reconciles via accelerator.prepare +
            # bf16 autocast. For this standalone eval we must do it ourselves,
            # otherwise the fp32 ISE output promotes inputs_embeds to fp32
            # and the bf16 base linear layers raise a dtype mismatch.
            eval_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
            eval_policy = create_policy_model(
                per_cfg, merged_dir, torch_dtype, special_token_ids,
            )
            adapter_state = load_peft_weights(str(best_ckpt))
            set_peft_model_state_dict(eval_policy.model, adapter_state)
            eval_policy.ise.load_state_dict(
                torch.load(ise_weights_path, weights_only=True),
            )
            eval_policy = eval_policy.to(device=eval_device, dtype=torch_dtype)
            eval_policy.eval()

            eval_ref = create_reference_model(per_cfg, merged_dir, torch_dtype)
            eval_ref = eval_ref.to(device=eval_device, dtype=torch_dtype)

            collator = DPOHierarchyCollator(
                pad_token_id=tokenizer.pad_token_id,
                tokenizer=tokenizer,
                margin_schedule=per_cfg.dpo.margin_schedule,
            )
            metrics = evaluate_reward_accuracies(
                policy_model=eval_policy, ref_model=eval_ref,
                dataset=hp_select_dataset, beta=per_cfg.dpo.beta,
                collator=collator, batch_size=per_cfg.dpo.per_device_batch_size,
            )

            wall = time.time() - start
            final_losses = read_final_losses(
                run_dir / f"stage{per_cfg.dpo.final_stage_index}",
            )
            row = {
                "config_id": config_id,
                "rho": entry["rho"], "beta": entry["beta"],
                "alpha": entry["alpha"],
                "curriculum_enabled": per_cfg.dpo.curriculum_enabled,
                "curriculum_stages": per_cfg.dpo.final_stage_index,
                "epochs_per_stage": per_cfg.dpo.epochs_per_stage,
                "final_train_loss": final_losses["final_train_loss"],
                "final_eval_loss_val_train": final_losses["final_eval_loss_val_train"],
                "hp_select": {
                    "per_gap_accuracy": {str(k): v for k, v in
                                           metrics["per_gap_accuracy"].items()},
                    "per_gap_count": {str(k): v for k, v in
                                        metrics["per_gap_count"].items()},
                    "macro_avg_accuracy": metrics["macro_avg_accuracy"],
                    "gap_weighted_accuracy": metrics["gap_weighted_accuracy"],
                    "mean_reward_margin": metrics["mean_reward_margin"],
                },
                "best_checkpoint_path": str(
                    best_ckpt.resolve().relative_to(_PROJECT_ROOT),
                ),
                "wall_time_seconds": wall,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            }
            hp_eval_path.write_text(json.dumps(row, indent=2))
            append_result(results_path, row)
            rewrite_summary(hp_root)

        except Exception as e:
            wall = time.time() - start
            logger.exception("[%s] Failed after %.1fs", config_id, wall)
            row = {
                "config_id": config_id,
                "rho": entry["rho"], "beta": entry["beta"],
                "alpha": entry["alpha"],
                "wall_time_seconds": wall,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_result(results_path, row)
            rewrite_summary(hp_root)
        finally:
            # Always drop every model reference between configs. A mid-eval
            # OOM previously left eval_policy/eval_ref bound in main()'s
            # scope, pinning ~32 GB of GPU weights so the next config OOM'd
            # a handful of training steps in. gc + empty_cache only help
            # once the Python bindings are gone.
            del policy_model, ref_model, eval_policy, eval_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Final summary + best config ---
    rewrite_summary(hp_root)
    write_best_config(hp_root)
    logger.info("HP search complete. Results in %s", hp_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
