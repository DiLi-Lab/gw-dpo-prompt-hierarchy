#!/bin/bash
# Run MT-Bench against every trained-model ablation in sequence.
#
# Iterates the seven nicknames the proposal Appendix B documents as
# ablation rows (the off-the-shelf `base_stock` is intentionally
# excluded — add it manually with --include-stock if needed).
#
# Per-model failures (missing checkpoint, OOM, judge-API error) are
# logged and the sweep continues with the next model so a single bad
# run doesn't waste hours of GPU + judge time.
#
# Resume behavior (default): each per-model invocation passes --resume,
# which (a) skips models whose latest run_<ts>/ already has metrics.json
# and (b) reuses the newest partial run_<ts>/ to pick up cached
# responses + judge-graded scoring. Pass --no-resume to force every
# model to start from a fresh run_<ts>/ (slow + extra judge-API spend;
# only when you want to overwrite results).
#
# Usage:
#   bin/run_mt_bench_all_ablations.sh
#   bin/run_mt_bench_all_ablations.sh --limit 8
#   bin/run_mt_bench_all_ablations.sh --override mt_bench.generation_batch_size=2
#   bin/run_mt_bench_all_ablations.sh --include-stock          # also runs base_stock
#   bin/run_mt_bench_all_ablations.sh --no-resume              # force fresh runs
#
# Per-model output lands at evaluation/external/mt_bench/<model>__chat_template/run_<UTC-ts>/.

set -uo pipefail   # NB: not -e — we WANT to continue past a failed model.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LIMIT_FLAG=()
INCLUDE_STOCK=false
RESUME_FLAG="--resume"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT_FLAG=(--limit "$2"); shift 2 ;;
    --override)
      EXTRA_OVERRIDES+=(--override "$2"); shift 2 ;;
    --include-stock)
      INCLUDE_STOCK=true; shift ;;
    --resume)
      RESUME_FLAG="--resume"; shift ;;
    --no-resume)
      RESUME_FLAG="--no-resume"; shift ;;
    -h|--help)
      awk '/^set -/{exit} NR>1{sub(/^# ?/,""); print}' "$0"
      exit 0 ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      echo "Run with --help to see usage." >&2
      exit 1 ;;
  esac
done

MODELS=(
  base_with_tokens     # (a) embedding-resized baseline, no training
  sft_only             # (b) SFT only
  standard_dpo         # (c) SFT + standard DPO (alpha=0)
  gw_dpo               # (d) SFT + GW-DPO, linear-schedule production run
  bilateral            #     SFT + GW-DPO, bilateral-schedule production run
  three_level          # (e) GW-DPO trained on 3-level data
  tokens_only          # (f) GW-DPO with tokens but no ISE
)
if [[ "$INCLUDE_STOCK" == true ]]; then
  MODELS=(base_stock "${MODELS[@]}")
fi

failures=()
start=$(date +%s)

for model in "${MODELS[@]}"; do
  printf '\n==============================================================\n'
  printf '[%s] MT-Bench: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$model"
  printf '==============================================================\n'
  if python bin/run_mt_bench.py \
        --model "$model" \
        "$RESUME_FLAG" \
        "${LIMIT_FLAG[@]}" \
        "${EXTRA_OVERRIDES[@]}"; then
    printf '\n[OK]   %s\n' "$model"
  else
    rc=$?
    printf '\n[FAIL] %s (exit=%d) — continuing with next model\n' "$model" "$rc"
    failures+=("$model")
  fi
done

elapsed=$(( $(date +%s) - start ))
printf '\n==============================================================\n'
printf 'MT-Bench sweep complete in %ds across %d model(s)\n' "$elapsed" "${#MODELS[@]}"
if (( ${#failures[@]} > 0 )); then
  printf 'Failed runs (%d): %s\n' "${#failures[@]}" "${failures[*]}"
  exit 1
fi
printf 'All %d models OK\n' "${#MODELS[@]}"
