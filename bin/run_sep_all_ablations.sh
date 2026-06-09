#!/bin/bash
# Run SEP against every trained-model ablation in sequence.
#
# Iterates the seven nicknames the proposal Appendix B documents as
# ablation rows (the off-the-shelf `base_stock` is intentionally
# excluded — add it manually with --include-stock if needed).
#
# Per-model failures (missing checkpoint, OOM, etc.) are logged and the
# sweep continues with the next model so a single bad run doesn't
# waste hours of GPU time.
#
# Resume behavior (default): each per-model invocation passes --resume,
# which (a) skips models whose latest run_<ts>/ already has metrics.json
# and (b) reuses the newest partial run_<ts>/ to pick up cached
# responses + scoring. Pass --no-resume to force every model to start
# from a fresh run_<ts>/ (slow; only when you want to overwrite results).
#
# Usage:
#   bin/run_sep_all_ablations.sh
#   bin/run_sep_all_ablations.sh --format chat_template
#   bin/run_sep_all_ablations.sh --mapping A
#   bin/run_sep_all_ablations.sh --limit 100
#   bin/run_sep_all_ablations.sh --override sep.generation_batch_size=2
#   bin/run_sep_all_ablations.sh --include-stock
#   bin/run_sep_all_ablations.sh --no-resume      # force fresh runs
#
# Parallel runs (e.g. across two GPUs or two terminals on a multi-GPU host):
#   # terminal 1
#   bin/run_sep_all_ablations.sh --models gw_dpo,sft_only,standard_dpo
#   # terminal 2
#   bin/run_sep_all_ablations.sh --models base_with_tokens,bilateral,three_level,tokens_only
#
# --models takes a comma-separated subset of nicknames and replaces the default
# 7-ablation list entirely. Disjoint --models sets across invocations are
# mutually exclusive: each invocation writes only to its own per-model output
# subdirectories (the path is keyed by model nickname), so no two parallel
# runs touch the same files. --models and --include-stock are mutually
# exclusive — to include base_stock under --models, list it explicitly.
#
# Per-model output lands at evaluation/external/sep/<model>__<format>__mappingA/run_<UTC-ts>/.

set -uo pipefail   # NB: not -e — we WANT to continue past a failed model.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

FORMAT="delimited"
MAPPING="A"
LIMIT_FLAG=()
INCLUDE_STOCK=false
RESUME_FLAG="--resume"
EXTRA_OVERRIDES=()
MODELS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      if [[ "${2:-}" != "delimited" && "${2:-}" != "chat_template" ]]; then
        echo "ERROR: --format must be 'delimited' or 'chat_template'" >&2
        exit 1
      fi
      FORMAT="$2"; shift 2 ;;
    --mapping)
      if [[ "${2:-}" != "A" && "${2:-}" != "B" ]]; then
        echo "ERROR: --mapping must be 'A' or 'B'" >&2
        exit 1
      fi
      MAPPING="$2"; shift 2 ;;
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
    --models)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --models requires a comma-separated list of nicknames" >&2
        exit 1
      fi
      MODELS_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      awk '/^set -/{exit} NR>1{sub(/^# ?/,""); print}' "$0"
      exit 0 ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      echo "Run with --help to see usage." >&2
      exit 1 ;;
  esac
done

if [[ -n "$MODELS_OVERRIDE" && "$INCLUDE_STOCK" == true ]]; then
  echo "ERROR: --models and --include-stock are mutually exclusive." >&2
  echo "       To include base_stock under --models, list it explicitly," >&2
  echo "       e.g. --models base_stock,gw_dpo" >&2
  exit 1
fi

if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
  # Strip whitespace from each entry so "a, b" and "a,b" both work.
  for i in "${!MODELS[@]}"; do
    MODELS[$i]="$(echo "${MODELS[$i]}" | tr -d '[:space:]')"
  done
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: --models parsed to an empty list" >&2
    exit 1
  fi
else
  MODELS=(
    base_with_tokens     # (a)
    sft_only             # (b)
    standard_dpo         # (c)
    gw_dpo               # (d) — linear-schedule production run
    bilateral            #     — bilateral-schedule production run
    three_level          # (e)
    tokens_only          # (f) — no ISE
  )
  if [[ "$INCLUDE_STOCK" == true ]]; then
    MODELS=(base_stock "${MODELS[@]}")
  fi
fi

failures=()
start=$(date +%s)

for model in "${MODELS[@]}"; do
  printf '\n==============================================================\n'
  printf '[%s] SEP: %s (format=%s, mapping=%s)\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$model" "$FORMAT" "$MAPPING"
  printf '==============================================================\n'
  if python bin/run_sep.py \
        --model "$model" \
        --format "$FORMAT" \
        --mapping "$MAPPING" \
        "$RESUME_FLAG" \
        ${LIMIT_FLAG[@]+"${LIMIT_FLAG[@]}"} \
        ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}; then
    printf '\n[OK]   %s\n' "$model"
  else
    rc=$?
    printf '\n[FAIL] %s (exit=%d) — continuing with next model\n' "$model" "$rc"
    failures+=("$model")
  fi
done

elapsed=$(( $(date +%s) - start ))
printf '\n==============================================================\n'
printf 'SEP sweep complete in %ds across %d model(s)\n' "$elapsed" "${#MODELS[@]}"
if (( ${#failures[@]} > 0 )); then
  printf 'Failed runs (%d): %s\n' "${#failures[@]}" "${failures[*]}"
  exit 1
fi
printf 'All %d models OK\n' "${#MODELS[@]}"
