#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-analysis/results/gol_transition_mspd/gpu_all_rules_${RUN_ID}}"

conda run --no-capture-output -n onerec python -u scripts/gol_transition_mspd_experiment.py \
  --experiment rule-sweep \
  --all-rules \
  --require-accelerator \
  --backend jax \
  --L 64 \
  --T 2048 \
  --burn-in "${BURN_IN:-512}" \
  --window-size 64 \
  --window-step 16 \
  --n-cell-sample 256 \
  --null-reps 1 \
  --distance js \
  --pair-sample 512 \
  --min-delta-h-nonzero-frac 0.5 \
  --delta-h-nonzero-eps "${DELTA_H_NONZERO_EPS:-1e-2}" \
  --rule-initial-density-min "${RULE_INITIAL_DENSITY_MIN:-0.05}" \
  --rule-initial-density-max "${RULE_INITIAL_DENSITY_MAX:-0.4}" \
  --random-seed 0 \
  --eval-batch-size "${RULE_EVAL_BATCH_SIZE:-64}" \
  --jax-metric-batch-size "${RULE_JAX_METRIC_BATCH_SIZE:-16}" \
  --n-rule-initial-boards "${N_RULE_INITIAL_BOARDS:-4}" \
  --progress-interval-rules "${PROGRESS_INTERVAL_RULES:-64}" \
  --output-dir "$OUT_DIR" \
  --no-videos
