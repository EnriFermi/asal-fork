#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${FLOWLENIA_PYTHON:-/home/coder/.conda/envs/torchjax/bin/python}"
OUTPUT_ROOT="${FLOWLENIA_RNG_SENSITIVITY_ROOT:-$ROOT/analysis/results/flowlenia_rng_sensitivity_trajectory20_shared4_9branch_10k_v1}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "FlowLenia Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/flowlenia_rng_sensitivity_experiment.py plan \
  --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" scripts/flowlenia_rng_sensitivity_experiment.py pilot \
  --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" scripts/flowlenia_rng_sensitivity_experiment.py simulate \
  --output-root "$OUTPUT_ROOT" \
  2>&1 | tee -a "$OUTPUT_ROOT/simulation_production.log"
"$PYTHON_BIN" scripts/analyze_flowlenia_rng_sensitivity.py all \
  --output-root "$OUTPUT_ROOT"
