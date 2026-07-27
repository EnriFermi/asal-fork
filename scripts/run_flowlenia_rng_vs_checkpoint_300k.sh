#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${FLOWLENIA_PYTHON:-/home/coder/.conda/envs/torchjax/bin/python}"
OUTPUT_ROOT="${FLOWLENIA_RNG_VS_CHECKPOINT_ROOT:-$ROOT/analysis/results/flowlenia_rng_vs_checkpoint_shared_init_8rng_dup_300k_v1}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "FlowLenia Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "[stage] prepare"
"$PYTHON_BIN" scripts/flowlenia_rng_vs_checkpoint_300k.py prepare \
  --output-root "$OUTPUT_ROOT"

echo "[stage] preflight"
"$PYTHON_BIN" scripts/flowlenia_rng_vs_checkpoint_300k.py preflight \
  --output-root "$OUTPUT_ROOT"

echo "[stage] ten optimized checkpoints through 300k"
"$PYTHON_BIN" scripts/flowlenia_rng_vs_checkpoint_300k.py simulate \
  --output-root "$OUTPUT_ROOT"

echo "[stage] full-batch reproducibility repeat"
"$PYTHON_BIN" scripts/flowlenia_rng_vs_checkpoint_300k.py audit \
  --output-root "$OUTPUT_ROOT"

echo "[stage] analysis"
"$PYTHON_BIN" scripts/analyze_flowlenia_rng_vs_checkpoint_300k.py \
  --output-root "$OUTPUT_ROOT"

echo "[complete] $OUTPUT_ROOT"
