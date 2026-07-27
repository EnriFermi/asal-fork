#!/usr/bin/env bash
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="/home/coder/.conda/envs/torchjax/bin/python"
OUTPUT_ROOT="$REPO_ROOT/analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
LOG_PATH="$OUTPUT_ROOT/production.log"
STATUS_PATH="$OUTPUT_ROOT/production_status.txt"
PID_PATH="$OUTPUT_ROOT/production.pid"
SCRIPT="$REPO_ROOT/scripts/flowlenia_c5_rng_only_horizon_grid.py"

mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "$$" > "$PID_PATH"
printf 'running plan %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$OUTPUT_ROOT/jax_compilation_cache"

cd "$REPO_ROOT"
"$PYTHON_BIN" "$SCRIPT" --phase plan
printf 'running preflight %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase preflight
printf 'running free %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase free
printf 'running walls %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase walls
printf 'running audit %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase audit
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
rm -f "$PID_PATH"

printf 'Production simulation pipeline complete. Log: %s\n' "$LOG_PATH"
