#!/usr/bin/env bash
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="/home/coder/.conda/envs/torchjax/bin/python"
OUTPUT_ROOT="$REPO_ROOT/analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
STATUS_PATH="$OUTPUT_ROOT/production_status.txt"
PID_PATH="$OUTPUT_ROOT/production.pid"
SCRIPT="$REPO_ROOT/scripts/flowlenia_c5_rng_only_horizon_grid.py"
CURRENT_PHASE="bootstrap"

record_failure() {
    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        printf 'failed %s exit=%s %s\n' \
            "$CURRENT_PHASE" "$exit_code" "$(date -u +%FT%TZ)" \
            > "$STATUS_PATH"
    fi
}
trap record_failure EXIT

test -s "$OUTPUT_ROOT/preflight.json"
printf '%s\n' "$$" > "$PID_PATH"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$OUTPUT_ROOT/jax_compilation_cache"

cd "$REPO_ROOT"

CURRENT_PHASE="free"
printf 'running free %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase free

CURRENT_PHASE="walls"
printf 'running walls %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase walls

CURRENT_PHASE="audit"
printf 'running audit %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase audit

CURRENT_PHASE="complete"
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
rm -f "$PID_PATH"
trap - EXIT
