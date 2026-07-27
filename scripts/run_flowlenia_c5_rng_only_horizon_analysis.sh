#!/usr/bin/env bash
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="/home/coder/.conda/envs/torchjax/bin/python"
OUTPUT_ROOT="$REPO_ROOT/analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
SCRIPT="$REPO_ROOT/scripts/flowlenia_c5_rng_only_horizon_analysis.py"
STATUS_PATH="$OUTPUT_ROOT/analysis_production_status.txt"
PID_PATH="$OUTPUT_ROOT/analysis_production.pid"
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

mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "$$" > "$PID_PATH"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$OUTPUT_ROOT/jax_analysis_compilation_cache"
export MPLBACKEND=Agg

cd "$REPO_ROOT"

CURRENT_PHASE="preflight"
printf 'running preflight %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase preflight

CURRENT_PHASE="embeddings"
printf 'running embeddings %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase embeddings

CURRENT_PHASE="metrics"
printf 'running metrics %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase metrics

CURRENT_PHASE="plots"
printf 'running plots %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase plots

CURRENT_PHASE="videos"
printf 'running videos %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase videos --video-fps 24 --video-hold-frames 6

CURRENT_PHASE="audit"
printf 'running audit %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
"$PYTHON_BIN" "$SCRIPT" --phase audit

CURRENT_PHASE="complete"
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$STATUS_PATH"
rm -f "$PID_PATH"
trap - EXIT
