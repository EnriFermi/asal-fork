#!/usr/bin/env bash

set -u

ROOT="${1:-/home/coder/project/analysis/results/flowlenia_rng_vs_interrandom_shared_init_8rng_dup_300k_v1}"
REPO="/home/coder/project"
ENV_ROOT="/home/coder/.conda/envs/torchjax"
PYTHON="$ENV_ROOT/bin/python"
CUDA_TOOL_BIN="$ENV_ROOT/lib/python3.11/site-packages/triton/backends/nvidia/bin"
SIM_SCRIPT="$REPO/scripts/flowlenia_rng_vs_interrandom_300k.py"
ANALYSIS_SCRIPT="$REPO/scripts/analyze_flowlenia_rng_vs_interrandom_300k.py"
LOG="$ROOT/pipeline_supervisor.log"

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

json_complete() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("status") == "complete" else 1)
PY
}

echo "[$(timestamp)] supervisor started"

if [ -f "$ROOT/production.pid" ]; then
    production_pid="$(cat "$ROOT/production.pid")"
    while kill -0 "$production_pid" 2>/dev/null; do
        if json_complete "$ROOT/simulation_completion.json"; then
            break
        fi
        sleep 60
    done
fi

if ! json_complete "$ROOT/simulation_completion.json"; then
    attempt=1
    while [ "$attempt" -le 3 ]; do
        echo "[$(timestamp)] production resume attempt $attempt"
        env \
            PATH="$ENV_ROOT/bin:$PATH" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            "$PYTHON" "$SIM_SCRIPT" simulate --output-root "$ROOT" \
            >>"$ROOT/production.log" 2>&1
        if json_complete "$ROOT/simulation_completion.json"; then
            break
        fi
        attempt=$((attempt + 1))
    done
fi

if ! json_complete "$ROOT/simulation_completion.json"; then
    echo "[$(timestamp)] production failed after resume attempts"
    exit 1
fi
echo "[$(timestamp)] production complete"

if ! json_complete "$ROOT/audit_simulation_completion.json"; then
    echo "[$(timestamp)] starting full 300k repeat audit"
    env \
        PATH="$ENV_ROOT/bin:$PATH" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" "$SIM_SCRIPT" audit --output-root "$ROOT" \
        >>"$ROOT/full_repeat_simulation.log" 2>&1
fi

if ! json_complete "$ROOT/audit_simulation_completion.json"; then
    echo "[$(timestamp)] full repeat audit simulation failed"
    exit 1
fi
echo "[$(timestamp)] full repeat audit simulation complete"

echo "[$(timestamp)] starting analysis"
env \
    PATH="$CUDA_TOOL_BIN:$ENV_ROOT/bin:$PATH" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$PYTHON" "$ANALYSIS_SCRIPT" \
    --random-root "$ROOT" \
    --optimized-root \
    "$REPO/analysis/results/flowlenia_rng_vs_checkpoint_shared_init_8rng_dup_300k_v1" \
    >"$ROOT/analysis.log" 2>&1
analysis_status=$?
if [ "$analysis_status" -ne 0 ]; then
    echo "[$(timestamp)] analysis failed with status $analysis_status"
    exit "$analysis_status"
fi

(
    cd "$ROOT/analysis" || exit 1
    find figures tables -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum >artifact_sha256.txt
)
timestamp >"$ROOT/pipeline_complete.txt"
echo "[$(timestamp)] pipeline complete"
