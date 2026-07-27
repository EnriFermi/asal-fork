#!/usr/bin/env bash
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LABEL=${1:-$(hostname)}
COMMIT=${FLOWLENIA_PROBE_COMMIT:-2e2152ff6d56481d922804a74b90556c39ce94cc}
RUN_DIR=${FLOWLENIA_PROBE_RUN_DIR:-$ROOT/experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization/run_006}
OUTPUT_ROOT=${FLOWLENIA_PROBE_OUTPUT_ROOT:-$ROOT/flowlenia_run006_probe}
OUTPUT_DIR=$OUTPUT_ROOT/$LABEL
PYTHON=${FLOWLENIA_PYTHON:-/home/coder/.conda/envs/torchjax/bin/python}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

mkdir -p "$OUTPUT_ROOT"
WORKTREE=$(mktemp -d /tmp/flowlenia-probe-source.XXXXXX)
RUNTIME_OVERLAY=$(mktemp -d /tmp/flowlenia-probe-runtime.XXXXXX)

cleanup() {
  git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  rm -rf "$RUNTIME_OVERLAY"
}
trap cleanup EXIT INT TERM

git -C "$ROOT" worktree add --detach "$WORKTREE" "$COMMIT"

"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "$RUNTIME_OVERLAY" \
  nvidia-cuda-nvcc-cu12==12.1.105
PTXAS=$(find "$RUNTIME_OVERLAY" -type f -name ptxas -perm -111 | head -1)
if [ -z "$PTXAS" ]; then
  echo "ptxas was not found after installing nvidia-cuda-nvcc-cu12" >&2
  exit 1
fi
PATH=$(dirname "$PTXAS"):$PATH
export PATH

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$WORKTREE"
"$PYTHON" "$ROOT/scripts/flowlenia_run006_divergence_probe.py" capture \
  --run-dir "$RUN_DIR" \
  --source-root "$WORKTREE" \
  --source-commit "$COMMIT" \
  --output-dir "$OUTPUT_DIR" &
PROBE_PID=$!
STARTED_AT=$(date +%s)
while kill -0 "$PROBE_PID" >/dev/null 2>&1; do
  sleep 60
  if kill -0 "$PROBE_PID" >/dev/null 2>&1; then
    NOW=$(date +%s)
    ELAPSED=$((NOW - STARTED_AT))
    echo "[probe] still running: elapsed=$((ELAPSED / 60))m$((ELAPSED % 60))s; expected total about 20-30m"
  fi
done
wait "$PROBE_PID"

ARCHIVE=$ROOT/flowlenia_run006_probe_${LABEL}.tar.gz
echo "[probe] packing $ARCHIVE"
tar -czf "$ARCHIVE" -C "$OUTPUT_ROOT" "$LABEL"
echo "[probe] complete"
echo "$ARCHIVE"
