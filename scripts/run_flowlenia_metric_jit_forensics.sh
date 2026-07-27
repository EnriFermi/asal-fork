#!/usr/bin/env bash
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LABEL=${1:-$(hostname)}
COMMIT=${FLOWLENIA_PROBE_COMMIT:-2e2152ff6d56481d922804a74b90556c39ce94cc}
RUN_DIR=${FLOWLENIA_PROBE_RUN_DIR:-$ROOT/experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization/run_006}
CAPTURE_DIR=${FLOWLENIA_PROBE_CAPTURE_DIR:-$ROOT/flowlenia_run006_probe/$LABEL}
OUTPUT_ROOT=${FLOWLENIA_FORENSICS_OUTPUT_ROOT:-$ROOT/flowlenia_run006_metric_forensics}
OUTPUT_DIR=$OUTPUT_ROOT/$LABEL
PYTHON=${FLOWLENIA_PYTHON:-/home/coder/.conda/envs/torchjax/bin/python}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi
if [ ! -f "$CAPTURE_DIR/trace_xy.npy" ]; then
  echo "Missing saved trace: $CAPTURE_DIR/trace_xy.npy" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
WORKTREE=$(mktemp -d /tmp/flowlenia-forensics-source.XXXXXX)
RUNTIME_OVERLAY=$(mktemp -d /tmp/flowlenia-forensics-runtime.XXXXXX)

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

if [ "${FLOWLENIA_XLA_DUMP:-0}" = "1" ]; then
  XLA_DUMP_DIR=$OUTPUT_DIR/xla_dump
  mkdir -p "$XLA_DUMP_DIR"
  XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text --xla_gpu_dump_llvmir"
  export XLA_FLAGS
fi

cd "$WORKTREE"
"$PYTHON" "$ROOT/scripts/flowlenia_metric_jit_forensics.py" \
  --run-dir "$RUN_DIR" \
  --source-root "$WORKTREE" \
  --capture-dir "$CAPTURE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --skip-exact-execution

find "$OUTPUT_DIR" -type f ! -name files.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUTPUT_DIR/files.sha256"

ARCHIVE=$ROOT/flowlenia_run006_metric_forensics_${LABEL}.tar.gz
echo "[forensics] packing $ARCHIVE"
tar -czf "$ARCHIVE" -C "$OUTPUT_ROOT" "$LABEL"
echo "[forensics] complete"
echo "$ARCHIVE"
