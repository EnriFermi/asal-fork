#!/usr/bin/env bash
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${FLOWLENIA_PYTHON:-/home/coder/.conda/envs/torchjax/bin/python}
CONFIG=$ROOT/experiments/paper_check_flow_lenia/config_lockheed_1_openai_es_fixed_init_opt003.yaml
LOCK=$ROOT/experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization/.run_003.lock

if [ ! -x "$PYTHON" ]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Protocol config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "run_003 is already running (lock: $LOCK)" >&2
  exit 1
fi

RUNTIME_OVERLAY=$(mktemp -d /tmp/flowlenia-opt003-runtime.XXXXXX)
cleanup() {
  rm -rf "$RUNTIME_OVERLAY"
}
trap cleanup EXIT INT TERM

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
export PYTHONUNBUFFERED=1

echo "[opt003] source_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "[opt003] ptxas=$($PTXAS --version | tail -1)"
echo "[opt003] config=$CONFIG"
cd "$ROOT"
"$PYTHON" scripts/run_paper_check_optimization.py "$CONFIG"
