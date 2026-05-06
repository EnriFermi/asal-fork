#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

if [ "$#" -gt 0 ]; then
  dataset_root="$1"
  shift
else
  dataset_root="experiments/opt_msc/checkpoints/test_run_longrun_check/minibang_golden_set"
fi

cd "${repo_root}"
"${python_bin}" scripts/flowlenia_minibang_detect.py "${dataset_root}" "$@"
