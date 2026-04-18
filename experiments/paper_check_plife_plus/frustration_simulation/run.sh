#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

"${python_bin}" "${repo_root}/scripts/run_paper_check_frustration.py" \
  "${repo_root}/experiments/paper_check_plife_plus/config.yaml"
