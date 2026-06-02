#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
  exec conda run -n "${CONDA_ENV:-onerec}" "${python_bin}" "${repo_root}/scripts/run_paper_suite.py" "${exp_dir}/config.yaml" "$@"
fi

exec "${python_bin}" "${repo_root}/scripts/run_paper_suite.py" "${exp_dir}/config.yaml" "$@"

