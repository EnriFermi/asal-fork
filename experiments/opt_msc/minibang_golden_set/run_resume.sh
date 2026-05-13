#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cd "${repo_root}"
"${python_bin}" scripts/flowlenia_minibang_resume.py "$@"
