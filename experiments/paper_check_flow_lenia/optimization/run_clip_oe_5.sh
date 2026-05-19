#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

export PAPER_NUM_OPTIMIZATIONS="${PAPER_NUM_OPTIMIZATIONS:-5}"
export PAPER_MACHINE_IDX="${PAPER_MACHINE_IDX:-0}"
export PAPER_NUM_MACHINES="${PAPER_NUM_MACHINES:-1}"

config_path="${repo_root}/experiments/paper_check_flow_lenia/config_clip_oe.yaml"
python_script="${repo_root}/scripts/run_paper_check_optimization.py"

echo "[clip_oe] num_optimizations=${PAPER_NUM_OPTIMIZATIONS}"
echo "[clip_oe] machine_idx=${PAPER_MACHINE_IDX} num_machines=${PAPER_NUM_MACHINES}"
echo "[clip_oe] config=${config_path}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  "${PYTHON_BIN}" "${python_script}" "${config_path}"
elif command -v conda >/dev/null 2>&1; then
  conda run -n "${CONDA_ENV:-onerec}" python "${python_script}" "${config_path}"
else
  python "${python_script}" "${config_path}"
fi
