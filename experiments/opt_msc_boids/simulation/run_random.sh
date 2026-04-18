#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cd "${repo_root}"

cfg="${CFG:-${exp_dir}/config_random.yaml}"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "${repo_root}/scripts/generate_random_best.py" "${cfg}"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "${repo_root}/scripts/simulate_after_training.py" "${cfg}"
