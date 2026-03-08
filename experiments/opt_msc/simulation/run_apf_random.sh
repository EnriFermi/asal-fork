#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cfg="${exp_dir}/config_apf_random.yaml"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "scripts/generate_random_best.py" "${cfg}"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "scripts/simulate_save_apf.py" "${cfg}"
