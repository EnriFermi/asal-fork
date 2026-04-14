#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${exp_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cd "${repo_root}"

cfg="${CFG:-${exp_dir}/config.yaml}"
jit_microbatch="${JIT_MICROBATCH:-1}"
batch_steps="${BATCH_STEPS:-64}"
img_size="${IMG_SIZE:-160}"
max_steps="${MAX_STEPS:-20000}"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "${repo_root}/scripts/simulate_after_training.py" "${cfg}" \
    "simulation.jit_microbatch=${jit_microbatch}" \
    "simulation.batch_steps=${batch_steps}" \
    "simulation.img_size=${img_size}" \
    "simulation.max_steps=${max_steps}"
