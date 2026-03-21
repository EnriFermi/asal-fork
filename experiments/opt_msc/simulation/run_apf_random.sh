#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cfg="${CFG:-${exp_dir}/config_apf_random.yaml}"
random_init_mode="sep_cma_es_ask"
cma_sigma_init="0.2"
cma_pop_size="8"
n_runs="3"

WANDB_MODE="${WANDB_MODE:-disabled}" \
  "${python_bin}" "scripts/generate_random_best.py" "${cfg}" --run-batch \
    "random_best.init_mode=${random_init_mode}" \
    "random_best.cma_sigma_init=${cma_sigma_init}" \
    "random_best.cma_pop_size=${cma_pop_size}" \
    "random_batch.n_runs=${n_runs}"
