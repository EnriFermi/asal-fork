#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

cfg="${CFG:-${exp_dir}/config.yaml}"
n_inits="${N_INITS:-5}"
rollout_steps="${ROLLOUT_STEPS:-300000}"
output_root="${OUTPUT_ROOT:-experiments/opt_msc/checkpoints/random_video_batch_300k}"
param_seed_start="${PARAM_SEED_START:-3}"
sim_seed_start="${SIM_SEED_START:-2}"
overwrite="${OVERWRITE:-0}"
wandb_mode="${WANDB_MODE:-disabled}"
jax_platforms="${JAX_PLATFORMS:-}"
xla_preallocate="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [ -n "${jax_platforms}" ]; then
  set -- \
    env JAX_PLATFORMS="${jax_platforms}" XLA_PYTHON_CLIENT_PREALLOCATE="${xla_preallocate}" \
    "${python_bin}" "scripts/simulate_random_flowlenia_video_batch.py" "${cfg}" \
    --n-inits "${n_inits}" \
    --rollout-steps "${rollout_steps}" \
    --output-root "${output_root}" \
    --param-seed-start "${param_seed_start}" \
    --sim-seed-start "${sim_seed_start}" \
    --wandb-mode "${wandb_mode}"
else
  set -- \
    env XLA_PYTHON_CLIENT_PREALLOCATE="${xla_preallocate}" \
    "${python_bin}" "scripts/simulate_random_flowlenia_video_batch.py" "${cfg}" \
    --n-inits "${n_inits}" \
    --rollout-steps "${rollout_steps}" \
    --output-root "${output_root}" \
    --param-seed-start "${param_seed_start}" \
    --sim-seed-start "${sim_seed_start}" \
    --wandb-mode "${wandb_mode}"
fi

if [ "${overwrite}" = "1" ]; then
  set -- "$@" --overwrite
fi

"$@"
