#!/usr/bin/env bash
set -euo pipefail

# Launch data-parallel CMA-ES (main_opt_parallel.py) mirroring the classic main_opt defaults.
# Adjust pop_size to be divisible by the number of local devices.

python3 scripts/main_opt_parallel.py --seed=42 --save_dir="./data/test_for_performance" --substrate="lenia_flow" \
  --time_sampling=16 --coef_prompt=0. --coef_softmax=0. \
  --coef_oe=1. --bs=1 --rollout_steps=8192 --pop_size=8 \
  --n_iters=500 --sigma=0.2 --mutations --mutation_p=0.1 \
  --mutation_sz=40 --seed_n_patches=20 --seed_mode="random_patches" \
  "$@"
