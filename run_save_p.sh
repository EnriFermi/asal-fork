#!/usr/bin/env bash

# Run simulation and save P snapshots with predefined parameters.

python simulate_save_p.py \
  --save_dir data/interp_supervised_0_1/interp_6 \
  --substrate lenia_flow \
  --time_sampling video \
  --rollout_steps 1000 \
  --mutations --mutation_p 0.05 --mutation_sz 40 \
  --n_seeds 16 \
  --seed_mode random_patches \
  --img_size 154 \
  --max_steps 1200000
