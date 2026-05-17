#!/usr/bin/env bash
set -eu
if ( set -o pipefail 2>/dev/null ); then
  set -o pipefail
fi


for i in $(seq 6 -1 1); do
  python scripts/simulate_save_p.py \
    --save_dir "data/interp_supervised_0_1/interp_${i}" \
    --substrate lenia_flow \
    --time_sampling video \
    --rollout_steps 1200000 \
    --mutations --mutation_p 0.05 --mutation_sz 40 \
    --n_seeds 16 \
    --seed_mode "random_patches" \
    --img_size 154 \
    --max_steps 1200000 \
    --snapshot_interval 256 \
    --jit_microbatch 64 \
    --save_A \
    --seed=0 \
    --save_rgb
done