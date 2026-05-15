#!/usr/bin/env bash
set -e

# Adjust this to your training run directory
SAVE_DIR="./data/supervised_pca_track"
SUBSTRATE="lenia_flow"
OUT_DIR="videos_lenia_flow"
FPS=200
MAX_STEPS=1500000  # or whatever you want

mkdir -p "$OUT_DIR"

for ITER in 0, 99; do
  echo "Running trajectory from iteration $ITER ..."
  python scripts/simulate_after_training.py \
    --save_dir "$SAVE_DIR" \
    --substrate "$SUBSTRATE" \
    --time_sampling video \
    --output "${OUT_DIR}/lenia_flow_iter${ITER}.mp4" \
    --fps "$FPS" \
    --max_steps "$MAX_STEPS" \
    --traj_iter "$ITER" \
    --img_size=154 \
    --n_seeds=16 \
    --mutations \
    --mutation_sz=40 \
    --mutation_p=0.05 \
    --seed=0 \
    --seed_mode="random_patches"
done
