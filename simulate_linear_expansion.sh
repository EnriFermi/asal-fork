#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="./data/linear_expansion"
OUT_DIR="./data/linear_expansion/videos"
SUBSTRATE="lenia_flow"
FPS=250
MAX_STEPS=1200000
ROLLOUT_STEPS=1200000

mkdir -p "$OUT_DIR"

for DIR in "$BASE_DIR"/scale_*; do
  if [[ ! -d "$DIR" ]]; then
    continue
  fi
  NAME="$(basename "$DIR")"
  echo "Running simulation for $NAME ..."
  python simulate_after_training.py \
    --save_dir "$DIR" \
    --substrate "$SUBSTRATE" \
    --rollout_steps "$ROLLOUT_STEPS" \
    --seed 0 \
    --n_seeds 16 \
    --img_size 154 \
    --seed_mode random_patches \
    --mutations --mutation_p 0.05 --mutation_sz 40 \
    --max_steps "$MAX_STEPS" \
    --output "${OUT_DIR}/${NAME}.mp4" \
    --fps "$FPS"
done
