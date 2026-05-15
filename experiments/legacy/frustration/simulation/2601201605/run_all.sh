#!/usr/bin/env bash

for n in 2 3 4 5; do
  echo "Running split ${n}x${n} ..."
  GRID_SPLIT="$n" python "scripts/simulate_frustration.py" \
    "experiments/legacy/frustration/simulation/2601201605/config.yaml"
done
