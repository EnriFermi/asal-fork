#!/usr/bin/env bash

exp_dir="experiments/log_apf/simulation/2602072147"

for i in {1..5}; do
  cfg="${exp_dir}/config_interp_${i}.yaml"
  echo "==> Running ${cfg}"
  python "scripts/simulate_save_apf.py" "${cfg}"
done
