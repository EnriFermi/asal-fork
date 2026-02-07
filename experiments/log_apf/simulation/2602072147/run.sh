#!/bin/sh
set -eu

exp_dir="experiments/log_apf/simulation/2602072147"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

i=1
while [ "${i}" -le 5 ]; do
  cfg="${exp_dir}/config_interp_${i}.yaml"
  echo "==> Running ${cfg}"
  "${python_bin}" "scripts/simulate_save_apf.py" "${cfg}"
  i=$((i + 1))
done
