#!/bin/sh
set -eu

cfg="experiments/log_apf/simulation/2602112129/config.yaml"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

echo "==> Running ${cfg}"
"${python_bin}" "scripts/simulate_save_apf.py" "${cfg}"
