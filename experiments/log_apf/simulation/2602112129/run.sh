#!/bin/sh
set -eu

cfg="$(cd "$(dirname "$0")" && pwd)/config.yaml"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

echo "==> Running ${cfg}"
"${python_bin}" "scripts/simulate_save_apf.py" "${cfg}"
