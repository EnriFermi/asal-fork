#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/scripts/main_opt_halving.py" \
  "${ROOT_DIR}/experiments/opt_halving/config.yaml"
