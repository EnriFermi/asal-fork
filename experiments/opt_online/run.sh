#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/scripts/main_opt_online.py" \
  "${ROOT_DIR}/experiments/opt_online/config.yaml"
