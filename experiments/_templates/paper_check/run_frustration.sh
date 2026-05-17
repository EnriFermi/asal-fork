#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

python "${repo_root}/scripts/run_paper_check_frustration.py" \
  "${script_dir}/config.yaml"
