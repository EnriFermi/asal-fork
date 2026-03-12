#!/bin/sh
set -eu

exp_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python"
fi

input_dir="${INPUT_DIR:-experiments/opt_msc/checkpoints/test_run_longer/apf_logs}"
output="${OUTPUT:-experiments/opt_msc/figures/test_run_longer.mp4}"
start_step="${START_STEP:-}"
end_step="${END_STEP:-}"
snapshot_stride="${SNAPSHOT_STRIDE:-1}"
fps="${FPS:-}"

cmd="\"${python_bin}\" scripts/render_apf_video.py --input_dir \"${input_dir}\" --output \"${output}\" --snapshot_stride \"${snapshot_stride}\""

if [ -n "${start_step}" ]; then
  cmd="${cmd} --start_step \"${start_step}\""
fi
if [ -n "${end_step}" ]; then
  cmd="${cmd} --end_step \"${end_step}\""
fi
if [ -n "${fps}" ]; then
  cmd="${cmd} --fps \"${fps}\""
fi

eval "${cmd}"
