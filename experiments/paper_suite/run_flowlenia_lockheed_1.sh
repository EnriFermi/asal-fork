#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
cfg="${CFG:-experiments/paper_suite/config_flowlenia_lockheed_1.yaml}"
paper_check_cfg="${PCFG:-experiments/paper_check_flow_lenia/config_lockheed_1.yaml}"
opt_root="${OPT_ROOT:-experiments/paper_check_flow_lenia/checkpoints_lockheed_1/optimization}"
expected_opts="${EXPECTED_OPTS:-9}"

run_frustration="${RUN_FRUSTRATION:-1}"
run_apf="${RUN_APF:-1}"
run_c1="${RUN_C1:-1}"
run_c2="${RUN_C2:-1}"
run_c5="${RUN_C5:-1}"
run_visualization="${RUN_VISUALIZATION:-1}"

force_apf="${FORCE_APF:-0}"
force_branches="${FORCE_BRANCHES:-1}"
force_metrics="${FORCE_METRICS:-1}"
force_visualization="${FORCE_VISUALIZATION:-1}"
allow_heavy="${ALLOW_HEAVY:-1}"

run_py() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n "${conda_env}" "${python_bin}" "$@"
  else
    "${python_bin}" "$@"
  fi
}

run_suite() {
  run_py scripts/run_paper_suite.py "${cfg}" "$@"
}

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "Missing required directory: ${path}" >&2
    exit 2
  fi
}

require_file "${cfg}"
require_file "${paper_check_cfg}"
require_dir "${opt_root}"

best_count="$(find "${opt_root}" -mindepth 2 -maxdepth 2 -name best.pkl -print | wc -l | tr -d ' ')"
echo "Flow-Lenia lockheed_1 paper-suite recompute"
echo "  cfg=${cfg}"
echo "  paper_check_cfg=${paper_check_cfg}"
echo "  opt_root=${opt_root}"
echo "  best.pkl count=${best_count}/${expected_opts}"
echo "  conda_env=${conda_env}"

if [ "${expected_opts}" != "0" ] && [ "${best_count}" -ne "${expected_opts}" ]; then
  echo "Expected ${expected_opts} completed optimizations, found ${best_count}. Set EXPECTED_OPTS=0 to skip this check." >&2
  exit 3
fi

heavy_args=()
if [ "${allow_heavy}" = "1" ]; then
  heavy_args+=(--allow-heavy)
fi

metric_force_args=()
if [ "${force_metrics}" = "1" ]; then
  metric_force_args+=(--force)
fi

vis_force_args=()
if [ "${force_visualization}" = "1" ]; then
  vis_force_args+=(--force)
fi

branch_force_args=()
if [ "${force_branches}" = "1" ]; then
  branch_force_args+=(--force)
fi

apf_force_args=()
if [ "${force_apf}" = "1" ]; then
  apf_force_args+=(--force)
fi

if [ "${run_frustration}" = "1" ] && [ "${run_c5}" = "1" ]; then
  echo
  echo "[1/9] Flow-Lenia C5 frustration simulation/check"
  run_suite --layer simulation --task c5 "${heavy_args[@]}"
fi

if [ "${run_apf}" = "1" ]; then
  echo
  echo "[2/9] Flow-Lenia A-run APF/Lagrangian trajectories"
  run_py scripts/paper_suite_flowlenia_arun_apf.py "${cfg}" "${apf_force_args[@]}"
fi

if [ "${run_c1}" = "1" ]; then
  echo
  echo "[3/9] Flow-Lenia C1 metrics"
  run_suite --layer metrics --task c1 "${metric_force_args[@]}"
fi

if [ "${run_c5}" = "1" ]; then
  echo
  echo "[4/9] Flow-Lenia C5 metrics"
  run_suite --layer metrics --task c5 "${metric_force_args[@]}"
fi

if [ "${run_c2}" = "1" ]; then
  echo
  echo "[5/9] Flow-Lenia C2 trajectory metrics"
  run_py scripts/paper_suite_c2_flowlenia_metrics.py "${cfg}" "${metric_force_args[@]}"

  echo
  echo "[6/9] Flow-Lenia C2 event tables"
  run_py scripts/paper_suite_c2_events.py "${cfg}"

  echo
  echo "[7/9] Flow-Lenia C2 branch simulation"
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer simulation "${heavy_args[@]}" "${branch_force_args[@]}"

  echo
  echo "[8/9] Flow-Lenia C2 branch metrics: APF and CLIP-Chamfer"
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer metrics "${metric_force_args[@]}"
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer metrics --branching-metric clip_chamfer "${metric_force_args[@]}"
fi

if [ "${run_visualization}" = "1" ]; then
  echo
  echo "[9/9] Flow-Lenia visualizations"
  if [ "${run_c1}" = "1" ]; then
    run_suite --layer visualization --task c1 "${vis_force_args[@]}"
  fi
  if [ "${run_c2}" = "1" ]; then
    run_suite --layer visualization --task c2 "${vis_force_args[@]}"
  fi
  if [ "${run_c5}" = "1" ]; then
    run_suite --layer visualization --task c5 "${vis_force_args[@]}"
  fi
fi

echo
echo "Done. Main result root: analysis/results/paper_suite_flowlenia_lockheed_1"
