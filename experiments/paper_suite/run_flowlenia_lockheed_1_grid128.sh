#!/bin/sh
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
cfg="${FLOWLENIA_LOCKHEED_GRID128_CFG:-experiments/paper_suite/config_flowlenia_lockheed_1_grid128.yaml}"
paper_check_cfg="${FLOWLENIA_LOCKHEED_GRID128_PCFG:-experiments/paper_check_flow_lenia/config_lockheed_1_grid128.yaml}"
opt_root="${OPT_ROOT:-experiments/paper_check_flow_lenia/checkpoints_lockheed_1_grid128_eval/optimization}"
opt_source_root="${OPT_SOURCE_ROOT:-experiments/paper_check_flow_lenia/checkpoints_lockheed_1/optimization}"
random_root="${RANDOM_ROOT:-$(dirname "${opt_root}")/frustration_simulation/random_params}"
random_source_root="${RANDOM_SOURCE_ROOT:-$(dirname "${opt_source_root}")/frustration_simulation/random_params}"
expected_opts="${EXPECTED_OPTS:-9}"
expected_random_baselines="${EXPECTED_RANDOM_BASELINES:-27}"
result_root="${RESULT_ROOT:-analysis/results/paper_suite_flowlenia_lockheed_1_grid128_eval}"
run_id="${PAPER_SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${PAPER_SUITE_LOG_DIR:-${result_root}/logs}"
master_log="${PAPER_SUITE_MASTER_LOG:-${log_dir}/${run_id}_master.log}"

run_frustration="${RUN_FRUSTRATION:-1}"
run_apf="${RUN_APF:-1}"
run_c1="${RUN_C1:-1}"
run_c2="${RUN_C2:-1}"
run_c5="${RUN_C5:-1}"
run_visualization="${RUN_VISUALIZATION:-1}"

force_apf="${FORCE_APF:-1}"
force_branches="${FORCE_BRANCHES:-1}"
force_metrics="${FORCE_METRICS:-1}"
force_visualization="${FORCE_VISUALIZATION:-1}"
allow_heavy="${ALLOW_HEAVY:-1}"
conda_no_capture_output="${CONDA_NO_CAPTURE_OUTPUT:-1}"
use_stdbuf="${USE_STDBUF:-1}"
command_counter=0

mkdir -p "${log_dir}"
export PAPER_SUITE_RUN_ID="${run_id}"
export PAPER_SUITE_LOG_DIR="${log_dir}"
export PAPER_SUITE_MASTER_LOG="${master_log}"
export PAPER_SUITE_LOG_PROGRESS="${PAPER_SUITE_LOG_PROGRESS:-plain}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

safe_log_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g' | sed 's/^__*/command/; s/_*$//'
}

log_line() {
  line="$(timestamp) [flowlenia-lockheed-wrapper] $*"
  echo "${line}"
  printf '%s\n' "${line}" >> "${master_log}"
}

run_stdout() {
  if [ "${use_stdbuf}" = "1" ] && command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$@"
  else
    "$@"
  fi
}

run_py() {
  command_counter=$((command_counter + 1))
  script_name="$(basename "$1" .py)"
  command_log="${log_dir}/${run_id}_wrapper_$(printf '%03d' "${command_counter}")_$(safe_log_name "${script_name}").log"
  status_file="${command_log}.status"
  rm -f "${status_file}"

  if command -v conda >/dev/null 2>&1; then
    if [ "${conda_no_capture_output}" = "1" ]; then
      log_line "command start log=${command_log} cmd=conda run --no-capture-output -n ${conda_env} ${python_bin} $*"
      (
        set +e
        run_stdout conda run --no-capture-output -n "${conda_env}" "${python_bin}" "$@" 2>&1
        status=$?
        printf '%s\n' "${status}" > "${status_file}"
        exit 0
      ) | tee -a "${command_log}"
    else
      log_line "command start log=${command_log} cmd=conda run -n ${conda_env} ${python_bin} $*"
      (
        set +e
        run_stdout conda run -n "${conda_env}" "${python_bin}" "$@" 2>&1
        status=$?
        printf '%s\n' "${status}" > "${status_file}"
        exit 0
      ) | tee -a "${command_log}"
    fi
  else
    log_line "command start log=${command_log} cmd=${python_bin} $*"
    (
      set +e
      run_stdout "${python_bin}" "$@" 2>&1
      status=$?
      printf '%s\n' "${status}" > "${status_file}"
      exit 0
    ) | tee -a "${command_log}"
  fi

  status="$(cat "${status_file}" 2>/dev/null || printf '1')"
  rm -f "${status_file}"
  if [ "${status}" -ne 0 ]; then
    log_line "command failed status=${status} log=${command_log}"
    exit "${status}"
  fi
  log_line "command done log=${command_log}"
}

run_suite() {
  run_py scripts/run_paper_suite.py "${cfg}" "$@"
}

require_file() {
  path="$1"
  if [ ! -f "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  path="$1"
  if [ ! -d "${path}" ]; then
    echo "Missing required directory: ${path}" >&2
    exit 2
  fi
}

if [ "${CFG+x}" = "x" ] && [ "${FLOWLENIA_LOCKHEED_GRID128_CFG+x}" != "x" ]; then
  echo "Ignoring generic CFG=${CFG}; use FLOWLENIA_LOCKHEED_GRID128_CFG to override this wrapper." >&2
fi
if [ "${PCFG+x}" = "x" ] && [ "${FLOWLENIA_LOCKHEED_GRID128_PCFG+x}" != "x" ]; then
  echo "Ignoring generic PCFG=${PCFG}; use FLOWLENIA_LOCKHEED_GRID128_PCFG to override this wrapper." >&2
fi
case "${cfg}" in
  *grid128*) ;;
  *)
    echo "Refusing non-grid128 paper-suite cfg in grid128 wrapper: ${cfg}" >&2
    exit 4
    ;;
esac
case "${paper_check_cfg}" in
  *grid128*) ;;
  *)
    echo "Refusing non-grid128 paper-check cfg in grid128 wrapper: ${paper_check_cfg}" >&2
    exit 4
    ;;
esac

require_file "${cfg}"
require_file "${paper_check_cfg}"
if [ ! -d "${opt_root}" ]; then
  require_dir "${opt_source_root}"
  echo "Isolated optimization input root is missing; copying read-only source checkpoints:"
  echo "  from=${opt_source_root}"
  echo "  to=${opt_root}"
  mkdir -p "${opt_root}"
  cp -R "${opt_source_root}/." "${opt_root}/"
fi
require_dir "${opt_root}"

if [ ! -d "${random_root}" ]; then
  require_dir "${random_source_root}"
  echo "Isolated random baseline input root is missing; copying read-only source checkpoints:"
  echo "  from=${random_source_root}"
  echo "  to=${random_root}"
  mkdir -p "${random_root}"
  cp -R "${random_source_root}/." "${random_root}/"
fi
require_dir "${random_root}"

best_count="$(find "${opt_root}" -mindepth 2 -maxdepth 2 -name best.pkl -print | wc -l | tr -d ' ')"
random_count="$(find "${random_root}" -mindepth 3 -maxdepth 3 -name best.pkl -print | wc -l | tr -d ' ')"
echo "Flow-Lenia lockheed_1 grid128 paper-suite recompute"
echo "  cfg=${cfg}"
echo "  paper_check_cfg=${paper_check_cfg}"
echo "  isolated_opt_root=${opt_root}"
echo "  opt_source_root(copy-only-if-missing)=${opt_source_root}"
echo "  best.pkl count=${best_count}/${expected_opts}"
echo "  isolated_random_root=${random_root}"
echo "  random_source_root(copy-only-if-missing)=${random_source_root}"
echo "  random best.pkl count=${random_count}/${expected_random_baselines}"
echo "  conda_env=${conda_env}"
echo "  run_id=${run_id}"
echo "  log_dir=${log_dir}"
echo "  master_log=${master_log}"
echo "  stdout_mode=tee conda_no_capture_output=${conda_no_capture_output} use_stdbuf=${use_stdbuf}"

if [ "${expected_opts}" != "0" ] && [ "${best_count}" -ne "${expected_opts}" ]; then
  echo "Expected ${expected_opts} completed optimizations, found ${best_count}. Set EXPECTED_OPTS=0 to skip this check." >&2
  exit 3
fi
if [ "${expected_random_baselines}" != "0" ] && [ "${random_count}" -ne "${expected_random_baselines}" ]; then
  echo "Expected ${expected_random_baselines} random baseline checkpoints, found ${random_count}. Set EXPECTED_RANDOM_BASELINES=0 to skip this check." >&2
  exit 3
fi

heavy_arg=""
if [ "${allow_heavy}" = "1" ]; then
  heavy_arg="--allow-heavy"
fi

metric_force_arg=""
if [ "${force_metrics}" = "1" ]; then
  metric_force_arg="--force"
fi

vis_force_arg=""
if [ "${force_visualization}" = "1" ]; then
  vis_force_arg="--force"
fi

branch_force_arg=""
if [ "${force_branches}" = "1" ]; then
  branch_force_arg="--force"
fi

apf_force_arg=""
if [ "${force_apf}" = "1" ]; then
  apf_force_arg="--force"
fi

if [ "${run_apf}" = "1" ]; then
  echo
  echo "[1/10] Flow-Lenia A-run APF/Lagrangian trajectories for C1/C2"
  run_py scripts/paper_suite_flowlenia_arun_apf.py "${cfg}" ${apf_force_arg}
fi

if [ "${run_c1}" = "1" ]; then
  echo
  echo "[2/10] Flow-Lenia C1 metrics"
  run_suite --layer metrics --task c1 ${metric_force_arg}
fi

if [ "${run_visualization}" = "1" ] && [ "${run_c1}" = "1" ]; then
  echo
  echo "[3/10] Flow-Lenia C1 visualization"
  run_suite --layer visualization --task c1 ${vis_force_arg}
fi

if [ "${run_frustration}" = "1" ] && [ "${run_c5}" = "1" ]; then
  echo
  echo "[4/10] Flow-Lenia C5 frustration simulation/check"
  run_py scripts/run_paper_check_frustration.py "${paper_check_cfg}"
fi

if [ "${run_c5}" = "1" ]; then
  echo
  echo "[5/10] Flow-Lenia C5 metrics"
  run_suite --layer metrics --task c5 ${metric_force_arg}
fi

if [ "${run_c2}" = "1" ]; then
  echo
  echo "[6/10] Flow-Lenia C2 trajectory metrics"
  run_py scripts/paper_suite_c2_flowlenia_metrics.py "${cfg}" ${metric_force_arg}

  echo
  echo "[7/10] Flow-Lenia C2 event tables"
  run_py scripts/paper_suite_c2_events.py "${cfg}"

  echo
  echo "[8/10] Flow-Lenia C2 branch simulation"
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer simulation ${heavy_arg} ${branch_force_arg}

  echo
  echo "[9/10] Flow-Lenia C2 branch metrics: APF and CLIP-Chamfer"
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer metrics ${metric_force_arg}
  run_py scripts/paper_suite_c2_branching.py "${cfg}" --layer metrics --branching-metric clip_chamfer ${metric_force_arg}
fi

if [ "${run_visualization}" = "1" ]; then
  echo
  echo "[10/10] Flow-Lenia remaining visualizations"
  if [ "${run_c2}" = "1" ]; then
    run_suite --layer visualization --task c2 ${vis_force_arg}
  fi
  if [ "${run_c5}" = "1" ]; then
    run_suite --layer visualization --task c5 ${vis_force_arg}
  fi
fi

echo
echo "Done. Main result root: analysis/results/paper_suite_flowlenia_lockheed_1_grid128_eval"
