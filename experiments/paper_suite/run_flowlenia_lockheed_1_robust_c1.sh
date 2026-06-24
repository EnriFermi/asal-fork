#!/bin/sh
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
cfg="${CFG:-experiments/paper_suite/config_flowlenia_lockheed_1_robust_mspd_candidates.yaml}"
selected_root="${SELECTED_ROOT:-analysis/results/flowlenia_lockheed_1_robust_mspd_candidate_selection/selected_checkpoints}"
selected_parent="$(dirname "${selected_root}")"
random_root="${RANDOM_ROOT:-${selected_parent}/frustration_simulation/random_params}"
random_source_root="${RANDOM_SOURCE_ROOT:-experiments/paper_check_flow_lenia/checkpoints_lockheed_1/frustration_simulation/random_params}"
expected_opts="${EXPECTED_OPTS:-9}"
expected_random_baselines="${EXPECTED_RANDOM_BASELINES:-27}"
result_root="${RESULT_ROOT:-analysis/results/paper_suite_flowlenia_lockheed_1_robust_mspd_candidates}"
run_id="${PAPER_SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${PAPER_SUITE_LOG_DIR:-${result_root}/logs}"
master_log="${PAPER_SUITE_MASTER_LOG:-${log_dir}/${run_id}_master.log}"

run_apf="${RUN_APF:-1}"
run_c1="${RUN_C1:-1}"
run_visualization="${RUN_VISUALIZATION:-1}"

force_apf="${FORCE_APF:-1}"
force_metrics="${FORCE_METRICS:-1}"
force_visualization="${FORCE_VISUALIZATION:-1}"
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
  line="$(timestamp) [flowlenia-robust-c1-wrapper] $*"
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

require_file "${cfg}"
require_file "${selected_parent}/selected_candidates.csv"
require_dir "${selected_root}"

if [ ! -d "${random_root}" ]; then
  require_dir "${random_source_root}"
  echo "Isolated random baseline input root is missing; copying read-only source checkpoints:"
  echo "  from=${random_source_root}"
  echo "  to=${random_root}"
  mkdir -p "${random_root}"
  cp -R "${random_source_root}/." "${random_root}/"
fi
require_dir "${random_root}"

best_count="$(find "${selected_root}" -mindepth 2 -maxdepth 2 -name best.pkl -print | wc -l | tr -d ' ')"
random_count="$(find "${random_root}" -mindepth 3 -maxdepth 3 -name best.pkl -print | wc -l | tr -d ' ')"

echo "Flow-Lenia lockheed_1 robust MSPD candidates C1 paper-suite run"
echo "  cfg=${cfg}"
echo "  selected_root=${selected_root}"
echo "  selected best.pkl count=${best_count}/${expected_opts}"
echo "  random_root=${random_root}"
echo "  random_source_root(copy-only-if-missing)=${random_source_root}"
echo "  random best.pkl count=${random_count}/${expected_random_baselines}"
echo "  result_root=${result_root}"
echo "  c1_apf_root=experiments/paper_check_flow_lenia/checkpoints_lockheed_1_robust_mspd_candidates/arun_lagrangian_apf_300k_train50"
echo "  conda_env=${conda_env}"
echo "  run_id=${run_id}"
echo "  log_dir=${log_dir}"
echo "  master_log=${master_log}"
echo "  stdout_mode=tee conda_no_capture_output=${conda_no_capture_output} use_stdbuf=${use_stdbuf}"

if [ "${expected_opts}" != "0" ] && [ "${best_count}" -ne "${expected_opts}" ]; then
  echo "Expected ${expected_opts} selected optimized checkpoints, found ${best_count}. Set EXPECTED_OPTS=0 to skip this check." >&2
  exit 3
fi
if [ "${expected_random_baselines}" != "0" ] && [ "${random_count}" -ne "${expected_random_baselines}" ]; then
  echo "Expected ${expected_random_baselines} random baseline checkpoints, found ${random_count}. Set EXPECTED_RANDOM_BASELINES=0 to skip this check." >&2
  exit 3
fi

apf_force_arg=""
if [ "${force_apf}" = "1" ]; then
  apf_force_arg="--force"
fi

metric_force_arg=""
if [ "${force_metrics}" = "1" ]; then
  metric_force_arg="--force"
fi

vis_force_arg=""
if [ "${force_visualization}" = "1" ]; then
  vis_force_arg="--force"
fi

if [ "${run_apf}" = "1" ]; then
  echo
  echo "[1/3] Flow-Lenia robust-candidate C1 APF/Lagrangian trajectories"
  run_py scripts/paper_suite_flowlenia_arun_apf.py "${cfg}" --section-key flow_lenia_arun_lagrangian_apf ${apf_force_arg}
fi

if [ "${run_c1}" = "1" ]; then
  echo
  echo "[2/3] Flow-Lenia robust-candidate C1 metrics"
  run_suite --layer metrics --task c1 ${metric_force_arg}
fi

if [ "${run_visualization}" = "1" ] && [ "${run_c1}" = "1" ]; then
  echo
  echo "[3/3] Flow-Lenia robust-candidate C1 visualization"
  run_suite --layer visualization --task c1 ${vis_force_arg}
fi

echo
echo "Done."
echo "  figures=${result_root}/figures"
echo "  c1 tables=${result_root}/flow_lenia"
