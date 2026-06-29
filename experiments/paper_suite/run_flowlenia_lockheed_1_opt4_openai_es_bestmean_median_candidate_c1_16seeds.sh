#!/bin/sh
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
cfg="experiments/paper_suite/config_flowlenia_lockheed_1_opt4_openai_es_bestmean_median_candidate_c1_16seeds.yaml"
source_run="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_opt4_openai_es/optimization/run_004"
selected_run="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_opt4_openai_es_bestmean_median_candidate/optimization/run_004"
result_root="analysis/results/paper_suite_flowlenia_lockheed_1_opt4_openai_es_bestmean_median_candidate_c1_16seeds"
apf_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_opt4_openai_es_bestmean_median_candidate/c1_lagrangian_apf_300k_train50_16seeds"
run_id="${PAPER_SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${PAPER_SUITE_LOG_DIR:-${result_root}/logs}"
master_log="${PAPER_SUITE_MASTER_LOG:-${log_dir}/${run_id}_master.log}"

force_export="${FORCE_EXPORT:-0}"
force_apf="${FORCE_APF:-0}"
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
  line="$(timestamp) [flowlenia-opt4-openai-median-c1] $*"
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

require_file() {
  path="$1"
  if [ ! -f "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
}

require_file "${cfg}"
require_file "${source_run}/pop_traj.pkl"

echo "Flow-Lenia lockheed_1 opt_004 OpenAI-ES best-mean-iteration median-candidate C1"
echo "  cfg=${cfg}"
echo "  source_run=${source_run}"
echo "  selected_run=${selected_run}"
echo "  apf_root=${apf_root}"
echo "  result_root=${result_root}"
echo "  rollout seeds per checkpoint=16"
echo "  random baselines=disabled"
echo "  conda_env=${conda_env}"
echo "  run_id=${run_id}"
echo "  log_dir=${log_dir}"
echo "  master_log=${master_log}"
echo "  force_export=${force_export} force_apf=${force_apf} force_metrics=${force_metrics} force_visualization=${force_visualization}"

export_force_arg=""
if [ "${force_export}" = "1" ]; then
  export_force_arg="--force"
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

echo
echo "[1/4] Export median perturbation candidate from best mean-score iteration"
run_py scripts/export_flowlenia_openai_es_median_candidate.py "${source_run}" --output-dir "${selected_run}" ${export_force_arg}

echo
echo "[2/4] Flow-Lenia dense C1 APF/Lagrangian trajectories"
run_py scripts/paper_suite_flowlenia_arun_apf.py "${cfg}" --section-key flow_lenia_arun_lagrangian_apf ${apf_force_arg}

echo
echo "[3/4] Flow-Lenia C1 metrics"
run_py scripts/run_paper_suite.py "${cfg}" --layer metrics --task c1 ${metric_force_arg}

echo
echo "[4/4] Flow-Lenia C1 visualization"
run_py scripts/run_paper_suite.py "${cfg}" --layer visualization --task c1 ${vis_force_arg}

echo
echo "Done. Candidate metadata:"
echo "  ${selected_run}/selected_candidate.json"
echo "Tables:"
echo "  ${result_root}/flow_lenia/checkpoint_scores.csv"
echo "Figures:"
echo "  ${result_root}/figures"
