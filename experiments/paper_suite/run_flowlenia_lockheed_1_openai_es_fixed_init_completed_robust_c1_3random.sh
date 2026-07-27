#!/bin/sh
set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
source_optimization_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization"
selected_optimization_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/optimization"
generated_cfg="experiments/paper_suite/config_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random.yaml"
result_root="${RESULT_ROOT:-analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random}"
apf_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/c1_lagrangian_apf_300k_train50_4seeds_replay_fixed_opt_native"
random_checkpoint_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1/frustration_simulation/random_params"
random_checkpoint_selection="optimization_iter0"
run_id="${PAPER_SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${PAPER_SUITE_LOG_DIR:-${result_root}/logs}"
master_log="${PAPER_SUITE_MASTER_LOG:-${log_dir}/${run_id}_master.log}"

force_prep_export="${FORCE_PREP_EXPORT:-0}"
run_preflight="${RUN_PREFLIGHT:-1}"
preflight_rollout_steps="${PREFLIGHT_ROLLOUT_STEPS:-200}"
c1_rollout_seeds="${C1_ROLLOUT_SEEDS:-4}"
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
  line="$(timestamp) [flowlenia-fixed-init-completed-robust-c1-3random] $*"
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

require_dir() {
  path="$1"
  if [ ! -d "${path}" ]; then
    echo "Missing required directory: ${path}" >&2
    exit 2
  fi
}

require_dir "${source_optimization_root}"
if [ "${random_checkpoint_selection}" != "optimization_iter0" ]; then
  require_dir "${random_checkpoint_root}"
fi

case "${apf_root}" in
  *replay_fixed*) ;;
  *)
    echo "Refusing to run: apf_root must be an isolated replay_fixed root, got ${apf_root}" >&2
    exit 2
    ;;
esac

prep_force_arg=""
if [ "${force_prep_export}" = "1" ]; then
  prep_force_arg="--force-export"
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

echo "Flow-Lenia lockheed_1 OpenAI-ES fixed-init completed-runs robust C1, 3 random per opt"
echo "  source_optimization_root=${source_optimization_root}"
echo "  selected_optimization_root=${selected_optimization_root}"
echo "  generated_cfg=${generated_cfg}"
echo "  result_root=${result_root}"
echo "  apf_root=${apf_root}"
echo "  random_checkpoint_root=${random_checkpoint_root}"
echo "  random_checkpoint_selection=${random_checkpoint_selection}"
echo "  anti_noise=robust_pioneer_lcb_in_top_trend"
echo "  lcb_z=2.0 trend_quantile=90 ewma_beta=0.85 trim_frac=0.125"
echo "  c1_rollout_seeds=${c1_rollout_seeds}"
echo "  random_baselines_per_opt=3"
echo "  conda_env=${conda_env}"
echo "  run_id=${run_id}"
echo "  log_dir=${log_dir}"
echo "  master_log=${master_log}"
echo "  force_prep_export=${force_prep_export} force_apf=${force_apf} force_metrics=${force_metrics} force_visualization=${force_visualization}"
echo "  run_preflight=${run_preflight} preflight_rollout_steps=${preflight_rollout_steps}"

echo
echo "[1/8] Discover completed runs, export robust candidates, write C1 config"
run_py scripts/prepare_flowlenia_completed_robust_c1.py \
  --source-optimization-root "${source_optimization_root}" \
  --selected-optimization-root "${selected_optimization_root}" \
  --output-config "${generated_cfg}" \
  --result-root "${result_root}" \
  --apf-root "${apf_root}" \
  --random-checkpoint-root "${random_checkpoint_root}" \
  --random-checkpoint-selection "${random_checkpoint_selection}" \
  --n-rollout-seeds "${c1_rollout_seeds}" \
  --num-random-baselines 3 \
  --batch-size 8 \
  --pair-seed-base 400003 \
  --lcb-z 2.0 \
  --trend-quantile 90 \
  --ewma-beta 0.85 \
  --trim-frac 0.125 \
  --exclude-source-run 8 \
  --legacy-optimization-sigma-collision \
  ${prep_force_arg}

echo
echo "[2/8] Static fail-fast guard: config, selected candidates, seeds, random checkpoints"
run_py scripts/check_flowlenia_c1_replay_preflight.py \
  --config "${generated_cfg}" \
  --selected-root "${selected_optimization_root}" \
  --source-optimization-root "${source_optimization_root}" \
  --skip-smoke \
  --skip-existing-results \
  --require-apf-root-contains "replay_fixed" \
  --summary-json "${result_root}/logs/${run_id}_preflight_static_guard.json"

echo
echo "[3/8] APF dry-run guard: validate trajectory plan without heavy simulation"
run_py scripts/paper_suite_flowlenia_arun_apf.py "${generated_cfg}" --section-key flow_lenia_arun_lagrangian_apf --dry-run

echo
echo "[4/8] Short replay smoke before heavy APF"
if [ "${run_preflight}" = "1" ]; then
  run_py scripts/check_flowlenia_c1_replay_preflight.py \
    --config "${generated_cfg}" \
    --selected-root "${selected_optimization_root}" \
    --source-optimization-root "${source_optimization_root}" \
    --skip-existing-results \
    --rollout-steps "${preflight_rollout_steps}" \
    --output-root "${result_root}/preflight_smoke/${run_id}" \
    --require-apf-root-contains "replay_fixed" \
    --allow-known-execution-divergence \
    --summary-json "${result_root}/logs/${run_id}_preflight_smoke_before_apf.json"
else
  echo "  skipped because RUN_PREFLIGHT=${run_preflight}"
fi

echo
echo "[5/8] Flow-Lenia C1 APF/Lagrangian trajectories"
run_py scripts/paper_suite_flowlenia_arun_apf.py "${generated_cfg}" --section-key flow_lenia_arun_lagrangian_apf ${apf_force_arg}

echo
echo "[6/8] Flow-Lenia C1 metrics"
run_py scripts/run_paper_suite.py "${generated_cfg}" --layer metrics --task c1 ${metric_force_arg}

echo
echo "[7/8] Post-metrics guard: table protocol audit"
if [ "${run_preflight}" = "1" ]; then
  run_py scripts/check_flowlenia_c1_replay_preflight.py \
    --config "${generated_cfg}" \
    --selected-root "${selected_optimization_root}" \
    --source-optimization-root "${source_optimization_root}" \
    --scores-csv "${result_root}/flow_lenia/checkpoint_scores.csv" \
    --skip-smoke \
    --require-apf-root-contains "replay_fixed" \
    --summary-json "${result_root}/logs/${run_id}_preflight_after_metrics.json"
else
  echo "  skipped because RUN_PREFLIGHT=${run_preflight}"
fi

echo
echo "[8/8] Flow-Lenia C1 visualization"
run_py scripts/run_paper_suite.py "${generated_cfg}" --layer visualization --task c1 ${vis_force_arg}

echo
echo "Done."
echo "Config:"
echo "  ${generated_cfg}"
echo "Selection summary:"
echo "  experiments/paper_suite/generated_manifests/$(basename "${generated_cfg}" .yaml)/summary.json"
echo "Tables:"
echo "  ${result_root}/flow_lenia/checkpoint_scores.csv"
echo "  ${result_root}/flow_lenia/group_contrasts.csv"
echo "Figures:"
echo "  ${result_root}/figures"
