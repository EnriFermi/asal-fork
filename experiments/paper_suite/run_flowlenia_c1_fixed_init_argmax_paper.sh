#!/usr/bin/env bash
set -eu

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

mode="${MODE:-pilot}"
case "$mode" in
  pilot|full) ;;
  *) echo "MODE must be pilot or full, got: $mode" >&2; exit 2 ;;
esac

conda_env="${CONDA_ENV:-torchjax}"
python_bin="${PYTHON_BIN:-python}"
source_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization"
campaign_root="experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper"
selected_root="${campaign_root}/optimization"
apf_root="${campaign_root}/c1_lagrangian_apf_300k_train50_4seeds_exact_parallel_zip"
full_result_root="analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper"

if [ "$mode" = "pilot" ]; then
  config="experiments/paper_suite/config_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper_pilot_run000.yaml"
  result_root="${full_result_root}_pilot_run000"
  include_args="--include-source-run 0"
else
  config="experiments/paper_suite/config_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper.yaml"
  result_root="$full_result_root"
  include_args="--include-source-run 0 --include-source-run 1 --include-source-run 2 --include-source-run 3 --include-source-run 4 --include-source-run 5 --include-source-run 6 --include-source-run 7 --include-source-run 8 --include-source-run 9"
fi

run_id="${PAPER_SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${result_root}/logs"
master_log="${log_dir}/${run_id}_master.log"
mkdir -p "$log_dir"
exec > >(tee -a "$master_log") 2>&1

run_py() {
  conda run --no-capture-output -n "$conda_env" "$python_bin" "$@"
}

echo "Flow-Lenia C1 fixed-init paper replay"
echo "  mode=$mode"
echo "  candidate_selection=global observed-MSPD argmax"
echo "  source_runs=$([ "$mode" = pilot ] && echo 0 || echo 0..9)"
echo "  APF replay=optimizer-native 8x4 batch, saved-APF posthoc"
echo "  random=3 original random-distribution theta per group in matched optimizer context"
echo "  exact_train_mspd_required=true"
echo "  config=$config"
echo "  apf_root=$apf_root"
echo "  result_root=$result_root"
echo "  master_log=$master_log"

echo "[1/6] Export argmax candidates and generate isolated C1 config"
# shellcheck disable=SC2086
run_py scripts/prepare_flowlenia_completed_robust_c1.py \
  --source-optimization-root "$source_root" \
  --selected-optimization-root "$selected_root" \
  --output-config "$config" \
  --result-root "$result_root" \
  --apf-root "$apf_root" \
  --random-checkpoint-root experiments/paper_check_flow_lenia/checkpoints_lockheed_1/frustration_simulation/random_params \
  --random-checkpoint-selection per_source_group_optimizer_context \
  --n-rollout-seeds 4 \
  --num-random-baselines 3 \
  --batch-size 8 \
  --pair-seed-base 400003 \
  --candidate-selection-rule argmax \
  --posthoc-replay-source apf \
  --require-exact-train-mspd \
  --optimizer-reference-cross-hardware-source-run 5 \
  --optimizer-reference-cross-hardware-source-run 6 \
  --optimizer-reference-cross-hardware-source-run 7 \
  --optimizer-reference-cross-hardware-source-run 8 \
  --optimizer-reference-cross-hardware-max-ulps 4 \
  --apf-flush-workers 16 \
  --legacy-optimization-sigma-collision \
  --legacy-optimization-sigma-collision-except-run 3 \
  --legacy-optimization-sigma-collision-except-run 9 \
  $include_args

echo "[2/6] Static protocol guard and APF execution-plan dry run"
run_py scripts/check_flowlenia_c1_replay_preflight.py \
  --config "$config" \
  --selected-root "$selected_root" \
  --source-optimization-root "$source_root" \
  --skip-smoke \
  --skip-existing-results \
  --require-apf-root-contains "argmax_paper" \
  --summary-json "${log_dir}/${run_id}_preflight_static.json"
run_py scripts/paper_suite_flowlenia_arun_apf.py "$config" \
  --section-key flow_lenia_arun_lagrangian_apf \
  --dry-run

echo "[3/6] Exact optimizer-context APF trajectories and videos"
run_py scripts/paper_suite_flowlenia_arun_apf.py "$config" \
  --section-key flow_lenia_arun_lagrangian_apf

echo "[4/6] C1 MSPD metrics; validates bit-exact or configured cross-hardware ULP replay"
run_py scripts/run_paper_suite.py "$config" --layer metrics --task c1 --force

echo "[5/6] Post-metric protocol and exact-MSPD audit"
run_py scripts/check_flowlenia_c1_replay_preflight.py \
  --config "$config" \
  --selected-root "$selected_root" \
  --source-optimization-root "$source_root" \
  --scores-csv "${result_root}/flow_lenia/checkpoint_scores.csv" \
  --skip-smoke \
  --require-apf-root-contains "argmax_paper" \
  --summary-json "${log_dir}/${run_id}_preflight_after_metrics.json"

echo "[6/6] All C1 paper-suite visualizations"
run_py scripts/run_paper_suite.py "$config" --layer visualization --task c1 --force

echo "Done mode=$mode"
echo "Tables: ${result_root}/flow_lenia"
echo "Figures: ${result_root}/figures"
echo "Videos/APF: ${apf_root}"
