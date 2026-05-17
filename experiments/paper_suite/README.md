# MSPD Paper Suite Runbook

Инструкция для запуска paper-suite экспериментов из `experiments/paper_suite/config.yaml`.

Запускать из корня репозитория:

```bash
cd /Users/enrifermi/Projects/asal-fork
```

Окружение:

```bash
conda run -n onerec python --version
```

Главный entry point:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all
```

## Главная идея

Suite разделен на три слоя:

- `simulation`: создает или проверяет raw artifacts. Тяжелые реальные симуляции не запускаются без `--allow-heavy`.
- `metrics`: считает MSPD/posthoc/statistical tables только из уже сохраненных artifacts.
- `visualization`: строит картинки только из готовых metrics tables.

Это сделано специально, чтобы после APF/trajectory logging можно было пересчитывать метрики и графики без пересимуляции.

Claims:

- C1: optimized systems vs matched random controls.
- C2: Delta-H marks transition-sensitive states.
- C5: blockwise frustration / history dependence.
- C6: transfer to Particle Life++; Boids secondary.
- N0: mandatory synthetic calibration `S0/S1/S3/S4/S5/S6/S7`.

Не запускаются как paper claims:

- C3.
- C4.

Важно: `S7` включен в synthetic calibration, но не используется как C3 claim.

## Безопасные команды

Минимальный smoke-test:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --smoke --force
```

Локальный default без heavy real simulations:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all
```

Проверить, что simulation layer сделал бы, без запуска тяжелых команд:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --dry-run
```

Запуск только метрик из готовых artifacts:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics
```

Запуск только графиков из готовых metrics tables:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer visualization
```

## Тяжелые запуски

Тяжелые реальные симуляции запускаются только с `--allow-heavy`.

Все simulation commands:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --allow-heavy
```

Только C2/APF/branching simulation:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --task c2 --allow-heavy
```

Перед реальным heavy запуском на A100 сначала сделать dry-run:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --task c2 --dry-run
```

## Запуск отдельных задач

Synthetic calibration:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --task synthetic
```

C1:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c1
```

C2:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --task c2
```

C5:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c5
```

C6:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c6
```

## Что считает код

### N0 synthetic calibration

Скрипт:

```bash
scripts/paper_suite_synthetic.py
```

Simulation layer генерирует synthetic trajectories:

- `S0`: static particles.
- `S1`: homogeneous Brownian/Gaussian motion.
- `S3`: one coherent moving blob.
- `S4`: two-role mixture.
- `S5`: synchronous global switch.
- `S6`: partial/staggered switch.
- `S7`: multi-scale moving blobs.

Metrics layer считает:

- MSPD score `D(tau)`.
- selected tau.
- Delta-H maps.
- role recovery ARI for `S4/S6/S7`.
- event localization for `S5/S6`.
- S7 selected-scale sanity check.

Основные outputs:

```text
analysis/results/paper_suite/synthetic_calibration/simulation/
analysis/results/paper_suite/synthetic_calibration/metrics/
analysis/results/paper_suite/synthetic_calibration/per_family_scores.csv
analysis/results/paper_suite/synthetic_calibration/tau_profiles.csv
analysis/results/paper_suite/synthetic_calibration/role_recovery.csv
analysis/results/paper_suite/synthetic_calibration/event_localization.csv
analysis/results/paper_suite/synthetic_calibration/synthetic_calibration_summary.json
```

Default synthetic settings intentionally small for local CPU/M5:

```yaml
n_particles: 64
time_steps: 240
seeds: 1
```

Для paper-quality run на A100 можно поднять эти значения в config.

### C1 optimized vs random

Скрипт:

```bash
scripts/paper_suite_posthoc.py
```

Код читает saved `trial_*_lagrangian.npz` из paper-check artifacts и считает selection-adjusted comparison:

- каждый optimized и random checkpoint выбирает tau по одинаковому rule;
- selection делается на `control_a`/selection windows;
- final score считается на held-out `control_b`/evaluation windows;
- random controls получают такое же право выбрать tau;
- итоговая статистика считается по matched groups: `optimized - median(random)`.

Outputs per dataset:

```text
analysis/results/paper_suite/<dataset>/checkpoint_scores.csv
analysis/results/paper_suite/<dataset>/group_contrasts.csv
analysis/results/paper_suite/<dataset>/dataset_summary.json
```

Datasets:

- `flow_lenia` required.
- `plife_plus` required.
- `boids` optional secondary substrate.

### C2 event summary

Скрипт:

```bash
scripts/paper_suite_c2_events.py
```

Код читает minibang/APF `metrics.npz` и извлекает:

- selected tau.
- Delta-H peak time.
- Delta-H peak value.
- Delta-H mean.
- optional `cluster_tv_peak`, если в metrics есть cluster turnover signal.

Outputs:

```text
analysis/results/paper_suite/c2_events/c2_event_summary.csv
analysis/results/paper_suite/c2_event_summary.json
```

Это offline summary. Он не запускает Flow-Lenia.

### C2 branching sensitivity

Скрипт:

```bash
scripts/paper_suite_c2_branching.py
```

Simulation layer:

- читает minibang `metrics.npz`;
- выбирает high Delta-H локальные времена;
- подбирает matched low/mid Delta-H времена;
- пишет `branch_plan.csv`;
- при `--allow-heavy` запускает resumed branches через `scripts/flowlenia_minibang_resume.py`;
- каждый branch получает small perturbation и отдельный `branch_seed`.

Без `--allow-heavy` real branches не запускаются.

Metrics layer:

- читает готовые branch directories;
- строит compact branch features из `branch_feature.npz`, `metrics.npz` или APF chunks;
- считает pairwise branch divergence внутри каждого selected time;
- считает paired contrast:

```text
B(high Delta-H) - B(low Delta-H)
```

Outputs:

```text
analysis/results/paper_suite/c2_branching/branch_plan.csv
analysis/results/paper_suite/c2_branching/branching_scores.csv
analysis/results/paper_suite/c2_branching/branching_pair_contrasts.csv
analysis/results/paper_suite/c2_branching_metrics_summary.json
```

Default config is deliberately small:

```yaml
max_trajectories: 2
m_pairs: 2
branches_per_time: 3
horizon_steps: 1000
```

For the paper-quality C2-B run on A100, increase to the design-doc target, e.g. `m_pairs: 5`, `branches_per_time: 4`, and top `2-3` MSPD-opt trajectories.

### C5 frustration / history dependence

Скрипт:

```bash
scripts/paper_suite_posthoc.py
```

Код читает saved paper-check artifacts:

- `trial_*_lagrangian.npz`
- `trial_*_embeddings.npz`
- `trial_results.csv` or trial json rows

И считает frustration effect:

```text
d(control_a, walls) - d(control_a, control_b)
```

По embedding axes и Delta-H map axes. Затем агрегирует per matched group:

```text
optimized - median(random)
```

Outputs per dataset:

```text
analysis/results/paper_suite/<dataset>/frustration_trial_metrics.csv
analysis/results/paper_suite/<dataset>/frustration_run_level.csv
analysis/results/paper_suite/<dataset>/frustration_metric_summary.csv
```

### C6 cross-substrate transfer

Скрипт:

```bash
scripts/paper_suite_posthoc.py
```

Код агрегирует C1/C5 summaries across substrates:

- Flow-Lenia.
- Particle Life++.
- Boids, if artifacts exist.

Outputs:

```text
analysis/results/paper_suite/cross_substrate_summary.csv
analysis/results/paper_suite/paper_suite_metrics_summary.json
```

## Visualization

Скрипт:

```bash
scripts/paper_suite_visualize.py
```

Код не пересчитывает метрики и не запускает симуляции. Он читает готовые CSV/JSON/NPZ и пишет figures:

```text
analysis/results/paper_suite/figures/synthetic_calibration_grid.png
analysis/results/paper_suite/figures/c2_branching_sensitivity.png
analysis/results/paper_suite/figures/c1_<dataset>_paired_contrast.png
analysis/results/paper_suite/figures/c5_<dataset>_frustration_contrast.png
analysis/results/paper_suite/figures/c6_cross_substrate_effects.png
```

## Какие artifacts должны быть на удаленной машине

Paper-check artifacts:

```text
experiments/paper_check_flow_lenia/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check_flow_lenia/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check_flow_lenia/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check_boids/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check_boids/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check_boids/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz
```

Minibang/APF artifacts:

```text
experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set/manifest.json
experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set/traj_*/metrics.npz
experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set/traj_*/config.yaml
experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set/traj_*/params.npy
experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set/traj_*/apf_logs/P_steps_*.npz
```

APF chunks should contain resume-capable keys where possible:

```text
A
P
F
lagrangian_xy
lagrangian_c
resume_batch_rng_key
state_t
state_mass_cycle_start
```

## Suggested A100 workflow

1. Validate command plan without running heavy compute:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --dry-run
```

2. Run missing APF/paper-check simulations only if expected artifacts are absent:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --allow-heavy
```

3. Compute metrics from saved artifacts:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics
```

4. Build figures:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer visualization
```

5. Full one-command paper suite after artifacts are ready:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all
```

## Troubleshooting

Check active paper-suite processes:

```bash
lsof -nP | rg 'paper_suite|paper_check|flowlenia_minibang|run_paper_suite|synthetic_calibration'
```

Check Docker containers:

```bash
docker ps
docker stats --no-stream
```

Stop a Docker container if needed:

```bash
docker stop <container_id>
```

If Docker Desktop still holds RAM after all containers stop, restart Docker Desktop. Docker VM memory may not be returned immediately.

If local RAM is tight, do not increase synthetic defaults. Keep:

```yaml
n_particles: 64
time_steps: 240
seeds: 1
```

Use `--smoke` for quick validation and reserve larger settings for A100.

If stale synthetic metrics are present, rerun with:

```bash
conda run -n onerec python scripts/paper_suite_synthetic.py experiments/paper_suite/config.yaml --layer all --force
```

Metric cache is keyed by trajectory metadata, shape, and metric config; stale cache should not be reused silently.

## File map

```text
experiments/paper_suite/config.yaml      main suite config
experiments/paper_suite/run.sh           shell wrapper
scripts/run_paper_suite.py               one-button orchestrator
scripts/paper_suite_simulation.py        simulation/validation layer
scripts/paper_suite_synthetic.py         N0 synthetic calibration
scripts/paper_suite_posthoc.py           C1/C5/C6 posthoc metrics
scripts/paper_suite_c2_events.py         C2 event summary
scripts/paper_suite_c2_branching.py      C2 branching sensitivity
scripts/paper_suite_visualize.py         figures
scripts/paper_suite_common.py            shared utilities
scripts/flowlenia_minibang_resume.py     resume branches from APF snapshots
```
