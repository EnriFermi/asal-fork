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

Python dependencies are listed in:

```text
requirements_paper_suite.txt
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
- N0: mandatory synthetic calibration `S0/S1/S3/S4/S5/S6/S7/S8`.

Не запускаются как paper claims:

- C3.
- C4.

Важно: `S7/S8` включены в synthetic calibration, но не используются как C3 claim.

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
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --task c2 --allow-heavy --dry-run
```

## Запуск отдельных задач

Synthetic calibration:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --task synthetic
```

Быстрый локальный цикл только для synthetic calibration, без C1/C2/C5/C6:

```bash
conda run -n onerec python scripts/paper_suite_synthetic.py experiments/paper_suite/config.yaml --layer simulation --smoke --force
conda run -n onerec python scripts/paper_suite_synthetic.py experiments/paper_suite/config.yaml --layer metrics --smoke --force
conda run -n onerec python scripts/paper_suite_synthetic.py experiments/paper_suite/config.yaml --layer visualization --smoke --force
```

То же самое одной командой:

```bash
conda run -n onerec python scripts/paper_suite_synthetic.py experiments/paper_suite/config.yaml --layer all --smoke --force
```

C1:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c1
```

C2:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --task c2 --allow-heavy
```

C5:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c5
```

C6 / C6.1:

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
- `S8`: one moving blob splits into multiple blobs with different directions, an S3-to-S7 transition.

Metrics layer считает:

- MSPD score `D(tau)`.
- selected tau.
- Delta-H maps.
- role recovery ARI for `S4/S6/S7/S8`.
- event localization for `S5/S6/S8`.
- S7/S8 selected-scale sanity check.

Small positive Delta-H values can be suppressed with:

```yaml
synthetic:
  metric_delta_h_floor: 0.0
```

This threshold is applied after `metric_preprocess_mode` and before `amp`/`msc`/`score` and `tau` selection. The stored raw `delta_h_map` remains raw for diagnostics and heatmaps; the exact thresholded values are saved as `delta_h_processed_map`.

Simulation layer также сразу рендерит trajectory videos:

- `analysis/results/paper_suite/synthetic_calibration/videos/S*_seed_*.mp4`

Visualization layer для synthetic calibration читает только готовые metrics tables/npz и строит:

- aggregate grid `analysis/results/paper_suite/figures/synthetic_calibration_grid.png`;
- per-run Delta-H heatmaps by tau under `analysis/results/paper_suite/synthetic_calibration/figures/delta_h_heatmaps/`;
- family median Delta-H heatmaps and combined overview `analysis/results/paper_suite/figures/synthetic_delta_h_heatmaps.png`.

Основные outputs:

```text
analysis/results/paper_suite/synthetic_calibration/simulation/
analysis/results/paper_suite/synthetic_calibration/metrics/
analysis/results/paper_suite/synthetic_calibration/videos/
analysis/results/paper_suite/synthetic_calibration/figures/delta_h_heatmaps/
analysis/results/paper_suite/synthetic_calibration/per_family_scores.csv
analysis/results/paper_suite/synthetic_calibration/tau_profiles.csv
analysis/results/paper_suite/synthetic_calibration/role_recovery.csv
analysis/results/paper_suite/synthetic_calibration/event_localization.csv
analysis/results/paper_suite/synthetic_calibration/synthetic_calibration_summary.json
analysis/results/paper_suite/synthetic_calibration/delta_h_heatmap_manifest.csv
analysis/results/paper_suite/synthetic_calibration/visualization_summary.json
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

Код читает saved lagrangian artifacts и считает selection-adjusted comparison:

- каждый optimized и random checkpoint выбирает tau по одинаковому rule;
- selection делается по четным metric windows `2k`;
- final score считается на held-out нечетных windows `2k+1`;
- для Flow-Lenia C1 и PLife++ C6.1 selection/eval maps считаются из одной reusable trajectory per candidate;
- `control_a/control_b` относятся к C5/C6.5 frustration/history-dependence, не к C6.1;
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

Дополнительный Flow-Lenia A-run APF/lagrangian logging для paper-check diagnostics запускается simulation layer отдельно от C1/C5 metrics:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --task c1 --allow-heavy
```

Он читает те же optimized checkpoint roots, что и paper-check Flow-Lenia/C2, использует параметры `paper_check_flow_lenia/frustration_simulation/config.yaml`, ограничивает rollout `500000` шагами и пишет batched sparse APF + lagrangian chunks:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/apf_logs/P_steps_*.npz
```

Batch size задается в `simulation.flow_lenia_arun_lagrangian_apf.batch_size`.
Инициализация использует paper-check control-A seed convention:
`run_seed_base + 2 * source_run_idx`.

### C2 event summary

Скрипт:

```bash
scripts/paper_suite_c2_events.py
```

Primary C2 source is no longer the old minibang golden set. The suite first
creates/reuses high-resolution Flow-Lenia rollouts from paper-check optimized
checkpoints:

```text
experiments/paper_check/checkpoints/optimization
experiments/paper_check/checkpoints_0/optimization
experiments/paper_check/checkpoints_reference/optimization
```

The dedicated rollout config is:

```text
experiments/paper_suite/flowlenia_arun_apf_500k.yaml
```

It uses the paper-check Flow-Lenia optimization config with:

```yaml
grid_size: 384
rollout_steps: 500000
max_steps: 500000
```

The output trajectory root is:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k
```

The simulation layer writes reusable APF chunks first:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/apf_logs/P_steps_*.npz
```

Those chunks are the durable source for C2 posthoc work. A rollout is treated
as simulation-ready when APF chunks, `config.yaml`, and `params.npy` exist;
`metrics.npz` is produced by the metrics layer and can be deleted/recomputed
without rerunning Flow-Lenia.

The posthoc metrics script is:

```bash
scripts/paper_suite_c2_flowlenia_metrics.py
```

It reads APF chunks and writes:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/metrics.npz
analysis/results/paper_suite/c2_highres_metrics/c2_highres_metrics_manifest.csv
```

Every C2 `metrics.npz` stores metric cache metadata: metric config JSON/hash,
metric code version, tau/window/floor/MSC settings, and an APF chunk input
identity hash. If those do not match the current config, the metrics layer
fails loudly unless `--force` is used, in which case metrics are recomputed from
saved APF chunks without resimulation.

Event summary then reads high-res C2 `metrics.npz` and extracts:

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

- читает high-res C2 `metrics.npz`;
- выбирает high Delta-H локальные времена;
- подбирает matched low/mid Delta-H времена by activity covariates rather than
  nearest time alone: total mass, active area, mean Lagrangian speed, and field
  activity when available;
- пишет `branch_plan.csv` and `branch_plan_meta.json`;
- при `--allow-heavy` запускает resumed branches через `scripts/flowlenia_minibang_resume.py`;
- каждый branch получает small perturbation и отдельный `branch_seed`.

Без `--allow-heavy` real branches не запускаются.

Metrics layer:

- читает готовые branch directories;
- primary metric считает explicit future divergence между branch trajectories
  по APF/field frames over horizon;
- default distance is multi-scale L2 over `A`, plus `P/F` when present with
  smaller weights;
- compact `branch_feature.npz` path оставлен только как smoke/debug fallback,
  not main evidence;
- считает mean pairwise branch divergence внутри каждого selected time;
- считает paired contrast:

```text
B(high Delta-H) - B(low Delta-H)
```

Outputs:

```text
analysis/results/paper_suite/c2_branching/branch_plan.csv
analysis/results/paper_suite/c2_branching/branch_plan_meta.json
analysis/results/paper_suite/c2_branching/branching_scores.csv
analysis/results/paper_suite/c2_branching/branching_pair_contrasts.csv
analysis/results/paper_suite/c2_branching_metrics_summary.json
```

The branching metrics layer refuses old branch plans without matching metadata
or with stale upstream `metrics.npz` hashes. Regenerate the branching simulation
layer after changing C2 metric config.

Default config is deliberately small:

```yaml
max_trajectories: 2
m_pairs: 2
branches_per_time: 3
horizon_steps: 1000
future_field_weights: {A: 1.0, P: 0.25, F: 0.25}
future_field_scales: [1, 2, 4]
future_max_frames: 32
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

По embedding axes и dynamic/MSPD axes. Embedding stays separate; the dynamic
axis uses the same averaged floor-aware MSC config as C1/C2. Затем агрегирует
per matched group:

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

Код агрегирует C1/C5 summaries across substrates; PLife++ C1-style transfer check is reported as C6.1:

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
experiments/paper_check/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check/checkpoints_0/frustration_simulation/trial_results.csv
experiments/paper_check/checkpoints_0/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check/checkpoints_0/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check/checkpoints_reference/frustration_simulation/trial_results.csv
experiments/paper_check/checkpoints_reference/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check/checkpoints_reference/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz

experiments/paper_check_boids/checkpoints/frustration_simulation/trial_results.csv
experiments/paper_check_boids/checkpoints/frustration_simulation/trial_data/trial_*_lagrangian.npz
experiments/paper_check_boids/checkpoints/frustration_simulation/trial_data/trial_*_embeddings.npz
```

C2 / Flow-Lenia paper-check control-A APF artifacts:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/manifest.json
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/config.yaml
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/params.npy
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/apf_logs/P_steps_*.npz
```

C2 high-res posthoc metric artifacts:

```text
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/metrics.npz
experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k/flow_opt_*/metrics_summary.json
analysis/results/paper_suite/c2_highres_metrics/c2_highres_metrics_manifest.csv
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

4. After C2 high-res `metrics.npz` exists, run C2 branch continuations if needed:

```bash
conda run -n onerec python scripts/paper_suite_c2_branching.py experiments/paper_suite/config.yaml --layer simulation --allow-heavy
conda run -n onerec python scripts/paper_suite_c2_branching.py experiments/paper_suite/config.yaml --layer metrics
```

5. Build figures:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer visualization
```

6. Full one-command paper suite after artifacts are ready:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all
```

## Troubleshooting

Every `scripts/run_paper_suite.py` run now creates timestamped logs under:

```bash
analysis/results/paper_suite/logs/
```

The main file is `*_master.log`; every subprocess also gets its own `*_paper_suite_*.log`.
Watch the current run with:

```bash
tail -f analysis/results/paper_suite/logs/*_master.log
```

Watch the quiet posthoc stage specifically:

```bash
tail -f analysis/results/paper_suite/logs/*_paper_suite_posthoc.log
```

The posthoc and C2 stages print progress lines for dataset loading, C1 trial scoring,
C5 history-distance rows, C2 APF metrics, event extraction, and branching metrics.

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
