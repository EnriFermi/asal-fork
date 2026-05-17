# Карта экспериментов из кода для claims

Дата анализа: 2026-05-14.

Уточнение: этот файл не оценивает наличие локальных результатов. Я рассматриваю репозиторий как набор кода, конфигов и ноутбуков, то есть ищу эксперименты, которые **могут подтвердить** claims после запуска и сбора outputs. Папки с result-like артефактами не используются как evidence; максимум они показывают ожидаемый формат outputs.

## Короткая карта

| Claim | Лучшие эксперименты в коде | Насколько закрывает claim |
|---|---|---|
| C1. MSPD отделяет optimized Flow-Lenia от random controls | `experiments/paper_check_flow_lenia/*`, `experiments/flow_lenia_mspd/optimization/*`, `experiments/flow_lenia_mspd/simulation/config_apf*.yaml`, `analysis/deltah_mssc_comparison.ipynb` | Direct for MSPD-opt vs random; NN-OEE comparison есть отдельным APF/log notebook route |
| C2. DeltaH spikes correspond to species-turnover events | `analysis/deltah_mssc_comparison.ipynb`, `experiments/flow_lenia_mspd/minibang_golden_set/*`, `scripts/flowlenia_minibang_*`, `scripts/parse_minibang_markup.py` | Partial; spike/event pipeline есть, но formal turnover/null tests надо усилить |
| C3. MSPD reveals scale separation at tau* | `analysis/deltah_mssc_comparison.ipynb`, `experiments/flow_lenia_mspd/optimization/config_longrun_check.yaml`, `experiments/flow_lenia_mspd/minibang_golden_set/config.yaml` | Partial-direct; tau grid and cluster separability есть, но selection rule `S(tau)` надо явно зафиксировать |
| C4. MSPD worlds match NN-OEE in ecological richness | `analysis/deltah_mssc_comparison.ipynb`, `experiments/flow_lenia_apf_rollouts/simulation/*`, `experiments/flow_lenia_mspd/simulation/config_apf.yaml`, minibang pipeline | Mostly partial; comparison protocol есть, richness taxonomy отсутствует |
| C5. Frustration is structural/snapshot, not dynamic/MSPD | `experiments/paper_check_flow_lenia/*`, `scripts/run_paper_check_frustration.py`, `scripts/paper_check_frustration_batch_eval.py`, `analysis/notebooks/flow_lenia/paper_check_log_analysis_fixed.ipynb`, `analysis/history_dependence/*` | Direct for frustration assay; literal 7/7 needs R=7 config/override, orthogonality needs extra correlation analysis |
| C6. MSPD generalizes to Boids and Particle Life | `experiments/paper_check_boids/*`, `experiments/paper_check_plife_plus/*`, `experiments/boids_mspd/*`, `experiments/plife_plus_mspd/*`, substrate notebooks | Direct for running MSPD on both substrates; paper-level proof still needs same stats/baselines as Flow-Lenia |

## Основные experiment families

| Family | Code/configs | What it runs | Expected outputs |
|---|---|---|---|
| Flow-Lenia MSPD optimization | `scripts/main_opt_msc.py`, `experiments/flow_lenia_mspd/optimization/config_longrun_check.yaml`, `experiments/paper_check_flow_lenia/optimization/config_longrun_check.yaml` | Evolves Flow-Lenia params with DeltaH/MSPD objective; supports fixed/max/trainable tau grids | `best.pkl`, optimization logs, MSPD scores, selected tau, later APF/trajectory rollouts |
| Paper-check orchestration | `scripts/run_paper_check_optimization.py`, `scripts/run_paper_check_frustration.py`, `scripts/paper_check_common.py`, `experiments/paper_check_flow_lenia/config.yaml` | Runs R optimized seeds and M random baselines, sharded over machines | `optimization/run_*/best.pkl`, `frustration_simulation/trial_results.csv`, trial artifacts |
| APF / Lagrangian logging | `experiments/flow_lenia_mspd/simulation/config_apf.yaml`, `config_apf_random.yaml`, `experiments/flow_lenia_apf_rollouts/simulation/*` | Long Flow-Lenia rollouts with dense snapshots and Lagrangian tracks | APF logs for DeltaH heatmaps and NN-OEE/MSPD/random comparison |
| DeltaH comparison notebook | `analysis/deltah_mssc_comparison.ipynb` | Computes DeltaH heatmaps for `t_mssc`, `random`, `nn_opt`; C2 spike-vs-mass derivative; C3 fast/slow clustering | Heatmaps, score tables, C2 correlation/scatter, C3 cluster diagnostics |
| Minibang / event pipeline | `experiments/flow_lenia_mspd/minibang_golden_set/config.yaml`, `scripts/flowlenia_minibang_simulate.py`, `flowlenia_minibang_resume.py`, `flowlenia_minibang_detect.py`, `flowlenia_minibang_plot_delta_h.py`, `parse_minibang_markup.py` | Samples optimized trajectories, computes DeltaH and color/cluster dynamics, detects candidate events and parses human labels | `metrics.npz`, `metrics_summary.json`, videos, event candidates, markup stats, ROC/correlation tables |
| Frustration / history dependence | `experiments/paper_check_flow_lenia/frustration_simulation/config.yaml`, `scripts/paper_check_frustration_batch_eval.py`, `analysis/history_dependence/*`, `analysis/notebooks/flow_lenia/paper_check_log_analysis_fixed.ipynb` | Runs control A, control B and walls/perturbed rollout; compares CLIP embedding distances and MSPD/DeltaH distances | `trial_results.csv`, `run_level_table.csv`, `primary_test_summary.csv`, `history_distance_metric_tests.csv` |
| Boids MSPD paper check | `experiments/paper_check_boids/*`, `experiments/boids_mspd/*`, `analysis/notebooks/boids/paper_check_log_analysis_fixed.ipynb` | Same paper-check design on Boids with `state_x` trajectories | Boids optimized/random runs, frustration/metric tables |
| Particle Life Plus paper check | `experiments/paper_check_plife_plus/*`, `experiments/plife_plus_mspd/*`, `analysis/notebooks/plife_plus/paper_check_log_analysis_fixed.ipynb` | Same paper-check design on PLife+ with `state_x` trajectories; also CLIP-OE and tau sweep configs | PLife optimized/random runs, CLIP-OE comparison, tau sweep plots, frustration/metric tables |
| Bootstrap/ranking stability | `experiments/legacy/opt_1/bs_variance/config.yaml`, `experiments/legacy/opt_1/bs_variance/run.sh` | Evaluates ranking stability for CLIP and MSC scores over bootstrap batch sizes | Stability CSVs/plots; useful as metric reliability support, not a main claim proof |

## Claim-by-claim mapping

### C1. MSPD отделяет optimized Flow-Lenia от random controls

**Главный experiment:** `experiments/paper_check_flow_lenia/config.yaml` + `scripts/run_paper_check_optimization.py`.

Почему подходит:

- `paper_check.num_optimizations: 5` задает R optimized Flow-Lenia runs.
- `paper_check.num_random_baselines: 3` задает M matched random baselines per optimized group.
- `optimization/base_config: optimization/config_longrun_check.yaml` использует Flow-Lenia, `main_opt_msc.py`, `metric_tau_mode: trainable_grid`, tau grid `1000..10000`, DeltaH/MSPD objective `metric_alpha: 0.0`, `metric_beta: 1.0`.
- `run_paper_check_frustration.py` умеет генерировать random checkpoints через тот же substrate/optimizer parameterization and writes trial-level rows containing `msc_score_control_mean` and related metrics.

How to run the intended package:

```bash
python scripts/run_paper_check_optimization.py experiments/paper_check_flow_lenia/config.yaml
python scripts/run_paper_check_frustration.py experiments/paper_check_flow_lenia/config.yaml
```

Expected claim evidence:

- table by run group: optimized `msc_score_control_mean` minus median random `msc_score_control_mean`;
- confidence interval / exact sign test or Mann-Whitney over run groups;
- DeltaH/MSPD heatmaps for optimized vs random if APF logging is also run.

**NN-OEE add-on:** `analysis/deltah_mssc_comparison.ipynb` is the code path that compares `t_mssc`, `random`, and `nn_opt`. It expects APF logs from:

- MSPD-opt: `experiments/flow_lenia_mspd/simulation/config_apf.yaml`;
- random: `experiments/flow_lenia_mspd/simulation/config_apf_random.yaml`;
- NN-OEE: `experiments/flow_lenia_apf_rollouts/simulation/2602112129/config.yaml` or similar `experiments/flow_lenia_apf_rollouts/simulation/*`.

**Gap:** C1’s minimal result in `claims.md` asks for `D_P`, mean DeltaH and variance with CIs/effect sizes. The code mostly gives MSPD/MSC scalar and DeltaH maps; a small analysis wrapper may be needed to emit exactly the paper table.

### C2. DeltaH spikes correspond to species-turnover events

**Direct notebook experiment:** `analysis/deltah_mssc_comparison.ipynb`.

Почему подходит:

- Notebook section `(b)` explicitly computes DeltaH spike vs selected species mass derivative `|dM/dt|`.
- It is designed to plot time series/scatter for a selected glider/species cluster.

**Event/minibang pipeline:** `experiments/flow_lenia_mspd/minibang_golden_set/config.yaml` with:

- `scripts/flowlenia_minibang_simulate.py` / `flowlenia_minibang_resume.py` for trajectory generation;
- `scripts/flowlenia_minibang_detect.py` for candidate event detection;
- `scripts/flowlenia_minibang_plot_delta_h.py` for DeltaH and cluster plots;
- `scripts/parse_minibang_markup.py` for human-labelled event markup.

Почему подходит:

- `config.yaml` computes DeltaH over tau grid `1000..10000`, window size `20000`, step `5000`;
- it computes color/cluster dynamics using `cluster_method: dpmeans`, `cluster_space: pcolor_chroma`;
- detector reasons include DeltaH spikes, cluster entropy changes and cluster mass shifts.

Expected claim evidence:

- event-level table: DeltaH spike time, turnover/mass-shift time, lag, event label;
- cross-correlation or lag distribution between DeltaH and `|dM/dt|`;
- temporal-shift/null-event baseline;
- human-labelled minibang ROC/AUC as supporting but not sufficient evidence.

**Gap:** The code has the ingredients, but the strong claim says "species-turnover". The current pipeline’s closest built-in proxy is cluster mass/entropy shift. To make C2 solid, the analysis should explicitly define turnover and add a null/permutation test.

### C3. MSPD reveals scale separation at system-identified tau*

**Direct notebook experiment:** `analysis/deltah_mssc_comparison.ipynb`.

Почему подходит:

- Section `(c)` is explicitly "Trajectory cluster separability at tau approx 3000".
- It builds per-particle trajectory signatures, fits a two-cluster fast/slow model, and reports cluster balance/separation diagnostics.

**Tau-selection optimization experiment:** `experiments/flow_lenia_mspd/optimization/config_longrun_check.yaml`.

Почему подходит:

- uses `metric_tau_mode: trainable_grid`;
- tau grid is `[1000, 2000, ..., 10000]`;
- objective can learn/select the tau latent under the MSPD objective.

**Posthoc tau-grid experiment:** `experiments/flow_lenia_mspd/minibang_golden_set/config.yaml`.

Почему подходит:

- posthoc metrics use `metric_tau_mode: max_grid`;
- generates DeltaH/MSPD by tau and can produce a distribution of selected tau values over many trajectories.

Expected claim evidence:

- a declared selection rule `S(tau)` or metric-selected tau protocol;
- plot/table of `S(tau)` or selected tau over seeds;
- 2D trajectory embedding/clustering at tau*;
- spatial overlay showing slow cluster as glider core and fast cluster as periphery/medium.

**Gap:** The code supports fixed tau, max-grid tau and trainable-grid tau, but the paper claim needs one unambiguous definition of "system-identified tau*". If the article says tau* is discovered, use trainable/max-grid and report selection stability; if it says tau is fixed near 3000, soften the wording.

### C4. MSPD-optimized worlds match NN-OEE baselines in ecological richness

**Comparison experiment:** `analysis/deltah_mssc_comparison.ipynb`.

Почему подходит:

- It is explicitly written to compare T-MSSC/MSPD optimized, random and NN-optimized systems.
- It consumes APF logs generated by `experiments/flow_lenia_mspd/simulation/config_apf.yaml`, `experiments/flow_lenia_mspd/simulation/config_apf_random.yaml` and `experiments/flow_lenia_apf_rollouts/simulation/*`.

**Event richness support:** minibang pipeline in `experiments/flow_lenia_mspd/minibang_golden_set/*`.

Почему подходит:

- It can generate videos, event candidates, DeltaH plots and cluster-mass dynamics.
- Human markup parsing can turn qualitative events into counts.

Expected claim evidence:

- table comparing MSPD-opt vs NN-OEE-opt on species/morphotype count, interaction-type count, Shannon diversity, turnover rate, replication/predation/collision rates;
- equal-budget or explicitly unequal-budget comparison;
- qualitative panels/videos as supplementary evidence.

**Gap:** I did not find a complete ecological-richness taxonomy implementation. The existing code can support qualitative/event-count evidence, but a strong C4 needs an added analysis layer that defines and counts richness.

### C5. Frustration structural/snapshot, not dynamic/MSPD

**Main experiment:** `experiments/paper_check_flow_lenia/config.yaml` + frustration simulation and fixed analysis notebook.

Run chain:

```bash
python scripts/run_paper_check_optimization.py experiments/paper_check_flow_lenia/config.yaml
python scripts/run_paper_check_frustration.py experiments/paper_check_flow_lenia/config.yaml
```

Then analyze with:

- `analysis/notebooks/flow_lenia/paper_check_log_analysis_fixed.ipynb`;
- or reusable code in `analysis/history_dependence/*`.

Почему подходит:

- `scripts/run_paper_check_frustration.py` pairs every optimized checkpoint with matched random baselines.
- `scripts/paper_check_frustration_batch_eval.py` runs three lanes: `control_a`, `control_b`, and `walls`.
- It records CLIP embedding distances and MSPD/MSC trajectory metrics.
- `analysis/history_dependence/paper_check_metrics.md` defines the key quantities:
  - `baseline_distance = d(control_a, control_b)`;
  - `walls_effect_distance = d(control, walls)`;
  - `anchor_effect_minus_baseline = d(control_a, walls) - d(control_a, control_b)`;
  - `msc_score_control_mean`, `msc_score_walls`, DeltaH map/distances.
- The fixed notebook computes run-level:
  - optimized minus random-control median;
  - exact one-sided sign test;
  - `primary_test_summary.csv`;
  - `history_distance_metric_tests.csv`.

Expected claim evidence:

- per-run table of structural/snapshot frustration significance;
- per-run table of dynamic/MSPD frustration significance;
- sign/binomial test for structural axis and dynamic axis;
- regression or correlation showing frustration gap explained by snapshot metric and not by MSPD;
- direct correlation/independence test between snapshot score and MSPD over a broader system pool.

**Important config mismatch:** current `experiments/paper_check_flow_lenia/config.yaml` has `num_optimizations: 5`, not 7. For the literal "7/7 vs 3/7" claim, use an override/config with `paper_check.num_optimizations: 7`, or do not state 7/7 in the paper.

**Gap:** The frustration assay itself is well represented in code. The "orthogonal axes" part is not fully closed by the existing notebook; it needs an explicit snapshot-vs-MSPD correlation/independence analysis.

### C6. MSPD generalizes beyond Flow-Lenia to Boids and Particle Life

**Boids paper-check experiment:** `experiments/paper_check_boids/config.yaml`.

Why it fits:

- same R/M paper-check structure as Flow-Lenia;
- `optimization/config_longrun_check.yaml` uses `substrate: boids`;
- trajectories come from `metric_trajectory_source: state_x`;
- tau grid is `[100, 200, 400, 800, 1200, 1600, 2000, 3000, 4000]`;
- fixed analysis notebook: `analysis/notebooks/boids/paper_check_log_analysis_fixed.ipynb`.

Run chain:

```bash
python scripts/run_paper_check_optimization.py experiments/paper_check_boids/config.yaml
python scripts/run_paper_check_frustration.py experiments/paper_check_boids/config.yaml
```

**Particle Life Plus paper-check experiment:** `experiments/paper_check_plife_plus/config.yaml`.

Why it fits:

- same R/M paper-check structure;
- `optimization/config_longrun_check.yaml` uses `substrate: plife_plus`;
- `metric_trajectory_source: state_x`;
- logs color diversity during optimization;
- fixed analysis notebook: `analysis/notebooks/plife_plus/paper_check_log_analysis_fixed.ipynb`.

Run chain:

```bash
python scripts/run_paper_check_optimization.py experiments/paper_check_plife_plus/config.yaml
python scripts/run_paper_check_frustration.py experiments/paper_check_plife_plus/config.yaml
```

**Tau calibration / OE-vs-MSPD-vs-random for PLife+:** `experiments/plife_plus_mspd/tau_sweep/config.yaml` and `scripts/plot_msc_tau_sweep.py`.

Why it fits:

- compares groups `oe`, `msc`, `random`;
- evaluates `score_by_tau`;
- can produce tau calibration plots and DeltaH heatmaps.

Expected claim evidence:

- learning curves and final MSPD score tables for Boids and PLife+;
- optimized-vs-random CI/effect sizes per substrate;
- substrate-specific tau calibration;
- optionally frustration-style tables if C6 is tied to the same paper-check protocol.

**Gap:** The code is ready for generalization experiments. For a paper-level C6, run both substrates with the same reporting format as Flow-Lenia and avoid claiming Boids if only PLife+ has completed outputs.

## Experiments that are useful but secondary

| Experiment | Where | Use |
|---|---|---|
| Bootstrap/ranking stability | `experiments/legacy/opt_1/bs_variance/config.yaml`, `experiments/legacy/opt_1/bs_variance/run.sh` | Supports reliability of metric evaluation and candidate ranking, but does not by itself prove C1-C6 |
| Legacy Flow-Lenia optimization | `experiments/legacy/opt_1/optimization/config.yaml`, `experiments/legacy/opt_online/*`, `experiments/legacy/opt_halving/*` | Historical/prototype optimization routes; useful for provenance, weaker for final proof package |
| Old frustration configs | `experiments/legacy/frustration/simulation/*`, `experiments/legacy/frustration/history_dependence/*` | Earlier history-dependence/frustration protocol; useful for method development, but `paper_check` is cleaner for the article |
| CLIP-OE PLife+ | `experiments/plife_plus_mspd/optimization/config_clip.yaml`, `experiments/paper_check_plife_plus/optimization/config_clip_oe.yaml` | Candidate NN/OE baseline for PLife+, especially for C4/C6-style comparisons |

## Minimal run plan by claim priority

1. **C1:** run `experiments/paper_check_flow_lenia/config.yaml`; add/export a table with optimized vs random `msc_score_control_mean`, DeltaH mean/variance and selected tau.
2. **C5:** run the same Flow-Lenia paper-check frustration chain; analyze with `paper_check_log_analysis_fixed.ipynb`; set R=7 if the manuscript keeps "7/7".
3. **C3:** from the Flow-Lenia outputs, report tau selection stability and cluster separability; decide fixed tau vs discovered tau wording.
4. **C2:** run minibang pipeline plus human markup; add event-level turnover/null analysis.
5. **C6:** repeat paper-check optimization/frustration for `paper_check_plife_plus` and `paper_check_boids`.
6. **C4:** generate NN-OEE/APF logs and add an ecological-richness taxonomy analysis; current code only partially covers this.
