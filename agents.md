# Paper Suite Execution Plan

This file is the implementation plan for the MSPD paper experiment suite. It is intentionally operational: every layer must be runnable independently, must write reusable artifacts, and must fail loudly on malformed outputs.

## Scope

Included claims:

- C1: MSPD-optimized systems vs matched random controls.
- C2: Delta-H marks transition-sensitive states.
- C5: blockwise frustration / history dependence.
- C6: transfer to Particle Life++; Boids is treated as a secondary limited-expressivity substrate.
- N0: mandatory synthetic calibration suite S0-S7.

Excluded claims:

- C3 as a paper claim is not run.
- C4 is deferred and must not be revived through ad hoc species/color/object-count metrics.

Important distinction:

- S7 is included in the mandatory synthetic calibration suite. It is not used to make C3 as a paper claim; it calibrates the instrument on ground-truth multi-scale dynamics.

## Layer Contract

The suite has three independent layers.

### 1. Simulation Layer

Purpose: create or validate reusable raw artifacts. This layer may run expensive simulations only when explicitly requested.

Outputs:

- Synthetic trajectory files under `analysis/results/paper_suite/synthetic_calibration/simulation/`.
- Paper-check A/B/C artifacts under each substrate's `experiments/paper_check_*/checkpoints/frustration_simulation/`.
- Flow-Lenia APF/minibang artifacts under the configured APF output root.
- C2 branching plan and optional resumed branch trajectories under `analysis/results/paper_suite/c2_branching/`.

Required reusable logs:

- For paper-check frustration: `trial_XXXXX_lagrangian.npz` and `trial_XXXXX_embeddings.npz`.
- For APF/minibang C2 work: `P_steps_*.npz` chunks containing `A`, `P`, `F`, `lagrangian_xy`, `lagrangian_c`, `resume_batch_rng_key`, `state_t`, and `state_mass_cycle_start` where possible.
- For C2 branching: `branch_plan.csv` plus one branch output directory per selected high/low time and branch seed. Branch outputs are resume-capable APF directories or compact `branch_feature.npz` smoke fixtures.

Policy:

- Default is `reuse_existing: true`.
- Heavy real simulations are skipped unless `--allow-heavy` is passed.
- Optimization reruns are disabled unless both config and CLI allow them.
- Missing required artifacts are reported in a manifest instead of being silently recomputed.

### 2. Metrics / Posthoc Layer

Purpose: compute all scores, contrasts, statistics, and machine-readable paper tables from saved artifacts only.

Outputs:

- Synthetic calibration metrics:
  - `per_family_scores.csv`
  - `tau_profiles.csv`
  - `role_recovery.csv`
  - `event_localization.csv`
  - `synthetic_calibration_summary.json`
  - per-run `*_metrics.npz`
- Paper-check metrics per substrate:
  - `checkpoint_scores.csv`
  - `group_contrasts.csv`
  - `frustration_run_level.csv`
  - `frustration_metric_summary.csv`
  - `dataset_summary.json`
- Cross-substrate summary:
  - `cross_substrate_summary.csv`
  - `paper_suite_metrics_summary.json`
- C2 branching:
  - `c2_branching/branching_scores.csv`
  - `c2_branching/branching_pair_contrasts.csv`
  - `c2_branching_metrics_summary.json`

Rules:

- C1 tau selection is posthoc and selection-adjusted: every optimized and random checkpoint gets the same right to select tau on `control_a`/selection windows, then the reported score is measured on `control_b`/held-out windows.
- C5 frustration is computed as `d(control_a, walls) - d(control_a, control_b)` for embedding and Delta-H map axes.
- C2 branching computes `B(high Delta-H) - B(matched low Delta-H)` from saved branch continuations only.
- The metrics layer must not invoke Flow-Lenia/Boids/PLife simulators.

### 3. Visualization Layer

Purpose: generate figures only from metrics tables/npz files.

Outputs:

- `figures/synthetic_calibration_grid.png`
- `figures/c1_<substrate>_paired_contrast.png`
- `figures/c2_branching_sensitivity.png`
- `figures/c5_<substrate>_frustration_contrast.png`
- `figures/c6_cross_substrate_effects.png`

Rules:

- Visualization must not recompute metrics.
- Missing metric tables produce a clear skip record, not a partial silent plot.

## Mandatory Synthetic Calibration N0

Families:

- S0 static particles.
- S1 homogeneous Brownian/Gaussian motion.
- S3 one coherent moving blob.
- S4 two-role mixture.
- S5 synchronous global switch.
- S6 partial/staggered switch.
- S7 multi-scale moving blobs.

Default compute target:

- CPU/JAX-light.
- `n_particles: 64`.
- `time_steps: 240`.
- `seeds: 1`.
- Tau grid: `[1, 2, 4, 8, 12, 16]`.
- This is intentionally a local-safe execution profile after the M5 RAM blow-up from the old full synthetic settings. Raise these values only for the final paper-quality run on the A100.

Smoke target:

- `n_particles: 32`.
- `time_steps: 96`.
- `seeds: 1`.
- Small tau/window grid.

Metrics:

- `D(tau)`, selected tau, Delta-H map.
- Role recovery ARI for S4, S6, S7.
- Event localization for S5 and S6.
- S7 scale-range recovery check.

## C2 Branching Sensitivity

Default local-safe target:

- `max_trajectories: 2`.
- `m_pairs: 2`.
- `branches_per_time: 3`.
- `horizon_steps: 1000`.
- Real branch resume commands are skipped unless `--allow-heavy` is passed.

Paper-quality target on A100 can be raised in config to the document's suggested `m_pairs: 5`, `branches_per_time: 4`, and top 2-3 MSPD-opt trajectories.

Layer behavior:

- Simulation layer selects high Delta-H and matched low/mid Delta-H times from existing minibang `metrics.npz`, writes `branch_plan.csv`, and optionally launches `flowlenia_minibang_resume.py` with small perturbations.
- Metrics layer loads branch outputs, summarizes future APF/metric trajectories into compact features, computes pairwise branch divergence, and reports paired high-low contrasts.
- Visualization layer reads only `branching_scores.csv` and `branching_pair_contrasts.csv`.

## C1 / C5 / C6 Artifact Expectations

Flow-Lenia required:

- `experiments/paper_check_flow_lenia/checkpoints/optimization/run_000/best.pkl` etc.
- `experiments/paper_check_flow_lenia/checkpoints/frustration_simulation/trial_results.csv` or `trial_data/trial_*.json`.
- `trial_XXXXX_lagrangian.npz`.
- `trial_XXXXX_embeddings.npz`.

Particle Life++ required for C6 primary:

- `experiments/paper_check_plife_plus/checkpoints/frustration_simulation/trial_results.csv` or `trial_data/trial_*.json`.
- Lagrangian and embedding npz artifacts as above.

Boids secondary:

- Same paper-check artifact format as PLife++.
- Missing Boids artifacts should not fail the full paper suite unless explicitly marked required.

## Entry Points

Main one-button run:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all
```

Layer-specific runs:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer visualization
```

Task-specific runs:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --task synthetic
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c1
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c2
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c5
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer metrics --task c6
```

Heavy simulation run:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer simulation --allow-heavy
```

Smoke run:

```bash
conda run -n onerec python scripts/run_paper_suite.py experiments/paper_suite/config.yaml --layer all --smoke
```

## Testing Requirements

Before considering this done:

- Run syntax/import checks for the new scripts.
- Run synthetic simulation, metrics, and visualization in smoke mode.
- Run posthoc paper-check metrics in smoke mode using generated tiny fake artifacts.
- Run the main orchestrator in smoke mode.
- Verify that generated CSV/JSON/PNG artifacts exist and have non-empty rows where expected.
