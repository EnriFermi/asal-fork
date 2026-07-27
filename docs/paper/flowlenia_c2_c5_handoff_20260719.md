# Flow-Lenia C2/C5 Handoff

Date: 2026-07-19

Status: discussion and current protocol decisions recorded for the next
paper-ready overnight task.

## C2 Decision

The scientific C2 claim is:

```text
States with higher Delta-H have greater uncertainty in their future.
```

External state noise is not part of this claim. The primary C2 condition must
therefore use the already completed `noise_scale = 0` branches: identical
saved state and parameters at the branch point, no additive state
perturbation, and independent continuation RNG keys.

The identical relative-step-zero frame must be excluded from future
divergence. Existing nonzero-noise results are a supplementary intervention
robustness analysis, not the primary test of C2.

The detailed decision, preliminary results, caveats, and final checklist are
recorded in:

```text
docs/paper/flowlenia_c2_protocol_decision_20260719.md
```

## C5 Delta-H Map Request

The immediate C5 request is to render a Delta-H map for every existing
wall-free and walled frustration trajectory. No new Flow-Lenia simulation is
needed or allowed for this visualization step.

The source root is:

```text
experiments/paper_check_flow_lenia/
checkpoints_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/
frustration_simulation/
```

The source contains 40 trials:

- 10 optimization groups, `opt_000` through `opt_009`;
- one optimized candidate and three random candidates per group;
- `control_a`, `control_b`, and `walls` trajectories per trial.

The complete visualization set therefore contains 120 numerical maps:

- 80 no-wall maps (`control_a` and `control_b`);
- 40 wall maps (`walls`).

The maps must be computed from the cached `trial_data/*_lagrangian.npz`
trajectories with the same implementation and parameters used by the current
C5 paper-suite evaluation:

```text
backend: scripts.clip_deltah_msc_metric.make_metric_loss_fn
tau mode: max_grid
tau steps: 1000, 2000, ..., 10000
window size: 20000 steps
window stride: 5000 steps
m samples: 48
m min: 4
projections: 16
null repetitions: 6
particle samples: 64
periodic: false
domain: 128 x 128
Delta-H floor: 0
MSC floor: 0.01
fixed distribution tau: 3000 steps
```

Metric RNG must match C5 exactly:

```text
metric_seed_base = seed_x + 10000000
control_a fold-in = 0
control_b fold-in = 1
walls fold-in = 2
```

The existing C5 table was evaluated with the JAX CPU backend. A trial-00000
probe reproduced all three saved scalar scores to CSV precision on CPU
(absolute differences below `9e-17`). The same calculation on GPU differed
by approximately `1e-7`. Therefore the final map cache must record and enforce
the CPU backend and verify every recomputed scalar score against
`frustration_trial_metrics.csv`.

## C5 Outputs

The dedicated cached visualizer is:

```text
scripts/plot_flowlenia_c5_delta_h_maps.py
```

Its output root is:

```text
analysis/results/
paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/
flow_lenia/c5_delta_h_maps/
```

Expected final artifacts:

- 120 versioned numerical map caches with source SHA-256 and metric config;
- 40 three-panel per-trial figures comparing both controls with walls;
- 10 per-optimization-group sheets containing all four candidates;
- a manifest with source, RNG, backend, map shape, score, and parity fields;
- a machine-readable summary and resolved metric configuration.

Per-trial plots use one exact shared color range across their three
trajectories. Per-run sheets use one shared 99.5th-percentile color range
across all 12 panels for visibility. A white horizontal line marks the
argmax-selected tau, and an orange dashed line marks the fixed 3000-step tau
used by C5 distribution comparisons.

## C5 Control-A Score Mismatch Found After Rendering

The rendered C5 `Control A` score must not be interpreted as the corresponding
C1 score.

For optimized `opt_004`, C5 `Control A` uses run seed `400011`, and its
training-reference guard passes against C1 seed 0 at step 300000. However, the
two displayed scores are evaluated on different metric trajectories:

- C1 scores the training trajectory over approximately steps 50000--300000;
- C5 scores a new late-window Lagrangian track over steps 1000000--1200000;
- the C5 Lagrangian particles are initialized at step 1000000 rather than
  carried from the C1 training track;
- the main C1 run value aggregates four rollout seeds, while C5 `Control A`
  is only optimizer-native seed 0.

Concrete `opt_004` values are:

```text
C1 seed 0 full/train-tau score: 0.0010119345970451, tau=8000
C1 seed 0 eval score:           0.0013496285701794, tau=8000
C1 four-seed eval mean:         0.00088807310377545
C5 late Control A posthoc:      0.000073982315370813, tau=9000
```

There is also a real C5 posthoc seed-provenance bug. The simulation job used
`metric_seed = 800019` for trial 16 and originally reported
`0.000063956540543586` at tau 6000. The posthoc trial table does not carry the
job's `metric_seed`, so `paper_check_metric_stats.py` falls back to
`seed_x + 10000000 = 10400011`, producing the rendered
`0.000073982315370813` at tau 9000.

Therefore the statement that the rendered maps exactly reproduce the C5
simulation protocol is too strong. They exactly reproduce the current
posthoc CSV, including its fallback seed, but not the source simulation's
metric RNG. Before final paper use:

1. propagate each trial's original `job.metric_seed` into posthoc rows;
2. rebuild the Delta-H maps from cached trajectories with that seed;
3. verify reconstructed scalar scores against the original
   `trial_results.csv`;
4. retain explicit labels that C1 is the training window and C5 is the late
   continuation window.

## Next Overnight Task

The user will provide a separate overnight instruction to carry the analyses
to final paper-ready results. Do not reinterpret the C2 claim as external
noise robustness, do not rerun completed RNG-only C2 branches, and do not
resimulate C5 merely to produce the Delta-H maps.
