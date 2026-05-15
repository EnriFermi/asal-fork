# Paper Check Metrics

This file describes the metrics used by the fixed paper-check analysis notebooks for `plife_plus` and `boids`.

The notebooks work at two levels:

- **trial-level**: one row per candidate trial, with `control_a`, `control_b`, and `walls` trajectories.
- **run-level**: one row per `optimized_run_idx`, comparing the optimized candidate to random controls from the same group.

## Trajectory Variants

| Name | Meaning |
| --- | --- |
| `control_a` | First unperturbed late-window rollout. Used as the anchor trajectory. |
| `control_b` | Second unperturbed late-window rollout. Used as the baseline control. |
| `walls` | Perturbed late-window rollout, for example after `cell_shuffle`. |

## Generic CLIP Distance Metrics

These are written by the frustration simulation into `trial_results.csv` or `trial_data/trial_*.json`.

| Metric | Meaning | Formula |
| --- | --- | --- |
| `baseline_distance` | Distance between the two control CLIP embedding sequences. | `d(control_a, control_b)` |
| `walls_effect_distance_ctrl_a` | Distance from anchor control to perturbed trajectory. | `d(control_a, walls)` |
| `walls_effect_distance_ctrl_b` | Distance from second control to perturbed trajectory. | `d(control_b, walls)` |
| `walls_effect_distance` | Average control-to-walls distance. | `0.5 * (d(control_a, walls) + d(control_b, walls))` |
| `effect_minus_baseline` | Perturbation effect over control-control baseline. | `walls_effect_distance - baseline_distance` |
| `anchor_effect_minus_baseline` | Anchor-style perturbation effect. This is the default primary metric in the fixed notebooks. | `walls_effect_distance_ctrl_a - baseline_distance` |
| `effect_over_baseline_ratio` | Relative perturbation effect. | `walls_effect_distance / max(baseline_distance, 1e-12)` |
| `anchor_effect_over_baseline_ratio` | Relative anchor-style perturbation effect. | `walls_effect_distance_ctrl_a / max(baseline_distance, 1e-12)` |

The simulation-side CLIP distance `d` is controlled by `evaluation.distance_metric`, for example:

| Distance | Formula |
| --- | --- |
| `cosine_mean` | `mean_t(1 - dot(z_a[t], z_b[t]))` |
| `euclidean_mean` | `mean_t(||z_a[t] - z_b[t]||)` |
| `sqeuclidean_mean` | `mean_t(sum((z_a[t] - z_b[t])^2))` |
| `cosine_last` | `1 - dot(z_a[-1], z_b[-1])` |

## CLIP OE Loss Metrics

These are also written by the frustration simulation. The fixed notebooks currently do not recompute them.

| Metric | Meaning | Formula |
| --- | --- | --- |
| `clip_oe_loss_control_a` | Open-endedness score/loss on CLIP embeddings of `control_a`. | `calc_open_endedness_score(z_control_a)` |
| `clip_oe_loss_control_b` | Same for `control_b`. | `calc_open_endedness_score(z_control_b)` |
| `clip_oe_loss_control_mean` | Mean control OE loss. | `0.5 * (clip_oe_loss_control_a + clip_oe_loss_control_b)` |
| `clip_oe_loss_walls` | OE loss on `walls`. | `calc_open_endedness_score(z_walls)` |
| `clip_oe_loss_walls_minus_control_mean` | Perturbed OE loss minus mean control OE loss. | `clip_oe_loss_walls - clip_oe_loss_control_mean` |
| `clip_oe_loss_walls_minus_control_a` | Perturbed OE loss minus anchor control OE loss. | `clip_oe_loss_walls - clip_oe_loss_control_a` |

## Recomputed Embedding Distance Metrics

These are recomputed in the notebook from saved `trial_XXXXX_embeddings.npz` artifacts.

The raw arrays are:

| Array | Meaning |
| --- | --- |
| `z_control_a` | Late-window CLIP embeddings for `control_a`. |
| `z_control_b` | Late-window CLIP embeddings for `control_b`. |
| `z_walls` | Late-window CLIP embeddings for `walls`. |

For each embedding distance family, the notebook computes the same triplet:

| Suffix | Meaning | Formula |
| --- | --- | --- |
| `__baseline_distance` | Control-control distance. | `d(control_a, control_b)` |
| `__walls_effect_distance_ctrl_a` | Anchor-to-walls distance. | `d(control_a, walls)` |
| `__walls_effect_distance_ctrl_b` | Second-control-to-walls distance. | `d(control_b, walls)` |
| `__walls_effect_distance` | Mean control-to-walls distance. | `0.5 * (d(control_a, walls) + d(control_b, walls))` |
| `__effect_minus_baseline` | Mean walls effect over baseline. | `__walls_effect_distance - __baseline_distance` |
| `__anchor_effect_minus_baseline` | Anchor walls effect over baseline. | `__walls_effect_distance_ctrl_a - __baseline_distance` |
| `__effect_over_baseline_ratio` | Relative mean walls effect. | `__walls_effect_distance / max(__baseline_distance, 1e-12)` |
| `__anchor_effect_over_baseline_ratio` | Relative anchor walls effect. | `__walls_effect_distance_ctrl_a / max(__baseline_distance, 1e-12)` |

Embedding distance families:

| Metric Prefix | Meaning | Formula |
| --- | --- | --- |
| `embedding_synced_cosine` | Time-synchronized cosine distance. | `mean_t(1 - dot(z_a[t], z_b[t]))` |
| `embedding_synced_euclidean` | Time-synchronized Euclidean distance. | `mean_t(||z_a[t] - z_b[t]||)` |
| `embedding_cloud_chamfer_cosine` | Chamfer distance between unordered clouds of embeddings. | `0.5 * (mean_i min_j d(z_a[i], z_b[j]) + mean_j min_i d(z_a[i], z_b[j]))` |

If `embeddings.normalize: true`, embeddings are L2-normalized before these distances are computed.

## MSC Scalar Metrics

These are recomputed in the notebook from saved `trial_XXXXX_lagrangian.npz` artifacts.

The raw arrays are:

| Array | Meaning |
| --- | --- |
| `xy_control_a` | Late-window Lagrangian trajectories for `control_a`. |
| `xy_control_b` | Late-window Lagrangian trajectories for `control_b`. |
| `xy_walls` | Late-window Lagrangian trajectories for `walls`. |

The notebook uses the same JAX scorer as paper-check/frustration optimization:

```python
make_metric_loss_fn(metric_cfg, include_maps=True)
```

The scorer computes `score = metric_alpha * amp + metric_beta * msc`. The `msc`
component is the paper-check scorer's weighted sum over configured scale pairs;
its absolute scale can therefore depend on the active scale-pair set. Use
`msc_amp_*`, `msc_metric_eps`, and the raw-vs-recomputed debug columns as sanity
checks when trajectories are almost static.

| Metric | Meaning | Formula |
| --- | --- | --- |
| `msc_score_control_a` | MSC score of `control_a`. | `score(xy_control_a)` |
| `msc_score_control_b` | MSC score of `control_b`. | `score(xy_control_b)` |
| `msc_score_control_mean` | Mean control MSC score. Best scalar for optimized-vs-random MSC comparison. | `0.5 * (msc_score_control_a + msc_score_control_b)` |
| `msc_score_walls` | MSC score of perturbed trajectory. | `score(xy_walls)` |
| `msc_score_walls_minus_control_a` | Perturbed score minus anchor control score. | `msc_score_walls - msc_score_control_a` |
| `msc_score_walls_minus_control_mean` | Perturbed score minus mean control score. | `msc_score_walls - msc_score_control_mean` |
| `msc_loss_control_a` | Loss version of `msc_score_control_a`. | `-msc_score_control_a` |
| `msc_loss_control_b` | Loss version of `msc_score_control_b`. | `-msc_score_control_b` |
| `msc_loss_control_mean` | Mean control loss. | `0.5 * (msc_loss_control_a + msc_loss_control_b)` |
| `msc_loss_walls` | Loss version of `msc_score_walls`. | `-msc_score_walls` |
| `msc_loss_walls_minus_control_a` | Perturbed loss minus anchor control loss. | `msc_loss_walls - msc_loss_control_a` |
| `msc_loss_walls_minus_control_mean` | Perturbed loss minus mean control loss. | `msc_loss_walls - msc_loss_control_mean` |
| `msc_amp_control_a` | Amplitude component of the MSC scorer for `control_a`. | `amp(xy_control_a)` |
| `msc_amp_control_b` | Amplitude component for `control_b`. | `amp(xy_control_b)` |
| `msc_amp_walls` | Amplitude component for `walls`. | `amp(xy_walls)` |
| `msc_component_control_a` | Multiscale component of the MSC scorer for `control_a`. | `msc(xy_control_a)` |
| `msc_component_control_b` | Multiscale component for `control_b`. | `msc(xy_control_b)` |
| `msc_component_walls` | Multiscale component for `walls`. | `msc(xy_walls)` |
| `msc_tau_best_steps_control_a` | Selected/best tau in simulation steps for `control_a`. | `tau_best_steps(xy_control_a)` |
| `msc_tau_best_steps_control_b` | Selected/best tau in simulation steps for `control_b`. | `tau_best_steps(xy_control_b)` |
| `msc_tau_best_steps_walls` | Selected/best tau in simulation steps for `walls`. | `tau_best_steps(xy_walls)` |
| `msc_score_anchor_absdiff_minus_baseline` | Whether walls is farther from anchor than the second control is, in absolute MSC-score space. | `abs(score_walls - score_control_a) - abs(score_control_b - score_control_a)` |
| `msc_loss_anchor_absdiff_minus_baseline` | Same as above in loss space. | `abs(loss_walls - loss_control_a) - abs(loss_control_b - loss_control_a)` |
| `msc_sample_every_steps` | Sampling stride used for MSC trajectories. | `metric_cfg["sample_every_steps"]` |
| `msc_time_sampling` | Number of saved trajectory frames used by MSC. | `xy_control_a.shape[0]` |

## Delta-H Map Distance Metrics

These are recomputed from the `delta_h_map` returned by the paper-check MSC scorer.

For each map distance family, the same triplet suffixes are used:

```text
__baseline_distance
__walls_effect_distance_ctrl_a
__walls_effect_distance_ctrl_b
__walls_effect_distance
__effect_minus_baseline
__anchor_effect_minus_baseline
__effect_over_baseline_ratio
__anchor_effect_over_baseline_ratio
```

Map distance families:

| Metric Prefix | Meaning | Formula |
| --- | --- | --- |
| `delta_h_l2` | L2 distance between flattened `delta_h_map`s. | `||flatten(map_a - map_b)||_2` |
| `delta_h_mean_abs` | Mean absolute difference between flattened maps. | `mean(abs(flatten(map_a - map_b)))` |
| `delta_h_cosine` | Cosine distance between flattened maps, if enabled. | `1 - dot(norm(map_a), norm(map_b))` |

## Delta-H Distribution Distance Metrics

These compare distributions of Delta-H values at one fixed tau slice.

The fixed tau is selected as follows:

- if `metric_tau_mode: fixed`, use `trajectories.metric_tau_steps` or `metric_tau_frames`;
- otherwise use `fixed_tau_distribution_steps` or `fixed_tau_distribution_frames`.

| Metric Prefix | Meaning | Formula |
| --- | --- | --- |
| `delta_h_dist_tauXX_wasserstein` | 1D Wasserstein distance between fixed-tau Delta-H distributions. | Integral distance between empirical CDFs. |
| `delta_h_dist_tauXX_ks` | Kolmogorov-Smirnov distance. | `max_x abs(CDF_a(x) - CDF_b(x))` |
| `delta_h_dist_tauXX_energy` | Energy distance on fixed-tau values. | `sqrt(max(0, 2E|A-B| - E|A-A'| - E|B-B'|))` |
| `delta_h_dist_tauXX_*_zscore` | Same distribution distance after per-vector z-score normalization. | `(values - mean(values)) / std(values)` before distance. |

## Run-Level Optimized-Vs-Random Metrics

The fixed notebooks aggregate trial rows into one row per `optimized_run_idx`.

For each metric `M`:

| Run-Level Metric | Meaning | Formula |
| --- | --- | --- |
| `M__optimized` | Value for the optimized candidate. | optimized row value |
| `M__random_median` | Median over random candidates in the same `optimized_run_idx`. | `median(M_randoms)` |
| `M__random_mean` | Mean over random candidates in the same `optimized_run_idx`. | `mean(M_randoms)` |
| `M__control_median` | Current configured control aggregate. With `CONTROL_KINDS=('random',)` and `CONTROL_AGG='median'`, this equals `M__random_median`. | `median(M_controls)` |
| `M__delta_vs_control_median` | Optimized improvement over random controls. | `M__optimized - M__control_median` |

The explicit MSC optimized-vs-random metric is:

| Metric | Meaning | Formula |
| --- | --- | --- |
| `msc_score_control_mean__optimized_minus_random_median` | Optimized MSC score minus median random MSC score within the same run group. | `msc_score_control_mean__optimized - msc_score_control_mean__random_median` |

Interpretation:

| Sign | Meaning |
| --- | --- |
| `> 0` | Optimized is higher than random for this metric. |
| `= 0` | No difference. |
| `< 0` | Optimized is lower than random for this metric. |

For loss metrics, lower is usually better, so the sign interpretation is inverted unless the metric name explicitly says `score`.

## Artifact Coverage Flags

These are added by the notebook to make stale or missing artifacts visible.

| Column | Meaning |
| --- | --- |
| `history_embeddings_artifact_found` | `true` if the saved embeddings `.npz` was found and embedding metrics were recomputed. |
| `history_trajectory_artifact_found` | `true` if the saved Lagrangian `.npz` was found and MSC/Delta-H metrics were recomputed. |

If a raw artifact is missing, recomputed metrics are left as `NaN` rather than silently using stale values from `trial_results.csv`.
