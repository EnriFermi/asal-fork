# Experiment Rename Map

This file records old experiment names and the cleaned-up names used after the repository organization pass.

Use this when searching older notebooks, logs, W&B runs, paper notes, or archived result folders that still refer to historical paths.

## Active Experiment Directories

| Old path | New path | Notes |
|---|---|---|
| `experiments/opt_msc` | `experiments/flow_lenia_mspd` | Flow-Lenia MSPD optimization, simulation, and minibang posthoc pipeline. |
| `experiments/opt_msc_boids` | `experiments/boids_mspd` | Boids MSPD optimization and simulation configs. |
| `experiments/opt_msc_plife_plus` | `experiments/plife_plus_mspd` | Particle Life Plus MSPD, CLIP-OE, simulation, and tau sweep configs. |
| `experiments/paper_check` | `experiments/paper_check_flow_lenia` | Flow-Lenia paper proof-package orchestration. |
| `experiments/log_apf` | `experiments/flow_lenia_apf_rollouts` | APF/Lagrangian rollout logging for Flow-Lenia comparison analyses. |

## Paper-Check Directories Kept As-Is

These names are already explicit enough and are not planned for rename.

| Path | Notes |
|---|---|
| `experiments/paper_check_boids` | Boids paper-check orchestration. |
| `experiments/paper_check_plife_plus` | Particle Life Plus paper-check orchestration. |

## Legacy Experiment Directories

| Old path | New path | Notes |
|---|---|---|
| `experiments/opt_1` | `experiments/legacy/opt_1` | Early CLIP/OE and bootstrap variance experiments. |
| `experiments/opt_halving` | `experiments/legacy/opt_halving` | Early halving optimization experiments. |
| `experiments/opt_online` | `experiments/legacy/opt_online` | Early online optimization experiments. |
| `experiments/root_best` | `experiments/legacy/root_best` | One-off root-best simulation configs. |
| `experiments/frustration` | `experiments/legacy/frustration` | Earlier frustration/history-dependence protocol; superseded by `paper_check_flow_lenia`. |

## Local Result/Artifact Moves

These are not active experiment definitions. They were moved out of the code/config tree.

| Old path | New path | Notes |
|---|---|---|
| `experiments 3` | `artifacts/legacy_experiment_runs/experiments_3` | Large local result folder. Ignored by Git. |
| `experiments/plots` | `artifacts/loose/experiments_plots` | Local generated plots. Ignored by Git. |
| `experiments/opt_msc/checkpoints/*` | `artifacts/experiment_checkpoints/flow_lenia_mspd/*` | Historical local checkpoint outputs. Ignored by Git. Fresh runs will recreate `experiments/flow_lenia_mspd/checkpoints/` as needed. |

## Root-Level Script Moves

Old one-off shell wrappers from the repository root now live in:

```text
scripts/legacy_entrypoints/
```

Tracking utilities that used to live at the root now live in:

```text
tools/tracking/
```
