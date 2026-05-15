# Experiments

This directory contains runnable experiment definitions: YAML configs, small runner scripts, and lightweight metadata needed to reproduce a run.

Generated checkpoints, videos, plots, APF logs, W&B caches, and temporary tables should stay out of Git. Put them under a configured `checkpoints/`, `outputs/`, or `figures/` directory, or under the top-level `artifacts/` directory for loose local files.

Historical names are recorded in [`../docs/experiment_rename_map.md`](../docs/experiment_rename_map.md).

## Active Layout

| Path | Purpose |
|---|---|
| `flow_lenia_mspd/` | Flow-Lenia MSPD optimization, simulation, APF logging, and minibang posthoc experiments. |
| `flow_lenia_apf_rollouts/` | Stored APF/Lagrangian rollout definitions for Flow-Lenia comparison analyses. |
| `paper_check_flow_lenia/` | Flow-Lenia paper-check orchestration over optimized runs and random baselines. |
| `boids_mspd/` | Boids MSPD optimization and simulation configs. |
| `paper_check_boids/` | Boids paper-check orchestration. |
| `plife_plus_mspd/` | Particle Life Plus MSPD, CLIP-OE, simulation, and tau-sweep configs. |
| `paper_check_plife_plus/` | Particle Life Plus paper-check orchestration. |
| `legacy/` | Older/prototype experiment definitions kept for provenance. Prefer active directories for new work. |
| `_templates/` | Copyable skeletons for adding new experiment families. |

## Naming

Use names that state the substrate and the experiment family:

```text
experiments/<substrate>_<method_or_question>/<stage>/
  config.yaml
  run.sh
```

Examples:

```text
experiments/flow_lenia_mspd/optimization/
experiments/plife_plus_mspd/tau_sweep/
experiments/paper_check_boids/frustration_simulation/
```

Avoid timestamp-only names for top-level directories. Timestamps are fine for run IDs inside `checkpoints/` or `outputs/`.

## Runner Conventions

- Keep `run.sh` next to the config it executes.
- Resolve `repo_root` inside shell scripts so they work from any current directory.
- Put long-lived defaults in `config.yaml`; use CLI overrides only for machine sharding or one-off local debugging.
- If an experiment supports several baselines, prefer explicit config names such as `config_random.yaml` or `config_clip_oe.yaml`.

## Templates

Start new experiment families from:

```text
experiments/_templates/optimization/
experiments/_templates/paper_check/
```

After copying a template, replace every `<...>` placeholder before running it.
