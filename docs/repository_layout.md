# Repository Layout

This fork keeps code, experiment definitions, analysis, paper notes, and local artifacts separated.

## Top Level

| Path | Purpose |
|---|---|
| `scripts/` | Main executable Python entrypoints and experiment helpers. |
| `experiments/` | Reproducible experiment configs and small runner scripts. |
| `analysis/` | Notebooks and reusable offline analysis code. |
| `tools/` | Research utilities that are not part of the core optimization pipeline. |
| `substrates/` | ALife substrate implementations. |
| `foundation_models/` | CLIP/DINO/SigLIP/pixel model wrappers. |
| `configs/` | Shared third-party/tool configs. |
| `docs/` | Paper notes, formal descriptions, and repo documentation. |
| `artifacts/` | Local/generated outputs. This directory is intentionally git-ignored. |

## Experiment Conventions

Use this layout for new experiments:

```text
experiments/<substrate>_<method_or_question>/<stage>/
  config.yaml
  run.sh
```

Current active families:

| Family | Use |
|---|---|
| `paper_check_flow_lenia/` | Flow-Lenia paper proof package. |
| `paper_check_boids/` | Boids proof package. |
| `paper_check_plife_plus/` | Particle Life Plus proof package. |
| `flow_lenia_mspd/` | Flow-Lenia MSPD optimization and posthoc minibang experiments. |
| `flow_lenia_apf_rollouts/` | Flow-Lenia APF/Lagrangian rollout definitions for comparison analyses. |
| `boids_mspd/` | Boids MSPD experiments. |
| `plife_plus_mspd/` | Particle Life Plus MSPD experiments. |
| `legacy/` | Older/prototype experiment definitions kept for provenance. |
| `_templates/` | Copyable skeletons for new experiment families. |

Generated checkpoints, videos, plots, and scratch results should go under either the configured `checkpoints/` directory or `artifacts/`, not the repository root.

See `experiments/README.md` for detailed conventions and `docs/experiment_rename_map.md` for historical names.

## Legacy Entrypoints

Old one-off shell wrappers were moved to:

```text
scripts/legacy_entrypoints/
```

They are kept for provenance and quick manual reruns, but new experiments should prefer YAML configs plus a small local `run.sh`.

## Tracking Tools

Flow-Lenia tracking utilities were moved out of the root into:

```text
tools/tracking/
```

The benchmark wrapper is still `scripts/bench.py`, now importing tracking helpers from `tools.tracking`.

## Paper Notes

Claim/proof-package notes live in:

```text
docs/paper/
```

Current files:

| File | Purpose |
|---|---|
| `docs/paper/claims.md` | Claims that need experimental support. |
| `docs/paper/claims_experiment_mapping.md` | Mapping from claims to experiments available in this codebase. |

Historical experiment renames are recorded in `docs/experiment_rename_map.md`.
