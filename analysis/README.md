# Analysis Notebooks

Unified notebook space for experiment analysis lives under [`analysis/notebooks`](notebooks).

Current layout:

- `flow_lenia/`: Flow-Lenia analysis notebooks, including `analyse_p.ipynb` and `flow_drift_metrics_demo.ipynb`
- `mspd/`: MSPD / DeltaH analysis notebooks
- `legacy/`: older exploratory and demo notebooks that were previously stored at repo root

Most notebooks in this folder now bootstrap `REPO_ROOT` automatically so they can be opened from their new location without manually fixing relative paths first.
