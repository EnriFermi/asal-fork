#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="eval_score_mspd")
    args = parser.parse_args()

    with args.scores_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["candidate_kind"] == "optimized"]
    if not rows:
        raise ValueError(f"No optimized rows in {args.scores_csv}")
    if args.metric not in rows[0]:
        raise ValueError(f"Missing metric column {args.metric!r} in {args.scores_csv}")

    groups = sorted({_int(row, "optimized_run_idx") for row in rows})
    seed_indices = sorted({_int(row, "rollout_seed_idx") for row in rows})
    group_pos = {value: idx for idx, value in enumerate(groups)}
    seed_pos = {value: idx for idx, value in enumerate(seed_indices)}
    values = np.full((len(seed_indices), len(groups)), np.nan, dtype=np.float64)
    absolute_seeds = np.full_like(values, -1, dtype=np.int64)

    for row in rows:
        group_idx = _int(row, "optimized_run_idx")
        seed_idx = _int(row, "rollout_seed_idx")
        pos = (seed_pos[seed_idx], group_pos[group_idx])
        if np.isfinite(values[pos]):
            raise ValueError(f"Duplicate optimized row for group={group_idx}, seed_idx={seed_idx}")
        values[pos] = float(row[args.metric])
        absolute_seeds[pos] = _int(row, "run_seed")

    if not np.isfinite(values).all():
        raise ValueError("Incomplete group x seed matrix")
    if np.any(values <= 0.0):
        raise ValueError("Log-scaled MSPD heatmap requires strictly positive scores")

    norm = LogNorm(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    fig, ax = plt.subplots(figsize=(14.5, 5.2), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", norm=norm)
    ax.set_xticks(np.arange(len(groups)), [f"opt_{group:03d}" for group in groups])
    ax.set_yticks(np.arange(len(seed_indices)), [f"seed_idx={seed}" for seed in seed_indices])
    ax.set_xlabel("optimization run")
    ax.set_ylabel("random seed slot")
    metric_label = args.metric.removesuffix("_score_mspd").replace("_", " ")
    ax.set_title(f"Flow-Lenia C1 {metric_label} MSPD by optimization run and seed")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = float(values[row_idx, col_idx])
            normalized = float(norm(value))
            color = "white" if normalized < 0.2 or normalized > 0.72 else "#111111"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2e}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="semibold",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("MSPD (log color scale)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
