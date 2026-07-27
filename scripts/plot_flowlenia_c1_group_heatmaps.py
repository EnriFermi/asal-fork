#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _load_rows(scores_csv: Path, group_idx: int) -> list[dict[str, object]]:
    with scores_csv.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if _int(row, "optimized_run_idx") == group_idx
        ]
    if not rows:
        raise ValueError(f"No rows for optimized_run_idx={group_idx} in {scores_csv}")

    loaded: list[dict[str, object]] = []
    for row in rows:
        maps_path = Path(row["maps_path"])
        if not maps_path.is_file():
            raise FileNotFoundError(maps_path)
        with np.load(maps_path) as maps:
            loaded.append(
                {
                    "row": row,
                    "selection": np.asarray(maps["delta_h_selection"], dtype=np.float64),
                    "eval": np.asarray(maps["delta_h_eval"], dtype=np.float64),
                    "selection_score_by_tau": np.asarray(
                        maps["selection_score_by_tau"], dtype=np.float64
                    ),
                    "eval_score_by_tau": np.asarray(maps["eval_score_by_tau"], dtype=np.float64),
                    "tau_steps": np.asarray(maps["tau_steps"], dtype=np.int64),
                    "selected_tau_idx": int(maps["selected_tau_idx"]),
                }
            )
    return loaded


def _ordered_grid(rows: list[dict[str, object]]) -> tuple[list[int], list[list[dict[str, object]]]]:
    seeds = sorted({_int(item["row"], "rollout_seed_idx") for item in rows})
    grid: list[list[dict[str, object]]] = []
    for seed in seeds:
        seed_rows = [item for item in rows if _int(item["row"], "rollout_seed_idx") == seed]
        optimized = [item for item in seed_rows if item["row"]["candidate_kind"] == "optimized"]
        randoms = sorted(
            (item for item in seed_rows if item["row"]["candidate_kind"] == "random"),
            key=lambda item: _int(item["row"], "candidate_idx"),
        )
        if len(optimized) != 1:
            raise ValueError(f"Expected one optimized row for seed={seed}, found {len(optimized)}")
        if not randoms:
            raise ValueError(f"No random rows for seed={seed}")
        grid.append([optimized[0], *randoms])
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent candidate count by seed: {sorted(widths)}")
    return seeds, grid


def _symmetric_limit(arrays: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(arr[np.isfinite(arr)]) for arr in arrays])
    return max(float(np.percentile(finite, 98.0)), 1e-12)


def _tau_ticks(ax: object, tau_steps: np.ndarray) -> None:
    idx = np.unique(np.linspace(0, tau_steps.size - 1, num=min(6, tau_steps.size), dtype=int))
    ax.set_yticks(idx, [str(int(tau_steps[i])) for i in idx])


def _plot_seed_grid(
    group_idx: int,
    split: str,
    seeds: list[int],
    grid: list[list[dict[str, object]]],
    output_dir: Path,
) -> Path:
    arrays = [np.asarray(item[split]) for row in grid for item in row]
    limit = _symmetric_limit(arrays)
    n_rows = len(grid)
    n_cols = len(grid[0])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols + 0.8, 2.45 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for row_idx, (seed, candidates) in enumerate(zip(seeds, grid)):
        for col_idx, item in enumerate(candidates):
            row = item["row"]
            arr = np.asarray(item[split])
            ax = axes[row_idx, col_idx]
            last_image = ax.imshow(
                arr,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            tau_idx = int(item["selected_tau_idx"])
            ax.axhline(tau_idx, color="#111111", linewidth=0.9, linestyle="--")
            if row["candidate_kind"] == "optimized":
                label = "optimized"
            else:
                label = f"random {int(float(row['candidate_idx']))}"
            score = float(row[f"{split}_score_mspd"])
            ax.set_title(f"{label}; MSPD={score:.3g}", fontsize=9)
            ax.set_xlabel(f"{split} window")
            if col_idx == 0:
                ax.set_ylabel(f"seed {seed}\ntau steps")
                _tau_ticks(ax, np.asarray(item["tau_steps"]))
            else:
                ax.set_yticks([])
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, shrink=0.82, pad=0.015, label="Delta-H")
    fig.suptitle(f"Flow-Lenia C1 run_{group_idx:03d}: {split} Delta-H by seed and candidate")
    output = output_dir / f"c1_flow_lenia_run_{group_idx:03d}_delta_h_{split}_all_opt_random.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_medians(group_idx: int, rows: list[dict[str, object]], output_dir: Path) -> Path:
    optimized = [item for item in rows if item["row"]["candidate_kind"] == "optimized"]
    randoms = [item for item in rows if item["row"]["candidate_kind"] == "random"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), squeeze=False, constrained_layout=True)
    for row_idx, split in enumerate(("selection", "eval")):
        opt_median = np.median(np.stack([np.asarray(item[split]) for item in optimized]), axis=0)
        random_median = np.median(np.stack([np.asarray(item[split]) for item in randoms]), axis=0)
        difference = opt_median - random_median
        base_limit = _symmetric_limit([opt_median, random_median])
        diff_limit = _symmetric_limit([difference])
        panels = (
            (opt_median, "optimized median", base_limit),
            (random_median, "random median", base_limit),
            (difference, "optimized - random", diff_limit),
        )
        for col_idx, (arr, label, limit) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                arr,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"{split}: {label}")
            ax.set_xlabel(f"{split} window")
            ax.set_ylabel("tau index")
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Delta-H")
    fig.suptitle(f"Flow-Lenia C1 run_{group_idx:03d}: aggregate Delta-H")
    output = output_dir / f"c1_flow_lenia_run_{group_idx:03d}_delta_h_medians.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_scores_by_tau(group_idx: int, rows: list[dict[str, object]], output_dir: Path) -> Path:
    optimized = [item for item in rows if item["row"]["candidate_kind"] == "optimized"]
    randoms = [item for item in rows if item["row"]["candidate_kind"] == "random"]
    tau_steps = np.asarray(rows[0]["tau_steps"], dtype=np.int64)
    selected_tau_idx = int(rows[0]["selected_tau_idx"])

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    styles = (
        ("optimized selection", optimized, "selection_score_by_tau", "#d62728", "-", "o"),
        ("optimized eval", optimized, "eval_score_by_tau", "#d62728", "--", "s"),
        ("random selection", randoms, "selection_score_by_tau", "#1f77b4", "-", "o"),
        ("random eval", randoms, "eval_score_by_tau", "#1f77b4", "--", "s"),
    )
    medians: dict[str, np.ndarray] = {}
    for label, items, key, color, linestyle, marker in styles:
        values = np.stack([np.asarray(item[key], dtype=np.float64) for item in items])
        median = np.nanmedian(values, axis=0)
        medians[label] = median
        ax.plot(
            tau_steps,
            median,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            linewidth=2.6,
            label=label,
        )

    selected_tau = int(tau_steps[selected_tau_idx])
    ax.axvline(selected_tau, color="#111111", linestyle=":", linewidth=1.4, label=f"train tau={selected_tau}")
    selected_values = "\n".join(
        f"{label}: {values[selected_tau_idx]:.6f}" for label, values in medians.items()
    )
    ax.text(
        0.02,
        0.97,
        f"At tau={selected_tau}\n{selected_values}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92, "pad": 5},
    )
    ax.set_xticks(tau_steps)
    ax.margins(x=0.04, y=0.15)
    ax.set_xlabel("tau steps")
    ax.set_ylabel("MSPD")
    ax.set_title(f"Flow-Lenia C1 run_{group_idx:03d}: median selection and eval MSPD by tau")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    output = output_dir / f"c1_flow_lenia_run_{group_idx:03d}_selection_eval_scores_by_tau.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-plot-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.scores_csv, args.group)
    if args.tau_plot_only:
        outputs = [_plot_scores_by_tau(args.group, rows, args.output_dir)]
    else:
        seeds, grid = _ordered_grid(rows)
        outputs = [
            _plot_seed_grid(args.group, split, seeds, grid, args.output_dir)
            for split in ("selection", "eval")
        ]
        outputs.append(_plot_medians(args.group, rows, args.output_dir))
        outputs.append(_plot_scores_by_tau(args.group, rows, args.output_dir))
    summary = {
        "group": args.group,
        "n_optimized": sum(item["row"]["candidate_kind"] == "optimized" for item in rows),
        "n_random": sum(item["row"]["candidate_kind"] == "random" for item in rows),
        "outputs": [str(path.resolve()) for path in outputs],
    }
    summary_path = args.output_dir / f"c1_flow_lenia_run_{args.group:03d}_delta_h_heatmaps.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
