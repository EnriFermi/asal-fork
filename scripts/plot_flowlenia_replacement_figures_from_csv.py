from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = (5000, 10000, 15000, 20000, 30000)

OPT_COLOR = "#1769AA"
RANDOM_COLOR = "#9AA0A6"
POSITIVE_COLOR = "#287D8E"
NEGATIVE_COLOR = "#B8574D"
MID_COLOR = "#D99000"
LOW_COLOR = "#377EB8"
HIGH_COLOR = "#C23B45"


def _matplotlib() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "flowlenia_article_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_figure(fig: Any, output_dir: Path, name: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 240}),
        (".pdf", {}),
    ):
        path = output_dir / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def _jitter(count: int, width: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    return np.linspace(-width, width, count, dtype=np.float64)


def _plot_c1(data_root: Path, output_dir: Path) -> list[str]:
    plt = _matplotlib()
    scores = pd.read_csv(data_root / "c1_rollout_scores.csv")
    expected_runs = list(range(10))

    fig, ax = plt.subplots(figsize=(7.15, 3.55), constrained_layout=True)
    for run_idx in expected_runs:
        group = scores[scores["run_idx"].astype(int) == run_idx]
        optimized = (
            group[group["candidate_kind"] == "optimized"]["eval_score_mspd"]
            .astype(float)
            .to_numpy()
            * 1000.0
        )
        random = (
            group[group["candidate_kind"] == "random"]["eval_score_mspd"]
            .astype(float)
            .to_numpy()
            * 1000.0
        )
        if optimized.size != 4 or random.size != 12:
            raise ValueError(
                f"C1 run {run_idx:03d}: expected 4 optimized and 12 random "
                f"rollout scores, found {optimized.size} and {random.size}."
            )
        ax.scatter(
            run_idx - 0.12 + _jitter(random.size, 0.07),
            np.sort(random),
            s=18,
            color=RANDOM_COLOR,
            alpha=0.72,
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            run_idx + 0.12 + _jitter(optimized.size, 0.045),
            np.sort(optimized),
            s=28,
            color=OPT_COLOR,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        random_median = float(np.median(random))
        optimized_median = float(np.median(optimized))
        ax.plot(
            [run_idx - 0.25, run_idx - 0.01],
            [random_median, random_median],
            color="#202124",
            linewidth=1.7,
            zorder=4,
        )
        ax.scatter(
            [run_idx + 0.12],
            [optimized_median],
            marker="D",
            s=35,
            color=OPT_COLOR,
            edgecolors="#202124",
            linewidths=0.45,
            zorder=4,
        )
        ax.plot(
            [run_idx - 0.01, run_idx + 0.12],
            [random_median, optimized_median],
            color=(
                POSITIVE_COLOR
                if optimized_median > random_median
                else NEGATIVE_COLOR
            ),
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )

    from matplotlib.lines import Line2D

    legend = [
        Line2D(
            [], [], marker="o", linestyle="none", color=RANDOM_COLOR,
            label="random rollouts"
        ),
        Line2D(
            [], [], marker="o", linestyle="none", color=OPT_COLOR,
            label="optimized rollouts"
        ),
        Line2D(
            [], [], color="#202124", linewidth=1.7, label="random median"
        ),
        Line2D(
            [], [], marker="D", linestyle="none", color=OPT_COLOR,
            markeredgecolor="#202124", label="optimized median"
        ),
    ]
    ax.legend(handles=legend, ncol=2, frameon=False, loc="upper left")
    ax.axhline(0.0, color="#DADCE0", linewidth=0.7, zorder=0)
    ax.set_xticks(expected_runs)
    ax.set_xticklabels([f"{idx:03d}" for idx in expected_runs])
    ax.set_xlabel("optimization run")
    ax.set_ylabel(r"optimizer-matched MSPD ($\times 10^{-3}$)")
    ax.set_title("Flow-Lenia C1: optimized and matched-random rollout scores")
    ax.text(
        0.99,
        0.96,
        "9/10 positive matched contrasts",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    outputs = _save_figure(fig, output_dir, "flow_c1_paired_raw_paper")
    plt.close(fig)
    return outputs


def _plot_c1_tau(data_root: Path, output_dir: Path) -> list[str]:
    plt = _matplotlib()
    tau_summary = pd.read_csv(data_root / "c1_tau_profiles_summary.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.05),
        sharey=True,
        constrained_layout=True,
    )
    for ax, split, title in zip(
        axes,
        ("selection", "evaluation"),
        ("interleaved split A", "interleaved split B"),
    ):
        for kind, color, label in (
            ("random", RANDOM_COLOR, "random"),
            ("optimized", OPT_COLOR, "optimized"),
        ):
            values = tau_summary[
                (tau_summary["split"] == split)
                & (tau_summary["candidate_kind"] == kind)
            ].sort_values("tau_steps")
            x = values["tau_steps"].to_numpy(dtype=float) / 1000.0
            median = values["median"].to_numpy(dtype=float) * 1000.0
            q25 = values["q25"].to_numpy(dtype=float) * 1000.0
            q75 = values["q75"].to_numpy(dtype=float) * 1000.0
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                x,
                median,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.2,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel(r"lag $\tau$ (thousand steps)")
        ax.set_xticks(np.arange(1, 11, 1))
        ax.grid(axis="y", color="#ECEFF1", linewidth=0.7)
    axes[0].set_ylabel(r"MSPD ($\times 10^{-3}$)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Flow-Lenia C1 lag-profile diagnostic", fontsize=10.5)
    outputs = _save_figure(fig, output_dir, "flow_c1_tau_profiles")
    plt.close(fig)
    return outputs


def _plot_c2(data_root: Path, output_dir: Path) -> list[str]:
    plt = _matplotlib()
    run_horizon = pd.read_csv(data_root / "c2_rng_only_run_horizon.csv")
    run_aggregate = pd.read_csv(data_root / "c2_rng_only_run_aggregate.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.45),
        constrained_layout=True,
    )
    left, right = axes
    x = np.asarray(HORIZONS, dtype=np.float64) / 1000.0
    pivot = run_horizon.pivot(
        index="run_idx",
        columns="horizon_steps",
        values="mean_high_minus_low",
    ).reindex(columns=HORIZONS)
    for _, values in pivot.iterrows():
        left.plot(
            x,
            values.to_numpy(dtype=float) * 1000.0,
            color="#B0B6BB",
            linewidth=0.8,
            alpha=0.7,
            marker="o",
            markersize=2.4,
        )
    median_by_horizon = pivot.median(axis=0).to_numpy(dtype=float) * 1000.0
    left.plot(
        x,
        median_by_horizon,
        color=POSITIVE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        label="median across runs",
    )
    left.axhline(0.0, color="#202124", linewidth=0.8)
    left.set_xticks(x)
    left.set_xlabel("continuation horizon (thousand steps)")
    left.set_ylabel(r"high $-$ low CLIP--Chamfer ($\times 10^{-3}$)")
    left.set_title("A. Horizon dependence")
    left.legend(frameon=False, loc="upper left")

    run_ids = run_aggregate["run_idx"].astype(int).to_numpy()
    effects = run_aggregate["mean_high_minus_low"].to_numpy(dtype=float)
    scaled_effects = effects * 1000.0
    colors = [
        POSITIVE_COLOR if value > 0.0 else NEGATIVE_COLOR
        for value in effects
    ]
    right.bar(run_ids, scaled_effects, color=colors, width=0.72)
    right.axhline(0.0, color="#202124", linewidth=0.8)
    right.set_xticks(run_ids)
    right.set_xticklabels([f"{value:03d}" for value in run_ids])
    right.set_xlabel("source optimization run")
    right.set_ylabel(
        r"mean high $-$ low CLIP--Chamfer ($\times 10^{-3}$)"
    )
    right.set_title("B. Run-level aggregate")
    right.text(
        0.98,
        0.95,
        "10/10 positive",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )
    fig.suptitle(
        "Flow-Lenia C2: intrinsic RNG-only future divergence",
        fontsize=10.5,
    )
    outputs = _save_figure(
        fig,
        output_dir,
        "flow_c2_pooled_and_within_paper",
    )
    plt.close(fig)
    return outputs


def _plot_c2_selection(data_root: Path, output_dir: Path) -> list[str]:
    plt = _matplotlib()
    points = pd.read_csv(data_root / "c2_branch_selection_points.csv")
    cells = pd.read_csv(data_root / "c2_branch_selection_heatmaps.csv")
    vmax = float(np.quantile(cells["processed_delta_h"].to_numpy(float), 0.99))

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12.2, 5.35),
        constrained_layout=True,
    )
    last_image = None
    marker_meta = {
        "high": (HIGH_COLOR, "v", 8.9),
        "mid": (MID_COLOR, "o", 4.5),
        "low": (LOW_COLOR, "^", 0.1),
    }
    for run_idx, ax in enumerate(axes.flat):
        run_cells = cells[cells["run_idx"].astype(int) == run_idx]
        heatmap = (
            run_cells.pivot(
                index="tau_idx",
                columns="window_idx",
                values="processed_delta_h",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
            .to_numpy(dtype=float)
        )
        tau_steps = (
            run_cells[["tau_idx", "tau_steps"]]
            .drop_duplicates()
            .sort_values("tau_idx")["tau_steps"]
            .to_numpy(dtype=np.int64)
        )
        centers = (
            run_cells[["window_idx", "window_center_steps"]]
            .drop_duplicates()
            .sort_values("window_idx")["window_center_steps"]
            .to_numpy(dtype=np.int64)
        )
        last_image = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        run_points = points[points["run_idx"].astype(int) == run_idx]
        for point in run_points.itertuples(index=False):
            color, marker, y = marker_meta[str(point.condition)]
            window_idx = int(np.argmin(np.abs(centers - int(point.step))))
            ax.axvline(window_idx, color=color, linewidth=0.7, alpha=0.65)
            ax.scatter(
                [window_idx],
                [y],
                marker=marker,
                s=22,
                color=color,
                edgecolors="white",
                linewidths=0.3,
                zorder=4,
            )
        x_ticks = np.linspace(0, centers.size - 1, 4).astype(int)
        y_ticks = np.asarray([0, 3, 6, 9], dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [f"{centers[idx] / 1000:.0f}" for idx in x_ticks]
        )
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"{tau_steps[idx] / 1000:.0f}" for idx in y_ticks]
        )
        ax.set_title(f"opt_{run_idx:03d}")
        if run_idx >= 5:
            ax.set_xlabel("branch time (k steps)")
        if run_idx % 5 == 0:
            ax.set_ylabel(r"lag $\tau$ (k steps)")
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            shrink=0.78,
            pad=0.012,
            label=r"processed $\Delta H$",
        )

    from matplotlib.lines import Line2D

    legend = [
        Line2D(
            [], [], marker="v", linestyle="none", color=HIGH_COLOR, label="high"
        ),
        Line2D(
            [], [], marker="o", linestyle="none", color=MID_COLOR, label="mid"
        ),
        Line2D(
            [], [], marker="^", linestyle="none", color=LOW_COLOR, label="low"
        ),
    ]
    fig.legend(
        handles=legend,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
    )
    outputs = _save_figure(
        fig,
        output_dir,
        "c2_branching_branch_selection_preview",
    )
    plt.close(fig)
    return outputs


def _plot_c5(data_root: Path, output_dir: Path) -> list[str]:
    plt = _matplotlib()
    table = pd.read_csv(data_root / "c5_5000_run_summary.csv")
    table = table.sort_values("run_idx").reset_index(drop=True)
    values = table["contrast_excess_clip_post_release"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 3.2), constrained_layout=True)
    run_ids = table["run_idx"].astype(int).to_numpy()
    scaled = values * 1000.0
    colors = [
        POSITIVE_COLOR if value > 0.0 else NEGATIVE_COLOR
        for value in values
    ]
    ax.bar(run_ids, scaled, color=colors, width=0.72)
    ax.axhline(0.0, color="#202124", linewidth=0.8)
    ax.set_xticks(run_ids)
    ax.set_xticklabels([f"{value:03d}" for value in run_ids])
    ax.set_xlabel("optimization run")
    ax.set_ylabel(
        r"optimized $-$ random-median frustration ($\times 10^{-3}$)"
    )
    ax.set_title(
        "Flow-Lenia C5: post-release excess CLIP--Chamfer contrast\n"
        "5k sensitivity horizon"
    )
    ax.text(
        0.02,
        0.95,
        "8/10 positive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.0,
        },
    )
    outputs = _save_figure(fig, output_dir, "flow_c5_frustration_paper")
    plt.close(fig)
    return outputs


def run(data_root: Path, output_dir: Path) -> dict[str, list[str]]:
    return {
        "flow_c1_paired_raw_paper": _plot_c1(data_root, output_dir),
        "flow_c1_tau_profiles": _plot_c1_tau(data_root, output_dir),
        "flow_c2_pooled_and_within_paper": _plot_c2(data_root, output_dir),
        "c2_branching_branch_selection_preview": _plot_c2_selection(
            data_root,
            output_dir,
        ),
        "flow_c5_frustration_paper": _plot_c5(data_root, output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replot the five canonical Flow-Lenia replacement figures using "
            "only the portable CSV tables shipped with the figure package."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced_figures"),
    )
    args = parser.parse_args()
    outputs = run(args.data_root.resolve(), args.output_dir.resolve())
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
