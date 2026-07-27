#!/usr/bin/env python3
"""Plot seed-matched C1 MSPD against C5 divergence for all 40 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C1 = (
    PROJECT_ROOT
    / "analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper"
    / "flow_lenia/checkpoint_scores.csv"
)
DEFAULT_C5_ROOT = (
    PROJECT_ROOT
    / "analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper"
    / "flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-scores", type=Path, default=DEFAULT_C1)
    parser.add_argument("--c5-root", type=Path, default=DEFAULT_C5_ROOT)
    return parser.parse_args()


def load_data(c1_scores: Path, c5_root: Path) -> pd.DataFrame:
    c1 = pd.read_csv(c1_scores)
    c1 = c1[c1["rollout_seed_idx"].eq(0)].copy()
    c1["candidate_idx_key"] = np.where(
        c1["candidate_kind"].eq("optimized"), 0, c1["candidate_idx"]
    ).astype(int)
    c1 = c1[
        [
            "optimized_run_idx",
            "candidate_kind",
            "candidate_idx_key",
            "candidate_label",
            "run_seed",
            "eval_score_mspd",
        ]
    ].rename(
        columns={
            "optimized_run_idx": "run_idx",
            "eval_score_mspd": "mspd",
        }
    )

    c5 = pd.read_csv(c5_root / "candidate_summary.csv")
    horizons = np.sort(c5["horizon_steps"].unique())
    expected_horizons = np.asarray([5000, 10000, 15000, 20000, 30000])
    if not np.array_equal(horizons, expected_horizons):
        raise ValueError(f"Unexpected C5 horizons: {horizons.tolist()}")

    keys = ["run_idx", "candidate_kind", "candidate_idx"]

    def integrate(metric: str, output_name: str) -> pd.DataFrame:
        values = c5.pivot(index=keys, columns="horizon_steps", values=metric)[
            horizons
        ]
        result = values.mean(axis=1).rename(output_name).reset_index()
        return result.rename(columns={"candidate_idx": "candidate_idx_key"})

    metrics = [
        integrate("excess_clip_post_release", "frustration_mean"),
        integrate(
            "paired_same_seed_clip_post_release", "wall_free_divergence_mean"
        ),
        integrate("free_within_clip_post_release", "natural_divergence_mean"),
    ]
    data = c1
    merge_keys = ["run_idx", "candidate_kind", "candidate_idx_key"]
    for metric in metrics:
        data = data.merge(metric, on=merge_keys, how="inner", validate="one_to_one")

    data = data.sort_values(merge_keys).reset_index(drop=True)
    if len(data) != 40:
        raise ValueError(f"Expected 40 candidates, found {len(data)}")
    counts = data.groupby("run_idx")["candidate_kind"].agg(
        optimized=lambda values: int((values == "optimized").sum()),
        random=lambda values: int((values == "random").sum()),
    )
    if not ((counts["optimized"] == 1) & (counts["random"] == 3)).all():
        raise ValueError("Each run must contain one optimized and three random candidates")

    data["candidate_code"] = np.where(
        data["candidate_kind"].eq("optimized"),
        "optimized",
        "random_" + data["candidate_idx_key"].astype(str),
    )
    return data


def plot(data: pd.DataFrame, c5_root: Path) -> tuple[Path, Path, Path]:
    figures_dir = c5_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "c5_mspd_vs_divergence_all_candidates.png"
    pdf_path = figures_dir / "c5_mspd_vs_divergence_all_candidates.pdf"
    csv_path = c5_root / "mspd_vs_divergence_all_candidates.csv"

    export_columns = [
        "run_idx",
        "candidate_kind",
        "candidate_idx_key",
        "candidate_label",
        "run_seed",
        "mspd",
        "frustration_mean",
        "wall_free_divergence_mean",
        "natural_divergence_mean",
    ]
    data[export_columns].to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), sharex=True)
    specs = [
        (
            "frustration_mean",
            "C5 excess frustration",
            "Walls divergence minus natural divergence",
        ),
        (
            "wall_free_divergence_mean",
            "Frustrated divergence",
            "Matched free-to-walls divergence",
        ),
        (
            "natural_divergence_mean",
            "Natural divergence",
            "Free-to-free divergence",
        ),
    ]
    markers = {"optimized": "*", "random_0": "o", "random_1": "^", "random_2": "s"}
    sizes = {"optimized": 150, "random_0": 58, "random_1": 62, "random_2": 56}
    colors = plt.get_cmap("tab10")
    x = data["mspd"].to_numpy(dtype=float) * 1e3

    for axis, (column, title, subtitle) in zip(axes, specs):
        y = data[column].to_numpy(dtype=float) * 1e3
        for row_idx, row in data.iterrows():
            code = str(row["candidate_code"])
            axis.scatter(
                x[row_idx],
                y[row_idx],
                marker=markers[code],
                s=sizes[code],
                color=colors(int(row["run_idx"])),
                edgecolor="white",
                linewidth=0.65,
                alpha=0.94,
                zorder=3,
            )

        fit_x = np.linspace(float(x.min()), float(x.max()), 200)
        slope, intercept = np.polyfit(x, y, 1)
        axis.plot(
            fit_x,
            slope * fit_x + intercept,
            color="#30343b",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )
        spearman = stats.spearmanr(x, y, alternative="greater")
        kendall = stats.kendalltau(x, y, alternative="greater")
        pearson = stats.pearsonr(x, y, alternative="greater")
        axis.text(
            0.035,
            0.965,
            f"Spearman rho = {spearman.statistic:+.3f}, p+ = {spearman.pvalue:.4f}\n"
            f"Kendall tau-b = {kendall.statistic:+.3f}, p+ = {kendall.pvalue:.4f}\n"
            f"Pearson r = {pearson.statistic:+.3f}, p+ = {pearson.pvalue:.4f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#c8cdd3",
                "alpha": 0.9,
            },
        )
        axis.axhline(0.0, color="#777d85", linewidth=0.9, zorder=1)
        axis.grid(True, color="#d9dde2", linewidth=0.7, alpha=0.75)
        axis.set_title(f"{title}\n{subtitle}")
        axis.set_xlabel(r"C1 MSPD on source seed ($\times 10^{-3}$)")
        axis.set_ylabel(r"Mean CLIP Chamfer distance ($\times 10^{-3}$)")

    run_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors(run_idx),
            markeredgecolor="white",
            markersize=8,
            label=f"run {run_idx:03d}",
        )
        for run_idx in range(10)
    ]
    candidate_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[code],
            color="none",
            markerfacecolor="#666b73",
            markeredgecolor="white",
            markersize=10 if code == "optimized" else 8,
            label=code,
        )
        for code in ("optimized", "random_0", "random_1", "random_2")
    ]
    fig.legend(
        handles=run_handles + candidate_handles,
        loc="lower center",
        ncol=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "C1 MSPD versus C5 divergence across all seed_000 candidates (n = 40)",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, csv_path


def main() -> None:
    args = parse_args()
    data = load_data(args.c1_scores.resolve(), args.c5_root.resolve())
    for output in plot(data, args.c5_root.resolve()):
        print(output)


if __name__ == "__main__":
    main()
