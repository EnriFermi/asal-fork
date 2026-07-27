from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP_ROOT = REPO_ROOT / (
    "analysis/results/"
    "flowlenia_rng_sensitivity_clip_chamfer_trajectory20_shared4_9branch_10k_v1"
)
VISUAL_ROOT = REPO_ROOT / (
    "analysis/results/"
    "flowlenia_rng_sensitivity_trajectory20_shared4_9branch_10k_v1"
)
OUTPUT_DIR = REPO_ROOT / "analysis/article_revision_20260722/figures"
OUTPUT_STEM = "flow_rng_sensitivity_exploratory"
REPRESENTATIVE_VISUAL_CACHE = REPO_ROOT / (
    "analysis/article_revision_20260722/cache/"
    "flow_rng_panel_a_opt005_seed000_step100k.npz"
)
HORIZON = 10_000
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260724
REPRESENTATIVE_CANDIDATE_ID = "run_005_optimized"
REPRESENTATIVE_ROLLOUT_SEED_IDX = 0

RANDOM_COLOR = "#9AA0A6"
OPTIMIZED_COLOR = "#1F6EAC"
INK = "#202124"
GRID = "#E2E5E9"


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(reps), values.size), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _candidate_table() -> pd.DataFrame:
    path = CLIP_ROOT / "clip_candidate_summary.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["arm"].astype(str) == "trajectory"].copy()
    if len(frame) != 40:
        raise RuntimeError(f"Expected 40 trajectory-arm candidates, found {len(frame)}.")
    counts = frame.groupby("candidate_kind").size().to_dict()
    if counts != {"optimized": 10, "random": 30}:
        raise RuntimeError(f"Unexpected candidate counts: {counts}")
    return frame


def _representative_example(
    candidates: pd.DataFrame,
) -> tuple[
    str,
    int,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    list[int],
    dict[int, int],
]:
    optimized = candidates.loc[candidates["candidate_kind"] == "optimized"]
    ranked = optimized.sort_values(
        ["clip_chamfer_10k", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    selected_rows = ranked.loc[
        ranked["candidate_id"].astype(str) == REPRESENTATIVE_CANDIDATE_ID
    ]
    if len(selected_rows) != 1:
        raise RuntimeError(
            "Missing representative candidate: "
            f"{REPRESENTATIVE_CANDIDATE_ID}"
        )
    selected_rank = int(selected_rows.index[0]) + 1
    selected = selected_rows.iloc[0]
    candidate_id = str(selected["candidate_id"])

    visual_path = REPRESENTATIVE_VISUAL_CACHE
    with np.load(visual_path, allow_pickle=False) as data:
        if str(np.asarray(data["candidate_id"]).item()) != candidate_id:
            raise RuntimeError(
                f"Representative cache candidate mismatch: {visual_path}"
            )
        visual_context_idx = int(np.asarray(data["visual_context_idx"]).item())
        context_indices = np.asarray(data["context_indices"], dtype=np.int32)
        context_position = int(np.flatnonzero(context_indices == visual_context_idx)[0])
        visual_steps = np.asarray(data["visual_steps"], dtype=np.int32)
        visual_rgb = np.asarray(data["visual_rgb"], dtype=np.uint8)

        if not np.all(visual_rgb[:, 0] == visual_rgb[0, 0]):
            raise RuntimeError("Representative branches do not share a bit-exact t=0 frame.")

        metric_steps = np.asarray(data["metric_steps"], dtype=np.int32)
        horizon_position = int(np.flatnonzero(metric_steps == HORIZON)[0])
        pair_left = np.asarray(data["pair_left"], dtype=np.int32)
        pair_right = np.asarray(data["pair_right"], dtype=np.int32)
        pair_values = np.asarray(
            data["render_l1"][context_position, horizon_position], dtype=np.float64
        )

    branches = pd.read_csv(CLIP_ROOT / "branches.csv")
    unique = branches.loc[branches["included_in_pairwise_metric"].astype(bool)].copy()
    branch_indices = sorted(unique["branch_idx"].astype(int).tolist())
    branch_seeds = {
        int(row.branch_idx): int(row.branch_seed) for row in unique.itertuples(index=False)
    }

    distance = np.zeros((max(branch_indices) + 1, max(branch_indices) + 1))
    for left, right, value in zip(pair_left, pair_right, pair_values):
        distance[left, right] = value
        distance[right, left] = value

    left, right = np.unravel_index(np.argmax(distance), distance.shape)
    selected_branches = [int(left), int(right)]
    while len(selected_branches) < 4:
        remaining = [idx for idx in branch_indices if idx not in selected_branches]
        next_branch = max(
            remaining,
            key=lambda idx: (
                min(distance[idx, chosen] for chosen in selected_branches),
                -idx,
            ),
        )
        selected_branches.append(int(next_branch))

    contexts = pd.read_csv(CLIP_ROOT / "contexts.csv")
    context = contexts.loc[
        (contexts["candidate_id"].astype(str) == candidate_id)
        & (contexts["context_idx"].astype(int) == visual_context_idx)
    ]
    if len(context) != 1:
        raise RuntimeError(
            f"Expected one context row for {candidate_id}/{visual_context_idx}."
        )
    rollout_seed_idx = int(context.iloc[0]["rollout_seed_idx"])
    if rollout_seed_idx != REPRESENTATIVE_ROLLOUT_SEED_IDX:
        raise RuntimeError(
            f"Expected visual context for rollout seed "
            f"{REPRESENTATIVE_ROLLOUT_SEED_IDX:03d}, found "
            f"{rollout_seed_idx:03d}."
        )
    source_step = int(context.iloc[0]["source_step"])
    if str(context.iloc[0]["arm"]) != "trajectory":
        raise RuntimeError("Representative example is not a visited C1 trajectory state.")

    return (
        candidate_id,
        selected_rank,
        rollout_seed_idx,
        source_step,
        visual_context_idx,
        visual_steps,
        visual_rgb,
        selected_branches,
        branch_seeds,
    )


def _plot_montage(
    fig: plt.Figure,
    spec,
    *,
    candidate_id: str,
    rollout_seed_idx: int,
    source_step: int,
    visual_steps: np.ndarray,
    visual_rgb: np.ndarray,
    selected_branches: list[int],
    branch_seeds: dict[int, int],
) -> None:
    frame_steps = [0, 2_500, 5_000, 10_000]
    frame_indices = [int(np.flatnonzero(visual_steps == step)[0]) for step in frame_steps]
    grid = spec.subgridspec(
        len(selected_branches),
        len(frame_steps),
        wspace=0.035,
        hspace=0.035,
    )

    axes: list[list[plt.Axes]] = []
    for row_idx, branch_idx in enumerate(selected_branches):
        row_axes = []
        for col_idx, (step, frame_idx) in enumerate(
            zip(frame_steps, frame_indices)
        ):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.imshow(visual_rgb[branch_idx, frame_idx], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                label = "0 (bit-exact start)" if step == 0 else f"{step / 1000:g}k"
                ax.set_title(label, fontsize=10, pad=5)
            if col_idx == 0:
                ax.set_ylabel(
                    f"seed {branch_seeds[branch_idx]}",
                    fontsize=8.5,
                    rotation=90,
                    labelpad=6,
                )
            row_axes.append(ax)
        axes.append(row_axes)

    bounds = spec.get_position(fig)
    fig.text(
        bounds.x0,
        bounds.y1 + 0.075,
        "A",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        bounds.x0 + 0.027,
        bounds.y1 + 0.075,
        "Exact-state continuation fork",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        bounds.x0 + 0.027,
        bounds.y1 + 0.044,
        (
            f"opt_{candidate_id.split('_')[1]}_seed_{rollout_seed_idx:03d}, "
            f"saved exact C1 state at step "
            f"{source_step:,}; same rule and state, continuation RNG only"
        ),
        fontsize=9.5,
        color="#4A4F55",
        ha="left",
        va="bottom",
    )
    fig.text(
        bounds.x0,
        bounds.y0 - 0.055,
        (
            "Four illustrative unique branches selected for visual separation at 10k; "
            "no external state perturbation."
        ),
        fontsize=8.5,
        color="#5F6368",
        ha="left",
        va="top",
    )


def _plot_candidate_panel(
    fig: plt.Figure,
    spec,
    candidates: pd.DataFrame,
    stats: dict,
) -> dict[str, list[float] | float]:
    ax = fig.add_subplot(spec)
    random_values = candidates.loc[
        candidates["candidate_kind"] == "random", "clip_chamfer_10k"
    ].to_numpy(dtype=np.float64)
    optimized_values = candidates.loc[
        candidates["candidate_kind"] == "optimized", "clip_chamfer_10k"
    ].to_numpy(dtype=np.float64)

    random_ci = _bootstrap_mean_ci(random_values, seed=BOOTSTRAP_SEED)
    optimized_ci = _bootstrap_mean_ci(optimized_values, seed=BOOTSTRAP_SEED + 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    random_x = rng.uniform(-0.16, 0.16, size=random_values.size)
    optimized_x = 1.0 + rng.uniform(-0.16, 0.16, size=optimized_values.size)

    ax.scatter(
        random_x,
        random_values,
        s=48,
        color=RANDOM_COLOR,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        optimized_x,
        optimized_values,
        s=52,
        color=OPTIMIZED_COLOR,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.95,
        zorder=3,
    )

    group_values = [random_values, optimized_values]
    group_cis = [random_ci, optimized_ci]
    for x, values, ci in zip((0.0, 1.0), group_values, group_cis):
        mean = float(np.mean(values))
        ax.errorbar(
            [x],
            [mean],
            yerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="D",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.5,
            ecolor=INK,
            elinewidth=2.0,
            capsize=6,
            capthick=2.0,
            zorder=5,
        )

    primary = stats["primary_test"]
    annotation = (
        "Exploratory comparison\n"
        f"mean difference = {primary['mean_difference']:+.2e}\n"
        "95% bootstrap CI for difference\n"
        f"[{primary['mean_difference_ci95_low']:+.2e}, "
        f"{primary['mean_difference_ci95_high']:+.2e}]\n"
        f"exact one-sided Mann-Whitney p = "
        f"{primary['mann_whitney_exact_one_sided_p']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.35,
        bbox={
            "boxstyle": "square,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#C9CDD2",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=6,
    )

    ax.set_xlim(-0.43, 1.43)
    y_min = min(float(random_values.min()), float(optimized_values.min()))
    y_max = max(float(random_values.max()), float(optimized_values.max()))
    padding = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - padding, y_max + 0.18 * (y_max - y_min))
    ax.set_xticks([0, 1], ["Random\n(n=30)", "Optimized\n(n=10)"])
    ax.set_ylabel("candidate CLIP-Chamfer at 10k", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)

    mean_handle = Line2D(
        [0],
        [0],
        marker="D",
        color=INK,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=1.8,
        markersize=7,
        label="mean and 95% bootstrap CI",
    )
    ax.legend(
        handles=[mean_handle],
        loc="lower center",
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )

    bounds = spec.get_position(fig)
    fig.text(
        bounds.x0,
        bounds.y1 + 0.075,
        "B",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y1 + 0.075,
        "Candidate-level RNG sensitivity at 10k",
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    fig.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y1 + 0.035,
        (
            "Exploratory motivation for fixed-context evaluation;\n"
            "not a confirmed optimized-vs-random claim"
        ),
        fontsize=8.7,
        color="#4A4F55",
        ha="center",
        va="bottom",
        linespacing=1.15,
    )

    return {
        "random_mean": float(np.mean(random_values)),
        "random_mean_bootstrap_ci95": [float(random_ci[0]), float(random_ci[1])],
        "optimized_mean": float(np.mean(optimized_values)),
        "optimized_mean_bootstrap_ci95": [
            float(optimized_ci[0]),
            float(optimized_ci[1]),
        ],
    }


def build(output_dir: Path) -> dict:
    candidates = _candidate_table()
    with (CLIP_ROOT / "clip_statistical_summary.json").open() as handle:
        stats = json.load(handle)

    (
        candidate_id,
        selected_rank,
        rollout_seed_idx,
        source_step,
        visual_context_idx,
        visual_steps,
        visual_rgb,
        selected_branches,
        branch_seeds,
    ) = _representative_example(candidates)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(17.2, 7.6), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.62, 1.0],
        left=0.045,
        right=0.985,
        bottom=0.14,
        top=0.83,
        wspace=0.16,
    )
    _plot_montage(
        fig,
        outer[0],
        candidate_id=candidate_id,
        rollout_seed_idx=rollout_seed_idx,
        source_step=source_step,
        visual_steps=visual_steps,
        visual_rgb=visual_rgb,
        selected_branches=selected_branches,
        branch_seeds=branch_seeds,
    )
    group_stats = _plot_candidate_panel(fig, outer[1], candidates, stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=240, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    provenance = {
        "figure": OUTPUT_STEM,
        "interpretation": (
            "Exploratory motivation for fixed-context evaluation; not a confirmed "
            "optimized-vs-random claim."
        ),
        "panel_a": {
            "candidate_id": candidate_id,
            "source_trajectory_id": (
                f"opt_{candidate_id.split('_')[1]}_seed_"
                f"{rollout_seed_idx:03d}"
            ),
            "selection": (
                "User-selected exact C1 trajectory opt_005_seed_000. Four unique "
                "branches are selected by farthest-point traversal of final render-L1 "
                "separation for the cached visual context."
            ),
            "optimized_clip_chamfer_rank_at_10k": selected_rank,
            "candidate_clip_chamfer_10k": float(
                candidates.loc[
                    candidates["candidate_id"].astype(str) == candidate_id,
                    "clip_chamfer_10k",
                ].iloc[0]
            ),
            "rollout_seed_idx": rollout_seed_idx,
            "visual_context_idx": visual_context_idx,
            "source_step": source_step,
            "branch_indices": selected_branches,
            "branch_seeds": [branch_seeds[idx] for idx in selected_branches],
            "frame_steps": [0, 2_500, 5_000, 10_000],
            "common_start_bit_exact": True,
            "external_state_perturbation": 0.0,
        },
        "panel_b": {
            "endpoint": "candidate mean trajectory-arm CLIP-Chamfer at 10k",
            "n_optimized": 10,
            "n_random": 30,
            "group_stats": group_stats,
            "difference_stats": stats["primary_test"],
            "bootstrap_reps_for_group_mean_intervals": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "sources": {
            "clip_root": str(CLIP_ROOT.relative_to(REPO_ROOT)),
            "visual_root": str(VISUAL_ROOT.relative_to(REPO_ROOT)),
            "panel_a_visual_cache": str(
                REPRESENTATIVE_VISUAL_CACHE.relative_to(REPO_ROOT)
            ),
        },
        "outputs": {
            "png": str(png_path.relative_to(REPO_ROOT)),
            "pdf": str(pdf_path.relative_to(REPO_ROOT)),
        },
    }
    provenance_path = output_dir / f"{OUTPUT_STEM}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the combined exploratory Flow-Lenia RNG-sensitivity figure."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    provenance = build(args.output_dir.resolve())
    print(json.dumps(provenance["outputs"], indent=2))


if __name__ == "__main__":
    main()
