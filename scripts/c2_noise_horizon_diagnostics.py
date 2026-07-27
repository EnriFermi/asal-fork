from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from c2_noise_horizon_sweep import _load_branch_arrays, _render_rgb
from paper_suite_common import ensure_dir, write_csv, write_json


DEFAULT_SUITE_ROOT = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper"
)
DEFAULT_SWEEP_ROOT = f"{DEFAULT_SUITE_ROOT}/c2_noise_horizon_sweep"
EVOLUTION_OFFSETS = (0, 4250, 10000, 20000, 30000)
CONDITION_COLORS = {
    "low": "#2878b5",
    "mid": "#ed8b2f",
    "high": "#2b9b46",
}


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def _fdr_bh(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    adjusted = np.full(raw.shape, np.nan, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(raw))
    if finite_idx.size == 0:
        return adjusted
    order = finite_idx[np.argsort(raw[finite_idx], kind="stable")]
    ranked = raw[order]
    scale = float(order.size) / np.arange(1, order.size + 1, dtype=np.float64)
    corrected = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted[order] = np.clip(corrected, 0.0, 1.0)
    return adjusted


def _pearson_fast(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[finite]
    yy = yy[finite]
    if xx.size < 3:
        return float("nan")
    xx = xx - np.mean(xx)
    yy = yy - np.mean(yy)
    denominator = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.sum(xx * yy) / denominator)


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_fast(rankdata(x), rankdata(y))


def _cluster_bootstrap(
    scores: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    run_ids = np.asarray(sorted(scores["run_idx"].unique()), dtype=np.int64)
    if not np.array_equal(run_ids, np.arange(10, dtype=np.int64)):
        raise ValueError(f"Expected run indices 0..9, got {run_ids.tolist()}.")
    score_groups = {
        int(run_idx): group
        for run_idx, group in scores.groupby("run_idx", sort=True)
    }
    contrast_groups = {
        int(run_idx): group
        for run_idx, group in contrasts.groupby("run_idx", sort=True)
    }
    rng = np.random.default_rng(int(seed))
    pearson_samples = np.empty(int(reps), dtype=np.float64)
    spearman_samples = np.empty(int(reps), dtype=np.float64)
    contrast_samples = np.empty(int(reps), dtype=np.float64)
    for rep_idx in range(int(reps)):
        sampled = rng.choice(run_ids, size=run_ids.size, replace=True)
        score_parts = [score_groups[int(run_idx)] for run_idx in sampled]
        contrast_parts = [contrast_groups[int(run_idx)] for run_idx in sampled]
        x = np.concatenate(
            [part["delta_h"].to_numpy(dtype=np.float64) for part in score_parts]
        )
        y = np.concatenate(
            [
                part["branching_score"].to_numpy(dtype=np.float64)
                for part in score_parts
            ]
        )
        delta = np.concatenate(
            [
                part["delta_branching_score"].to_numpy(dtype=np.float64)
                for part in contrast_parts
            ]
        )
        pearson_samples[rep_idx] = _pearson_fast(x, y)
        spearman_samples[rep_idx] = _spearman_fast(x, y)
        contrast_samples[rep_idx] = float(np.median(delta))

    def interval(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size != values.size:
            raise ValueError("Cluster bootstrap produced non-finite statistics.")
        low, high = np.quantile(finite, [0.025, 0.975])
        return float(low), float(high)

    pearson_low, pearson_high = interval(pearson_samples)
    spearman_low, spearman_high = interval(spearman_samples)
    contrast_low, contrast_high = interval(contrast_samples)
    return {
        "bootstrap_reps": int(reps),
        "bootstrap_seed": int(seed),
        "pearson_cluster_ci_low": pearson_low,
        "pearson_cluster_ci_high": pearson_high,
        "spearman_cluster_ci_low": spearman_low,
        "spearman_cluster_ci_high": spearman_high,
        "contrast_median_cluster_ci_low": contrast_low,
        "contrast_median_cluster_ci_high": contrast_high,
    }


def _compute_inference(
    *,
    scores: pd.DataFrame,
    correlations: pd.DataFrame,
    contrasts: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped_scores = scores.groupby(["strength", "horizon_steps"], sort=True)
    grouped_contrasts = contrasts.groupby(
        ["strength", "horizon_steps"],
        sort=True,
    )
    for cell_idx, source in correlations.sort_values(
        ["strength", "horizon_steps"]
    ).reset_index(drop=True).iterrows():
        key = (float(source["strength"]), int(source["horizon_steps"]))
        score_group = grouped_scores.get_group(key)
        contrast_group = grouped_contrasts.get_group(key)
        n_nonzero = int(source["contrast_n_nonzero"])
        n_positive = int(source["contrast_n_positive"])
        cell_seed = int(seed) + 1009 * int(cell_idx)
        rows.append(
            {
                **source.to_dict(),
                "contrast_sign_test_less_p": float(
                    binomtest(
                        n_positive,
                        n_nonzero,
                        p=0.5,
                        alternative="less",
                    ).pvalue
                ),
                "contrast_sign_test_two_sided_p": float(
                    binomtest(
                        n_positive,
                        n_nonzero,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                ),
                **_cluster_bootstrap(
                    score_group,
                    contrast_group,
                    reps=int(reps),
                    seed=cell_seed,
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["pearson_fdr_bh_q"] = _fdr_bh(
        result["pearson_p"].to_numpy(dtype=np.float64)
    )
    result["spearman_fdr_bh_q"] = _fdr_bh(
        result["spearman_p"].to_numpy(dtype=np.float64)
    )
    result["contrast_greater_fdr_bh_q"] = _fdr_bh(
        result["contrast_sign_test_greater_p"].to_numpy(dtype=np.float64)
    )
    result["contrast_less_fdr_bh_q"] = _fdr_bh(
        result["contrast_sign_test_less_p"].to_numpy(dtype=np.float64)
    )
    return result


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_per_run_heatmaps(
    *,
    within: pd.DataFrame,
    figures: Path,
    value: str,
    title: str,
) -> Path:
    plt = _matplotlib()
    strengths = sorted(within["strength"].unique().tolist())
    horizons = sorted(within["horizon_steps"].unique().astype(int).tolist())
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(12.5, 20.0),
        constrained_layout=True,
    )
    image = None
    for run_idx, ax in enumerate(axes.flat):
        run = within[within["run_idx"] == run_idx].set_index(
            ["strength", "horizon_steps"]
        )
        values = np.asarray(
            [
                [
                    float(run.loc[(strength, horizon), value])
                    for horizon in horizons
                ]
                for strength in strengths
            ],
            dtype=np.float64,
        )
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                current = float(values[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{current:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(current) >= 0.45 else "black",
                )
        ax.set_xticks(
            np.arange(len(horizons)),
            [f"{horizon // 1000}k" for horizon in horizons],
        )
        ax.set_yticks(
            np.arange(len(strengths)),
            [f"{strength:g}" for strength in strengths],
        )
        ax.set_xlabel("branch horizon")
        ax.set_ylabel("noise scale")
        ax.set_title(f"opt_{run_idx:03d}")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.52, pad=0.02, label=title)
    fig.suptitle(f"Flow-Lenia C2 per-run {title}", fontsize=16)
    output = figures / f"c2_per_run_{value}_heatmaps.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_per_run_curves(
    *,
    within: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt = _matplotlib()
    strengths = sorted(within["strength"].unique().tolist())
    colors = plt.cm.viridis(np.linspace(0.04, 0.96, len(strengths)))
    rows: list[dict[str, Any]] = []
    for run_idx in range(10):
        run = within[within["run_idx"] == run_idx]
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.3),
            constrained_layout=True,
        )
        for strength, color in zip(strengths, colors, strict=True):
            group = run[np.isclose(run["strength"], strength)].sort_values(
                "horizon_steps"
            )
            x = group["horizon_steps"].to_numpy(dtype=np.float64) / 1000.0
            axes[0].plot(
                x,
                group["pearson_r"],
                marker="o",
                color=color,
                label=f"{strength:g}",
            )
            axes[1].plot(
                x,
                group["spearman_rho"],
                marker="o",
                color=color,
            )
        for ax in axes:
            ax.axhline(0.0, color="#555555", linewidth=0.9)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlabel("branch horizon (k steps)")
            ax.grid(color="#dddddd", linewidth=0.65)
        axes[0].set_ylabel("Pearson r")
        axes[1].set_ylabel("Spearman rho")
        axes[0].legend(
            title="noise scale",
            frameon=False,
            ncol=2,
            fontsize=8,
        )
        fig.suptitle(f"Flow-Lenia C2 sensitivity: opt_{run_idx:03d}")
        output = output_dir / f"opt_{run_idx:03d}_noise_horizon_curves.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        rows.append({"run_idx": run_idx, "figure": str(output)})
    manifest = output_dir / "manifest.csv"
    write_csv(manifest, rows)

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(13.0, 18.0),
        constrained_layout=True,
    )
    for run_idx, ax in enumerate(axes.flat):
        run = within[within["run_idx"] == run_idx]
        for strength, color in zip(strengths, colors, strict=True):
            group = run[np.isclose(run["strength"], strength)].sort_values(
                "horizon_steps"
            )
            ax.plot(
                group["horizon_steps"].to_numpy(dtype=np.float64) / 1000.0,
                group["spearman_rho"],
                marker="o",
                markersize=3.5,
                color=color,
                linewidth=1.2,
                label=f"{strength:g}",
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"opt_{run_idx:03d}")
        ax.set_xlabel("horizon (k steps)")
        ax.set_ylabel("Spearman rho")
        ax.grid(color="#dddddd", linewidth=0.55)
    axes.flat[0].legend(
        title="noise",
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle("C2 within-run association curves across noise and horizon")
    combined = output_dir.parent / "c2_per_run_spearman_curves.png"
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return manifest, combined


def _plot_uncertainty(
    *,
    inference: pd.DataFrame,
    figures: Path,
) -> Path:
    plt = _matplotlib()
    strengths = sorted(inference["strength"].unique().tolist())
    colors = plt.cm.viridis(np.linspace(0.04, 0.96, len(strengths)))
    panels = (
        (
            "pearson_r",
            "pearson_cluster_ci_low",
            "pearson_cluster_ci_high",
            "pooled Pearson r",
        ),
        (
            "spearman_rho",
            "spearman_cluster_ci_low",
            "spearman_cluster_ci_high",
            "pooled Spearman rho",
        ),
        (
            "contrast_median",
            "contrast_median_cluster_ci_low",
            "contrast_median_cluster_ci_high",
            "median high-minus-low divergence",
        ),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.6, 4.8),
        constrained_layout=True,
    )
    for strength, color in zip(strengths, colors, strict=True):
        group = inference[np.isclose(inference["strength"], strength)].sort_values(
            "horizon_steps"
        )
        x = group["horizon_steps"].to_numpy(dtype=np.float64) / 1000.0
        for ax, (value, low, high, _) in zip(axes, panels, strict=True):
            y = group[value].to_numpy(dtype=np.float64)
            ax.plot(
                x,
                y,
                marker="o",
                color=color,
                linewidth=1.4,
                label=f"{strength:g}",
            )
            ax.fill_between(
                x,
                group[low].to_numpy(dtype=np.float64),
                group[high].to_numpy(dtype=np.float64),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    for ax, (_, _, _, ylabel) in zip(axes, panels, strict=True):
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.set_xlabel("branch horizon (k steps)")
        ax.set_ylabel(ylabel)
        ax.grid(color="#dddddd", linewidth=0.65)
    axes[0].legend(
        title="noise scale",
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    fig.suptitle("C2 sensitivity with 95% run-cluster bootstrap intervals")
    output = figures / "c2_noise_horizon_cluster_bootstrap_ci.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _state_rows(
    plan: pd.DataFrame,
    *,
    run_idx: int,
    condition: str,
    pair_id: int,
) -> pd.DataFrame:
    selected = plan[
        (plan["run_idx"].astype(int) == int(run_idx))
        & (plan["condition"].astype(str) == str(condition))
        & (plan["pair_id"].astype(int) == int(pair_id))
    ].sort_values(["strength", "branch_id"])
    expected = 7 * 3
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} rows for opt_{run_idx:03d}/{condition}/"
            f"pair={pair_id}, found {len(selected)}."
        )
    return selected


def _load_evolution_frames(
    selected: pd.DataFrame,
) -> tuple[
    dict[tuple[float, int, int], np.ndarray],
    pd.DataFrame,
]:
    frames: dict[tuple[float, int, int], np.ndarray] = {}
    rms_rows: list[dict[str, Any]] = []
    for strength, group in selected.groupby("strength", sort=True):
        branch_frames: dict[int, dict[int, np.ndarray]] = {}
        for row in group.sort_values("branch_id").itertuples(index=False):
            arrays = _load_branch_arrays(Path(str(row.branch_dir)))
            steps = np.asarray(arrays["steps"], dtype=np.int64)
            relative = steps - int(row.step)
            rgb = _render_rgb(arrays["A"], arrays["P"])
            branch_id = int(row.branch_id)
            branch_frames[branch_id] = {}
            for offset in EVOLUTION_OFFSETS:
                hit = np.flatnonzero(relative == int(offset))
                if hit.size != 1:
                    raise ValueError(
                        f"Missing offset {offset} in {row.branch_dir}."
                    )
                frame = np.asarray(rgb[int(hit[0])], dtype=np.float32)
                frames[(float(strength), branch_id, int(offset))] = frame
                branch_frames[branch_id][int(offset)] = frame
        for offset in EVOLUTION_OFFSETS:
            pair_rms = [
                float(
                    np.sqrt(
                        np.mean(
                            (
                                branch_frames[left][int(offset)]
                                - branch_frames[right][int(offset)]
                            )
                            ** 2
                        )
                    )
                )
                for left, right in combinations(sorted(branch_frames), 2)
            ]
            rms_rows.append(
                {
                    "strength": float(strength),
                    "relative_step": int(offset),
                    "rgb_pair_rms_median": float(np.median(pair_rms)),
                    "rgb_pair_rms_mean": float(np.mean(pair_rms)),
                    "rgb_pair_rms_std": float(np.std(pair_rms, ddof=1)),
                }
            )
    return frames, pd.DataFrame(rms_rows)


def _plot_evolution_montage(
    *,
    frames: dict[tuple[float, int, int], np.ndarray],
    strengths: list[float],
    condition: str,
    run_idx: int,
    pair_id: int,
    figures: Path,
) -> Path:
    plt = _matplotlib()
    columns = [
        (branch_id, offset)
        for branch_id in range(3)
        for offset in EVOLUTION_OFFSETS
    ]
    fig, axes = plt.subplots(
        len(strengths),
        len(columns),
        figsize=(30.0, 14.5),
        constrained_layout=True,
    )
    for row_idx, strength in enumerate(strengths):
        for col_idx, (branch_id, offset) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                frames[(float(strength), int(branch_id), int(offset))],
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                label = "0" if offset == 0 else f"{offset / 1000:g}k"
                ax.set_title(f"b{branch_id} +{label}", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"noise={strength:g}",
                    fontsize=9,
                    color=CONDITION_COLORS[condition],
                )
    fig.suptitle(
        f"C2 branch evolution: opt_{run_idx:03d}, {condition}, pair {pair_id}",
        fontsize=15,
    )
    output = (
        figures
        / f"c2_opt_{run_idx:03d}_{condition}_pair{pair_id}_noise_evolution.png"
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def _write_evolution_video(
    *,
    selected: pd.DataFrame,
    condition: str,
    run_idx: int,
    pair_id: int,
    output: Path,
    fps: float,
) -> None:
    import imageio.v2 as imageio

    plt = _matplotlib()
    strengths = sorted(selected["strength"].unique().tolist())
    all_frames: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    relative_steps: np.ndarray | None = None
    for row in selected.itertuples(index=False):
        arrays = _load_branch_arrays(Path(str(row.branch_dir)))
        steps = np.asarray(arrays["steps"], dtype=np.int64)
        relative = steps - int(row.step)
        if relative_steps is None:
            relative_steps = relative
        elif not np.array_equal(relative_steps, relative):
            raise ValueError("Evolution video branch step grids differ.")
        all_frames[(float(row.strength), int(row.branch_id))] = (
            relative,
            _render_rgb(arrays["A"], arrays["P"]),
        )
    assert relative_steps is not None

    with imageio.get_writer(
        output,
        fps=float(fps),
        codec="libx264",
        macro_block_size=2,
        quality=7,
    ) as writer:
        for frame_idx, relative_step in enumerate(relative_steps):
            fig, axes = plt.subplots(
                len(strengths),
                3,
                figsize=(6.6, 13.0),
                constrained_layout=True,
            )
            for row_idx, strength in enumerate(strengths):
                for branch_id in range(3):
                    ax = axes[row_idx, branch_id]
                    ax.imshow(
                        all_frames[(float(strength), branch_id)][1][frame_idx],
                        interpolation="nearest",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row_idx == 0:
                        ax.set_title(f"branch {branch_id}", fontsize=9)
                    if branch_id == 0:
                        ax.set_ylabel(f"noise={strength:g}", fontsize=8)
            fig.suptitle(
                (
                    f"opt_{run_idx:03d} {condition} pair {pair_id}; "
                    f"relative step +{int(relative_step)}"
                ),
                fontsize=11,
            )
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
                height,
                width,
                4,
            )
            writer.append_data(rgba[..., :3])
            plt.close(fig)


def _selection_linkage_audit(
    *,
    suite_root: Path,
    sweep_plan: pd.DataFrame,
    figures: Path,
) -> dict[str, Any]:
    source_plan_path = suite_root / "c2_branching" / "branch_plan.csv"
    source_plan = pd.read_csv(source_plan_path)
    keys = ["traj_id", "pair_id", "condition", "step"]
    sweep_states = (
        sweep_plan[keys]
        .drop_duplicates()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    source_states = (
        source_plan[keys]
        .drop_duplicates()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    exact = bool(sweep_states.equals(source_states))
    source_heatmap = suite_root / "figures" / "c2_per_run_delta_h_heatmaps.png"
    linked_heatmap = figures / "c2_selected_branch_points_delta_h_heatmaps.png"
    if not source_heatmap.exists():
        raise FileNotFoundError(source_heatmap)
    shutil.copy2(source_heatmap, linked_heatmap)
    payload = {
        "status": "exact" if exact else "mismatch",
        "keys": keys,
        "n_sweep_states": len(sweep_states),
        "n_source_states": len(source_states),
        "source_plan": str(source_plan_path),
        "source_heatmap": str(source_heatmap),
        "linked_heatmap": str(linked_heatmap),
    }
    if not exact:
        raise RuntimeError("Sweep states do not match original C2 selected states.")
    return payload


def _write_report(
    *,
    output: Path,
    inference: pd.DataFrame,
    linkage: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    selected = inference[inference["horizon_steps"] == 20000].sort_values(
        "strength"
    )
    lines = [
        "# Flow-Lenia C2 noise/horizon sensitivity report",
        "",
        "## Completion",
        "",
        "- 7 perturbation scales x 5 horizons x 150 selected states.",
        "- 3 branches per state; 3150 validated branch continuations.",
        "- All five horizons reuse exact prefixes from one 30k continuation.",
        "- Original noise=1/horizon=20k A/P and saved scores pass exact parity.",
        (
            "- Selection linkage: "
            f"{linkage['n_sweep_states']}/{linkage['n_source_states']} states "
            "match the original Delta-H branch plan exactly."
        ),
        "",
        "## Noise definition",
        "",
        "Each scale multiplies A_std=0.02, P_std=0.02, and "
        "lagrangian_xy_std=1.0. Scale 0 applies no state perturbation but keeps "
        "the protocol's independent branch RNG seeds.",
        "",
        "## Primary 20k results",
        "",
        "| scale | Pearson r [run-cluster 95% CI] | FDR q | "
        "within-run median rho | median high-low [95% CI] | positive pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            "| "
            f"{float(row.strength):g} | "
            f"{float(row.pearson_r):+.3f} "
            f"[{float(row.pearson_cluster_ci_low):+.3f}, "
            f"{float(row.pearson_cluster_ci_high):+.3f}] | "
            f"{float(row.pearson_fdr_bh_q):.3g} | "
            f"{float(row.within_spearman_median):+.3f} | "
            f"{float(row.contrast_median):+.5f} "
            f"[{float(row.contrast_median_cluster_ci_low):+.5f}, "
            f"{float(row.contrast_median_cluster_ci_high):+.5f}] | "
            f"{int(row.contrast_n_positive)}/"
            f"{int(row.contrast_n_nonzero)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The expected positive C2 association is present only in the "
            "zero-state-perturbation branch-RNG control.",
            "- At scale 0.1 the association is approximately null.",
            "- At the original scale 1 the association is negative across all "
            "five horizons and is also negative within most runs.",
            "- Strong perturbations increasingly measure perturbation damage; "
            "low-Delta-H states often diverge more than high-Delta-H states.",
            "- Therefore the positive C2 claim is not robust to perturbation "
            "amplitude under this protocol.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in sorted(outputs.items()):
        lines.append(f"- `{name}`: `{path}`")
    output.write_text("\n".join(lines) + "\n")


def _publish_outputs(
    *,
    suite_root: Path,
    full_root: Path,
    figures: Path,
    per_run_dir: Path,
    report_path: Path,
) -> Path:
    published = ensure_dir(suite_root / "figures" / "c2_noise_horizon")
    sources = [
        full_root / "correlation_grid_inference.csv",
        full_root / "visual_branch_rms_diagnostics.csv",
        full_root / "figures" / "c2_noise_horizon_heatmaps.png",
        full_root / "figures" / "c2_noise_horizon_correlation_curves.png",
        full_root / "figures" / "c2_noise_horizon_divergence_by_stratum.png",
        figures / "c2_noise_horizon_cluster_bootstrap_ci.png",
        figures / "c2_per_run_pearson_r_heatmaps.png",
        figures / "c2_per_run_spearman_rho_heatmaps.png",
        figures / "c2_per_run_spearman_curves.png",
        figures / "c2_selected_branch_points_delta_h_heatmaps.png",
        figures / "c2_opt_005_low_pair0_noise_evolution.png",
        figures / "c2_opt_005_high_pair0_noise_evolution.png",
        figures / "c2_opt_005_low_pair0_noise_grid.mp4",
        figures / "c2_opt_005_high_pair0_noise_grid.mp4",
        report_path,
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish incomplete diagnostics: " + ", ".join(missing)
        )
    for source in sources:
        shutil.copy2(source, published / source.name)
    shutil.copytree(
        per_run_dir,
        published / "per_run",
        dirs_exist_ok=True,
    )
    return published


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite_root = _resolve(args.suite_root)
    sweep_root = _resolve(args.sweep_root)
    full_root = sweep_root / "full"
    figures = ensure_dir(full_root / "figures" / "diagnostics")
    per_run_dir = ensure_dir(figures / "per_run")

    scores = pd.read_csv(full_root / "scores_clip_chamfer.csv")
    correlations = pd.read_csv(full_root / "correlation_grid.csv")
    within = pd.read_csv(full_root / "within_run_correlations.csv")
    contrasts = pd.read_csv(full_root / "matched_high_low_contrasts.csv")
    plan = pd.read_csv(full_root / "sweep_plan.csv")
    if (
        len(scores) != 5250
        or len(correlations) != 35
        or len(within) != 350
        or len(contrasts) != 1750
        or len(plan) != 3150
    ):
        raise ValueError("Sweep tables do not have the expected complete sizes.")

    inference = _compute_inference(
        scores=scores,
        correlations=correlations,
        contrasts=contrasts,
        reps=int(args.bootstrap_reps),
        seed=int(args.seed),
    )
    inference_path = full_root / "correlation_grid_inference.csv"
    inference.to_csv(inference_path, index=False)

    pearson_heatmaps = _plot_per_run_heatmaps(
        within=within,
        figures=figures,
        value="pearson_r",
        title="Pearson r",
    )
    spearman_heatmaps = _plot_per_run_heatmaps(
        within=within,
        figures=figures,
        value="spearman_rho",
        title="Spearman rho",
    )
    per_run_manifest, per_run_curves = _plot_per_run_curves(
        within=within,
        output_dir=per_run_dir,
    )
    uncertainty = _plot_uncertainty(
        inference=inference,
        figures=figures,
    )
    linkage = _selection_linkage_audit(
        suite_root=suite_root,
        sweep_plan=plan,
        figures=figures,
    )
    write_json(full_root / "selection_linkage_audit.json", linkage)

    visual_outputs: dict[str, str] = {}
    visual_rows: list[pd.DataFrame] = []
    strengths = sorted(plan["strength"].unique().tolist())
    for condition in ("low", "high"):
        selected = _state_rows(
            plan,
            run_idx=int(args.visual_run_idx),
            condition=condition,
            pair_id=int(args.visual_pair_id),
        )
        frames, rms = _load_evolution_frames(selected)
        rms.insert(0, "condition", condition)
        rms.insert(0, "pair_id", int(args.visual_pair_id))
        rms.insert(0, "run_idx", int(args.visual_run_idx))
        visual_rows.append(rms)
        montage = _plot_evolution_montage(
            frames=frames,
            strengths=strengths,
            condition=condition,
            run_idx=int(args.visual_run_idx),
            pair_id=int(args.visual_pair_id),
            figures=figures,
        )
        video = (
            figures
            / (
                f"c2_opt_{int(args.visual_run_idx):03d}_{condition}_"
                f"pair{int(args.visual_pair_id)}_noise_grid.mp4"
            )
        )
        _write_evolution_video(
            selected=selected,
            condition=condition,
            run_idx=int(args.visual_run_idx),
            pair_id=int(args.visual_pair_id),
            output=video,
            fps=float(args.video_fps),
        )
        visual_outputs[f"{condition}_evolution_montage"] = str(montage)
        visual_outputs[f"{condition}_evolution_video"] = str(video)
    visual_diagnostics = pd.concat(visual_rows, ignore_index=True)
    visual_diagnostics_path = full_root / "visual_branch_rms_diagnostics.csv"
    visual_diagnostics.to_csv(visual_diagnostics_path, index=False)

    outputs = {
        "inference_table": str(inference_path),
        "per_run_pearson_heatmaps": str(pearson_heatmaps),
        "per_run_spearman_heatmaps": str(spearman_heatmaps),
        "per_run_curves": str(per_run_curves),
        "per_run_manifest": str(per_run_manifest),
        "cluster_bootstrap_figure": str(uncertainty),
        "selection_heatmaps": str(linkage["linked_heatmap"]),
        "visual_branch_rms": str(visual_diagnostics_path),
        **visual_outputs,
    }
    report_path = full_root / "C2_NOISE_HORIZON_REPORT.md"
    outputs["report"] = str(report_path)
    _write_report(
        output=report_path,
        inference=inference,
        linkage=linkage,
        outputs=outputs,
    )
    published_dir = _publish_outputs(
        suite_root=suite_root,
        full_root=full_root,
        figures=figures,
        per_run_dir=per_run_dir,
        report_path=report_path,
    )
    outputs["published_dir"] = str(published_dir)
    _write_report(
        output=report_path,
        inference=inference,
        linkage=linkage,
        outputs=outputs,
    )
    shutil.copy2(report_path, published_dir / report_path.name)
    summary = {
        "status": "complete",
        "bootstrap_reps": int(args.bootstrap_reps),
        "seed": int(args.seed),
        "n_grid_cells": len(inference),
        "n_within_run_cells": len(within),
        "selection_linkage": linkage,
        "outputs": outputs,
    }
    write_json(full_root / "diagnostics_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Complete cached Flow-Lenia C2 noise/horizon diagnostics with "
            "run-level plots, cluster-bootstrap inference, and branch visuals."
        )
    )
    parser.add_argument("--suite-root", default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--bootstrap-reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--visual-run-idx", type=int, default=5)
    parser.add_argument("--visual-pair-id", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
