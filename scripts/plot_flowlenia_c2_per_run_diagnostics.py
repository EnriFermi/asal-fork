from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from c2_branch_selection_preview import _ensure_matplotlib, _load_heatmap_for_plot
from paper_suite_c2_branching import _get
from paper_suite_common import ensure_dir, load_config, write_csv, write_json


_RUN_RE = re.compile(r"run_(\d{3})(?:_|$)")
_CONDITION_ORDER = ("low", "mid", "high")
_PALETTE = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
_MARKERS = {"low": "o", "mid": "s", "high": "^"}
_LABEL_PREFIX = {"low": "L", "mid": "M", "high": "H"}


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_idx(traj_id: Any) -> int:
    match = _RUN_RE.search(str(traj_id))
    if match is None:
        raise ValueError(f"Could not parse source run index from trajectory id: {traj_id!r}")
    return int(match.group(1))


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot make coordinate edges for an empty axis.")
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    if not np.all(np.diff(values) > 0):
        raise ValueError(f"Heatmap coordinates must be strictly increasing: {values}")
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        [
            np.asarray([values[0] - (midpoints[0] - values[0])]),
            midpoints,
            np.asarray([values[-1] + (values[-1] - midpoints[-1])]),
        ]
    )


def _finite_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    result: dict[str, float | int] = {
        "n": int(x.size),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }
    if x.size < 3 or float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return result


def _padded_limits(values: np.ndarray, *, fraction: float = 0.06) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if span <= 0.0:
        span = max(abs(lo), 1.0)
    pad = fraction * span
    return lo - pad, hi + pad


def _draw_correlation(
    ax: Any,
    rows: pd.DataFrame,
    *,
    stats: dict[str, float | int],
    show_legend: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for condition in _CONDITION_ORDER:
        group = rows[rows["condition"] == condition]
        if group.empty:
            continue
        x = group["delta_h"].to_numpy(dtype=np.float64)
        y = group["branching_score"].to_numpy(dtype=np.float64)
        low = group["branching_score_ci_low"].to_numpy(dtype=np.float64)
        high = group["branching_score_ci_high"].to_numpy(dtype=np.float64)
        yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=_MARKERS[condition],
            linestyle="none",
            markersize=5.4,
            color=_PALETTE[condition],
            markerfacecolor=_PALETTE[condition],
            markeredgecolor="white",
            markeredgewidth=0.5,
            ecolor=_PALETTE[condition],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.82,
            label=condition,
            zorder=3,
        )

    x = rows["delta_h"].to_numpy(dtype=np.float64)
    y = rows["branching_score"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) >= 2 and float(np.std(x[finite])) > 1e-15:
        slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
        x_grid = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 160)
        ax.plot(x_grid, slope * x_grid + intercept, color="#252525", linewidth=1.5, zorder=2)

    ax.text(
        0.03,
        0.97,
        (
            f"n={int(stats['n'])}\n"
            f"Pearson r={float(stats['pearson_r']):.3f}\n"
            f"Spearman rho={float(stats['spearman_rho']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        zorder=5,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(color="#dedede", linewidth=0.65, alpha=0.72)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    if show_legend:
        ax.legend(title="stratum", frameon=False, loc="lower right", ncol=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_heatmap(
    ax: Any,
    *,
    centers: np.ndarray,
    tau_steps: np.ndarray,
    heatmap: np.ndarray,
    admissible_tau_idx: list[int],
    points: pd.DataFrame,
    vmax: float,
    annotate_points: bool,
    selection_ax: Any | None = None,
) -> Any:
    centers = np.asarray(centers, dtype=np.float64)
    tau_steps = np.asarray(tau_steps, dtype=np.float64)
    x_edges = _coordinate_edges(centers)
    y_edges = _coordinate_edges(tau_steps)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )

    admissible = set(int(idx) for idx in admissible_tau_idx)
    for idx in range(tau_steps.size):
        if idx not in admissible:
            ax.axhspan(y_edges[idx], y_edges[idx + 1], color="white", alpha=0.42, linewidth=0)

    for condition in _CONDITION_ORDER:
        group = points[points["condition"] == condition].sort_values("pair_id")
        for row in group.itertuples(index=False):
            step = float(row.window_center_step)
            ax.axvline(step, color=_PALETTE[condition], linewidth=1.05, alpha=0.82, zorder=3)
            if selection_ax is None:
                ax.scatter(
                    [step],
                    [1.015],
                    transform=ax.get_xaxis_transform(),
                    marker=_MARKERS[condition],
                    s=42,
                    color=_PALETTE[condition],
                    edgecolor="white",
                    linewidth=0.55,
                    clip_on=False,
                    zorder=5,
                )
                if annotate_points:
                    ax.annotate(
                        f"{_LABEL_PREFIX[condition]}{int(row.pair_id) + 1}",
                        xy=(step, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=7.5,
                        color=_PALETTE[condition],
                        clip_on=False,
                        zorder=6,
                    )
            else:
                selection_y = float(_CONDITION_ORDER.index(condition))
                selection_ax.scatter(
                    [step],
                    [selection_y],
                    marker=_MARKERS[condition],
                    s=44,
                    color=_PALETTE[condition],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=5,
                )
                if annotate_points:
                    selection_ax.annotate(
                        f"{_LABEL_PREFIX[condition]}{int(row.pair_id) + 1}",
                        xy=(step, selection_y),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=7.2,
                        color=_PALETTE[condition],
                        zorder=6,
                    )

    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if selection_ax is not None:
        selection_ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        selection_ax.set_ylim(-0.5, len(_CONDITION_ORDER) - 0.25)
        selection_ax.set_yticks(
            np.arange(len(_CONDITION_ORDER), dtype=np.float64),
            list(_CONDITION_ORDER),
        )
        selection_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        selection_ax.grid(axis="x", color="#dedede", linewidth=0.55, alpha=0.55)
        selection_ax.spines["top"].set_visible(False)
        selection_ax.spines["right"].set_visible(False)
        selection_ax.spines["bottom"].set_visible(False)
        selection_ax.spines["left"].set_linewidth(0.8)
    return mesh


def _load_inputs(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    branch_root = output_root / "c2_branching"
    plan_path = branch_root / "branch_plan.csv"
    scores_path = branch_root / "branching_scores_clip_chamfer.csv"
    for path in (plan_path, scores_path):
        if not path.exists():
            raise FileNotFoundError(path)

    plan = pd.read_csv(plan_path)
    scores = pd.read_csv(scores_path)
    required_plan = {
        "traj_id",
        "pair_id",
        "condition",
        "step",
        "window_center_step",
        "delta_h_energy",
        "source_metrics_path",
    }
    required_scores = {
        "traj_id",
        "pair_id",
        "condition",
        "step",
        "delta_h",
        "branching_score",
        "branching_score_ci_low",
        "branching_score_ci_high",
    }
    if missing := required_plan.difference(plan.columns):
        raise ValueError(f"{plan_path} is missing columns: {sorted(missing)}")
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"{scores_path} is missing columns: {sorted(missing)}")

    point_keys = ["traj_id", "pair_id", "condition"]
    points = (
        plan.sort_values(point_keys + ["step"])
        .drop_duplicates(point_keys, keep="first")
        .copy()
    )
    if len(points) != 150:
        raise ValueError(f"Expected 150 unique C2 branch points, found {len(points)}.")
    if scores.duplicated(point_keys).any():
        raise ValueError("C2 score table contains duplicate trajectory/condition/pair rows.")
    if len(scores) != 150:
        raise ValueError(f"Expected 150 C2 score rows, found {len(scores)}.")

    check = scores.merge(
        points[point_keys + ["step", "window_center_step", "delta_h_energy", "source_metrics_path"]],
        on=point_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_score", "_plan"),
    )
    if not bool((check["_merge"] == "both").all()):
        bad = check[check["_merge"] != "both"]
        raise ValueError(f"Score/plan branch-point keys differ:\n{bad[point_keys + ['_merge']]}")
    score_steps = check["step_score"].to_numpy(dtype=np.float64)
    plan_steps = check["step_plan"].to_numpy(dtype=np.float64)
    center_steps = check["window_center_step"].to_numpy(dtype=np.float64)
    if not np.array_equal(score_steps, plan_steps) or not np.array_equal(plan_steps, center_steps):
        raise ValueError("Score steps, plan steps, and Delta-H window centers are not identical.")
    if not np.allclose(
        check["delta_h"].to_numpy(dtype=np.float64),
        check["delta_h_energy"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=5e-12,
    ):
        raise ValueError("Score Delta-H values do not match the branch-plan selection energy.")

    points["run_idx"] = points["traj_id"].map(_run_idx)
    scores["run_idx"] = scores["traj_id"].map(_run_idx)
    expected_runs = list(range(10))
    if sorted(points["run_idx"].unique().tolist()) != expected_runs:
        raise ValueError(f"Expected source runs {expected_runs}, found {sorted(points['run_idx'].unique().tolist())}.")
    if sorted(scores["run_idx"].unique().tolist()) != expected_runs:
        raise ValueError(f"Expected score runs {expected_runs}, found {sorted(scores['run_idx'].unique().tolist())}.")

    for run_idx in expected_runs:
        run_points = points[points["run_idx"] == run_idx]
        run_scores = scores[scores["run_idx"] == run_idx]
        if len(run_points) != 15 or len(run_scores) != 15:
            raise ValueError(
                f"opt_{run_idx:03d}: expected 15 points/scores, found {len(run_points)}/{len(run_scores)}."
            )
        counts = run_points["condition"].value_counts().to_dict()
        if any(int(counts.get(condition, 0)) != 5 for condition in _CONDITION_ORDER):
            raise ValueError(f"opt_{run_idx:03d}: expected 5 low/mid/high points, found {counts}.")
        if run_points["traj_id"].nunique() != 1 or run_scores["traj_id"].nunique() != 1:
            raise ValueError(f"opt_{run_idx:03d}: expected exactly one source trajectory.")

    return points, scores, plan_path, scores_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config)
    output_root = _resolve(
        args.output_root
        or str(_get(cfg.get("meta", {}), "output_root", "analysis/results/paper_suite"))
    )
    figures_root = ensure_dir(output_root / "figures")
    detail_root = ensure_dir(
        _resolve(args.output_dir) if args.output_dir else figures_root / "c2_per_run"
    )
    points, scores, plan_path, scores_path = _load_inputs(output_root)

    heatmaps: dict[int, dict[str, Any]] = {}
    finite_heatmap_values: list[np.ndarray] = []
    for run_idx in range(10):
        run_points = points[points["run_idx"] == run_idx]
        metric_paths = run_points["source_metrics_path"].drop_duplicates().tolist()
        if len(metric_paths) != 1:
            raise ValueError(f"opt_{run_idx:03d}: expected one source metrics path, found {metric_paths}.")
        metrics_path = _resolve(str(metric_paths[0]))
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        centers, tau_steps, heatmap, admissible_tau_idx, meta = _load_heatmap_for_plot(metrics_path)
        selected_steps = run_points["window_center_step"].to_numpy(dtype=np.float64)
        if not np.isin(selected_steps, centers).all():
            raise ValueError(f"opt_{run_idx:03d}: selected branch times are absent from the Delta-H heatmap grid.")
        heatmaps[run_idx] = {
            "centers": centers,
            "tau_steps": tau_steps,
            "heatmap": heatmap,
            "admissible_tau_idx": admissible_tau_idx,
            "meta": meta,
            "metrics_path": metrics_path,
        }
        finite = heatmap[np.isfinite(heatmap)]
        if finite.size:
            finite_heatmap_values.append(finite)
    if not finite_heatmap_values:
        raise ValueError("All source Delta-H heatmaps are empty or non-finite.")
    all_heatmap_values = np.concatenate(finite_heatmap_values)
    vmax = max(float(np.percentile(all_heatmap_values, float(args.vmax_percentile))), 1e-12)

    all_x = scores["delta_h"].to_numpy(dtype=np.float64)
    all_y = np.concatenate(
        [
            scores["branching_score"].to_numpy(dtype=np.float64),
            scores["branching_score_ci_low"].to_numpy(dtype=np.float64),
            scores["branching_score_ci_high"].to_numpy(dtype=np.float64),
        ]
    )
    xlim = _padded_limits(all_x)
    ylim = _padded_limits(all_y)

    plt = _ensure_matplotlib()
    style = {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
    }
    manifest_rows: list[dict[str, Any]] = []

    with plt.rc_context(style):
        correlation_overview, corr_axes = plt.subplots(
            5,
            2,
            figsize=(12.5, 20.0),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        heatmap_overview, heat_axes = plt.subplots(
            5,
            2,
            figsize=(14.0, 20.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        overview_mesh = None

        for run_idx in range(10):
            run_scores = scores[scores["run_idx"] == run_idx].copy()
            run_points = points[points["run_idx"] == run_idx].copy()
            stats = _finite_correlation(
                run_scores["delta_h"].to_numpy(dtype=np.float64),
                run_scores["branching_score"].to_numpy(dtype=np.float64),
            )

            correlation_path = detail_root / f"opt_{run_idx:03d}_correlation.png"
            fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
            _draw_correlation(
                ax,
                run_scores,
                stats=stats,
                show_legend=True,
                xlim=xlim,
                ylim=ylim,
            )
            ax.set_title(f"opt_{run_idx:03d}: Delta-H vs future divergence")
            ax.set_xlabel("branch energy E_b = mean_tau phi(Delta-H)")
            ax.set_ylabel("future divergence B_b (CLIP-Chamfer cosine)")
            fig.savefig(correlation_path, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)

            overview_ax = corr_axes.flat[run_idx]
            _draw_correlation(
                overview_ax,
                run_scores,
                stats=stats,
                show_legend=False,
                xlim=xlim,
                ylim=ylim,
            )
            overview_ax.set_title(f"opt_{run_idx:03d}")
            if run_idx // 2 == 4:
                overview_ax.set_xlabel("branch energy E_b")
            if run_idx % 2 == 0:
                overview_ax.set_ylabel("future divergence B_b")

            heatmap_path = detail_root / f"opt_{run_idx:03d}_delta_h_heatmap.png"
            heatmap_data = heatmaps[run_idx]
            fig, (selection_ax, ax) = plt.subplots(
                2,
                1,
                figsize=(10.8, 6.6),
                sharex=True,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [1.0, 4.5]},
            )
            mesh = _draw_heatmap(
                ax,
                centers=heatmap_data["centers"],
                tau_steps=heatmap_data["tau_steps"],
                heatmap=heatmap_data["heatmap"],
                admissible_tau_idx=heatmap_data["admissible_tau_idx"],
                points=run_points,
                vmax=vmax,
                annotate_points=True,
                selection_ax=selection_ax,
            )
            fig.suptitle(
                f"opt_{run_idx:03d}: Delta-H map with exact C2 branch times",
                fontsize=13,
            )
            selection_ax.set_title(
                "Selected branch times by stratum; energy = mean_tau phi(Delta-H)",
                fontsize=9,
                loc="left",
            )
            ax.set_xlabel("window center step")
            ax.set_ylabel("tau (steps)")
            fig.colorbar(mesh, ax=ax, pad=0.02, label="phi(Delta-H)")
            fig.savefig(heatmap_path, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)

            overview_heat_ax = heat_axes.flat[run_idx]
            overview_mesh = _draw_heatmap(
                overview_heat_ax,
                centers=heatmap_data["centers"],
                tau_steps=heatmap_data["tau_steps"],
                heatmap=heatmap_data["heatmap"],
                admissible_tau_idx=heatmap_data["admissible_tau_idx"],
                points=run_points,
                vmax=vmax,
                annotate_points=False,
            )
            overview_heat_ax.set_title(f"opt_{run_idx:03d}")
            if run_idx // 2 == 4:
                overview_heat_ax.set_xlabel("window center step")
            if run_idx % 2 == 0:
                overview_heat_ax.set_ylabel("tau (steps)")

            by_condition = {
                condition: run_points[run_points["condition"] == condition]
                .sort_values("pair_id")["window_center_step"]
                .astype(int)
                .tolist()
                for condition in _CONDITION_ORDER
            }
            manifest_rows.append(
                {
                    "run_idx": run_idx,
                    "run_id": f"opt_{run_idx:03d}",
                    "traj_id": str(run_scores["traj_id"].iloc[0]),
                    **stats,
                    "n_low": int((run_points["condition"] == "low").sum()),
                    "n_mid": int((run_points["condition"] == "mid").sum()),
                    "n_high": int((run_points["condition"] == "high").sum()),
                    "low_steps": ",".join(str(value) for value in by_condition["low"]),
                    "mid_steps": ",".join(str(value) for value in by_condition["mid"]),
                    "high_steps": ",".join(str(value) for value in by_condition["high"]),
                    "source_metrics_path": str(heatmap_data["metrics_path"]),
                    "correlation_figure": str(correlation_path),
                    "delta_h_heatmap_figure": str(heatmap_path),
                }
            )

        correlation_overview.suptitle(
            "Flow-Lenia C2 within-run associations: Delta-H vs future divergence",
            fontsize=14,
        )
        correlation_overview_path = figures_root / "c2_per_run_correlations.png"
        correlation_overview.savefig(
            correlation_overview_path,
            dpi=int(args.dpi),
            bbox_inches="tight",
        )
        plt.close(correlation_overview)

        heatmap_overview.suptitle(
            "Flow-Lenia C2 Delta-H maps with exact selected branch times\n"
            "Blue circles: low; orange squares: mid; green triangles: high. Markers encode time only, not tau",
            fontsize=14,
        )
        if overview_mesh is not None:
            heatmap_overview.colorbar(
                overview_mesh,
                ax=list(heat_axes.flat),
                pad=0.015,
                shrink=0.9,
                label="phi(Delta-H)",
            )
        heatmap_overview_path = figures_root / "c2_per_run_delta_h_heatmaps.png"
        heatmap_overview.savefig(
            heatmap_overview_path,
            dpi=int(args.dpi),
            bbox_inches="tight",
        )
        plt.close(heatmap_overview)

    manifest_path = detail_root / "manifest.csv"
    write_csv(manifest_path, manifest_rows)
    summary_path = detail_root / "summary.json"
    write_json(
        summary_path,
        {
            "status": "complete",
            "runs": 10,
            "points_per_run": 15,
            "selection_counts_per_run": {"low": 5, "mid": 5, "high": 5},
            "selection_coordinate": "window_center_step",
            "selection_energy": "mean_tau phi(Delta-H)",
            "marker_semantics": "Markers denote selected branch times only; selection has no single tau coordinate.",
            "heatmap_vmax_percentile": float(args.vmax_percentile),
            "heatmap_vmax": vmax,
            "inputs": {
                "branch_plan": str(plan_path),
                "branch_plan_sha256": _sha256(plan_path),
                "branching_scores": str(scores_path),
                "branching_scores_sha256": _sha256(scores_path),
            },
            "outputs": {
                "correlation_overview": str(correlation_overview_path),
                "heatmap_overview": str(heatmap_overview_path),
                "per_run_directory": str(detail_root),
                "manifest": str(manifest_path),
            },
        },
    )
    return {
        "correlation_overview": str(correlation_overview_path),
        "heatmap_overview": str(heatmap_overview_path),
        "per_run_directory": str(detail_root),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-run Flow-Lenia C2 correlations and exact selected Delta-H branch times."
    )
    parser.add_argument("config")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vmax-percentile", type=float, default=98.0)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if not 50.0 <= float(args.vmax_percentile) <= 100.0:
        parser.error("--vmax-percentile must be between 50 and 100.")
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
