from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

from paper_suite_common import dataset_items, ensure_dir, load_config, log_event, resolve_path, write_json


def _ensure_matplotlib():
    import tempfile

    cache_root = Path(tempfile.gettempdir()) / "paper_suite_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _output_root(cfg: Any) -> Path:
    return ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))


def _plot_synthetic(output_root: Path, figures: Path) -> dict[str, str]:
    tau_path = output_root / "synthetic_calibration" / "tau_profiles.csv"
    score_path = output_root / "synthetic_calibration" / "per_family_scores.csv"
    if not tau_path.exists() or not score_path.exists():
        return {}
    plt = _ensure_matplotlib()
    tau = pd.read_csv(tau_path)
    scores = pd.read_csv(score_path)
    families = [f for f in ["S0", "S1", "S3", "S4", "S5", "S6", "S7"] if f in set(tau["family"])]
    if not families:
        return {}
    fig, axes = plt.subplots(len(families), 2, figsize=(9, max(2.2, 2.0 * len(families))), squeeze=False)
    for row_idx, family in enumerate(families):
        ax0, ax1 = axes[row_idx]
        sub = tau[tau["family"] == family]
        grouped = sub.groupby("tau_steps")["score_by_tau"].agg(["median", "min", "max"]).reset_index()
        ax0.plot(grouped["tau_steps"], grouped["median"], marker="o", color="#1f77b4")
        ax0.fill_between(grouped["tau_steps"], grouped["min"], grouped["max"], color="#1f77b4", alpha=0.15)
        ax0.set_xscale("log")
        ax0.set_ylabel(family)
        ax0.set_xlabel("tau")
        ax0.set_title("D(tau)")

        vals = scores[scores["family"] == family]["score"].astype(float).to_numpy()
        ax1.scatter(np.arange(vals.size), vals, color="#444444", s=24)
        if vals.size:
            ax1.axhline(float(np.median(vals)), color="#d62728", linewidth=1)
        ax1.set_xlabel("seed")
        ax1.set_title("selected score")
    fig.tight_layout()
    path = figures / "synthetic_calibration_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"synthetic_calibration_grid": str(path)}


def _plot_c1(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    path = ds_dir / "group_contrasts.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "delta_vs_random_median" not in df.columns:
        return {}
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(df.shape[0])
    y = df["delta_vs_random_median"].astype(float).to_numpy()
    ax.axhline(0.0, color="#777777", linewidth=1)
    ax.scatter(x, y, color=np.where(y >= 0, "#2ca02c", "#d62728"), s=42)
    if y.size:
        ax.axhline(float(np.median(y)), color="#111111", linestyle="--", linewidth=1)
    ax.set_xlabel("matched group")
    ax.set_ylabel("optimized - random median")
    ax.set_title(f"C1 selection-adjusted contrast: {dataset}")
    fig.tight_layout()
    out = figures / f"c1_{dataset}_paired_contrast.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {f"c1_{dataset}_paired_contrast": str(out)}


def _plot_c5(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    path = ds_dir / "frustration_run_level.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    delta_cols = [c for c in df.columns if c.endswith("__delta_vs_random_median")]
    preferred = [c for c in delta_cols if "embedding_cloud_chamfer_cosine" in c]
    col = preferred[0] if preferred else (delta_cols[0] if delta_cols else None)
    if col is None:
        return {}
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = df[col].astype(float).to_numpy()
    x = np.arange(y.size)
    ax.axhline(0.0, color="#777777", linewidth=1)
    ax.bar(x, y, color=np.where(y >= 0, "#2ca02c", "#d62728"))
    ax.set_xlabel("matched group")
    ax.set_ylabel("optimized - random median")
    ax.set_title(f"C5 frustration contrast: {dataset}")
    fig.tight_layout()
    out = figures / f"c5_{dataset}_frustration_contrast.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {f"c5_{dataset}_frustration_contrast": str(out)}


def _plot_cross(output_root: Path, figures: Path) -> dict[str, str]:
    path = output_root / "cross_substrate_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "median" not in df.columns:
        return {}
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [f"{r.dataset}\n{r.claim}" for r in df.itertuples()]
    vals = df["median"].astype(float).to_numpy()
    ax.axhline(0.0, color="#777777", linewidth=1)
    ax.bar(np.arange(vals.size), vals, color="#4c78a8")
    ax.set_xticks(np.arange(vals.size), labels, rotation=30, ha="right")
    ax.set_ylabel("median delta")
    ax.set_title("Cross-substrate paper-suite effects")
    fig.tight_layout()
    out = figures / "c6_cross_substrate_effects.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"c6_cross_substrate_effects": str(out)}


def _plot_c2_branching(output_root: Path, figures: Path) -> dict[str, str]:
    scores_path = output_root / "c2_branching" / "branching_scores.csv"
    contrasts_path = output_root / "c2_branching" / "branching_pair_contrasts.csv"
    if not scores_path.exists() or not contrasts_path.exists():
        return {}
    scores = pd.read_csv(scores_path)
    contrasts = pd.read_csv(contrasts_path)
    if scores.empty or contrasts.empty:
        return {}
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax0, ax1 = axes

    for idx, row in enumerate(contrasts.itertuples()):
        low = float(row.low_branching_score)
        high = float(row.high_branching_score)
        ax0.plot([0, 1], [low, high], color="#888888", alpha=0.7, linewidth=1)
        ax0.scatter([0, 1], [low, high], color=["#4c78a8", "#d62728"], s=28, zorder=3)
    diffs = contrasts["delta_branching_score"].astype(float).to_numpy()
    if diffs.size:
        ax0.set_title(f"paired branching; median delta={np.nanmedian(diffs):.3g}")
    else:
        ax0.set_title("paired branching")
    ax0.set_xticks([0, 1], ["low", "high"])
    ax0.set_ylabel("branch divergence")

    colors = np.where(scores["condition"].astype(str).to_numpy() == "high", "#d62728", "#4c78a8")
    ax1.scatter(scores["delta_h"].astype(float), scores["branching_score"].astype(float), c=colors, s=36)
    ax1.set_xlabel("Delta-H at branch time")
    ax1.set_ylabel("branch divergence")
    ax1.set_title("Delta-H vs future divergence")
    fig.tight_layout()
    out = figures / "c2_branching_sensitivity.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"c2_branching_sensitivity": str(out)}


def run(config_path: str | Path, *, task: str = "all", smoke: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    figures = ensure_dir(output_root / "figures")
    paths: dict[str, str] = {}
    log_event(f"visualization start task={task} smoke={smoke} figures={figures}", component="visualization")
    if task in {"all", "synthetic"}:
        paths.update(_plot_synthetic(output_root, figures))
    if task in {"all", "c2"}:
        paths.update(_plot_c2_branching(output_root, figures))
    if task in {"all", "c1", "c5", "c6"}:
        for dataset, _ds in dataset_items(cfg):
            ds_dir = output_root / dataset
            if task in {"all", "c1", "c6"}:
                paths.update(_plot_c1(dataset, ds_dir, figures))
            if task in {"all", "c5", "c6"}:
                paths.update(_plot_c5(dataset, ds_dir, figures))
        paths.update(_plot_cross(output_root, figures))
    write_json(output_root / "visualization_summary.json", {"figure_paths": paths})
    log_event(f"visualization done n_figures={len(paths)} summary={output_root / 'visualization_summary.json'}", component="visualization")
    return {"n_figures": len(paths), "figure_paths": paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-suite visualization layer.")
    parser.add_argument("config")
    parser.add_argument("--task", choices=["all", "synthetic", "c1", "c2", "c5", "c6"], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, task=args.task, smoke=args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
