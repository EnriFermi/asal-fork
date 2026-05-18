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
from paper_suite_synthetic import visualize as visualize_synthetic


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
    role_path = output_root / "synthetic_calibration" / "role_recovery.csv"
    event_path = output_root / "synthetic_calibration" / "event_localization.csv"
    if not tau_path.exists() or not score_path.exists():
        return {}
    plt = _ensure_matplotlib()
    tau = pd.read_csv(tau_path)
    scores = pd.read_csv(score_path)
    role = pd.read_csv(role_path) if role_path.exists() and role_path.stat().st_size > 1 else pd.DataFrame()
    event = pd.read_csv(event_path) if event_path.exists() and event_path.stat().st_size > 1 else pd.DataFrame()
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
    out_paths = {"synthetic_calibration_grid": str(path)}
    if {"amp_by_tau", "msc_by_tau", "delta_h_mean"}.issubset(tau.columns):
        fig, axes = plt.subplots(len(families), 6, figsize=(17, max(2.2, 1.8 * len(families))), squeeze=False)
        for row_idx, family in enumerate(families):
            sub = tau[tau["family"] == family].sort_values("tau_steps")
            for col_idx, (col, title) in enumerate(
                [
                    ("score_by_tau", "score"),
                    ("amp_by_tau", "amp"),
                    ("msc_by_tau", "msc"),
                    ("delta_h_mean", "Delta-H"),
                ]
            ):
                ax = axes[row_idx, col_idx]
                ax.plot(sub["tau_steps"], sub[col], marker="o", linewidth=1.2)
                selected = sub[sub["selected"].astype(bool)]
                if not selected.empty:
                    ax.scatter(selected["tau_steps"], selected[col], color="#d62728", s=28, zorder=3)
                ax.set_xscale("log")
                ax.set_title(title if row_idx == 0 else "")
                ax.set_ylabel(family if col_idx == 0 else "")
            ax_event = axes[row_idx, 4]
            event_error_col = "event_error_steps" if "event_error_steps" in event.columns else "peak_error_steps"
            if not event.empty and "family" in event.columns and event_error_col in event.columns:
                ev = event[event["family"] == family][event_error_col].astype(float).to_numpy()
            else:
                ev = np.asarray([], dtype=np.float64)
            if ev.size:
                ax_event.scatter(np.arange(ev.size), ev, color="#9467bd", s=28)
                ax_event.axhline(float(np.nanmedian(ev)), color="#111111", linewidth=1)
            else:
                ax_event.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax_event.transAxes)
            ax_event.set_title("event error" if row_idx == 0 else "")
            ax_event.set_xticks([])

            ax_role = axes[row_idx, 5]
            if not role.empty and "family" in role.columns and "ari" in role.columns:
                ari = pd.to_numeric(role[role["family"] == family]["ari"], errors="coerce").dropna().to_numpy()
            else:
                ari = np.asarray([], dtype=np.float64)
            if ari.size:
                ax_role.scatter(np.arange(ari.size), ari, color="#2ca02c", s=28)
                ax_role.set_ylim(-0.05, 1.05)
                ax_role.axhline(float(np.nanmedian(ari)), color="#111111", linewidth=1)
            else:
                ax_role.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax_role.transAxes)
            ax_role.set_title("ARI" if row_idx == 0 else "")
            ax_role.set_xticks([])
        fig.tight_layout()
        decomp = figures / "synthetic_decomposition_grid.png"
        fig.savefig(decomp, dpi=180)
        plt.close(fig)
        out_paths["synthetic_decomposition_grid"] = str(decomp)
    return out_paths


def _plot_c1_tau_profiles(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    maps_dir = ds_dir / "c1_delta_h_maps"
    score_path = ds_dir / "checkpoint_scores.csv"
    if not maps_dir.exists() or not score_path.exists():
        return {}
    scores = pd.read_csv(score_path)
    if scores.empty or "maps_path" not in scores.columns:
        return {}
    rows = []
    for rec in scores.itertuples():
        path = Path(str(rec.maps_path))
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            if "selection_score_by_tau" not in data.files or "eval_score_by_tau" not in data.files:
                continue
            tau = np.asarray(data["tau_steps"], dtype=np.float64)
            sel = np.asarray(data["selection_score_by_tau"], dtype=np.float64)
            ev = np.asarray(data["eval_score_by_tau"], dtype=np.float64)
        for t, s, e in zip(tau, sel, ev):
            rows.append({"kind": str(rec.candidate_kind), "tau": float(t), "selection": float(s), "eval": float(e)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, col in zip(axes, ["selection", "eval"]):
        for kind, color in (("optimized", "#d62728"), ("random", "#4c78a8")):
            sub = df[df["kind"] == kind]
            if sub.empty:
                continue
            g = sub.groupby("tau")[col].median().reset_index()
            ax.plot(g["tau"], g[col], marker="o", label=kind, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("tau steps")
        ax.set_ylabel("MSPD score")
        ax.set_title(col)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out = figures / f"c1_{dataset}_tau_profiles.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {f"c1_{dataset}_tau_profiles": str(out)}


def _plot_c1_heatmaps(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    score_path = ds_dir / "checkpoint_scores.csv"
    if not score_path.exists():
        return {}
    scores = pd.read_csv(score_path)
    if scores.empty or "maps_path" not in scores.columns:
        return {}
    picks = []
    for kind in ("optimized", "random"):
        sub = scores[scores["candidate_kind"] == kind]
        if not sub.empty:
            picks.append((kind, Path(str(sub.iloc[0]["maps_path"]))))
    if not picks:
        return {}
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(len(picks), 2, figsize=(8, max(2.8, 2.5 * len(picks))), squeeze=False)
    for row_idx, (kind, path) in enumerate(picks):
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            sel = np.asarray(data["delta_h_selection"], dtype=np.float64)
            ev = np.asarray(data["delta_h_eval"], dtype=np.float64)
        for col_idx, (arr, title) in enumerate(((sel, "selection"), (ev, "eval"))):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis")
            ax.set_title(f"{kind} {title}")
            ax.set_xlabel("window")
            ax.set_ylabel("tau index")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    out = figures / f"c1_{dataset}_delta_h_heatmaps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {f"c1_{dataset}_delta_h_heatmaps": str(out)}


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
    paths = {f"c1_{dataset}_paired_contrast": str(out)}
    paths.update(_plot_c1_tau_profiles(dataset, ds_dir, figures))
    paths.update(_plot_c1_heatmaps(dataset, ds_dir, figures))
    return paths


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
    paths = {f"c5_{dataset}_frustration_contrast": str(out)}
    emb_cols = [c for c in delta_cols if "embedding" in c]
    mspd_cols = [c for c in delta_cols if c.startswith("delta_h_") or c.startswith("msc_")]
    if emb_cols and mspd_cols:
        emb = emb_cols[0]
        mspd = mspd_cols[0]
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.axhline(0.0, color="#777777", linewidth=1)
        ax.axvline(0.0, color="#777777", linewidth=1)
        ax.scatter(df[emb].astype(float), df[mspd].astype(float), s=38, color="#4c78a8")
        ax.set_xlabel("embedding frustration contrast")
        ax.set_ylabel("MSPD frustration contrast")
        ax.set_title(f"C5 axes: {dataset}")
        fig.tight_layout()
        out2 = figures / f"c5_{dataset}_embedding_vs_mspd.png"
        fig.savefig(out2, dpi=180)
        plt.close(fig)
        paths[f"c5_{dataset}_embedding_vs_mspd"] = str(out2)
    return paths


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
    out_alias = figures / "c2_high_vs_low_branching_divergence.png"
    fig.savefig(out_alias, dpi=180)
    plt.close(fig)
    return {"c2_branching_sensitivity": str(out), "c2_high_vs_low_branching_divergence": str(out_alias)}


def run(config_path: str | Path, *, task: str = "all", smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    figures = ensure_dir(output_root / "figures")
    paths: dict[str, str] = {}
    log_event(f"visualization start task={task} smoke={smoke} force={force} figures={figures}", component="visualization")
    if task in {"all", "synthetic"}:
        paths.update(_plot_synthetic(output_root, figures))
        synthetic_result = visualize_synthetic(config_path, smoke=smoke, force=force)
        paths.update({str(k): str(v) for k, v in synthetic_result.get("figure_paths", {}).items()})
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, task=args.task, smoke=args.smoke, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
