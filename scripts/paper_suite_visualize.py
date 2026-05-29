from __future__ import annotations

import argparse
import json
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

from paper_suite_common import dataset_items, ensure_dir, load_config, log_event, resolve_path, sign_test_greater, write_json
from paper_suite_synthetic import visualize as visualize_synthetic


_VISUALIZATION_SKIPS: list[dict[str, str]] = []


def _record_skip(plot: str, reason: str) -> None:
    _VISUALIZATION_SKIPS.append({"plot": plot, "reason": reason})
    log_event(f"{plot} skipped: {reason}", component="visualization")


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


_FAMILY_ORDER = ["S0", "S1", "S3", "S4", "S5", "S6", "S7", "S8"]


def _read_csv_if_exists(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists() and path.stat().st_size > 1:
            return pd.read_csv(path)
    return pd.DataFrame()


def _synthetic_table(output_root: Path, short_name: str) -> pd.DataFrame:
    return _read_csv_if_exists(
        output_root / "synthetic_calibration" / f"{short_name}.csv",
        output_root / f"synthetic_calibration_{short_name}.csv",
        output_root / f"{short_name}.csv",
    )


def _finite_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _median_or_nan(values: Any) -> float:
    arr = _finite_array(values)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def _mad_or_eps(values: Any, eps: float = 1e-12) -> float:
    arr = _finite_array(values)
    if arr.size == 0:
        return eps
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    return max(mad, eps)


def _format_p(value: Any) -> str:
    try:
        p = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def _maybe_log_y(ax: Any, values: Any) -> None:
    arr = _finite_array(values)
    positive = arr[arr > 0]
    if positive.size < 2:
        return
    ratio = float(np.nanmax(positive) / max(np.nanmin(positive), 1e-300))
    if ratio > 200.0:
        if np.any(arr <= 0):
            ax.set_yscale("symlog", linthresh=max(float(np.nanmin(positive)) * 0.5, 1e-12))
        else:
            ax.set_yscale("log")


def _save(fig: Any, path: Path, *, dpi: int = 220) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return str(path)


def _plot_family_points(
    ax: Any,
    df: pd.DataFrame,
    *,
    value_col: str,
    families: list[str],
    color: str,
    ylabel: str,
) -> np.ndarray:
    x = np.arange(len(families), dtype=np.float64)
    medians = []
    for i, family in enumerate(families):
        sub = df[df["family"].astype(str) == family]
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        med = float(np.nanmedian(vals)) if vals.size else float("nan")
        medians.append(med)
        if vals.size:
            ax.bar(i, med, color=color, alpha=0.22, width=0.68, edgecolor="none", zorder=1)
            jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.asarray([0.0])
            ax.scatter(i + jitter, vals, s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(x, families)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    return np.asarray(medians, dtype=np.float64)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def _plot_synthetic(output_root: Path, figures: Path) -> dict[str, str]:
    tau_path = output_root / "synthetic_calibration" / "tau_profiles.csv"
    score_path = output_root / "synthetic_calibration" / "per_family_scores.csv"
    if not tau_path.exists() or not score_path.exists():
        return {}
    plt = _ensure_matplotlib()
    tau = pd.read_csv(tau_path)
    scores = pd.read_csv(score_path)
    families = [f for f in ["S0", "S1", "S3", "S4", "S5", "S6", "S7", "S8"] if f in set(tau["family"])]
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
        fig, axes = plt.subplots(len(families), 4, figsize=(12.0, max(2.2, 1.8 * len(families))), squeeze=False)
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
        fig.tight_layout()
        decomp = figures / "synthetic_decomposition_grid.png"
        fig.savefig(decomp, dpi=180)
        plt.close(fig)
        out_paths["synthetic_decomposition_grid"] = str(decomp)
    return out_paths


def _plot_synthetic_summary_clean(output_root: Path, figures: Path) -> dict[str, str]:
    scores = _synthetic_table(output_root, "per_family_scores")
    if scores.empty or "family" not in scores.columns or "score" not in scores.columns:
        return {}
    families = [f for f in _FAMILY_ORDER if f in set(scores["family"].astype(str))]
    if not families:
        return {}
    amp_col = "amp" if "amp" in scores.columns else "delta_h_processed_mean"
    if amp_col not in scores.columns:
        amp_col = "delta_h_mean" if "delta_h_mean" in scores.columns else ""

    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), constrained_layout=True)
    ax_a, ax_b = axes.ravel()

    score_medians = _plot_family_points(
        ax_a,
        scores,
        value_col="score",
        families=families,
        color="#4c78a8",
        ylabel="selected MSPD score",
    )
    _maybe_log_y(ax_a, scores["score"])
    ax_a.set_title("Selected MSPD by synthetic family")
    _add_panel_label(ax_a, "A")
    ax_a.text(0.5, 0.95, "null controls", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#555555")
    positive_x = [families.index(fam) for fam in ("S6", "S8") if fam in families]
    if positive_x:
        ax_a.text(float(np.mean(positive_x)), 0.95, "positive controls", transform=ax_a.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#6f4e9b")

    if amp_col:
        _plot_family_points(
            ax_b,
            scores,
            value_col=amp_col,
            families=families,
            color="#f58518",
            ylabel="Delta-H amplitude / processed mean",
        )
        _maybe_log_y(ax_b, scores[amp_col])
    ax_b.set_title("Delta-H amplitude")
    _add_panel_label(ax_b, "B")

    for ax in axes:
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = figures / "synthetic_calibration_summary_clean.png"
    _save(fig, out)
    plt.close(fig)
    return {"synthetic_calibration_summary_clean": str(out)}


def _metric_window_centers(data: Any) -> np.ndarray:
    starts = np.asarray(data["window_start_steps"], dtype=np.float64) if "window_start_steps" in data.files else np.arange(np.asarray(data["delta_h_map"]).shape[1], dtype=np.float64)
    window_size = 0.0
    if "_metric_config_json" in data.files:
        try:
            cfg = json.loads(str(np.asarray(data["_metric_config_json"]).item()))
            window_size = float(cfg.get("window_size_steps", cfg.get("window_size_frames", 0.0)))
        except Exception:
            window_size = 0.0
    return starts + 0.5 * window_size


def _load_synthetic_heatmap_item(row: pd.Series) -> dict[str, Any] | None:
    path = Path(str(row.get("metrics_path", "")))
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            key = "delta_h_processed_map" if "delta_h_processed_map" in data.files else "delta_h_map"
            z = np.asarray(data[key], dtype=np.float64)
            tau = np.asarray(data["tau_steps"], dtype=np.float64) if "tau_steps" in data.files else np.arange(z.shape[0], dtype=np.float64)
            centers = _metric_window_centers(data)
    except Exception:
        return None
    if z.ndim != 2 or z.size == 0:
        return None
    return {"z": z, "tau": tau, "centers": centers, "path": str(path)}


def _plot_synthetic_delta_h_heatmaps_clean(output_root: Path, figures: Path) -> dict[str, str]:
    scores = _synthetic_table(output_root, "per_family_scores")
    events = _synthetic_table(output_root, "event_localization")
    if scores.empty or "family" not in scores.columns or "metrics_path" not in scores.columns:
        return {}
    families = [f for f in ["S0", "S1", "S4", "S5", "S6", "S8"] if f in set(scores["family"].astype(str))]
    loaded: list[tuple[str, dict[str, Any]]] = []
    for family in families:
        sub = scores[scores["family"].astype(str) == family].sort_values("seed" if "seed" in scores.columns else "family")
        if sub.empty:
            continue
        item = _load_synthetic_heatmap_item(sub.iloc[0])
        if item is not None:
            loaded.append((family, item))
    if not loaded:
        return {}
    finite_vals = np.concatenate([item["z"][np.isfinite(item["z"])] for _family, item in loaded if np.any(np.isfinite(item["z"]))])
    vmax = float(np.nanpercentile(finite_vals, 99.0)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-12)

    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.0 * len(loaded) + 0.9, 3.8), constrained_layout=True, squeeze=False)
    ims = []
    for ax, (family, item) in zip(axes.ravel(), loaded, strict=True):
        z = item["z"]
        tau = item["tau"]
        centers = item["centers"]
        x0, x1 = float(np.nanmin(centers)), float(np.nanmax(centers))
        if centers.size > 1:
            step = float(np.nanmedian(np.diff(centers)))
            x0 -= 0.5 * step
            x1 += 0.5 * step
        y0, y1 = float(np.nanmin(tau)), float(np.nanmax(tau))
        im = ax.imshow(z, aspect="auto", origin="lower", interpolation="nearest", extent=[x0, x1, y0, y1], cmap="magma", vmin=0.0, vmax=vmax)
        ims.append(im)
        if family in {"S6", "S8"} and not events.empty:
            ev = events[events["family"].astype(str) == family]
            if not ev.empty and {"event_start", "event_end"}.issubset(ev.columns):
                for rec in ev.itertuples():
                    ax.axvspan(
                        float(rec.event_start),
                        float(rec.event_end),
                        facecolor="#6baed6",
                        edgecolor="#2166ac",
                        alpha=0.22,
                        linewidth=1.0,
                        zorder=3,
                    )
        ax.set_title(family, fontsize=13, fontweight="bold")
        ax.set_xlabel("window center step", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
    axes.ravel()[0].set_ylabel("tau", fontsize=10)
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
    cbar.set_label("processed Delta-H", fontsize=10)
    out = figures / "synthetic_delta_h_heatmaps_clean.png"
    _save(fig, out)
    plt.close(fig)
    return {"synthetic_delta_h_heatmaps_clean": str(out)}


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
    paths = {f"c1_{dataset}_delta_h_heatmaps": str(out)}
    paths.update(_plot_c1_optimized_vs_random_delta_h_heatmaps(dataset, scores, figures))
    return paths


def _load_c1_eval_map(row: pd.Series) -> dict[str, Any] | None:
    path = Path(str(row.get("maps_path", "")))
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "delta_h_eval" not in data.files:
                return None
            delta_h_eval = np.asarray(data["delta_h_eval"], dtype=np.float64)
            tau_steps = (
                np.asarray(data["tau_steps"], dtype=np.int64)
                if "tau_steps" in data.files
                else np.arange(delta_h_eval.shape[0], dtype=np.int64)
            )
            selected_tau_idx = (
                int(np.asarray(data["selected_tau_idx"]).reshape(-1)[0])
                if "selected_tau_idx" in data.files
                else 0
            )
            out = {
                "delta_h_eval": delta_h_eval,
                "tau_steps": tau_steps,
                "selected_tau_idx": selected_tau_idx,
            }
    except Exception:
        return None
    return out


def _selected_eval_scores_from_tau_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {"group", "candidate_kind", "candidate_idx", "split", "tau_steps", "score"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    rows = []
    for (group, kind, candidate_idx), sub in raw.groupby(["group", "candidate_kind", "candidate_idx"], dropna=False):
        sel = sub[sub["split"].astype(str) == "selection"].copy()
        ev = sub[sub["split"].astype(str) == "eval"].copy()
        if sel.empty:
            continue
        sel["score_numeric"] = pd.to_numeric(sel["score"], errors="coerce")
        sel = sel.dropna(subset=["score_numeric"])
        if sel.empty:
            continue
        tau = sel.loc[sel["score_numeric"].idxmax(), "tau_steps"]
        ev_same = ev[pd.to_numeric(ev["tau_steps"], errors="coerce") == float(tau)]
        if ev_same.empty:
            continue
        score = pd.to_numeric(ev_same.iloc[0]["score"], errors="coerce")
        if not np.isfinite(float(score)):
            continue
        rows.append(
            {
                "optimized_run_idx": group,
                "candidate_kind": str(kind),
                "candidate_idx": candidate_idx,
                "eval_score_mspd": float(score),
            }
        )
    return pd.DataFrame(rows)


def _load_c1_raw_scores(ds_dir: Path) -> pd.DataFrame:
    path = ds_dir / "checkpoint_scores.csv"
    if path.exists() and path.stat().st_size > 1:
        df = pd.read_csv(path)
        if not df.empty and {"candidate_kind", "eval_score_mspd"}.issubset(df.columns):
            if "optimized_run_idx" not in df.columns:
                group_col = "group" if "group" in df.columns else None
                if group_col is not None:
                    df["optimized_run_idx"] = df[group_col]
            if "optimized_run_idx" in df.columns:
                return df
    return _selected_eval_scores_from_tau_table(ds_dir / "c1_tau_opt_random_by_item.csv")


def _c1_group_deltas(raw: pd.DataFrame) -> tuple[list[Any], np.ndarray, dict[Any, np.ndarray], dict[Any, float]]:
    groups = []
    deltas = []
    random_values: dict[Any, np.ndarray] = {}
    optimized_values: dict[Any, float] = {}
    if raw.empty:
        return groups, np.asarray([], dtype=np.float64), random_values, optimized_values
    raw = raw.copy()
    raw["score_numeric"] = pd.to_numeric(raw["eval_score_mspd"], errors="coerce")
    for group, sub in raw.groupby("optimized_run_idx", sort=True):
        opt = sub[sub["candidate_kind"].astype(str) == "optimized"]["score_numeric"].dropna().to_numpy(dtype=np.float64)
        rand = sub[sub["candidate_kind"].astype(str) == "random"]["score_numeric"].dropna().to_numpy(dtype=np.float64)
        if opt.size == 0 or rand.size == 0:
            continue
        opt_val = float(opt[0])
        rand_med = float(np.nanmedian(rand))
        groups.append(group)
        deltas.append(opt_val - rand_med)
        random_values[group] = rand
        optimized_values[group] = opt_val
    return groups, np.asarray(deltas, dtype=np.float64), random_values, optimized_values


def _plot_c1_paired_raw_clean(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    raw = _load_c1_raw_scores(ds_dir)
    groups, deltas, random_values, optimized_values = _c1_group_deltas(raw)
    if not groups:
        return {}
    stats = sign_test_greater(deltas)
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(6.2, 0.62 * len(groups) + 2.2), 4.2), constrained_layout=True)
    for i, group in enumerate(groups):
        rand = random_values[group]
        opt = optimized_values[group]
        jitter = np.linspace(-0.11, 0.11, rand.size) if rand.size > 1 else np.asarray([0.0])
        ax.scatter(i + jitter, rand, s=28, color="#9a9a9a", alpha=0.85, edgecolor="white", linewidth=0.4, zorder=2)
        rand_med = float(np.nanmedian(rand))
        ax.plot([i - 0.18, i + 0.18], [rand_med, rand_med], color="#222222", linewidth=1.4, zorder=3)
        ax.plot([i, i], [rand_med, opt], color="#777777", linewidth=0.9, alpha=0.65, zorder=1)
        ax.scatter(i, opt, s=76, color="#4c78a8", edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_xticks(np.arange(len(groups)), [str(g) for g in groups], rotation=35 if len(groups) > 10 else 0, ha="right" if len(groups) > 10 else "center")
    ax.set_xlabel("matched group id")
    ax.set_ylabel("held-out MSPD score")
    text = (
        f"n_positive={stats['n_positive']}/{stats['n_nonzero']}   "
        f"median_delta={float(stats['median']):.3g}   "
        f"sign_test_p={_format_p(stats['sign_test_greater_p'])}"
    )
    ax.set_title(f"C1 optimized vs matched random controls: {dataset}\n{text}")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = figures / f"c1_{dataset}_paired_raw_clean.png"
    _save(fig, out)
    plt.close(fig)
    return {f"c1_{dataset}_paired_raw_clean": str(out)}


def _symmetric_heatmap_limit(arrays: list[np.ndarray], *, percentile: float = 98.0) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size:
            vals.append(np.abs(finite))
    if not vals:
        return 1.0
    lim = float(np.nanpercentile(np.concatenate(vals), percentile))
    return max(lim, 1e-12)


def _format_tau_ticks(ax: Any, tau_steps: np.ndarray) -> None:
    tau = np.asarray(tau_steps, dtype=np.int64).reshape(-1)
    if tau.size <= 8:
        ax.set_yticks(np.arange(tau.size), [str(int(x)) for x in tau])
    else:
        idx = np.unique(np.linspace(0, tau.size - 1, num=6, dtype=int))
        ax.set_yticks(idx, [str(int(tau[i])) for i in idx])


def _plot_c1_optimized_vs_random_delta_h_heatmaps(
    dataset: str,
    scores: pd.DataFrame,
    figures: Path,
    *,
    max_groups: int = 8,
) -> dict[str, str]:
    stale_outputs = [
        figures / f"c1_{dataset}_delta_h_eval_optimized_vs_random_median.png",
        figures / f"c1_{dataset}_delta_h_eval_optimized_vs_random_grid.png",
    ]
    required = {"optimized_run_idx", "candidate_kind", "maps_path"}
    if scores.empty or not required.issubset(scores.columns):
        for stale in stale_outputs:
            if stale.exists():
                stale.unlink()
        return {}

    pair_records: list[dict[str, Any]] = []
    opt_maps: list[np.ndarray] = []
    random_maps: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None

    for group_idx, group in scores.groupby("optimized_run_idx"):
        opt = group[group["candidate_kind"] == "optimized"]
        randoms = group[group["candidate_kind"] == "random"]
        if opt.empty or randoms.empty:
            continue
        opt_row = opt.iloc[0]
        opt_loaded = _load_c1_eval_map(opt_row)
        if opt_loaded is None:
            continue

        if "eval_score_mspd" in randoms.columns:
            rand_scores = pd.to_numeric(randoms["eval_score_mspd"], errors="coerce")
            med = float(np.nanmedian(rand_scores.to_numpy(dtype=np.float64)))
            if np.isfinite(med):
                pick_idx = (rand_scores - med).abs().sort_values().index[0]
                rand_row = randoms.loc[pick_idx]
            else:
                rand_row = randoms.iloc[0]
        else:
            rand_row = randoms.iloc[0]
        rand_loaded = _load_c1_eval_map(rand_row)
        if rand_loaded is None:
            continue

        opt_map = np.asarray(opt_loaded["delta_h_eval"], dtype=np.float64)
        rand_map = np.asarray(rand_loaded["delta_h_eval"], dtype=np.float64)
        if opt_map.shape != rand_map.shape:
            continue
        if reference_shape is None:
            reference_shape = opt_map.shape
        if opt_map.shape == reference_shape:
            opt_maps.append(opt_map)
            random_maps.append(rand_map)

        pair_records.append(
            {
                "group_idx": int(group_idx),
                "opt_row": opt_row,
                "rand_row": rand_row,
                "opt": opt_loaded,
                "random": rand_loaded,
            }
        )

    paths: dict[str, str] = {}
    if not pair_records:
        for stale in stale_outputs:
            if stale.exists():
                stale.unlink()
        return paths

    plt = _ensure_matplotlib()

    if opt_maps and random_maps:
        opt_med = np.nanmedian(np.stack(opt_maps, axis=0), axis=0)
        rand_med = np.nanmedian(np.stack(random_maps, axis=0), axis=0)
        diff = opt_med - rand_med
        lim = _symmetric_heatmap_limit([opt_med, rand_med])
        diff_lim = _symmetric_heatmap_limit([diff])
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), squeeze=False)
        panels = [
            (opt_med, "optimized median eval Delta-H", "coolwarm", -lim, lim),
            (rand_med, "random median eval Delta-H", "coolwarm", -lim, lim),
            (diff, "optimized - random median", "coolwarm", -diff_lim, diff_lim),
        ]
        for ax, (arr, title, cmap, vmin, vmax) in zip(axes[0], panels):
            im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("window")
            ax.set_ylabel("tau index")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        out = figures / f"c1_{dataset}_delta_h_eval_optimized_vs_random_median.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths[f"c1_{dataset}_delta_h_eval_optimized_vs_random_median"] = str(out)

    selected_pairs = pair_records[: max(1, int(max_groups))]
    arrays = []
    for rec in selected_pairs:
        arrays.append(np.asarray(rec["opt"]["delta_h_eval"], dtype=np.float64))
        arrays.append(np.asarray(rec["random"]["delta_h_eval"], dtype=np.float64))
    lim = _symmetric_heatmap_limit(arrays)
    fig, axes = plt.subplots(
        len(selected_pairs),
        2,
        figsize=(9.2, max(2.8, 2.35 * len(selected_pairs))),
        squeeze=False,
    )
    last_im = None
    for row_idx, rec in enumerate(selected_pairs):
        for col_idx, (label, loaded, row) in enumerate(
            (
                ("optimized", rec["opt"], rec["opt_row"]),
                ("random / non-optimized", rec["random"], rec["rand_row"]),
            )
        ):
            arr = np.asarray(loaded["delta_h_eval"], dtype=np.float64)
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-lim, vmax=lim)
            sel_idx = int(loaded.get("selected_tau_idx", 0))
            if 0 <= sel_idx < arr.shape[0]:
                ax.axhline(sel_idx, color="#111111", linewidth=0.9, linestyle="--")
            score = row.get("eval_score_mspd", np.nan)
            tau_steps = np.asarray(loaded.get("tau_steps", []), dtype=np.int64)
            selected_tau = int(tau_steps[sel_idx]) if tau_steps.size and 0 <= sel_idx < tau_steps.size else -1
            ax.set_title(f"{label}; score={float(score):.3g}; tau={selected_tau}")
            ax.set_xlabel("window")
            ax.set_ylabel(f"group {int(rec['group_idx'])}\ntau")
            _format_tau_ticks(ax, tau_steps)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    fig.suptitle(f"C1 eval Delta-H heatmaps: optimized vs matched random ({dataset})", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    out = figures / f"c1_{dataset}_delta_h_eval_optimized_vs_random_grid.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    paths[f"c1_{dataset}_delta_h_eval_optimized_vs_random_grid"] = str(out)
    return paths


def _plot_c1(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    path = ds_dir / "group_contrasts.csv"
    paired_out = figures / f"c1_{dataset}_paired_contrast.png"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty and "delta_vs_random_median" in df.columns:
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
            fig.savefig(paired_out, dpi=180)
            plt.close(fig)
            paths[f"c1_{dataset}_paired_contrast"] = str(paired_out)
        elif paired_out.exists():
            paired_out.unlink()
    elif paired_out.exists():
        paired_out.unlink()
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


def _preferred_c5_delta_col(df: pd.DataFrame) -> str | None:
    delta_cols = [c for c in df.columns if c.endswith("__delta_vs_random_median")]
    preferred_names = [
        "embedding_cloud_chamfer_cosine__anchor_effect_minus_baseline__delta_vs_random_median",
        "anchor_effect_minus_baseline__delta_vs_random_median",
        "embedding_synced_cosine__anchor_effect_minus_baseline__delta_vs_random_median",
    ]
    for name in preferred_names:
        if name in delta_cols:
            return name
    preferred = [c for c in delta_cols if "embedding" in c and "minus_baseline" in c]
    return preferred[0] if preferred else (delta_cols[0] if delta_cols else None)


def _preferred_c5_metric_name(metrics: list[str]) -> str | None:
    preferred = [
        "embedding_cloud_chamfer_cosine__anchor_effect_minus_baseline",
        "anchor_effect_minus_baseline",
        "embedding_synced_cosine__anchor_effect_minus_baseline",
    ]
    for name in preferred:
        if name in metrics:
            return name
    emb = [m for m in metrics if "embedding" in m and "minus_baseline" in m]
    return emb[0] if emb else (metrics[0] if metrics else None)


def _parse_random_values(value: Any) -> np.ndarray:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.asarray([], dtype=np.float64)
    parts = str(value).replace(",", ";").split(";")
    vals = []
    for part in parts:
        try:
            v = float(part)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def _c5_raw_delta_frame(ds_dir: Path) -> pd.DataFrame:
    run_path = ds_dir / "frustration_run_level.csv"
    if run_path.exists() and run_path.stat().st_size > 1:
        run_df = pd.read_csv(run_path)
        col = _preferred_c5_delta_col(run_df)
        if col is not None:
            rand_col = col.replace("__delta_vs_random_median", "__random_median")
            metric = col.replace("__delta_vs_random_median", "")
            out = pd.DataFrame(
                {
                    "group": run_df["optimized_run_idx"].to_numpy() if "optimized_run_idx" in run_df.columns else np.arange(len(run_df)),
                    "delta": pd.to_numeric(run_df[col], errors="coerce"),
                    "random_median": pd.to_numeric(run_df[rand_col], errors="coerce") if rand_col in run_df.columns else np.nan,
                    "metric": metric,
                }
            )
            return out.dropna(subset=["delta"])

    point_path = ds_dir / "frustration_pointwise_opt_gt_random_by_group.csv"
    if not point_path.exists() or point_path.stat().st_size <= 1:
        return pd.DataFrame()
    point_df = pd.read_csv(point_path)
    if point_df.empty or not {"metric", "optimized_run_idx", "opt_value", "random_values"}.issubset(point_df.columns):
        return pd.DataFrame()
    metric = _preferred_c5_metric_name([str(x) for x in point_df["metric"].dropna().unique().tolist()])
    if metric is None:
        return pd.DataFrame()
    rows = []
    sub = point_df[point_df["metric"].astype(str) == metric]
    for rec in sub.itertuples():
        rand = _parse_random_values(getattr(rec, "random_values"))
        try:
            opt = float(getattr(rec, "opt_value"))
        except Exception:
            continue
        if rand.size == 0 or not np.isfinite(opt):
            continue
        rows.append(
            {
                "group": getattr(rec, "optimized_run_idx"),
                "delta": opt - float(np.nanmedian(rand)),
                "random_median": float(np.nanmedian(rand)),
                "random_values": rand,
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def _plot_c5_frustration_clean(dataset: str, ds_dir: Path, figures: Path) -> dict[str, str]:
    df = _c5_raw_delta_frame(ds_dir)
    if df.empty or "delta" not in df.columns:
        return {}
    y = pd.to_numeric(df["delta"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(y)
    if not np.any(finite):
        return {}
    groups = df["group"].to_numpy() if "group" in df.columns else np.arange(len(df))
    stats = sign_test_greater(y[finite])
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(6.2, 0.62 * len(y) + 2.2), 4.0), constrained_layout=True)
    x = np.arange(len(y), dtype=np.float64)
    colors = np.where(y >= 0, "#2ca02c", "#d62728")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.bar(x[finite], y[finite], color=colors[finite], alpha=0.82, width=0.62)
    ax.scatter(x[finite], y[finite], color=colors[finite], edgecolor="white", linewidth=0.6, s=44, zorder=3)
    ax.set_xticks(x, [str(g) for g in groups], rotation=35 if len(groups) > 10 else 0, ha="right" if len(groups) > 10 else "center")
    ax.set_xlabel("matched group id")
    ax.set_ylabel("F(opt) - median F(random)")
    metric_label = str(df["metric"].iloc[0]) if "metric" in df.columns and not df.empty else "frustration"
    text = (
        f"n_positive={stats['n_positive']}/{stats['n_nonzero']}   "
        f"median_delta={float(stats['median']):.3g}   "
        f"sign_test_p={_format_p(stats['sign_test_greater_p'])}"
    )
    ax.set_title(f"C5 frustration contrast: {dataset}\n{text}")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, -0.22, f"metric: {metric_label}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555555")
    out = figures / f"c5_{dataset}_frustration_clean.png"
    _save(fig, out)
    plt.close(fig)
    return {f"c5_{dataset}_frustration_clean": str(out)}


def _bootstrap_median_ci(values: np.ndarray, *, n_boot: int = 2000) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    rng = np.random.default_rng(12345)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = float(np.nanmedian(rng.choice(arr, size=arr.size, replace=True)))
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _claim_deltas_for_cross(output_root: Path, dataset: str, kind: str) -> tuple[np.ndarray, float, str] | None:
    ds_dir = output_root / dataset
    if kind == "c1":
        raw = _load_c1_raw_scores(ds_dir)
        _groups, deltas, random_values, _optimized = _c1_group_deltas(raw)
        if deltas.size == 0:
            return None
        rand_all = np.concatenate(list(random_values.values())) if random_values else deltas
        return deltas, _mad_or_eps(rand_all), "held-out MSPD"
    df = _c5_raw_delta_frame(ds_dir)
    if df.empty:
        return None
    deltas = pd.to_numeric(df["delta"], errors="coerce").to_numpy(dtype=np.float64)
    if "random_values" in df.columns:
        random_chunks = [np.asarray(v, dtype=np.float64) for v in df["random_values"] if isinstance(v, np.ndarray) and v.size]
        denom_source = np.concatenate(random_chunks) if random_chunks else pd.to_numeric(df.get("random_median", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    else:
        denom_source = pd.to_numeric(df.get("random_median", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    metric = str(df["metric"].iloc[0]) if "metric" in df.columns and not df.empty else "frustration"
    return deltas[np.isfinite(deltas)], _mad_or_eps(denom_source), metric


def _plot_cross(output_root: Path, figures: Path) -> dict[str, str]:
    rows = [
        ("flow_lenia", "Flow-Lenia C1 MSPD contrast", "c1"),
        ("flow_lenia", "Flow-Lenia C5 frustration contrast", "c5"),
        ("plife_plus", "PLife++ C6.1 MSPD contrast", "c1"),
        ("plife_plus", "PLife++ C6.5 frustration contrast", "c5"),
    ]
    items = []
    for dataset, label, kind in rows:
        loaded = _claim_deltas_for_cross(output_root, dataset, kind)
        if loaded is None:
            continue
        deltas, denom, metric = loaded
        z = np.asarray(deltas, dtype=np.float64) / float(denom)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue
        items.append((label, z, sign_test_greater(deltas), metric, denom))
    if not items:
        return {}
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.0, max(3.8, 0.82 * len(items) + 1.4)), constrained_layout=True)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    y_positions = np.arange(len(items), dtype=np.float64)
    for y, (label, z, stats, metric, denom) in zip(y_positions, items, strict=True):
        jitter = np.linspace(-0.09, 0.09, z.size) if z.size > 1 else np.asarray([0.0])
        ax.scatter(z, y + jitter, color="#4c78a8", alpha=0.72, s=36, edgecolor="white", linewidth=0.5)
        med = float(np.nanmedian(z))
        ci = _bootstrap_median_ci(z)
        if ci is not None:
            ax.plot([ci[0], ci[1]], [y, y], color="#111111", linewidth=2.0)
        ax.scatter(med, y, s=90, color="#111111", edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(
            0.99,
            y,
            f"n+={stats['n_positive']}/{stats['n_nonzero']}, p={_format_p(stats['sign_test_greater_p'])}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_yticks(y_positions, [item[0] for item in items])
    ax.invert_yaxis()
    ax.set_xlabel("normalized matched delta, delta / (MAD random + eps)")
    ax.set_title("Cross-substrate normalized effect-size summary")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = figures / "cross_substrate_effects_clean.png"
    _save(fig, out)
    alias = figures / "c6_cross_substrate_effects.png"
    _save(fig, alias)
    plt.close(fig)
    return {"cross_substrate_effects_clean": str(out), "c6_cross_substrate_effects": str(alias)}


def _plot_c2_branching_one(output_root: Path, figures: Path, *, suffix: str, label: str) -> dict[str, str]:
    scores_path = output_root / "c2_branching" / f"branching_scores{suffix}.csv"
    contrasts_path = output_root / "c2_branching" / f"branching_pair_contrasts{suffix}.csv"
    if not scores_path.exists():
        return {}
    scores = pd.read_csv(scores_path)
    contrasts = pd.read_csv(contrasts_path) if contrasts_path.exists() else pd.DataFrame()
    if scores.empty:
        return {}
    plt = _ensure_matplotlib()
    if contrasts.empty:
        fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        ax0, ax1 = axes
        for _idx, row in enumerate(contrasts.itertuples()):
            low = float(row.low_branching_score)
            high = float(row.high_branching_score)
            ax0.plot([0, 1], [low, high], color="#888888", alpha=0.7, linewidth=1)
            ax0.scatter([0, 1], [low, high], color=["#4c78a8", "#d62728"], s=28, zorder=3)
        diffs = contrasts["delta_branching_score"].astype(float).to_numpy()
        ax0.set_title(f"{label}; median delta={np.nanmedian(diffs):.3g}" if diffs.size else label)
        ax0.set_xticks([0, 1], ["low", "high"])
        ax0.set_ylabel("branch divergence")

    conditions = scores["condition"].astype(str).to_numpy() if "condition" in scores.columns else np.asarray(["sampled"] * len(scores))
    palette = {"high": "#d62728", "mid": "#ff7f0e", "low": "#4c78a8", "sampled": "#6f4e9b"}
    colors = np.asarray([palette.get(str(cond), "#6f4e9b") for cond in conditions])
    x = scores["delta_h"].astype(float).to_numpy()
    y = scores["branching_score"].astype(float).to_numpy()
    ax1.scatter(x, y, c=colors, s=36)
    finite = np.isfinite(x) & np.isfinite(y)
    title = f"Delta-H vs future divergence\n{label}"
    corr_text = ""
    if int(np.sum(finite)) >= 2 and float(np.std(x[finite])) > 1e-12 and float(np.std(y[finite])) > 1e-12:
        r = float(np.corrcoef(x[finite], y[finite])[0, 1])
        rx = pd.Series(x[finite]).rank(method="average").to_numpy(dtype=float)
        ry = pd.Series(y[finite]).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(rx, ry)[0, 1]) if float(np.std(rx)) > 1e-12 and float(np.std(ry)) > 1e-12 else float("nan")
        corr_text = f"Pearson r = {r:.3g}\nSpearman rho = {spearman:.3g}\nn = {int(np.sum(finite))}"
        coef = np.polyfit(x[finite], y[finite], deg=1)
        xline = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 100)
        ax1.plot(xline, coef[0] * xline + coef[1], color="#333333", linewidth=1, alpha=0.8)
    ax1.set_xlabel("Delta-H at branch time")
    ax1.set_ylabel("branch divergence")
    ax1.set_title(title)
    if corr_text:
        ax1.text(
            0.03,
            0.97,
            corr_text,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
    fig.tight_layout()
    tag = suffix.lstrip("_")
    name_suffix = f"_{tag}" if tag else ""
    out = figures / f"c2_branching_sensitivity{name_suffix}.png"
    fig.savefig(out, dpi=180)
    out_alias = figures / f"c2_high_vs_low_branching_divergence{name_suffix}.png"
    fig.savefig(out_alias, dpi=180)
    out_corr = figures / f"c2_delta_h_branching_correlation{name_suffix}.png"
    fig.savefig(out_corr, dpi=180)
    plt.close(fig)
    key_suffix = f"_{tag}" if tag else ""
    return {
        f"c2_branching_sensitivity{key_suffix}": str(out),
        f"c2_high_vs_low_branching_divergence{key_suffix}": str(out_alias),
        f"c2_delta_h_branching_correlation{key_suffix}": str(out_corr),
    }


def _plot_c2_branching(output_root: Path, figures: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    paths.update(_plot_c2_branching_one(output_root, figures, suffix="", label="APF multiscale L2"))
    paths.update(_plot_c2_branching_one(output_root, figures, suffix="_clip_chamfer", label="CLIP Chamfer"))
    return paths


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if rx.size < 2 or float(np.nanstd(rx)) <= 1e-12 or float(np.nanstd(ry)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_regression_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, *, n_boot: int = 1000) -> tuple[np.ndarray, np.ndarray] | None:
    if x.size < 3:
        return None
    rng = np.random.default_rng(54321)
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        xb = x[idx]
        yb = y[idx]
        if float(np.nanstd(xb)) <= 1e-12:
            continue
        coef = np.polyfit(xb, yb, deg=1)
        preds.append(coef[0] * x_grid + coef[1])
    if not preds:
        return None
    arr = np.vstack(preds)
    return np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def _plot_c2_clip_chamfer_association_clean(output_root: Path, figures: Path) -> dict[str, str]:
    scores_path = output_root / "c2_branching" / "branching_scores_clip_chamfer.csv"
    if not scores_path.exists() or scores_path.stat().st_size <= 1:
        return {}
    scores = pd.read_csv(scores_path)
    if scores.empty or not {"delta_h", "branching_score"}.issubset(scores.columns):
        return {}
    x = pd.to_numeric(scores["delta_h"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(scores["branching_score"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    sub = scores.loc[finite].copy()
    if x.size < 2:
        return {}
    if "condition" in sub.columns:
        strata = sub["condition"].astype(str).to_numpy()
    else:
        q = pd.qcut(pd.Series(x), q=min(3, np.unique(x).size), labels=False, duplicates="drop")
        labels = np.asarray(["low", "mid", "high"])
        strata = labels[np.asarray(q.fillna(0), dtype=int).clip(0, labels.size - 1)]
    palette = {"low": "#4c78a8", "mid": "#f58518", "high": "#d62728"}
    colors = [palette.get(str(s), "#6f4e9b") for s in strata]
    pearson = float(np.corrcoef(x, y)[0, 1]) if float(np.nanstd(x)) > 1e-12 and float(np.nanstd(y)) > 1e-12 else float("nan")
    spearman = _rank_correlation(x, y)

    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for stratum in ["low", "mid", "high"]:
        mask = np.asarray([str(s) == stratum for s in strata])
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=48, color=palette[stratum], alpha=0.86, edgecolor="white", linewidth=0.6, label=stratum)
    other = np.asarray([str(s) not in palette for s in strata])
    if np.any(other):
        ax.scatter(x[other], y[other], s=48, color="#6f4e9b", alpha=0.86, edgecolor="white", linewidth=0.6, label="sampled")
    if x.size >= 2 and float(np.nanstd(x)) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 160)
        y_hat = coef[0] * x_grid + coef[1]
        band = _bootstrap_regression_band(x, y, x_grid)
        if band is not None:
            ax.fill_between(x_grid, band[0], band[1], color="#333333", alpha=0.14, linewidth=0)
        ax.plot(x_grid, y_hat, color="#333333", linewidth=1.5)
    ax.set_xlabel("Delta-H at branch time")
    ax.set_ylabel("future CLIP-Chamfer branch divergence")
    ax.set_title(f"C2 Delta-H / CLIP-Chamfer association\nPearson r={pearson:.3g}, Spearman r={spearman:.3g}, n={x.size}")
    ax.text(
        0.02,
        0.02,
        "stratified branch sample; correlation is descriptive across sampled branch states.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 3},
    )
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Delta-H stratum", frameon=False, loc="best")
    out = figures / "c2_clip_chamfer_association_clean.png"
    _save(fig, out)
    plt.close(fig)
    return {"c2_clip_chamfer_association_clean": str(out)}


def _plot_c2_plife_plus_branching(output_root: Path, figures: Path) -> dict[str, str]:
    scores_path = output_root / "c2_plife_plus_branching" / "branching_scores.csv"
    corr_path = output_root / "c2_plife_plus_branching" / "branching_delta_h_correlation.csv"
    if not scores_path.exists():
        _record_skip("c2_plife_plus_branching", f"missing scores table {scores_path}")
        return {}
    try:
        scores = pd.read_csv(scores_path)
    except pd.errors.EmptyDataError:
        _record_skip("c2_plife_plus_branching", f"empty scores table {scores_path}")
        return {}
    if scores.empty or not {"delta_h", "branching_score"}.issubset(scores.columns):
        _record_skip("c2_plife_plus_branching", f"no score rows or missing columns in {scores_path}")
        return {}
    x = pd.to_numeric(scores["delta_h"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(scores["branching_score"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        _record_skip("c2_plife_plus_branching", f"no finite delta_h/branching_score pairs in {scores_path}")
        return {}
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        _record_skip("c2_plife_plus_branching", f"fewer than two finite score rows in {scores_path}")
        return {}
    cond = scores.loc[finite, "condition"].astype(str).to_numpy() if "condition" in scores.columns else np.asarray(["sampled"] * x.size)
    palette = {"low": "#4c78a8", "mid": "#f58518", "high": "#d62728", "sampled": "#6f4e9b"}
    pearson = float("nan")
    spearman = float("nan")
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        if not corr.empty:
            pearson = float(pd.to_numeric(corr.get("pearson_r", pd.Series([np.nan])).iloc[0], errors="coerce"))
            spearman = float(pd.to_numeric(corr.get("spearman_r", pd.Series([np.nan])).iloc[0], errors="coerce"))
    if not np.isfinite(pearson) and x.size >= 2 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = _rank_correlation(x, y)

    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for stratum in ["low", "mid", "high"]:
        mask = np.asarray([str(c) == stratum for c in cond])
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=48, color=palette[stratum], alpha=0.86, edgecolor="white", linewidth=0.6, label=stratum)
    other = np.asarray([str(c) not in {"low", "mid", "high"} for c in cond])
    if np.any(other):
        ax.scatter(x[other], y[other], s=48, color=palette["sampled"], alpha=0.86, edgecolor="white", linewidth=0.6, label="sampled")
    if x.size >= 2 and float(np.nanstd(x)) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 160)
        band = _bootstrap_regression_band(x, y, xs)
        if band is not None:
            ax.fill_between(xs, band[0], band[1], color="#333333", alpha=0.14, linewidth=0)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1.5)
    ax.set_xlabel("Delta-H at branch time")
    ax.set_ylabel("future CLIP-Chamfer branch divergence")
    ax.set_title(f"C2 PLife++ Delta-H / CLIP-Chamfer association\nPearson r={pearson:.3g}, Spearman r={spearman:.3g}, n={x.size}")
    ax.text(
        0.02,
        0.02,
        "stratified branch sample; correlation is descriptive across sampled PLife++ branch states.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 3},
    )
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Delta-H stratum", frameon=False, loc="best")
    out = figures / "c2_plife_plus_branching_sensitivity.png"
    _save(fig, out)
    out_corr = figures / "c2_plife_plus_delta_h_branching_correlation.png"
    _save(fig, out_corr)
    out_clean = figures / "c2_plife_plus_clip_chamfer_association_clean.png"
    _save(fig, out_clean)
    plt.close(fig)
    log_event(f"C2 PLife++ plots wrote {out}, {out_corr}, and {out_clean}", component="visualization")
    return {
        "c2_plife_plus_branching_sensitivity": str(out),
        "c2_plife_plus_delta_h_branching_correlation": str(out_corr),
        "c2_plife_plus_clip_chamfer_association_clean": str(out_clean),
    }


def run(config_path: str | Path, *, task: str = "all", smoke: bool = False, force: bool = False) -> dict[str, Any]:
    _VISUALIZATION_SKIPS.clear()
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    figures = ensure_dir(output_root / "figures")
    paths: dict[str, str] = {}
    log_event(f"visualization start task={task} smoke={smoke} force={force} figures={figures}", component="visualization")
    if task in {"all", "synthetic"}:
        paths.update(_plot_synthetic(output_root, figures))
        paths.update(_plot_synthetic_summary_clean(output_root, figures))
        paths.update(_plot_synthetic_delta_h_heatmaps_clean(output_root, figures))
        synthetic_result = visualize_synthetic(config_path, smoke=smoke, force=force)
        paths.update({str(k): str(v) for k, v in synthetic_result.get("figure_paths", {}).items()})
    if task in {"all", "c2"}:
        paths.update(_plot_c2_branching(output_root, figures))
        paths.update(_plot_c2_clip_chamfer_association_clean(output_root, figures))
        paths.update(_plot_c2_plife_plus_branching(output_root, figures))
    if task in {"all", "c1", "c5", "c6"}:
        for dataset, _ds in dataset_items(cfg):
            ds_dir = output_root / dataset
            if task in {"all", "c1", "c6"}:
                paths.update(_plot_c1_paired_raw_clean(dataset, ds_dir, figures))
                paths.update(_plot_c1(dataset, ds_dir, figures))
            if task in {"all", "c5", "c6"}:
                paths.update(_plot_c5_frustration_clean(dataset, ds_dir, figures))
                paths.update(_plot_c5(dataset, ds_dir, figures))
        paths.update(_plot_cross(output_root, figures))
    summary = {"figure_paths": paths, "skipped_plots": list(_VISUALIZATION_SKIPS)}
    write_json(output_root / "visualization_summary.json", summary)
    log_event(
        f"visualization done n_figures={len(paths)} n_skipped={len(_VISUALIZATION_SKIPS)} "
        f"summary={output_root / 'visualization_summary.json'}",
        component="visualization",
    )
    return {"n_figures": len(paths), **summary}


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
