from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from paper_suite_c2_branching import (
    _branch_cfg,
    _get,
    _iter_metric_items,
    _load_delta_h_energy,
    _nearest_apf_step,
    _npz_json,
    _npz_scalar,
    _preprocess_delta_h_heatmap,
    _safe_arr,
    _select_ranked_high_low_points,
    _trajectory_end_step,
    _trajectory_root,
)
from paper_suite_common import ensure_dir, load_config, write_csv


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper_suite_matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _load_heatmap_for_plot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        dh_map = np.asarray(_safe_arr(data, "delta_h_map"), dtype=np.float64)
        tau_steps = np.asarray(_safe_arr(data, "delta_h_tau_steps", np.arange(dh_map.shape[0])), dtype=np.int64).reshape(-1)
        if dh_map.ndim != 2:
            raise ValueError(f"{path} delta_h_map must be 2D, got {dh_map.shape}.")
        if dh_map.shape[0] != tau_steps.size and dh_map.shape[1] == tau_steps.size:
            dh_map = dh_map.T
        if dh_map.shape[0] != tau_steps.size:
            raise ValueError(f"{path} delta_h_map shape {dh_map.shape} is incompatible with tau grid size {tau_steps.size}.")

        centers = _safe_arr(data, "delta_h_window_center_steps")
        if centers is None:
            starts = _safe_arr(data, "delta_h_window_start_steps", np.arange(dh_map.shape[1]))
            centers = np.asarray(starts, dtype=np.float64).reshape(-1)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1)

        metric_cfg = _npz_json(data, "metric_config_json")
        sample_every = int(
            _npz_scalar(
                data,
                "delta_h_sample_every_steps",
                metric_cfg.get("sample_every_steps", metric_cfg.get("sample_stride_steps", 1)),
            )
        )
        window_size_steps = _npz_scalar(data, "delta_h_window_size_steps", None)
        if window_size_steps is None:
            window_size_steps = int(metric_cfg.get("window_size_frames", 0)) * max(1, sample_every)
        window_size_steps = int(window_size_steps)
        m_min = int(metric_cfg.get("m_min", 4))
        min_gap_steps = int(m_min) * max(1, sample_every)
        admissible = np.isfinite(tau_steps) & (tau_steps < window_size_steps) & ((window_size_steps - tau_steps) >= min_gap_steps)
        mode = str(metric_cfg.get("preprocess_mode", "clip")).strip().lower()
        floor = float(metric_cfg.get("delta_h_floor", 0.0) or 0.0)
        processed = _preprocess_delta_h_heatmap(dh_map, mode=mode, floor=floor)
    n = min(int(centers.size), int(processed.shape[1]))
    return centers[:n], tau_steps, processed[:, :n], [int(i) for i in np.flatnonzero(admissible)], {
        "preprocess_mode": mode,
        "delta_h_floor": floor,
        "window_size_steps": window_size_steps,
        "min_gap_steps": min_gap_steps,
    }


def _fallback_trajectory_end(metrics_path: Path, centers: np.ndarray, horizon_steps: int) -> int:
    try:
        with np.load(metrics_path, allow_pickle=False) as data:
            ends = _safe_arr(data, "delta_h_window_end_steps")
            if ends is not None and np.asarray(ends).size:
                return int(np.nanmax(np.asarray(ends, dtype=np.float64)))
    except Exception:
        pass
    if centers.size:
        return int(np.nanmax(centers) + int(horizon_steps))
    return int(horizon_steps)


def _snapped_step(apf_dir: Path, requested_step: int) -> int:
    try:
        return _nearest_apf_step(apf_dir, requested_step)
    except Exception:
        return int(requested_step)


def _selected_for_item(
    *,
    item: dict[str, Any],
    traj_order: int,
    n_high: int,
    n_low: int,
    q_high: float,
    q_low: float,
    horizon_steps: int,
    selection_seed: int,
    energy_min_remaining_steps: int | None,
    energy_min_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[dict[str, Any]], dict[str, Any]]:
    metrics_path = Path(item["metrics_path"])
    traj_dir = Path(item["traj_dir"])
    apf_dir = Path(item.get("apf_dir", traj_dir / "apf_logs"))
    centers, tau_steps, heatmap, admissible_tau_idx, _plot_meta = _load_heatmap_for_plot(metrics_path)
    energy_centers, energy, energy_meta = _load_delta_h_energy(
        metrics_path,
        min_remaining_steps=energy_min_remaining_steps,
        min_remaining_samples=energy_min_samples,
    )
    try:
        trajectory_end = _trajectory_end_step(apf_dir)
    except Exception:
        trajectory_end = None
    if trajectory_end is None:
        trajectory_end = _fallback_trajectory_end(metrics_path, energy_centers, horizon_steps)
    points, summary = _select_ranked_high_low_points(
        centers=energy_centers,
        energy=energy,
        covariates=None,
        n_high=n_high,
        n_low=n_low,
        q_high=q_high,
        q_low=q_low,
        horizon_steps=horizon_steps,
        trajectory_end_step=trajectory_end,
        seed=selection_seed + 10007 * traj_order,
        energy_meta=energy_meta,
    )
    for row in points:
        requested = int(row["step"])
        row["snapped_step"] = _snapped_step(apf_dir, requested)
        row["step_snap_delta"] = int(row["snapped_step"]) - requested
    return centers, tau_steps, heatmap, admissible_tau_idx, points, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config, smoke=args.smoke)
    bcfg = _branch_cfg(cfg)
    c2_cfg = cfg.get("c2", {})
    trajectory_root = _trajectory_root(c2_cfg)
    if trajectory_root is None:
        raise ValueError("Could not resolve C2 trajectory root.")
    items = _iter_metric_items(trajectory_root)
    if not items:
        raise ValueError(f"No C2 metrics.npz items found under {trajectory_root}.")

    n_high = int(args.n_high if args.n_high is not None else _get(bcfg, "n_high", _get(bcfg, "m_pairs", 2)))
    n_low = int(args.n_low if args.n_low is not None else _get(bcfg, "n_low", _get(bcfg, "m_pairs", 2)))
    q_high = float(args.q_high if args.q_high is not None else _get(bcfg, "high_quantile", 0.8))
    q_low = float(args.q_low if args.q_low is not None else _get(bcfg, "low_quantile", 0.2))
    horizon_steps = int(args.horizon_steps if args.horizon_steps is not None else _get(bcfg, "horizon_steps", 1000))
    selection_seed = int(args.selection_seed if args.selection_seed is not None else _get(bcfg, "selection_seed", 12345))
    energy_min_remaining_steps = _get(bcfg, "energy_min_remaining_steps", None)
    energy_min_samples = _get(bcfg, "energy_min_samples", None)
    energy_min_remaining_steps = None if energy_min_remaining_steps is None else int(energy_min_remaining_steps)
    energy_min_samples = None if energy_min_samples is None else int(energy_min_samples)

    ranked: list[tuple[float, dict[str, Any], np.ndarray]] = []
    for item in items:
        centers, energy, _meta = _load_delta_h_energy(
            Path(item["metrics_path"]),
            min_remaining_steps=energy_min_remaining_steps,
            min_remaining_samples=energy_min_samples,
        )
        ranked.append((float(np.nanmean(energy)), item, centers))
    ranked.sort(key=lambda x: -x[0])
    max_trajectories = len(ranked) if args.max_trajectories == "all" else int(args.max_trajectories or _get(bcfg, "max_trajectories", 2))
    selected_items = ranked[:max_trajectories]

    out_path = Path(args.output) if args.output else Path("analysis/results/paper_suite/c2_branching/branch_selection_preview.png")
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    ensure_dir(out_path.parent)
    csv_path = out_path.with_suffix(".csv")

    plt = _ensure_matplotlib()
    n_panels = max(1, len(selected_items))
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig_w = max(6.0, 5.6 * n_cols)
    fig_h = max(4.6, 4.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)

    all_rows: list[dict[str, Any]] = []
    last_im = None
    for panel_idx, (_rank_score, item, _centers_for_rank) in enumerate(selected_items):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        centers, tau_steps, heatmap, admissible_tau_idx, points, summary = _selected_for_item(
            item=item,
            traj_order=panel_idx,
            n_high=n_high,
            n_low=n_low,
            q_high=q_high,
            q_low=q_low,
            horizon_steps=horizon_steps,
            selection_seed=selection_seed,
            energy_min_remaining_steps=energy_min_remaining_steps,
            energy_min_samples=energy_min_samples,
        )
        if heatmap.size:
            vmax = float(np.nanpercentile(heatmap[np.isfinite(heatmap)], 98.0)) if np.isfinite(heatmap).any() else 1.0
            vmax = max(vmax, 1e-12)
            last_im = ax.imshow(
                heatmap,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
            )
        for idx in range(len(tau_steps)):
            if idx not in admissible_tau_idx:
                ax.axhspan(idx - 0.5, idx + 0.5, color="white", alpha=0.35, linewidth=0)

        n_windows = max(1, heatmap.shape[1])
        step_to_x = {int(round(float(step))): i for i, step in enumerate(centers)}
        for row in points:
            window_idx = int(row.get("window_index", step_to_x.get(int(row["window_center_step"]), 0)))
            color = "#d62728" if row["condition"] == "high" else "#1f77b4"
            marker = "v" if row["condition"] == "high" else "^"
            ax.axvline(window_idx, color=color, linewidth=1.5, alpha=0.9)
            ax.scatter(
                [window_idx],
                [len(tau_steps) - 0.7 if row["condition"] == "high" else 0.7],
                c=color,
                marker=marker,
                s=44,
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )
            all_rows.append(
                {
                    "traj_id": str(item["traj_id"]),
                    "condition": row["condition"],
                    "point_id": int(row["point_id"]),
                    "window_index": int(row["window_index"]),
                    "window_center_step": int(row["window_center_step"]),
                    "snapped_step": int(row["snapped_step"]),
                    "step_snap_delta": int(row["step_snap_delta"]),
                    "delta_h_energy": float(row["delta_h_energy"]),
                    "delta_h_quantile_rank": float(row["delta_h_quantile_rank"]),
                    "metrics_path": str(item["metrics_path"]),
                }
            )

        if centers.size:
            ticks = np.linspace(0, n_windows - 1, min(5, n_windows)).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(round(float(centers[i])))) for i in ticks], rotation=30, ha="right")
        y_ticks = np.linspace(0, max(0, len(tau_steps) - 1), min(6, len(tau_steps))).astype(int) if len(tau_steps) else []
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(tau_steps[i])) for i in y_ticks])
        ax.set_title(
            f"{item['traj_id']}\nH {summary.get('n_high_selected', 0)}/{summary.get('n_high_pool', 0)}; "
            f"L {summary.get('n_low_selected', 0)}/{summary.get('n_low_pool', 0)}",
            fontsize=9,
        )
        ax.set_xlabel("window center step")
        ax.set_ylabel("tau steps")

    for panel_idx in range(len(selected_items), n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].axis("off")

    fig.suptitle(
        f"C2 branching start preview\nmean_tau phi(Delta-H), high q>={q_high:g}, low q<={q_low:g}, "
        f"N_high={n_high}, N_low={n_low}",
        fontsize=12,
    )
    if last_im is not None:
        used_axes = [axes[i // n_cols][i % n_cols] for i in range(len(selected_items))]
        fig.colorbar(last_im, ax=used_axes, shrink=0.82, pad=0.02, label="phi(Delta-H)")
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    write_csv(csv_path, all_rows)
    return {"figure": str(out_path), "csv": str(csv_path), "n_points": len(all_rows), "n_trajectories": len(selected_items)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview C2 branching high/low start windows as Delta-H heatmaps.")
    parser.add_argument("config")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-trajectories", default=None, help="Integer or 'all'. Defaults to c2.branching.max_trajectories.")
    parser.add_argument("--n-high", type=int, default=None)
    parser.add_argument("--n-low", type=int, default=None)
    parser.add_argument("--q-high", type=float, default=None)
    parser.add_argument("--q-low", type=float, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
