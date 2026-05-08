from __future__ import annotations

import argparse
import csv
import json
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
from omegaconf import OmegaConf

from flowlenia_minibang_common import load_config, resolve_path


DELTA_H_CONFIG_KEYS = [
    "metric_tau_mode",
    "minibang_metric_tau_mode",
    "metric_tau_grid_steps",
    "metric_tau_grid_frames",
    "metric_tau_steps",
    "metric_window_size_steps",
    "metric_window_step_steps",
    "metric_range_start_steps",
    "metric_range_end_steps",
    "metric_m_samples",
    "metric_m_min",
    "metric_n_proj",
    "metric_null_reps",
    "metric_particle_samples",
    "metric_preprocess_mode",
    "metric_alpha",
    "metric_beta",
    "metric_eps",
    "metric_dirs_seed",
]


def _load_manifest_rows(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("trajectories", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid manifest format: {manifest_path}")
        return manifest, rows

    rows = []
    for traj_dir in sorted(dataset_root.glob("traj_*")):
        if traj_dir.is_dir():
            rows.append(
                {
                    "traj_id": traj_dir.name,
                    "traj_dir": str(traj_dir),
                    "metrics_path": str(traj_dir / "metrics.npz"),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No manifest.json or traj_* dirs found in {dataset_root}")
    return {}, rows


def _candidate_paths(*paths: Any) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if raw is None or raw == "":
            continue
        path = Path(str(raw))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _traj_id(row: dict[str, Any]) -> str:
    if row.get("traj_id", None):
        return str(row["traj_id"])
    if row.get("traj_dir", None):
        return Path(str(row["traj_dir"])).name
    return f"traj_{int(row.get('selection_idx', 0)):05d}"


def _local_traj_dir(dataset_root: Path, row: dict[str, Any]) -> Path:
    traj_id = _traj_id(row)
    local = dataset_root / traj_id
    if local.exists():
        return local
    raw = row.get("traj_dir", None)
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
        mapped = dataset_root / path.name
        if mapped.exists():
            return mapped
    return local


def _metrics_path(dataset_root: Path, row: dict[str, Any]) -> tuple[Path | None, Path]:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        row.get("metrics_path", None),
        traj_dir / "metrics.npz",
        Path(str(row["traj_dir"])) / "metrics.npz" if row.get("traj_dir", None) else None,
    )
    existing = _first_existing(candidates)
    preferred = traj_dir / "metrics.npz"
    return existing, preferred


def _apf_dir(dataset_root: Path, row: dict[str, Any]) -> Path | None:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        row.get("apf_dir", None),
        traj_dir / "apf_logs",
        Path(str(row["traj_dir"])) / "apf_logs" if row.get("traj_dir", None) else None,
    )
    return _first_existing(candidates)


def _config_path(dataset_root: Path, row: dict[str, Any], manifest: dict[str, Any]) -> Path | None:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        traj_dir / "config.yaml",
        Path(str(row["traj_dir"])) / "config.yaml" if row.get("traj_dir", None) else None,
        manifest.get("config_path", None),
    )
    return _first_existing(candidates)


def _npz_has_keys(path: Path, keys: tuple[str, ...]) -> bool:
    try:
        with np.load(path) as data:
            return all(key in data.files for key in keys)
    except Exception:
        return False


def _compute_metrics_if_needed(
    *,
    dataset_root: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    metrics_path: Path | None,
    preferred_metrics_path: Path,
    metrics_seed: int,
    overwrite_delta_h: bool,
    overwrite_cluster_mass: bool,
    need_delta_h: bool,
    need_cluster_mass: bool,
    metric_overrides: dict[str, Any],
) -> Path | None:
    has_delta_h = metrics_path is not None and _npz_has_keys(
        metrics_path,
        ("delta_h_map", "delta_h_tau_steps"),
    )
    has_cluster_mass = metrics_path is not None and _npz_has_keys(metrics_path, ("cluster_steps", "cluster_mass_prob"))
    compute_delta_h = need_delta_h and (overwrite_delta_h or not has_delta_h)
    compute_cluster_mass = need_cluster_mass and (overwrite_cluster_mass or not has_cluster_mass)
    if not compute_delta_h and not compute_cluster_mass:
        return metrics_path

    apf_dir = _apf_dir(dataset_root, row)
    config_path = _config_path(dataset_root, row, manifest)
    if apf_dir is None or config_path is None:
        return metrics_path

    from flowlenia_minibang_simulate import _compute_cluster_metrics, _compute_delta_h_metrics

    _cfg, flat = load_config(config_path)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    if not isinstance(flat_args, dict):
        raise ValueError(f"Could not flatten config: {config_path}")
    flat_args.update(metric_overrides)

    selection_idx = int(row.get("selection_idx", row.get("traj_selection_idx", 0)))
    seed = int(metrics_seed) + selection_idx
    computed: dict[str, Any] = {}
    if compute_cluster_mass:
        print(f"[{_traj_id(row)}] computing cluster mass metrics from {apf_dir}")
        computed.update(_compute_cluster_metrics(apf_dir, argparse.Namespace(**flat_args), seed=seed + 17))
    if compute_delta_h:
        print(f"[{_traj_id(row)}] computing deltaH metrics from {apf_dir}")
        computed.update(_compute_delta_h_metrics(apf_dir, flat_args, seed=seed + 31))

    out_path = metrics_path if metrics_path is not None else preferred_metrics_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Any] = {}
    if out_path.exists():
        with np.load(out_path) as old:
            merged.update({key: old[key] for key in old.files})
    merged.update(computed)
    np.savez_compressed(out_path, **merged)
    return out_path


def _load_delta_h_heatmap(metrics_path: Path) -> dict[str, Any]:
    with np.load(metrics_path) as data:
        if "delta_h_map" not in data.files:
            raise KeyError(f"{metrics_path} does not contain delta_h_map")
        z = np.asarray(data["delta_h_map"], dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"delta_h_map must be 2D (tau, window), got {z.shape}")
        if "delta_h_window_center_steps" in data.files:
            x = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        elif "delta_h_window_start_steps" in data.files and "delta_h_window_end_steps" in data.files:
            s0 = np.asarray(data["delta_h_window_start_steps"], dtype=np.float64).reshape(-1)
            s1 = np.asarray(data["delta_h_window_end_steps"], dtype=np.float64).reshape(-1)
            x = 0.5 * (s0 + s1)
        else:
            x = np.arange(z.shape[1], dtype=np.float64)
        if "delta_h_tau_steps" not in data.files:
            raise KeyError(f"{metrics_path} does not contain delta_h_tau_steps")
        tau_steps = np.asarray(data["delta_h_tau_steps"], dtype=np.float64).reshape(-1)
        if x.size != z.shape[1]:
            x = np.arange(z.shape[1], dtype=np.float64)
        if tau_steps.size != z.shape[0]:
            tau_steps = np.arange(z.shape[0], dtype=np.float64)
    return {"steps": x, "tau_steps": tau_steps, "delta_h_map": z}


def _colors_from_cluster_centers(data: np.lib.npyio.NpzFile, n_clusters: int) -> np.ndarray:
    if "cluster_centers_rgb" in data.files:
        colors = np.asarray(data["cluster_centers_rgb"], dtype=np.float64)
        if colors.ndim == 2 and colors.shape[0] >= n_clusters and colors.shape[1] >= 3:
            return np.clip(colors[:n_clusters, :3], 0.0, 1.0)
    if "cluster_centers_raw" not in data.files:
        return np.ones((n_clusters, 3), dtype=np.float64) * 0.25
    centers = np.asarray(data["cluster_centers_raw"], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] < n_clusters:
        return np.ones((n_clusters, 3), dtype=np.float64) * 0.25
    p = centers[:n_clusters]
    if p.shape[1] >= 3:
        rgb = p[:, :3]
    else:
        reps = int(np.ceil(3 / max(1, p.shape[1])))
        rgb = np.tile(p, (1, reps))[:, :3]
    return np.clip(rgb, 0.0, 1.0)


def _load_cluster_mass(metrics_path: Path, *, mode: str) -> dict[str, Any]:
    mass_key = "cluster_mass_prob" if mode == "prob" else "cluster_mass"
    with np.load(metrics_path) as data:
        if "cluster_steps" not in data.files:
            raise KeyError(f"{metrics_path} does not contain cluster_steps")
        if mass_key not in data.files:
            raise KeyError(f"{metrics_path} does not contain {mass_key}")
        steps = np.asarray(data["cluster_steps"], dtype=np.float64).reshape(-1)
        mass = np.asarray(data[mass_key], dtype=np.float64)
        if mass.ndim != 2:
            raise ValueError(f"{mass_key} must be 2D, got {mass.shape}")
        if steps.size != mass.shape[0]:
            steps = np.arange(mass.shape[0], dtype=np.float64)
        colors = _colors_from_cluster_centers(data, int(mass.shape[1]))
        tv_lag = None
        if "cluster_tv_lag" in data.files:
            tv_lag = np.asarray(data["cluster_tv_lag"], dtype=np.float64).reshape(-1)
        entropy = None
        if "cluster_entropy_norm" in data.files:
            entropy = np.asarray(data["cluster_entropy_norm"], dtype=np.float64).reshape(-1)
    return {
        "steps": steps,
        "mass": mass,
        "mode": mode,
        "colors": colors,
        "tv_lag": tv_lag,
        "entropy": entropy,
    }


def _scalar_from_npz(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0].item()


def _list_from_npz(data: np.lib.npyio.NpzFile, key: str) -> list[Any]:
    if key not in data.files:
        return []
    return np.asarray(data[key]).reshape(-1).tolist()


def _delta_h_metadata(
    *,
    dataset_root: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "traj_id": _traj_id(row),
        "metrics_path": str(metrics_path),
    }
    config_path = _config_path(dataset_root, row, manifest)
    if config_path is not None:
        payload["config_path"] = str(config_path)
        try:
            _cfg, flat = load_config(config_path)
            payload["config_values"] = {
                key: OmegaConf.to_container(flat.get(key), resolve=True)
                if OmegaConf.is_config(flat.get(key))
                else flat.get(key)
                for key in DELTA_H_CONFIG_KEYS
                if flat.get(key, None) is not None
            }
        except Exception as exc:
            payload["config_error"] = str(exc)

    with np.load(metrics_path) as data:
        tau_steps = _list_from_npz(data, "delta_h_tau_steps")
        window_centers = np.asarray(_list_from_npz(data, "delta_h_window_center_steps"), dtype=np.float64)
        window_starts = np.asarray(_list_from_npz(data, "delta_h_window_start_steps"), dtype=np.float64)
        payload["actual_values"] = {
            "delta_h_tau_steps": tau_steps,
            "delta_h_window_size_steps": _scalar_from_npz(data, "delta_h_window_size_steps"),
            "delta_h_sample_every_steps": _scalar_from_npz(data, "delta_h_sample_every_steps"),
            "n_tau": len(tau_steps),
            "n_windows": int(window_centers.size),
            "window_center_min_step": float(np.nanmin(window_centers)) if window_centers.size else None,
            "window_center_max_step": float(np.nanmax(window_centers)) if window_centers.size else None,
            "window_start_min_step": float(np.nanmin(window_starts)) if window_starts.size else None,
            "window_start_max_step": float(np.nanmax(window_starts)) if window_starts.size else None,
        }
    return payload


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _plot_delta_h_heatmap_one(
    *,
    plt: Any,
    row: dict[str, Any],
    series: dict[str, Any],
    out_path: Path,
    detect_start_step: int | None,
    detect_end_step: int | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> dict[str, Any]:
    traj_id = _traj_id(row)
    steps = np.asarray(series["steps"], dtype=np.float64)
    tau_steps = np.asarray(series["tau_steps"], dtype=np.float64)
    dh_map = np.asarray(series["delta_h_map"], dtype=np.float64)
    if dh_map.size == 0 or not np.any(np.isfinite(dh_map)):
        raise ValueError(f"No finite deltaH map values for {traj_id}")

    max_tau_i, max_step_i = np.unravel_index(int(np.nanargmax(dh_map)), dh_map.shape)
    max_step = float(steps[max_step_i])
    max_tau = float(tau_steps[max_tau_i])
    max_dh = float(dh_map[max_tau_i, max_step_i])
    mean_dh = float(np.nanmean(dh_map))

    fig, ax = plt.subplots(figsize=(10.8, 5.0), constrained_layout=True)
    im = ax.imshow(
        dh_map,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            float(np.nanmin(steps)),
            float(np.nanmax(steps)),
            float(np.nanmin(tau_steps)),
            float(np.nanmax(tau_steps)),
        ],
    )
    ax.scatter([max_step], [max_tau], color="white", edgecolor="black", s=32, linewidth=0.8, zorder=3)
    if detect_start_step is not None:
        ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if detect_end_step is not None:
        ax.axvline(int(detect_end_step), color="#666666", linestyle=":", linewidth=1.0, alpha=0.65)

    subtitle = []
    for key, label in (("loss", "loss"), ("iter", "iter"), ("saturation_T", "T")):
        if row.get(key, None) not in (None, ""):
            value = row[key]
            if isinstance(value, float):
                subtitle.append(f"{label}={value:.4g}")
            else:
                subtitle.append(f"{label}={value}")

    ax.set_title(f"{traj_id} deltaH heatmap" + (f" ({', '.join(subtitle)})" if subtitle else ""))
    ax.set_xlabel("simulation step")
    ax.set_ylabel("tau step")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("deltaH")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "traj_id": traj_id,
        "status": "ok",
        "delta_h_plot_path": str(out_path),
        "plot_path": str(out_path),
        "n_points": int(dh_map.size),
        "step_min": float(np.nanmin(steps)),
        "step_max": float(np.nanmax(steps)),
        "delta_h_max": max_dh,
        "delta_h_max_step": max_step,
        "delta_h_max_tau_step": max_tau,
        "delta_h_mean": mean_dh,
        "delta_h_n_tau": int(tau_steps.size),
        "delta_h_tau_grid": ",".join(str(int(x)) if float(x).is_integer() else f"{x:g}" for x in tau_steps),
    }


def _plot_delta_h_heatmap_grid(
    *,
    plt: Any,
    plotted: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: Path,
    detect_start_step: int | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if not plotted:
        return
    cols = 4
    rows_n = int(math.ceil(len(plotted) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.7 * rows_n), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    last_im = None
    for ax, (row, series) in zip(axes_arr, plotted):
        steps = np.asarray(series["steps"], dtype=np.float64)
        tau_steps = np.asarray(series["tau_steps"], dtype=np.float64)
        dh_map = np.asarray(series["delta_h_map"], dtype=np.float64)
        last_im = ax.imshow(
            dh_map,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[
                float(np.nanmin(steps)),
                float(np.nanmax(steps)),
                float(np.nanmin(tau_steps)),
                float(np.nanmax(tau_steps)),
            ],
        )
        if detect_start_step is not None:
            ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(_traj_id(row), fontsize=9)
        ax.grid(False)
    for ax in axes_arr[len(plotted) :]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_arr[: len(plotted)].tolist(), fraction=0.018, pad=0.01)
    fig.suptitle("deltaH heatmap by trajectory", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cluster_plot_indices(mean_mass: np.ndarray, *, top_k: int | None, min_mean: float) -> np.ndarray:
    mean = np.asarray(mean_mass, dtype=np.float64).reshape(-1)
    finite_mean = np.where(np.isfinite(mean), mean, -np.inf)
    idx = np.arange(mean.size, dtype=np.int32)
    if float(min_mean) > 0.0:
        idx = idx[finite_mean >= float(min_mean)]
    if idx.size == 0 and mean.size:
        idx = np.asarray([int(np.nanargmax(finite_mean))], dtype=np.int32)
    idx = idx[np.argsort(-finite_mean[idx])]
    if top_k is not None:
        idx = idx[: max(1, int(top_k))]
    return idx.astype(np.int32, copy=False)


def _plot_cluster_mass_one(
    *,
    plt: Any,
    row: dict[str, Any],
    series: dict[str, Any],
    out_path: Path,
    detect_start_step: int | None,
    detect_end_step: int | None,
    plot_top_k: int | None,
    plot_min_mean: float,
) -> dict[str, Any]:
    traj_id = _traj_id(row)
    steps = np.asarray(series["steps"], dtype=np.float64)
    mass = np.asarray(series["mass"], dtype=np.float64)
    finite = np.isfinite(steps) & np.all(np.isfinite(mass), axis=1)
    steps_f = steps[finite]
    mass_f = mass[finite]
    if mass_f.size == 0:
        raise ValueError(f"No finite cluster mass points for {traj_id}")

    n_clusters = int(mass_f.shape[1])
    colors = np.asarray(series.get("colors", np.ones((n_clusters, 3)) * 0.25), dtype=np.float64)
    if colors.shape[0] < n_clusters:
        colors = np.pad(colors, ((0, n_clusters - colors.shape[0]), (0, 0)), constant_values=0.25)
    mean_mass = np.nanmean(mass_f, axis=0)
    dominant_cluster = int(np.nanargmax(mean_mass))
    dominant_mean = float(mean_mass[dominant_cluster])
    plot_idx = _cluster_plot_indices(mean_mass, top_k=plot_top_k, min_mean=plot_min_mean)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for i_cluster in plot_idx:
        ax.plot(
            steps_f,
            mass_f[:, i_cluster],
            linewidth=1.65,
            color=colors[i_cluster, :3],
            label=f"c{i_cluster}",
        )
    if detect_start_step is not None:
        ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if detect_end_step is not None:
        ax.axvline(int(detect_end_step), color="#666666", linestyle=":", linewidth=1.0, alpha=0.65)

    subtitle = []
    for key, label in (("loss", "loss"), ("iter", "iter"), ("saturation_T", "T")):
        if row.get(key, None) not in (None, ""):
            value = row[key]
            subtitle.append(f"{label}={value:.4g}" if isinstance(value, float) else f"{label}={value}")
    mode_label = "mass fraction" if series.get("mode") == "prob" else "raw mass"
    ax.set_title(f"{traj_id} cluster mass" + (f" ({', '.join(subtitle)})" if subtitle else ""))
    ax.set_xlabel("simulation step")
    ax.set_ylabel(mode_label)
    if series.get("mode") == "prob":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    tv = series.get("tv_lag", None)
    entropy = series.get("entropy", None)
    return {
        "cluster_mass_plot_path": str(out_path),
        "cluster_mass_n_points": int(mass_f.shape[0]),
        "cluster_mass_n_clusters": n_clusters,
        "cluster_mass_n_plotted_clusters": int(plot_idx.size),
        "cluster_mass_dominant_cluster": dominant_cluster,
        "cluster_mass_dominant_mean": dominant_mean,
        "cluster_tv_lag_max": float(np.nanmax(tv)) if tv is not None and np.asarray(tv).size else "",
        "cluster_entropy_norm_min": float(np.nanmin(entropy)) if entropy is not None and np.asarray(entropy).size else "",
        "cluster_entropy_norm_max": float(np.nanmax(entropy)) if entropy is not None and np.asarray(entropy).size else "",
    }


def _plot_cluster_mass_grid(
    *,
    plt: Any,
    plotted: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: Path,
    detect_start_step: int | None,
    plot_top_k: int | None,
    plot_min_mean: float,
) -> None:
    if not plotted:
        return
    cols = 4
    rows_n = int(math.ceil(len(plotted) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.7 * rows_n), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (row, series) in zip(axes_arr, plotted):
        steps = np.asarray(series["steps"], dtype=np.float64)
        mass = np.asarray(series["mass"], dtype=np.float64)
        finite = np.isfinite(steps) & np.all(np.isfinite(mass), axis=1)
        steps_f = steps[finite]
        mass_f = mass[finite]
        n_clusters = int(mass_f.shape[1]) if mass_f.ndim == 2 else 0
        colors = np.asarray(series.get("colors", np.ones((n_clusters, 3)) * 0.25), dtype=np.float64)
        if colors.shape[0] < n_clusters:
            colors = np.pad(colors, ((0, n_clusters - colors.shape[0]), (0, 0)), constant_values=0.25)
        mean_mass = np.nanmean(mass_f, axis=0) if mass_f.size else np.zeros((n_clusters,), dtype=np.float64)
        for i_cluster in _cluster_plot_indices(mean_mass, top_k=plot_top_k, min_mean=plot_min_mean):
            ax.plot(steps_f, mass_f[:, i_cluster], linewidth=1.0, color=colors[i_cluster, :3], alpha=0.92)
        if detect_start_step is not None:
            ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=0.8, alpha=0.55)
        if series.get("mode") == "prob":
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(_traj_id(row), fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(plotted) :]:
        ax.axis("off")
    fig.suptitle("cluster mass by trajectory", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "traj_id",
        "status",
        "metrics_path",
        "delta_h_plot_path",
        "cluster_mass_plot_path",
        "n_points",
        "step_min",
        "step_max",
        "delta_h_max",
        "delta_h_max_step",
        "delta_h_max_tau_step",
        "delta_h_mean",
        "delta_h_n_tau",
        "delta_h_tau_grid",
        "cluster_mass_n_points",
        "cluster_mass_n_clusters",
        "cluster_mass_n_plotted_clusters",
        "cluster_mass_dominant_cluster",
        "cluster_mass_dominant_mean",
        "cluster_tv_lag_max",
        "cluster_entropy_norm_min",
        "cluster_entropy_norm_max",
        "loss",
        "iter",
        "saturation_T",
        "source",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_index(path: Path, rows: list[dict[str, Any]], delta_grid_path: Path, cluster_grid_path: Path) -> None:
    lines = ["# Minibang Metric Plots", ""]
    if delta_grid_path.exists():
        lines.extend(["## deltaH overview", "", f"![deltaH overview]({delta_grid_path.name})", ""])
    if cluster_grid_path.exists():
        lines.extend(["## cluster mass overview", "", f"![cluster mass overview]({cluster_grid_path.name})", ""])
    lines.append("| traj | status | max step | max tau | max deltaH | mean deltaH | deltaH | cluster mass | message |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        dh_plot = Path(str(row.get("delta_h_plot_path", ""))).name if row.get("delta_h_plot_path", "") else ""
        dh_plot_link = f"[png]({dh_plot})" if dh_plot else ""
        cluster_plot = (
            Path(str(row.get("cluster_mass_plot_path", ""))).name if row.get("cluster_mass_plot_path", "") else ""
        )
        cluster_plot_link = f"[png]({cluster_plot})" if cluster_plot else ""
        lines.append(
            "| {traj} | {status} | {step} | {tau} | {maxv} | {meanv} | {dh_plot} | {cluster_plot} | {message} |".format(
                traj=row.get("traj_id", ""),
                status=row.get("status", ""),
                step=_fmt_num(row.get("delta_h_max_step", "")),
                tau=_fmt_num(row.get("delta_h_max_tau_step", "")),
                maxv=_fmt_num(row.get("delta_h_max", "")),
                meanv=_fmt_num(row.get("delta_h_mean", "")),
                dh_plot=dh_plot_link,
                cluster_plot=cluster_plot_link,
                message=str(row.get("message", "")).replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt_num(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _parse_optional_int_arg(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "auto", ""}:
        return None
    return int(text)


def _parse_optional_bool_arg(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _recompute_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.metric_range_start_steps is not None:
        overrides["metric_range_start_steps"] = _parse_optional_int_arg(args.metric_range_start_steps)
    if args.metric_range_end_steps is not None:
        overrides["metric_range_end_steps"] = _parse_optional_int_arg(args.metric_range_end_steps)
    if args.cluster_method is not None:
        overrides["cluster_method"] = str(args.cluster_method)
    if args.cluster_space is not None:
        overrides["cluster_space"] = str(args.cluster_space)
    if args.cluster_dp_lambda is not None:
        overrides["cluster_dp_lambda"] = float(args.cluster_dp_lambda)
    if args.cluster_dp_iters is not None:
        overrides["cluster_dp_iters"] = int(args.cluster_dp_iters)
    if args.cluster_dp_max_clusters is not None:
        overrides["cluster_dp_max_clusters"] = int(args.cluster_dp_max_clusters)
    if args.cluster_standardize is not None:
        overrides["cluster_standardize"] = _parse_optional_bool_arg(args.cluster_standardize)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot deltaH and cluster-mass curves for FlowLenia minibang trajectory datasets."
    )
    parser.add_argument("dataset_root", help="Dataset root with manifest.json or traj_* directories.")
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/metric_plots.")
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="If metrics.npz misses deltaH or cluster mass metrics, compute them from apf_logs/*.npz first.",
    )
    parser.add_argument(
        "--overwrite-computed",
        action="store_true",
        help="With --compute-missing, recompute requested metrics even if metrics.npz already has them.",
    )
    parser.add_argument(
        "--recompute-delta-h",
        action="store_true",
        help="With --compute-missing, recompute deltaH even if metrics.npz already contains delta_h_map.",
    )
    parser.add_argument(
        "--recompute-cluster-mass",
        action="store_true",
        help="With --compute-missing, recompute cluster mass metrics even if metrics.npz already contains them.",
    )
    parser.add_argument("--metrics-seed", type=int, default=12345, help="Base seed for recomputing deltaH.")
    parser.add_argument(
        "--metric-range-start-steps",
        default=None,
        help="Override metric_range_start_steps when recomputing metrics.",
    )
    parser.add_argument(
        "--metric-range-end-steps",
        default=None,
        help="Override metric_range_end_steps when recomputing metrics. Use null/none/auto for rollout end.",
    )
    parser.add_argument("--cluster-method", choices=["kmeans", "dpmeans"], default=None)
    parser.add_argument(
        "--cluster-space",
        choices=["p", "p_rgb", "pcolor", "rendered", "pcolor_chroma", "rendered_chroma", "chroma"],
        default=None,
    )
    parser.add_argument("--cluster-dp-lambda", type=float, default=None)
    parser.add_argument("--cluster-dp-iters", type=int, default=None)
    parser.add_argument("--cluster-dp-max-clusters", type=int, default=None)
    parser.add_argument("--cluster-standardize", default=None, help="Override cluster_standardize: true/false.")
    parser.add_argument("--start-step", type=int, default=None, help="Optional vertical marker. Defaults to manifest.")
    parser.add_argument("--end-step", type=int, default=None, help="Optional vertical marker. Defaults to manifest.")
    parser.add_argument("--delta-h-cmap", default="magma", help="Matplotlib colormap for deltaH heatmaps.")
    parser.add_argument(
        "--delta-h-vmax-quantile",
        type=float,
        default=0.995,
        help="Global finite-value quantile used as heatmap vmax. Use 1.0 for exact max.",
    )
    parser.add_argument(
        "--cluster-mass-mode",
        choices=["prob", "raw"],
        default="prob",
        help="Plot normalized cluster mass fractions or raw cluster masses.",
    )
    parser.add_argument(
        "--cluster-plot-top-k",
        type=int,
        default=None,
        help="Only draw the top K clusters by mean plotted mass. Metrics are not modified.",
    )
    parser.add_argument(
        "--cluster-plot-min-mean",
        type=float,
        default=0.0,
        help="Only draw clusters whose mean plotted mass is at least this value. Metrics are not modified.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Only process the first N trajectories from manifest/traj_*; useful for debugging recomputation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    output_dir = resolve_path(args.output_dir, dataset_root) if args.output_dir else dataset_root / "metric_plots"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl_cache = Path(tempfile.gettempdir()) / "flowlenia_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, rows = _load_manifest_rows(dataset_root)
    total_rows = len(rows)
    if args.max_trials is not None:
        if int(args.max_trials) < 1:
            raise ValueError("--max-trials must be >= 1")
        rows = rows[: int(args.max_trials)]
        print(f"Processing first {len(rows)} of {total_rows} trajectories (--max-trials={args.max_trials})")
    detect_start_step = args.start_step if args.start_step is not None else _as_optional_int(manifest.get("detect_start_step"))
    detect_end_step = args.end_step if args.end_step is not None else _as_optional_int(manifest.get("detect_end_step"))
    metric_overrides = _recompute_cli_overrides(args)

    summary_rows: list[dict[str, Any]] = []
    delta_h_ready: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    cluster_mass_ready: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    delta_h_metadata_rows: list[dict[str, Any]] = []
    tau_grid_ref: np.ndarray | None = None
    for row in rows:
        traj_id = _traj_id(row)
        metrics_path, preferred_metrics_path = _metrics_path(dataset_root, row)
        summary = {
            "traj_id": traj_id,
            "status": "missing_or_failed",
            "metrics_path": str(metrics_path or preferred_metrics_path),
            "loss": row.get("loss", ""),
            "iter": row.get("iter", ""),
            "saturation_T": row.get("saturation_T", ""),
            "source": row.get("source", ""),
            "message": "",
        }
        messages: list[str] = []
        plotted_any = False
        try:
            if args.compute_missing:
                metrics_path = _compute_metrics_if_needed(
                    dataset_root=dataset_root,
                    row=row,
                    manifest=manifest,
                    metrics_path=metrics_path,
                    preferred_metrics_path=preferred_metrics_path,
                    metrics_seed=int(args.metrics_seed),
                    overwrite_delta_h=bool(args.overwrite_computed or args.recompute_delta_h),
                    overwrite_cluster_mass=bool(args.overwrite_computed or args.recompute_cluster_mass),
                    need_delta_h=True,
                    need_cluster_mass=True,
                    metric_overrides=metric_overrides,
                )
                summary["metrics_path"] = str(metrics_path or preferred_metrics_path)
            if metrics_path is None or not metrics_path.exists():
                raise FileNotFoundError("metrics.npz not found")
            summary["metrics_path"] = str(metrics_path)

            try:
                delta_h_series = _load_delta_h_heatmap(metrics_path)
                tau_grid = np.asarray(delta_h_series["tau_steps"], dtype=np.float64)
                if tau_grid_ref is None:
                    tau_grid_ref = tau_grid
                elif tau_grid.shape != tau_grid_ref.shape or not np.allclose(tau_grid, tau_grid_ref):
                    raise ValueError(
                        "deltaH tau grid differs from previous trajectories: "
                        f"{tau_grid.tolist()} != {tau_grid_ref.tolist()}"
                    )
                delta_h_metadata_rows.append(
                    _delta_h_metadata(
                        dataset_root=dataset_root,
                        row=row,
                        manifest=manifest,
                        metrics_path=metrics_path,
                    )
                )
                delta_h_ready.append((row, delta_h_series, summary))
                plotted_any = True
            except Exception as exc:
                messages.append(f"deltaH: {exc}")

            try:
                cluster_mass_series = _load_cluster_mass(metrics_path, mode=str(args.cluster_mass_mode))
                cluster_mass_ready.append((row, cluster_mass_series, summary))
                plotted_any = True
            except Exception as exc:
                messages.append(f"cluster_mass: {exc}")

            if plotted_any:
                summary["status"] = "partial" if messages else "ok"
            else:
                summary["status"] = "missing_or_failed"
                print(f"[{traj_id}] skipped: {'; '.join(messages) if messages else 'no plottable metrics'}")
            summary["message"] = "; ".join(messages)
            summary_rows.append(summary)
        except Exception as exc:
            summary["status"] = "missing_or_failed"
            summary["message"] = str(exc)
            summary_rows.append(summary)
            print(f"[{traj_id}] skipped: {exc}")

    delta_h_values = []
    for _row, series, _summary in delta_h_ready:
        vals = np.asarray(series["delta_h_map"], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            delta_h_values.append(vals)
    if delta_h_values:
        all_delta_h = np.concatenate(delta_h_values)
        vmin = float(np.nanmin(all_delta_h))
        q = min(max(float(args.delta_h_vmax_quantile), 0.0), 1.0)
        vmax = float(np.nanquantile(all_delta_h, q))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(all_delta_h))
    else:
        vmin = None
        vmax = None

    delta_h_plotted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row, series, summary in delta_h_ready:
        traj_id = _traj_id(row)
        try:
            delta_h_plot_path = output_dir / f"{traj_id}_delta_h_heatmap.png"
            summary.update(
                _plot_delta_h_heatmap_one(
                    plt=plt,
                    row=row,
                    series=series,
                    out_path=delta_h_plot_path,
                    detect_start_step=detect_start_step,
                    detect_end_step=detect_end_step,
                    cmap=str(args.delta_h_cmap),
                    vmin=vmin,
                    vmax=vmax,
                )
            )
            delta_h_plotted.append((row, series))
        except Exception as exc:
            summary["status"] = "partial" if summary.get("cluster_mass_plot_path") else "missing_or_failed"
            summary["message"] = "; ".join([x for x in [summary.get("message", ""), f"deltaH_plot: {exc}"] if x])
            print(f"[{traj_id}] deltaH heatmap failed: {exc}")

    cluster_mass_plotted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row, series, summary in cluster_mass_ready:
        traj_id = _traj_id(row)
        try:
            cluster_mass_plot_path = output_dir / f"{traj_id}_cluster_mass.png"
            summary.update(
                _plot_cluster_mass_one(
                    plt=plt,
                    row=row,
                    series=series,
                    out_path=cluster_mass_plot_path,
                    detect_start_step=detect_start_step,
                    detect_end_step=detect_end_step,
                    plot_top_k=args.cluster_plot_top_k,
                    plot_min_mean=float(args.cluster_plot_min_mean),
                )
            )
            cluster_mass_plotted.append((row, series))
        except Exception as exc:
            summary["status"] = "partial" if summary.get("delta_h_plot_path") else "missing_or_failed"
            summary["message"] = "; ".join([x for x in [summary.get("message", ""), f"cluster_mass_plot: {exc}"] if x])
            print(f"[{traj_id}] cluster mass plot failed: {exc}")

    for summary in summary_rows:
        has_dh = bool(summary.get("delta_h_plot_path", ""))
        has_cluster = bool(summary.get("cluster_mass_plot_path", ""))
        if has_dh and has_cluster and not summary.get("message", ""):
            summary["status"] = "ok"
        elif has_dh or has_cluster:
            summary["status"] = "partial"

    delta_grid_path = output_dir / "delta_h_heatmap_grid.png"
    _plot_delta_h_heatmap_grid(
        plt=plt,
        plotted=delta_h_plotted,
        out_path=delta_grid_path,
        detect_start_step=detect_start_step,
        cmap=str(args.delta_h_cmap),
        vmin=vmin,
        vmax=vmax,
    )
    cluster_grid_path = output_dir / "cluster_mass_grid.png"
    _plot_cluster_mass_grid(
        plt=plt,
        plotted=cluster_mass_plotted,
        out_path=cluster_grid_path,
        detect_start_step=detect_start_step,
        plot_top_k=args.cluster_plot_top_k,
        plot_min_mean=float(args.cluster_plot_min_mean),
    )
    _write_csv(output_dir / "metric_plot_summary.csv", summary_rows)
    (output_dir / "delta_h_config.json").write_text(json.dumps(delta_h_metadata_rows, indent=2, sort_keys=True) + "\n")
    _write_index(output_dir / "index.md", summary_rows, delta_grid_path, cluster_grid_path)
    print(
        f"Wrote {len(delta_h_plotted)} deltaH heatmaps and "
        f"{len(cluster_mass_plotted)} cluster-mass plots to {output_dir}"
    )
    if len(delta_h_plotted) != len(rows) or len(cluster_mass_plotted) != len(rows):
        print(
            "Some trajectories were skipped or partially plotted; "
            "see metric_plot_summary.csv"
        )


if __name__ == "__main__":
    main()
