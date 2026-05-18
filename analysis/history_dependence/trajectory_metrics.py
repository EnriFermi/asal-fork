from __future__ import annotations

import json
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .io import infer_lagrangian_metadata
from .utils import pair_type, progress, progress_bar
from scripts.clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config


def _periodic_delta(dx: np.ndarray, *, periodic: bool, domain_y: float, domain_x: float) -> np.ndarray:
    out = np.asarray(dx, dtype=np.float64).copy()
    if periodic and domain_y > 0:
        out[..., 0] = (out[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
    if periodic and domain_x > 0:
        out[..., 1] = (out[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
    return out


def _resolve_fixed_tau_index(
    tau_steps: np.ndarray,
    tau_frames: np.ndarray,
    *,
    fixed_tau_steps: int | None = None,
    fixed_tau_frames: int | None = None,
) -> int | None:
    if fixed_tau_steps is not None:
        matches = np.flatnonzero(np.asarray(tau_steps, dtype=np.int64) == int(fixed_tau_steps))
        if matches.size < 1:
            raise ValueError(
                f"Requested fixed tau_steps={fixed_tau_steps}, but available tau_steps are "
                f"{np.asarray(tau_steps, dtype=np.int64).tolist()}."
            )
        return int(matches[0])
    if fixed_tau_frames is not None:
        matches = np.flatnonzero(np.asarray(tau_frames, dtype=np.int64) == int(fixed_tau_frames))
        if matches.size < 1:
            raise ValueError(
                f"Requested fixed tau_frames={fixed_tau_frames}, but available tau_frames are "
                f"{np.asarray(tau_frames, dtype=np.int64).tolist()}."
            )
        return int(matches[0])
    return None


def delta_h_distribution_distance(values_a: np.ndarray, values_b: np.ndarray, metric: str = "wasserstein") -> float:
    a = np.sort(np.asarray(values_a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(values_b, dtype=np.float64).reshape(-1))
    if a.size < 1 or b.size < 1:
        raise ValueError("Delta-H distribution vectors must be non-empty.")

    metric = str(metric).strip().lower()
    if metric == "wasserstein":
        x = np.concatenate([a, b])
        x.sort()
        if x.size < 2:
            return 0.0
        dx = np.diff(x)
        cdf_a = np.searchsorted(a, x[:-1], side="right") / float(a.size)
        cdf_b = np.searchsorted(b, x[:-1], side="right") / float(b.size)
        return float(np.sum(np.abs(cdf_a - cdf_b) * dx))
    if metric == "ks":
        x = np.concatenate([a, b])
        x.sort()
        cdf_a = np.searchsorted(a, x, side="right") / float(a.size)
        cdf_b = np.searchsorted(b, x, side="right") / float(b.size)
        return float(np.max(np.abs(cdf_a - cdf_b)))
    if metric == "energy":
        cross = np.mean(np.abs(a[:, None] - b[None, :]))
        within_a = np.mean(np.abs(a[:, None] - a[None, :]))
        within_b = np.mean(np.abs(b[:, None] - b[None, :]))
        val = max(0.0, 2.0 * cross - within_a - within_b)
        return float(np.sqrt(val))
    raise ValueError(f"Unsupported delta-h distribution distance metric={metric!r}.")


def derive_metric_config(cfg: dict[str, Any], run_collection) -> dict[str, Any] | None:
    traj_cfg = dict(cfg.get("trajectories", {}))
    if not bool(traj_cfg.get("enabled", True)):
        return None

    source_metric = dict(run_collection.metric_summary or {})
    lag_meta = infer_lagrangian_metadata(run_collection)
    rollout_steps = (
        traj_cfg.get("rollout_steps")
        or source_metric.get("trajectory_window_steps")
        or lag_meta.get("trajectory_window_steps")
    )
    sample_every_steps = (
        traj_cfg.get("sample_every_steps")
        or source_metric.get("sample_every_steps")
        or lag_meta.get("sample_every_steps")
    )
    time_sampling = traj_cfg.get("time_sampling") or source_metric.get("time_sampling") or lag_meta.get("time_sampling")
    if rollout_steps is None and sample_every_steps is not None and time_sampling is not None:
        rollout_steps = int(sample_every_steps) * int(time_sampling)
    if rollout_steps is None or sample_every_steps is None:
        return None

    args = SimpleNamespace(
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
        time_sampling=None if time_sampling is None else int(time_sampling),
        metric_window_size_frames=traj_cfg.get("metric_window_size_frames", source_metric.get("window_size_frames")),
        metric_window_size_steps=traj_cfg.get(
            "metric_window_size_steps",
            lag_meta.get("metric_window_size_steps"),
        ),
        metric_window_step_frames=traj_cfg.get("metric_window_step_frames", source_metric.get("window_step_frames")),
        metric_window_step_steps=traj_cfg.get(
            "metric_window_step_steps",
            lag_meta.get("metric_window_step_steps"),
        ),
        metric_tau_mode=traj_cfg.get("metric_tau_mode", source_metric.get("tau_mode", "fixed")),
        metric_tau_frames=traj_cfg.get("metric_tau_frames", source_metric.get("tau_frames")),
        metric_tau_steps=traj_cfg.get(
            "metric_tau_steps",
            lag_meta.get("metric_tau_steps", source_metric.get("tau_steps")),
        ),
        metric_tau_grid_frames=traj_cfg.get("metric_tau_grid_frames"),
        metric_tau_grid_steps=traj_cfg.get("metric_tau_grid_steps"),
        metric_range_start_steps=traj_cfg.get("metric_range_start_steps", source_metric.get("range_start_steps")),
        metric_range_end_steps=traj_cfg.get("metric_range_end_steps", source_metric.get("range_end_steps")),
        metric_m_samples=traj_cfg.get("metric_m_samples", source_metric.get("m_count", 48)),
        metric_m_min=traj_cfg.get("metric_m_min", 4),
        metric_n_proj=traj_cfg.get("metric_n_proj", source_metric.get("n_proj", 16)),
        metric_null_reps=traj_cfg.get("metric_null_reps", source_metric.get("null_reps", 6)),
        metric_particle_samples=traj_cfg.get("metric_particle_samples", source_metric.get("particle_samples", 64)),
        metric_dirs_seed=traj_cfg.get("metric_dirs_seed", 123),
        metric_periodic=traj_cfg.get("metric_periodic", source_metric.get("periodic", False)),
        metric_domain_y=traj_cfg.get("metric_domain_y", source_metric.get("domain_y", 0.0)),
        metric_domain_x=traj_cfg.get("metric_domain_x", source_metric.get("domain_x", 0.0)),
        metric_preprocess_mode=traj_cfg.get("metric_preprocess_mode", source_metric.get("preprocess_mode", "clip")),
        metric_scales=traj_cfg.get("metric_scales", source_metric.get("scales")),
        metric_scale_weights=traj_cfg.get("metric_scale_weights"),
        metric_alpha=traj_cfg.get("metric_alpha", source_metric.get("alpha", 0.0)),
        metric_beta=traj_cfg.get("metric_beta", source_metric.get("beta", 1.0)),
        metric_eps=traj_cfg.get("metric_eps", 1e-12),
    )
    return resolve_metric_config(args)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


@lru_cache(maxsize=32)
def _cached_metric_eval(metric_cfg_json: str):
    metric_cfg = json.loads(metric_cfg_json)
    return jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))


def _metric_eval(metric_cfg: dict[str, Any]):
    key = json.dumps(metric_cfg, sort_keys=True, default=_json_default)
    return _cached_metric_eval(key)


def compute_delta_h_summary(
    xy_seq: np.ndarray,
    metric_cfg: dict[str, Any],
    *,
    selected_tau_index: int | None = None,
    metric_rng_seed: int | None = None,
    metric_rng_fold_in: int | None = None,
    progress_desc: str | None = None,
    progress_enabled: bool = False,
) -> dict[str, Any]:
    xy = np.asarray(xy_seq, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"Lagrangian trajectories must have shape (T, N, 2), got {xy.shape}.")

    if progress_enabled:
        desc = progress_desc or "Delta-H"
        print(f"[{desc}] scoring with scripts.clip_deltah_msc_metric.make_metric_loss_fn")

    metric_eval = _metric_eval(metric_cfg)
    rng_seed = int(metric_cfg["dirs_seed"]) if metric_rng_seed is None else int(metric_rng_seed)
    rng = jax.random.PRNGKey(rng_seed)
    if metric_rng_fold_in is not None:
        rng = jax.random.fold_in(rng, int(metric_rng_fold_in))

    tau_selector = None
    if selected_tau_index is not None:
        tau_count = max(1, len(metric_cfg.get("tau_frames_list", [metric_cfg["tau_frames"]])))
        clipped_idx = int(np.clip(int(selected_tau_index), 0, tau_count - 1))
        if tau_count <= 1:
            tau_selector = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            frac = np.clip(float(clipped_idx) / float(tau_count - 1), 1e-6, 1.0 - 1e-6)
            tau_selector = jnp.asarray(np.log(frac / (1.0 - frac)), dtype=jnp.float32)

    _, info = metric_eval(rng, jnp.asarray(xy), tau_selector=tau_selector)
    info = jax.device_get(info)

    h_all = np.asarray(info["delta_h_map"], dtype=np.float64)
    h_best = np.asarray(info["delta_h_best"], dtype=np.float64)
    score_all = np.asarray(info["score_by_tau"], dtype=np.float64)
    amp_all = np.asarray(info["amp_by_tau"], dtype=np.float64)
    msc_all = np.asarray(info["msc_by_tau"], dtype=np.float64)
    tau_frames = np.asarray(info["tau_frames"], dtype=np.int32)
    tau_steps = np.asarray(info["tau_steps"], dtype=np.int32)
    starts = np.asarray(info["window_start_frames"], dtype=np.int32)
    starts_steps = np.asarray(info["window_start_steps"], dtype=np.int32)
    best_idx = int(np.asarray(info["tau_selected_idx"]).item())
    return {
        "delta_h_map": h_all,
        "delta_h_best": h_best,
        "score_by_tau": score_all,
        "amp_by_tau": amp_all,
        "msc_by_tau": msc_all,
        "tau_frames": tau_frames,
        "tau_steps": tau_steps,
        "window_start_frames": starts.astype(np.int32),
        "window_start_steps": starts_steps.astype(np.int32),
        "tau_best_idx": int(best_idx),
        "tau_best_frames": int(np.asarray(info["tau_best_frames"]).item()),
        "tau_best_steps": int(np.asarray(info["tau_best_steps"]).item()),
        "score_scalar": float(np.asarray(info["score"]).item()),
        "amp_scalar": float(np.asarray(info["amp"]).item()),
        "msc_scalar": float(np.asarray(info["msc"]).item()),
        "delta_h_best_mean": float(np.mean(h_best)),
        "delta_h_best_std": float(np.std(h_best)),
    }


def compute_coarse_observables(xy_seq: np.ndarray, metric_cfg: dict[str, Any], *, occupancy_bins: int = 64) -> dict[str, float]:
    xy = np.asarray(xy_seq, dtype=np.float64)
    dt = max(float(metric_cfg["sample_every_steps"]), 1e-12)
    dx = _periodic_delta(
        xy[1:] - xy[:-1],
        periodic=bool(metric_cfg["periodic"]),
        domain_y=float(metric_cfg["domain_y"]),
        domain_x=float(metric_cfg["domain_x"]),
    )
    speeds = np.linalg.norm(dx, axis=-1) / dt
    centers = np.mean(xy, axis=1, keepdims=True)
    spatial_spread = np.linalg.norm(xy - centers, axis=-1)

    domain_y = float(metric_cfg["domain_y"])
    domain_x = float(metric_cfg["domain_x"])
    if domain_y > 0 and domain_x > 0:
        y0, y1 = 0.0, domain_y
        x0, x1 = 0.0, domain_x
        area_norm = domain_y * domain_x
    else:
        y0 = float(np.min(xy[..., 0]))
        y1 = float(np.max(xy[..., 0]))
        x0 = float(np.min(xy[..., 1]))
        x1 = float(np.max(xy[..., 1]))
        area_norm = max((y1 - y0) * (x1 - x0), 1e-12)

    occ_vals = []
    bbox_vals = []
    y_scale = max(y1 - y0, 1e-12)
    x_scale = max(x1 - x0, 1e-12)
    for frame in xy:
        yy = np.clip(((frame[:, 0] - y0) / y_scale) * occupancy_bins, 0, occupancy_bins - 1e-9).astype(np.int32)
        xx = np.clip(((frame[:, 1] - x0) / x_scale) * occupancy_bins, 0, occupancy_bins - 1e-9).astype(np.int32)
        occ = np.unique(yy * occupancy_bins + xx).size / float(occupancy_bins * occupancy_bins)
        occ_vals.append(float(occ))
        bbox_area = max(float(np.max(frame[:, 0]) - np.min(frame[:, 0])), 0.0) * max(
            float(np.max(frame[:, 1]) - np.min(frame[:, 1])),
            0.0,
        )
        bbox_vals.append(float(bbox_area / area_norm))

    return {
        "mean_speed": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "occupied_area_fraction": float(np.mean(occ_vals)),
        "bbox_area_fraction": float(np.mean(bbox_vals)),
        "spatial_spread": float(np.mean(spatial_spread)),
    }


def delta_h_map_distance(map_a: np.ndarray, map_b: np.ndarray, metric: str = "l2") -> float:
    a = np.asarray(map_a, dtype=np.float64).reshape(-1)
    b = np.asarray(map_b, dtype=np.float64).reshape(-1)
    metric = str(metric).strip().lower()
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "mean_abs":
        return float(np.mean(np.abs(a - b)))
    if metric == "cosine":
        an = a / np.clip(np.linalg.norm(a), 1e-12, None)
        bn = b / np.clip(np.linalg.norm(b), 1e-12, None)
        return float(1.0 - np.dot(an, bn))
    raise ValueError(f"Unsupported delta-h distance metric={metric!r}.")


def compute_trajectory_observables(
    runs: pd.DataFrame,
    load_lagrangian_fn,
    cfg: dict[str, Any],
    metric_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    traj_cfg = dict(cfg.get("trajectories", {}))
    progress_cfg = dict(cfg.get("progress", {}))
    show_progress = bool(progress_cfg.get("enabled", True))
    show_inner = bool(progress_cfg.get("show_inner", True))
    occupancy_bins = int(traj_cfg.get("occupancy_bins", 64))
    selected_tau_index = traj_cfg.get("selected_tau_index")
    fixed_tau_steps = traj_cfg.get("fixed_tau_distribution_steps", None)
    fixed_tau_frames = traj_cfg.get("fixed_tau_distribution_frames", None)
    available = runs[runs["has_lagrangian"]].copy().reset_index(drop=True)
    if available.empty:
        return pd.DataFrame(), {}

    run_rows = []
    per_run: dict[str, dict[str, Any]] = {}
    run_iter = progress(
        available.to_dict(orient="records"),
        total=int(available.shape[0]),
        desc="Trajectory observables",
        enabled=show_progress,
        leave=False,
    )
    for row in run_iter:
        xy = load_lagrangian_fn(row)
        delta_h_summary = compute_delta_h_summary(
            xy,
            metric_cfg,
            selected_tau_index=None if selected_tau_index is None else int(selected_tau_index),
            progress_desc=f"Delta-H {row['run_id']}",
            progress_enabled=show_progress and show_inner,
        )
        coarse = compute_coarse_observables(xy, metric_cfg, occupancy_bins=occupancy_bins)
        metrics = dict(delta_h_summary)
        fixed_tau_idx = _resolve_fixed_tau_index(
            metrics["tau_steps"],
            metrics["tau_frames"],
            fixed_tau_steps=None if fixed_tau_steps is None else int(fixed_tau_steps),
            fixed_tau_frames=None if fixed_tau_frames is None else int(fixed_tau_frames),
        )
        if fixed_tau_idx is not None:
            fixed_tau_values = np.asarray(metrics["delta_h_map"][fixed_tau_idx], dtype=np.float64)
            metrics.update(
                {
                    "delta_h_fixed_tau": fixed_tau_values,
                    "delta_h_fixed_tau_idx": int(fixed_tau_idx),
                    "delta_h_fixed_tau_steps": int(np.asarray(metrics["tau_steps"])[fixed_tau_idx]),
                    "delta_h_fixed_tau_frames": int(np.asarray(metrics["tau_frames"])[fixed_tau_idx]),
                    "delta_h_fixed_tau_mean": float(np.mean(fixed_tau_values)),
                    "delta_h_fixed_tau_std": float(np.std(fixed_tau_values)),
                }
            )
        metrics.update(coarse)
        per_run[row["run_id"]] = metrics
        run_rows.append(
            {
                "run_id": row["run_id"],
                "condition": row["condition"],
                "variant": row["variant"],
                "pair_group_id": row["pair_group_id"],
                "tau_best_idx": int(metrics["tau_best_idx"]),
                "tau_best_frames": int(metrics["tau_best_frames"]),
                "tau_best_steps": int(metrics["tau_best_steps"]),
                "score_scalar": float(metrics["score_scalar"]),
                "amp_scalar": float(metrics["amp_scalar"]),
                "msc_scalar": float(metrics["msc_scalar"]),
                "delta_h_best_mean": float(metrics["delta_h_best_mean"]),
                "delta_h_best_std": float(metrics["delta_h_best_std"]),
                "delta_h_fixed_tau_idx": None if "delta_h_fixed_tau_idx" not in metrics else int(metrics["delta_h_fixed_tau_idx"]),
                "delta_h_fixed_tau_steps": None if "delta_h_fixed_tau_steps" not in metrics else int(metrics["delta_h_fixed_tau_steps"]),
                "delta_h_fixed_tau_frames": None if "delta_h_fixed_tau_frames" not in metrics else int(metrics["delta_h_fixed_tau_frames"]),
                "delta_h_fixed_tau_mean": None if "delta_h_fixed_tau_mean" not in metrics else float(metrics["delta_h_fixed_tau_mean"]),
                "delta_h_fixed_tau_std": None if "delta_h_fixed_tau_std" not in metrics else float(metrics["delta_h_fixed_tau_std"]),
                "mean_speed": float(metrics["mean_speed"]),
                "speed_std": float(metrics["speed_std"]),
                "occupied_area_fraction": float(metrics["occupied_area_fraction"]),
                "bbox_area_fraction": float(metrics["bbox_area_fraction"]),
                "spatial_spread": float(metrics["spatial_spread"]),
            }
        )
    return pd.DataFrame(run_rows), per_run


def compute_trajectory_pairwise(
    runs: pd.DataFrame,
    run_metrics: pd.DataFrame,
    per_run: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    traj_cfg = dict(cfg.get("trajectories", {}))
    progress_cfg = dict(cfg.get("progress", {}))
    show_progress = bool(progress_cfg.get("enabled", True))
    map_metrics = [str(x) for x in traj_cfg.get("pairwise_map_metrics", ["l2"])]
    distribution_metrics = [str(x) for x in traj_cfg.get("pairwise_distribution_metrics", [])]
    scalar_keys = [
        "msc_scalar",
        "score_scalar",
        "amp_scalar",
        "mean_speed",
        "speed_std",
        "occupied_area_fraction",
        "bbox_area_fraction",
        "spatial_spread",
    ]

    available = runs[runs["run_id"].isin(run_metrics["run_id"])].copy().reset_index(drop=True)
    n_runs = int(available.shape[0])
    run_ids = available["run_id"].tolist()
    matrix_names = [f"delta_h_{metric}" for metric in map_metrics]
    fixed_tau_steps = traj_cfg.get("fixed_tau_distribution_steps", None)
    fixed_tau_frames = traj_cfg.get("fixed_tau_distribution_frames", None)
    dist_name_suffix = None
    if distribution_metrics:
        if fixed_tau_steps is not None:
            dist_name_suffix = f"tau{int(fixed_tau_steps)}"
        elif fixed_tau_frames is not None:
            dist_name_suffix = f"tauf{int(fixed_tau_frames)}"
        else:
            dist_name_suffix = "taufixed"
        matrix_names.extend(f"delta_h_dist_{dist_name_suffix}_{metric}" for metric in distribution_metrics)
    matrix_names.extend(f"absdiff_{key}" for key in scalar_keys)
    matrices = {
        name: np.zeros((n_runs, n_runs), dtype=np.float64)
        for name in matrix_names
    }

    metrics_by_id = run_metrics.set_index("run_id").to_dict(orient="index")
    pair_rows = []
    pair_total = n_runs * (n_runs - 1) // 2
    with progress_bar(total=pair_total, desc="Trajectory pairwise", enabled=show_progress, leave=False) as pbar:
        for i in range(n_runs):
            row_i = available.iloc[i]
            data_i = per_run[row_i["run_id"]]
            scal_i = metrics_by_id[row_i["run_id"]]
            for j in range(i + 1, n_runs):
                row_j = available.iloc[j]
                data_j = per_run[row_j["run_id"]]
                scal_j = metrics_by_id[row_j["run_id"]]
                record = {
                    "run_a": row_i["run_id"],
                    "run_b": row_j["run_id"],
                    "condition_a": row_i["condition"],
                    "condition_b": row_j["condition"],
                    "pair_type": pair_type(row_i["condition"], row_j["condition"]),
                    "pair_group_a": row_i["pair_group_id"],
                    "pair_group_b": row_j["pair_group_id"],
                    "same_pair_group": bool(row_i["pair_group_id"] == row_j["pair_group_id"]),
                }
                for metric in map_metrics:
                    name = f"delta_h_{metric}"
                    value = delta_h_map_distance(data_i["delta_h_map"], data_j["delta_h_map"], metric=metric)
                    matrices[name][i, j] = value
                    matrices[name][j, i] = value
                    record[name] = value
                if distribution_metrics:
                    if "delta_h_fixed_tau" not in data_i or "delta_h_fixed_tau" not in data_j:
                        raise ValueError(
                            "Requested pairwise_distribution_metrics, but delta_h_fixed_tau is unavailable. "
                            "Set trajectories.fixed_tau_distribution_steps or fixed_tau_distribution_frames."
                        )
                    for metric in distribution_metrics:
                        name = f"delta_h_dist_{dist_name_suffix}_{metric}"
                        value = delta_h_distribution_distance(data_i["delta_h_fixed_tau"], data_j["delta_h_fixed_tau"], metric=metric)
                        matrices[name][i, j] = value
                        matrices[name][j, i] = value
                        record[name] = value
                for key in scalar_keys:
                    name = f"absdiff_{key}"
                    value = abs(float(scal_i[key]) - float(scal_j[key]))
                    matrices[name][i, j] = value
                    matrices[name][j, i] = value
                    record[name] = value
                pair_rows.append(record)
                pbar.update(1)

    matrix_frames = {
        name: pd.DataFrame(value, index=run_ids, columns=run_ids)
        for name, value in matrices.items()
    }
    return pd.DataFrame(pair_rows), matrix_frames
