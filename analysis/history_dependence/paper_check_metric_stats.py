from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.clip_deltah_msc_metric import resolve_metric_config

from .embedding_metrics import cloud_distance, synchronized_distance
from .pipeline import load_analysis_config
from .trajectory_metrics import compute_delta_h_summary, delta_h_distribution_distance, delta_h_map_distance


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _prepare_embeddings(z: np.ndarray, *, normalize: bool) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must have shape (T, D), got {arr.shape}.")
    if not normalize:
        return arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _resolve_artifact_path(row: dict[str, Any], field: str) -> Path | None:
    raw = row.get(field)
    if _is_missing(raw):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    base = row.get("frustration_root") or row.get("source_root")
    if _is_missing(base):
        return path
    return Path(str(base)) / path


def _load_source_metric_summary(frustration_root: Path) -> dict[str, Any]:
    summary_path = frustration_root / "summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text())
    summary = payload.get("msc_metric_summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _infer_lag_meta(lagrangian_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    with np.load(lagrangian_path, allow_pickle=False) as data:
        for key in (
            "sample_every_steps",
            "trajectory_start_steps",
            "trajectory_end_steps",
            "trajectory_window_steps",
            "metric_window_size_steps",
            "metric_window_step_steps",
            "metric_tau_steps",
        ):
            if key not in data.files:
                continue
            arr = np.asarray(data[key])
            if arr.shape == ():
                meta[key] = arr.item()
        for key in ("xy_control_a", "xy_control_b", "xy_walls"):
            if key in data.files:
                meta["time_sampling"] = int(np.asarray(data[key]).shape[0])
                break
    return meta


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


def _build_metric_cfg(
    analysis_cfg: dict[str, Any],
    *,
    lagrangian_path: Path,
    source_metric_summary: dict[str, Any],
) -> dict[str, Any]:
    traj_cfg = dict(analysis_cfg.get("trajectories", {}))
    lag_meta = _infer_lag_meta(lagrangian_path)
    source_metric = dict(source_metric_summary or {})

    rollout_steps = (
        traj_cfg.get("rollout_steps")
        or source_metric.get("rollout_steps")
        or lag_meta.get("trajectory_window_steps")
    )
    sample_every_steps = (
        traj_cfg.get("sample_every_steps")
        or source_metric.get("sample_every_steps")
        or lag_meta.get("sample_every_steps")
    )
    time_sampling = (
        traj_cfg.get("time_sampling")
        or source_metric.get("time_sampling")
        or lag_meta.get("time_sampling")
    )

    if rollout_steps is None and sample_every_steps is not None and time_sampling is not None:
        rollout_steps = int(sample_every_steps) * int(time_sampling)
    if rollout_steps is None or sample_every_steps is None:
        raise ValueError(
            f"Could not derive trajectory metric config for {lagrangian_path}: "
            f"rollout_steps={rollout_steps}, sample_every_steps={sample_every_steps}"
        )

    args = SimpleNamespace(
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
        time_sampling=None if time_sampling is None else int(time_sampling),
        metric_window_size_frames=traj_cfg.get("metric_window_size_frames", source_metric.get("window_size_frames")),
        metric_window_size_steps=traj_cfg.get("metric_window_size_steps", lag_meta.get("metric_window_size_steps")),
        metric_window_step_frames=traj_cfg.get("metric_window_step_frames", source_metric.get("window_step_frames")),
        metric_window_step_steps=traj_cfg.get("metric_window_step_steps", lag_meta.get("metric_window_step_steps")),
        metric_tau_mode=traj_cfg.get("metric_tau_mode", source_metric.get("tau_mode", "fixed")),
        metric_tau_frames=traj_cfg.get("metric_tau_frames", source_metric.get("tau_frames")),
        metric_tau_steps=traj_cfg.get("metric_tau_steps", lag_meta.get("metric_tau_steps", source_metric.get("tau_steps"))),
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


def history_distance_base_names(analysis_cfg_or_path: dict[str, Any] | str | Path) -> list[str]:
    cfg = (
        analysis_cfg_or_path
        if isinstance(analysis_cfg_or_path, dict)
        else load_analysis_config(analysis_cfg_or_path)
    )
    emb_cfg = dict(cfg.get("embeddings", {}))
    traj_cfg = dict(cfg.get("trajectories", {}))

    names = [f"embedding_synced_{str(metric)}" for metric in emb_cfg.get("synced_metrics", ["cosine"])]
    cloud_method = str(emb_cfg.get("cloud_method", "chamfer"))
    names.extend(
        f"embedding_cloud_{cloud_method}_{str(metric)}"
        for metric in emb_cfg.get("cloud_metrics", ["cosine"])
    )

    names.extend(f"delta_h_{str(metric)}" for metric in traj_cfg.get("pairwise_map_metrics", ["l2"]))

    distribution_metrics = [str(metric) for metric in traj_cfg.get("pairwise_distribution_metrics", [])]
    if distribution_metrics:
        fixed_tau_steps = traj_cfg.get("fixed_tau_distribution_steps")
        fixed_tau_frames = traj_cfg.get("fixed_tau_distribution_frames")
        if fixed_tau_steps is not None:
            suffix = f"tau{int(fixed_tau_steps)}"
        elif fixed_tau_frames is not None:
            suffix = f"tauf{int(fixed_tau_frames)}"
        else:
            suffix = "taufixed"
        names.extend(f"delta_h_dist_{suffix}_{metric}" for metric in distribution_metrics)
    return names


def history_distance_effect_columns(analysis_cfg_or_path: dict[str, Any] | str | Path) -> list[str]:
    return [f"{name}__effect_minus_baseline" for name in history_distance_base_names(analysis_cfg_or_path)]


def _pairwise_effect_triplet(distance_fn, a: Any, b: Any, c: Any) -> dict[str, float]:
    baseline = float(distance_fn(a, b))
    walls_a = float(distance_fn(a, c))
    walls_b = float(distance_fn(b, c))
    walls_effect = 0.5 * (walls_a + walls_b)
    return {
        "baseline_distance": baseline,
        "walls_effect_distance": walls_effect,
        "walls_effect_distance_ctrl_a": walls_a,
        "walls_effect_distance_ctrl_b": walls_b,
        "effect_minus_baseline": walls_effect - baseline,
        "effect_over_baseline_ratio": walls_effect / max(baseline, 1e-12),
    }


def _compute_embedding_metrics(embeddings_path: Path, analysis_cfg: dict[str, Any]) -> dict[str, float]:
    emb_cfg = dict(analysis_cfg.get("embeddings", {}))
    normalize = bool(emb_cfg.get("normalize", True))
    out: dict[str, float] = {}
    with np.load(embeddings_path, allow_pickle=False) as data:
        z_a = _prepare_embeddings(data["z_control_a"], normalize=normalize)
        z_b = _prepare_embeddings(data["z_control_b"], normalize=normalize)
        z_c = _prepare_embeddings(data["z_walls"], normalize=normalize)

    for metric in [str(x) for x in emb_cfg.get("synced_metrics", ["cosine"])]:
        base_name = f"embedding_synced_{metric}"
        stats = _pairwise_effect_triplet(
            lambda x, y, metric=metric: synchronized_distance(x, y, metric=metric),
            z_a,
            z_b,
            z_c,
        )
        out.update({f"{base_name}__{key}": value for key, value in stats.items()})

    cloud_method = str(emb_cfg.get("cloud_method", "chamfer"))
    for metric in [str(x) for x in emb_cfg.get("cloud_metrics", ["cosine"])]:
        base_name = f"embedding_cloud_{cloud_method}_{metric}"
        stats = _pairwise_effect_triplet(
            lambda x, y, metric=metric: cloud_distance(x, y, metric=metric, method=cloud_method),
            z_a,
            z_b,
            z_c,
        )
        out.update({f"{base_name}__{key}": value for key, value in stats.items()})
    return out


def _compute_trajectory_metrics(
    lagrangian_path: Path,
    analysis_cfg: dict[str, Any],
    *,
    source_metric_summary: dict[str, Any],
) -> dict[str, float]:
    traj_cfg = dict(analysis_cfg.get("trajectories", {}))
    metric_cfg = _build_metric_cfg(
        analysis_cfg,
        lagrangian_path=lagrangian_path,
        source_metric_summary=source_metric_summary,
    )

    with np.load(lagrangian_path, allow_pickle=False) as data:
        xy_a = np.asarray(data["xy_control_a"], dtype=np.float64)
        xy_b = np.asarray(data["xy_control_b"], dtype=np.float64)
        xy_c = np.asarray(data["xy_walls"], dtype=np.float64)

    summary_a = compute_delta_h_summary(xy_a, metric_cfg)
    summary_b = compute_delta_h_summary(xy_b, metric_cfg)
    summary_c = compute_delta_h_summary(xy_c, metric_cfg)
    out: dict[str, float] = {}

    for metric in [str(x) for x in traj_cfg.get("pairwise_map_metrics", ["l2"])]:
        base_name = f"delta_h_{metric}"
        stats = _pairwise_effect_triplet(
            lambda x, y, metric=metric: delta_h_map_distance(x["delta_h_map"], y["delta_h_map"], metric=metric),
            summary_a,
            summary_b,
            summary_c,
        )
        out.update({f"{base_name}__{key}": value for key, value in stats.items()})

    distribution_metrics = [str(metric) for metric in traj_cfg.get("pairwise_distribution_metrics", [])]
    if distribution_metrics:
        fixed_tau_idx = _resolve_fixed_tau_index(
            np.asarray(summary_a["tau_steps"], dtype=np.int64),
            np.asarray(summary_a["tau_frames"], dtype=np.int64),
            fixed_tau_steps=None if traj_cfg.get("fixed_tau_distribution_steps") is None else int(traj_cfg["fixed_tau_distribution_steps"]),
            fixed_tau_frames=None if traj_cfg.get("fixed_tau_distribution_frames") is None else int(traj_cfg["fixed_tau_distribution_frames"]),
        )
        if fixed_tau_idx is None:
            raise ValueError(
                "pairwise_distribution_metrics are configured, but neither "
                "fixed_tau_distribution_steps nor fixed_tau_distribution_frames is set."
            )
        fixed_tau_steps = int(np.asarray(summary_a["tau_steps"], dtype=np.int64)[fixed_tau_idx])
        suffix = f"tau{fixed_tau_steps}"
        fixed_a = np.asarray(summary_a["delta_h_map"][fixed_tau_idx], dtype=np.float64)
        fixed_b = np.asarray(summary_b["delta_h_map"][fixed_tau_idx], dtype=np.float64)
        fixed_c = np.asarray(summary_c["delta_h_map"][fixed_tau_idx], dtype=np.float64)
        for metric in distribution_metrics:
            base_name = f"delta_h_dist_{suffix}_{metric}"
            stats = _pairwise_effect_triplet(
                lambda x, y, metric=metric: delta_h_distribution_distance(x, y, metric=metric),
                fixed_a,
                fixed_b,
                fixed_c,
            )
            out.update({f"{base_name}__{key}": value for key, value in stats.items()})
    return out


def augment_rows_with_history_dependence_distances(
    rows: pd.DataFrame,
    *,
    analysis_config_path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if rows.empty:
        cfg = load_analysis_config(analysis_config_path)
        base_names = history_distance_base_names(cfg)
        return rows.copy(), [f"{name}__effect_minus_baseline" for name in base_names], base_names

    analysis_cfg = load_analysis_config(analysis_config_path)
    base_names = history_distance_base_names(analysis_cfg)
    effect_cols = [f"{name}__effect_minus_baseline" for name in base_names]
    out = rows.copy()

    source_metric_cache: dict[Path, dict[str, Any]] = {}

    for idx, row in out.iterrows():
        row_dict = dict(row)
        computed: dict[str, float] = {}

        embeddings_path = _resolve_artifact_path(row_dict, "embeddings_path")
        if embeddings_path is not None and embeddings_path.exists():
            computed.update(_compute_embedding_metrics(embeddings_path, analysis_cfg))

        lagrangian_path = _resolve_artifact_path(row_dict, "lagrangian_path")
        if lagrangian_path is not None and lagrangian_path.exists():
            frustration_root = Path(str(row_dict.get("frustration_root"))) if not _is_missing(row_dict.get("frustration_root")) else lagrangian_path.parent.parent
            if frustration_root not in source_metric_cache:
                source_metric_cache[frustration_root] = _load_source_metric_summary(frustration_root)
            computed.update(
                _compute_trajectory_metrics(
                    lagrangian_path,
                    analysis_cfg,
                    source_metric_summary=source_metric_cache[frustration_root],
                )
            )

        for key, value in computed.items():
            out.at[idx, key] = value

    return out, [col for col in effect_cols if col in out.columns], base_names
