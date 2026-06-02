from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.clip_deltah_msc_metric import resolve_metric_config

from .embedding_metrics import cloud_distance, synchronized_distance
from .pipeline import load_analysis_config
from .trajectory_metrics import compute_delta_h_summary, delta_h_distribution_distance, delta_h_map_distance


_PAIRWISE_STAT_KEYS = (
    "baseline_distance",
    "walls_effect_distance",
    "walls_effect_distance_ctrl_a",
    "walls_effect_distance_ctrl_b",
    "effect_minus_baseline",
    "effect_over_baseline_ratio",
)

_MSC_SCALAR_COLUMNS = (
    "msc_loss_control_a",
    "msc_loss_control_b",
    "msc_loss_control_mean",
    "msc_loss_walls",
    "msc_loss_walls_minus_control_mean",
    "msc_loss_walls_minus_control_a",
    "msc_score_control_a",
    "msc_score_control_b",
    "msc_score_control_mean",
    "msc_score_walls",
    "msc_score_walls_minus_control_mean",
    "msc_score_walls_minus_control_a",
    "msc_amp_control_a",
    "msc_amp_control_b",
    "msc_amp_walls",
    "msc_component_control_a",
    "msc_component_control_b",
    "msc_component_walls",
    "msc_tau_best_steps_control_a",
    "msc_tau_best_steps_control_b",
    "msc_tau_best_steps_walls",
    "msc_score_anchor_absdiff_minus_baseline",
    "msc_loss_anchor_absdiff_minus_baseline",
    "msc_metric_backend",
    "msc_metric_rng_seed_base",
    "msc_sample_every_steps",
    "msc_time_sampling",
    "msc_metric_periodic",
    "msc_metric_positions_unwrapped",
    "msc_metric_scale_weight_sum",
    "msc_metric_normalized_by_scale_weight_sum",
    "msc_metric_delta_h_floor",
    "msc_metric_msc_floor",
    "msc_metric_msc_term",
    "msc_metric_eps",
    "msc_metric_alpha",
    "msc_metric_beta",
)


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


def _progress(iterable, *, total: int, enabled: bool, desc: str):
    if not enabled:
        for item in iterable:
            yield item
        return
    if os.environ.get("PAPER_SUITE_LOG_PROGRESS") == "plain":
        for idx, item in enumerate(iterable, start=1):
            if idx == 1 or idx == total or idx % 5 == 0:
                print(f"[{desc}] {idx}/{total}", flush=True)
            yield item
        return
    try:
        from tqdm.auto import tqdm  # type: ignore

        yield from tqdm(iterable, total=total, desc=desc, leave=False)
        return
    except Exception:
        pass
    for idx, item in enumerate(iterable, start=1):
        if idx == 1 or idx == total or idx % 5 == 0:
            print(f"[{desc}] {idx}/{total}")
        yield item


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
                xy_len = int(np.asarray(data[key]).shape[0])
                meta["time_sampling"] = xy_len
                break
        for key in ("xy_late_sample_steps", "sample_offsets_steps"):
            if key not in data.files:
                continue
            steps = np.asarray(data[key], dtype=np.int64).reshape(-1)
            if steps.size != int(meta.get("time_sampling", steps.size)):
                continue
            positive = np.diff(steps)
            positive = positive[positive > 0]
            if positive.size:
                meta["sample_every_steps"] = int(round(float(np.median(positive))))
                if "trajectory_window_steps" not in meta and steps.size:
                    meta["trajectory_window_steps"] = int(steps[-1] - steps[0] + int(meta["sample_every_steps"]))
                break
    return meta


def _cfg_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [x.strip() for x in text.split(",") if x.strip()] if text else None
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray, dict)):
            return list(value)
    except Exception:
        pass
    return [value]


def _safe_metric_timing(
    traj_cfg: dict[str, Any],
    lag_meta: dict[str, Any],
    source_metric: dict[str, Any],
    *,
    rollout_steps: int,
    sample_every_steps: int,
) -> dict[str, Any]:
    sample_every_steps = max(1, int(sample_every_steps))
    rollout_steps = max(sample_every_steps, int(rollout_steps))
    time_sampling = max(1, rollout_steps // sample_every_steps)
    m_min = max(1, int(traj_cfg.get("metric_m_min", 4)))

    win_frames_raw = traj_cfg.get("metric_window_size_frames", source_metric.get("window_size_frames"))
    if win_frames_raw is not None:
        win_frames = int(win_frames_raw)
    else:
        win_steps_raw = traj_cfg.get("metric_window_size_steps", lag_meta.get("metric_window_size_steps"))
        if win_steps_raw is None:
            win_steps_raw = source_metric.get("window_size_steps", 20_000)
        win_frames = int(max(1, round(float(win_steps_raw) / float(sample_every_steps))))
    win_frames = max(1, min(int(win_frames), int(time_sampling)))
    win_steps = int(win_frames * sample_every_steps)

    step_frames_raw = traj_cfg.get("metric_window_step_frames", source_metric.get("window_step_frames"))
    if step_frames_raw is not None:
        step_frames = int(step_frames_raw)
    else:
        step_steps_raw = traj_cfg.get("metric_window_step_steps", lag_meta.get("metric_window_step_steps"))
        if step_steps_raw is None:
            step_steps_raw = source_metric.get("window_step_steps", 5_000)
        step_frames = int(max(1, round(float(step_steps_raw) / float(sample_every_steps))))
    step_frames = max(1, min(int(step_frames), int(win_frames)))
    step_steps = int(step_frames * sample_every_steps)

    max_tau_frames = max(1, int(win_frames) - int(m_min))
    tau_frames_raw = traj_cfg.get("metric_tau_frames", source_metric.get("tau_frames"))
    if tau_frames_raw is not None:
        tau_frames = int(tau_frames_raw)
    else:
        tau_steps_raw = traj_cfg.get("metric_tau_steps", lag_meta.get("metric_tau_steps", source_metric.get("tau_steps", 3_000)))
        tau_frames = int(max(1, round(float(tau_steps_raw) / float(sample_every_steps))))
    tau_frames = max(1, min(int(tau_frames), int(max_tau_frames)))
    tau_steps = int(tau_frames * sample_every_steps)

    grid_frames_raw = traj_cfg.get("metric_tau_grid_frames")
    if grid_frames_raw is not None:
        grid_frames = [int(x) for x in (_cfg_list(grid_frames_raw) or [])]
    else:
        grid_steps_raw = traj_cfg.get("metric_tau_grid_steps")
        grid_frames = [
            int(max(1, round(float(x) / float(sample_every_steps))))
            for x in (_cfg_list(grid_steps_raw) or [])
        ]
    grid_frames = sorted({int(x) for x in grid_frames if 0 < int(x) <= int(max_tau_frames)})
    if not grid_frames:
        grid_frames = [tau_frames]
    grid_steps = [int(x * sample_every_steps) for x in grid_frames]

    range_end = traj_cfg.get("metric_range_end_steps", source_metric.get("range_end_steps"))
    if range_end is not None:
        range_end = min(int(range_end), int(rollout_steps))

    return {
        "metric_window_size_steps": win_steps,
        "metric_window_step_steps": step_steps,
        "metric_tau_steps": tau_steps,
        "metric_tau_grid_steps": grid_steps,
        "metric_range_end_steps": range_end,
    }


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


def _distribution_tau_request(traj_cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    tau_mode = str(traj_cfg.get("metric_tau_mode", "fixed")).strip().lower()
    if tau_mode == "fixed":
        metric_tau_steps = traj_cfg.get("metric_tau_steps")
        metric_tau_frames = traj_cfg.get("metric_tau_frames")
        if metric_tau_steps is not None:
            return int(metric_tau_steps), None
        if metric_tau_frames is not None:
            return None, int(metric_tau_frames)

    fixed_tau_steps = traj_cfg.get("fixed_tau_distribution_steps")
    fixed_tau_frames = traj_cfg.get("fixed_tau_distribution_frames")
    return (
        None if fixed_tau_steps is None else int(fixed_tau_steps),
        None if fixed_tau_frames is None else int(fixed_tau_frames),
    )


def _distribution_tau_suffix(traj_cfg: dict[str, Any]) -> str:
    fixed_tau_steps, fixed_tau_frames = _distribution_tau_request(traj_cfg)
    if fixed_tau_steps is not None:
        return f"tau{int(fixed_tau_steps)}"
    if fixed_tau_frames is not None:
        return f"tauf{int(fixed_tau_frames)}"
    return "taufixed"


def _normalize_delta_h_values(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 1:
        raise ValueError("Delta-H distribution vectors must be non-empty.")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std < eps:
        return arr - mean
    return (arr - mean) / std


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
        or lag_meta.get("trajectory_window_steps")
        or source_metric.get("rollout_steps")
    )
    sample_every_steps = (
        traj_cfg.get("sample_every_steps")
        or lag_meta.get("sample_every_steps")
        or source_metric.get("sample_every_steps")
    )
    time_sampling = (
        traj_cfg.get("time_sampling")
        or lag_meta.get("time_sampling")
        or source_metric.get("time_sampling")
    )

    if rollout_steps is None and sample_every_steps is not None and time_sampling is not None:
        rollout_steps = int(sample_every_steps) * int(time_sampling)
    if rollout_steps is None or sample_every_steps is None:
        raise ValueError(
            f"Could not derive trajectory metric config for {lagrangian_path}: "
            f"rollout_steps={rollout_steps}, sample_every_steps={sample_every_steps}"
        )
    timing = _safe_metric_timing(
        traj_cfg,
        lag_meta,
        source_metric,
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
    )

    args = SimpleNamespace(
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
        time_sampling=None if time_sampling is None else int(time_sampling),
        metric_window_size_frames=None,
        metric_window_size_steps=timing["metric_window_size_steps"],
        metric_window_step_frames=None,
        metric_window_step_steps=timing["metric_window_step_steps"],
        metric_tau_mode=traj_cfg.get("metric_tau_mode", source_metric.get("tau_mode", "fixed")),
        metric_tau_frames=None,
        metric_tau_steps=timing["metric_tau_steps"],
        metric_tau_grid_frames=None,
        metric_tau_grid_steps=timing["metric_tau_grid_steps"],
        metric_range_start_steps=traj_cfg.get("metric_range_start_steps", source_metric.get("range_start_steps")),
        metric_range_end_steps=timing["metric_range_end_steps"],
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
        metric_delta_h_floor=traj_cfg.get("metric_delta_h_floor", source_metric.get("delta_h_floor", 0.0)),
        metric_msc_floor=traj_cfg.get("metric_msc_floor", source_metric.get("msc_floor", 0.01)),
        metric_msc_term=traj_cfg.get("metric_msc_term", source_metric.get("msc_term", "floor_reconstruction_error")),
        metric_msc_normalize_by_weight_sum=traj_cfg.get(
            "metric_msc_normalize_by_weight_sum",
            str(source_metric.get("scale_normalization", "sum_weight_r")) == "sum_weight_r",
        ),
        metric_alpha=traj_cfg.get("metric_alpha", source_metric.get("alpha", 0.0)),
        metric_beta=traj_cfg.get("metric_beta", source_metric.get("beta", 1.0)),
        metric_eps=traj_cfg.get("metric_eps", source_metric.get("eps", 1e-12)),
    )
    metric_cfg = resolve_metric_config(args)
    positions_unwrapped = traj_cfg.get(
        "metric_positions_unwrapped",
        traj_cfg.get("positions_unwrapped", source_metric.get("positions_unwrapped", False)),
    )
    metric_cfg["positions_unwrapped"] = bool(positions_unwrapped)
    return metric_cfg


def _paper_check_metric_seed_base(row: dict[str, Any]) -> int | None:
    metric_seed = row.get("metric_seed")
    if not _is_missing(metric_seed):
        return int(metric_seed)
    seed_x = row.get("seed_x")
    if _is_missing(seed_x):
        seed_x = row.get("x_seed")
    if _is_missing(seed_x):
        return None
    return int(seed_x) + 10_000_000


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
        suffix = _distribution_tau_suffix(traj_cfg)
        names.extend(f"delta_h_dist_{suffix}_{metric}" for metric in distribution_metrics)
        names.extend(f"delta_h_dist_{suffix}_{metric}_zscore" for metric in distribution_metrics)
    return names


def history_distance_effect_columns(analysis_cfg_or_path: dict[str, Any] | str | Path) -> list[str]:
    return [f"{name}__effect_minus_baseline" for name in history_distance_base_names(analysis_cfg_or_path)]


def _recomputed_metric_columns(base_names: list[str]) -> list[str]:
    columns = list(_MSC_SCALAR_COLUMNS)
    for base_name in base_names:
        columns.extend(f"{base_name}__{key}" for key in _PAIRWISE_STAT_KEYS)
    return columns


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
    row_dict: dict[str, Any],
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

    metric_seed_base = _paper_check_metric_seed_base(row_dict)
    summary_a = compute_delta_h_summary(
        xy_a,
        metric_cfg,
        metric_rng_seed=metric_seed_base,
        metric_rng_fold_in=0,
    )
    summary_b = compute_delta_h_summary(
        xy_b,
        metric_cfg,
        metric_rng_seed=metric_seed_base,
        metric_rng_fold_in=1,
    )
    summary_c = compute_delta_h_summary(
        xy_c,
        metric_cfg,
        metric_rng_seed=metric_seed_base,
        metric_rng_fold_in=2,
    )
    out: dict[str, float] = _msc_scalar_metrics_from_summaries(
        summary_a,
        summary_b,
        summary_c,
        metric_cfg=metric_cfg,
        metric_seed_base=metric_seed_base,
        time_sampling=int(xy_a.shape[0]),
    )

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
        fixed_tau_steps, fixed_tau_frames = _distribution_tau_request(traj_cfg)
        fixed_tau_idx = _resolve_fixed_tau_index(
            np.asarray(summary_a["tau_steps"], dtype=np.int64),
            np.asarray(summary_a["tau_frames"], dtype=np.int64),
            fixed_tau_steps=fixed_tau_steps,
            fixed_tau_frames=fixed_tau_frames,
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
        fixed_a_z = _normalize_delta_h_values(fixed_a)
        fixed_b_z = _normalize_delta_h_values(fixed_b)
        fixed_c_z = _normalize_delta_h_values(fixed_c)
        for metric in distribution_metrics:
            base_name = f"delta_h_dist_{suffix}_{metric}_zscore"
            stats = _pairwise_effect_triplet(
                lambda x, y, metric=metric: delta_h_distribution_distance(x, y, metric=metric),
                fixed_a_z,
                fixed_b_z,
                fixed_c_z,
            )
            out.update({f"{base_name}__{key}": value for key, value in stats.items()})
    return out


def _msc_scalar_metrics_from_summaries(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    summary_c: dict[str, Any],
    *,
    metric_cfg: dict[str, Any],
    metric_seed_base: int | None,
    time_sampling: int,
) -> dict[str, float]:
    score_a = float(summary_a["score_scalar"])
    score_b = float(summary_b["score_scalar"])
    score_w = float(summary_c["score_scalar"])
    loss_a = -score_a
    loss_b = -score_b
    loss_w = -score_w
    score_control_mean = 0.5 * (score_a + score_b)
    loss_control_mean = 0.5 * (loss_a + loss_b)
    return {
        "msc_metric_backend": "scripts.clip_deltah_msc_metric.make_metric_loss_fn",
        "msc_metric_rng_seed_base": np.nan if metric_seed_base is None else float(metric_seed_base),
        "msc_metric_periodic": float(bool(metric_cfg["periodic"])),
        "msc_metric_positions_unwrapped": float(bool(metric_cfg.get("positions_unwrapped", False))),
        "msc_metric_scale_weight_sum": float(metric_cfg.get("scale_weight_sum", sum(float(w) for _, w in metric_cfg["scale_pairs"]))),
        "msc_metric_normalized_by_scale_weight_sum": float(
            str(metric_cfg.get("scale_normalization", "sum_weight_r")) == "sum_weight_r"
        ),
        "msc_metric_delta_h_floor": float(metric_cfg.get("delta_h_floor", 0.0)),
        "msc_metric_msc_floor": float(metric_cfg.get("msc_floor", metric_cfg.get("delta_h_floor", 0.0))),
        "msc_metric_msc_term": str(metric_cfg.get("msc_term", "overlap")),
        "msc_metric_eps": float(metric_cfg["eps"]),
        "msc_metric_alpha": float(metric_cfg["alpha"]),
        "msc_metric_beta": float(metric_cfg["beta"]),
        "msc_loss_control_a": float(loss_a),
        "msc_loss_control_b": float(loss_b),
        "msc_loss_control_mean": float(loss_control_mean),
        "msc_loss_walls": float(loss_w),
        "msc_loss_walls_minus_control_mean": float(loss_w - loss_control_mean),
        "msc_loss_walls_minus_control_a": float(loss_w - loss_a),
        "msc_score_control_a": float(score_a),
        "msc_score_control_b": float(score_b),
        "msc_score_control_mean": float(score_control_mean),
        "msc_score_walls": float(score_w),
        "msc_score_walls_minus_control_mean": float(score_w - score_control_mean),
        "msc_score_walls_minus_control_a": float(score_w - score_a),
        "msc_amp_control_a": float(summary_a["amp_scalar"]),
        "msc_amp_control_b": float(summary_b["amp_scalar"]),
        "msc_amp_walls": float(summary_c["amp_scalar"]),
        "msc_component_control_a": float(summary_a["msc_scalar"]),
        "msc_component_control_b": float(summary_b["msc_scalar"]),
        "msc_component_walls": float(summary_c["msc_scalar"]),
        "msc_tau_best_steps_control_a": float(summary_a["tau_best_steps"]),
        "msc_tau_best_steps_control_b": float(summary_b["tau_best_steps"]),
        "msc_tau_best_steps_walls": float(summary_c["tau_best_steps"]),
        "msc_score_anchor_absdiff_minus_baseline": float(abs(score_w - score_a) - abs(score_b - score_a)),
        "msc_loss_anchor_absdiff_minus_baseline": float(abs(loss_w - loss_a) - abs(loss_b - loss_a)),
        "msc_sample_every_steps": float(metric_cfg["sample_every_steps"]),
        "msc_time_sampling": float(time_sampling),
    }


def augment_rows_with_history_dependence_distances(
    rows: pd.DataFrame,
    *,
    analysis_config_path: str | Path,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if rows.empty:
        cfg = load_analysis_config(analysis_config_path)
        base_names = history_distance_base_names(cfg)
        return rows.copy(), [f"{name}__effect_minus_baseline" for name in base_names], base_names

    analysis_cfg = load_analysis_config(analysis_config_path)
    base_names = history_distance_base_names(analysis_cfg)
    effect_cols = [f"{name}__effect_minus_baseline" for name in base_names]
    out = rows.copy()
    for col in _recomputed_metric_columns(base_names):
        if col in out.columns:
            out[col] = np.nan
    out["history_embeddings_artifact_found"] = False
    out["history_trajectory_artifact_found"] = False

    row_iter = list(out.iterrows())
    for idx, row in _progress(
        row_iter,
        total=len(row_iter),
        enabled=show_progress,
        desc="history distances",
    ):
        row_dict = dict(row)
        computed: dict[str, float] = {}

        embeddings_path = _resolve_artifact_path(row_dict, "embeddings_path")
        if embeddings_path is not None and embeddings_path.exists():
            out.at[idx, "history_embeddings_artifact_found"] = True
            computed.update(_compute_embedding_metrics(embeddings_path, analysis_cfg))

        lagrangian_path = _resolve_artifact_path(row_dict, "lagrangian_path")
        if lagrangian_path is not None and lagrangian_path.exists():
            out.at[idx, "history_trajectory_artifact_found"] = True
            frustration_root = (
                Path(str(row_dict.get("frustration_root")))
                if not _is_missing(row_dict.get("frustration_root"))
                else lagrangian_path.parent.parent
            )
            computed.update(
                _compute_trajectory_metrics(
                    lagrangian_path,
                    analysis_cfg,
                    row_dict=row_dict,
                    source_metric_summary=_load_source_metric_summary(frustration_root),
                )
            )

        for key, value in computed.items():
            out.at[idx, key] = value

    return out, [col for col in effect_cols if col in out.columns], base_names
