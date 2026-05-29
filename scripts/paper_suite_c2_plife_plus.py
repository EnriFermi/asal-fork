from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from clip_deltah_msc_metric import make_metric_loss_fn
from paper_suite_common import command_to_str, ensure_dir, load_config, log_event, read_csv, resolve_path, to_plain, write_csv, write_json
from paper_suite_c2_branching import (
    _as_float_list,
    _is_delta_h_sweep_mode,
    _is_ranked_high_low_mode,
    _select_delta_h_points,
    _select_ranked_high_low_points,
)
from paper_suite_metric_cache import sha256_text, stable_json
from paper_suite_plife_c1_lagrangian import (
    _apply_section_base_overrides,
    _flatten_base_config,
    _load_base_config,
    _make_substrate,
)
from paper_suite_posthoc import _metric_config_from_lagrangian, _primary_lagrangian_xy_key, _score_maps


METRICS_MANIFEST_COLUMNS = [
    "traj_id",
    "trial_idx",
    "optimized_run_idx",
    "candidate_kind",
    "candidate_idx",
    "candidate_label",
    "seed_x",
    "metric_seed",
    "lagrangian_path",
    "params_path",
    "status",
    "message",
    "metrics_path",
    "n_points",
    "sample_every_steps",
    "trajectory_start_steps",
    "trajectory_end_steps",
]

PLIFE_BRANCH_PLAN_VERSION = "c2_plife_branch_plan_v2_flow_style"

BRANCH_PLAN_COLUMNS = [
    "point_id",
    "pair_id",
    "traj_id",
    "trial_idx",
    "optimized_run_idx",
    "condition",
    "window_idx",
    "step",
    "requested_step",
    "step_snap_delta",
    "delta_h",
    "delta_h_energy",
    "target_quantile",
    "delta_h_quantile",
    "delta_h_quantile_rank",
    "selection_mode",
    "selection_method",
    "metrics_path",
    "lagrangian_path",
    "params_path",
    "seed_x",
    "sample_every_steps",
    "trajectory_start_steps",
    "trajectory_end_steps",
    "horizon_steps",
    "branch_id",
    "branch_seed",
    "branch_dir",
    "branch_output_path",
    "perturb_x_std",
    "perturb_v_std",
    "perturb_c_std",
    "status",
    "command",
]

BRANCHING_SCORE_COLUMNS = [
    "traj_id",
    "point_id",
    "pair_id",
    "condition",
    "step",
    "delta_h",
    "branching_score",
    "branching_metric",
    "n_branches",
    "n_branch_pairs",
    "foundation_model",
    "max_future_frames",
    "clip_embedding_cache_n",
]


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _progress_now(idx: int, total: int, *, every: int = 5) -> bool:
    return idx <= 1 or idx == total or (every > 0 and idx % every == 0)


def _output_root(cfg: Any) -> Path:
    return ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))


def _plife_dataset_cfg(cfg: Any) -> Any:
    return _get(cfg.get("datasets", {}), "plife_plus", {})


def _plife_c2_cfg(cfg: Any) -> Any:
    return _get(_get(cfg.get("c2", {}), "plife_plus", {}), "config", _get(cfg.get("c2", {}), "plife_plus", {}))


def _out_dir(cfg: Any, output_root: Path) -> Path:
    raw = _get(_plife_c2_cfg(cfg), "output_dir", None)
    if raw:
        path = resolve_path(raw)
        return ensure_dir(path if path is not None else output_root / "c2_plife_plus_branching")
    return ensure_dir(output_root / "c2_plife_plus_branching")


def _branch_root(cfg: Any, output_root: Path) -> Path:
    raw = _get(_plife_c2_cfg(cfg), "branch_root", None)
    if raw:
        path = resolve_path(raw)
        return ensure_dir(path if path is not None else output_root / "c2_plife_plus_branching" / "branches")
    return ensure_dir(output_root / "c2_plife_plus_branching" / "branches")


def _trajectory_root(cfg: Any, *, smoke: bool) -> Path | None:
    c2_cfg = _plife_c2_cfg(cfg)
    raw = _get(c2_cfg, "trajectory_root", None)
    if raw is None:
        raw = _get(_get(_plife_dataset_cfg(cfg), "c1", {}), "lagrangian_root", None)
    if raw is None and smoke:
        raw = "analysis/results/paper_suite_smoke/smoke_inputs/plife_plus/frustration_simulation"
    return resolve_path(raw)


def _simulation_section(cfg: Any) -> Any:
    return _get(cfg.get("simulation", {}), "plife_plus_c1_lagrangian", {})


def _base_config_path(cfg: Any) -> Path | None:
    raw = _get(_plife_c2_cfg(cfg), "base_config", None)
    if raw is None:
        raw = _get(_simulation_section(cfg), "base_config", None)
    if raw is None:
        raw = "experiments/paper_check_plife_plus/frustration_simulation/config.yaml"
    return resolve_path(raw)


def _metric_cfg_raw(cfg: Any) -> Any:
    c2_metric = _get(_plife_c2_cfg(cfg), "metric", None)
    if c2_metric is not None:
        return c2_metric
    return _get(_get(_plife_dataset_cfg(cfg), "c1", {}), "metric", {})


def _manifest_path(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _canonical_kind(kind: Any, label: Any = None) -> str:
    text = f"{kind} {'' if label is None else label}".lower()
    if "opt" in text or "best" in text:
        return "optimized"
    if "rand" in text:
        return "random"
    return str(kind or "other")


def _iter_trajectory_items(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            kind = _canonical_kind(row.get("candidate_kind", "optimized"), row.get("candidate_label", None))
            lagrangian = _manifest_path(root, row.get("lagrangian_path"), default=root / "trial_data" / f"trial_{idx:05d}_lagrangian.npz")
            params = _manifest_path(root, row.get("params_path"), default=root / "params" / f"trial_{idx:05d}_params.npy")
            items.append(
                {
                    "traj_id": str(row.get("trial_uid", row.get("traj_id", f"plife_traj_{idx:05d}"))),
                    "trial_idx": int(row.get("trial_idx", idx)),
                    "optimized_run_idx": int(row.get("optimized_run_idx", idx)),
                    "candidate_kind": kind,
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "candidate_label": str(row.get("candidate_label", kind)),
                    "seed_x": int(row.get("seed_x", row.get("trajectory_seed", -1))),
                    "metric_seed": int(row.get("metric_seed", 900_000 + idx)),
                    "lagrangian_path": lagrangian,
                    "params_path": params,
                }
            )
    else:
        trial_dir = root / "trial_data"
        json_paths = sorted(trial_dir.glob("trial_*.json")) if trial_dir.exists() else []
        for idx, path in enumerate(p for p in json_paths if not p.name.endswith("_summary.json")):
            row = json.loads(path.read_text())
            kind = _canonical_kind(row.get("candidate_kind", "optimized"), row.get("candidate_label", None))
            lagrangian = _manifest_path(root, row.get("lagrangian_path"), default=path.with_name(path.stem + "_lagrangian.npz"))
            params = _manifest_path(root, row.get("params_path"), default=root / "params" / f"trial_{int(row.get('trial_idx', idx)):05d}_params.npy")
            items.append(
                {
                    "traj_id": str(row.get("trial_uid", f"plife_traj_{idx:05d}")),
                    "trial_idx": int(row.get("trial_idx", idx)),
                    "optimized_run_idx": int(row.get("optimized_run_idx", idx)),
                    "candidate_kind": kind,
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "candidate_label": str(row.get("candidate_label", kind)),
                    "seed_x": int(row.get("seed_x", row.get("trajectory_seed", -1))),
                    "metric_seed": int(row.get("metric_seed", 900_000 + idx)),
                    "lagrangian_path": lagrangian,
                    "params_path": params,
                }
            )
    return [item for item in items if str(item.get("candidate_kind")) == "optimized"]


def _load_lagrangian(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        key = _primary_lagrangian_xy_key(data)
        xy = np.asarray(data[key], dtype=np.float32)
        sample_every = int(np.asarray(data["sample_every_steps"]).item()) if "sample_every_steps" in data.files else 1
        start = int(np.asarray(data["trajectory_start_steps"]).item()) if "trajectory_start_steps" in data.files else 0
        end = int(np.asarray(data["trajectory_end_steps"]).item()) if "trajectory_end_steps" in data.files else start + xy.shape[0] * sample_every
        if "xy_late_sample_steps" in data.files:
            sample_steps = np.asarray(data["xy_late_sample_steps"], dtype=np.int64).reshape(-1)
        elif "sample_offsets_steps" in data.files:
            sample_steps = start + np.asarray(data["sample_offsets_steps"], dtype=np.int64).reshape(-1)
        else:
            sample_steps = start + sample_every * np.arange(1, xy.shape[0] + 1, dtype=np.int64)
    return {"xy": xy, "sample_every": sample_every, "start": start, "end": end, "sample_steps": sample_steps, "trajectory_key": key}


def _processed_delta_h(delta_h_map: np.ndarray, metric_cfg: dict[str, Any]) -> np.ndarray:
    x = np.asarray(delta_h_map, dtype=np.float64)
    mode = str(metric_cfg.get("preprocess_mode", "clip")).strip().lower()
    if mode == "clip":
        out = np.maximum(x, 0.0)
    elif mode == "shift":
        out = x - np.nanmin(x, axis=1, keepdims=True)
    elif mode == "none":
        out = x.copy()
    else:
        out = np.maximum(x, 0.0)
    floor = float(metric_cfg.get("delta_h_floor", 0.0) or 0.0)
    if floor > 0.0:
        out = np.where(out >= floor, out, 0.0)
    return out


def _window_centers(info: dict[str, np.ndarray], metric_cfg: dict[str, Any], traj_start: int) -> np.ndarray:
    starts = np.asarray(info.get("window_start_steps", np.arange(np.asarray(info["delta_h_map"]).shape[1])), dtype=np.float64).reshape(-1)
    window = float(metric_cfg.get("window_size_steps", metric_cfg.get("window_size_frames", 0.0)) or 0.0)
    return float(traj_start) + starts + 0.5 * window


def _metric_timing_aliases(info: dict[str, np.ndarray], metric_cfg: dict[str, Any], lag: dict[str, Any]) -> dict[str, np.ndarray]:
    delta_h_map = np.asarray(info["delta_h_map"])
    sample_every = int(metric_cfg.get("sample_every_steps", lag.get("sample_every", 1)) or lag.get("sample_every", 1) or 1)
    sample_steps = np.asarray(lag.get("sample_steps", []), dtype=np.int64).reshape(-1)
    if sample_steps.size:
        step_offset = int(sample_steps[0])
    else:
        step_offset = int(lag.get("start", 0)) + sample_every
    base_step = max(0, step_offset - sample_every)
    starts_rel = np.asarray(info.get("window_start_steps", np.arange(delta_h_map.shape[1])), dtype=np.int64).reshape(-1)
    window_start_steps = starts_rel + int(base_step)
    window_size_steps = int(metric_cfg.get("window_size_steps", int(metric_cfg.get("window_size_frames", 1)) * sample_every))
    window_end_steps = window_start_steps + int(window_size_steps)
    window_center_steps = window_start_steps + int(window_size_steps // 2)
    tau_steps = np.asarray(info.get("tau_steps", metric_cfg.get("tau_steps_list", np.arange(delta_h_map.shape[0]))), dtype=np.int64).reshape(-1)
    selected_idx = int(np.asarray(info.get("tau_selected_idx", info.get("delta_h_selected_tau_idx", np.asarray(0)))).reshape(-1)[0])
    selected_tau = int(tau_steps[selected_idx]) if 0 <= selected_idx < tau_steps.size else int(tau_steps[0]) if tau_steps.size else 0
    return {
        "delta_h_tau_steps": tau_steps.astype(np.int32),
        "delta_h_tau_frames": np.asarray(info.get("tau_frames", np.maximum(1, np.round(tau_steps / max(1, sample_every))).astype(np.int32)), dtype=np.int32),
        "delta_h_selected_tau_idx": np.asarray(selected_idx, dtype=np.int32),
        "delta_h_selected_tau_steps": np.asarray(selected_tau, dtype=np.int32),
        "delta_h_window_start_steps": window_start_steps.astype(np.int64),
        "delta_h_window_end_steps": window_end_steps.astype(np.int64),
        "delta_h_window_center_steps": window_center_steps.astype(np.int64),
        "delta_h_window_size_steps": np.asarray(window_size_steps, dtype=np.int32),
        "delta_h_sample_every_steps": np.asarray(sample_every, dtype=np.int32),
    }


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    try:
        return np.asarray(data[key]).reshape(-1)[0].item()
    except Exception:
        return default


def _npz_json(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    raw = _npz_scalar(data, key, None)
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _load_plife_delta_h(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "delta_h_best" in data.files:
            dh = np.asarray(data["delta_h_best"], dtype=np.float64).reshape(-1)
        elif "delta_h_map" in data.files:
            dh_map = np.asarray(data["delta_h_map"], dtype=np.float64)
            selected = int(_npz_scalar(data, "delta_h_selected_tau_idx", _npz_scalar(data, "tau_selected_idx", 0)) or 0)
            dh = np.asarray(dh_map[selected], dtype=np.float64).reshape(-1)
        else:
            raise ValueError(f"{path} has neither delta_h_best nor delta_h_map.")
        if "delta_h_window_center_steps" in data.files:
            centers = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        else:
            starts = np.asarray(data["window_start_steps"] if "window_start_steps" in data.files else np.arange(dh.size), dtype=np.float64).reshape(-1)
            metric_cfg = _npz_json(data, "metric_config_json")
            sample_every = int(_npz_scalar(data, "sample_every_steps", metric_cfg.get("sample_every_steps", 1)) or 1)
            traj_start = int(_npz_scalar(data, "trajectory_start_steps", 0) or 0)
            window_size = int(metric_cfg.get("window_size_steps", int(metric_cfg.get("window_size_frames", 1)) * sample_every))
            centers = traj_start + starts + int(window_size // 2)
    n = min(int(dh.size), int(centers.size))
    if n == 0:
        raise ValueError(f"{path} has an empty Delta-H series.")
    return centers[:n], dh[:n]


def _load_plife_delta_h_energy(
    path: Path,
    *,
    min_remaining_steps: int | None = None,
    min_remaining_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if "delta_h_map" not in data.files:
            centers, dh = _load_plife_delta_h(path)
            return centers, dh, {
                "selection_energy_source": "delta_h_best_fallback",
                "admissible_tau_count": 1,
                "admissible_tau_steps": "",
            }
        dh_map = np.asarray(data["delta_h_map"], dtype=np.float64)
        if dh_map.ndim != 2:
            raise ValueError(f"{path} delta_h_map must be 2D, got {dh_map.shape}.")
        tau_steps = np.asarray(
            data["delta_h_tau_steps"] if "delta_h_tau_steps" in data.files else data["tau_steps"] if "tau_steps" in data.files else np.arange(dh_map.shape[0]),
            dtype=np.float64,
        ).reshape(-1)
        if dh_map.shape[0] != tau_steps.size and dh_map.shape[1] == tau_steps.size:
            dh_map = dh_map.T
        if dh_map.shape[0] != tau_steps.size:
            raise ValueError(f"{path} delta_h_map shape {dh_map.shape} is incompatible with tau grid size {tau_steps.size}.")
        if "delta_h_window_center_steps" in data.files:
            centers = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        else:
            centers, _dh = _load_plife_delta_h(path)
        metric_cfg = _npz_json(data, "metric_config_json")
        sample_every = int(_npz_scalar(data, "delta_h_sample_every_steps", metric_cfg.get("sample_every_steps", 1)) or 1)
        window_size_steps = _npz_scalar(data, "delta_h_window_size_steps", None)
        if window_size_steps is None:
            window_size_steps = int(metric_cfg.get("window_size_frames", 1)) * max(1, sample_every)
        window_size_steps = int(window_size_steps)
        if min_remaining_steps is not None:
            min_gap_steps = int(min_remaining_steps)
        elif min_remaining_samples is not None:
            min_gap_steps = int(min_remaining_samples) * max(1, sample_every)
        else:
            min_gap_steps = int(metric_cfg.get("m_min", 4)) * max(1, sample_every)
        admissible = np.isfinite(tau_steps) & (tau_steps < float(window_size_steps)) & (
            (float(window_size_steps) - tau_steps) >= float(min_gap_steps)
        )
        if not np.any(admissible):
            raise ValueError(
                f"{path} has no admissible tau for branching energy: "
                f"window_size_steps={window_size_steps}, min_remaining_steps={min_gap_steps}, tau_steps={tau_steps.tolist()}."
            )
        processed = _processed_delta_h(dh_map, metric_cfg)
        energy = np.nanmean(processed[admissible], axis=0)
        n = min(int(centers.size), int(energy.size))
        meta = {
            "selection_energy_source": "mean_tau_phi_delta_h_map",
            "selection_preprocess_mode": str(metric_cfg.get("preprocess_mode", "clip")),
            "selection_delta_h_floor": float(metric_cfg.get("delta_h_floor", 0.0) or 0.0),
            "selection_window_size_steps": int(window_size_steps),
            "selection_sample_every_steps": int(sample_every),
            "selection_min_remaining_steps": int(min_gap_steps),
            "admissible_tau_count": int(np.sum(admissible)),
            "admissible_tau_steps": ",".join(str(int(round(x))) for x in tau_steps[admissible]),
        }
    if n == 0:
        raise ValueError(f"{path} produced an empty branching energy profile.")
    return centers[:n], energy[:n], meta


def _pick_condition_indices(values: np.ndarray, centers: np.ndarray, *, n_high: int, n_mid: int, n_low: int, refractory: int) -> list[tuple[str, int]]:
    finite = np.isfinite(values) & np.isfinite(centers)
    order = np.where(finite)[0]
    if order.size == 0:
        return []
    ranked = order[np.argsort(values[order])]
    picks: list[tuple[str, int]] = []
    used_centers: list[float] = []

    def far_enough(idx: int) -> bool:
        if refractory <= 0:
            return True
        return all(abs(float(centers[idx]) - c) >= float(refractory) for c in used_centers)

    def add(condition: str, candidates: np.ndarray, n: int) -> None:
        for idx in candidates:
            if len([p for p in picks if p[0] == condition]) >= int(n):
                break
            if far_enough(int(idx)):
                picks.append((condition, int(idx)))
                used_centers.append(float(centers[int(idx)]))

    add("high", ranked[::-1], n_high)
    mid_value = float(np.nanmedian(values[finite]))
    mid_ranked = order[np.argsort(np.abs(values[order] - mid_value))]
    add("mid", mid_ranked, n_mid)
    add("low", ranked, n_low)
    return picks


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if rx.size < 2 or float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _branching_metric_mode(raw: Any) -> str:
    value = str(raw or "clip_chamfer").strip().lower().replace("-", "_")
    aliases = {
        "clip": "clip_chamfer",
        "clip_cloud": "clip_chamfer",
        "clip_chamfer_cosine": "clip_chamfer",
        "future_clip_chamfer_cosine": "clip_chamfer",
        "position": "position_chamfer",
        "position_chamfer": "position_chamfer",
        "particle_position_chamfer": "position_chamfer",
        "future_position_chamfer": "position_chamfer",
    }
    value = aliases.get(value, value)
    if value not in {"clip_chamfer", "position_chamfer"}:
        raise ValueError(f"Unknown PLife++ C2 branching_metric {raw!r}; use 'clip_chamfer' or 'position_chamfer'.")
    return value


def _correlation_summary(rows: list[dict[str, Any]], *, metric_name: str) -> dict[str, Any]:
    x = np.asarray([row["delta_h"] for row in rows], dtype=np.float64)
    y = np.asarray([row["branching_score"] for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) >= 2 and float(np.std(x[finite])) > 1e-12 and float(np.std(y[finite])) > 1e-12:
        pearson = float(np.corrcoef(x[finite], y[finite])[0, 1])
        spearman = _rank_correlation(x[finite], y[finite])
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {
        "claim": "C2_PLIFE_PLUS",
        "branching_metric": metric_name,
        "n": int(np.sum(finite)),
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def _normalize_embeddings(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _embedding_chamfer_cosine(z_a: np.ndarray, z_b: np.ndarray) -> float:
    a = _normalize_embeddings(z_a)
    b = _normalize_embeddings(z_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[0] < 1:
        return float("nan")
    d = 1.0 - (a @ b.T)
    return float(0.5 * (np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))))


def _frame_subset(arr: np.ndarray, max_frames: int) -> np.ndarray:
    frames = np.asarray(arr, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError(f"expected RGB frame stack with shape (T,H,W,3), got {frames.shape}")
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    if frames.shape[-1] < 3:
        frames = np.repeat(frames, int(math.ceil(3 / max(1, frames.shape[-1]))), axis=-1)[..., :3]
    if max_frames > 0 and frames.shape[0] > int(max_frames):
        idx = np.linspace(0, frames.shape[0] - 1, int(max_frames)).astype(int)
        frames = frames[idx]
    return np.clip(frames, 0.0, 1.0).astype(np.float32)


def _clip_embedding_cache_path(path: Path, *, cache_dir: Path, foundation_model: str, max_frames: int) -> Path:
    stat = path.stat()
    payload = json.dumps(
        {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "foundation_model": str(foundation_model),
            "max_frames": int(max_frames),
            "version": "plife_c2_branch_clip_embeddings_v1",
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{path.stem}_{digest}.npz"


def _load_or_compute_branch_clip_embeddings(
    path: Path,
    *,
    fm: Any,
    cache_dir: Path,
    foundation_model: str,
    max_frames: int,
    force: bool,
) -> tuple[np.ndarray, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _clip_embedding_cache_path(path, cache_dir=cache_dir, foundation_model=foundation_model, max_frames=max_frames)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            return np.asarray(data["z"], dtype=np.float32), cache_path
    with np.load(path, allow_pickle=False) as data:
        if "rgb_future" not in data.files:
            raise ValueError(f"{path} has no rgb_future; rerun PLife++ C2 branch simulation")
        frames = _frame_subset(np.asarray(data["rgb_future"], dtype=np.float32), max_frames=max_frames)
    zs: list[np.ndarray] = []
    for frame in frames:
        z = jax.device_get(fm.embed_img(frame))
        zs.append(np.asarray(z, dtype=np.float32).reshape(-1))
    z_arr = _normalize_embeddings(np.stack(zs, axis=0)).astype(np.float32)
    np.savez_compressed(
        cache_path,
        z=z_arr,
        branch_output=np.asarray(str(path)),
        foundation_model=np.asarray(str(foundation_model)),
        max_frames=np.asarray(int(max_frames), dtype=np.int32),
    )
    return z_arr, cache_path


def _future_clip_chamfer(
    branch_paths: list[Path],
    *,
    fm: Any,
    cache_dir: Path,
    foundation_model: str,
    max_frames: int,
    force_cache: bool,
) -> tuple[float, dict[str, Any]]:
    embeddings: list[np.ndarray] = []
    cache_paths: list[str] = []
    for path in branch_paths:
        try:
            z, cache_path = _load_or_compute_branch_clip_embeddings(
                path,
                fm=fm,
                cache_dir=cache_dir,
                foundation_model=foundation_model,
                max_frames=max_frames,
                force=force_cache,
            )
        except Exception:
            continue
        embeddings.append(z)
        cache_paths.append(str(cache_path))
    if len(embeddings) < 2:
        return float("nan"), {"metric": "future_clip_chamfer_cosine", "n_branches": len(embeddings), "n_pairs": 0}
    vals = [_embedding_chamfer_cosine(embeddings[i], embeddings[j]) for i, j in combinations(range(len(embeddings)), 2)]
    vals = [float(v) for v in vals if np.isfinite(v)]
    return (
        float(np.mean(vals)) if vals else float("nan"),
        {
            "metric": "future_clip_chamfer_cosine",
            "n_branches": len(embeddings),
            "n_pairs": len(vals),
            "foundation_model": str(foundation_model),
            "max_future_frames": int(max_frames),
            "clip_embedding_cache_n": len(cache_paths),
        },
    )


def _periodic_delta(dx: np.ndarray, domain: float = 1.0) -> np.ndarray:
    return (dx + 0.5 * domain) % domain - 0.5 * domain


def _chamfer_frame(a: np.ndarray, b: np.ndarray, *, domain: float, max_particles: int) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.shape[0], bb.shape[0], int(max_particles))
    if n <= 0:
        return float("nan")
    aa = aa[:n]
    bb = bb[:n]
    d = _periodic_delta(aa[:, None, :] - bb[None, :, :], domain=domain)
    dist = np.sqrt(np.sum(d * d, axis=-1))
    return float(0.5 * (np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))))


def _future_chamfer(a: np.ndarray, b: np.ndarray, *, domain: float, max_particles: int) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n_frames = min(aa.shape[0], bb.shape[0])
    if n_frames <= 0:
        return float("nan")
    vals = [_chamfer_frame(aa[i], bb[i], domain=domain, max_particles=max_particles) for i in range(n_frames)]
    return float(np.nanmean(vals))


def _branch_output_path(branch_root: Path, row: dict[str, Any], rep: int) -> Path:
    traj = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(row["traj_id"]))
    return branch_root / traj / f"point_{int(row['point_id']):04d}" / f"branch_{int(rep):03d}.npz"


def _branch_output_npz_from_row(branch_root: Path, row: dict[str, Any]) -> Path:
    raw = str(row.get("branch_output_path", "")).strip()
    if raw:
        return Path(raw)
    branch_dir_raw = str(row.get("branch_dir", "")).strip()
    if branch_dir_raw:
        branch_dir = Path(branch_dir_raw)
        if branch_dir.suffix == ".npz":
            return branch_dir
        return branch_dir / "branch_output.npz"
    rep = int(float(row.get("branch_id", row.get("branch_rep", 0)) or 0))
    return _branch_output_path(branch_root, row, rep)


def _branch_plan_meta_current(plan_path: Path) -> tuple[bool, str]:
    if not plan_path.exists():
        return False, f"missing branch plan {plan_path}"
    meta_path = plan_path.with_name("branch_plan_meta.json")
    if not meta_path.exists():
        return False, f"missing branch plan metadata {meta_path}"
    try:
        rows = read_csv(plan_path)
        with meta_path.open("r") as f:
            meta = json.load(f)
    except Exception as exc:
        return False, f"cannot read branch plan metadata: {type(exc).__name__}: {exc}"
    if str(meta.get("branch_plan_version", "")) != PLIFE_BRANCH_PLAN_VERSION:
        return False, f"stale branch plan version {meta.get('branch_plan_version')!r}; expected {PLIFE_BRANCH_PLAN_VERSION}"
    current_hash = sha256_text(stable_json(rows))
    if str(meta.get("branch_plan_rows_hash", "")) != current_hash:
        return False, "branch_plan.csv no longer matches branch_plan_meta.json"
    metrics_hash = _metrics_manifest_identity_hash(plan_path.parent)
    if metrics_hash and str(meta.get("metrics_manifest_identity_hash", "")) != metrics_hash:
        return False, "branch_plan.csv was built from stale PLife++ C2 metrics_manifest/metrics files"
    return True, "ok"


def _metrics_manifest_identity_hash(out_dir: Path) -> str:
    manifest_path = out_dir / "metrics_manifest.csv"
    if not manifest_path.exists():
        return ""
    records: list[dict[str, Any]] = []
    for row in read_csv(manifest_path):
        path = Path(str(row.get("metrics_path", "")))
        rec: dict[str, Any] = {
            "traj_id": row.get("traj_id", ""),
            "status": row.get("status", ""),
            "metrics_path": str(path),
        }
        if path.exists():
            stat = path.stat()
            rec["size"] = int(stat.st_size)
            rec["mtime_ns"] = int(stat.st_mtime_ns)
        records.append(rec)
    return sha256_text(stable_json(records))


def _score_branch_plan(
    *,
    plan_rows: list[dict[str, Any]],
    branch_root: Path,
    out_dir: Path,
    c2_cfg: Any,
    branch_cfg: Any,
    metric_mode: str,
    clip_foundation_model: str,
    clip_cache_dir: Path,
    clip_max_frames: int,
    domain: float,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    max_particles = int(_get(c2_cfg, "divergence_max_particles", 128))
    expected_branch_outputs = 0
    existing_branch_outputs = 0
    missing_min_branch_points = 0
    invalid_score_points = 0
    missing_rgb_outputs = 0
    clip_fm = None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in plan_rows:
        key = (str(row.get("traj_id", "")), str(row.get("point_id", row.get("pair_id", ""))), str(row.get("condition", "sampled")))
        groups.setdefault(key, []).append(row)

    for point_idx, ((traj_id, point_id, condition), rows) in enumerate(sorted(groups.items()), start=1):
        branch_paths: list[Path] = []
        for row in rows:
            expected_branch_outputs += 1
            path = _branch_output_npz_from_row(branch_root, row)
            if not path.exists():
                continue
            existing_branch_outputs += 1
            branch_paths.append(path)
        if _progress_now(point_idx, len(groups), every=10):
            log_event(
                f"PLife++ C2 score point {point_idx}/{len(groups)} point={point_id} "
                f"traj={traj_id} condition={condition} existing_branches={len(branch_paths)}/{len(rows)}",
                component="c2-plife",
            )
        if len(branch_paths) < 2:
            missing_min_branch_points += 1
            continue
        if metric_mode == "clip_chamfer":
            for path in branch_paths:
                try:
                    with np.load(path, allow_pickle=False) as data:
                        if "rgb_future" not in data.files:
                            missing_rgb_outputs += 1
                except Exception:
                    missing_rgb_outputs += 1
            if clip_fm is None:
                import foundation_models

                log_event(f"PLife++ C2 loading foundation model {clip_foundation_model!r} for branch CLIP-Chamfer", component="c2-plife")
                clip_fm = foundation_models.create_foundation_model(clip_foundation_model)
            score, detail = _future_clip_chamfer(
                branch_paths,
                fm=clip_fm,
                cache_dir=clip_cache_dir,
                foundation_model=clip_foundation_model,
                max_frames=clip_max_frames,
                force_cache=force,
            )
        else:
            branches = []
            for path in branch_paths:
                try:
                    with np.load(path, allow_pickle=False) as data:
                        branches.append(np.asarray(data["xy_future"], dtype=np.float32))
                except Exception:
                    continue
            vals = [
                _future_chamfer(branches[i], branches[j], domain=domain, max_particles=max_particles)
                for i, j in combinations(range(len(branches)), 2)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            score = float(np.mean(vals)) if vals else float("nan")
            detail = {"metric": "future_position_chamfer", "n_branches": len(branches), "n_pairs": len(vals)}
        if not np.isfinite(score):
            invalid_score_points += 1
            continue
        used = branch_paths
        first = rows[0]
        step_vals = [float(row.get("step", "nan")) for row in rows]
        dh_vals = [float(row.get("delta_h", "nan")) for row in rows]
        score_row = {
            "traj_id": traj_id,
            "point_id": int(float(point_id)),
            "pair_id": int(float(first.get("pair_id", point_id) or point_id)),
            "condition": condition,
            "step": float(np.nanmedian(step_vals)),
            "delta_h": float(np.nanmedian(dh_vals)),
            "branching_score": float(score),
            "branching_metric": str(detail.get("metric", metric_mode)),
            "n_branches": int(detail.get("n_branches", len(used))),
            "n_branch_pairs": int(detail.get("n_pairs", 0)),
        }
        for key, value in detail.items():
            if key not in score_row:
                score_row[str(key)] = value
        score_rows.append(score_row)

    metric_name = "future_clip_chamfer_cosine" if metric_mode == "clip_chamfer" else "future_position_chamfer"
    corr = _correlation_summary(score_rows, metric_name=metric_name)
    write_csv(out_dir / "branching_scores.csv", score_rows, fieldnames=BRANCHING_SCORE_COLUMNS)
    write_csv(out_dir / "branching_delta_h_correlation.csv", [corr])
    score_summary = {
        "n_plan_rows": len(plan_rows),
        "n_plan_points": len(groups),
        "n_scores": len(score_rows),
        "n_expected_branch_outputs": int(expected_branch_outputs),
        "n_existing_branch_outputs": int(existing_branch_outputs),
        "n_points_with_fewer_than_two_branches": int(missing_min_branch_points),
        "n_points_with_invalid_score": int(invalid_score_points),
        "n_existing_branch_outputs_missing_rgb_future": int(missing_rgb_outputs),
        "branching_metric_mode": metric_mode,
        "correlation": corr,
    }
    return score_rows, score_summary


def metrics(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    out_dir = _out_dir(cfg, output_root)
    c2_cfg = _plife_c2_cfg(cfg)
    log_event(
        f"PLife++ C2 metrics start smoke={smoke} force={force} output={out_dir}",
        component="c2-plife",
    )
    enabled = bool(_get(c2_cfg, "enabled", True))
    if not enabled:
        summary = {"status": "disabled"}
        write_json(out_dir / "c2_plife_plus_metrics_summary.json", summary)
        log_event("PLife++ C2 metrics disabled by config", component="c2-plife")
        return summary
    root = _trajectory_root(cfg, smoke=smoke)
    required = bool(_get(c2_cfg, "required", False))
    if root is None or not root.exists():
        if required:
            raise FileNotFoundError(f"PLife++ C2 trajectory root not found: {root}")
        summary = {"status": "skipped", "reason": f"missing trajectory root {root}"}
        write_json(out_dir / "c2_plife_plus_metrics_summary.json", summary)
        log_event(f"PLife++ C2 metrics skipped: {summary['reason']}", component="c2-plife")
        return summary

    items = _iter_trajectory_items(root)
    max_trajectories = int(_get(c2_cfg, "max_trajectories", _get(_get(cfg.get("c2", {}), "branching", {}), "max_trajectories", 9)))
    items = items[:max(0, max_trajectories)]
    log_event(
        f"PLife++ C2 metrics discovered n_optimized={len(items)} max_trajectories={max_trajectories} root={root}",
        component="c2-plife",
    )
    if not items:
        summary = {"status": "skipped", "reason": f"no optimized PLife++ trajectories under {root}"}
        write_json(out_dir / "c2_plife_plus_metrics_summary.json", summary)
        log_event(f"PLife++ C2 metrics skipped: {summary['reason']}", component="c2-plife")
        return summary

    metrics_dir = ensure_dir(out_dir / "metrics")
    branch_root = _branch_root(cfg, output_root)
    metric_raw = _metric_cfg_raw(cfg)
    branch_cfg = _get(cfg.get("c2", {}), "branching", {})
    domain = float(_get(c2_cfg, "domain_size", 1.0))
    metric_mode = _branching_metric_mode(_get(c2_cfg, "branching_metric", "clip_chamfer"))
    clip_foundation_model = str(_get(c2_cfg, "clip_foundation_model", _get(branch_cfg, "clip_foundation_model", "clip")))
    clip_cache_raw = _get(c2_cfg, "clip_embedding_cache_dir", None)
    clip_cache_dir = ensure_dir(resolve_path(clip_cache_raw) if clip_cache_raw else out_dir / "clip_embedding_cache")
    clip_max_frames = int(_get(c2_cfg, "clip_max_future_frames", _get(branch_cfg, "future_max_frames", 32)))
    log_event(
        "PLife++ C2 metrics config "
        f"metric={metric_mode} branch_root={branch_root} clip_max_frames={clip_max_frames}",
        component="c2-plife",
    )

    manifest_rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items, start=1):
        lag_path = Path(item["lagrangian_path"])
        if _progress_now(item_idx, len(items)):
            log_event(
                f"PLife++ C2 metric item {item_idx}/{len(items)} traj={item['traj_id']} lagrangian={lag_path}",
                component="c2-plife",
            )
        if not lag_path.exists():
            log_event(
                f"PLife++ C2 metric item missing lagrangian traj={item['traj_id']} path={lag_path}",
                component="c2-plife",
            )
            manifest_rows.append({**item, "status": "missing_lagrangian", "message": str(lag_path)})
            continue
        lag = _load_lagrangian(lag_path)
        metric_cfg = _metric_config_from_lagrangian(lag_path, metric_raw)
        metrics_path = metrics_dir / f"{item['traj_id']}_metrics.npz"
        if metrics_path.exists() and not force:
            if _progress_now(item_idx, len(items)):
                log_event(f"PLife++ C2 metric cache hit traj={item['traj_id']} metrics={metrics_path}", component="c2-plife")
            with np.load(metrics_path, allow_pickle=False) as data:
                info = {key: np.asarray(data[key]) for key in data.files}
        else:
            log_event(
                f"PLife++ C2 metric compute traj={item['traj_id']} T={lag['xy'].shape[0]} N={lag['xy'].shape[1]} "
                f"sample_every={lag['sample_every']} range={lag['start']}..{lag['end']} metrics={metrics_path}",
                component="c2-plife",
            )
            metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))
            info = _score_maps(metric_eval, int(item["metric_seed"]), np.asarray(lag["xy"], dtype=np.float32))
            timing_aliases = _metric_timing_aliases(info, metric_cfg, lag)
            np.savez_compressed(
                metrics_path,
                **info,
                **timing_aliases,
                metric_config_json=np.asarray(json.dumps(to_plain(metric_cfg), sort_keys=True)),
                lagrangian_path=np.asarray(str(lag_path)),
                params_path=np.asarray(str(item["params_path"])),
                trajectory_start_steps=np.asarray(int(lag["start"]), dtype=np.int32),
                trajectory_end_steps=np.asarray(int(lag["end"]), dtype=np.int32),
                sample_every_steps=np.asarray(int(lag["sample_every"]), dtype=np.int32),
            )
        if "delta_h_map" not in info:
            log_event(f"PLife++ C2 metric item has no delta_h_map traj={item['traj_id']} metrics={metrics_path}", component="c2-plife")
            manifest_rows.append({**item, "status": "missing_delta_h_map", "metrics_path": str(metrics_path)})
            continue
        n_windows = int(np.asarray(info["delta_h_map"]).shape[1]) if np.asarray(info["delta_h_map"]).ndim == 2 else 0
        manifest_rows.append(
            {
                **item,
                "status": "ok",
                "metrics_path": str(metrics_path),
                "n_points": n_windows,
                "sample_every_steps": int(lag["sample_every"]),
                "trajectory_start_steps": int(lag["start"]),
                "trajectory_end_steps": int(lag["end"]),
            }
        )

    write_csv(out_dir / "metrics_manifest.csv", manifest_rows, fieldnames=METRICS_MANIFEST_COLUMNS)
    log_event(
        f"PLife++ C2 wrote metrics manifest n_items={len(manifest_rows)} manifest={out_dir / 'metrics_manifest.csv'}",
        component="c2-plife",
    )

    plan_path = out_dir / "branch_plan.csv"
    plan_ok, plan_reason = _branch_plan_meta_current(plan_path)
    if plan_ok:
        plan_rows = read_csv(plan_path)
        score_rows, score_summary = _score_branch_plan(
            plan_rows=plan_rows,
            branch_root=branch_root,
            out_dir=out_dir,
            c2_cfg=c2_cfg,
            branch_cfg=branch_cfg,
            metric_mode=metric_mode,
            clip_foundation_model=clip_foundation_model,
            clip_cache_dir=clip_cache_dir,
            clip_max_frames=clip_max_frames,
            domain=domain,
            force=force,
        )
    else:
        score_rows = []
        metric_name = "future_clip_chamfer_cosine" if metric_mode == "clip_chamfer" else "future_position_chamfer"
        corr = _correlation_summary(score_rows, metric_name=metric_name)
        write_csv(out_dir / "branching_scores.csv", score_rows, fieldnames=BRANCHING_SCORE_COLUMNS)
        write_csv(out_dir / "branching_delta_h_correlation.csv", [corr])
        score_summary = {
            "n_plan_rows": 0,
            "n_plan_points": 0,
            "n_scores": 0,
            "n_expected_branch_outputs": 0,
            "n_existing_branch_outputs": 0,
            "n_points_with_fewer_than_two_branches": 0,
            "n_points_with_invalid_score": 0,
            "n_existing_branch_outputs_missing_rgb_future": 0,
            "branching_metric_mode": metric_mode,
            "correlation": corr,
            "score_skip_reason": plan_reason,
        }
        log_event(f"PLife++ C2 metrics branch scoring skipped: {plan_reason}", component="c2-plife")
    summary = {
        "status": "ok",
        "trajectory_root": str(root),
        "branch_root": str(branch_root),
        "n_metric_items": len(manifest_rows),
        **score_summary,
    }
    write_json(out_dir / "c2_plife_plus_metrics_summary.json", summary)
    log_event(
        "PLife++ C2 metrics done "
        f"n_plan_rows={summary['n_plan_rows']} n_plan_points={summary['n_plan_points']} n_scores={len(score_rows)} "
        f"branch_outputs={summary['n_existing_branch_outputs']}/{summary['n_expected_branch_outputs']} "
        f"points_lt2_branches={summary['n_points_with_fewer_than_two_branches']} "
        f"invalid_scores={summary['n_points_with_invalid_score']} "
        f"missing_rgb={summary['n_existing_branch_outputs_missing_rgb_future']} "
        f"summary={out_dir / 'c2_plife_plus_metrics_summary.json'}",
        component="c2-plife",
    )
    return summary


def _build_substrate(cfg: Any, *, smoke: bool):
    base_config = _base_config_path(cfg)
    if base_config is None or not base_config.exists():
        raise FileNotFoundError(f"PLife++ C2 base config not found: {base_config}")
    log_event(f"PLife++ C2 loading base substrate config {base_config}", component="c2-plife")
    base_cfg, _unused = _load_base_config(base_config)
    _apply_section_base_overrides(base_cfg, _simulation_section(cfg))
    if smoke:
        base_cfg.substrate.rollout_steps = min(int(base_cfg.substrate.rollout_steps), 96)
        base_cfg.substrate.n_particles = min(int(base_cfg.substrate.n_particles), 32)
    rollout_steps = _get(base_cfg.substrate, "rollout_steps", "?")
    n_particles = _get(base_cfg.substrate, "n_particles", "?")
    log_event(
        f"PLife++ C2 substrate settings rollout_steps={rollout_steps} n_particles={n_particles}",
        component="c2-plife",
    )
    flat = _flatten_base_config(base_cfg)
    args = SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))
    return _make_substrate(args)


def _advance_cache(substrate):
    cache: dict[int, Any] = {}

    def get(n_steps: int):
        n_steps = int(n_steps)
        if n_steps not in cache:
            def advance(rng, state, params):
                def body(carry, _):
                    rng_cur, st_cur = carry
                    rng_next, step_key = jax.random.split(rng_cur)
                    st_next = substrate.step_state(step_key, st_cur, params)
                    return (rng_next, st_next), None
                return jax.lax.scan(body, (rng, state), None, length=n_steps)[0]
            cache[n_steps] = jax.jit(advance)
        return cache[n_steps]

    return get


def _simulate_one_branch(
    *,
    substrate,
    params: np.ndarray,
    seed: int,
    branch_step: int,
    rep_seed: int,
    horizon_steps: int,
    sample_every: int,
    perturb: dict[str, float],
    render_img_size: int,
    include_initial_frame: bool = False,
) -> dict[str, np.ndarray]:
    params_j = jnp.asarray(params, dtype=jnp.float32)
    init_rng, init_key = jax.random.split(jax.random.PRNGKey(int(seed)), 2)
    state = substrate.init_state(init_key, params_j)
    rng = init_rng
    advance = _advance_cache(substrate)
    chunk = 512
    remaining = int(branch_step)
    while remaining > 0:
        n = min(chunk, remaining)
        rng, state = advance(n)(rng, state, params_j)
        remaining -= n

    state_np = {key: np.asarray(value) for key, value in state.items()}
    x_frames = []
    c_frames = []
    rgb_frames = []
    if include_initial_frame:
        x_frames.append(np.asarray(state_np["x"], dtype=np.float32))
        if "c" in state_np:
            c_frames.append(np.asarray(state_np["c"], dtype=np.float32))
        if render_img_size > 0:
            rgb = substrate.render_state(state, params_j, img_size=int(render_img_size))
            rgb_frames.append(np.asarray(jax.device_get(rgb), dtype=np.float32))

    prng = np.random.default_rng(int(rep_seed))
    if float(perturb.get("x_std", 0.0)) > 0.0 and "x" in state_np:
        state_np["x"] = (state_np["x"] + prng.normal(0.0, float(perturb["x_std"]), size=state_np["x"].shape)) % 1.0
    if float(perturb.get("v_std", 0.0)) > 0.0 and "v" in state_np:
        state_np["v"] = state_np["v"] + prng.normal(0.0, float(perturb["v_std"]), size=state_np["v"].shape)
    if float(perturb.get("c_std", 0.0)) > 0.0 and "c" in state_np:
        c = state_np["c"] + prng.normal(0.0, float(perturb["c_std"]), size=state_np["c"].shape)
        state_np["c"] = c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)
    state = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in state_np.items()}

    remaining = int(horizon_steps)
    sample_every = max(1, int(sample_every))
    while remaining > 0:
        n = min(sample_every, remaining)
        rng, state = advance(n)(rng, state, params_j)
        x_frames.append(np.asarray(jax.device_get(state["x"]), dtype=np.float32))
        if "c" in state:
            c_frames.append(np.asarray(jax.device_get(state["c"]), dtype=np.float32))
        if render_img_size > 0:
            rgb = substrate.render_state(state, params_j, img_size=int(render_img_size))
            rgb_frames.append(np.asarray(jax.device_get(rgb), dtype=np.float32))
        remaining -= n
    out = {
        "xy_future": np.stack(x_frames, axis=0) if x_frames else np.zeros((0, int(state_np["x"].shape[0]), 2), dtype=np.float32),
    }
    if c_frames:
        out["c_future"] = np.stack(c_frames, axis=0).astype(np.float32)
    if rgb_frames:
        out["rgb_future"] = np.stack(rgb_frames, axis=0).astype(np.float32)
    return out


def _safe_path_id(value: Any) -> str:
    text = str(value)
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return out or "traj"


def _nearest_window_idx(centers: np.ndarray, step: int) -> int:
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if c.size == 0:
        return 0
    return int(np.argmin(np.abs(c - float(step))))


def _int_or(value: Any, default: int) -> int:
    try:
        x = float(value)
        if not np.isfinite(x):
            return int(default)
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _write_branch_plan_with_meta(out_dir: Path, rows: list[dict[str, Any]], *, matching: str, config_summary: dict[str, Any]) -> Path:
    plan_path = out_dir / "branch_plan.csv"
    write_csv(plan_path, rows, fieldnames=BRANCH_PLAN_COLUMNS)
    csv_rows = read_csv(plan_path)
    meta = {
        "branch_plan_version": PLIFE_BRANCH_PLAN_VERSION,
        "branch_plan_rows_hash": sha256_text(stable_json(csv_rows)),
        "metrics_manifest_identity_hash": _metrics_manifest_identity_hash(out_dir),
        "matching": matching,
        "branching_config": config_summary,
    }
    write_json(out_dir / "branch_plan_meta.json", meta)
    return plan_path


def _build_branch_plan_rows(cfg: Any, output_root: Path, out_dir: Path, branch_root: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    c2_cfg = _plife_c2_cfg(cfg)
    branch_cfg = _get(cfg.get("c2", {}), "branching", {})
    manifest_path = out_dir / "metrics_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing PLife++ C2 metrics manifest {manifest_path}; run PLife++ C2 metrics first")
    manifest_rows = [row for row in read_csv(manifest_path) if str(row.get("status", "")) == "ok"]
    selection_mode = str(_get(c2_cfg, "selection_mode", _get(branch_cfg, "selection_mode", "tau_mean_ranked_high_low")))
    n_high = int(_get(c2_cfg, "n_high", _get(branch_cfg, "n_high", 5)))
    n_mid = int(_get(c2_cfg, "n_mid", _get(branch_cfg, "n_mid", 5)))
    n_low = int(_get(c2_cfg, "n_low", _get(branch_cfg, "n_low", 5)))
    m_pairs = int(_get(c2_cfg, "m_pairs", _get(branch_cfg, "m_pairs", 5)))
    selection_seed = int(_get(c2_cfg, "selection_seed", _get(branch_cfg, "selection_seed", 12345)))
    n_points_per_trajectory = int(_get(c2_cfg, "n_points_per_trajectory", _get(branch_cfg, "n_points_per_trajectory", max(1, 2 * m_pairs))))
    point_quantiles = _as_float_list(_get(c2_cfg, "point_quantiles", _get(branch_cfg, "point_quantiles", None)))
    point_quantile_min = float(_get(c2_cfg, "point_quantile_min", _get(branch_cfg, "point_quantile_min", 0.05)))
    point_quantile_max = float(_get(c2_cfg, "point_quantile_max", _get(branch_cfg, "point_quantile_max", 0.95)))
    high_quantile = float(_get(c2_cfg, "high_quantile", _get(branch_cfg, "high_quantile", 0.8)))
    low_quantile = float(_get(c2_cfg, "low_quantile", _get(branch_cfg, "low_quantile", 0.2)))
    mid_quantile_low = float(_get(c2_cfg, "mid_quantile_low", _get(branch_cfg, "mid_quantile_low", 0.4)))
    mid_quantile_high = float(_get(c2_cfg, "mid_quantile_high", _get(branch_cfg, "mid_quantile_high", 0.6)))
    refractory_steps = int(_get(c2_cfg, "refractory_steps", _get(branch_cfg, "refractory_steps", 0)))
    min_branch_step = int(_get(c2_cfg, "min_branch_step", _get(c2_cfg, "selection_min_step", 0)))
    horizon_steps = int(_get(c2_cfg, "horizon_steps", _get(branch_cfg, "horizon_steps", 1000)))
    branches_per_time = int(_get(c2_cfg, "branches_per_time", _get(branch_cfg, "branches_per_time", 3)))
    energy_min_remaining_steps_raw = _get(c2_cfg, "energy_min_remaining_steps", _get(branch_cfg, "energy_min_remaining_steps", None))
    energy_min_remaining_samples_raw = _get(c2_cfg, "energy_min_samples", _get(branch_cfg, "energy_min_samples", None))
    energy_min_remaining_steps = None if energy_min_remaining_steps_raw is None else int(energy_min_remaining_steps_raw)
    energy_min_remaining_samples = None if energy_min_remaining_samples_raw is None else int(energy_min_remaining_samples_raw)
    perturb_cfg = _get(c2_cfg, "perturb", {})
    perturb_x_std = float(_get(perturb_cfg, "x_std", 0.003))
    perturb_v_std = float(_get(perturb_cfg, "v_std", 0.0))
    perturb_c_std = float(_get(perturb_cfg, "c_std", 0.01))

    config_summary = {
        "selection_mode": selection_mode,
        "selection_seed": selection_seed,
        "n_high": n_high,
        "n_mid": n_mid,
        "n_low": n_low,
        "m_pairs": m_pairs,
        "n_points_per_trajectory": n_points_per_trajectory,
        "point_quantiles": point_quantiles,
        "point_quantile_min": point_quantile_min,
        "point_quantile_max": point_quantile_max,
        "high_quantile": high_quantile,
        "mid_quantile_low": mid_quantile_low,
        "mid_quantile_high": mid_quantile_high,
        "low_quantile": low_quantile,
        "refractory_steps": refractory_steps,
        "min_branch_step": min_branch_step,
        "horizon_steps": horizon_steps,
        "branches_per_time": branches_per_time,
        "energy_min_remaining_steps": energy_min_remaining_steps,
        "energy_min_samples": energy_min_remaining_samples,
        "perturb_x_std": perturb_x_std,
        "perturb_v_std": perturb_v_std,
        "perturb_c_std": perturb_c_std,
    }

    ranked: list[tuple[float, int, dict[str, str], np.ndarray, np.ndarray, dict[str, Any]]] = []
    for item_idx, row in enumerate(manifest_rows):
        metrics_path = Path(str(row.get("metrics_path", "")))
        if not metrics_path.exists():
            continue
        if _is_ranked_high_low_mode(selection_mode):
            centers, energy, selection_meta = _load_plife_delta_h_energy(
                metrics_path,
                min_remaining_steps=energy_min_remaining_steps,
                min_remaining_samples=energy_min_remaining_samples,
            )
            rank_score = float(np.nanmean(energy))
        else:
            centers, energy = _load_plife_delta_h(metrics_path)
            selection_meta = {}
            rank_score = float(np.nanmax(energy))
        ranked.append((rank_score, item_idx, row, centers, energy, selection_meta))
    ranked.sort(key=lambda x: -x[0])
    max_trajectories = int(_get(c2_cfg, "max_trajectories", _get(branch_cfg, "max_trajectories", len(ranked))))
    ranked = ranked[: max(0, max_trajectories)]
    rows: list[dict[str, Any]] = []
    global_point_id = 0
    matching_used = "none"

    log_event(
        f"PLife++ C2 branch planning n_metric_items={len(ranked)} selection_mode={selection_mode} "
        f"n_high={n_high} n_mid={n_mid} n_low={n_low} branches_per_time={branches_per_time}",
        component="c2-plife",
    )
    for traj_order, (_rank, _item_idx, item, centers, energy, selection_meta) in enumerate(ranked):
        traj_id = str(item.get("traj_id", Path(str(item.get("metrics_path", ""))).stem))
        trajectory_end_raw = item.get("trajectory_end_steps", "")
        trajectory_end = _int_or(trajectory_end_raw, -1)
        trajectory_end = None if trajectory_end < 0 else trajectory_end
        if _is_ranked_high_low_mode(selection_mode):
            points, select_summary = _select_ranked_high_low_points(
                centers=centers,
                energy=energy,
                covariates={},
                n_high=n_high,
                n_low=n_low,
                n_mid=n_mid,
                q_high=high_quantile,
                q_low=low_quantile,
                q_mid_low=mid_quantile_low,
                q_mid_high=mid_quantile_high,
                horizon_steps=horizon_steps,
                trajectory_end_step=trajectory_end,
                seed=selection_seed + 10007 * traj_order,
                min_step=min_branch_step,
                energy_meta=selection_meta,
            )
            log_event(
                f"PLife++ C2 branch planning traj={traj_id} selected_high={select_summary.get('n_high_selected', 0)} "
                f"selected_mid={select_summary.get('n_mid_selected', 0)} selected_low={select_summary.get('n_low_selected', 0)}",
                component="c2-plife",
            )
            matching = "mean_tau_phi_delta_h_quantile_rank_sampling_without_replacement"
            matching_used = matching
        elif _is_delta_h_sweep_mode(selection_mode):
            points = _select_delta_h_points(
                centers=centers,
                dh=energy,
                covariates={},
                n_points=n_points_per_trajectory,
                refractory_steps=refractory_steps,
                quantiles=point_quantiles or None,
                quantile_min=point_quantile_min,
                quantile_max=point_quantile_max,
                min_step=min_branch_step,
            )
            matching = "delta_h_quantile_sweep"
            matching_used = matching
            log_event(f"PLife++ C2 branch planning traj={traj_id} selected_points={len(points)}", component="c2-plife")
        else:
            points = _select_delta_h_points(
                centers=centers,
                dh=energy,
                covariates={},
                n_points=max(1, n_high + n_mid + n_low),
                refractory_steps=refractory_steps,
                quantile_min=point_quantile_min,
                quantile_max=point_quantile_max,
                min_step=min_branch_step,
            )
            matching = "delta_h_quantile_sweep_fallback"
            matching_used = matching
            log_event(f"PLife++ C2 branch planning fallback traj={traj_id} selected_points={len(points)}", component="c2-plife")

        for point in points:
            local_point_id = int(point.get("point_id", global_point_id))
            condition = str(point.get("condition", "sampled"))
            step = int(point.get("step", point.get("window_center_step", 0)))
            window_idx = int(point.get("window_index", _nearest_window_idx(centers, step)))
            delta_h = float(point.get("delta_h", point.get("delta_h_energy", np.nan)))
            pair_id = int(point.get("pair_id", local_point_id))
            try:
                target_q_value = float(point.get("target_quantile", 0.5))
            except (TypeError, ValueError):
                target_q_value = float("nan")
            q_tag = int(round(1000.0 * target_q_value)) if np.isfinite(target_q_value) else global_point_id
            point_dir = branch_root / _safe_path_id(traj_id) / f"point_{global_point_id:04d}_{condition}_q_{q_tag:04d}_step_{step}"
            for branch_id in range(branches_per_time):
                condition_offset = {"high": 0, "mid": 3967, "low": 7919, "sampled": 12347}.get(condition, 12347)
                branch_seed = int(selection_seed + 1000003 * traj_order + 1009 * global_point_id + 131 * branch_id + condition_offset)
                branch_dir = point_dir / f"branch_{branch_id:02d}"
                branch_output = branch_dir / "branch_output.npz"
                rows.append(
                    {
                        "point_id": int(global_point_id),
                        "pair_id": int(pair_id),
                        "traj_id": traj_id,
                        "trial_idx": _int_or(item.get("trial_idx", -1), -1),
                        "optimized_run_idx": _int_or(item.get("optimized_run_idx", -1), -1),
                        "condition": condition,
                        "window_idx": int(window_idx),
                        "step": int(step),
                        "requested_step": int(step),
                        "step_snap_delta": 0,
                        "delta_h": float(delta_h),
                        "delta_h_energy": float(point.get("delta_h_energy", delta_h)),
                        "target_quantile": point.get("target_quantile", ""),
                        "delta_h_quantile": point.get("delta_h_quantile", ""),
                        "delta_h_quantile_rank": point.get("delta_h_quantile_rank", ""),
                        "selection_mode": selection_mode,
                        "selection_method": point.get("selection_method", matching),
                        "metrics_path": str(item.get("metrics_path", "")),
                        "lagrangian_path": str(item.get("lagrangian_path", "")),
                        "params_path": str(item.get("params_path", "")),
                        "seed_x": _int_or(item.get("seed_x", -1), -1),
                        "sample_every_steps": _int_or(item.get("sample_every_steps", 1), 1),
                        "trajectory_start_steps": _int_or(item.get("trajectory_start_steps", 0), 0),
                        "trajectory_end_steps": _int_or(item.get("trajectory_end_steps", 0), 0),
                        "horizon_steps": int(horizon_steps),
                        "branch_id": int(branch_id),
                        "branch_seed": int(branch_seed),
                        "branch_dir": str(branch_dir),
                        "branch_output_path": str(branch_output),
                        "perturb_x_std": float(perturb_x_std),
                        "perturb_v_std": float(perturb_v_std),
                        "perturb_c_std": float(perturb_c_std),
                        "status": "planned",
                        "command": "",
                    }
                )
            global_point_id += 1
    return rows, matching_used, config_summary


def simulation(
    config_path: str | Path,
    *,
    smoke: bool = False,
    force: bool = False,
    allow_heavy: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    out_dir = _out_dir(cfg, output_root)
    c2_cfg = _plife_c2_cfg(cfg)
    log_event(
        f"PLife++ C2 simulation start smoke={smoke} force={force} allow_heavy={allow_heavy} dry_run={dry_run} output={out_dir}",
        component="c2-plife",
    )
    if not bool(_get(c2_cfg, "enabled", True)):
        log_event("PLife++ C2 simulation disabled by config", component="c2-plife")
        return {"status": "disabled"}
    branch_root = _branch_root(cfg, output_root)
    plan_path = out_dir / "branch_plan.csv"
    plan_ok, plan_reason = _branch_plan_meta_current(plan_path)
    if force or not plan_ok:
        try:
            rows, matching, config_summary = _build_branch_plan_rows(cfg, output_root, out_dir, branch_root)
        except FileNotFoundError as exc:
            summary = {"status": "skipped", "reason": str(exc), "branch_plan": str(plan_path)}
            write_json(out_dir / "c2_plife_plus_simulation_summary.json", summary)
            log_event(f"PLife++ C2 simulation skipped: {summary['reason']}", component="c2-plife")
            return summary
        log_event(
            f"PLife++ C2 simulation rebuilt branch plan reason={plan_reason} n_plan_rows={len(rows)}",
            component="c2-plife",
        )
    else:
        rows = read_csv(plan_path)
        try:
            with (out_dir / "branch_plan_meta.json").open("r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        matching = str(meta.get("matching", "unknown"))
        config_summary = dict(meta.get("branching_config", {}))
        log_event(f"PLife++ C2 simulation using current branch plan n_plan_rows={len(rows)}", component="c2-plife")

    if not rows:
        _write_branch_plan_with_meta(out_dir, rows, matching=matching, config_summary=config_summary)
        summary = {"status": "skipped", "reason": "empty branch plan", "branch_plan": str(plan_path)}
        write_json(out_dir / "c2_plife_plus_simulation_summary.json", summary)
        log_event(f"PLife++ C2 simulation skipped: {summary['reason']}", component="c2-plife")
        return summary

    future_sample_every = int(_get(c2_cfg, "future_sample_every_steps", _get(c2_cfg, "sample_every_steps", 25)))
    metric_mode = _branching_metric_mode(_get(c2_cfg, "branching_metric", "clip_chamfer"))
    render_img_size = int(_get(c2_cfg, "render_img_size", 128)) if metric_mode == "clip_chamfer" else 0
    total_expected = len(rows)

    def _mark_without_running(status: str) -> dict[str, Any]:
        existing = 0
        for row in rows:
            out = _branch_output_npz_from_row(branch_root, row)
            if out.exists():
                row["status"] = "exists"
                existing += 1
            else:
                row["status"] = status
            row["branch_output_path"] = str(out)
            row["command"] = command_to_str(
                [
                    "internal_plife_branch",
                    str(row.get("params_path", "")),
                    "--step",
                    str(row.get("step", "")),
                    "--horizon",
                    str(row.get("horizon_steps", "")),
                    "--branch-seed",
                    str(row.get("branch_seed", "")),
                ]
            )
        _write_branch_plan_with_meta(out_dir, rows, matching=matching, config_summary=config_summary)
        return {
            "n_existing": existing,
            "n_expected_branch_outputs": total_expected,
        }

    if not allow_heavy:
        counts = _mark_without_running("skipped_heavy")
        summary = {
            "status": "skipped_heavy",
            "reason": "PLife++ C2 branch simulation requires --allow-heavy",
            "branch_plan": str(plan_path),
            "n_plan_rows": len(rows),
            **counts,
        }
        write_json(out_dir / "c2_plife_plus_simulation_summary.json", summary)
        log_event(
            f"PLife++ C2 simulation skipped heavy n_plan_rows={len(rows)} existing={counts['n_existing']}/{total_expected} plan={plan_path}",
            component="c2-plife",
        )
        return summary

    if dry_run:
        counts = _mark_without_running("dry_run")
        summary = {"status": "dry_run", "n_plan_rows": len(rows), "branch_plan": str(plan_path), **counts}
        write_json(out_dir / "c2_plife_plus_simulation_summary.json", summary)
        log_event(f"PLife++ C2 simulation dry-run n_plan_rows={len(rows)}", component="c2-plife")
        return summary

    log_event(f"PLife++ C2 building substrate branch_root={branch_root}", component="c2-plife")
    substrate = _build_substrate(cfg, smoke=smoke)
    log_event(
        "PLife++ C2 simulation config "
        f"n_plan_rows={len(rows)} expected_branch_outputs={total_expected} metric={metric_mode} "
        f"future_sample_every={future_sample_every} render_img_size={render_img_size}",
        component="c2-plife",
    )
    done = 0
    written = 0
    existing = 0
    skipped = 0
    errors = []
    for branch_idx, row in enumerate(rows, start=1):
        params_path = Path(str(row["params_path"]))
        seed = _int_or(row.get("seed_x", -1), -1)
        out = _branch_output_npz_from_row(branch_root, row)
        row["branch_output_path"] = str(out)
        row["command"] = command_to_str(
            [
                "internal_plife_branch",
                str(params_path),
                "--step",
                str(row.get("step", "")),
                "--horizon",
                str(row.get("horizon_steps", "")),
                "--branch-seed",
                str(row.get("branch_seed", "")),
            ]
        )
        if seed < 0 or not params_path.exists():
            skipped += 1
            errors.append(f"point={row.get('point_id')} missing seed/params")
            row["status"] = "skipped_missing_params"
            log_event(
                f"PLife++ C2 simulation skip branch {branch_idx}/{len(rows)} point={row.get('point_id')} "
                f"seed={seed} params={params_path}",
                component="c2-plife",
            )
            continue
        params = np.load(params_path, allow_pickle=True)
        if _progress_now(branch_idx, len(rows), every=10):
            log_event(
                f"PLife++ C2 simulation branch {branch_idx}/{len(rows)} point={row.get('point_id')} branch={row.get('branch_id')} "
                f"traj={row.get('traj_id')} condition={row.get('condition')} step={row.get('step')}",
                component="c2-plife",
            )
        if out.exists() and not force:
            done += 1
            existing += 1
            row["status"] = "exists"
            continue
        ensure_dir(out.parent)
        perturb = {
            "x_std": float(row.get("perturb_x_std", _get(_get(c2_cfg, "perturb", {}), "x_std", 0.003))),
            "v_std": float(row.get("perturb_v_std", _get(_get(c2_cfg, "perturb", {}), "v_std", 0.0))),
            "c_std": float(row.get("perturb_c_std", _get(_get(c2_cfg, "perturb", {}), "c_std", 0.01))),
        }
        branch_payload = _simulate_one_branch(
            substrate=substrate,
            params=params,
            seed=seed,
            branch_step=_int_or(row["step"], 0),
            rep_seed=_int_or(row.get("branch_seed", -1), 10_000_000 + 1009 * _int_or(row.get("point_id", 0), 0) + _int_or(row.get("branch_id", 0), 0)),
            horizon_steps=_int_or(row["horizon_steps"], 0),
            sample_every=future_sample_every,
            perturb=perturb,
            render_img_size=render_img_size,
        )
        if written == 0:
            shape_text = ", ".join(f"{key}={tuple(value.shape)}" for key, value in branch_payload.items())
            log_event(f"PLife++ C2 first branch payload shapes {shape_text}", component="c2-plife")
        np.savez_compressed(
            out,
            **{key: np.asarray(value, dtype=np.float32) for key, value in branch_payload.items()},
            point_id=np.asarray(_int_or(row["point_id"], -1), dtype=np.int32),
            branch_id=np.asarray(_int_or(row.get("branch_id", 0), 0), dtype=np.int32),
            branch_seed=np.asarray(_int_or(row.get("branch_seed", -1), -1), dtype=np.int64),
            step=np.asarray(_int_or(row["step"], 0), dtype=np.int32),
            horizon_steps=np.asarray(_int_or(row["horizon_steps"], 0), dtype=np.int32),
            future_sample_every_steps=np.asarray(future_sample_every, dtype=np.int32),
            branching_metric=np.asarray(metric_mode),
            render_img_size=np.asarray(int(render_img_size), dtype=np.int32),
        )
        row["status"] = "exists" if out.exists() else "missing"
        done += 1
        written += 1
        if written == 1 or written % 25 == 0 or done == total_expected:
            log_event(
                f"PLife++ C2 branch outputs progress done={done}/{total_expected} written={written} existing={existing} last={out}",
                component="c2-plife",
            )
    _write_branch_plan_with_meta(out_dir, rows, matching=matching, config_summary=config_summary)
    summary = {
        "status": "ok",
        "n_written_or_existing": done,
        "n_written": written,
        "n_existing": existing,
        "n_expected_branch_outputs": total_expected,
        "n_skipped_points": skipped,
        "errors": errors[:20],
        "branch_root": str(branch_root),
        "branch_plan": str(plan_path),
    }
    write_json(out_dir / "c2_plife_plus_simulation_summary.json", summary)
    log_event(
        f"PLife++ C2 simulation done written={written} existing={existing} done={done}/{total_expected} "
        f"skipped_points={skipped} summary={out_dir / 'c2_plife_plus_simulation_summary.json'}",
        component="c2-plife",
    )
    return summary


def visualization(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    log_event(f"PLife++ C2 visualization start smoke={smoke} force={force}", component="c2-plife")
    from paper_suite_visualize import run as run_visualization

    result = run_visualization(config_path, task="c2", smoke=smoke, force=force)
    figure_paths = result.get("figure_paths", {})
    plife_figures = {
        key: value
        for key, value in figure_paths.items()
        if str(key).startswith("c2_plife_plus")
    }
    log_event(f"PLife++ C2 visualization done n_plife_figures={len(plife_figures)}", component="c2-plife")
    return {"status": "ok", "n_plife_figures": len(plife_figures), "figure_paths": plife_figures, "visualization": result}


def _normalize_layer(layer: str) -> str:
    value = str(layer).strip().lower().replace("-", "_")
    aliases = {
        "viz": "visualization",
        "visualisation": "visualization",
        "vizualisation": "visualization",
    }
    return aliases.get(value, value)


def run(
    config_path: str | Path,
    *,
    layer: str = "all",
    smoke: bool = False,
    force: bool = False,
    allow_heavy: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    layer = _normalize_layer(layer)
    if layer == "metrics":
        return metrics(config_path, smoke=smoke, force=force)
    if layer == "simulation":
        return simulation(config_path, smoke=smoke, force=force, allow_heavy=allow_heavy, dry_run=dry_run)
    if layer == "visualization":
        return visualization(config_path, smoke=smoke, force=force)
    if layer != "all":
        raise ValueError(f"Unknown PLife++ C2 layer {layer!r}; use simulation, metrics, visualization, or all.")
    first = metrics(config_path, smoke=smoke, force=force)
    sim = simulation(config_path, smoke=smoke, force=force, allow_heavy=allow_heavy, dry_run=dry_run)
    second = metrics(config_path, smoke=smoke, force=False)
    vis = visualization(config_path, smoke=smoke, force=force)
    return {"metrics_before": first, "simulation": sim, "metrics_after": second, "visualization": vis}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLife++ C2 Delta-H branch sensitivity layer.")
    parser.add_argument("config")
    parser.add_argument(
        "--layer",
        choices=["simulation", "metrics", "visualization", "visualisation", "vizualisation", "viz", "all"],
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, layer=args.layer, smoke=args.smoke, force=args.force, allow_heavy=args.allow_heavy, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
