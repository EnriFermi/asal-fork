from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import list_apf_chunks
from flowlenia_minibang_simulate import expected_delta_h_metric_metadata
from paper_suite_common import (
    REPO_ROOT,
    command_to_str,
    current_python,
    ensure_dir,
    load_config,
    log_event,
    read_csv,
    resolve_path,
    run_subprocess,
    sign_test_greater,
    write_csv,
    write_json,
)
from paper_suite_c2_flowlenia_metrics import _flat_metric_args as _c2_flat_metric_args
from paper_suite_c2_flowlenia_metrics import _rollout_config as _c2_rollout_config
from paper_suite_metric_cache import compare_metrics_npz_metadata, sha256_text, stable_json


BRANCH_PLAN_VERSION = "c2_branch_plan_v6_mid_stratum"


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _output_root(cfg: Any) -> Path:
    return ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))


def _branch_cfg(cfg: Any) -> Any:
    return _get(cfg.get("c2", {}), "branching", {})


def _branch_root(cfg: Any, output_root: Path) -> Path:
    raw = _get(_branch_cfg(cfg), "branch_root", None)
    if raw is None:
        return ensure_dir(output_root / "c2_branching" / "branches")
    path = resolve_path(raw)
    return ensure_dir(path if path is not None else output_root / "c2_branching" / "branches")


def _trajectory_root(c2_cfg: Any) -> Path | None:
    raw = _get(c2_cfg, "trajectory_root", None)
    if raw is None:
        raw = _get(c2_cfg, "minibang_root", "experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set")
    return resolve_path(raw)


def _manifest_path(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists() or len(path.parts) > 1:
        return candidate
    return default.parent / path


def _iter_metric_items(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    items: list[dict[str, Any]] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for row in payload.get("trajectories", []):
            if str(row.get("candidate_kind", "optimized")).strip().lower() != "optimized":
                continue
            traj_id = str(row.get("traj_id", ""))
            traj_dir = _manifest_path(root, row.get("traj_dir"), default=root / traj_id)
            apf_dir = _manifest_path(root, row.get("apf_dir"), default=traj_dir / "apf_logs")
            path = _manifest_path(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            if path.exists():
                items.append(
                    {
                        "traj_id": str(row.get("traj_id", path.parent.name)),
                        "metrics_path": path,
                        "traj_dir": traj_dir,
                        "apf_dir": apf_dir,
                    }
                )
            traj_id = row.get("traj_id")
            if traj_id:
                candidate = root / str(traj_id) / "metrics.npz"
                if candidate.exists() and all(candidate != item["metrics_path"] for item in items):
                    items.append(
                        {
                            "traj_id": str(traj_id),
                            "metrics_path": candidate,
                            "traj_dir": candidate.parent,
                            "apf_dir": candidate.parent / "apf_logs",
                        }
                    )
    if not items:
        items = [
            {"traj_id": path.parent.name, "metrics_path": path, "traj_dir": path.parent, "apf_dir": path.parent / "apf_logs"}
            for path in sorted(root.glob("traj_*/metrics.npz"))
        ]
    return items


def _safe_arr(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key])


def _load_delta_h(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dh = _safe_arr(data, "delta_h_best")
        if dh is None:
            dh_map = _safe_arr(data, "delta_h_map")
            if dh_map is None:
                raise ValueError(f"{path} has neither delta_h_best nor delta_h_map.")
            selected = int(np.asarray(_safe_arr(data, "delta_h_selected_tau_idx", np.asarray(0))).item())
            dh = np.asarray(dh_map[selected], dtype=np.float64)
        dh = np.asarray(dh, dtype=np.float64).reshape(-1)
        centers = _safe_arr(data, "delta_h_window_center_steps")
        if centers is None:
            starts = _safe_arr(data, "delta_h_window_start_steps", np.arange(dh.size))
            centers = np.asarray(starts, dtype=np.float64).reshape(-1)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    n = min(dh.size, centers.size)
    if n == 0:
        raise ValueError(f"{path} has an empty delta-H series.")
    return centers[:n], dh[:n]


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    arr = _safe_arr(data, key)
    if arr is None:
        return default
    try:
        return np.asarray(arr).reshape(-1)[0].item()
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


def _preprocess_delta_h_heatmap(delta_h_map: np.ndarray, *, mode: str, floor: float) -> np.ndarray:
    x = np.asarray(delta_h_map, dtype=np.float64)
    mode = str(mode or "clip").strip().lower()
    if mode == "clip":
        out = np.maximum(x, 0.0)
    elif mode == "shift":
        row_min = np.nanmin(x, axis=1, keepdims=True)
        out = x - row_min
    elif mode == "none":
        out = x.copy()
    else:
        raise ValueError(f"Unknown Delta-H preprocessing mode {mode!r}.")
    floor = float(floor or 0.0)
    if floor > 0.0:
        out = np.where(out >= floor, out, 0.0)
    return out


def _load_delta_h_energy(
    path: Path,
    *,
    min_remaining_steps: int | None = None,
    min_remaining_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        dh_map_raw = _safe_arr(data, "delta_h_map")
        if dh_map_raw is None:
            centers, dh = _load_delta_h(path)
            return centers, dh, {
                "selection_energy_source": "delta_h_best_fallback",
                "admissible_tau_count": 1,
                "admissible_tau_steps": "",
            }
        dh_map = np.asarray(dh_map_raw, dtype=np.float64)
        tau_steps = np.asarray(_safe_arr(data, "delta_h_tau_steps", np.arange(dh_map.shape[0])), dtype=np.float64).reshape(-1)
        if dh_map.ndim != 2:
            raise ValueError(f"{path} delta_h_map must be 2D, got {dh_map.shape}.")
        if dh_map.shape[0] == tau_steps.size:
            pass
        elif dh_map.shape[1] == tau_steps.size:
            dh_map = dh_map.T
        else:
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
        if window_size_steps <= 0:
            raise ValueError(f"{path} does not contain a usable Delta-H window size.")

        m_min = int(metric_cfg.get("m_min", 4))
        if min_remaining_steps is not None:
            min_gap_steps = int(min_remaining_steps)
        elif min_remaining_samples is not None:
            min_gap_steps = int(min_remaining_samples) * max(1, sample_every)
        else:
            min_gap_steps = int(m_min) * max(1, sample_every)
        admissible = np.isfinite(tau_steps) & (tau_steps < float(window_size_steps)) & (
            (float(window_size_steps) - tau_steps) >= float(min_gap_steps)
        )
        if not np.any(admissible):
            raise ValueError(
                f"{path} has no admissible tau for branching energy: "
                f"window_size_steps={window_size_steps}, min_remaining_steps={min_gap_steps}, tau_steps={tau_steps.tolist()}."
            )

        mode = str(metric_cfg.get("preprocess_mode", "clip")).strip().lower()
        floor = float(metric_cfg.get("delta_h_floor", 0.0) or 0.0)
        processed = _preprocess_delta_h_heatmap(dh_map, mode=mode, floor=floor)
        energy = np.nanmean(processed[admissible], axis=0)
        n = min(int(centers.size), int(energy.size))
        meta = {
            "selection_energy_source": "mean_tau_phi_delta_h_map",
            "selection_preprocess_mode": mode,
            "selection_delta_h_floor": floor,
            "selection_window_size_steps": window_size_steps,
            "selection_sample_every_steps": sample_every,
            "selection_min_remaining_steps": int(min_gap_steps),
            "admissible_tau_count": int(np.sum(admissible)),
            "admissible_tau_steps": ",".join(str(int(round(x))) for x in tau_steps[admissible]),
        }
    if n == 0:
        raise ValueError(f"{path} produced an empty branching energy profile.")
    return centers[:n], energy[:n], meta


def _validate_metric_item(item: dict[str, Any], flat_args: dict[str, Any]) -> dict[str, Any]:
    metrics_path = Path(item["metrics_path"])
    apf_dir = Path(item.get("apf_dir", Path(item["traj_dir"]) / "apf_logs"))
    metric_cfg, input_identity, _metadata = expected_delta_h_metric_metadata(apf_dir, flat_args)
    fresh, reason, expected = compare_metrics_npz_metadata(metrics_path, metric_cfg, input_identity)
    if not fresh:
        raise ValueError(
            f"C2 branching refuses stale upstream metrics for {item['traj_id']} at {metrics_path}: {reason}. "
            "Run the C2 metrics layer with --force before planning branches."
        )
    return {
        "traj_id": str(item["traj_id"]),
        "metrics_path": str(metrics_path),
        "apf_dir": str(apf_dir),
        "metric_config_hash": str(expected["metric_config_hash"]),
        "metric_input_identity_hash": str(expected["metric_input_identity_hash"]),
    }


def _nearest_indices(steps: np.ndarray, centers: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    if steps.size == 0:
        return np.zeros_like(centers, dtype=np.int64)
    order = np.argsort(steps)
    sorted_steps = steps[order]
    pos = np.searchsorted(sorted_steps, centers, side="left")
    pos = np.clip(pos, 0, sorted_steps.size - 1)
    prev = np.clip(pos - 1, 0, sorted_steps.size - 1)
    choose_prev = np.abs(sorted_steps[prev] - centers) <= np.abs(sorted_steps[pos] - centers)
    nearest = np.where(choose_prev, prev, pos)
    return order[nearest].astype(np.int64)


def _apf_steps_from_chunk(data: np.lib.npyio.NpzFile, *, start: int, end: int, n: int) -> np.ndarray:
    if "state_t" in data.files:
        arr = np.asarray(data["state_t"], dtype=np.float64).reshape(-1)
        if arr.size == n:
            return arr
    if n <= 1:
        return np.asarray([float(start)], dtype=np.float64)
    return np.linspace(float(start), float(end), int(n), dtype=np.float64)


def _activity_covariates(apf_dir: Path, centers: np.ndarray) -> dict[str, np.ndarray]:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    step_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    active_parts: list[np.ndarray] = []
    field_parts: list[np.ndarray] = []
    lag_parts: list[np.ndarray] = []
    for path, start, end, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            n = 0
            if "A" in data.files:
                a = np.asarray(data["A"], dtype=np.float32)
                n = int(a.shape[0])
                mass_parts.append(np.sum(a, axis=tuple(range(1, a.ndim)), dtype=np.float64))
                grid_mass = np.sum(a, axis=-1) if a.ndim >= 4 else a
                active_parts.append(np.mean(grid_mass > 1e-6, axis=tuple(range(1, grid_mass.ndim)), dtype=np.float64))
            if "F" in data.files:
                f = np.asarray(data["F"], dtype=np.float32)
                if n == 0:
                    n = int(f.shape[0])
                field_parts.append(np.mean(np.abs(f), axis=tuple(range(1, f.ndim)), dtype=np.float64))
            if "lagrangian_xy" in data.files:
                lag = np.asarray(data["lagrangian_xy"], dtype=np.float32)
                if n == 0:
                    n = int(lag.shape[0])
                lag_parts.append(lag)
            if n > 0:
                step_parts.append(_apf_steps_from_chunk(data, start=int(start), end=int(end), n=n))
    if not step_parts or not mass_parts:
        raise ValueError(f"No activity covariates could be read from APF chunks in {apf_dir}")
    steps = np.concatenate(step_parts)
    order = np.argsort(steps)
    steps = steps[order]
    idx = _nearest_indices(steps, centers)
    out: dict[str, np.ndarray] = {}
    mass = np.concatenate(mass_parts)[order]
    out["total_mass"] = mass[idx]
    if active_parts:
        active = np.concatenate(active_parts)[order]
        out["active_fraction"] = active[idx]
    if field_parts:
        field = np.concatenate(field_parts)[order]
        out["field_activity"] = field[idx]
    if lag_parts:
        lag = np.concatenate(lag_parts, axis=0)[order]
        speed = np.full((lag.shape[0],), np.nan, dtype=np.float64)
        if lag.shape[0] > 1:
            dxy = np.linalg.norm(np.diff(lag.astype(np.float64), axis=0), axis=-1)
            dt = np.maximum(np.diff(steps), 1e-12)
            speed[1:] = np.nanmean(dxy, axis=1) / dt
            speed[0] = speed[1] if speed.size > 1 else np.nan
        out["mean_lagrangian_speed"] = speed[idx]
    return out


def _select_events(
    *,
    centers: np.ndarray,
    dh: np.ndarray,
    covariates: dict[str, np.ndarray] | None = None,
    m_pairs: int,
    refractory_steps: int,
    high_quantile: float,
    low_quantile: float,
    min_step: int = 0,
) -> list[dict[str, Any]]:
    finite = np.isfinite(dh) & np.isfinite(centers)
    if min_step > 0:
        finite &= np.asarray(centers, dtype=np.float64).reshape(-1) >= float(min_step)
    if int(np.sum(finite)) < 2:
        return []
    c = centers[finite]
    h = dh[finite]
    cov_full = covariates or {}
    cov = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)[finite]
        for key, value in cov_full.items()
        if np.asarray(value).reshape(-1).size == finite.size
    }
    z_cov: dict[str, np.ndarray] = {}
    for key, value in cov.items():
        finite_v = value[np.isfinite(value)]
        if finite_v.size < 2:
            continue
        std = float(np.nanstd(finite_v))
        if std <= 1e-12:
            z_cov[key] = np.zeros_like(value, dtype=np.float64)
        else:
            z_cov[key] = (value - float(np.nanmean(finite_v))) / std
    high_thr = float(np.nanquantile(h, high_quantile))
    low_thr = float(np.nanquantile(h, low_quantile))
    high_order = np.argsort(-h)
    selected_high: list[int] = []
    for idx in high_order:
        if h[idx] < high_thr:
            continue
        step = float(c[idx])
        if any(abs(step - float(c[j])) < refractory_steps for j in selected_high):
            continue
        selected_high.append(int(idx))
        if len(selected_high) >= int(m_pairs):
            break

    low_pool = [int(i) for i in np.argsort(h) if h[int(i)] <= low_thr]
    used_low: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for pair_id, hi in enumerate(selected_high):
        if not low_pool:
            break
        candidates = [i for i in low_pool if i not in used_low]
        if not candidates:
            break
        if z_cov:
            def _cov_dist(i: int) -> tuple[float, float]:
                vals = []
                for key, z in z_cov.items():
                    a = float(z[hi])
                    b = float(z[i])
                    if np.isfinite(a) and np.isfinite(b):
                        vals.append((a - b) ** 2)
                dist = math.sqrt(float(np.mean(vals))) if vals else float("inf")
                return dist, abs(float(c[i]) - float(c[hi]))

            lo = min(candidates, key=_cov_dist)
            match_dist = float(_cov_dist(lo)[0])
            match_method = "activity_covariate_nearest_in_low_delta_h_pool"
        else:
            lo = min(candidates, key=lambda i: abs(float(c[i]) - float(c[hi])))
            match_dist = float("nan")
            match_method = "temporal_fallback_no_activity_covariates"
        used_low.add(lo)
        row = {
            "pair_id": int(pair_id),
            "high_step": int(round(float(c[hi]))),
            "high_delta_h": float(h[hi]),
            "low_step": int(round(float(c[lo]))),
            "low_delta_h": float(h[lo]),
            "match_method": match_method,
            "match_covariate_distance": match_dist,
        }
        for key, value in cov.items():
            row[f"high_{key}"] = float(value[hi])
            row[f"low_{key}"] = float(value[lo])
        pairs.append(row)
    return pairs


def _is_delta_h_sweep_mode(mode: Any) -> bool:
    return str(mode or "paired_high_low").strip().lower() in {
        "continuous",
        "delta_h_sweep",
        "delta_h_quantile_sweep",
        "quantile_sweep",
        "sampled_delta_h",
    }


def _is_ranked_high_low_mode(mode: Any) -> bool:
    return str(mode or "paired_high_low").strip().lower() in {
        "tau_mean_ranked_high_low",
        "ranked_high_low",
        "sampled_high_low",
        "quantile_rank_high_low",
        "mean_tau_high_low",
    }


def _quantile_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full_like(x, np.nan, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(x))
    if finite_idx.size == 0:
        return out
    vals = x[finite_idx]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=np.float64)
    i = 0
    while i < vals.size:
        j = i + 1
        while j < vals.size and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    denom = max(1, vals.size - 1)
    out[finite_idx] = ranks / float(denom)
    return out


def _trajectory_end_step(apf_dir: Path) -> int | None:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        return None
    return int(chunks[-1][2])


def _apf_saved_steps(apf_dir: Path) -> np.ndarray:
    chunks = list_apf_chunks(apf_dir)
    parts: list[np.ndarray] = []
    for path, start, end, _idx in chunks:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "steps" in data.files:
                    steps = np.asarray(data["steps"], dtype=np.int64).reshape(-1)
                    if steps.size:
                        parts.append(steps)
                        continue
                n = int(np.asarray(data["A"]).shape[0]) if "A" in data.files else 0
                if n > 0:
                    parts.append(_apf_steps_from_chunk(data, start=int(start), end=int(end), n=n).astype(np.int64))
        except Exception:
            continue
    if not parts:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64))


def _nearest_apf_step(apf_dir: Path, requested_step: int, *, cache: dict[str, np.ndarray] | None = None) -> int:
    key = str(apf_dir.resolve())
    if cache is not None and key in cache:
        steps = cache[key]
    else:
        steps = _apf_saved_steps(apf_dir)
        if cache is not None:
            cache[key] = steps
    if steps.size == 0:
        return int(requested_step)
    idx = int(np.argmin(np.abs(steps.astype(np.float64) - float(requested_step))))
    return int(steps[idx])


def _as_float_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        try:
            parts = list(raw)
        except TypeError:
            parts = [raw]
    out: list[float] = []
    for value in parts:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _empirical_quantile(values: np.ndarray, idx: int) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or idx < 0 or idx >= x.size or not np.isfinite(x[idx]):
        return float("nan")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite <= float(x[idx])))


def _select_delta_h_points(
    *,
    centers: np.ndarray,
    dh: np.ndarray,
    covariates: dict[str, np.ndarray] | None = None,
    n_points: int,
    refractory_steps: int,
    quantiles: list[float] | None = None,
    quantile_min: float = 0.05,
    quantile_max: float = 0.95,
    min_step: int = 0,
) -> list[dict[str, Any]]:
    finite = np.isfinite(dh) & np.isfinite(centers)
    if min_step > 0:
        finite &= np.asarray(centers, dtype=np.float64).reshape(-1) >= float(min_step)
    if int(np.sum(finite)) < 1:
        return []
    c = np.asarray(centers, dtype=np.float64).reshape(-1)[finite]
    h = np.asarray(dh, dtype=np.float64).reshape(-1)[finite]
    cov_full = covariates or {}
    cov = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)[finite]
        for key, value in cov_full.items()
        if np.asarray(value).reshape(-1).size == finite.size
    }

    if quantiles:
        targets = [float(np.clip(q, 0.0, 1.0)) for q in quantiles]
    else:
        n = max(1, int(n_points))
        q0 = float(np.clip(quantile_min, 0.0, 1.0))
        q1 = float(np.clip(quantile_max, 0.0, 1.0))
        if q1 < q0:
            q0, q1 = q1, q0
        targets = [float(q) for q in np.linspace(q0, q1, n)]

    selected: list[int] = []
    points: list[dict[str, Any]] = []
    for target_q in targets:
        target_h = float(np.nanquantile(h, target_q))
        order = np.argsort(np.abs(h - target_h))
        chosen: int | None = None
        for idx in order:
            i = int(idx)
            if i in selected:
                continue
            step = float(c[i])
            if any(abs(step - float(c[j])) < refractory_steps for j in selected):
                continue
            chosen = i
            break
        if chosen is None:
            for idx in order:
                i = int(idx)
                if i not in selected:
                    chosen = i
                    break
        if chosen is None:
            continue
        point_id = len(points)
        selected.append(chosen)
        row: dict[str, Any] = {
            "point_id": int(point_id),
            "step": int(round(float(c[chosen]))),
            "delta_h": float(h[chosen]),
            "target_quantile": float(target_q),
            "delta_h_quantile": _empirical_quantile(h, chosen),
            "selection_method": "nearest_delta_h_quantile_with_refractory",
        }
        for key, value in cov.items():
            row[key] = float(value[chosen])
        points.append(row)
    return points


def _select_ranked_high_low_points(
    *,
    centers: np.ndarray,
    energy: np.ndarray,
    covariates: dict[str, np.ndarray] | None,
    n_high: int,
    n_low: int,
    n_mid: int,
    q_high: float,
    q_low: float,
    q_mid_low: float = 0.4,
    q_mid_high: float = 0.6,
    horizon_steps: int,
    trajectory_end_step: int | None,
    seed: int,
    min_step: int = 0,
    energy_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    n = min(c.size, e.size)
    if n == 0:
        return [], {
            "n_high_pool": 0,
            "n_mid_pool": 0,
            "n_low_pool": 0,
            "n_high_selected": 0,
            "n_mid_selected": 0,
            "n_low_selected": 0,
        }
    c = c[:n]
    e = e[:n]
    finite = np.isfinite(c) & np.isfinite(e)
    if min_step > 0:
        finite &= c >= float(min_step)
    if trajectory_end_step is not None:
        finite &= (c + float(horizon_steps)) <= float(trajectory_end_step)
    if int(np.sum(finite)) < 1:
        return [], {
            "n_high_pool": 0,
            "n_mid_pool": 0,
            "n_low_pool": 0,
            "n_high_selected": 0,
            "n_mid_selected": 0,
            "n_low_selected": 0,
            "min_branch_step": int(min_step),
            "trajectory_end_step": "" if trajectory_end_step is None else int(trajectory_end_step),
        }

    q_high = float(np.clip(q_high, 0.0, 1.0))
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_mid_low = float(np.clip(q_mid_low, 0.0, 1.0))
    q_mid_high = float(np.clip(q_mid_high, 0.0, 1.0))
    if q_mid_high < q_mid_low:
        q_mid_low, q_mid_high = q_mid_high, q_mid_low
    finite_values = e[finite]
    high_threshold = float(np.nanquantile(finite_values, q_high))
    low_threshold = float(np.nanquantile(finite_values, q_low))
    qrank = np.full_like(e, np.nan, dtype=np.float64)
    qrank[finite] = _quantile_ranks(e[finite])
    high_pool = np.flatnonzero(finite & (e >= high_threshold))
    mid_pool = np.flatnonzero(finite & (qrank >= q_mid_low) & (qrank <= q_mid_high))
    low_pool = np.flatnonzero(finite & (e <= low_threshold))
    rng = np.random.default_rng(int(seed))

    def _draw(pool: np.ndarray, count: int) -> np.ndarray:
        if pool.size == 0 or int(count) <= 0:
            return np.asarray([], dtype=np.int64)
        k = min(int(pool.size), int(count))
        return rng.choice(pool, size=k, replace=False).astype(np.int64)

    selected_high = _draw(high_pool, int(n_high))
    selected_mid = _draw(mid_pool, int(n_mid))
    selected_low = _draw(low_pool, int(n_low))
    cov_full = covariates or {}
    cov = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)[:n]
        for key, value in cov_full.items()
        if np.asarray(value).reshape(-1).size >= n
    }
    common_meta = {
        "selection_method": "mean_tau_phi_delta_h_quantile_rank_sampling_without_replacement",
        "selection_high_quantile": q_high,
        "selection_low_quantile": q_low,
        "selection_mid_quantile_low": q_mid_low,
        "selection_mid_quantile_high": q_mid_high,
        "selection_high_threshold": high_threshold,
        "selection_low_threshold": low_threshold,
        "n_high_pool": int(high_pool.size),
        "n_mid_pool": int(mid_pool.size),
        "n_low_pool": int(low_pool.size),
        "n_high_requested": int(n_high),
        "n_mid_requested": int(n_mid),
        "n_low_requested": int(n_low),
        "n_high_selected": int(selected_high.size),
        "n_mid_selected": int(selected_mid.size),
        "n_low_selected": int(selected_low.size),
        "min_branch_step": int(min_step),
        "trajectory_end_step": "" if trajectory_end_step is None else int(trajectory_end_step),
    }
    if energy_meta:
        common_meta.update(energy_meta)

    rows: list[dict[str, Any]] = []
    for condition, selected in (("high", selected_high), ("mid", selected_mid), ("low", selected_low)):
        for local_id, original_idx in enumerate(selected.tolist()):
            target_q = q_high if condition == "high" else q_low if condition == "low" else 0.5 * (q_mid_low + q_mid_high)
            row: dict[str, Any] = {
                "pair_id": int(local_id),
                "point_id": int(local_id),
                "condition": condition,
                "window_index": int(original_idx),
                "window_center_step": int(round(float(c[original_idx]))),
                "step": int(round(float(c[original_idx]))),
                "delta_h": float(e[original_idx]),
                "delta_h_energy": float(e[original_idx]),
                "delta_h_quantile_rank": float(qrank[original_idx]),
                "target_quantile": float(target_q),
                **common_meta,
            }
            for key, value in cov.items():
                row[f"point_{key}"] = float(value[original_idx])
            rows.append(row)
    return rows, common_meta


def _expected_resume_metadata_from_plan_row(row: dict[str, Any]) -> dict[str, Any] | None:
    required = ("step", "horizon_steps", "branch_seed", "perturb_a_std", "perturb_p_std", "perturb_lagrangian_xy_std")
    if any(str(row.get(key, "")).strip() == "" for key in required):
        return None
    try:
        step = int(float(row["step"]))
        horizon = int(float(row["horizon_steps"]))
        return {
            "start_step": step,
            "end_step": step + horizon,
            "branch_seed": int(float(row["branch_seed"])),
            "perturb_a_std": float(row["perturb_a_std"]),
            "perturb_p_std": float(row["perturb_p_std"]),
            "perturb_lagrangian_xy_std": float(row["perturb_lagrangian_xy_std"]),
        }
    except (TypeError, ValueError):
        return None


def _metadata_matches(found: dict[str, Any], expected: dict[str, Any] | None) -> bool:
    if expected is None:
        return True
    for key, value in expected.items():
        if key not in found:
            return False
        try:
            if isinstance(value, float):
                if not math.isclose(float(found[key]), float(value), rel_tol=1e-9, abs_tol=1e-12):
                    return False
            else:
                if int(found[key]) != int(value):
                    return False
        except (TypeError, ValueError):
            return False
    return True


def _branch_output_ok(branch_dir: Path, *, expected_metadata: dict[str, Any] | None = None) -> bool:
    if (branch_dir / "branch_feature.npz").exists() or (branch_dir / "metrics.npz").exists():
        return expected_metadata is None
    apf_dir = branch_dir / "apf_logs"
    metadata_path = branch_dir / "resume_metadata.json"
    if not apf_dir.exists() or not metadata_path.exists():
        return False
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        return False
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if not _metadata_matches(metadata, expected_metadata):
            return False
        end_step = int(metadata["end_step"])
    except Exception:
        return False
    return int(chunks[-1][2]) >= end_step


def _make_resume_command(
    *,
    source_traj_dir: Path,
    step: int,
    horizon_steps: int,
    branch_dir: Path,
    branch_seed: int,
    perturb_a_std: float,
    perturb_p_std: float,
    perturb_lag_xy_std: float,
    force: bool,
) -> list[str]:
    cmd = [
        current_python(),
        "scripts/flowlenia_minibang_resume.py",
        str(source_traj_dir),
        "--step",
        str(int(step)),
        "--additional-steps",
        str(int(horizon_steps)),
        "--output-dir",
        str(branch_dir),
        "--branch-seed",
        str(int(branch_seed)),
        "--perturb-a-std",
        str(float(perturb_a_std)),
        "--perturb-p-std",
        str(float(perturb_p_std)),
        "--perturb-lagrangian-xy-std",
        str(float(perturb_lag_xy_std)),
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def _make_resume_batch_command(*, jobs_path: Path, batch_size: int, force: bool) -> list[str]:
    cmd = [
        current_python(),
        "scripts/flowlenia_minibang_resume_batch.py",
        "--jobs-json",
        str(jobs_path),
        "--batch-size",
        str(int(batch_size)),
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def _write_smoke_fixture(output_root: Path, branch_root: Path) -> Path:
    out_dir = ensure_dir(output_root / "c2_branching")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(123)
    yy, xx = np.mgrid[0:8, 0:8].astype(np.float32)
    base_a = ((yy + xx) / 14.0)[None, :, :, None]
    base_p = np.repeat(base_a, 3, axis=-1)
    base_f = np.concatenate([base_a, 1.0 - base_a], axis=-1)
    for pair_id in range(2):
        for condition, step, dh, spread in (
            ("high", 100 + pair_id * 100, 1.0 + 0.2 * pair_id, 0.35),
            ("low", 150 + pair_id * 100, 0.1 + 0.05 * pair_id, 0.04),
        ):
            center = rng.normal(0.0, 0.05, size=(1, 1, 1, 1)).astype(np.float32)
            for branch_id in range(3):
                branch_dir = ensure_dir(branch_root / "smoke_traj" / f"pair_{pair_id:03d}_{condition}" / f"branch_{branch_id:02d}")
                apf_dir = ensure_dir(branch_dir / "apf_logs")
                drift = center + np.float32(branch_id * spread)
                noise = rng.normal(0.0, spread * 0.02, size=(6, 8, 8, 1)).astype(np.float32)
                a = np.clip(base_a + drift + noise, 0.0, 1.0).astype(np.float32)
                p = np.clip(base_p + np.repeat(noise, 3, axis=-1), 0.0, 1.0).astype(np.float32)
                f = (base_f + np.repeat(noise[..., :1], 2, axis=-1)).astype(np.float32)
                np.savez_compressed(
                    apf_dir / "P_steps_000000000_000000006__secs_0.000_0.006__idx_000000.npz",
                    A=a,
                    P=p,
                    F=f,
                    state_t=np.arange(6, dtype=np.int32),
                )
                rows.append(
                    {
                        "traj_id": "smoke_traj",
                        "pair_id": pair_id,
                        "condition": condition,
                        "step": step,
                        "delta_h": dh,
                        "branch_id": branch_id,
                        "source_metrics_path": "",
                        "source_traj_dir": "",
                        "branch_dir": str(branch_dir),
                        "status": "smoke_written",
                        "command": "",
                    }
                )
    path = out_dir / "branch_plan.csv"
    write_csv(path, rows)
    csv_rows = read_csv(path)
    write_json(
        out_dir / "branch_plan_meta.json",
        {
            "branch_plan_version": BRANCH_PLAN_VERSION,
            "branch_plan_rows_hash": sha256_text(stable_json(csv_rows)),
            "smoke": True,
            "matching": "smoke_fixture",
        },
    )
    write_json(output_root / "c2_branching_simulation_summary.json", {"status": "ok", "n_branches": len(rows), "plan": str(path)})
    return path


def _validate_branch_plan(plan_path: Path, plan_rows: list[dict[str, str]], flat_args: dict[str, Any]) -> None:
    meta_path = plan_path.with_name("branch_plan_meta.json")
    if not meta_path.exists():
        raise ValueError(f"C2 branching refuses old branch_plan without metadata: {plan_path}. Regenerate the branch simulation layer.")
    with meta_path.open("r") as f:
        meta = json.load(f)
    if str(meta.get("branch_plan_version", "")) != BRANCH_PLAN_VERSION:
        raise ValueError(
            f"C2 branching refuses stale branch_plan version {meta.get('branch_plan_version')!r}; "
            f"expected {BRANCH_PLAN_VERSION}. Regenerate the branch simulation layer."
        )
    current_hash = sha256_text(stable_json(plan_rows))
    if str(meta.get("branch_plan_rows_hash", "")) != current_hash:
        raise ValueError("C2 branching refuses branch_plan.csv because it no longer matches branch_plan_meta.json.")

    unique: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        metrics_path = str(row.get("source_metrics_path", "")).strip()
        if not metrics_path:
            continue
        traj_dir_raw = str(row.get("source_traj_dir", "")).strip()
        traj_dir = Path(traj_dir_raw) if traj_dir_raw else Path(metrics_path).parent
        apf_dir_raw = str(row.get("source_apf_dir", "")).strip()
        apf_dir = Path(apf_dir_raw) if apf_dir_raw else traj_dir / "apf_logs"
        unique.setdefault(
            metrics_path,
            {
                "traj_id": str(row.get("traj_id", Path(metrics_path).parent.name)),
                "metrics_path": Path(metrics_path),
                "traj_dir": traj_dir,
                "apf_dir": apf_dir,
            },
        )
    for item in unique.values():
        _validate_metric_item(item, flat_args)


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
    out_dir = ensure_dir(output_root / "c2_branching")
    bcfg = _branch_cfg(cfg)
    branch_root = _branch_root(cfg, output_root)
    log_event(
        f"C2 branching simulation start smoke={smoke} force={force} allow_heavy={allow_heavy} dry_run={dry_run} branch_root={branch_root}",
        component="c2-branch",
    )
    if smoke:
        plan = _write_smoke_fixture(output_root, branch_root)
        log_event(f"C2 branching simulation smoke fixture plan={plan}", component="c2-branch")
        return {"status": "ok", "n_branches": len(read_csv(plan)), "plan": str(plan)}

    c2_cfg = cfg.get("c2", {})
    trajectory_root = _trajectory_root(c2_cfg)
    if trajectory_root is None or not trajectory_root.exists():
        summary = {"status": "skipped", "reason": f"missing trajectory root {trajectory_root}"}
        write_json(output_root / "c2_branching_simulation_summary.json", summary)
        log_event(f"C2 branching simulation skipped missing trajectory_root={trajectory_root}", component="c2-branch")
        return summary

    metric_items = _iter_metric_items(trajectory_root)
    if not metric_items:
        summary = {
            "status": "skipped",
            "reason": f"no metrics.npz found under {trajectory_root}; run the C2 metrics layer after APF simulation first",
        }
        write_json(output_root / "c2_branching_simulation_summary.json", summary)
        log_event(f"C2 branching simulation skipped no metrics trajectory_root={trajectory_root}", component="c2-branch")
        return summary

    rollout_config = _c2_rollout_config(c2_cfg)
    flat_args = _c2_flat_metric_args(rollout_config)
    max_trajectories = int(_get(bcfg, "max_trajectories", 2))
    m_pairs = int(_get(bcfg, "m_pairs", 2))
    selection_mode = str(_get(bcfg, "selection_mode", "paired_high_low"))
    n_points_per_trajectory = int(_get(bcfg, "n_points_per_trajectory", max(1, 2 * m_pairs)))
    point_quantiles = _as_float_list(_get(bcfg, "point_quantiles", None))
    point_quantile_min = float(_get(bcfg, "point_quantile_min", 0.05))
    point_quantile_max = float(_get(bcfg, "point_quantile_max", 0.95))
    branches_per_time = int(_get(bcfg, "branches_per_time", 3))
    resume_batch_size = int(_get(bcfg, "resume_batch_size", 1))
    refractory_steps = int(_get(bcfg, "refractory_steps", 5000))
    high_quantile = float(_get(bcfg, "high_quantile", 0.8))
    low_quantile = float(_get(bcfg, "low_quantile", 0.2))
    mid_quantile_low = float(_get(bcfg, "mid_quantile_low", 0.4))
    mid_quantile_high = float(_get(bcfg, "mid_quantile_high", 0.6))
    n_high = int(_get(bcfg, "n_high", m_pairs))
    n_low = int(_get(bcfg, "n_low", m_pairs))
    n_mid = int(_get(bcfg, "n_mid", m_pairs))
    selection_seed = int(_get(bcfg, "selection_seed", 12345))
    energy_min_remaining_steps_raw = _get(bcfg, "energy_min_remaining_steps", None)
    energy_min_remaining_samples_raw = _get(bcfg, "energy_min_samples", None)
    energy_min_remaining_steps = None if energy_min_remaining_steps_raw is None else int(energy_min_remaining_steps_raw)
    energy_min_remaining_samples = None if energy_min_remaining_samples_raw is None else int(energy_min_remaining_samples_raw)
    min_branch_step = int(_get(bcfg, "min_branch_step", _get(bcfg, "selection_min_step", 0)))
    horizon_steps = int(_get(bcfg, "horizon_steps", 1000))
    perturb = _get(bcfg, "perturb", {})
    perturb_a_std = float(_get(perturb, "a_std", 1e-4))
    perturb_p_std = float(_get(perturb, "p_std", 1e-4))
    perturb_lag_xy_std = float(_get(perturb, "lagrangian_xy_std", 0.01))

    ranked: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]] = []
    metric_records: list[dict[str, Any]] = []
    for item in metric_items:
        path = Path(item["metrics_path"])
        record = _validate_metric_item(item, flat_args)
        apf_dir = Path(item.get("apf_dir", Path(item["traj_dir"]) / "apf_logs"))
        if _is_ranked_high_low_mode(selection_mode):
            centers, dh, selection_meta = _load_delta_h_energy(
                path,
                min_remaining_steps=energy_min_remaining_steps,
                min_remaining_samples=energy_min_remaining_samples,
            )
            rank_score = float(np.nanmean(dh))
            covariates = {}
        else:
            centers, dh = _load_delta_h(path)
            selection_meta = {}
            rank_score = float(np.nanmax(dh))
            covariates = _activity_covariates(apf_dir, centers)
        metric_records.append(record)
        ranked.append((rank_score, item, centers, dh, covariates, record, selection_meta))
    ranked.sort(key=lambda x: -x[0])
    log_event(
        f"C2 branching simulation ranked n_metric_items={len(ranked)} max_trajectories={max_trajectories} "
        f"selection_mode={selection_mode} m_pairs={m_pairs} n_high={n_high} n_mid={n_mid} n_low={n_low} "
        f"n_points_per_trajectory={n_points_per_trajectory} "
        f"branches_per_time={branches_per_time}",
        component="c2-branch",
    )

    rows: list[dict[str, Any]] = []
    pending_batch_jobs: list[tuple[int, dict[str, Any]]] = []
    saved_steps_cache: dict[str, np.ndarray] = {}

    def _append_branch_row(
        *,
        source_traj_dir: Path,
        source_apf_dir: Path,
        metrics_path: Path,
        traj_id: str,
        pair_id: int,
        condition: str,
        step: int,
        delta_h: float,
        branch_id: int,
        branch_seed: int,
        branch_dir: Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        requested_step = int(step)
        snapped_step = _nearest_apf_step(source_apf_dir, requested_step, cache=saved_steps_cache)
        expected_metadata = {
            "start_step": int(snapped_step),
            "end_step": int(snapped_step) + int(horizon_steps),
            "branch_seed": int(branch_seed),
            "perturb_a_std": float(perturb_a_std),
            "perturb_p_std": float(perturb_p_std),
            "perturb_lagrangian_xy_std": float(perturb_lag_xy_std),
        }
        branch_ready = _branch_output_ok(branch_dir, expected_metadata=expected_metadata)
        overwrite_incomplete = branch_dir.exists() and not branch_ready
        cmd = _make_resume_command(
            source_traj_dir=source_traj_dir,
            step=snapped_step,
            horizon_steps=horizon_steps,
            branch_dir=branch_dir,
            branch_seed=branch_seed,
            perturb_a_std=perturb_a_std,
            perturb_p_std=perturb_p_std,
            perturb_lag_xy_std=perturb_lag_xy_std,
            force=force or overwrite_incomplete,
        )
        if branch_ready and not force:
            status = "exists"
        elif not allow_heavy:
            status = "skipped_heavy"
        else:
            if overwrite_incomplete:
                log_event(
                    f"C2 branching simulation removing incomplete branch output {branch_dir}",
                    component="c2-branch",
                )
                if not dry_run:
                    shutil.rmtree(branch_dir)
            if resume_batch_size > 1:
                status = "queued_batch" if not dry_run else "dry_run"
            else:
                log_event(
                    f"C2 branching simulation running traj={traj_id} pair={pair_id} condition={condition} "
                    f"branch={branch_id} step={snapped_step} requested_step={requested_step}",
                    component="c2-branch",
                )
                run_subprocess(cmd, dry_run=dry_run)
                status = "dry_run" if dry_run else ("exists" if _branch_output_ok(branch_dir, expected_metadata=expected_metadata) else "missing")
        row = {
            "traj_id": traj_id,
            "pair_id": int(pair_id),
            "condition": condition,
            "step": int(snapped_step),
            "requested_step": int(requested_step),
            "step_snap_delta": int(snapped_step - requested_step),
            "delta_h": float(delta_h),
            "branch_id": int(branch_id),
            "branch_seed": int(branch_seed),
            "horizon_steps": int(horizon_steps),
            "perturb_a_std": float(perturb_a_std),
            "perturb_p_std": float(perturb_p_std),
            "perturb_lagrangian_xy_std": float(perturb_lag_xy_std),
            "source_metrics_path": str(metrics_path),
            "source_traj_dir": str(source_traj_dir),
            "source_apf_dir": str(source_apf_dir),
            "branch_dir": str(branch_dir),
            "selection_mode": selection_mode,
            "status": status,
            "command": command_to_str(cmd),
        }
        if extra:
            row.update(extra)
        rows.append(row)
        if status == "queued_batch":
            pending_batch_jobs.append(
                (
                    len(rows) - 1,
                    {
                        "source_traj_dir": str(source_traj_dir),
                        "step": int(snapped_step),
                        "additional_steps": int(horizon_steps),
                        "output_dir": str(branch_dir),
                        "branch_seed": int(branch_seed),
                        "perturb_a_std": float(perturb_a_std),
                        "perturb_p_std": float(perturb_p_std),
                        "perturb_lagrangian_xy_std": float(perturb_lag_xy_std),
                    },
                )
            )

    for traj_order, (_peak, item, centers, dh, covariates, _record, selection_meta) in enumerate(ranked[:max_trajectories]):
        metrics_path = Path(item["metrics_path"])
        source_traj_dir = Path(item["traj_dir"])
        source_apf_dir = Path(item.get("apf_dir", source_traj_dir / "apf_logs"))
        traj_id = str(item["traj_id"])
        if _is_ranked_high_low_mode(selection_mode):
            trajectory_end = _trajectory_end_step(source_apf_dir)
            points, select_summary = _select_ranked_high_low_points(
                centers=centers,
                energy=dh,
                covariates=covariates,
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
                f"C2 branching simulation traj={traj_id} selected_high={select_summary.get('n_high_selected', 0)} "
                f"selected_mid={select_summary.get('n_mid_selected', 0)} selected_low={select_summary.get('n_low_selected', 0)} "
                f"high_pool={select_summary.get('n_high_pool', 0)} mid_pool={select_summary.get('n_mid_pool', 0)} "
                f"low_pool={select_summary.get('n_low_pool', 0)} source={metrics_path}",
                component="c2-branch",
            )
            for point in points:
                point_id = int(point["point_id"])
                pair_id = int(point["pair_id"])
                condition = str(point["condition"])
                step = int(point["step"])
                delta_h = float(point["delta_h"])
                window_idx = int(point.get("window_index", point_id))
                extra = {
                    key: value
                    for key, value in point.items()
                    if key
                    not in {
                        "pair_id",
                        "point_id",
                        "condition",
                        "step",
                        "delta_h",
                    }
                }
                for branch_id in range(branches_per_time):
                    condition_offset = {"high": 0, "mid": 3967, "low": 7919}.get(condition, 12347)
                    branch_seed = int(
                        selection_seed
                        + 1000003 * traj_order
                        + 1009 * point_id
                        + 131 * branch_id
                        + condition_offset
                    )
                    branch_dir = branch_root / traj_id / f"rank_{condition}_{point_id:03d}_w_{window_idx:04d}_step_{step}" / f"branch_{branch_id:02d}"
                    _append_branch_row(
                        source_traj_dir=source_traj_dir,
                        source_apf_dir=source_apf_dir,
                        metrics_path=metrics_path,
                        traj_id=traj_id,
                        pair_id=pair_id,
                        condition=condition,
                        step=step,
                        delta_h=delta_h,
                        branch_id=branch_id,
                        branch_seed=branch_seed,
                        branch_dir=branch_dir,
                        extra=extra,
                    )
            continue

        if _is_delta_h_sweep_mode(selection_mode):
            points = _select_delta_h_points(
                centers=centers,
                dh=dh,
                covariates=covariates,
                n_points=n_points_per_trajectory,
                refractory_steps=refractory_steps,
                quantiles=point_quantiles or None,
                quantile_min=point_quantile_min,
                quantile_max=point_quantile_max,
                min_step=min_branch_step,
            )
            log_event(
                f"C2 branching simulation traj={traj_id} selected_points={len(points)} source={metrics_path}",
                component="c2-branch",
            )
            for point in points:
                point_id = int(point["point_id"])
                step = int(point["step"])
                delta_h = float(point["delta_h"])
                target_q = float(point.get("target_quantile", float("nan")))
                q_tag = int(round(1000.0 * target_q)) if np.isfinite(target_q) else point_id
                extra = {
                    "point_id": point_id,
                    "target_quantile": point.get("target_quantile", ""),
                    "delta_h_quantile": point.get("delta_h_quantile", ""),
                    "selection_method": point.get("selection_method", ""),
                    "point_total_mass": point.get("total_mass", ""),
                    "point_active_fraction": point.get("active_fraction", ""),
                    "point_mean_lagrangian_speed": point.get("mean_lagrangian_speed", ""),
                    "point_field_activity": point.get("field_activity", ""),
                }
                for branch_id in range(branches_per_time):
                    branch_seed = int(2000003 + 1009 * point_id + 131 * branch_id)
                    branch_dir = branch_root / traj_id / f"point_{point_id:03d}_q_{q_tag:04d}_step_{step}" / f"branch_{branch_id:02d}"
                    _append_branch_row(
                        source_traj_dir=source_traj_dir,
                        source_apf_dir=source_apf_dir,
                        metrics_path=metrics_path,
                        traj_id=traj_id,
                        pair_id=point_id,
                        condition="sampled",
                        step=step,
                        delta_h=delta_h,
                        branch_id=branch_id,
                        branch_seed=branch_seed,
                        branch_dir=branch_dir,
                        extra=extra,
                    )
            continue

        pairs = _select_events(
            centers=centers,
            dh=dh,
            covariates=covariates,
            m_pairs=m_pairs,
            refractory_steps=refractory_steps,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
            min_step=min_branch_step,
        )
        log_event(
            f"C2 branching simulation traj={traj_id} selected_pairs={len(pairs)} source={metrics_path}",
            component="c2-branch",
        )
        for pair in pairs:
            for condition in ("high", "low"):
                step = int(pair[f"{condition}_step"])
                delta_h = float(pair[f"{condition}_delta_h"])
                for branch_id in range(branches_per_time):
                    branch_seed = int(1000003 + 1009 * int(pair["pair_id"]) + 131 * branch_id + (0 if condition == "high" else 7919))
                    branch_dir = branch_root / traj_id / f"pair_{int(pair['pair_id']):03d}_{condition}_step_{step}" / f"branch_{branch_id:02d}"
                    _append_branch_row(
                        source_traj_dir=source_traj_dir,
                        source_apf_dir=source_apf_dir,
                        metrics_path=metrics_path,
                        traj_id=traj_id,
                        pair_id=int(pair["pair_id"]),
                        condition=condition,
                        step=step,
                        delta_h=delta_h,
                        branch_id=branch_id,
                        branch_seed=branch_seed,
                        branch_dir=branch_dir,
                        extra={
                            "match_method": str(pair.get("match_method", "")),
                            "match_covariate_distance": pair.get("match_covariate_distance", ""),
                            "high_total_mass": pair.get("high_total_mass", ""),
                            "low_total_mass": pair.get("low_total_mass", ""),
                            "high_active_fraction": pair.get("high_active_fraction", ""),
                            "low_active_fraction": pair.get("low_active_fraction", ""),
                            "high_mean_lagrangian_speed": pair.get("high_mean_lagrangian_speed", ""),
                            "low_mean_lagrangian_speed": pair.get("low_mean_lagrangian_speed", ""),
                            "high_field_activity": pair.get("high_field_activity", ""),
                            "low_field_activity": pair.get("low_field_activity", ""),
                        },
                    )

    if pending_batch_jobs:
        jobs_path = out_dir / "branch_resume_jobs.json"
        write_json(jobs_path, {"jobs": [job for _idx, job in pending_batch_jobs]})
        batch_cmd = _make_resume_batch_command(jobs_path=jobs_path, batch_size=resume_batch_size, force=force)
        log_event(
            f"C2 branching simulation running batched resume jobs={len(pending_batch_jobs)} batch_size={resume_batch_size}",
            component="c2-branch",
        )
        run_subprocess(batch_cmd, dry_run=dry_run)
        for row_idx, job in pending_batch_jobs:
            branch_dir = Path(str(job["output_dir"]))
            expected_metadata = {
                "start_step": int(job["step"]),
                "end_step": int(job["step"]) + int(job["additional_steps"]),
                "branch_seed": int(job["branch_seed"]),
                "perturb_a_std": float(job["perturb_a_std"]),
                "perturb_p_std": float(job["perturb_p_std"]),
                "perturb_lagrangian_xy_std": float(job["perturb_lagrangian_xy_std"]),
            }
            rows[row_idx]["command"] = command_to_str(batch_cmd)
            rows[row_idx]["status"] = "dry_run" if dry_run else (
                "exists" if _branch_output_ok(branch_dir, expected_metadata=expected_metadata) else "missing"
            )

    plan_path = out_dir / "branch_plan.csv"
    write_csv(plan_path, rows)
    csv_rows = read_csv(plan_path)
    if _is_ranked_high_low_mode(selection_mode):
        matching = "mean_tau_phi_delta_h_quantile_rank_sampling_without_replacement"
    elif _is_delta_h_sweep_mode(selection_mode):
        matching = "delta_h_quantile_sweep"
    else:
        matching = "activity_covariate_nearest_in_low_delta_h_pool"
    plan_meta = {
        "branch_plan_version": BRANCH_PLAN_VERSION,
        "branch_plan_rows_hash": sha256_text(stable_json(csv_rows)),
        "matching": matching,
        "rollout_config": str(rollout_config),
        "metric_records": metric_records,
        "branching_config": {
            "max_trajectories": max_trajectories,
            "m_pairs": m_pairs,
            "n_high": n_high,
            "n_mid": n_mid,
            "n_low": n_low,
            "selection_mode": selection_mode,
            "selection_seed": selection_seed,
            "n_points_per_trajectory": n_points_per_trajectory,
            "point_quantiles": point_quantiles,
            "point_quantile_min": point_quantile_min,
            "point_quantile_max": point_quantile_max,
            "branches_per_time": branches_per_time,
            "horizon_steps": horizon_steps,
            "min_branch_step": min_branch_step,
            "refractory_steps": refractory_steps,
            "high_quantile": high_quantile,
            "mid_quantile_low": mid_quantile_low,
            "mid_quantile_high": mid_quantile_high,
            "low_quantile": low_quantile,
            "perturb_a_std": perturb_a_std,
            "perturb_p_std": perturb_p_std,
            "perturb_lagrangian_xy_std": perturb_lag_xy_std,
            "energy_min_remaining_steps": energy_min_remaining_steps,
            "energy_min_samples": energy_min_remaining_samples,
        },
    }
    write_json(out_dir / "branch_plan_meta.json", plan_meta)
    summary = {
        "status": "ok",
        "n_branches": len(rows),
        "n_ready": sum(1 for row in rows if row["status"] == "exists"),
        "allow_heavy": bool(allow_heavy),
        "plan": str(plan_path),
    }
    write_json(output_root / "c2_branching_simulation_summary.json", summary)
    log_event(
        f"C2 branching simulation done n_branches={len(rows)} n_ready={summary['n_ready']} plan={plan_path}",
        component="c2-branch",
    )
    return summary


def _stats_feature(arr: np.ndarray) -> list[float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.quantile(finite, 0.10)),
        float(np.quantile(finite, 0.90)),
    ]


def _feature_from_metrics(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        parts: list[float] = []
        for key in ("delta_h_best", "cluster_tv"):
            arr = _safe_arr(data, key)
            if arr is not None:
                parts.extend(_stats_feature(arr))
        if not parts and "delta_h_map" in data.files:
            parts.extend(_stats_feature(np.asarray(data["delta_h_map"])))
    if not parts:
        raise ValueError(f"No metric feature arrays found in {path}")
    return np.asarray(parts, dtype=np.float64)


def _feature_from_apf(apf_dir: Path, *, max_chunks: int, max_snapshots_per_chunk: int) -> np.ndarray:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    selected = chunks[-max(1, int(max_chunks)) :]
    parts: list[float] = []
    for path, _s0, _s1, _idx in selected:
        with np.load(path, allow_pickle=False) as data:
            for key in ("A", "P", "F", "lagrangian_xy"):
                if key not in data.files:
                    continue
                arr = np.asarray(data[key])
                if arr.ndim > 0 and arr.shape[0] > max_snapshots_per_chunk:
                    idxs = np.linspace(0, arr.shape[0] - 1, max_snapshots_per_chunk).astype(int)
                    arr = arr[idxs]
                parts.extend(_stats_feature(arr))
    if not parts:
        raise ValueError(f"No usable APF arrays found in {apf_dir}")
    return np.asarray(parts, dtype=np.float64)


def _feature_from_branch(branch_dir: Path, *, max_chunks: int, max_snapshots_per_chunk: int) -> np.ndarray:
    feature_path = branch_dir / "branch_feature.npz"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as data:
            key = "feature" if "feature" in data.files else data.files[0]
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    metrics_path = branch_dir / "metrics.npz"
    if metrics_path.exists():
        return _feature_from_metrics(metrics_path)
    return _feature_from_apf(branch_dir / "apf_logs", max_chunks=max_chunks, max_snapshots_per_chunk=max_snapshots_per_chunk)


def _pairwise_mean_distance(features: list[np.ndarray]) -> float:
    if len(features) < 2:
        return float("nan")
    dists = []
    min_len = min(int(f.size) for f in features)
    for a, b in combinations(features, 2):
        aa = np.asarray(a[:min_len], dtype=np.float64)
        bb = np.asarray(b[:min_len], dtype=np.float64)
        dists.append(float(np.linalg.norm(aa - bb) / math.sqrt(max(1, min_len))))
    return float(np.mean(dists)) if dists else float("nan")


def _field_weights(raw: Any) -> dict[str, float]:
    if raw is None:
        return {"A": 1.0, "P": 0.25, "F": 0.25}
    try:
        items = raw.items()
    except Exception:
        return {"A": 1.0, "P": 0.25, "F": 0.25}
    out: dict[str, float] = {}
    for key, value in items:
        weight = float(value)
        if weight > 0.0:
            out[str(key)] = weight
    return out or {"A": 1.0}


def _branching_metric_mode(raw: Any) -> str:
    value = str(raw or "apf").strip().lower()
    aliases = {
        "field": "apf",
        "field_l2": "apf",
        "apf_l2": "apf",
        "future_apf_multiscale_l2": "apf",
        "clip": "clip_chamfer",
        "clip_cloud": "clip_chamfer",
        "embedding_chamfer": "clip_chamfer",
        "clip_chamfer_cosine": "clip_chamfer",
        "future_clip_chamfer_cosine": "clip_chamfer",
    }
    value = aliases.get(value, value)
    if value not in {"apf", "clip_chamfer"}:
        raise ValueError(f"Unknown C2 branching metric {raw!r}; use 'apf' or 'clip_chamfer'.")
    return value


def _pool_spatial(arr: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    scale = int(scale)
    if scale <= 1 or x.ndim < 4:
        return x
    y_axis, x_axis = 1, 2
    h = int(x.shape[y_axis])
    w = int(x.shape[x_axis])
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 < scale or w2 < scale:
        return x
    slicer = [slice(None)] * x.ndim
    slicer[y_axis] = slice(0, h2)
    slicer[x_axis] = slice(0, w2)
    x = x[tuple(slicer)]
    if x.ndim == 4:
        t, _h, _w = x.shape[:3]
        rest = x.shape[3:]
        return x.reshape((t, h2 // scale, scale, w2 // scale, scale) + rest).mean(axis=(2, 4))
    prefix = x.shape[:y_axis]
    suffix = x.shape[x_axis + 1 :]
    return x.reshape(prefix + (h2 // scale, scale, w2 // scale, scale) + suffix).mean(axis=(y_axis + 1, y_axis + 3))


def _field_l2(a: np.ndarray, b: np.ndarray, *, scales: list[int]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n = min(int(aa.shape[0]), int(bb.shape[0]))
    if n < 1:
        return float("nan")
    aa = aa[:n]
    bb = bb[:n]
    vals = []
    for scale in scales:
        pa = _pool_spatial(aa, int(scale))
        pb = _pool_spatial(bb, int(scale))
        diff = np.asarray(pa, dtype=np.float32) - np.asarray(pb, dtype=np.float32)
        vals.append(float(np.sqrt(np.mean(diff * diff))))
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _branch_field_series(
    branch_dir: Path,
    *,
    weights: dict[str, float],
    max_chunks: int,
    max_frames: int,
) -> dict[str, np.ndarray]:
    apf_dir = branch_dir / "apf_logs"
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    selected = chunks[: max(1, int(max_chunks))]
    series: dict[str, list[np.ndarray]] = {key: [] for key in weights}
    for path, _s0, _s1, _idx in selected:
        with np.load(path, allow_pickle=False) as data:
            for key in weights:
                if key not in data.files:
                    continue
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim < 3 or arr.shape[0] < 1:
                    continue
                series[key].append(arr)
    out: dict[str, np.ndarray] = {}
    for key, parts in series.items():
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0)
        if max_frames > 0 and arr.shape[0] > max_frames:
            idxs = np.linspace(0, arr.shape[0] - 1, int(max_frames)).astype(int)
            arr = arr[idxs]
        out[key] = arr
    if not out:
        raise ValueError(f"No weighted APF fields {sorted(weights)} found in {apf_dir}")
    return out


def _pairwise_future_field_divergence(
    branch_dirs: list[Path],
    *,
    weights: dict[str, float],
    scales: list[int],
    max_chunks: int,
    max_frames: int,
) -> tuple[float, dict[str, Any]]:
    fields = []
    used_dirs = []
    for branch_dir in branch_dirs:
        try:
            fields.append(
                _branch_field_series(
                    branch_dir,
                    weights=weights,
                    max_chunks=max_chunks,
                    max_frames=max_frames,
                )
            )
            used_dirs.append(branch_dir)
        except Exception:
            continue
    if len(fields) < 2:
        return float("nan"), {"metric": "future_apf_multiscale_l2", "n_branches": len(fields), "n_pairs": 0}
    pair_vals = []
    key_vals: dict[str, list[float]] = {key: [] for key in weights}
    for i, j in combinations(range(len(fields)), 2):
        common = [key for key in weights if key in fields[i] and key in fields[j]]
        weighted = []
        for key in common:
            d = _field_l2(fields[i][key], fields[j][key], scales=scales)
            if np.isfinite(d):
                key_vals[key].append(d)
                weighted.append(float(weights[key]) * d)
        if weighted:
            denom = sum(float(weights[key]) for key in common)
            pair_vals.append(float(sum(weighted) / max(denom, 1e-12)))
    score = float(np.mean(pair_vals)) if pair_vals else float("nan")
    detail: dict[str, Any] = {
        "metric": "future_apf_multiscale_l2",
        "n_branches": len(fields),
        "n_pairs": len(pair_vals),
        "field_keys": ",".join(key for key in weights if key_vals.get(key)),
        "scales": ",".join(str(int(s)) for s in scales),
        "max_future_frames": int(max_frames),
    }
    for key, vals in key_vals.items():
        if vals:
            detail[f"{key}_divergence"] = float(np.mean(vals))
    return score, detail


def _render_apf_rgb(fields: dict[str, np.ndarray]) -> np.ndarray:
    if "P" not in fields:
        raise ValueError("CLIP divergence requires P in branch APF fields.")
    p = np.asarray(fields["P"], dtype=np.float32)
    if p.ndim != 4:
        raise ValueError(f"P must have shape (T,H,W,C), got {p.shape}.")
    if p.shape[-1] < 3:
        reps = int(math.ceil(3 / max(1, int(p.shape[-1]))))
        p3 = np.tile(p, (1, 1, 1, reps))[..., :3]
    else:
        p3 = p[..., :3]
    if "A" in fields:
        a = np.asarray(fields["A"], dtype=np.float32)
        if a.ndim != 4 or a.shape[0] != p.shape[0]:
            raise ValueError(f"A must have shape (T,H,W,C) with T={p.shape[0]}, got {a.shape}.")
        return np.clip(np.sum(a, axis=-1, keepdims=True) * p3, 0.0, 1.0).astype(np.float32)
    return np.clip(p3, 0.0, 1.0).astype(np.float32)


def _normalize_embeddings(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _embedding_chamfer_cosine(z_a: np.ndarray, z_b: np.ndarray) -> float:
    a = _normalize_embeddings(np.asarray(z_a, dtype=np.float64))
    b = _normalize_embeddings(np.asarray(z_b, dtype=np.float64))
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[0] < 1:
        return float("nan")
    d = 1.0 - (a @ b.T)
    return float(0.5 * (np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))))


def _clip_embedding_cache_path(
    branch_dir: Path,
    *,
    cache_dir: Path,
    foundation_model: str,
    max_chunks: int,
    max_frames: int,
) -> Path:
    payload = stable_json(
        {
            "branch_dir": str(branch_dir.resolve()),
            "foundation_model": str(foundation_model),
            "max_chunks": int(max_chunks),
            "max_frames": int(max_frames),
            "version": "c2_branch_clip_embeddings_v1",
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    safe_name = branch_dir.name.replace("/", "__")
    return cache_dir / f"{safe_name}_{digest}.npz"


def _load_or_compute_branch_clip_embeddings(
    branch_dir: Path,
    *,
    fm: Any,
    cache_dir: Path,
    foundation_model: str,
    max_chunks: int,
    max_frames: int,
    force: bool,
) -> tuple[np.ndarray, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _clip_embedding_cache_path(
        branch_dir,
        cache_dir=cache_dir,
        foundation_model=foundation_model,
        max_chunks=max_chunks,
        max_frames=max_frames,
    )
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            return np.asarray(data["z"], dtype=np.float32), cache_path

    import jax

    fields = _branch_field_series(
        branch_dir,
        weights={"A": 1.0, "P": 1.0},
        max_chunks=max_chunks,
        max_frames=max_frames,
    )
    frames = _render_apf_rgb(fields)
    zs: list[np.ndarray] = []
    for frame in frames:
        z = jax.device_get(fm.embed_img(frame))
        zs.append(np.asarray(z, dtype=np.float32).reshape(-1))
    z_arr = _normalize_embeddings(np.stack(zs, axis=0)).astype(np.float32)
    np.savez_compressed(
        cache_path,
        z=z_arr,
        branch_dir=np.asarray(str(branch_dir)),
        foundation_model=np.asarray(str(foundation_model)),
        max_chunks=np.asarray(int(max_chunks), dtype=np.int32),
        max_frames=np.asarray(int(max_frames), dtype=np.int32),
    )
    return z_arr, cache_path


def _pairwise_future_clip_chamfer_divergence(
    branch_dirs: list[Path],
    *,
    fm: Any,
    cache_dir: Path,
    foundation_model: str,
    max_chunks: int,
    max_frames: int,
    force_cache: bool,
) -> tuple[float, dict[str, Any]]:
    embeddings: list[np.ndarray] = []
    cache_paths: list[str] = []
    for branch_dir in branch_dirs:
        try:
            z, cache_path = _load_or_compute_branch_clip_embeddings(
                branch_dir,
                fm=fm,
                cache_dir=cache_dir,
                foundation_model=foundation_model,
                max_chunks=max_chunks,
                max_frames=max_frames,
                force=force_cache,
            )
            embeddings.append(z)
            cache_paths.append(str(cache_path))
        except Exception:
            continue
    if len(embeddings) < 2:
        return float("nan"), {"metric": "future_clip_chamfer_cosine", "n_branches": len(embeddings), "n_pairs": 0}
    pair_vals = []
    for i, j in combinations(range(len(embeddings)), 2):
        d = _embedding_chamfer_cosine(embeddings[i], embeddings[j])
        if np.isfinite(d):
            pair_vals.append(float(d))
    score = float(np.mean(pair_vals)) if pair_vals else float("nan")
    return score, {
        "metric": "future_clip_chamfer_cosine",
        "n_branches": len(embeddings),
        "n_pairs": len(pair_vals),
        "foundation_model": str(foundation_model),
        "max_future_frames": int(max_frames),
        "clip_embedding_cache_n": len(cache_paths),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(finite)) < 2:
        return float("nan")
    xx = xx[finite]
    yy = yy[finite]
    sx = float(np.std(xx))
    sy = float(np.std(yy))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _branching_correlation_summary(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    x: list[float] = []
    y: list[float] = []
    for row in score_rows:
        try:
            dh = float(row.get("delta_h", "nan"))
            score = float(row.get("branching_score", "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(dh) and np.isfinite(score):
            x.append(dh)
            y.append(score)
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    pearson = _pearson_corr(xx, yy)
    spearman = _pearson_corr(_average_ranks(xx), _average_ranks(yy)) if xx.size >= 2 else float("nan")
    return {
        "n": int(xx.size),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "delta_h_min": float(np.nanmin(xx)) if xx.size else float("nan"),
        "delta_h_max": float(np.nanmax(xx)) if xx.size else float("nan"),
        "branching_score_min": float(np.nanmin(yy)) if yy.size else float("nan"),
        "branching_score_max": float(np.nanmax(yy)) if yy.size else float("nan"),
    }


def metrics(
    config_path: str | Path,
    *,
    smoke: bool = False,
    branching_metric: str | None = None,
    allow_stale_branch_plan: bool = False,
    force_clip_cache: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    bcfg = _branch_cfg(cfg)
    branch_root = _branch_root(cfg, output_root)
    out_dir = ensure_dir(output_root / "c2_branching")
    plan_path = out_dir / "branch_plan.csv"
    metric_mode = _branching_metric_mode(branching_metric or _get(bcfg, "branching_metric", _get(bcfg, "divergence_metric", "apf")))
    log_event(f"C2 branching metrics start smoke={smoke} metric={metric_mode} plan={plan_path}", component="c2-branch")
    if smoke and not plan_path.exists():
        _write_smoke_fixture(output_root, branch_root)
    if not plan_path.exists():
        summary = {"status": "skipped", "reason": f"missing branch plan {plan_path}"}
        write_json(output_root / "c2_branching_metrics_summary.json", summary)
        log_event(f"C2 branching metrics skipped missing plan={plan_path}", component="c2-branch")
        return summary

    max_chunks = int(_get(bcfg, "feature_max_apf_chunks", 4))
    max_snapshots = int(_get(bcfg, "feature_max_snapshots_per_chunk", 8))
    max_future_frames = int(_get(bcfg, "future_max_frames", max_snapshots))
    field_scales = [int(x) for x in _get(bcfg, "future_field_scales", [1, 2, 4])]
    weights = _field_weights(_get(bcfg, "future_field_weights", None))
    clip_foundation_model = str(_get(bcfg, "clip_foundation_model", "clip"))
    clip_cache_raw = _get(bcfg, "clip_embedding_cache_dir", None)
    clip_cache_dir = ensure_dir(resolve_path(clip_cache_raw) if clip_cache_raw else out_dir / "clip_embedding_cache")
    clip_fm = None
    if metric_mode == "clip_chamfer" and not smoke:
        import foundation_models

        clip_fm = foundation_models.create_foundation_model(clip_foundation_model)
    allow_feature_fallback = bool(smoke or _get(bcfg, "allow_debug_feature_fallback", False))
    plan_rows = read_csv(plan_path)
    if not smoke and not allow_stale_branch_plan:
        c2_cfg = cfg.get("c2", {})
        rollout_config = _c2_rollout_config(c2_cfg)
        flat_args = _c2_flat_metric_args(rollout_config)
        _validate_branch_plan(plan_path, plan_rows, flat_args)
    elif allow_stale_branch_plan:
        log_event("C2 branching metrics using branch_plan without version/upstream validation by explicit request", component="c2-branch")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in plan_rows:
        key = (str(row["traj_id"]), str(row["pair_id"]), str(row["condition"]))
        groups.setdefault(key, []).append(row)
    log_event(f"C2 branching metrics loaded n_plan_rows={len(plan_rows)} n_groups={len(groups)}", component="c2-branch")

    score_rows: list[dict[str, Any]] = []
    group_items = sorted(groups.items())
    for group_idx, ((traj_id, pair_id, condition), rows) in enumerate(group_items, start=1):
        if group_idx == 1 or group_idx == len(group_items) or group_idx % 5 == 0:
            log_event(
                f"C2 branching metrics group {group_idx}/{len(group_items)} traj={traj_id} pair={pair_id} condition={condition}",
                component="c2-branch",
        )
        valid_rows = [
            row
            for row in rows
            if _branch_output_ok(
                Path(str(row["branch_dir"])),
                expected_metadata=_expected_resume_metadata_from_plan_row(row),
            )
        ]
        branch_dirs = [Path(str(row["branch_dir"])) for row in valid_rows]
        if metric_mode == "clip_chamfer":
            if clip_fm is None:
                raise RuntimeError("CLIP foundation model was not initialized.")
            score, detail = _pairwise_future_clip_chamfer_divergence(
                branch_dirs,
                fm=clip_fm,
                cache_dir=clip_cache_dir,
                foundation_model=clip_foundation_model,
                max_chunks=max_chunks,
                max_frames=max_future_frames,
                force_cache=force_clip_cache,
            )
        else:
            score, detail = _pairwise_future_field_divergence(
                branch_dirs,
                weights=weights,
                scales=field_scales,
                max_chunks=max_chunks,
                max_frames=max_future_frames,
            )
        used = list(valid_rows)
        metric_name = str(detail.get("metric", "future_apf_multiscale_l2"))
        fallback_used = False
        if metric_mode == "apf" and not np.isfinite(score) and allow_feature_fallback:
            features = []
            used = []
            for row in valid_rows:
                branch_dir = Path(str(row["branch_dir"]))
                try:
                    features.append(_feature_from_branch(branch_dir, max_chunks=max_chunks, max_snapshots_per_chunk=max_snapshots))
                    used.append(row)
                except Exception:
                    continue
            if len(features) >= 2:
                score = _pairwise_mean_distance(features)
                metric_name = "debug_compact_feature_l2"
                fallback_used = True
                detail = {"metric": metric_name, "n_branches": len(features), "n_pairs": int(len(features) * (len(features) - 1) // 2)}
        if not np.isfinite(score):
            continue
        if not used:
            used = rows
        delta_h_vals = [float(row.get("delta_h", "nan")) for row in used]
        step_vals = [float(row.get("step", "nan")) for row in used]
        row_out = {
            "traj_id": traj_id,
            "pair_id": int(float(pair_id)),
            "condition": condition,
            "step": float(np.nanmedian(step_vals)),
            "delta_h": float(np.nanmedian(delta_h_vals)),
            "branching_score": float(score),
            "branching_metric": metric_name,
            "used_debug_feature_fallback": bool(fallback_used),
            "n_branches": int(detail.get("n_branches", len(used))),
            "n_branch_pairs": int(detail.get("n_pairs", 0)),
        }
        for key, value in detail.items():
            if key in row_out:
                continue
            row_out[str(key)] = value
        score_rows.append(row_out)

    contrast_rows: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in score_rows:
        key = (str(row["traj_id"]), int(row["pair_id"]))
        by_pair.setdefault(key, {})[str(row["condition"])] = row
    for (traj_id, pair_id), conds in sorted(by_pair.items()):
        if "high" not in conds or "low" not in conds:
            continue
        high = conds["high"]
        low = conds["low"]
        contrast_rows.append(
            {
                "traj_id": traj_id,
                "pair_id": pair_id,
                "high_step": high["step"],
                "low_step": low["step"],
                "high_delta_h": high["delta_h"],
                "low_delta_h": low["delta_h"],
                "high_branching_score": high["branching_score"],
                "low_branching_score": low["branching_score"],
                "delta_branching_score": float(high["branching_score"]) - float(low["branching_score"]),
            }
        )

    suffix = "" if metric_mode == "apf" else f"_{metric_mode}"
    scores_path = out_dir / f"branching_scores{suffix}.csv"
    contrasts_path = out_dir / f"branching_pair_contrasts{suffix}.csv"
    correlation_path = out_dir / f"branching_delta_h_correlation{suffix}.csv"
    correlation_summary = _branching_correlation_summary(score_rows)
    write_csv(scores_path, score_rows)
    write_csv(contrasts_path, contrast_rows)
    write_csv(correlation_path, [correlation_summary])
    summary = {
        "status": "ok",
        "branching_metric_mode": metric_mode,
        "n_scores": len(score_rows),
        "n_pairs": len(contrast_rows),
        "branching_sign_test": sign_test_greater(row["delta_branching_score"] for row in contrast_rows),
        "branching_delta_h_correlation": correlation_summary,
        "scores": str(scores_path),
        "contrasts": str(contrasts_path),
        "correlation": str(correlation_path),
    }
    summary_path = output_root / f"c2_branching_metrics_summary{suffix}.json"
    write_json(summary_path, summary)
    log_event(
        f"C2 branching metrics done metric={metric_mode} n_scores={len(score_rows)} n_pairs={len(contrast_rows)} scores={scores_path}",
        component="c2-branch",
    )
    return summary


def run(
    config_path: str | Path,
    *,
    layer: str = "all",
    smoke: bool = False,
    force: bool = False,
    allow_heavy: bool = False,
    dry_run: bool = False,
    branching_metric: str | None = None,
    allow_stale_branch_plan: bool = False,
    force_clip_cache: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if layer in {"simulation", "all"}:
        out["simulation"] = simulation(config_path, smoke=smoke, force=force, allow_heavy=allow_heavy, dry_run=dry_run)
    if layer in {"metrics", "all"}:
        out["metrics"] = metrics(
            config_path,
            smoke=smoke,
            branching_metric=branching_metric,
            allow_stale_branch_plan=allow_stale_branch_plan,
            force_clip_cache=force_clip_cache,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2 branching-sensitivity experiment for the paper suite.")
    parser.add_argument("config")
    parser.add_argument("--layer", choices=["simulation", "metrics", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--branching-metric", choices=["apf", "clip_chamfer"], default=None)
    parser.add_argument("--allow-stale-branch-plan", action="store_true")
    parser.add_argument("--force-clip-cache", action="store_true")
    args = parser.parse_args(argv)
    print(
        run(
            args.config,
            layer=args.layer,
            smoke=args.smoke,
            force=args.force,
            allow_heavy=args.allow_heavy,
            dry_run=args.dry_run,
            branching_metric=args.branching_metric,
            allow_stale_branch_plan=args.allow_stale_branch_plan,
            force_clip_cache=args.force_clip_cache,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
