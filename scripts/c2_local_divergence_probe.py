from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import list_apf_chunks
from paper_suite_common import ensure_dir, load_config, resolve_path, write_csv, write_json


DEFAULT_FIELD_WEIGHTS = {"A": 1.0, "P": 0.25, "F": 0.25}
DEFAULT_DIVERGENCE_MODES = ("apf", "clip_chamfer")


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _parse_ints(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(default)


def _parse_field_weights(raw: str | None) -> dict[str, float]:
    if raw is None or str(raw).strip() == "":
        return dict(DEFAULT_FIELD_WEIGHTS)
    out: dict[str, float] = {}
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, value = part.split(":", 1)
            weight = float(value)
        else:
            key, weight = part, 1.0
        key = key.strip()
        if key and weight > 0.0:
            out[key] = float(weight)
    return out or dict(DEFAULT_FIELD_WEIGHTS)


def _parse_modes(raw: str | None) -> set[str]:
    if raw is None or str(raw).strip() == "":
        return set(DEFAULT_DIVERGENCE_MODES)
    aliases = {
        "old": "apf",
        "field": "apf",
        "field_l2": "apf",
        "apf_l2": "apf",
        "clip": "clip_chamfer",
        "clip_cloud": "clip_chamfer",
        "embedding_chamfer": "clip_chamfer",
    }
    out: set[str] = set()
    for part in str(raw).split(","):
        key = part.strip().lower()
        if not key:
            continue
        key = aliases.get(key, key)
        if key not in {"apf", "clip_chamfer"}:
            raise ValueError(f"Unknown divergence mode {part!r}; use apf,clip_chamfer.")
        out.add(key)
    return out or set(DEFAULT_DIVERGENCE_MODES)


def _trajectory_root(cfg: Any) -> Path:
    c2_cfg = cfg.get("c2", {})
    raw = _get(c2_cfg, "trajectory_root", "experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k")
    path = resolve_path(raw)
    if path is None:
        raise ValueError("Could not resolve c2.trajectory_root.")
    return path


def _path_from_manifest(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw).strip() == "":
        return default
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return root / path


def _iter_trajectories(root: Path, *, include_random: bool) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    items: list[dict[str, Any]] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            kind = str(row.get("candidate_kind", "optimized")).strip().lower()
            if kind != "optimized" and not include_random:
                continue
            traj_id = str(row.get("traj_id", f"traj_{idx:05d}"))
            traj_dir = _path_from_manifest(root, row.get("traj_dir"), default=root / traj_id)
            apf_dir = _path_from_manifest(root, row.get("apf_dir"), default=traj_dir / "apf_logs")
            metrics_path = _path_from_manifest(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            items.append(
                {
                    "traj_id": traj_id,
                    "candidate_kind": kind,
                    "candidate_label": str(row.get("candidate_label", kind)),
                    "traj_dir": traj_dir,
                    "apf_dir": apf_dir,
                    "metrics_path": metrics_path,
                }
            )
    if items:
        return items

    for traj_dir in sorted(root.glob("flow_opt*")):
        if not traj_dir.is_dir():
            continue
        kind = "random" if "_random_" in traj_dir.name else "optimized"
        if kind != "optimized" and not include_random:
            continue
        items.append(
            {
                "traj_id": traj_dir.name,
                "candidate_kind": kind,
                "candidate_label": kind,
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
            }
        )
    return items


def _load_delta_h(metrics_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(metrics_path, allow_pickle=False) as data:
        if "delta_h_best" in data.files:
            delta_h = np.asarray(data["delta_h_best"], dtype=np.float64).reshape(-1)
        elif "delta_h_map" in data.files:
            dh_map = np.asarray(data["delta_h_map"], dtype=np.float64)
            selected = int(np.asarray(data.get("delta_h_selected_tau_idx", np.asarray(0))).reshape(-1)[0])
            delta_h = np.asarray(dh_map[selected], dtype=np.float64).reshape(-1)
        else:
            raise ValueError(f"{metrics_path} has neither delta_h_best nor delta_h_map.")
        if "delta_h_window_center_steps" in data.files:
            centers = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        elif "delta_h_window_start_steps" in data.files and "delta_h_window_end_steps" in data.files:
            starts = np.asarray(data["delta_h_window_start_steps"], dtype=np.float64).reshape(-1)
            ends = np.asarray(data["delta_h_window_end_steps"], dtype=np.float64).reshape(-1)
            centers = 0.5 * (starts + ends)
        else:
            centers = np.arange(delta_h.size, dtype=np.float64)
        meta = {}
        for key in ("delta_h_selected_tau_steps", "delta_h_selected_tau_idx"):
            if key in data.files:
                arr = np.asarray(data[key]).reshape(-1)
                meta[key] = arr[0].item() if arr.size else None
    n = min(delta_h.size, centers.size)
    if n == 0:
        raise ValueError(f"{metrics_path} has empty Delta-H arrays.")
    return centers[:n], delta_h[:n], meta


def _steps_for_chunk(data: np.lib.npyio.NpzFile, *, start: int, end: int, n: int) -> np.ndarray:
    if "steps" in data.files:
        arr = np.asarray(data["steps"], dtype=np.float64).reshape(-1)
        if arr.size == n:
            return arr
    if "state_t" in data.files:
        state_t = np.asarray(data["state_t"], dtype=np.float64).reshape(-1)
        if state_t.size == n:
            return state_t
    if n <= 1:
        return np.asarray([float(start)], dtype=np.float64)
    return np.linspace(float(start), float(end), int(n), dtype=np.float64)


def _metadata_time_len(data: np.lib.npyio.NpzFile) -> int | None:
    for key in ("steps", "state_t"):
        if key in data.files:
            n = int(np.asarray(data[key]).reshape(-1).size)
            if n >= 1:
                return n
    return None


def _normalize_apf_array(arr: np.ndarray, data: np.lib.npyio.NpzFile) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    time_len = _metadata_time_len(data)
    if time_len is not None:
        time_axes = [axis for axis, size in enumerate(x.shape) if int(size) == int(time_len)]
        if time_axes:
            axis = 0 if 0 in time_axes else time_axes[0]
            if axis != 0:
                x = np.moveaxis(x, axis, 0)

    while x.ndim > 4:
        if int(x.shape[1]) == 1:
            x = np.squeeze(x, axis=1)
            continue
        tail_singletons = [axis for axis in range(4, x.ndim) if int(x.shape[axis]) == 1]
        if tail_singletons:
            x = np.squeeze(x, axis=tail_singletons[0])
            continue
        other_singletons = [axis for axis in range(1, x.ndim) if int(x.shape[axis]) == 1]
        if other_singletons:
            x = np.squeeze(x, axis=other_singletons[0])
            continue
        break

    if time_len is None and x.ndim >= 5 and int(x.shape[0]) == 1:
        x = x[0]
    if x.ndim >= 5 and int(x.shape[1]) == 1:
        x = x[:, 0]
    return x


def _avg_pool_spatial(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    factor = int(factor)
    if factor <= 1 or x.ndim < 4:
        return x
    h = int(x.shape[1])
    w = int(x.shape[2])
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    if h2 < factor or w2 < factor:
        return x
    x = x[:, :h2, :w2, ...]
    rest = x.shape[3:]
    return x.reshape((x.shape[0], h2 // factor, factor, w2 // factor, factor) + rest).mean(axis=(2, 4))


def _load_apf_series(
    apf_dir: Path,
    *,
    field_weights: dict[str, float],
    spatial_downsample: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    step_parts: list[np.ndarray] = []
    field_parts: dict[str, list[np.ndarray]] = {key: [] for key in field_weights}
    for path, start, end, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            n = 0
            for key in field_weights:
                if key not in data.files:
                    continue
                arr = _normalize_apf_array(np.asarray(data[key], dtype=np.float32), data)
                if arr.ndim < 4 or arr.shape[0] < 1:
                    continue
                arr = _avg_pool_spatial(arr, spatial_downsample)
                field_parts[key].append(arr)
                n = max(n, int(arr.shape[0]))
            if n > 0:
                step_parts.append(_steps_for_chunk(data, start=start, end=end, n=n))
    fields = {key: np.concatenate(parts, axis=0) for key, parts in field_parts.items() if parts}
    if not step_parts or not fields:
        raise ValueError(f"No usable APF field arrays found in {apf_dir}.")
    steps = np.concatenate(step_parts).astype(np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    fields = {key: value[order] for key, value in fields.items()}
    keep = np.ones(steps.shape, dtype=bool)
    if steps.size > 1:
        keep[1:] = np.diff(steps) > 0
    steps = steps[keep]
    fields = {key: value[keep] for key, value in fields.items()}
    return steps, fields


def _nearest_indices(steps: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(steps, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    pos = np.searchsorted(s, t, side="left")
    pos = np.clip(pos, 0, s.size - 1)
    prev = np.clip(pos - 1, 0, s.size - 1)
    choose_prev = np.abs(s[prev] - t) <= np.abs(s[pos] - t)
    idx = np.where(choose_prev, prev, pos).astype(np.int64)
    err = np.abs(s[idx] - t)
    return idx, err


def _pool_spatial(arr: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    scale = int(scale)
    if scale <= 1 or x.ndim < 4:
        return x
    h = int(x.shape[1])
    w = int(x.shape[2])
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 < scale or w2 < scale:
        return x
    x = x[:, :h2, :w2, ...]
    rest = x.shape[3:]
    return x.reshape((x.shape[0], h2 // scale, scale, w2 // scale, scale) + rest).mean(axis=(2, 4))


def _field_l2(a: np.ndarray, b: np.ndarray, *, scales: list[int]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n = min(int(aa.shape[0]), int(bb.shape[0]))
    if n < 1:
        return float("nan")
    aa = aa[:n]
    bb = bb[:n]
    vals: list[float] = []
    for scale in scales:
        pa = _pool_spatial(aa, int(scale))
        pb = _pool_spatial(bb, int(scale))
        diff = pa - pb
        vals.append(float(np.sqrt(np.mean(diff * diff))))
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _future_distance(
    *,
    steps: np.ndarray,
    fields: dict[str, np.ndarray],
    t0: float,
    offset_steps: int,
    horizon_steps: int,
    max_future_frames: int,
    field_weights: dict[str, float],
    scales: list[int],
    max_step_error: float,
) -> float:
    if offset_steps == 0:
        return float("nan")
    t1 = float(t0) + float(offset_steps)
    if t0 < steps[0] or t1 < steps[0] or t0 + horizon_steps > steps[-1] or t1 + horizon_steps > steps[-1]:
        return float("nan")
    n_frames = max(2, int(max_future_frames))
    rel = np.linspace(0.0, float(horizon_steps), n_frames, dtype=np.float64)
    idx0, err0 = _nearest_indices(steps, float(t0) + rel)
    idx1, err1 = _nearest_indices(steps, float(t1) + rel)
    valid = (err0 <= max_step_error) & (err1 <= max_step_error)
    if int(np.sum(valid)) < 2:
        return float("nan")
    weighted: list[float] = []
    denom = 0.0
    for key, weight in field_weights.items():
        if key not in fields:
            continue
        d = _field_l2(fields[key][idx0[valid]], fields[key][idx1[valid]], scales=scales)
        if np.isfinite(d):
            weighted.append(float(weight) * d)
            denom += float(weight)
    if not weighted or denom <= 0.0:
        return float("nan")
    return float(sum(weighted) / max(denom, 1e-12))


def _future_indices(
    *,
    steps: np.ndarray,
    t0: float,
    offset_steps: int,
    horizon_steps: int,
    max_future_frames: int,
    max_step_error: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if offset_steps == 0:
        return None
    t1 = float(t0) + float(offset_steps)
    if t0 < steps[0] or t1 < steps[0] or t0 + horizon_steps > steps[-1] or t1 + horizon_steps > steps[-1]:
        return None
    n_frames = max(2, int(max_future_frames))
    rel = np.linspace(0.0, float(horizon_steps), n_frames, dtype=np.float64)
    idx0, err0 = _nearest_indices(steps, float(t0) + rel)
    idx1, err1 = _nearest_indices(steps, float(t1) + rel)
    valid = (err0 <= max_step_error) & (err1 <= max_step_error)
    if int(np.sum(valid)) < 2:
        return None
    return idx0[valid], idx1[valid]


def _render_apf_rgb(fields: dict[str, np.ndarray]) -> np.ndarray:
    if "P" not in fields:
        raise ValueError("CLIP divergence requires P in APF fields.")
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
        intensity = np.sum(a, axis=-1, keepdims=True)
        return np.clip(intensity * p3, 0.0, 1.0).astype(np.float32)
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


def _future_clip_chamfer(
    *,
    embeddings: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
) -> float:
    if embeddings is None:
        return float("nan")
    if idx0.size < 1 or idx1.size < 1:
        return float("nan")
    return _embedding_chamfer_cosine(embeddings[idx0], embeddings[idx1])


def _load_or_compute_clip_embeddings(
    *,
    item: dict[str, Any],
    steps: np.ndarray,
    fields: dict[str, np.ndarray],
    cache_dir: Path,
    foundation_model: str,
    spatial_downsample: int,
    force: bool,
) -> tuple[np.ndarray, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    traj_id = str(item["traj_id"]).replace("/", "__")
    cache_path = cache_dir / f"{traj_id}_{foundation_model.replace('/', '_')}_ds{int(spatial_downsample)}_embeddings.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            z = np.asarray(data["z"], dtype=np.float32)
            cached_steps = np.asarray(data["steps"], dtype=np.float64).reshape(-1)
            if "spatial_downsample" in data.files:
                cached_ds = int(np.asarray(data["spatial_downsample"]).reshape(-1)[0])
            else:
                cached_ds = -1
        if (
            z.shape[0] == steps.size
            and cached_steps.shape == steps.shape
            and np.allclose(cached_steps, steps)
            and cached_ds == int(spatial_downsample)
        ):
            return z, cache_path

    import jax
    import foundation_models

    frames = _render_apf_rgb(fields)
    fm = foundation_models.create_foundation_model(str(foundation_model))
    zs: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i == 0 or i == frames.shape[0] - 1 or (i + 1) % 50 == 0:
            print(f"[c2-local-divergence] CLIP embed {traj_id} {i + 1}/{frames.shape[0]}", flush=True)
        z = jax.device_get(fm.embed_img(frame))
        zs.append(np.asarray(z, dtype=np.float32).reshape(-1))
    emb = _normalize_embeddings(np.stack(zs, axis=0)).astype(np.float32)
    np.savez_compressed(
        cache_path,
        z=emb,
        steps=np.asarray(steps, dtype=np.float64),
        traj_id=np.asarray(traj_id),
        foundation_model=np.asarray(str(foundation_model)),
        spatial_downsample=np.asarray(int(spatial_downsample), dtype=np.int32),
    )
    return emb, cache_path


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


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(finite)) < 2:
        return float("nan")
    xx = xx[finite]
    yy = yy[finite]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _summary(rows: list[dict[str, Any]], *, label: str, value_key: str) -> dict[str, Any]:
    usable = [row for row in rows if value_key in row and str(row.get(value_key, "")).strip() != ""]
    x = np.asarray([float(row["delta_h"]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row[value_key]) for row in usable], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    return {
        "label": label,
        "n": int(x.size),
        "pearson_r": _corr(x, y),
        "spearman_r": _corr(_average_ranks(x), _average_ranks(y)) if x.size >= 2 else float("nan"),
        "delta_h_min": float(np.nanmin(x)) if x.size else float("nan"),
        "delta_h_max": float(np.nanmax(x)) if x.size else float("nan"),
        "local_divergence_min": float(np.nanmin(y)) if y.size else float("nan"),
        "local_divergence_max": float(np.nanmax(y)) if y.size else float("nan"),
    }


def _prefixed_summary(rows: list[dict[str, Any]], *, label: str, value_key: str, prefix: str) -> dict[str, Any]:
    summary = _summary(rows, label=label, value_key=value_key)
    out = {"label": label}
    for key, value in summary.items():
        if key == "label":
            continue
        out[f"{prefix}_{key}"] = value
    return out


def _write_scatter(rows: list[dict[str, Any]], out_path: Path, title: str, *, value_key: str, ylabel: str) -> str | None:
    if not rows:
        return None
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-cache-c2-local-divergence"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    usable = [row for row in rows if value_key in row and str(row.get(value_key, "")).strip() != ""]
    if not usable:
        return None
    x = np.asarray([float(row["delta_h"]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row[value_key]) for row in usable], dtype=np.float64)
    labels = [str(row.get("traj_id", "")) for row in usable]
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    colors = np.asarray([unique[label] for label in labels], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(x[finite], y[finite], c=colors[finite], cmap="tab10", s=24, alpha=0.85)
    if int(np.sum(finite)) >= 2 and float(np.std(x[finite])) > 1e-12 and float(np.std(y[finite])) > 1e-12:
        coef = np.polyfit(x[finite], y[finite], 1)
        xs = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1)
        title = f"{title}; r={np.corrcoef(x[finite], y[finite])[0, 1]:.3g}"
    ax.set_xlabel("Delta-H")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _analyze_item(
    item: dict[str, Any],
    *,
    divergence_modes: set[str],
    field_weights: dict[str, float],
    scales: list[int],
    horizon_steps: int,
    offset_steps: list[int],
    max_future_frames: int,
    spatial_downsample: int,
    max_step_error: float | None,
    clip_cache_dir: Path,
    foundation_model: str,
    force_clip_cache: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = Path(item["metrics_path"])
    apf_dir = Path(item["apf_dir"])
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not apf_dir.exists():
        raise FileNotFoundError(f"Missing APF dir: {apf_dir}")
    centers, delta_h, metric_meta = _load_delta_h(metrics_path)
    load_field_weights = dict(field_weights)
    if "clip_chamfer" in divergence_modes:
        load_field_weights.setdefault("A", 1.0)
        load_field_weights.setdefault("P", 1.0)
    steps, fields = _load_apf_series(apf_dir, field_weights=load_field_weights, spatial_downsample=spatial_downsample)
    if steps.size < 2:
        raise ValueError(f"Too few APF snapshots in {apf_dir}.")
    sample_step = float(np.nanmedian(np.diff(steps)))
    tolerance = float(max_step_error) if max_step_error is not None else max(1.0, 0.55 * sample_step)
    clip_embeddings = None
    clip_cache_path = None
    if "clip_chamfer" in divergence_modes:
        clip_embeddings, clip_cache_path = _load_or_compute_clip_embeddings(
            item=item,
            steps=steps,
            fields=fields,
            cache_dir=clip_cache_dir,
            foundation_model=foundation_model,
            spatial_downsample=spatial_downsample,
            force=force_clip_cache,
        )

    rows: list[dict[str, Any]] = []
    for idx, (center, dh) in enumerate(zip(centers, delta_h)):
        apf_vals: list[float] = []
        clip_vals: list[float] = []
        apf_offsets: list[int] = []
        clip_offsets: list[int] = []
        for offset in offset_steps:
            idx_pair = _future_indices(
                steps=steps,
                t0=float(center),
                offset_steps=int(offset),
                horizon_steps=int(horizon_steps),
                max_future_frames=int(max_future_frames),
                max_step_error=tolerance,
            )
            if idx_pair is None:
                continue
            idx0, idx1 = idx_pair
            if "apf" in divergence_modes:
                weighted: list[float] = []
                denom = 0.0
                for key, weight in field_weights.items():
                    if key not in fields:
                        continue
                    d = _field_l2(fields[key][idx0], fields[key][idx1], scales=scales)
                    if np.isfinite(d):
                        weighted.append(float(weight) * d)
                        denom += float(weight)
                if weighted and denom > 0.0:
                    apf_vals.append(float(sum(weighted) / max(denom, 1e-12)))
                    apf_offsets.append(int(offset))
            if "clip_chamfer" in divergence_modes and clip_embeddings is not None:
                d_clip = _future_clip_chamfer(embeddings=clip_embeddings, idx0=idx0, idx1=idx1)
                if np.isfinite(d_clip):
                    clip_vals.append(float(d_clip))
                    clip_offsets.append(int(offset))
        if not apf_vals and not clip_vals:
            continue
        row = {
            "traj_id": str(item["traj_id"]),
            "candidate_kind": str(item.get("candidate_kind", "")),
            "candidate_label": str(item.get("candidate_label", "")),
            "window_idx": int(idx),
            "step": int(round(float(center))),
            "delta_h": float(dh),
            "metrics_path": str(metrics_path),
            "apf_dir": str(apf_dir),
        }
        if apf_vals:
            row.update(
                {
                    "local_divergence": float(np.mean(apf_vals)),
                    "local_divergence_apf": float(np.mean(apf_vals)),
                    "local_divergence_apf_median": float(np.median(apf_vals)),
                    "n_offsets_used_apf": int(len(apf_vals)),
                    "offset_steps_used_apf": ",".join(str(offset) for offset in apf_offsets),
                }
            )
        if clip_vals:
            if "local_divergence" not in row:
                row["local_divergence"] = float(np.mean(clip_vals))
            row.update(
                {
                    "local_divergence_clip_chamfer": float(np.mean(clip_vals)),
                    "local_divergence_clip_chamfer_median": float(np.median(clip_vals)),
                    "n_offsets_used_clip_chamfer": int(len(clip_vals)),
                    "offset_steps_used_clip_chamfer": ",".join(str(offset) for offset in clip_offsets),
                    "clip_cache_path": str(clip_cache_path) if clip_cache_path is not None else "",
                    "foundation_model": str(foundation_model),
                }
            )
        if "local_divergence" in row:
            row.update(
                {
                    "local_divergence_median": row.get("local_divergence_apf_median", row.get("local_divergence_clip_chamfer_median", "")),
                    "n_offsets_used": row.get("n_offsets_used_apf", row.get("n_offsets_used_clip_chamfer", "")),
                    "offset_steps_used": row.get("offset_steps_used_apf", row.get("offset_steps_used_clip_chamfer", "")),
                }
            )
        for key, value in metric_meta.items():
            row[key] = value
        rows.append(row)
    summary: dict[str, Any] = {"label": str(item["traj_id"])}
    if "apf" in divergence_modes:
        summary.update(_prefixed_summary(rows, label=str(item["traj_id"]), value_key="local_divergence_apf", prefix="apf"))
    if "clip_chamfer" in divergence_modes:
        summary.update(
            _prefixed_summary(rows, label=str(item["traj_id"]), value_key="local_divergence_clip_chamfer", prefix="clip_chamfer")
        )
    summary.update(
        {
            "traj_id": str(item["traj_id"]),
            "candidate_kind": str(item.get("candidate_kind", "")),
            "metrics_path": str(metrics_path),
            "apf_dir": str(apf_dir),
            "n_apf_steps": int(steps.size),
            "apf_step_min": float(steps[0]),
            "apf_step_max": float(steps[-1]),
            "apf_sample_step": sample_step,
            "field_keys_loaded": ",".join(sorted(fields)),
            "clip_cache_path": str(clip_cache_path) if clip_cache_path is not None else "",
        }
    )
    return rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config, smoke=False)
    root = resolve_path(args.trajectory_root) if args.trajectory_root else _trajectory_root(cfg)
    if root is None:
        raise ValueError("Could not resolve trajectory root.")
    if not root.exists():
        raise FileNotFoundError(f"Trajectory root not found: {root}")
    output_dir = ensure_dir(
        resolve_path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "analysis" / "results" / "paper_suite" / "c2_local_divergence_probe"
    )
    field_weights = _parse_field_weights(args.field_weights)
    divergence_modes = _parse_modes(args.divergence_modes)
    scales = _parse_ints(args.scales, [1, 2, 4])
    offsets = _parse_ints(args.offset_steps, [5000, 10000, 20000])
    clip_cache_dir = ensure_dir(
        resolve_path(args.clip_cache_dir)
        if args.clip_cache_dir
        else output_dir / "clip_embedding_cache"
    )
    items = _iter_trajectories(root, include_random=bool(args.include_random))
    if args.max_trajectories is not None:
        items = items[: max(0, int(args.max_trajectories))]
    if not items:
        raise ValueError(f"No trajectories discovered under {root}")

    all_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        print(f"[c2-local-divergence] {idx}/{len(items)} {item['traj_id']}", flush=True)
        try:
            rows, summary = _analyze_item(
                item,
                divergence_modes=divergence_modes,
                field_weights=field_weights,
                scales=scales,
                horizon_steps=int(args.horizon_steps),
                offset_steps=offsets,
                max_future_frames=int(args.max_future_frames),
                spatial_downsample=int(args.spatial_downsample),
                max_step_error=float(args.max_step_error) if args.max_step_error is not None else None,
                clip_cache_dir=clip_cache_dir,
                foundation_model=str(args.foundation_model),
                force_clip_cache=bool(args.force_clip_cache),
            )
            all_rows.extend(rows)
            traj_rows.append(summary)
        except Exception as exc:
            failures.append({"traj_id": str(item.get("traj_id", "")), "error": f"{type(exc).__name__}: {exc}"})
            if args.strict:
                raise

    rows_path = output_dir / "local_divergence_rows.csv"
    per_traj_path = output_dir / "local_divergence_by_trajectory.csv"
    summary_path = output_dir / "local_divergence_summary.json"
    figure_apf_path = output_dir / "local_divergence_apf_vs_delta_h.png"
    figure_clip_path = output_dir / "local_divergence_clip_chamfer_vs_delta_h.png"
    write_csv(rows_path, all_rows)
    write_csv(per_traj_path, traj_rows)
    pooled: dict[str, Any] = {"label": "pooled"}
    figures: dict[str, str | None] = {}
    if "apf" in divergence_modes:
        pooled["apf"] = _summary(all_rows, label="pooled", value_key="local_divergence_apf")
        figures["apf"] = _write_scatter(
            all_rows,
            figure_apf_path,
            "C2 local APF divergence probe",
            value_key="local_divergence_apf",
            ylabel="single-trajectory local APF divergence",
        )
    if "clip_chamfer" in divergence_modes:
        pooled["clip_chamfer"] = _summary(all_rows, label="pooled", value_key="local_divergence_clip_chamfer")
        figures["clip_chamfer"] = _write_scatter(
            all_rows,
            figure_clip_path,
            "C2 local CLIP-chamfer divergence probe",
            value_key="local_divergence_clip_chamfer",
            ylabel="single-trajectory local CLIP chamfer divergence",
        )
    summary = {
        "status": "ok" if all_rows else "empty",
        "trajectory_root": str(root),
        "output_dir": str(output_dir),
        "n_trajectories_requested": len(items),
        "n_trajectories_scored": len(traj_rows),
        "n_rows": len(all_rows),
        "pooled": pooled,
        "divergence_modes": sorted(divergence_modes),
        "field_weights": field_weights,
        "scales": scales,
        "horizon_steps": int(args.horizon_steps),
        "offset_steps": offsets,
        "max_future_frames": int(args.max_future_frames),
        "spatial_downsample": int(args.spatial_downsample),
        "foundation_model": str(args.foundation_model),
        "clip_cache_dir": str(clip_cache_dir),
        "rows_csv": str(rows_path),
        "per_trajectory_csv": str(per_traj_path),
        "figures": figures,
        "figure": figures.get("apf") or figures.get("clip_chamfer"),
        "failures": failures,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="c2_local_divergence_probe_"))
    try:
        root = tmp / "root"
        traj = root / "flow_opt_test"
        apf = traj / "apf_logs"
        apf.mkdir(parents=True)
        steps = np.arange(0, 60000, 1000, dtype=np.int32)
        t = np.linspace(0, 1, steps.size, dtype=np.float32)
        yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
        base = ((yy + xx) / 30.0)[None, :, :, None]
        wave = np.sin(12.0 * t)[:, None, None, None].astype(np.float32)
        a = np.clip(base + 0.1 * wave, 0.0, 1.0).astype(np.float32)
        p = np.repeat(a, 3, axis=-1)
        f = np.concatenate([a, 1.0 - a], axis=-1)
        np.savez_compressed(
            apf / "P_steps_000000_059000__secs_0.000_59.000__idx_0000.npz",
            A=a,
            P=p,
            F=f,
            state_t=steps,
        )
        centers = np.arange(5000, 40000, 5000, dtype=np.int32)
        delta_h = np.linspace(0.0, 1.0, centers.size, dtype=np.float32)
        np.savez_compressed(traj / "metrics.npz", delta_h_best=delta_h, delta_h_window_center_steps=centers)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "trajectories": [
                        {
                            "traj_id": "flow_opt_test",
                            "candidate_kind": "optimized",
                            "traj_dir": "flow_opt_test",
                            "apf_dir": "flow_opt_test/apf_logs",
                            "metrics_path": "flow_opt_test/metrics.npz",
                        }
                    ]
                }
            )
        )
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text("meta:\n  output_root: analysis/results/paper_suite\nc2:\n  trajectory_root: unused\n")
        ns = argparse.Namespace(
            config=str(cfg_path),
            trajectory_root=str(root),
            output_dir=str(tmp / "out"),
            include_random=False,
            max_trajectories=None,
            divergence_modes="apf",
            field_weights="A:1,P:0.25,F:0.25",
            scales="1,2",
            horizon_steps=10000,
            offset_steps="1000,2000",
            max_future_frames=6,
            spatial_downsample=2,
            max_step_error=None,
            clip_cache_dir=None,
            foundation_model="clip",
            force_clip_cache=False,
            strict=True,
        )
        summary = run(ns)
        if summary["n_rows"] <= 0:
            raise AssertionError("self-test produced no rows")

        traj_batched = root / "flow_opt_batched_axis"
        apf_batched = traj_batched / "apf_logs"
        apf_batched.mkdir(parents=True)
        np.savez_compressed(
            apf_batched / "P_steps_000000_059000__secs_0.000_59.000__idx_0000.npz",
            A=a[None, ...],
            P=p[None, ...],
            F=f[None, ...],
            steps=steps,
        )
        steps2 = steps + 60000
        np.savez_compressed(
            apf_batched / "P_steps_060000_119000__secs_60.000_119.000__idx_0001.npz",
            A=a,
            P=p,
            F=f,
            steps=steps2,
        )
        steps_loaded, fields_loaded = _load_apf_series(
            apf_batched,
            field_weights={"A": 1.0, "P": 0.25, "F": 0.25},
            spatial_downsample=2,
        )
        expected_steps = 2 * steps.size
        if steps_loaded.size != expected_steps:
            raise AssertionError(f"mixed-axis APF load failed: got {steps_loaded.size}, expected {expected_steps}")
        if fields_loaded["A"].shape[0] != expected_steps:
            raise AssertionError("batched-axis APF field has wrong time length")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate C2 future divergence from a single saved APF trajectory, "
            "then correlate it with Delta-H from metrics.npz. This never runs branching simulation."
        )
    )
    parser.add_argument("config", nargs="?", default="experiments/paper_suite/config.yaml")
    parser.add_argument("--trajectory-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--divergence-modes",
        default="apf,clip_chamfer",
        help="Comma-separated divergence modes: apf,clip_chamfer. Default computes both.",
    )
    parser.add_argument("--field-weights", default="A:1.0,P:0.25,F:0.25")
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--horizon-steps", type=int, default=30000)
    parser.add_argument("--offset-steps", default="5000,10000,20000")
    parser.add_argument("--max-future-frames", type=int, default=24)
    parser.add_argument("--spatial-downsample", type=int, default=4)
    parser.add_argument("--max-step-error", type=float, default=None)
    parser.add_argument("--foundation-model", default="clip")
    parser.add_argument("--clip-cache-dir", default=None)
    parser.add_argument("--force-clip-cache", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        print("self-test ok")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
