from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np

from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config
from paper_suite_common import (
    ensure_dir,
    init_suite_logging,
    load_config,
    log_event,
    resolve_path,
    sign_test_greater,
    to_plain,
    write_csv,
    write_json,
)


FAMILIES = ("S0", "S1", "S3", "S4", "S5", "S6", "S7")


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    value = cfg.get(key, default) if cfg is not None else default
    return int(default if value is None else value)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    value = cfg.get(key, default) if cfg is not None else default
    return float(default if value is None else value)


def _cfg_bool(cfg: Any, key: str, default: bool) -> bool:
    value = cfg.get(key, default) if cfg is not None else default
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_optional_int(cfg: Any, key: str) -> int | None:
    if cfg is None:
        return None
    value = cfg.get(key, None)
    return None if value is None else int(value)


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    value = float(seconds)
    if not np.isfinite(value):
        return "unknown"
    value = max(0.0, value)
    if value < 1.0:
        return f"{value * 1000.0:.0f}ms"
    total = int(round(value))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_text(done: int, total: int, start_time: float, *, item_seconds: float | None = None) -> str:
    elapsed = time.perf_counter() - start_time
    total = max(0, int(total))
    done = max(0, int(done))
    avg = elapsed / done if done > 0 else None
    remaining = max(0, total - done)
    eta = None if avg is None else avg * remaining
    parts = [
        f"{done}/{total}",
        f"elapsed={_format_duration(elapsed)}",
    ]
    if item_seconds is not None:
        parts.append(f"item={_format_duration(item_seconds)}")
    if avg is not None:
        parts.append(f"avg={_format_duration(avg)}")
    parts.append(f"eta={_format_duration(eta)}")
    return " ".join(parts)


def _tau_grid(cfg: Any) -> list[int]:
    vals = cfg.get("tau_grid_steps", [1, 2, 4, 8, 16, 32, 64]) if cfg is not None else [1, 2, 4, 8, 16, 32, 64]
    return [int(x) for x in vals]


def _periodic_delta(dx: np.ndarray, domain: float = 1.0) -> np.ndarray:
    return (dx + 0.5 * domain) % domain - 0.5 * domain


def _unwrap_periodic_xy(xy: np.ndarray, *, domain: float = 1.0) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected xy with shape (T,N,2), got {arr.shape}.")
    if arr.shape[0] <= 1:
        return arr.copy()
    if not np.isfinite(domain) or float(domain) <= 0.0:
        raise ValueError(f"domain must be positive for torus unwrapping, got {domain!r}.")
    step_delta = _periodic_delta(arr[1:] - arr[:-1], domain=float(domain))
    unwrapped = np.empty_like(arr, dtype=np.float32)
    unwrapped[0] = arr[0]
    unwrapped[1:] = arr[0][None, :, :] + np.cumsum(step_delta, axis=0, dtype=np.float32)
    return unwrapped


def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack((np.sin(theta), np.cos(theta)), axis=-1).astype(np.float32)


def _simulate_s0(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    x0 = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.repeat(x0[None, :, :], T, axis=0)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "static_null"}}


def _simulate_s1(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    sigma = 0.006
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        x = np.mod(x + rng.normal(0.0, sigma, size=(N, 2)).astype(np.float32), L)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "homogeneous_motion_null"}}


def _simulate_s3(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    radius = 0.08
    direction = _random_unit_vectors(rng, 1)[0]
    speed = 0.002
    center0 = rng.uniform(0.2, 0.8, size=(2,)).astype(np.float32)
    offsets = _random_unit_vectors(rng, N) * rng.uniform(0.0, radius, size=(N, 1)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        center = np.mod(center0 + direction * speed * t, L)
        jitter = rng.normal(0.0, 0.0015, size=(N, 2)).astype(np.float32)
        xy[t] = np.mod(center[None, :] + offsets + jitter, L)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "coherent_motion_low_complexity"}}


def _simulate_s4(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    labels = np.arange(N, dtype=np.int32) % 2
    rng.shuffle(labels)
    v = np.zeros((N, 2), dtype=np.float32)
    v[labels == 0] = np.asarray([0.0008, 0.0], dtype=np.float32)
    v[labels == 1] = np.asarray([0.0, 0.0050], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        x = np.mod(x + v + rng.normal(0.0, 0.0008, size=(N, 2)).astype(np.float32), L)
    return {"xy": xy, "labels": labels, "metadata": {"expected": "two_role_positive_control"}}


def _simulate_s5(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    t0 = int(0.5 * T)
    v0 = np.asarray([0.0015, 0.0], dtype=np.float32)
    v1 = np.asarray([0.0, 0.0015], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        v = v0 if t < t0 else v1
        x = np.mod(x + v + rng.normal(0.0, 0.0009, size=(N, 2)).astype(np.float32), L)
    return {
        "xy": xy,
        "labels": np.zeros(N, dtype=np.int32),
        "metadata": {"expected": "global_switch_not_generic_changepoint", "event_interval": [t0, t0]},
    }


def _simulate_s6(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    t0 = int(0.40 * T)
    t1 = int(0.65 * T)
    switch_times = rng.integers(t0, max(t0 + 1, t1), size=N)
    v_old = np.asarray([0.0010, 0.0], dtype=np.float32)
    v_new = np.asarray([0.0, 0.0050], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    labels_t = np.empty((T, N), dtype=np.int8)
    for t in range(T):
        switched = t >= switch_times
        labels_t[t] = switched.astype(np.int8)
        xy[t] = x
        v = np.where(switched[:, None], v_new[None, :], v_old[None, :]).astype(np.float32)
        x = np.mod(x + v + rng.normal(0.0, 0.0008, size=(N, 2)).astype(np.float32), L)
    labels_mid = labels_t[(t0 + t1) // 2].astype(np.int32)
    return {
        "xy": xy,
        "labels": labels_mid,
        "labels_t": labels_t,
        "metadata": {"expected": "partial_transition_positive_control", "event_interval": [t0, t1]},
    }


def _simulate_s7(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    specs = [
        {"radius": 0.11, "speed": 0.0016, "direction": np.asarray([1.0, 0.2], dtype=np.float32)},
        {"radius": 0.04, "speed": 0.0055, "direction": np.asarray([-0.2, 1.0], dtype=np.float32)},
        {"radius": 0.025, "speed": 0.0100, "direction": np.asarray([0.8, -0.6], dtype=np.float32)},
    ]
    n_groups = len(specs)
    labels = np.arange(N, dtype=np.int32) % n_groups
    rng.shuffle(labels)
    centers0 = rng.uniform(0.15, 0.85, size=(n_groups, 2)).astype(np.float32)
    offsets = np.zeros((N, 2), dtype=np.float32)
    for group, spec in enumerate(specs):
        idx = np.flatnonzero(labels == group)
        if idx.size:
            offsets[idx] = _random_unit_vectors(rng, idx.size) * rng.uniform(0.0, spec["radius"], size=(idx.size, 1)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        frame = np.empty((N, 2), dtype=np.float32)
        for group, spec in enumerate(specs):
            direction = spec["direction"] / np.linalg.norm(spec["direction"])
            center = np.mod(centers0[group] + direction * float(spec["speed"]) * t, L)
            idx = np.flatnonzero(labels == group)
            if idx.size:
                frame[idx] = center[None, :] + offsets[idx] + rng.normal(0.0, 0.001, size=(idx.size, 2)).astype(np.float32)
        xy[t] = np.mod(frame, L)
    crosses = [float(spec["radius"]) / max(float(spec["speed"]), 1e-12) for spec in specs]
    return {
        "xy": xy,
        "labels": labels,
        "metadata": {
            "expected": "multiscale_tau_calibration",
            "scale_range": [float(min(crosses)), float(max(crosses))],
            "crossing_times": crosses,
        },
    }


SIMULATORS = {
    "S0": _simulate_s0,
    "S1": _simulate_s1,
    "S3": _simulate_s3,
    "S4": _simulate_s4,
    "S5": _simulate_s5,
    "S6": _simulate_s6,
    "S7": _simulate_s7,
}


def _synthetic_dirs(cfg: Any) -> dict[str, Path]:
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    root = ensure_dir(output_root / "synthetic_calibration")
    return {
        "root": root,
        "simulation": ensure_dir(root / "simulation"),
        "metrics": ensure_dir(root / "metrics"),
        "videos": ensure_dir(root / "videos"),
        "figures": ensure_dir(root / "figures"),
        "heatmaps": ensure_dir(root / "figures" / "delta_h_heatmaps"),
    }


def _cached_simulation_matches(path: Path, *, family: str, seed: int, T: int, N: int, L: float) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            xy_shape = tuple(np.asarray(data["xy"]).shape)
            metadata = __import__("json").loads(str(np.asarray(data["metadata_json"]).item()))
    except Exception:
        return False
    return (
        xy_shape == (int(T), int(N), 2)
        and str(metadata.get("family")) == str(family)
        and int(metadata.get("seed", -1)) == int(seed)
        and int(metadata.get("time_steps", -1)) == int(T)
        and int(metadata.get("n_particles", -1)) == int(N)
        and abs(float(metadata.get("domain_size", float("nan"))) - float(L)) < 1e-12
    )


def _video_path(dirs: dict[str, Path], family: str, seed: int) -> Path:
    return dirs["videos"] / f"{family}_seed_{seed:03d}.mp4"


def _video_is_valid(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(path))
            try:
                return bool(cap.isOpened()) and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            finally:
                cap.release()
        except Exception:
            return True
    except OSError:
        return False


def _load_simulation_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, Any] = {
            "xy": np.asarray(data["xy"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.int32),
            "metadata": json.loads(str(np.asarray(data["metadata_json"]).item())),
        }
        if "labels_t" in data.files:
            payload["labels_t"] = np.asarray(data["labels_t"], dtype=np.int8)
    return payload


def _label_palette_rgb() -> np.ndarray:
    return np.asarray(
        [
            [36, 95, 160],
            [215, 72, 64],
            [45, 140, 94],
            [142, 86, 164],
            [230, 159, 0],
            [73, 152, 148],
            [92, 84, 112],
            [188, 96, 58],
        ],
        dtype=np.uint8,
    )


def _blend_rgb(color: np.ndarray, background: np.ndarray, alpha: float) -> tuple[int, int, int]:
    c = alpha * color.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)
    return tuple(int(x) for x in c[::-1])


def _render_synthetic_video(
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    labels_t: np.ndarray | None,
    metadata: dict[str, Any],
    output_path: Path,
    render_cfg: Any,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional wheel.
        raise RuntimeError(
            "Synthetic video rendering requires opencv-python. Install requirements_paper_suite.txt "
            "or set synthetic.render.enabled=false."
        ) from exc

    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"Synthetic video expects xy with shape (T,N,2), got {xy.shape}.")
    T, N, _ = xy.shape
    if T < 1 or N < 1:
        raise ValueError(f"Synthetic video expects non-empty trajectory, got T={T}, N={N}.")

    size = int(max(128, _cfg_int(render_cfg, "size_px", 512)))
    fps = int(max(1, _cfg_int(render_cfg, "fps", 18)))
    trail_steps = int(max(0, _cfg_int(render_cfg, "trail_steps", 16)))
    radius = int(max(1, _cfg_int(render_cfg, "particle_radius_px", 3)))
    max_frames = _cfg_optional_int(render_cfg, "max_frames")
    particle_stride = int(max(1, _cfg_int(render_cfg, "particle_stride", 1)))
    max_particles = _cfg_optional_int(render_cfg, "max_particles")
    codec = str(render_cfg.get("codec", "mp4v") if render_cfg is not None else "mp4v")
    if len(codec) != 4:
        raise ValueError(f"synthetic.render.codec must be a four-character OpenCV fourcc code, got {codec!r}.")

    frame_count = T if max_frames is None else min(T, max(1, int(max_frames)))
    frame_idx = np.unique(np.linspace(0, T - 1, num=frame_count, dtype=np.int32))
    keep_ids = np.arange(N, dtype=np.int32)[::particle_stride]
    if max_particles is not None:
        keep_ids = keep_ids[: max(0, int(max_particles))]
    if keep_ids.size == 0:
        raise ValueError("No particles selected for synthetic video. Check synthetic.render particle settings.")

    domain = float(metadata.get("domain_size", 1.0))
    if not np.isfinite(domain) or domain <= 0:
        domain = 1.0
    family = str(metadata.get("family", "synthetic"))
    seed = int(metadata.get("seed", -1))
    palette = _label_palette_rgb()
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels.size != N:
        labels = np.zeros(N, dtype=np.int32)
    dynamic_labels = labels_t is not None and np.asarray(labels_t).shape[:2] == (T, N)
    fallback_labels = np.arange(N, dtype=np.int32) if np.unique(labels).size <= 1 and not dynamic_labels else labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (size, size),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV video writer for {tmp_path} with codec={codec!r}.")

    bg_rgb = np.asarray([248, 248, 246], dtype=np.uint8)
    frame_bgr_bg = tuple(int(x) for x in bg_rgb[::-1])
    margin = max(18, int(round(0.055 * size)))
    span = max(1, size - 2 * margin - 1)

    def to_px_continuous(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float32) / domain
        x = margin + p[:, 0] * span
        y = margin + (1.0 - p[:, 1]) * span
        return np.stack([x, y], axis=-1).astype(np.int32)

    periodic_offsets = np.asarray(
        [[dy, dx] for dy in (-domain, 0.0, domain) for dx in (-domain, 0.0, domain)],
        dtype=np.float32,
    )

    try:
        for t in frame_idx:
            frame = np.full((size, size, 3), frame_bgr_bg, dtype=np.uint8)
            cv2.rectangle(frame, (margin, margin), (size - margin, size - margin), (210, 210, 205), 1, lineType=cv2.LINE_AA)
            draw_labels = np.asarray(labels_t[t], dtype=np.int32) if dynamic_labels else fallback_labels
            positions_wrapped = np.mod(xy[t], domain)

            if trail_steps > 0:
                start_t = max(0, int(t) - trail_steps)
                for pid in keep_ids:
                    color_rgb = palette[int(draw_labels[pid]) % len(palette)]
                    hist = xy[start_t : int(t) + 1, pid, :]
                    if hist.shape[0] < 2:
                        continue
                    hist_unwrapped = _unwrap_periodic_xy(hist[:, None, :], domain=domain)[:, 0, :]
                    hist_end_wrapped = np.mod(hist_unwrapped[-1], domain)
                    hist_aligned = hist_unwrapped + (positions_wrapped[pid] - hist_end_wrapped)[None, :]
                    denom = max(1, hist_aligned.shape[0] - 2)
                    for offset in periodic_offsets:
                        hist_px = to_px_continuous(hist_aligned + offset[None, :])
                        for seg_idx in range(hist_px.shape[0] - 1):
                            alpha = 0.14 + 0.28 * (seg_idx / denom)
                            trail_color = _blend_rgb(color_rgb, bg_rgb, alpha)
                            cv2.line(
                                frame,
                                tuple(int(x) for x in hist_px[seg_idx]),
                                tuple(int(x) for x in hist_px[seg_idx + 1]),
                                trail_color,
                                1,
                                lineType=cv2.LINE_AA,
                            )

            for pid in keep_ids:
                color_rgb = palette[int(draw_labels[pid]) % len(palette)]
                color_bgr = tuple(int(x) for x in color_rgb[::-1])
                for offset in periodic_offsets:
                    center_px = to_px_continuous(positions_wrapped[pid : pid + 1] + offset[None, :])[0]
                    center = tuple(int(x) for x in center_px)
                    cv2.circle(frame, center, radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, center, radius, color_bgr, -1, lineType=cv2.LINE_AA)

            cv2.putText(
                frame,
                f"{family} seed={seed:03d} t={int(t):04d}",
                (margin, max(14, margin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, size / 1200.0),
                (45, 45, 45),
                1,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    if not _video_is_valid(tmp_path):
        raise RuntimeError(f"Synthetic video writer produced an invalid or empty file: {tmp_path}")
    os.replace(tmp_path, output_path)


def simulate(
    config_path: str | Path,
    *,
    smoke: bool = False,
    force: bool = False,
    render_videos: bool | None = None,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    syn = cfg.get("synthetic", {})
    dirs = _synthetic_dirs(cfg)
    render_cfg = syn.get("render", {}) if syn is not None else {}
    video_enabled = _cfg_bool(render_cfg, "enabled", True) if render_videos is None else bool(render_videos)
    T = _cfg_int(syn, "time_steps", 2000)
    N = _cfg_int(syn, "n_particles", 256)
    L = _cfg_float(syn, "domain_size", 1.0)
    seeds = _cfg_int(syn, "seeds", 3)
    families = [str(x) for x in (syn.get("families", list(FAMILIES)) or list(FAMILIES))]
    manifest_rows = []
    total_runs = len(families) * seeds
    layer_start = time.perf_counter()
    run_idx = 0
    log_event(
        f"synthetic simulation start smoke={smoke} force={force} videos={video_enabled} "
        f"families={families} seeds={seeds} T={T} N={N} n_runs={total_runs}",
        component="synthetic",
    )
    for family in families:
        if family not in SIMULATORS:
            raise ValueError(f"Unknown synthetic family {family!r}. Expected one of {sorted(SIMULATORS)}.")
        for seed in range(seeds):
            item_start = time.perf_counter()
            item_idx = run_idx + 1
            out_path = dirs["simulation"] / f"{family}_seed_{seed:03d}.npz"
            video_path = _video_path(dirs, family, seed)
            had_existing = out_path.exists()
            had_video = _video_is_valid(video_path)
            payload_for_video: dict[str, Any] | None = None
            if out_path.exists() and not force and _cached_simulation_matches(out_path, family=family, seed=seed, T=T, N=N, L=L):
                status = "exists"
            else:
                rng = np.random.default_rng(_cfg_int(syn, "seed_base", 100) + seed + 1009 * families.index(family))
                payload = SIMULATORS[family](rng, T=T, N=N, L=L)
                metadata = dict(payload.get("metadata", {}))
                metadata.update({"family": family, "seed": int(seed), "time_steps": int(T), "n_particles": int(N), "domain_size": float(L)})
                save_payload = {
                    "xy": np.asarray(payload["xy"], dtype=np.float32),
                    "labels": np.asarray(payload.get("labels", np.zeros(N, dtype=np.int32)), dtype=np.int32),
                    "metadata_json": np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
                }
                if "labels_t" in payload:
                    save_payload["labels_t"] = np.asarray(payload["labels_t"], dtype=np.int8)
                np.savez_compressed(out_path, **save_payload)
                payload_for_video = {
                    "xy": save_payload["xy"],
                    "labels": save_payload["labels"],
                    "labels_t": save_payload.get("labels_t"),
                    "metadata": metadata,
                }
                status = "rewritten_stale" if had_existing and not force else "written"
            video_status = "disabled"
            if video_enabled:
                video_force = force or status != "exists"
                if had_video and not video_force:
                    video_status = "exists"
                else:
                    if payload_for_video is None:
                        payload_for_video = _load_simulation_payload(out_path)
                    video_start = time.perf_counter()
                    log_event(
                        f"synthetic video start {item_idx}/{total_runs} {family} seed={seed} -> {video_path}",
                        component="synthetic",
                    )
                    _render_synthetic_video(
                        xy=np.asarray(payload_for_video["xy"], dtype=np.float32),
                        labels=np.asarray(payload_for_video["labels"], dtype=np.int32),
                        labels_t=None
                        if payload_for_video.get("labels_t") is None
                        else np.asarray(payload_for_video["labels_t"], dtype=np.int8),
                        metadata=dict(payload_for_video["metadata"]),
                        output_path=video_path,
                        render_cfg=render_cfg,
                    )
                    log_event(
                        f"synthetic video done {item_idx}/{total_runs} {family} seed={seed} "
                        f"video_time={_format_duration(time.perf_counter() - video_start)}",
                        component="synthetic",
                    )
                    video_status = "rewritten" if had_video else "written"
            manifest_rows.append({"family": family, "seed": seed, "path": str(out_path), "status": status})
            manifest_rows[-1].update({"video_path": str(video_path) if video_enabled else "", "video_status": video_status})
            run_idx += 1
            item_seconds = time.perf_counter() - item_start
            manifest_rows[-1].update({"elapsed_seconds": f"{item_seconds:.6f}"})
            log_event(
                f"synthetic simulation progress {_progress_text(run_idx, total_runs, layer_start, item_seconds=item_seconds)} "
                f"{family} seed={seed} status={status} video_status={video_status}",
                component="synthetic",
            )
    write_csv(dirs["root"] / "simulation_manifest.csv", manifest_rows)
    elapsed = time.perf_counter() - layer_start
    write_json(
        dirs["root"] / "simulation_summary.json",
        {
            "n_runs": len(manifest_rows),
            "time_steps": T,
            "n_particles": N,
            "videos_enabled": bool(video_enabled),
            "elapsed_seconds": elapsed,
            "elapsed": _format_duration(elapsed),
        },
    )
    log_event(
        f"synthetic simulation done n_runs={len(manifest_rows)} elapsed={_format_duration(elapsed)} "
        f"manifest={dirs['root'] / 'simulation_manifest.csv'}",
        component="synthetic",
    )
    return {"simulation_manifest": str(dirs["root"] / "simulation_manifest.csv"), "n_runs": len(manifest_rows)}


def _build_metric_cfg(syn_cfg: Any, T: int) -> dict[str, Any]:
    args = SimpleNamespace(
        rollout_steps=int(T),
        sample_every_steps=1,
        time_sampling=None,
        metric_window_size_steps=_cfg_int(syn_cfg, "metric_window_size_steps", 200),
        metric_window_step_steps=_cfg_int(syn_cfg, "metric_window_step_steps", 50),
        metric_tau_mode="max_grid",
        metric_tau_steps=_tau_grid(syn_cfg)[0],
        metric_tau_grid_steps=_tau_grid(syn_cfg),
        metric_window_size_frames=None,
        metric_window_step_frames=None,
        metric_tau_frames=None,
        metric_tau_grid_frames=None,
        metric_range_start_steps=_cfg_int(syn_cfg, "metric_range_start_steps", 0),
        metric_range_end_steps=None,
        metric_m_samples=_cfg_int(syn_cfg, "metric_m_samples", 32),
        metric_m_min=_cfg_int(syn_cfg, "metric_m_min", 4),
        metric_n_proj=_cfg_int(syn_cfg, "metric_n_proj", 12),
        metric_null_reps=_cfg_int(syn_cfg, "metric_null_reps", 4),
        metric_particle_samples=_cfg_int(syn_cfg, "metric_particle_samples", 64),
        metric_dirs_seed=_cfg_int(syn_cfg, "metric_dirs_seed", 123),
        metric_periodic=True,
        metric_domain_y=_cfg_float(syn_cfg, "domain_size", 1.0),
        metric_domain_x=_cfg_float(syn_cfg, "domain_size", 1.0),
        metric_preprocess_mode="clip",
        metric_delta_h_floor=_cfg_float(syn_cfg, "metric_delta_h_floor", 0.0),
        metric_scales=None,
        metric_scale_weights=None,
        metric_alpha=0.0,
        metric_beta=1.0,
        metric_eps=1e-12,
    )
    cfg = resolve_metric_config(args)
    # Synthetic trajectories are stored wrapped on the torus. Delta-H must see
    # continuous trajectories, otherwise large tau values alias velocities once
    # displacement crosses half the domain.
    cfg["positions_unwrapped"] = True
    return cfg


def _kmeans(features: np.ndarray, k: int, *, seed: int, n_iters: int = 40) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"kmeans expects 2D features, got {x.shape}.")
    n = x.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=min(k, n), replace=False)].copy()
    if centers.shape[0] < k:
        centers = np.concatenate([centers, np.repeat(centers[-1:], k - centers.shape[0], axis=0)], axis=0)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iters):
        d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(d, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = np.mean(x[labels == j], axis=0)
    return labels


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.int64).reshape(-1)
    y = np.asarray(b, dtype=np.int64).reshape(-1)
    if x.size != y.size:
        raise ValueError("ARI label vectors must have the same length.")
    n = int(x.size)
    if n < 2:
        return float("nan")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    contingency = np.zeros((xi.max() + 1, yi.max() + 1), dtype=np.int64)
    for i, j in zip(xi, yi):
        contingency[i, j] += 1

    def comb2(v: np.ndarray) -> float:
        vv = np.asarray(v, dtype=np.float64)
        return float(np.sum(vv * (vv - 1.0) / 2.0))

    sum_comb = comb2(contingency)
    row_comb = comb2(np.sum(contingency, axis=1))
    col_comb = comb2(np.sum(contingency, axis=0))
    total = n * (n - 1.0) / 2.0
    expected = row_comb * col_comb / total if total > 0 else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb - expected) < 1e-12 else 0.0
    return float((sum_comb - expected) / denom)


def _role_recovery(
    xy: np.ndarray,
    labels: np.ndarray,
    tau: int,
    *,
    seed: int,
    domain: float,
    positions_unwrapped: bool = False,
) -> dict[str, Any] | None:
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    tau = int(max(1, min(tau, xy.shape[0] - 1)))
    dx = xy[tau:] - xy[:-tau] if positions_unwrapped else _periodic_delta(xy[tau:] - xy[:-tau], domain=domain)
    speed = np.linalg.norm(dx, axis=-1)
    feats = np.concatenate(
        [
            np.mean(dx, axis=0),
            np.std(dx, axis=0),
            np.mean(speed, axis=0, keepdims=True).T,
            np.std(speed, axis=0, keepdims=True).T,
        ],
        axis=1,
    )
    pred = _kmeans(feats, int(unique.size), seed=seed)
    return {"ari": _adjusted_rand_index(labels, pred), "n_roles": int(unique.size)}


def _event_error(window_centers: np.ndarray, values: np.ndarray, interval: list[int]) -> dict[str, Any]:
    idx = int(np.nanargmax(values))
    peak = float(window_centers[idx])
    lo, hi = float(interval[0]), float(interval[1])
    if lo <= peak <= hi:
        error = 0.0
    else:
        error = min(abs(peak - lo), abs(peak - hi))
    return {"peak_step": peak, "event_start": lo, "event_end": hi, "event_error_steps": float(error)}


def _array_sha256(arr: np.ndarray) -> str:
    x = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(x.view(np.uint8))
    return h.hexdigest()


def _metric_cache_payload(
    *,
    metric_cfg: dict[str, Any],
    metric_seed: int,
    trajectory_path: Path,
    trajectory_metadata: dict[str, Any],
    xy: np.ndarray,
    metric_xy: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "_paper_suite_cache_version": np.asarray(3, dtype=np.int32),
        "_metric_config_json": np.asarray(json.dumps(to_plain(metric_cfg), sort_keys=True)),
        "_metric_seed": np.asarray(int(metric_seed), dtype=np.int64),
        "_trajectory_path": np.asarray(str(trajectory_path)),
        "_trajectory_metadata_json": np.asarray(json.dumps(to_plain(trajectory_metadata), sort_keys=True)),
        "_trajectory_shape": np.asarray(xy.shape, dtype=np.int32),
        "_trajectory_xy_sha256": np.asarray(_array_sha256(xy)),
        "_metric_input_xy_sha256": np.asarray(_array_sha256(metric_xy)),
        "_metric_input_positions_unwrapped": np.asarray(bool(metric_cfg.get("positions_unwrapped", False))),
        "_trajectory_labels_sha256": np.asarray(_array_sha256(labels)),
    }


def _metric_cache_matches(
    cached: np.lib.npyio.NpzFile,
    *,
    metric_cfg: dict[str, Any],
    metric_seed: int,
    trajectory_path: Path,
    trajectory_metadata: dict[str, Any],
    xy: np.ndarray,
    metric_xy: np.ndarray,
    labels: np.ndarray,
) -> bool:
    required = {
        "_paper_suite_cache_version",
        "_metric_config_json",
        "_metric_seed",
        "_trajectory_path",
        "_trajectory_metadata_json",
        "_trajectory_shape",
        "_trajectory_xy_sha256",
        "_metric_input_xy_sha256",
        "_metric_input_positions_unwrapped",
        "_trajectory_labels_sha256",
    }
    if not required.issubset(set(cached.files)):
        return False
    try:
        return (
            int(np.asarray(cached["_paper_suite_cache_version"]).item()) == 3
            and str(np.asarray(cached["_metric_config_json"]).item()) == json.dumps(to_plain(metric_cfg), sort_keys=True)
            and int(np.asarray(cached["_metric_seed"]).item()) == int(metric_seed)
            and str(np.asarray(cached["_trajectory_path"]).item()) == str(trajectory_path)
            and str(np.asarray(cached["_trajectory_metadata_json"]).item()) == json.dumps(to_plain(trajectory_metadata), sort_keys=True)
            and tuple(np.asarray(cached["_trajectory_shape"], dtype=np.int32).tolist()) == tuple(int(x) for x in xy.shape)
            and str(np.asarray(cached["_trajectory_xy_sha256"]).item()) == _array_sha256(xy)
            and str(np.asarray(cached["_metric_input_xy_sha256"]).item()) == _array_sha256(metric_xy)
            and bool(np.asarray(cached["_metric_input_positions_unwrapped"]).item()) == bool(metric_cfg.get("positions_unwrapped", False))
            and str(np.asarray(cached["_trajectory_labels_sha256"]).item()) == _array_sha256(labels)
        )
    except Exception:
        return False


def metrics(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    syn = cfg.get("synthetic", {})
    dirs = _synthetic_dirs(cfg)
    families = [str(x) for x in (syn.get("families", list(FAMILIES)) or list(FAMILIES))]
    seeds = _cfg_int(syn, "seeds", 3)
    sim_files = [dirs["simulation"] / f"{family}_seed_{seed:03d}.npz" for family in families for seed in range(seeds)]
    layer_start = time.perf_counter()
    log_event(
        f"synthetic metrics start smoke={smoke} force={force} families={families} seeds={seeds} n_runs={len(sim_files)}",
        component="synthetic",
    )
    missing = [str(path) for path in sim_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} configured synthetic simulation files in {dirs['simulation']}. "
            "Run layer=simulation/task=synthetic first. First missing file: "
            f"{missing[0]}"
        )
    with np.load(sim_files[0], allow_pickle=False) as first:
        T = int(np.asarray(first["xy"]).shape[0])
    metric_cfg = _build_metric_cfg(syn, T)
    metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))
    domain = _cfg_float(syn, "domain_size", 1.0)
    metric_seed_base = _cfg_int(syn, "metric_seed", 12345)

    score_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(sim_files, start=1):
        item_start = time.perf_counter()
        with np.load(path, allow_pickle=False) as data:
            xy = np.asarray(data["xy"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int32)
            metadata = __import__("json").loads(str(np.asarray(data["metadata_json"]).item()))
        xy_metric = _unwrap_periodic_xy(xy, domain=domain)
        family = str(metadata["family"])
        seed = int(metadata["seed"])
        metric_seed = int(metric_seed_base + seed)
        metrics_path = dirs["metrics"] / f"{family}_seed_{seed:03d}_metrics.npz"
        if metrics_path.exists() and not force:
            with np.load(metrics_path, allow_pickle=False) as cached:
                if _metric_cache_matches(
                    cached,
                    metric_cfg=metric_cfg,
                    metric_seed=metric_seed,
                    trajectory_path=path,
                    trajectory_metadata=metadata,
                    xy=xy,
                    metric_xy=xy_metric,
                    labels=labels,
                ):
                    info_np = {key: np.asarray(cached[key]) for key in cached.files}
                else:
                    info_np = {}
        else:
            info_np = {}
        if not info_np:
            item_status = "computed"
            log_event(
                f"synthetic metrics compute start {idx}/{len(sim_files)} {family} seed={seed} from {path.name}",
                component="synthetic",
            )
            rng = jax.random.PRNGKey(metric_seed)
            _loss, info = metric_eval(rng, jnp.asarray(xy_metric))
            info_np = {key: np.asarray(jax.device_get(value)) for key, value in info.items()}
            info_np.update(
                _metric_cache_payload(
                    metric_cfg=metric_cfg,
                    metric_seed=metric_seed,
                    trajectory_path=path,
                    trajectory_metadata=metadata,
                    xy=xy,
                    metric_xy=xy_metric,
                    labels=labels,
                )
            )
            np.savez_compressed(metrics_path, **info_np)
        else:
            item_status = "exists"

        tau_steps = np.asarray(info_np["tau_steps"], dtype=np.int32)
        score_by_tau = np.asarray(info_np["score_by_tau"], dtype=np.float64)
        amp_by_tau = np.asarray(info_np["amp_by_tau"], dtype=np.float64)
        msc_by_tau = np.asarray(info_np["msc_by_tau"], dtype=np.float64)
        delta_h_map = np.asarray(info_np["delta_h_map"], dtype=np.float64)
        delta_h_processed_map = np.asarray(info_np.get("delta_h_processed_map", delta_h_map), dtype=np.float64)
        best_idx = int(np.asarray(info_np["tau_selected_idx"]).item())
        best_tau = int(tau_steps[best_idx])
        win_starts = np.asarray(info_np["window_start_steps"], dtype=np.float64)
        win_centers = win_starts + 0.5 * int(metric_cfg["window_size_frames"])

        score_rows.append(
            {
                "family": family,
                "seed": seed,
                "score": float(np.asarray(info_np["score"]).item()),
                "msc": float(np.asarray(info_np["msc"]).item()),
                "amp": float(np.asarray(info_np["amp"]).item()),
                "delta_h_mean": float(np.asarray(info_np["delta_h_mean"]).item()),
                "delta_h_std": float(np.asarray(info_np["delta_h_std"]).item()),
                "delta_h_processed_mean": float(np.asarray(info_np.get("delta_h_processed_mean", info_np["delta_h_mean"])).item()),
                "delta_h_processed_std": float(np.asarray(info_np.get("delta_h_processed_std", info_np["delta_h_std"])).item()),
                "tau_best_steps": best_tau,
                "metrics_path": str(metrics_path),
                "trajectory_path": str(path),
            }
        )
        for i, tau in enumerate(tau_steps):
            tau_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "tau_steps": int(tau),
                    "score_by_tau": float(score_by_tau[i]),
                    "amp_by_tau": float(amp_by_tau[i]),
                    "msc_by_tau": float(msc_by_tau[i]),
                    "delta_h_median": float(np.nanmedian(delta_h_map[i])),
                    "delta_h_mean": float(np.nanmean(delta_h_map[i])),
                    "delta_h_std": float(np.nanstd(delta_h_map[i])),
                    "delta_h_processed_median": float(np.nanmedian(delta_h_processed_map[i])),
                    "delta_h_processed_mean": float(np.nanmean(delta_h_processed_map[i])),
                    "delta_h_processed_std": float(np.nanstd(delta_h_processed_map[i])),
                    "selected": bool(i == best_idx),
                }
            )

        rec = _role_recovery(xy_metric, labels, best_tau, seed=seed, domain=domain, positions_unwrapped=True)
        if rec is not None:
            role_rows.append({"family": family, "seed": seed, "tau_steps": best_tau, **rec})

        event_interval = metadata.get("event_interval")
        if event_interval is not None:
            event_rows.append({"family": family, "seed": seed, "tau_steps": best_tau, **_event_error(win_centers, delta_h_map[best_idx], event_interval)})

        scale_range = metadata.get("scale_range")
        if scale_range is not None:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            role_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "tau_steps": best_tau,
                    "ari": "" if rec is None else rec["ari"],
                    "n_roles": "" if rec is None else rec["n_roles"],
                    "scale_low": lo,
                    "scale_high": hi,
                    "tau_in_scale_range": bool(lo <= best_tau <= hi),
                }
            )
        item_seconds = time.perf_counter() - item_start
        log_event(
            f"synthetic metrics progress {_progress_text(idx, len(sim_files), layer_start, item_seconds=item_seconds)} "
            f"{family} seed={seed} status={item_status} tau_best={best_tau} metrics={metrics_path}",
            component="synthetic",
        )

    write_csv(dirs["root"] / "per_family_scores.csv", score_rows)
    write_csv(dirs["root"] / "tau_profiles.csv", tau_rows)
    write_csv(dirs["root"] / "role_recovery.csv", role_rows)
    write_csv(dirs["root"] / "event_localization.csv", event_rows)

    summary: dict[str, Any] = {"n_runs": len(score_rows), "families": {}}
    for family in sorted({row["family"] for row in score_rows}):
        vals = [row["score"] for row in score_rows if row["family"] == family]
        tau_vals = [row["tau_best_steps"] for row in score_rows if row["family"] == family]
        summary["families"][family] = {
            "n": len(vals),
            "score_median": float(np.median(vals)) if vals else float("nan"),
            "tau_best_median": float(np.median(tau_vals)) if tau_vals else float("nan"),
        }
    if role_rows:
        aris = [float(row["ari"]) for row in role_rows if row.get("ari") not in ("", None)]
        summary["role_recovery"] = sign_test_greater(aris)
    elapsed = time.perf_counter() - layer_start
    summary["elapsed_seconds"] = elapsed
    summary["elapsed"] = _format_duration(elapsed)
    write_json(dirs["root"] / "synthetic_calibration_summary.json", summary)
    log_event(
        f"synthetic metrics done n_runs={len(score_rows)} elapsed={_format_duration(elapsed)} "
        f"summary={dirs['root'] / 'synthetic_calibration_summary.json'}",
        component="synthetic",
    )
    return {"n_runs": len(score_rows), "summary_path": str(dirs["root"] / "synthetic_calibration_summary.json")}


def _ensure_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "paper_suite_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _suite_figures_dir(cfg: Any) -> Path:
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    return ensure_dir(output_root / "figures")


def _figure_fresh(output_path: Path, sources: list[Path], *, force: bool) -> bool:
    if force or not output_path.exists():
        return False
    try:
        out_mtime = output_path.stat().st_mtime
        return all(src.exists() and out_mtime >= src.stat().st_mtime for src in sources)
    except OSError:
        return False


def _save_figure_to_many(fig: Any, paths: list[Path], *, dpi: int = 180) -> None:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    if not unique_paths:
        return
    fig.savefig(unique_paths[0], dpi=dpi)
    for path in unique_paths[1:]:
        shutil.copyfile(unique_paths[0], path)


def _expected_metric_paths(cfg: Any, dirs: dict[str, Path]) -> list[tuple[str, int, Path]]:
    syn = cfg.get("synthetic", {})
    families = [str(x) for x in (syn.get("families", list(FAMILIES)) or list(FAMILIES))]
    seeds = _cfg_int(syn, "seeds", 3)
    return [(family, seed, dirs["metrics"] / f"{family}_seed_{seed:03d}_metrics.npz") for family in families for seed in range(seeds)]


def _plot_synthetic_grid_from_tables(
    *,
    dirs: dict[str, Path],
    suite_figures: Path,
    force: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    tau_path = dirs["root"] / "tau_profiles.csv"
    score_path = dirs["root"] / "per_family_scores.csv"
    out_suite = suite_figures / "synthetic_calibration_grid.png"
    out_local = dirs["figures"] / "synthetic_calibration_grid.png"
    missing = [str(path) for path in (tau_path, score_path) if not path.exists()]
    if missing:
        return {}, [
            {
                "artifact": "synthetic_calibration_grid",
                "status": "missing_inputs",
                "message": ";".join(missing),
                "path": str(out_suite),
            }
        ]
    if _figure_fresh(out_suite, [tau_path, score_path], force=force) and _figure_fresh(out_local, [tau_path, score_path], force=force):
        return {"synthetic_calibration_grid": str(out_suite), "synthetic_calibration_grid_local": str(out_local)}, []

    import pandas as pd

    tau = pd.read_csv(tau_path)
    scores = pd.read_csv(score_path)
    if tau.empty or scores.empty:
        return {}, [
            {
                "artifact": "synthetic_calibration_grid",
                "status": "empty_inputs",
                "message": f"{tau_path};{score_path}",
                "path": str(out_suite),
            }
        ]
    plt = _ensure_matplotlib()
    families = [f for f in FAMILIES if f in set(tau["family"])]
    if not families:
        return {}, [
            {
                "artifact": "synthetic_calibration_grid",
                "status": "empty_families",
                "message": f"{tau_path}",
                "path": str(out_suite),
            }
        ]
    fig, axes = plt.subplots(len(families), 2, figsize=(9, max(2.2, 2.0 * len(families))), squeeze=False)
    for row_idx, family in enumerate(families):
        ax0, ax1 = axes[row_idx]
        sub = tau[tau["family"] == family]
        grouped = sub.groupby("tau_steps")["score_by_tau"].agg(["median", "min", "max"]).reset_index()
        ax0.plot(grouped["tau_steps"], grouped["median"], marker="o", color="#2f6f9f")
        ax0.fill_between(grouped["tau_steps"], grouped["min"], grouped["max"], color="#2f6f9f", alpha=0.15)
        if np.all(grouped["tau_steps"].astype(float).to_numpy() > 0):
            ax0.set_xscale("log")
        ax0.set_ylabel(family)
        ax0.set_xlabel("tau")
        ax0.set_title("D(tau)")

        vals = scores[scores["family"] == family]["score"].astype(float).to_numpy()
        ax1.scatter(np.arange(vals.size), vals, color="#444444", s=24)
        if vals.size:
            ax1.axhline(float(np.median(vals)), color="#c43c39", linewidth=1)
        ax1.set_xlabel("seed")
        ax1.set_title("selected score")
    fig.tight_layout()
    _save_figure_to_many(fig, [out_suite, out_local])
    plt.close(fig)
    return {"synthetic_calibration_grid": str(out_suite), "synthetic_calibration_grid_local": str(out_local)}, []


def _heatmap_tick_positions(n: int, *, max_ticks: int = 7) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.linspace(0, n - 1, num=min(max_ticks, n), dtype=np.int32))


def _load_metric_for_visualization(metrics_path: Path) -> dict[str, Any]:
    with np.load(metrics_path, allow_pickle=False) as data:
        required = {"delta_h_map", "tau_steps", "window_start_steps"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{metrics_path} is missing required visualization keys: {sorted(missing)}")
        delta_h_map = np.asarray(data["delta_h_map"], dtype=np.float64)
        tau_steps = np.asarray(data["tau_steps"], dtype=np.int32).reshape(-1)
        window_start_steps = np.asarray(data["window_start_steps"], dtype=np.int32).reshape(-1)
        if delta_h_map.ndim != 2:
            raise ValueError(f"{metrics_path} delta_h_map must be 2D (tau, window), got {delta_h_map.shape}.")
        if tau_steps.size != delta_h_map.shape[0]:
            raise ValueError(
                f"{metrics_path} tau_steps length {tau_steps.size} does not match delta_h_map rows {delta_h_map.shape[0]}."
            )
        if window_start_steps.size != delta_h_map.shape[1]:
            raise ValueError(
                f"{metrics_path} window_start_steps length {window_start_steps.size} "
                f"does not match delta_h_map columns {delta_h_map.shape[1]}."
            )
        metadata: dict[str, Any] = {}
        if "_trajectory_metadata_json" in data.files:
            metadata = json.loads(str(np.asarray(data["_trajectory_metadata_json"]).item()))
        selected_idx = int(np.asarray(data["tau_selected_idx"]).item()) if "tau_selected_idx" in data.files else None
        score_by_tau = np.asarray(data["score_by_tau"], dtype=np.float64) if "score_by_tau" in data.files else None
    return {
        "delta_h_map": delta_h_map,
        "tau_steps": tau_steps,
        "window_start_steps": window_start_steps,
        "metadata": metadata,
        "selected_idx": selected_idx,
        "score_by_tau": score_by_tau,
    }


def _plot_delta_h_heatmap(
    *,
    arr: np.ndarray,
    tau_steps: np.ndarray,
    window_start_steps: np.ndarray,
    title: str,
    output_path: Path,
    selected_idx: int | None = None,
    event_interval: list[int] | None = None,
    cmap: str = "coolwarm",
    vabs: float | None = None,
) -> None:
    plt = _ensure_matplotlib()
    z = np.asarray(arr, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Delta-H heatmap expects 2D array, got {z.shape}.")
    if vabs is None:
        finite = z[np.isfinite(z)]
        vabs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(vabs) or vabs <= 0.0:
        vabs = 1.0

    fig, ax = plt.subplots(figsize=(10.0, 3.2), dpi=180)
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-float(vabs),
        vmax=float(vabs),
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_ylabel("tau (steps)")
    y_idx = np.arange(len(tau_steps), dtype=np.int32)
    ax.set_yticks(y_idx)
    ax.set_yticklabels([str(int(x)) for x in tau_steps])
    x_idx = _heatmap_tick_positions(len(window_start_steps))
    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(int(window_start_steps[i])) for i in x_idx])
    ax.set_xlabel("window start (steps)")
    if selected_idx is not None and 0 <= int(selected_idx) < z.shape[0]:
        ax.axhline(float(selected_idx), color="#111111", linewidth=1.1, alpha=0.85)
    if event_interval is not None and len(event_interval) == 2 and len(window_start_steps):
        lo = int(event_interval[0])
        hi = int(event_interval[1])
        x0 = max(-0.5, float(np.searchsorted(window_start_steps, lo, side="left")) - 0.5)
        x1 = min(float(z.shape[1]) - 0.5, float(np.searchsorted(window_start_steps, hi, side="right")) - 0.5)
        if x1 >= x0:
            ax.axvspan(x0, x1, color="#222222", alpha=0.08, linewidth=0)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Delta-H")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_family_heatmap_grid(
    *,
    family_maps: dict[str, np.ndarray],
    tau_steps: np.ndarray,
    window_start_steps: np.ndarray,
    output_paths: list[Path],
    cmap: str,
) -> None:
    if not family_maps:
        return
    plt = _ensure_matplotlib()
    families = [family for family in FAMILIES if family in family_maps]
    finite_parts = [np.asarray(family_maps[family], dtype=np.float64) for family in families]
    finite_chunks = [arr[np.isfinite(arr)].reshape(-1) for arr in finite_parts if np.isfinite(arr).any()]
    finite = np.concatenate(finite_chunks) if finite_chunks else np.asarray([], dtype=np.float64)
    vabs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(vabs) or vabs <= 0.0:
        vabs = 1.0
    fig, axes = plt.subplots(len(families), 1, figsize=(10.5, max(2.2, 2.0 * len(families))), dpi=180, squeeze=False)
    x_idx = _heatmap_tick_positions(len(window_start_steps))
    for ax, family in zip(axes[:, 0], families, strict=False):
        im = ax.imshow(
            family_maps[family],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-vabs,
            vmax=vabs,
            interpolation="nearest",
        )
        ax.set_title(f"{family}: median Delta-H across seeds")
        ax.set_ylabel("tau")
        ax.set_yticks(np.arange(len(tau_steps), dtype=np.int32))
        ax.set_yticklabels([str(int(x)) for x in tau_steps])
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(int(window_start_steps[i])) for i in x_idx])
        ax.set_xlabel("window start")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    _save_figure_to_many(fig, output_paths)
    plt.close(fig)


def visualize(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    syn = cfg.get("synthetic", {})
    dirs = _synthetic_dirs(cfg)
    suite_figures = _suite_figures_dir(cfg)
    vis_cfg = syn.get("visualization", {}) if syn is not None else {}
    heatmaps_enabled = _cfg_bool(vis_cfg, "heatmaps_enabled", True)
    heatmap_max_runs = _cfg_optional_int(vis_cfg, "heatmap_max_runs")
    heatmap_cmap = str(vis_cfg.get("heatmap_cmap", "coolwarm") if vis_cfg is not None else "coolwarm")
    paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    layer_start = time.perf_counter()
    metric_items = _expected_metric_paths(cfg, dirs)

    log_event(
        f"synthetic visualization start smoke={smoke} force={force} heatmaps={heatmaps_enabled} n_metric_runs={len(metric_items)}",
        component="synthetic",
    )
    grid_start = time.perf_counter()
    grid_paths, grid_skips = _plot_synthetic_grid_from_tables(dirs=dirs, suite_figures=suite_figures, force=force)
    paths.update(grid_paths)
    skip_rows.extend(grid_skips)
    log_event(
        f"synthetic visualization grid done status={'written_or_exists' if grid_paths else 'skipped'} "
        f"time={_format_duration(time.perf_counter() - grid_start)}",
        component="synthetic",
    )

    family_runs: dict[str, list[np.ndarray]] = {}
    family_refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if heatmaps_enabled:
        rendered_runs = 0
        for item_idx, (family, seed, metrics_path) in enumerate(metric_items, start=1):
            item_start = time.perf_counter()
            out_path = dirs["heatmaps"] / f"{family}_seed_{seed:03d}_delta_h_heatmap.png"
            row = {
                "artifact": "delta_h_heatmap",
                "family": family,
                "seed": seed,
                "metrics_path": str(metrics_path),
                "path": str(out_path),
            }
            if not metrics_path.exists():
                row.update({"status": "missing_metrics", "message": "run synthetic metrics first"})
                rows.append(row)
                skip_rows.append(row.copy())
                item_seconds = time.perf_counter() - item_start
                log_event(
                    f"synthetic visualization heatmap progress "
                    f"{_progress_text(item_idx, len(metric_items), layer_start, item_seconds=item_seconds)} "
                    f"{family} seed={seed} status=missing_metrics",
                    component="synthetic",
                )
                continue
            metric_data = _load_metric_for_visualization(metrics_path)
            metadata = dict(metric_data.get("metadata", {}))
            metadata.setdefault("family", family)
            metadata.setdefault("seed", seed)
            event_interval = metadata.get("event_interval")

            ref = family_refs.get(family)
            if ref is None:
                family_refs[family] = (metric_data["tau_steps"], metric_data["window_start_steps"])
                family_runs.setdefault(family, []).append(metric_data["delta_h_map"])
            elif np.array_equal(ref[0], metric_data["tau_steps"]) and np.array_equal(ref[1], metric_data["window_start_steps"]):
                family_runs.setdefault(family, []).append(metric_data["delta_h_map"])
            else:
                rows.append(
                    {
                        **row,
                        "status": "aggregate_skipped_shape_mismatch",
                        "message": "metric tau/window grid differs inside family",
                    }
                )

            if heatmap_max_runs is not None and rendered_runs >= int(heatmap_max_runs):
                row.update({"status": "skipped_limit", "message": f"heatmap_max_runs={heatmap_max_runs}"})
                rows.append(row)
                skip_rows.append(row.copy())
                item_seconds = time.perf_counter() - item_start
                log_event(
                    f"synthetic visualization heatmap progress "
                    f"{_progress_text(item_idx, len(metric_items), layer_start, item_seconds=item_seconds)} "
                    f"{family} seed={seed} status=skipped_limit",
                    component="synthetic",
                )
                continue
            if _figure_fresh(out_path, [metrics_path], force=force):
                status = "exists"
            else:
                _plot_delta_h_heatmap(
                    arr=metric_data["delta_h_map"],
                    tau_steps=metric_data["tau_steps"],
                    window_start_steps=metric_data["window_start_steps"],
                    title=f"{family} seed={seed:03d}: Delta-H by tau",
                    output_path=out_path,
                    selected_idx=metric_data["selected_idx"],
                    event_interval=event_interval,
                    cmap=heatmap_cmap,
                )
                status = "written"
            rendered_runs += 1
            key = f"synthetic_delta_h_heatmap_{family}_seed_{seed:03d}"
            paths[key] = str(out_path)
            row.update({"status": status, "message": ""})
            rows.append(row)
            item_seconds = time.perf_counter() - item_start
            log_event(
                f"synthetic visualization heatmap progress "
                f"{_progress_text(item_idx, len(metric_items), layer_start, item_seconds=item_seconds)} "
                f"{family} seed={seed} status={status}",
                component="synthetic",
            )

        aggregate_start = time.perf_counter()
        aggregate_maps: dict[str, np.ndarray] = {}
        aggregate_tau: np.ndarray | None = None
        aggregate_windows: np.ndarray | None = None
        npz_payload: dict[str, np.ndarray] = {}
        for family in [f for f in FAMILIES if f in family_runs]:
            maps = family_runs[family]
            if not maps:
                continue
            tau_steps, window_steps = family_refs[family]
            arr = np.median(np.stack(maps, axis=0), axis=0)
            aggregate_maps[family] = arr
            npz_payload[f"{family}_delta_h_median"] = arr
            if aggregate_tau is None:
                aggregate_tau = tau_steps
                aggregate_windows = window_steps
            out_path = dirs["heatmaps"] / f"{family}_delta_h_heatmap_median.png"
            if not _figure_fresh(out_path, [dirs["metrics"] / f"{family}_seed_{seed:03d}_metrics.npz" for seed in range(_cfg_int(syn, "seeds", 3))], force=force):
                _plot_delta_h_heatmap(
                    arr=arr,
                    tau_steps=tau_steps,
                    window_start_steps=window_steps,
                    title=f"{family}: median Delta-H by tau",
                    output_path=out_path,
                    selected_idx=None,
                    event_interval=None,
                    cmap=heatmap_cmap,
                )
            paths[f"synthetic_delta_h_heatmap_{family}_median"] = str(out_path)
        if aggregate_maps and aggregate_tau is not None and aggregate_windows is not None:
            np.savez_compressed(
                dirs["heatmaps"] / "delta_h_family_heatmaps.npz",
                tau_steps=aggregate_tau,
                window_start_steps=aggregate_windows,
                **npz_payload,
            )
            same_grid = all(
                np.array_equal(family_refs[family][0], aggregate_tau) and np.array_equal(family_refs[family][1], aggregate_windows)
                for family in aggregate_maps
            )
            if same_grid:
                out_suite = suite_figures / "synthetic_delta_h_heatmaps.png"
                out_local = dirs["figures"] / "synthetic_delta_h_heatmaps.png"
                _plot_family_heatmap_grid(
                    family_maps=aggregate_maps,
                    tau_steps=aggregate_tau,
                    window_start_steps=aggregate_windows,
                    output_paths=[out_suite, out_local],
                    cmap=heatmap_cmap,
                )
                paths["synthetic_delta_h_heatmaps"] = str(out_suite)
                paths["synthetic_delta_h_heatmaps_local"] = str(out_local)
            else:
                skip_rows.append(
                    {
                        "artifact": "synthetic_delta_h_heatmaps",
                        "status": "skipped_shape_mismatch",
                        "message": "family aggregate grids differ",
                        "path": str(suite_figures / "synthetic_delta_h_heatmaps.png"),
                    }
                )
        log_event(
            f"synthetic visualization aggregates done families={len(aggregate_maps)} "
            f"time={_format_duration(time.perf_counter() - aggregate_start)}",
            component="synthetic",
        )
    else:
        skip_rows.append(
            {
                "artifact": "delta_h_heatmaps",
                "status": "disabled",
                "message": "synthetic.visualization.heatmaps_enabled=false",
                "path": str(dirs["heatmaps"]),
            }
        )

    write_csv(
        dirs["root"] / "delta_h_heatmap_manifest.csv",
        rows,
        fieldnames=["artifact", "family", "seed", "metrics_path", "path", "status", "message"],
    )
    write_csv(
        dirs["root"] / "visualization_skips.csv",
        skip_rows,
        fieldnames=["artifact", "family", "seed", "metrics_path", "path", "status", "message"],
    )
    elapsed = time.perf_counter() - layer_start
    summary = {
        "n_figures": len(paths),
        "figure_paths": paths,
        "n_skips": len(skip_rows),
        "skip_manifest": str(dirs["root"] / "visualization_skips.csv"),
        "heatmap_manifest": str(dirs["root"] / "delta_h_heatmap_manifest.csv"),
        "elapsed_seconds": elapsed,
        "elapsed": _format_duration(elapsed),
    }
    write_json(dirs["root"] / "visualization_summary.json", summary)
    log_event(
        f"synthetic visualization done n_figures={len(paths)} elapsed={_format_duration(elapsed)} "
        f"summary={dirs['root'] / 'visualization_summary.json'}",
        component="synthetic",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mandatory synthetic MSPD calibration S0-S7.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--layer", choices=["simulation", "metrics", "visualization", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Skip synthetic trajectory mp4 rendering during the simulation layer.")
    args = parser.parse_args(argv)
    master_log = init_suite_logging(args.config, smoke=args.smoke, layer=args.layer, task="synthetic")
    log_event(f"synthetic direct log: {master_log}", component="synthetic")
    if args.layer in {"simulation", "all"}:
        print(simulate(args.config, smoke=args.smoke, force=args.force, render_videos=False if args.no_videos else None))
    if args.layer in {"metrics", "all"}:
        print(metrics(args.config, smoke=args.smoke, force=args.force))
    if args.layer in {"visualization", "all"}:
        print(visualize(args.config, smoke=args.smoke, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
