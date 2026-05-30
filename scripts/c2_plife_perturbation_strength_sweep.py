from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from paper_suite_c2_plife_plus import (
    _branch_plan_meta_current,
    _build_substrate,
    _future_chamfer,
    _get,
    _int_or,
    _out_dir,
    _output_root,
    _plife_c2_cfg,
    _simulate_one_branch,
    metrics as c2_plife_metrics,
    simulation as c2_plife_simulation,
)
from paper_suite_common import command_to_str, ensure_dir, load_config, log_event, read_csv, resolve_path, write_csv, write_json
from render_plife_videos import _frame_indices, _open_video_writer, _write_rgb_video, _write_xy_video, _xy_frame


def _parse_floats(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = list(raw)
    out = [float(part) for part in parts if str(part).strip() != ""]
    if not out:
        raise ValueError("Empty strength list.")
    return out


def _parse_conditions(raw: Any) -> set[str]:
    if raw is None:
        return {"high"}
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = [str(part).strip() for part in raw]
    return {part for part in parts if part}


def _safe_tag(value: Any) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", text)


def _safe_path_id(value: Any) -> str:
    text = str(value)
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return out.strip("_") or "traj"


def _legacy_strength_output_path(point_dir: Path, strength: float, rep_id: int) -> Path:
    return point_dir / f"strength_{_safe_tag(strength)}" / f"rep_{int(rep_id):03d}" / "branch_output.npz"


def _float_or(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _sweep_cfg(c2_cfg: Any) -> Any:
    return _get(c2_cfg, "perturbation_strength_sweep", {})


def _resolve_output_root(cfg: Any, c2_cfg: Any, args: argparse.Namespace) -> Path:
    raw = args.output_root if args.output_root is not None else _get(_sweep_cfg(c2_cfg), "output_dir", None)
    if raw:
        path = resolve_path(raw)
        return ensure_dir(path if path is not None else _output_root(cfg) / "c2_plife_plus_perturbation_strength_sweep")
    return ensure_dir(_output_root(cfg) / "c2_plife_plus_perturbation_strength_sweep")


def _ensure_branch_plan(config_path: str | Path, *, smoke: bool, cfg: Any, output_root: Path, force_plan: bool) -> Path:
    out_dir = _out_dir(cfg, output_root)
    plan_path = out_dir / "branch_plan.csv"
    plan_ok, plan_reason = _branch_plan_meta_current(plan_path)
    if force_plan or not plan_ok:
        log_event(
            f"PLife++ perturbation sweep preparing C2 branch plan reason={plan_reason} force_plan={force_plan}",
            component="c2-plife-sweep",
        )
        c2_plife_metrics(config_path, smoke=smoke, force=False)
        c2_plife_simulation(config_path, smoke=smoke, force=bool(force_plan), allow_heavy=False, dry_run=False)
        plan_ok, plan_reason = _branch_plan_meta_current(plan_path)
    if not plan_ok:
        raise FileNotFoundError(f"PLife++ C2 branch_plan is not ready: {plan_reason}")
    log_event(f"PLife++ perturbation sweep using branch plan {plan_path}", component="c2-plife-sweep")
    return plan_path


def _select_points(plan_rows: list[dict[str, str]], *, conditions: set[str], n_points: int) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for row in plan_rows:
        condition = str(row.get("condition", ""))
        if conditions and condition not in conditions:
            continue
        key = (str(row.get("traj_id", "")), str(row.get("point_id", row.get("pair_id", ""))))
        existing = groups.get(key)
        if existing is None or _int_or(row.get("branch_id", 0), 0) < _int_or(existing.get("branch_id", 0), 0):
            groups[key] = row
    points = list(groups.values())
    points.sort(key=lambda row: -_float_or(row.get("delta_h_energy", row.get("delta_h", "nan"))))
    return points[: max(0, int(n_points))]


def _load_branch_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
            if key
            in {
                "xy_future",
                "rgb_future",
                "c_future",
                "strength",
                "x_std",
                "v_std",
                "c_std",
                "pre_perturb_initial_frame",
            }
        }


def _rgb_uint8(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"expected RGB frame stack shape (T,H,W,C), got {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] < 3:
        arr = np.repeat(arr, int(math.ceil(3 / max(1, arr.shape[-1]))), axis=-1)[..., :3]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def _resize_nearest(frame: np.ndarray, panel_size: int) -> np.ndarray:
    size = int(panel_size)
    if size <= 0:
        return np.asarray(frame, dtype=np.uint8)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h == size and w == size:
        return np.asarray(frame, dtype=np.uint8)
    yy = np.linspace(0, h - 1, size).round().astype(np.int64)
    xx = np.linspace(0, w - 1, size).round().astype(np.int64)
    return np.asarray(frame[yy][:, xx], dtype=np.uint8)


def _color_strip(c_future: np.ndarray, frame_idx: int, *, width: int, height: int) -> np.ndarray:
    c = np.asarray(c_future, dtype=np.float32)
    if c.ndim != 3 or c.shape[-1] < 3 or c.shape[1] <= 0:
        return np.zeros((0, int(width), 3), dtype=np.uint8)
    t = int(np.clip(frame_idx, 0, c.shape[0] - 1))
    rgb = np.clip((c[t, :, :3] + 1.0) * 0.5, 0.0, 1.0)
    cols = np.linspace(0, rgb.shape[0] - 1, max(1, int(width))).round().astype(np.int64)
    strip = (rgb[cols][None, :, :] * 255.0).astype(np.uint8)
    return np.repeat(strip, max(1, int(height)), axis=0)


def _append_color_strip(frame: np.ndarray, arrays: dict[str, np.ndarray], frame_idx: int, *, enabled: bool, height: int) -> np.ndarray:
    out = np.asarray(frame, dtype=np.uint8)
    if not enabled or "c_future" not in arrays or int(height) <= 0:
        return out
    strip = _color_strip(arrays["c_future"], frame_idx, width=out.shape[1], height=int(height))
    if strip.size == 0:
        return out
    return np.concatenate([out, strip], axis=0)


def _load_video_frames(
    path: Path,
    *,
    panel_size: int,
    max_frames: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
    debug_color_strip: bool,
    color_strip_height: int,
) -> list[np.ndarray]:
    arrays = _load_branch_arrays(path)
    if "rgb_future" in arrays:
        frames = _rgb_uint8(arrays["rgb_future"])
        idx = _frame_indices(frames.shape[0], max_frames)
        return [
            _append_color_strip(
                _resize_nearest(frames[int(i)], panel_size),
                arrays,
                int(i),
                enabled=debug_color_strip,
                height=color_strip_height,
            )
            for i in idx
        ]
    if "xy_future" in arrays:
        xy = np.asarray(arrays["xy_future"], dtype=np.float32)
        idx = _frame_indices(xy.shape[0], max_frames)
        return [
            _append_color_strip(
                _xy_frame(
                    xy,
                    int(i),
                    img_size=int(panel_size),
                    radius=int(radius),
                    trail_steps=int(trail_steps),
                    domain_size=float(domain_size),
                    wrap=True,
                ),
                arrays,
                int(i),
                enabled=debug_color_strip,
                height=color_strip_height,
            )
            for i in idx
        ]
    raise ValueError(f"No rgb_future or xy_future in {path}")


def _load_frames_at_indices(
    path: Path,
    indices: list[int],
    *,
    panel_size: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
    debug_color_strip: bool,
    color_strip_height: int,
) -> list[tuple[int, np.ndarray]]:
    arrays = _load_branch_arrays(path)
    if "rgb_future" in arrays:
        frames = _rgb_uint8(arrays["rgb_future"])
        out: list[tuple[int, np.ndarray]] = []
        for idx in indices:
            i = int(idx)
            if 0 <= i < int(frames.shape[0]):
                out.append(
                    (
                        i,
                        _append_color_strip(
                            _resize_nearest(frames[i], panel_size),
                            arrays,
                            i,
                            enabled=debug_color_strip,
                            height=color_strip_height,
                        ),
                    )
                )
        return out
    if "xy_future" in arrays:
        xy = np.asarray(arrays["xy_future"], dtype=np.float32)
        out = []
        for idx in indices:
            i = int(idx)
            if 0 <= i < int(xy.shape[0]):
                out.append(
                    (
                        i,
                        _append_color_strip(
                            _xy_frame(
                                xy,
                                i,
                                img_size=int(panel_size),
                                radius=int(radius),
                                trail_steps=int(trail_steps),
                                domain_size=float(domain_size),
                                wrap=True,
                            ),
                            arrays,
                            i,
                            enabled=debug_color_strip,
                            height=color_strip_height,
                        ),
                    )
                )
        return out
    raise ValueError(f"No rgb_future or xy_future in {path}")


def _write_frame_list_video(
    frames: list[np.ndarray],
    output: Path,
    *,
    fps: float,
    codec: str,
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "video_path": str(output), "n_frames": 0}
    if not frames:
        return {"status": "skipped_empty", "video_path": str(output), "n_frames": 0}
    ensure_dir(output.parent)
    writer, cv2 = _open_video_writer(output, fps=fps, codec=codec, frame_shape=frames[0].shape)
    written = 0
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()
    return {"status": "written", "video_path": str(output), "n_frames": int(written)}


def _write_image(output: Path, frame: np.ndarray, *, force: bool) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "image_path": str(output)}
    import cv2  # type: ignore

    ensure_dir(output.parent)
    ok = cv2.imwrite(str(output), cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not write image {output}")
    return {"status": "written", "image_path": str(output)}


def _grid_canvas(frames: list[np.ndarray], labels: list[str], *, n_cols: int):
    import cv2  # type: ignore

    if not frames:
        raise ValueError("No frames for grid canvas.")
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(len(frames) / n_cols))
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    label_h = 24
    canvas = np.full((n_rows * (h + label_h), n_cols * w, 3), 255, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // n_cols
        col = idx % n_cols
        y0 = row * (h + label_h)
        x0 = col * w
        cv2.putText(canvas, labels[idx], (x0 + 6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[y0 + label_h : y0 + label_h + h, x0 : x0 + w] = frame
    return canvas


def _frame_label(frame_idx: int, sample_every: int, *, has_initial_frame: bool) -> str:
    if has_initial_frame and int(frame_idx) == 0:
        return "pre"
    if has_initial_frame:
        return f"+{int(frame_idx) * max(1, int(sample_every))}"
    return f"+{(int(frame_idx) + 1) * max(1, int(sample_every))}"


def _write_first_frame_montage(
    *,
    output: Path,
    items: list[tuple[float, int, Path]],
    frame_count: int,
    panel_size: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
    sample_every: int,
    has_initial_frame: bool,
    debug_color_strip: bool,
    color_strip_height: int,
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "image_path": str(output), "n_tiles": 0}
    n_frames = max(1, int(frame_count))
    frame_indices = list(range(n_frames))
    frames: list[np.ndarray] = []
    labels: list[str] = []
    for strength, rep_id, path in sorted(items, key=lambda item: (item[0], item[1], str(item[2]))):
        selected = _load_frames_at_indices(
            path,
            frame_indices,
            panel_size=panel_size,
            radius=radius,
            trail_steps=trail_steps,
            domain_size=domain_size,
            debug_color_strip=debug_color_strip,
            color_strip_height=color_strip_height,
        )
        for frame_idx, frame in selected:
            frames.append(frame)
            labels.append(f"s={strength:g} r={rep_id} {_frame_label(frame_idx, sample_every, has_initial_frame=has_initial_frame)}")
    if not frames:
        return {"status": "skipped_empty", "image_path": str(output), "n_tiles": 0}
    canvas = _grid_canvas(frames, labels, n_cols=n_frames)
    result = _write_image(output, canvas, force=force)
    result["n_tiles"] = int(len(frames))
    return result


def _write_grid_video(
    *,
    output: Path,
    items: list[tuple[Path, str]],
    fps: float,
    codec: str,
    panel_size: int,
    max_frames: int,
    grid_cols: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
    debug_color_strip: bool,
    color_strip_height: int,
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "video_path": str(output), "n_frames": 0}
    series: list[tuple[list[np.ndarray], str]] = []
    for path, label in items:
        frames = _load_video_frames(
            path,
            panel_size=panel_size,
            max_frames=max_frames,
            radius=radius,
            trail_steps=trail_steps,
            domain_size=domain_size,
            debug_color_strip=debug_color_strip,
            color_strip_height=color_strip_height,
        )
        if frames:
            series.append((frames, label))
    if not series:
        return {"status": "skipped_empty", "video_path": str(output), "n_frames": 0}
    n_frames = max(len(frames) for frames, _label in series)
    if n_frames <= 0:
        return {"status": "skipped_empty", "video_path": str(output), "n_frames": 0}
    labels = [label for _frames, label in series]
    first = _grid_canvas([frames[0] for frames, _label in series], labels, n_cols=grid_cols)
    ensure_dir(output.parent)
    writer, cv2 = _open_video_writer(output, fps=fps, codec=codec, frame_shape=first.shape)
    written = 0
    try:
        for idx in range(n_frames):
            frame = (
                first
                if idx == 0
                else _grid_canvas([frames[min(idx, len(frames) - 1)] for frames, _label in series], labels, n_cols=grid_cols)
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()
    return {"status": "written", "video_path": str(output), "n_frames": int(written)}


def _compute_divergences(rows: list[dict[str, Any]], *, domain: float, max_particles: int) -> None:
    by_point_rep: dict[tuple[int, int], Path] = {}
    for row in rows:
        if abs(float(row["strength"])) <= 1e-12 and str(row.get("status")) in {"written", "exists"}:
            by_point_rep[(int(row["sweep_point_id"]), int(row["rep_id"]))] = Path(str(row["output_path"]))

    cache: dict[Path, dict[str, np.ndarray]] = {}

    def load(path: Path) -> dict[str, np.ndarray]:
        if path not in cache:
            cache[path] = _load_branch_arrays(path)
        return cache[path]

    for row in rows:
        if str(row.get("status")) not in {"written", "exists"}:
            continue
        baseline = by_point_rep.get((int(row["sweep_point_id"]), int(row["rep_id"])))
        if baseline is None or not baseline.exists():
            row["divergence_status"] = "missing_baseline"
            continue
        output = Path(str(row["output_path"]))
        try:
            base = load(baseline)
            cur = load(output)
            for key in ("strength", "x_std", "v_std", "c_std"):
                if key in cur:
                    saved = float(np.asarray(cur[key]).reshape(-1)[0])
                    row[f"saved_{key}"] = saved
                    planned = _float_or(row.get(key), float("nan"))
                    if np.isfinite(planned):
                        row[f"{key}_saved_minus_planned"] = float(saved - planned)
            if "xy_future" in base and "xy_future" in cur:
                cur_xy = np.asarray(cur["xy_future"])
                base_xy = np.asarray(base["xy_future"])
                if bool(row.get("pre_perturb_initial_frame", False)) and cur_xy.shape[0] > 1 and base_xy.shape[0] > 1:
                    cur_xy = cur_xy[1:]
                    base_xy = base_xy[1:]
                row["xy_chamfer_mean_vs_baseline"] = _future_chamfer(
                    cur_xy,
                    base_xy,
                    domain=float(domain),
                    max_particles=int(max_particles),
                )
                row["xy_chamfer_final_vs_baseline"] = _future_chamfer(
                    cur_xy[-1:],
                    base_xy[-1:],
                    domain=float(domain),
                    max_particles=int(max_particles),
                )
            if "rgb_future" in base and "rgb_future" in cur:
                a = np.asarray(cur["rgb_future"], dtype=np.float32)
                b = np.asarray(base["rgb_future"], dtype=np.float32)
                if bool(row.get("pre_perturb_initial_frame", False)) and a.shape[0] > 1 and b.shape[0] > 1:
                    a = a[1:]
                    b = b[1:]
                n = min(int(a.shape[0]), int(b.shape[0]))
                if n > 0:
                    diff = a[:n] - b[:n]
                    row["rgb_mse_mean_vs_baseline"] = float(np.mean(diff**2))
                    row["rgb_mean_abs_vs_baseline"] = float(np.mean(np.abs(diff)))
                    row["rgb_max_abs_vs_baseline"] = float(np.max(np.abs(diff)))
                else:
                    row["rgb_mse_mean_vs_baseline"] = float("nan")
            if "c_future" in base and "c_future" in cur:
                c_cur = np.asarray(cur["c_future"], dtype=np.float32)
                c_base = np.asarray(base["c_future"], dtype=np.float32)
                if bool(row.get("pre_perturb_initial_frame", False)) and c_cur.shape[0] > 1 and c_base.shape[0] > 1:
                    c_cur = c_cur[1:]
                    c_base = c_base[1:]
                n = min(int(c_cur.shape[0]), int(c_base.shape[0]))
                if n > 0:
                    c_delta = c_cur[:n] - c_base[:n]
                    c3_delta = c_cur[:n, :, :3] - c_base[:n, :, :3]
                    c3 = c_cur[:n, :, :3]
                    row["c_mean_abs_vs_baseline"] = float(np.mean(np.abs(c_delta)))
                    row["c_max_abs_vs_baseline"] = float(np.max(np.abs(c_delta)))
                    row["c_first3_mean_abs_vs_baseline"] = float(np.mean(np.abs(c3_delta)))
                    row["c_first3_max_abs_vs_baseline"] = float(np.max(np.abs(c3_delta)))
                    row["c_first3_min"] = float(np.min(c3))
                    row["c_first3_max"] = float(np.max(c3))
                    row["c_first3_std"] = float(np.std(c3))
            row["baseline_output_path"] = str(baseline)
            row["divergence_status"] = "ok"
        except Exception as exc:
            row["divergence_status"] = "error"
            row["divergence_message"] = f"{type(exc).__name__}: {exc}"


def _divergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = np.asarray([_float_or(row.get("xy_chamfer_mean_vs_baseline", "nan")) for row in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n_xy_divergence": 0}
    return {
        "n_xy_divergence": int(vals.size),
        "xy_chamfer_mean_min": float(np.min(vals)),
        "xy_chamfer_mean_median": float(np.median(vals)),
        "xy_chamfer_mean_max": float(np.max(vals)),
    }


def _color_debug_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c_vals = np.asarray([_float_or(row.get("c_first3_mean_abs_vs_baseline", "nan")) for row in rows], dtype=np.float64)
    c_vals = c_vals[np.isfinite(c_vals)]
    rgb_vals = np.asarray([_float_or(row.get("rgb_mean_abs_vs_baseline", "nan")) for row in rows], dtype=np.float64)
    rgb_vals = rgb_vals[np.isfinite(rgb_vals)]
    summary: dict[str, Any] = {
        "n_c_first3_debug": int(c_vals.size),
        "n_rgb_debug": int(rgb_vals.size),
    }
    if c_vals.size:
        summary.update(
            {
                "c_first3_mean_abs_max": float(np.max(c_vals)),
                "c_first3_mean_abs_median": float(np.median(c_vals)),
            }
        )
    if rgb_vals.size:
        summary.update(
            {
                "rgb_mean_abs_max": float(np.max(rgb_vals)),
                "rgb_mean_abs_median": float(np.median(rgb_vals)),
            }
        )
    return summary


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _subsample_vectors(arr: np.ndarray, max_items: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        return x.reshape((0, 0))
    n = int(x.shape[0])
    if n <= int(max_items) or int(max_items) <= 0:
        return x
    idx = np.linspace(0, n - 1, int(max_items)).round().astype(np.int64)
    return x[np.unique(idx)]


def _vector_chamfer(a: np.ndarray, b: np.ndarray, *, max_items: int) -> float:
    aa = _subsample_vectors(np.asarray(a, dtype=np.float32), max_items=max_items)
    bb = _subsample_vectors(np.asarray(b, dtype=np.float32), max_items=max_items)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    d = aa[:, None, :] - bb[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    return float(0.5 * (np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))))


def _identity_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n = min(int(aa.shape[0]), int(bb.shape[0]))
    if n <= 0:
        return float("nan")
    return float(np.mean(np.sqrt(np.sum((aa[:n] - bb[:n]) ** 2, axis=-1))))


def _elapsed_steps(frame_idx: int, sample_every: int, *, has_initial_frame: bool) -> int:
    if has_initial_frame:
        return int(frame_idx) * max(1, int(sample_every))
    return (int(frame_idx) + 1) * max(1, int(sample_every))


def _write_color_relaxation_plot(timeseries_rows: list[dict[str, Any]], output_path: Path, *, metric_key: str) -> str | None:
    if not timeseries_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    point_ids = sorted({_int_or(row.get("sweep_point_id", -1), -1) for row in timeseries_rows})
    point_ids = [pid for pid in point_ids if pid >= 0]
    if not point_ids:
        return None
    n_cols = min(3, len(point_ids))
    n_rows = int(math.ceil(len(point_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    strengths = sorted({_float_or(row.get("strength"), float("nan")) for row in timeseries_rows if np.isfinite(_float_or(row.get("strength"), float("nan")))})
    denom = max(1, len(strengths) - 1)
    color_by_strength = {strength: cmap(i / denom) for i, strength in enumerate(strengths)}

    for ax in axes.ravel():
        ax.axis("off")
    for ax, point_id in zip(axes.ravel(), point_ids):
        ax.axis("on")
        rows = [row for row in timeseries_rows if _int_or(row.get("sweep_point_id", -1), -1) == point_id]
        groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
        for row in rows:
            strength = _float_or(row.get("strength"), float("nan"))
            rep = _int_or(row.get("rep_id", 0), 0)
            if np.isfinite(strength):
                groups.setdefault((strength, rep), []).append(row)
        for (strength, rep), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
            group = sorted(group, key=lambda row: _int_or(row.get("elapsed_steps", 0), 0))
            x = np.asarray([_int_or(row.get("elapsed_steps", 0), 0) for row in group], dtype=np.float64)
            y = np.asarray([_float_or(row.get(metric_key), float("nan")) for row in group], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue
            style = "--" if abs(strength) <= 1e-12 else "-"
            ax.plot(x[finite], y[finite], style, lw=2.0, color=color_by_strength[strength], label=f"s={strength:g} r={rep}")
        ax.set_title(f"point {point_id}")
        ax.set_xlabel("steps after branch")
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _write_color_relaxation_outputs(
    rows: list[dict[str, Any]],
    *,
    out_root: Path,
    max_particles: int,
    force_plot: bool,
) -> dict[str, Any]:
    baseline_paths: dict[tuple[int, int], Path] = {}
    for row in rows:
        if str(row.get("status")) in {"written", "exists"} and abs(_float_or(row.get("strength"), float("nan"))) <= 1e-12:
            baseline_paths[(_int_or(row.get("sweep_point_id", -1), -1), _int_or(row.get("rep_id", 0), 0))] = Path(str(row["output_path"]))

    cache: dict[Path, dict[str, np.ndarray]] = {}

    def load(path: Path) -> dict[str, np.ndarray]:
        if path not in cache:
            cache[path] = _load_branch_arrays(path)
        return cache[path]

    ts_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status")) not in {"written", "exists"}:
            continue
        path = Path(str(row.get("output_path", "")))
        if not path.exists():
            continue
        arrays = load(path)
        if "c_future" not in arrays:
            continue
        c = np.asarray(arrays["c_future"], dtype=np.float32)
        if c.ndim != 3 or c.shape[0] < 2:
            continue
        has_initial = _boolish(row.get("pre_perturb_initial_frame", False))
        if not has_initial:
            continue
        pre = c[0]
        baseline_c = None
        baseline_path = baseline_paths.get((_int_or(row.get("sweep_point_id", -1), -1), _int_or(row.get("rep_id", 0), 0)))
        if baseline_path is not None and baseline_path.exists():
            base_arrays = load(baseline_path)
            if "c_future" in base_arrays:
                baseline_c = np.asarray(base_arrays["c_future"], dtype=np.float32)
        sample_every = _int_or(row.get("future_sample_every_steps", 1), 1)
        for frame_idx in range(int(c.shape[0])):
            cur = c[frame_idx]
            rgb_cur = np.clip((cur[:, :3] + 1.0) * 0.5, 0.0, 1.0)
            rgb_pre = np.clip((pre[:, :3] + 1.0) * 0.5, 0.0, 1.0)
            rec = {
                "sweep_point_id": row.get("sweep_point_id", ""),
                "traj_id": row.get("traj_id", ""),
                "condition": row.get("condition", ""),
                "step": row.get("step", ""),
                "strength": row.get("strength", ""),
                "rep_id": row.get("rep_id", ""),
                "frame_idx": int(frame_idx),
                "elapsed_steps": _elapsed_steps(frame_idx, sample_every, has_initial_frame=has_initial),
                "c_chamfer_to_pre": _vector_chamfer(cur, pre, max_items=max_particles),
                "c_rgb_chamfer_to_pre": _vector_chamfer(rgb_cur, rgb_pre, max_items=max_particles),
                "c_identity_l2_to_pre": _identity_l2(cur, pre),
                "c_rgb_identity_l2_to_pre": _identity_l2(rgb_cur, rgb_pre),
                "output_path": str(path),
            }
            if baseline_c is not None and frame_idx < int(baseline_c.shape[0]):
                base = baseline_c[frame_idx]
                rgb_base = np.clip((base[:, :3] + 1.0) * 0.5, 0.0, 1.0)
                rec.update(
                    {
                        "c_chamfer_to_baseline_same_frame": _vector_chamfer(cur, base, max_items=max_particles),
                        "c_rgb_chamfer_to_baseline_same_frame": _vector_chamfer(rgb_cur, rgb_base, max_items=max_particles),
                        "c_identity_l2_to_baseline_same_frame": _identity_l2(cur, base),
                        "c_rgb_identity_l2_to_baseline_same_frame": _identity_l2(rgb_cur, rgb_base),
                    }
                )
            ts_rows.append(rec)

    groups: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for rec in ts_rows:
        point_id = _int_or(rec.get("sweep_point_id", -1), -1)
        strength = _float_or(rec.get("strength"), float("nan"))
        rep = _int_or(rec.get("rep_id", 0), 0)
        if point_id >= 0 and np.isfinite(strength):
            groups.setdefault((point_id, strength, rep), []).append(rec)
    for (point_id, strength, rep), group in sorted(groups.items()):
        group = sorted(group, key=lambda rec: _int_or(rec.get("elapsed_steps", 0), 0))
        post = [rec for rec in group if _int_or(rec.get("frame_idx", 0), 0) > 0]
        for metric_key in ("c_rgb_chamfer_to_pre", "c_chamfer_to_pre", "c_rgb_chamfer_to_baseline_same_frame"):
            vals = np.asarray([_float_or(rec.get(metric_key), float("nan")) for rec in post], dtype=np.float64)
            steps = np.asarray([_int_or(rec.get("elapsed_steps", 0), 0) for rec in post], dtype=np.int64)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            vals_f = vals[finite]
            steps_f = steps[finite]
            first = float(vals_f[0])
            final = float(vals_f[-1])
            min_idx = int(np.argmin(vals_f))
            half_step = ""
            if first > 0:
                below = np.flatnonzero(vals_f <= 0.5 * first)
                if below.size:
                    half_step = int(steps_f[int(below[0])])
            summary_rows.append(
                {
                    "sweep_point_id": int(point_id),
                    "strength": float(strength),
                    "rep_id": int(rep),
                    "metric": metric_key,
                    "first_post": first,
                    "final": final,
                    "min": float(vals_f[min_idx]),
                    "min_elapsed_steps": int(steps_f[min_idx]),
                    "final_over_first": float(final / first) if first > 0 else "",
                    "half_relax_elapsed_steps": half_step,
                    "n_post_frames": int(vals_f.size),
                }
            )

    ts_path = out_root / "plife_color_relaxation_timeseries.csv"
    summary_path = out_root / "plife_color_relaxation_summary.csv"
    write_csv(ts_path, ts_rows)
    write_csv(summary_path, summary_rows)
    plot_path = None
    if force_plot and ts_rows:
        plot_path = _write_color_relaxation_plot(
            ts_rows,
            out_root / "plife_color_relaxation.png",
            metric_key="c_rgb_chamfer_to_pre",
        )
    return {
        "color_relaxation_timeseries": str(ts_path),
        "color_relaxation_summary": str(summary_path),
        "color_relaxation_plot": plot_path or "",
        "n_color_relaxation_rows": len(ts_rows),
        "n_color_relaxation_summary_rows": len(summary_rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config, smoke=bool(args.smoke))
    output_root = _output_root(cfg)
    c2_cfg = _plife_c2_cfg(cfg)
    sweep_cfg = _sweep_cfg(c2_cfg)
    out_root = _resolve_output_root(cfg, c2_cfg, args)

    plan_path = _ensure_branch_plan(
        args.config,
        smoke=bool(args.smoke),
        cfg=cfg,
        output_root=output_root,
        force_plan=bool(args.force_plan),
    )
    plan_rows = read_csv(plan_path)
    log_event(f"PLife++ perturbation sweep loaded branch plan rows={len(plan_rows)}", component="c2-plife-sweep")
    conditions = _parse_conditions(args.conditions if args.conditions is not None else _get(sweep_cfg, "conditions", ["high"]))
    n_points = int(args.n_points if args.n_points is not None else _get(sweep_cfg, "n_points", 3))
    points = _select_points(plan_rows, conditions=conditions, n_points=n_points)
    if not points:
        raise ValueError(f"No PLife++ C2 branch points found for conditions={sorted(conditions)} in {plan_path}")

    strengths = _parse_floats(args.strengths if args.strengths is not None else _get(sweep_cfg, "strengths", [0, 0.1, 1, 3, 10]))
    if not any(abs(value) <= 1e-12 for value in strengths):
        strengths = [0.0, *strengths]
    reps_per_strength = int(args.reps_per_strength if args.reps_per_strength is not None else _get(sweep_cfg, "reps_per_strength", 1))
    horizon_steps = int(args.horizon_steps if args.horizon_steps is not None else _get(sweep_cfg, "horizon_steps", _get(c2_cfg, "horizon_steps", 1000)))
    future_sample_every = int(
        args.future_sample_every_steps
        if args.future_sample_every_steps is not None
        else _get(sweep_cfg, "future_sample_every_steps", _get(c2_cfg, "future_sample_every_steps", _get(c2_cfg, "sample_every_steps", 25)))
    )
    render_img_size = int(args.render_img_size if args.render_img_size is not None else _get(sweep_cfg, "render_img_size", _get(c2_cfg, "render_img_size", 128)))
    domain = float(args.domain_size if args.domain_size is not None else _get(sweep_cfg, "domain_size", _get(c2_cfg, "domain_size", 1.0)))
    include_initial_frame = bool(_get(sweep_cfg, "include_pre_perturb_initial_frame", True)) and not bool(args.no_initial_frame)
    video_fps = float(args.video_fps if args.video_fps is not None else _get(sweep_cfg, "video_fps", 12.0))
    codec = str(args.codec if args.codec is not None else _get(sweep_cfg, "codec", "mp4v"))
    panel_size = int(args.panel_size if args.panel_size is not None else _get(sweep_cfg, "panel_size", 192))
    max_video_frames = int(args.max_video_frames if args.max_video_frames is not None else _get(sweep_cfg, "max_video_frames", 128))
    first_frame_count = int(args.first_frame_count if args.first_frame_count is not None else _get(sweep_cfg, "first_frame_count", 4))
    radius = int(args.radius if args.radius is not None else _get(sweep_cfg, "radius", 3))
    trail_steps = int(args.trail_steps if args.trail_steps is not None else _get(sweep_cfg, "trail_steps", 6))
    debug_color_strip = bool(_get(sweep_cfg, "debug_color_strip", True)) and not bool(args.no_debug_color_strip)
    color_strip_height = int(args.color_strip_height if args.color_strip_height is not None else _get(sweep_cfg, "color_strip_height", 24))
    render_force = bool(args.force_render or args.force)
    max_particles = int(_get(c2_cfg, "divergence_max_particles", 128))
    perturb_cfg = _get(c2_cfg, "perturb", {})
    base_x_std = float(args.base_x_std if args.base_x_std is not None else _get(sweep_cfg, "base_x_std", _get(perturb_cfg, "x_std", 0.003)))
    base_v_std = float(args.base_v_std if args.base_v_std is not None else _get(sweep_cfg, "base_v_std", _get(perturb_cfg, "v_std", 0.0)))
    base_c_std = float(args.base_c_std if args.base_c_std is not None else _get(sweep_cfg, "base_c_std", _get(perturb_cfg, "c_std", 0.01)))

    log_event(
        "PLife++ perturbation sweep start "
        f"points={len(points)} conditions={sorted(conditions)} strengths={strengths} reps={reps_per_strength} "
        f"horizon={horizon_steps} sample_every={future_sample_every} render_img_size={render_img_size} output={out_root}",
        component="c2-plife-sweep",
    )
    log_event(
        f"PLife++ perturbation sweep base noise x_std={base_x_std:g} v_std={base_v_std:g} c_std={base_c_std:g} "
        f"domain={domain:g} max_particles={max_particles} include_pre_perturb_initial_frame={include_initial_frame}",
        component="c2-plife-sweep",
    )
    log_event(
        f"PLife++ perturbation sweep video debug_color_strip={debug_color_strip} color_strip_height={color_strip_height}",
        component="c2-plife-sweep",
    )
    log_event(
        f"PLife++ perturbation sweep first-frame montages frame_count={first_frame_count}",
        component="c2-plife-sweep",
    )
    for point_idx, point in enumerate(points):
        log_event(
            "PLife++ perturbation sweep selected point "
            f"{point_idx + 1}/{len(points)} source_point={point.get('point_id')} traj={point.get('traj_id')} "
            f"condition={point.get('condition')} step={point.get('step')} "
            f"delta_h={_float_or(point.get('delta_h'), float('nan')):.6g} "
            f"delta_h_energy={_float_or(point.get('delta_h_energy'), float('nan')):.6g}",
            component="c2-plife-sweep",
        )

    rows: list[dict[str, Any]] = []
    if args.allow_heavy and not args.dry_run:
        log_event("PLife++ perturbation sweep building PLife++ substrate for branch resumes", component="c2-plife-sweep")
        substrate = _build_substrate(cfg, smoke=bool(args.smoke))
    else:
        substrate = None
        log_event(
            f"PLife++ perturbation sweep not running substrate allow_heavy={bool(args.allow_heavy)} dry_run={bool(args.dry_run)}",
            component="c2-plife-sweep",
        )

    params_cache: dict[Path, np.ndarray] = {}
    written = 0
    existing = 0
    skipped = 0
    errors: list[str] = []
    total = len(points) * len(strengths) * max(1, reps_per_strength)
    for point_idx, point in enumerate(points):
        traj_tag = _safe_path_id(point.get("traj_id", f"traj_{point_idx:03d}"))
        point_dir = out_root / f"point_{point_idx:03d}_{traj_tag}_src_{_int_or(point.get('point_id', point_idx), point_idx):04d}_step_{_int_or(point.get('step', 0), 0)}"
        params_path = Path(str(point.get("params_path", "")))
        seed_x = _int_or(point.get("seed_x", -1), -1)
        log_event(
            f"PLife++ perturbation sweep point {point_idx + 1}/{len(points)} output={point_dir} "
            f"params={params_path} seed_x={seed_x}",
            component="c2-plife-sweep",
        )
        for strength_idx, strength in enumerate(strengths):
            actual_x_std = float(base_x_std * float(strength))
            actual_v_std = float(base_v_std * float(strength))
            actual_c_std = float(base_c_std * float(strength))
            log_event(
                f"PLife++ perturbation sweep point {point_idx + 1}/{len(points)} strength={float(strength):g} "
                f"noise=x:{actual_x_std:g} v:{actual_v_std:g} c:{actual_c_std:g} "
                f"reps={max(1, reps_per_strength)}",
                component="c2-plife-sweep",
            )
            strength_tag = _safe_tag(strength)
            noise_tag = _safe_path_id(f"s_{float(strength):.6g}_x_{actual_x_std:.6g}_v_{actual_v_std:.6g}_c_{actual_c_std:.6g}")
            for rep_id in range(max(1, reps_per_strength)):
                branch_seed = int(args.seed_base) + 100_003 * point_idx + 1009 * rep_id
                if args.vary_rng_with_strength:
                    branch_seed += 131 * strength_idx
                output_path = point_dir / f"strength_{noise_tag}" / f"rep_{rep_id:03d}" / "branch_output.npz"
                legacy_output_path = _legacy_strength_output_path(point_dir, float(strength), rep_id)
                video_path = output_path.with_name("video.mp4")
                row = {
                    "sweep_point_id": int(point_idx),
                    "source_point_id": point.get("point_id", ""),
                    "source_pair_id": point.get("pair_id", ""),
                    "traj_id": point.get("traj_id", ""),
                    "trial_idx": point.get("trial_idx", ""),
                    "optimized_run_idx": point.get("optimized_run_idx", ""),
                    "condition": point.get("condition", ""),
                    "window_idx": point.get("window_idx", ""),
                    "step": _int_or(point.get("step", 0), 0),
                    "requested_step": point.get("requested_step", ""),
                    "delta_h": point.get("delta_h", ""),
                    "delta_h_energy": point.get("delta_h_energy", ""),
                    "delta_h_quantile": point.get("delta_h_quantile", ""),
                    "strength": float(strength),
                    "strength_tag": strength_tag,
                    "rep_id": int(rep_id),
                    "x_std": actual_x_std,
                    "v_std": actual_v_std,
                    "c_std": actual_c_std,
                    "seed_x": int(seed_x),
                    "branch_seed": int(branch_seed),
                    "horizon_steps": int(horizon_steps),
                    "future_sample_every_steps": int(future_sample_every),
                    "pre_perturb_initial_frame": bool(include_initial_frame),
                    "params_path": str(params_path),
                    "metrics_path": point.get("metrics_path", ""),
                    "output_path": str(output_path),
                    "legacy_output_path": str(legacy_output_path),
                    "video_path": str(video_path),
                    "status": "planned",
                    "command": command_to_str(
                        [
                            "internal_plife_perturbation_sweep",
                            str(params_path),
                            "--step",
                            str(_int_or(point.get("step", 0), 0)),
                            "--horizon",
                            str(horizon_steps),
                            "--strength",
                            str(float(strength)),
                            "--branch-seed",
                            str(branch_seed),
                        ]
                    ),
                }
                if not output_path.exists() and legacy_output_path.exists() and not args.force:
                    output_path = legacy_output_path
                    video_path = output_path.with_name("video.mp4")
                    row["output_path"] = str(output_path)
                    row["video_path"] = str(video_path)
                    row["used_legacy_output_path"] = True
                if output_path.exists() and not args.force:
                    row["status"] = "exists"
                    existing += 1
                    log_event(
                        f"PLife++ perturbation sweep exists point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id} output={output_path} legacy={bool(row.get('used_legacy_output_path', False))}",
                        component="c2-plife-sweep",
                    )
                    rows.append(row)
                    continue
                if not args.allow_heavy:
                    row["status"] = "skipped_heavy"
                    skipped += 1
                    log_event(
                        f"PLife++ perturbation sweep planned/skipped-heavy point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id} output={output_path}",
                        component="c2-plife-sweep",
                    )
                    rows.append(row)
                    continue
                if args.dry_run:
                    row["status"] = "dry_run"
                    log_event(
                        f"PLife++ perturbation sweep dry-run point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id} command={row['command']}",
                        component="c2-plife-sweep",
                    )
                    rows.append(row)
                    continue
                if seed_x < 0 or not params_path.exists():
                    row["status"] = "skipped_missing_params"
                    row["message"] = f"seed_x={seed_x} params_exists={params_path.exists()}"
                    skipped += 1
                    errors.append(f"point={point_idx} missing seed/params")
                    log_event(
                        f"PLife++ perturbation sweep skipped missing params point={point_idx} seed={seed_x} params={params_path}",
                        component="c2-plife-sweep",
                    )
                    rows.append(row)
                    continue
                try:
                    if params_path not in params_cache:
                        params_cache[params_path] = np.load(params_path, allow_pickle=True)
                    if written == 0 or (written + existing + skipped) % 10 == 0:
                        log_event(
                            f"PLife++ perturbation sweep branch {len(rows) + 1}/{total} "
                            f"point={point_idx} strength={strength:g} rep={rep_id} step={row['step']}",
                            component="c2-plife-sweep",
                        )
                    payload = _simulate_one_branch(
                        substrate=substrate,
                        params=params_cache[params_path],
                        seed=seed_x,
                        branch_step=int(row["step"]),
                        rep_seed=branch_seed,
                        horizon_steps=horizon_steps,
                        sample_every=future_sample_every,
                        perturb={"x_std": row["x_std"], "v_std": row["v_std"], "c_std": row["c_std"]},
                        render_img_size=render_img_size,
                        include_initial_frame=include_initial_frame,
                    )
                    ensure_dir(output_path.parent)
                    np.savez_compressed(
                        output_path,
                        **{key: np.asarray(value, dtype=np.float32) for key, value in payload.items()},
                        sweep_point_id=np.asarray(point_idx, dtype=np.int32),
                        source_point_id=np.asarray(_int_or(point.get("point_id", -1), -1), dtype=np.int32),
                        rep_id=np.asarray(rep_id, dtype=np.int32),
                        strength=np.asarray(float(strength), dtype=np.float32),
                        x_std=np.asarray(row["x_std"], dtype=np.float32),
                        v_std=np.asarray(row["v_std"], dtype=np.float32),
                        c_std=np.asarray(row["c_std"], dtype=np.float32),
                        branch_seed=np.asarray(branch_seed, dtype=np.int64),
                        step=np.asarray(int(row["step"]), dtype=np.int32),
                        horizon_steps=np.asarray(horizon_steps, dtype=np.int32),
                        future_sample_every_steps=np.asarray(future_sample_every, dtype=np.int32),
                        pre_perturb_initial_frame=np.asarray(bool(include_initial_frame)),
                    )
                    row["status"] = "written"
                    written += 1
                    log_event(
                        f"PLife++ perturbation sweep wrote branch point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id} output={output_path}",
                        component="c2-plife-sweep",
                    )
                except Exception as exc:
                    row["status"] = "error"
                    row["message"] = f"{type(exc).__name__}: {exc}"
                    errors.append(f"point={point_idx} strength={strength:g} rep={rep_id}: {type(exc).__name__}: {exc}")
                    log_event(
                        f"PLife++ perturbation sweep error point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id}: {type(exc).__name__}: {exc}",
                        component="c2-plife-sweep",
                    )
                rows.append(row)

    log_event("PLife++ perturbation sweep computing divergence against strength=0 baselines", component="c2-plife-sweep")
    _compute_divergences(rows, domain=domain, max_particles=max_particles)
    div_summary = _divergence_summary(rows)
    color_summary = _color_debug_summary(rows)
    log_event(f"PLife++ perturbation sweep divergence summary {div_summary}", component="c2-plife-sweep")
    log_event(f"PLife++ perturbation sweep color debug summary {color_summary}", component="c2-plife-sweep")
    for row in rows:
        if str(row.get("status")) not in {"written", "exists"}:
            continue
        log_event(
            "PLife++ perturbation sweep color debug "
            f"point={row.get('sweep_point_id')} strength={_float_or(row.get('strength'), float('nan')):g} "
            f"status={row.get('status')} planned_c_std={_float_or(row.get('c_std'), float('nan')):g} "
            f"saved_c_std={_float_or(row.get('saved_c_std'), float('nan')):g} "
            f"c3_abs={_float_or(row.get('c_first3_mean_abs_vs_baseline'), float('nan')):.6g} "
            f"c3_range=[{_float_or(row.get('c_first3_min'), float('nan')):.6g},{_float_or(row.get('c_first3_max'), float('nan')):.6g}] "
            f"c3_std={_float_or(row.get('c_first3_std'), float('nan')):.6g} "
            f"rgb_abs={_float_or(row.get('rgb_mean_abs_vs_baseline'), float('nan')):.6g} "
            f"rgb_max={_float_or(row.get('rgb_max_abs_vs_baseline'), float('nan')):.6g}",
            component="c2-plife-sweep",
        )
    log_event("PLife++ perturbation sweep writing color relaxation timeseries", component="c2-plife-sweep")
    color_relaxation_summary = _write_color_relaxation_outputs(
        rows,
        out_root=out_root,
        max_particles=max_particles,
        force_plot=not bool(args.skip_color_relaxation_plot),
    )
    log_event(f"PLife++ perturbation sweep color relaxation outputs {color_relaxation_summary}", component="c2-plife-sweep")

    video_rows = 0
    if not args.dry_run and not args.skip_videos:
        log_event("PLife++ perturbation sweep rendering individual videos", component="c2-plife-sweep")
        for row in rows:
            if str(row.get("status")) not in {"written", "exists"}:
                continue
            output_path = Path(str(row["output_path"]))
            if not output_path.exists():
                continue
            arrays = _load_branch_arrays(output_path)
            try:
                if debug_color_strip:
                    frames = _load_video_frames(
                        output_path,
                        panel_size=panel_size,
                        max_frames=max_video_frames,
                        radius=radius,
                        trail_steps=trail_steps,
                        domain_size=float(domain),
                        debug_color_strip=debug_color_strip,
                        color_strip_height=color_strip_height,
                    )
                    result = _write_frame_list_video(
                        frames,
                        Path(str(row["video_path"])),
                        fps=video_fps,
                        codec=codec,
                        force=render_force,
                    )
                elif "rgb_future" in arrays:
                    result = _write_rgb_video(
                        arrays["rgb_future"],
                        Path(str(row["video_path"])),
                        fps=video_fps,
                        codec=codec,
                        macro_block_size=1,
                        max_frames=max_video_frames,
                        force=render_force,
                    )
                elif "xy_future" in arrays:
                    result = _write_xy_video(
                        arrays["xy_future"],
                        Path(str(row["video_path"])),
                        fps=video_fps,
                        codec=codec,
                        macro_block_size=1,
                        img_size=panel_size,
                        radius=radius,
                        trail_steps=trail_steps,
                        max_frames=max_video_frames,
                        domain_size=float(domain),
                        wrap=True,
                        force=render_force,
                    )
                else:
                    result = {"status": "skipped_missing_frames", "video_path": row["video_path"], "n_frames": 0}
                row["video_status"] = result.get("status", "")
                row["video_n_frames"] = result.get("n_frames", "")
                if result.get("status") in {"written", "exists"}:
                    video_rows += 1
                log_event(
                    f"PLife++ perturbation sweep video {row['video_status']} point={row['sweep_point_id']} "
                    f"strength={float(row['strength']):g} rep={row['rep_id']} path={row['video_path']}",
                    component="c2-plife-sweep",
                )
            except Exception as exc:
                row["video_status"] = "error"
                row["video_message"] = f"{type(exc).__name__}: {exc}"
                log_event(
                    f"PLife++ perturbation sweep video error point={row['sweep_point_id']} "
                    f"strength={float(row['strength']):g} rep={row['rep_id']}: {type(exc).__name__}: {exc}",
                    component="c2-plife-sweep",
                )
    else:
        log_event(
            f"PLife++ perturbation sweep skipping individual videos dry_run={bool(args.dry_run)} skip_videos={bool(args.skip_videos)}",
            component="c2-plife-sweep",
        )

    grid_videos: list[str] = []
    if not args.dry_run and not args.skip_grid_videos:
        log_event("PLife++ perturbation sweep rendering per-point strength grid videos", component="c2-plife-sweep")
        grid_cols = int(args.grid_cols if args.grid_cols is not None else _get(sweep_cfg, "grid_cols", len(strengths)))
        for point_idx in range(len(points)):
            items_with_strength: list[tuple[float, Path, str]] = []
            for row in rows:
                if int(row["sweep_point_id"]) != point_idx or str(row.get("status")) not in {"written", "exists"}:
                    continue
                path = Path(str(row["output_path"]))
                if path.exists():
                    items_with_strength.append((float(row["strength"]), path, f"s={float(row['strength']):g} r={int(row['rep_id'])}"))
            if not items_with_strength:
                continue
            items = [(path, label) for _strength, path, label in sorted(items_with_strength, key=lambda item: (item[0], str(item[1])))]
            grid_path = out_root / f"point_{point_idx:03d}_strength_grid.mp4"
            try:
                result = _write_grid_video(
                    output=grid_path,
                    items=items,
                    fps=video_fps,
                    codec=codec,
                    panel_size=panel_size,
                    max_frames=max_video_frames,
                    grid_cols=grid_cols,
                    radius=radius,
                    trail_steps=trail_steps,
                    domain_size=float(domain),
                    debug_color_strip=debug_color_strip,
                    color_strip_height=color_strip_height,
                    force=render_force,
                )
                if result.get("status") in {"written", "exists"}:
                    grid_videos.append(str(grid_path))
                log_event(
                    f"PLife++ perturbation sweep grid video {result.get('status')} point={point_idx} path={grid_path}",
                    component="c2-plife-sweep",
                )
            except Exception as exc:
                errors.append(f"grid point={point_idx}: {type(exc).__name__}: {exc}")
                log_event(
                    f"PLife++ perturbation sweep grid video error point={point_idx}: {type(exc).__name__}: {exc}",
                    component="c2-plife-sweep",
                )
    else:
        log_event(
            f"PLife++ perturbation sweep skipping grid videos dry_run={bool(args.dry_run)} skip_grid_videos={bool(args.skip_grid_videos)}",
            component="c2-plife-sweep",
        )

    first_frame_montages: list[str] = []
    if not args.dry_run and not args.skip_first_frame_montages:
        log_event("PLife++ perturbation sweep rendering first-frame perturbation montages", component="c2-plife-sweep")
        for point_idx in range(len(points)):
            items_with_strength: list[tuple[float, int, Path]] = []
            for row in rows:
                if int(row["sweep_point_id"]) != point_idx or str(row.get("status")) not in {"written", "exists"}:
                    continue
                path = Path(str(row["output_path"]))
                if path.exists():
                    items_with_strength.append((float(row["strength"]), int(row["rep_id"]), path))
            if not items_with_strength:
                continue
            montage_path = out_root / f"point_{point_idx:03d}_first_frames.png"
            try:
                result = _write_first_frame_montage(
                    output=montage_path,
                    items=items_with_strength,
                    frame_count=first_frame_count,
                    panel_size=panel_size,
                    radius=radius,
                    trail_steps=trail_steps,
                    domain_size=float(domain),
                    sample_every=future_sample_every,
                    has_initial_frame=include_initial_frame,
                    debug_color_strip=debug_color_strip,
                    color_strip_height=color_strip_height,
                    force=render_force,
                )
                if result.get("status") in {"written", "exists"}:
                    first_frame_montages.append(str(montage_path))
                log_event(
                    f"PLife++ perturbation sweep first-frame montage {result.get('status')} "
                    f"point={point_idx} tiles={result.get('n_tiles')} path={montage_path}",
                    component="c2-plife-sweep",
                )
            except Exception as exc:
                errors.append(f"first-frame montage point={point_idx}: {type(exc).__name__}: {exc}")
                log_event(
                    f"PLife++ perturbation sweep first-frame montage error point={point_idx}: {type(exc).__name__}: {exc}",
                    component="c2-plife-sweep",
                )
    else:
        log_event(
            "PLife++ perturbation sweep skipping first-frame montages "
            f"dry_run={bool(args.dry_run)} skip_first_frame_montages={bool(args.skip_first_frame_montages)}",
            component="c2-plife-sweep",
        )

    plan_out = out_root / "plife_perturbation_strength_sweep_plan.csv"
    write_csv(plan_out, rows)
    summary = {
        "status": "ok",
        "allow_heavy": bool(args.allow_heavy),
        "dry_run": bool(args.dry_run),
        "force_render": bool(args.force_render),
        "branch_plan": str(plan_path),
        "output_root": str(out_root),
        "plan": str(plan_out),
        "n_points": len(points),
        "n_rows": len(rows),
        "n_written": int(written),
        "n_existing": int(existing),
        "n_skipped": int(skipped),
        "n_videos": int(video_rows),
        "grid_videos": grid_videos,
        "first_frame_montages": first_frame_montages,
        "n_first_frame_montages": int(len(first_frame_montages)),
        "conditions": sorted(conditions),
        "strengths": strengths,
        "reps_per_strength": int(reps_per_strength),
        "horizon_steps": int(horizon_steps),
        "future_sample_every_steps": int(future_sample_every),
        "include_pre_perturb_initial_frame": bool(include_initial_frame),
        "base_x_std": float(base_x_std),
        "base_v_std": float(base_v_std),
        "base_c_std": float(base_c_std),
        "video_fps": float(video_fps),
        "panel_size": int(panel_size),
        "max_video_frames": int(max_video_frames),
        "first_frame_count": int(first_frame_count),
        "debug_color_strip": bool(debug_color_strip),
        "color_strip_height": int(color_strip_height),
        **div_summary,
        **color_summary,
        **color_relaxation_summary,
        "errors": errors[:20],
    }
    write_json(out_root / "plife_perturbation_strength_sweep_summary.json", summary)
    log_event(
        f"PLife++ perturbation sweep done rows={len(rows)} written={written} existing={existing} "
        f"videos={video_rows} grids={len(grid_videos)} first_frame_montages={len(first_frame_montages)} "
        f"summary={out_root / 'plife_perturbation_strength_sweep_summary.json'}",
        component="c2-plife-sweep",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLife++ C2 high-Delta-H perturbation strength sweep videos.")
    parser.add_argument("config")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--conditions", default=None, help="Comma-separated branch-plan conditions to sweep, default from config or high.")
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--strengths", default=None, help="Comma-separated strength multipliers; 0 baseline is added if absent.")
    parser.add_argument("--reps-per-strength", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--future-sample-every-steps", type=int, default=None)
    parser.add_argument("--render-img-size", type=int, default=None)
    parser.add_argument("--domain-size", type=float, default=None)
    parser.add_argument("--base-x-std", type=float, default=None)
    parser.add_argument("--base-v-std", type=float, default=None)
    parser.add_argument("--base-c-std", type=float, default=None)
    parser.add_argument("--no-initial-frame", action="store_true", help="Do not prepend the shared pre-perturb branch frame to sweep videos.")
    parser.add_argument("--seed-base", type=int, default=9_300_003)
    parser.add_argument("--vary-rng-with-strength", action="store_true")
    parser.add_argument("--force-plan", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-render", action="store_true", help="Overwrite rendered PNG/MP4 files without forcing branch resimulation.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-grid-videos", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--panel-size", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--first-frame-count", type=int, default=None, help="Number of initial saved frames to show in perturbation montage PNGs.")
    parser.add_argument("--skip-first-frame-montages", action="store_true")
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--trail-steps", type=int, default=None)
    parser.add_argument("--no-debug-color-strip", action="store_true", help="Do not append c[:, :3] particle color strips to rendered sweep videos.")
    parser.add_argument("--color-strip-height", type=int, default=None)
    parser.add_argument("--skip-color-relaxation-plot", action="store_true")
    args = parser.parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
