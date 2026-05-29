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
        return {key: np.asarray(data[key]) for key in data.files if key in {"xy_future", "rgb_future", "c_future"}}


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


def _load_video_frames(
    path: Path,
    *,
    panel_size: int,
    max_frames: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
) -> list[np.ndarray]:
    arrays = _load_branch_arrays(path)
    if "rgb_future" in arrays:
        frames = _rgb_uint8(arrays["rgb_future"])
        idx = _frame_indices(frames.shape[0], max_frames)
        return [_resize_nearest(frames[int(i)], panel_size) for i in idx]
    if "xy_future" in arrays:
        xy = np.asarray(arrays["xy_future"], dtype=np.float32)
        idx = _frame_indices(xy.shape[0], max_frames)
        return [
            _xy_frame(
                xy,
                int(i),
                img_size=int(panel_size),
                radius=int(radius),
                trail_steps=int(trail_steps),
                domain_size=float(domain_size),
                wrap=True,
            )
            for i in idx
        ]
    raise ValueError(f"No rgb_future or xy_future in {path}")


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
                row["rgb_mse_mean_vs_baseline"] = float(np.mean((a[:n] - b[:n]) ** 2)) if n > 0 else float("nan")
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
    radius = int(args.radius if args.radius is not None else _get(sweep_cfg, "radius", 3))
    trail_steps = int(args.trail_steps if args.trail_steps is not None else _get(sweep_cfg, "trail_steps", 6))
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
            log_event(
                f"PLife++ perturbation sweep point {point_idx + 1}/{len(points)} strength={float(strength):g} "
                f"noise=x:{base_x_std * float(strength):g} v:{base_v_std * float(strength):g} c:{base_c_std * float(strength):g} "
                f"reps={max(1, reps_per_strength)}",
                component="c2-plife-sweep",
            )
            strength_tag = _safe_tag(strength)
            for rep_id in range(max(1, reps_per_strength)):
                branch_seed = int(args.seed_base) + 100_003 * point_idx + 1009 * rep_id
                if args.vary_rng_with_strength:
                    branch_seed += 131 * strength_idx
                output_path = point_dir / f"strength_{strength_tag}" / f"rep_{rep_id:03d}" / "branch_output.npz"
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
                    "x_std": float(base_x_std * float(strength)),
                    "v_std": float(base_v_std * float(strength)),
                    "c_std": float(base_c_std * float(strength)),
                    "seed_x": int(seed_x),
                    "branch_seed": int(branch_seed),
                    "horizon_steps": int(horizon_steps),
                    "future_sample_every_steps": int(future_sample_every),
                    "pre_perturb_initial_frame": bool(include_initial_frame),
                    "params_path": str(params_path),
                    "metrics_path": point.get("metrics_path", ""),
                    "output_path": str(output_path),
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
                if output_path.exists() and not args.force:
                    row["status"] = "exists"
                    existing += 1
                    log_event(
                        f"PLife++ perturbation sweep exists point={point_idx} strength={float(strength):g} "
                        f"rep={rep_id} output={output_path}",
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
    log_event(f"PLife++ perturbation sweep divergence summary {div_summary}", component="c2-plife-sweep")

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
                if "rgb_future" in arrays:
                    result = _write_rgb_video(
                        arrays["rgb_future"],
                        Path(str(row["video_path"])),
                        fps=video_fps,
                        codec=codec,
                        macro_block_size=1,
                        max_frames=max_video_frames,
                        force=bool(args.force),
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
                        force=bool(args.force),
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
                    force=bool(args.force),
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

    plan_out = out_root / "plife_perturbation_strength_sweep_plan.csv"
    write_csv(plan_out, rows)
    summary = {
        "status": "ok",
        "allow_heavy": bool(args.allow_heavy),
        "dry_run": bool(args.dry_run),
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
        **div_summary,
        "errors": errors[:20],
    }
    write_json(out_root / "plife_perturbation_strength_sweep_summary.json", summary)
    log_event(
        f"PLife++ perturbation sweep done rows={len(rows)} written={written} existing={existing} "
        f"videos={video_rows} grids={len(grid_videos)} summary={out_root / 'plife_perturbation_strength_sweep_summary.json'}",
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-grid-videos", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--panel-size", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--trail-steps", type=int, default=None)
    args = parser.parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
