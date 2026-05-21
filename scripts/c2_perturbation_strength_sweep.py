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

from paper_suite_c2_branching import (
    _branch_cfg,
    _get,
    _iter_metric_items,
    _load_delta_h_energy,
    _make_resume_command,
    _nearest_apf_step,
    _trajectory_end_step,
    _trajectory_root,
)
from paper_suite_common import command_to_str, current_python, ensure_dir, load_config, run_subprocess, write_csv, write_json


def _parse_floats(raw: str) -> list[float]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("Empty strength list.")
    return out


def _safe_tag(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", text)


def _chunk_sort_key(path: Path) -> tuple[int, int, int, str]:
    match = re.search(r"P_steps_(\d+)_(\d+).*idx_(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), path.name
    return 10**18, 10**18, 10**18, path.name


def _render_pcolor(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    pp = np.asarray(p, dtype=np.float32)
    if pp.shape[-1] < 3:
        reps = int(math.ceil(3.0 / max(1, int(pp.shape[-1]))))
        p3 = np.tile(pp, reps)[..., :3]
    else:
        p3 = pp[..., :3]
    mass = np.sum(aa, axis=-1, keepdims=True) if aa.ndim == 3 else aa[..., None]
    return (np.clip(mass * p3, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _resize_nearest(frame: np.ndarray, size: int) -> np.ndarray:
    if int(size) <= 0:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h == int(size) and w == int(size):
        return frame
    yy = np.linspace(0, h - 1, int(size)).astype(np.int64)
    xx = np.linspace(0, w - 1, int(size)).astype(np.int64)
    return frame[yy][:, xx]


def _load_branch_frames(branch_dir: Path, *, panel_size: int, max_frames: int) -> list[np.ndarray]:
    apf_dir = branch_dir / "apf_logs"
    chunks = sorted(apf_dir.glob("P_steps_*.npz"), key=_chunk_sort_key)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    frames: list[np.ndarray] = []
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as data:
            a = np.asarray(data["A"])
            p = np.asarray(data["P"])
            for i in range(int(a.shape[0])):
                frames.append(_resize_nearest(_render_pcolor(a[i], p[i]), panel_size))
    if max_frames > 0 and len(frames) > int(max_frames):
        idx = np.linspace(0, len(frames) - 1, int(max_frames)).astype(np.int64)
        frames = [frames[int(i)] for i in idx]
    return frames


def _make_grid_frame(frames: list[np.ndarray], *, n_cols: int) -> np.ndarray:
    if not frames:
        raise ValueError("No frames for grid.")
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(len(frames) / n_cols))
    h, w, c = frames[0].shape
    canvas = np.zeros((n_rows * h, n_cols * w, c), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // n_cols
        col = idx % n_cols
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame
    return canvas


def _write_strength_grid_video(
    *,
    output_path: Path,
    branch_dirs: list[Path],
    fps: float,
    panel_size: int,
    max_frames: int,
) -> None:
    import imageio.v2 as imageio

    series = [_load_branch_frames(path, panel_size=panel_size, max_frames=max_frames) for path in branch_dirs]
    n_frames = max(len(frames) for frames in series)
    n_cols = int(math.ceil(math.sqrt(len(series))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=float(fps), codec="libx264", macro_block_size=1)
    try:
        for frame_idx in range(n_frames):
            row_frames = [frames[min(frame_idx, len(frames) - 1)] for frames in series]
            writer.append_data(_make_grid_frame(row_frames, n_cols=n_cols))
    finally:
        writer.close()


def _select_high_points(
    *,
    items: list[dict[str, Any]],
    max_trajectories: int,
    n_points: int,
    q_high: float,
    horizon_steps: int,
    min_branch_step: int,
    energy_min_remaining_steps: int | None,
    energy_min_samples: int | None,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]] = []
    for item in items:
        centers, energy, meta = _load_delta_h_energy(
            Path(item["metrics_path"]),
            min_remaining_steps=energy_min_remaining_steps,
            min_remaining_samples=energy_min_samples,
        )
        ranked.append((float(np.nanmean(energy)), item, centers, energy, meta))
    ranked.sort(key=lambda row: -row[0])

    candidates: list[dict[str, Any]] = []
    for traj_order, (_score, item, centers, energy, meta) in enumerate(ranked[: int(max_trajectories)]):
        traj_dir = Path(item["traj_dir"])
        apf_dir = Path(item.get("apf_dir", traj_dir / "apf_logs"))
        try:
            trajectory_end = _trajectory_end_step(apf_dir)
        except Exception:
            trajectory_end = int(np.nanmax(centers) + int(horizon_steps)) if centers.size else None
        c = np.asarray(centers, dtype=np.float64).reshape(-1)
        e = np.asarray(energy, dtype=np.float64).reshape(-1)
        n = min(c.size, e.size)
        c = c[:n]
        e = e[:n]
        finite = np.isfinite(c) & np.isfinite(e)
        if min_branch_step > 0:
            finite &= c >= float(min_branch_step)
        if trajectory_end is not None:
            finite &= (c + float(horizon_steps)) <= float(trajectory_end)
        if not np.any(finite):
            continue
        threshold = float(np.nanquantile(e[finite], float(q_high)))
        pool = np.flatnonzero(finite & (e >= threshold))
        for idx in pool.tolist():
            requested_step = int(round(float(c[idx])))
            try:
                snapped_step = _nearest_apf_step(apf_dir, requested_step)
            except Exception:
                snapped_step = requested_step
            candidates.append(
                {
                    "traj_order": int(traj_order),
                    "traj_id": str(item["traj_id"]),
                    "traj_dir": str(traj_dir),
                    "apf_dir": str(apf_dir),
                    "metrics_path": str(item["metrics_path"]),
                    "window_index": int(idx),
                    "window_center_step": requested_step,
                    "step": int(snapped_step),
                    "step_snap_delta": int(snapped_step - requested_step),
                    "delta_h_energy": float(e[idx]),
                    "high_threshold": threshold,
                    "admissible_tau_count": int(meta.get("admissible_tau_count", 0)),
                    "admissible_tau_steps": str(meta.get("admissible_tau_steps", "")),
                }
            )
    candidates.sort(key=lambda row: -float(row["delta_h_energy"]))
    return candidates[: int(n_points)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config, smoke=False)
    bcfg = _branch_cfg(cfg)
    c2_cfg = cfg.get("c2", {})
    trajectory_root = _trajectory_root(c2_cfg)
    if trajectory_root is None or not trajectory_root.exists():
        raise FileNotFoundError(f"Missing C2 trajectory root: {trajectory_root}")
    items = _iter_metric_items(trajectory_root)
    if not items:
        raise ValueError(f"No optimized metrics.npz items found under {trajectory_root}.")

    strengths = _parse_floats(args.strengths)
    max_trajectories = int(args.max_trajectories if args.max_trajectories is not None else _get(bcfg, "max_trajectories", 2))
    n_points = int(args.n_points)
    q_high = float(args.q_high if args.q_high is not None else _get(bcfg, "high_quantile", 0.8))
    horizon_steps = int(args.horizon_steps if args.horizon_steps is not None else _get(bcfg, "horizon_steps", 1000))
    min_branch_step = int(args.min_branch_step if args.min_branch_step is not None else _get(bcfg, "min_branch_step", _get(bcfg, "selection_min_step", 0)))
    perturb = _get(bcfg, "perturb", {})
    base_a_std = float(args.base_a_std if args.base_a_std is not None else _get(perturb, "a_std", 1e-4))
    base_p_std = float(args.base_p_std if args.base_p_std is not None else _get(perturb, "p_std", 1e-4))
    base_lag_xy_std = float(args.base_lag_xy_std if args.base_lag_xy_std is not None else _get(perturb, "lagrangian_xy_std", 0.01))
    energy_min_remaining_steps_raw = _get(bcfg, "energy_min_remaining_steps", None)
    energy_min_samples_raw = _get(bcfg, "energy_min_samples", None)
    energy_min_remaining_steps = None if energy_min_remaining_steps_raw is None else int(energy_min_remaining_steps_raw)
    energy_min_samples = None if energy_min_samples_raw is None else int(energy_min_samples_raw)

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root
    ensure_dir(out_root)

    points = _select_high_points(
        items=items,
        max_trajectories=max_trajectories,
        n_points=n_points,
        q_high=q_high,
        horizon_steps=horizon_steps,
        min_branch_step=min_branch_step,
        energy_min_remaining_steps=energy_min_remaining_steps,
        energy_min_samples=energy_min_samples,
    )
    if not points:
        raise ValueError("No high Delta-H points selected.")

    rows: list[dict[str, Any]] = []
    by_strength: dict[str, list[Path]] = {}
    for point_id, point in enumerate(points):
        source_traj_dir = Path(str(point["traj_dir"]))
        if args.allow_heavy and not (source_traj_dir / "apf_logs").exists():
            raise FileNotFoundError(f"Cannot resume without APF logs: {source_traj_dir / 'apf_logs'}")
        branch_seed_base = int(args.seed_base) + 1009 * point_id
        for strength_idx, strength in enumerate(strengths):
            tag = _safe_tag(strength)
            branch_seed = branch_seed_base + (131 * strength_idx if args.vary_rng_with_strength else 0)
            out_dir = out_root / f"point_{point_id:03d}_{point['traj_id']}_w_{point['window_index']:04d}_step_{point['step']}" / f"strength_{tag}"
            a_std = base_a_std * float(strength)
            p_std = base_p_std * float(strength)
            lag_std = base_lag_xy_std * float(strength)
            cmd = _make_resume_command(
                source_traj_dir=source_traj_dir,
                step=int(point["step"]),
                horizon_steps=horizon_steps,
                branch_dir=out_dir,
                branch_seed=branch_seed,
                perturb_a_std=a_std,
                perturb_p_std=p_std,
                perturb_lag_xy_std=lag_std,
                force=bool(args.force),
            )
            status = "planned"
            if args.allow_heavy:
                run_subprocess(cmd, dry_run=bool(args.dry_run))
                status = "dry_run" if args.dry_run else "written"
            row = {
                **point,
                "point_id": int(point_id),
                "strength": float(strength),
                "strength_tag": tag,
                "a_std": float(a_std),
                "p_std": float(p_std),
                "lagrangian_xy_std": float(lag_std),
                "branch_seed": int(branch_seed),
                "output_dir": str(out_dir),
                "video_path": str(out_dir / "video.mp4"),
                "status": status,
                "command": command_to_str(cmd),
            }
            rows.append(row)
            by_strength.setdefault(tag, []).append(out_dir)

    plan_path = out_root / "perturbation_strength_sweep_plan.csv"
    write_csv(plan_path, rows)

    grid_videos: list[str] = []
    if args.allow_heavy and not args.dry_run and not args.skip_grid_videos:
        for tag, dirs in by_strength.items():
            existing = [path for path in dirs if (path / "apf_logs").exists()]
            if not existing:
                continue
            grid_path = out_root / f"strength_{tag}_grid.mp4"
            _write_strength_grid_video(
                output_path=grid_path,
                branch_dirs=existing,
                fps=float(args.video_fps),
                panel_size=int(args.panel_size),
                max_frames=int(args.max_video_frames),
            )
            grid_videos.append(str(grid_path))

    summary = {
        "status": "ok",
        "allow_heavy": bool(args.allow_heavy),
        "dry_run": bool(args.dry_run),
        "n_points": len(points),
        "strengths": strengths,
        "horizon_steps": horizon_steps,
        "min_branch_step": min_branch_step,
        "base_a_std": base_a_std,
        "base_p_std": base_p_std,
        "base_lagrangian_xy_std": base_lag_xy_std,
        "plan": str(plan_path),
        "grid_videos": grid_videos,
    }
    write_json(out_root / "perturbation_strength_sweep_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run C2 high-DeltaH perturbation strength sweep videos.")
    parser.add_argument("config")
    parser.add_argument("--output-root", default="analysis/results/paper_suite/c2_perturbation_strength_sweep")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=3)
    parser.add_argument("--q-high", type=float, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--min-branch-step", type=int, default=None)
    parser.add_argument("--strengths", default="0,0.1,1,3,10")
    parser.add_argument("--base-a-std", type=float, default=None)
    parser.add_argument("--base-p-std", type=float, default=None)
    parser.add_argument("--base-lag-xy-std", type=float, default=None)
    parser.add_argument("--seed-base", type=int, default=9100003)
    parser.add_argument("--vary-rng-with-strength", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-grid-videos", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--max-video-frames", type=int, default=256)
    args = parser.parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
