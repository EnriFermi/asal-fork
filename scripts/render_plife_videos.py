from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from paper_suite_common import ensure_dir, load_config, log_event, read_csv, resolve_path, write_csv, write_json


XY_KEYS_DEFAULT = ("xy_trajectory", "xy_control_a", "xy_control_b", "xy_walls")


def _open_video_writer(output: Path, *, fps: float, codec: str, frame_shape: tuple[int, int, int]):
    import cv2  # type: ignore

    h, w = int(frame_shape[0]), int(frame_shape[1])
    fourcc = cv2.VideoWriter_fourcc(*str(codec)[:4].ljust(4, " "))
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    return writer, cv2


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _output_root(cfg: Any) -> Path:
    return ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))


def _resolve(raw: Any) -> Path | None:
    return resolve_path(raw) if raw is not None and str(raw) else None


def _path_from_root(root: Path, raw: Any, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _slug(text: Any) -> str:
    value = str(text)
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return out.strip("_") or "item"


def _palette(n: int) -> np.ndarray:
    base = np.asarray(
        [
            [31, 119, 180],
            [214, 39, 40],
            [44, 160, 44],
            [255, 127, 14],
            [148, 103, 189],
            [140, 86, 75],
            [23, 190, 207],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
        ],
        dtype=np.float32,
    )
    if n <= base.shape[0]:
        return base[:n]
    reps = int(math.ceil(float(n) / float(base.shape[0])))
    return np.tile(base, (reps, 1))[:n]


def _frame_indices(n_frames: int, max_frames: int | None) -> np.ndarray:
    n = int(n_frames)
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    if max_frames is None or int(max_frames) <= 0 or n <= int(max_frames):
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, int(max_frames)).round().astype(np.int64)
    return np.unique(idx)


def _draw_disk(img: np.ndarray, row: float, col: float, radius: int, color: np.ndarray, alpha: float) -> None:
    h, w = img.shape[:2]
    r = max(1, int(radius))
    rr0 = max(0, int(math.floor(row)) - r)
    rr1 = min(h, int(math.floor(row)) + r + 1)
    cc0 = max(0, int(math.floor(col)) - r)
    cc1 = min(w, int(math.floor(col)) + r + 1)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    yy, xx = np.ogrid[rr0:rr1, cc0:cc1]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= float(r * r)
    if not np.any(mask):
        return
    patch = img[rr0:rr1, cc0:cc1]
    color_u8 = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    a = float(np.clip(alpha, 0.0, 1.0))
    patch[mask] = np.clip((1.0 - a) * patch[mask].astype(np.float32) + a * color_u8, 0.0, 255.0).astype(np.uint8)


def _xy_frame(
    xy_seq: np.ndarray,
    frame_idx: int,
    *,
    img_size: int,
    radius: int,
    trail_steps: int,
    domain_size: float,
    wrap: bool,
) -> np.ndarray:
    xy = np.asarray(xy_seq, dtype=np.float32)
    n_particles = int(xy.shape[1])
    colors = _palette(n_particles)
    img = np.full((int(img_size), int(img_size), 3), 255, dtype=np.uint8)
    t0 = max(0, int(frame_idx) - max(0, int(trail_steps)))
    denom = max(1, int(frame_idx) - t0)
    size_minus = max(1, int(img_size) - 1)
    domain = float(domain_size) if float(domain_size) > 0 else 1.0
    for t in range(t0, int(frame_idx) + 1):
        alpha = 0.16 + 0.84 * float(t - t0) / float(denom)
        rad = int(radius) if t == int(frame_idx) else max(1, int(radius) - 1)
        pts = np.asarray(xy[t], dtype=np.float32)
        if wrap:
            pts = np.mod(pts, domain)
        else:
            pts = np.clip(pts, 0.0, domain)
        pts = pts / domain
        finite = np.isfinite(pts).all(axis=1)
        for i in np.flatnonzero(finite):
            col = float(pts[i, 0]) * size_minus
            row = float(pts[i, 1]) * size_minus
            _draw_disk(img, row=row, col=col, radius=rad, color=colors[i], alpha=alpha)
    return img


def _write_rgb_video(
    frames: np.ndarray,
    output: Path,
    *,
    fps: float,
    codec: str,
    macro_block_size: int,
    max_frames: int | None,
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "video_path": str(output), "n_frames": 0}
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"expected RGB frame stack shape (T,H,W,C), got {arr.shape}")
    idx = _frame_indices(arr.shape[0], max_frames)
    if idx.size == 0:
        raise ValueError("empty RGB frame stack")
    ensure_dir(output.parent)
    first = np.asarray(arr[int(idx[0])])
    if first.shape[-1] > 3:
        first = first[..., :3]
    if first.shape[-1] < 3:
        first = np.repeat(first, int(math.ceil(3 / max(1, first.shape[-1]))), axis=-1)[..., :3]
    if first.dtype != np.uint8:
        first = (np.clip(first, 0.0, 1.0) * 255).astype(np.uint8)
    writer, cv2 = _open_video_writer(output, fps=fps, codec=codec, frame_shape=first.shape)
    written = 0
    try:
        for pos, i in enumerate(idx):
            frame = first if pos == 0 else np.asarray(arr[int(i)])
            if frame.shape[-1] > 3:
                frame = frame[..., :3]
            if frame.shape[-1] < 3:
                frame = np.repeat(frame, int(math.ceil(3 / max(1, frame.shape[-1]))), axis=-1)[..., :3]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()
    return {"status": "written", "video_path": str(output), "n_frames": int(written)}


def _write_xy_video(
    xy: np.ndarray,
    output: Path,
    *,
    fps: float,
    codec: str,
    macro_block_size: int,
    img_size: int,
    radius: int,
    trail_steps: int,
    max_frames: int | None,
    domain_size: float,
    wrap: bool,
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        return {"status": "exists", "video_path": str(output), "n_frames": 0}
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"expected XY trajectory shape (T,N,2), got {arr.shape}")
    idx = _frame_indices(arr.shape[0], max_frames)
    if idx.size == 0:
        raise ValueError("empty XY trajectory")
    ensure_dir(output.parent)
    first = _xy_frame(
        arr,
        int(idx[0]),
        img_size=int(img_size),
        radius=int(radius),
        trail_steps=int(trail_steps),
        domain_size=float(domain_size),
        wrap=bool(wrap),
    )
    writer, cv2 = _open_video_writer(output, fps=fps, codec=codec, frame_shape=first.shape)
    written = 0
    try:
        for pos, i in enumerate(idx):
            frame = first if pos == 0 else _xy_frame(
                arr,
                int(i),
                img_size=int(img_size),
                radius=int(radius),
                trail_steps=int(trail_steps),
                domain_size=float(domain_size),
                wrap=bool(wrap),
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()
    return {"status": "written", "video_path": str(output), "n_frames": int(written)}


def _load_trajectory_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            trial_idx = int(row.get("trial_idx", idx))
            rows.append(
                {
                    "trial_idx": trial_idx,
                    "trial_uid": str(row.get("trial_uid", row.get("traj_id", f"trial_{trial_idx:05d}"))),
                    "candidate_kind": str(row.get("candidate_kind", "")),
                    "candidate_label": str(row.get("candidate_label", "")),
                    "lagrangian_path": _path_from_root(root, row.get("lagrangian_path"), root / "trial_data" / f"trial_{trial_idx:05d}_lagrangian.npz"),
                }
            )
    elif (root / "trial_results.csv").exists():
        for idx, row in enumerate(read_csv(root / "trial_results.csv")):
            trial_idx = int(float(row.get("trial_idx", idx)))
            rows.append(
                {
                    "trial_idx": trial_idx,
                    "trial_uid": str(row.get("trial_uid", f"trial_{trial_idx:05d}")),
                    "candidate_kind": str(row.get("candidate_kind", "")),
                    "candidate_label": str(row.get("candidate_label", "")),
                    "lagrangian_path": _path_from_root(root, row.get("lagrangian_path"), root / "trial_data" / f"trial_{trial_idx:05d}_lagrangian.npz"),
                }
            )
    else:
        for path in sorted((root / "trial_data").glob("trial_*_lagrangian.npz")):
            stem = path.name.replace("_lagrangian.npz", "")
            try:
                trial_idx = int(stem.split("_")[-1])
            except ValueError:
                trial_idx = len(rows)
            rows.append(
                {
                    "trial_idx": trial_idx,
                    "trial_uid": stem,
                    "candidate_kind": "",
                    "candidate_label": "",
                    "lagrangian_path": path,
                }
            )
    return rows


def _render_lagrangian_root(
    *,
    root: Path,
    out_dir: Path,
    source_kind: str,
    xy_keys: tuple[str, ...],
    args: argparse.Namespace,
    limit_state: dict[str, int],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    if not root.exists():
        rows_out.append({"source_kind": source_kind, "source_root": str(root), "status": "missing_root"})
        return rows_out
    for row in _load_trajectory_rows(root):
        if args.limit is not None and limit_state["n"] >= int(args.limit):
            break
        path = Path(row["lagrangian_path"])
        if not path.exists():
            rows_out.append({**row, "source_kind": source_kind, "source_path": str(path), "status": "missing_lagrangian"})
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = [key for key in xy_keys if key in data.files]
                if not keys:
                    rows_out.append({**row, "source_kind": source_kind, "source_path": str(path), "status": "missing_xy"})
                    continue
                for key in keys:
                    if args.limit is not None and limit_state["n"] >= int(args.limit):
                        break
                    xy = np.asarray(data[key], dtype=np.float32)
                    name = f"{int(row['trial_idx']):05d}_{_slug(row['trial_uid'])}_{key}.mp4"
                    output = out_dir / source_kind / name
                    if args.dry_run:
                        result = {"status": "dry_run", "video_path": str(output), "n_frames": 0}
                    else:
                        result = _write_xy_video(
                            xy,
                            output,
                            fps=args.fps,
                            codec=args.codec,
                            macro_block_size=args.macro_block_size,
                            img_size=args.img_size,
                            radius=args.radius,
                            trail_steps=args.trail_steps,
                            max_frames=args.max_frames,
                            domain_size=args.domain_size,
                            wrap=not args.no_wrap,
                            force=args.force,
                        )
                    rows_out.append(
                        {
                            **row,
                            "source_kind": source_kind,
                            "source_path": str(path),
                            "xy_key": key,
                            **result,
                        }
                    )
                    limit_state["n"] += 1
        except Exception as exc:
            rows_out.append({**row, "source_kind": source_kind, "source_path": str(path), "status": "error", "message": f"{type(exc).__name__}: {exc}"})
    return rows_out


def _render_c2_branches(
    *,
    c2_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    limit_state: dict[str, int],
) -> list[dict[str, Any]]:
    plan_path = c2_dir / "branch_plan.csv"
    rows_out: list[dict[str, Any]] = []
    if not plan_path.exists():
        return [{"source_kind": "c2_branches", "source_root": str(c2_dir), "status": "missing_branch_plan"}]
    try:
        plan_rows = read_csv(plan_path)
    except Exception as exc:
        return [{"source_kind": "c2_branches", "source_root": str(c2_dir), "status": "error", "message": f"{type(exc).__name__}: {exc}"}]
    for row in plan_rows:
        if args.limit is not None and limit_state["n"] >= int(args.limit):
            break
        raw_output = str(row.get("branch_output_path", "")).strip()
        if raw_output:
            branch_path = Path(raw_output)
        else:
            branch_dir = Path(str(row.get("branch_dir", "")))
            branch_path = branch_dir / "branch_output.npz"
        if not branch_path.exists():
            rows_out.append({"source_kind": "c2_branches", "source_path": str(branch_path), "status": "missing_branch_output", **row})
            continue
        try:
            with np.load(branch_path, allow_pickle=False) as data:
                stem = (
                    f"{_slug(row.get('traj_id', 'traj'))}_point_{int(float(row.get('point_id', 0))):04d}"
                    f"_branch_{int(float(row.get('branch_id', 0))):03d}.mp4"
                )
                output = out_dir / "c2_branches" / stem
                if args.dry_run:
                    result = {"status": "dry_run", "video_path": str(output), "n_frames": 0}
                elif "rgb_future" in data.files:
                    result = _write_rgb_video(
                        np.asarray(data["rgb_future"]),
                        output,
                        fps=args.fps,
                        codec=args.codec,
                        macro_block_size=args.macro_block_size,
                        max_frames=args.max_frames,
                        force=args.force,
                    )
                elif "xy_future" in data.files:
                    result = _write_xy_video(
                        np.asarray(data["xy_future"], dtype=np.float32),
                        output,
                        fps=args.fps,
                        codec=args.codec,
                        macro_block_size=args.macro_block_size,
                        img_size=args.img_size,
                        radius=args.radius,
                        trail_steps=args.trail_steps,
                        max_frames=args.max_frames,
                        domain_size=args.domain_size,
                        wrap=not args.no_wrap,
                        force=args.force,
                    )
                else:
                    rows_out.append({"source_kind": "c2_branches", "source_path": str(branch_path), "status": "missing_rgb_or_xy", **row})
                    continue
            rows_out.append({"source_kind": "c2_branches", "source_path": str(branch_path), **row, **result})
            limit_state["n"] += 1
        except Exception as exc:
            rows_out.append({"source_kind": "c2_branches", "source_path": str(branch_path), "status": "error", "message": f"{type(exc).__name__}: {exc}", **row})
    return rows_out


def _plife_roots(cfg: Any, *, smoke: bool) -> dict[str, Path]:
    output_root = _output_root(cfg)
    ds = _get(_get(cfg.get("datasets", {}), "plife_plus", {}), "c1", {})
    plife_ds = _get(cfg.get("datasets", {}), "plife_plus", {})
    sim = _get(_get(cfg.get("simulation", {}), "plife_plus_c1_lagrangian", {}), "smoke_output_root" if smoke else "output_root", None)
    c1_root = _resolve(sim) or _resolve(_get(ds, "lagrangian_root", None)) or Path("experiments/paper_check_plife_plus/checkpoints/c1_lagrangian_24k")
    c5_root = _resolve(_get(plife_ds, "frustration_root", None)) or Path("experiments/paper_check_plife_plus/checkpoints/frustration_simulation")
    if smoke:
        smoke_c5 = output_root / "smoke_inputs" / "plife_plus" / "frustration_simulation"
        if smoke_c5.exists():
            c5_root = smoke_c5
    c2_cfg = _get(_get(cfg.get("c2", {}), "plife_plus", {}), "config", _get(cfg.get("c2", {}), "plife_plus", {}))
    c2_dir = _resolve(_get(c2_cfg, "output_dir", None)) or output_root / "c2_plife_plus_branching"
    return {"c1_lagrangian": c1_root, "c5_frustration": c5_root, "c2_branches": c2_dir}


def run(
    config_path: str | Path,
    *,
    smoke: bool = False,
    include: str = "all",
    force: bool = False,
    dry_run: bool = False,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    fps: float = 18.0,
    img_size: int = 256,
    radius: int = 3,
    trail_steps: int = 12,
    max_frames: int | None = 240,
    codec: str = "mp4v",
    macro_block_size: int = 1,
    domain_size: float = 1.0,
    no_wrap: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    root = _output_root(cfg)
    out_dir = ensure_dir(Path(output_dir) if output_dir is not None else root / "videos" / "plife_plus")
    include_set = {"c1", "c5", "c2"} if include == "all" else {part.strip() for part in include.split(",") if part.strip()}
    args = argparse.Namespace(
        force=force,
        dry_run=dry_run,
        limit=limit,
        fps=fps,
        img_size=img_size,
        radius=radius,
        trail_steps=trail_steps,
        max_frames=max_frames,
        codec=codec,
        macro_block_size=macro_block_size,
        domain_size=domain_size,
        no_wrap=no_wrap,
    )
    roots = _plife_roots(cfg, smoke=smoke)
    rows: list[dict[str, Any]] = []
    limit_state = {"n": 0}
    log_event(f"PLife++ video render start include={include} output={out_dir}", component="plife-video")
    if "c1" in include_set:
        rows.extend(
            _render_lagrangian_root(
                root=roots["c1_lagrangian"],
                out_dir=out_dir,
                source_kind="c1_lagrangian",
                xy_keys=("xy_trajectory", "xy_control_a"),
                args=args,
                limit_state=limit_state,
            )
        )
    if "c5" in include_set:
        rows.extend(
            _render_lagrangian_root(
                root=roots["c5_frustration"],
                out_dir=out_dir,
                source_kind="c5_frustration",
                xy_keys=("xy_control_a", "xy_control_b", "xy_walls"),
                args=args,
                limit_state=limit_state,
            )
        )
    if "c2" in include_set:
        rows.extend(_render_c2_branches(c2_dir=roots["c2_branches"], out_dir=out_dir, args=args, limit_state=limit_state))
    manifest = out_dir / "plife_video_manifest.csv"
    write_csv(manifest, rows)
    summary = {
        "status": "ok",
        "output_dir": str(out_dir),
        "manifest": str(manifest),
        "n_rows": len(rows),
        "n_written": sum(1 for row in rows if str(row.get("status")) == "written"),
        "n_exists": sum(1 for row in rows if str(row.get("status")) == "exists"),
        "n_skipped": sum(1 for row in rows if str(row.get("status", "")).startswith("skipped")),
        "n_errors": sum(1 for row in rows if str(row.get("status")) == "error"),
        "n_missing": sum(1 for row in rows if str(row.get("status", "")).startswith("missing")),
    }
    write_json(out_dir / "plife_video_summary.json", summary)
    log_event(
        f"PLife++ video render done n_written={summary['n_written']} n_exists={summary['n_exists']} "
        f"n_skipped={summary['n_skipped']} n_missing={summary['n_missing']} manifest={manifest}",
        component="plife-video",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render PLife++ videos from saved paper-suite trajectory artifacts.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include", default="all", help="Comma-separated subset: c1,c5,c2, or all.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of videos to write across selected sources.")
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--trail-steps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240, help="0 means all frames.")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--macro-block-size", type=int, default=1)
    parser.add_argument("--domain-size", type=float, default=1.0)
    parser.add_argument("--no-wrap", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(
        run(
            args.config,
            smoke=args.smoke,
            include=args.include,
            force=args.force,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            limit=args.limit,
            fps=args.fps,
            img_size=args.img_size,
            radius=args.radius,
            trail_steps=args.trail_steps,
            max_frames=None if int(args.max_frames) <= 0 else int(args.max_frames),
            codec=args.codec,
            macro_block_size=args.macro_block_size,
            domain_size=args.domain_size,
            no_wrap=args.no_wrap,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
