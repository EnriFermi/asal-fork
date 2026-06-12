from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import list_apf_chunks
from paper_suite_common import load_config, resolve_path, write_json


class _Mp4Writer:
    def __init__(self, path: Path, *, fps: float, first_frame: np.ndarray):
        self.path = Path(path)
        self._imageio_writer = None
        self._cv2_writer = None
        try:
            import imageio  # type: ignore

            self._imageio_writer = imageio.get_writer(
                str(self.path),
                fps=float(fps),
                codec="libx264",
                macro_block_size=1,
            )
            return
        except ModuleNotFoundError:
            pass

        import cv2  # type: ignore

        h, w = int(first_frame.shape[0]), int(first_frame.shape[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv2_writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), (w, h))
        if not self._cv2_writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV video writer for {self.path}")

    def append_data(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if self._imageio_writer is not None:
            self._imageio_writer.append_data(frame)
            return
        import cv2  # type: ignore

        self._cv2_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._imageio_writer is not None:
            self._imageio_writer.close()
        if self._cv2_writer is not None:
            self._cv2_writer.release()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return default


def _default_output_root(config: Path | None) -> Path:
    if config is None:
        return _REPO_ROOT / "analysis/results/paper_suite"
    cfg, _ = load_config(config)
    raw = cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")
    resolved = resolve_path(raw)
    return resolved if resolved is not None else _REPO_ROOT / "analysis/results/paper_suite"


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exists:\n" + "\n".join(str(p) for p in paths))


def _resolve_data_path(raw: str, *, rewrite_prefix: tuple[str, str] | None) -> Path:
    path_s = str(raw)
    if rewrite_prefix is not None:
        old, new = rewrite_prefix
        if path_s.startswith(old):
            path_s = new + path_s[len(old) :]
    path = Path(path_s)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _parse_rewrite_prefix(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    if "=" not in raw:
        raise ValueError("--rewrite-prefix must have form OLD=NEW")
    old, new = raw.split("=", 1)
    if not old:
        raise ValueError("--rewrite-prefix OLD side must be non-empty")
    return old, new


def _select_score_row(
    rows: list[dict[str, str]],
    *,
    plan_rows: list[dict[str, str]] | None = None,
    min_branches: int = 1,
    min_delta_h_quantile: float,
    traj_id: str | None,
    pair_id: int | None,
    condition: str | None,
) -> dict[str, str]:
    finite = [
        row
        for row in rows
        if np.isfinite(_as_float(row.get("delta_h"))) and np.isfinite(_as_float(row.get("branching_score")))
    ]
    if not finite:
        raise ValueError("No finite rows with delta_h and branching_score found in scores CSV.")

    if traj_id is not None:
        finite = [row for row in finite if str(row.get("traj_id")) == str(traj_id)]
    if pair_id is not None:
        finite = [row for row in finite if _as_int(row.get("pair_id")) == int(pair_id)]
    if condition is not None:
        finite = [row for row in finite if str(row.get("condition")) == str(condition)]
    if not finite:
        raise ValueError("No score rows remain after --traj-id/--pair-id/--condition filters.")

    if traj_id is None and pair_id is None and condition is None:
        qs = np.asarray([_as_float(row.get("delta_h")) for row in finite], dtype=np.float64)
        threshold = float(np.quantile(qs, float(min_delta_h_quantile)))
        candidates = [row for row in finite if _as_float(row.get("delta_h")) >= threshold]
        if candidates:
            finite = candidates

    if plan_rows is not None and int(min_branches) > 1:
        counts = [(len(_matching_plan_rows(plan_rows, row)), row) for row in finite]
        enough = [row for count, row in counts if count >= int(min_branches)]
        if not enough:
            max_count = max((count for count, _row in counts), default=0)
            raise RuntimeError(
                f"No selected C2 score row has at least {int(min_branches)} branch rows; "
                f"maximum available after filters is {max_count}. The branch plan was likely "
                "generated with c2.branching.branches_per_time < 4. Re-run the C2 branching "
                "simulation layer with c2.branching.branches_per_time=4, then recompute the "
                "C2 branching metrics."
            )
        finite = enough

    return max(finite, key=lambda row: _as_float(row.get("branching_score")))


def _matching_plan_rows(plan_rows: list[dict[str, str]], selected: dict[str, str]) -> list[dict[str, str]]:
    traj_id = str(selected.get("traj_id"))
    pair_id = _as_int(selected.get("pair_id"))
    condition = str(selected.get("condition"))
    step = _as_float(selected.get("step"))

    rows = [
        row
        for row in plan_rows
        if str(row.get("traj_id")) == traj_id
        and _as_int(row.get("pair_id")) == pair_id
        and str(row.get("condition")) == condition
    ]
    if not rows:
        return []

    if np.isfinite(step):
        same_step = [row for row in rows if abs(_as_float(row.get("step")) - step) < 0.5]
        if same_step:
            rows = same_step
    rows.sort(key=lambda row: (_as_int(row.get("branch_id"), 0) or 0, _as_int(row.get("branch_seed"), 0) or 0))
    return rows


def _to_rgb_u8(p: np.ndarray, a: np.ndarray | None) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.shape[-1] >= 3:
        p3 = p[..., :3]
    else:
        reps = int(np.ceil(3.0 / max(1, p.shape[-1])))
        p3 = np.tile(p, (1, 1, reps))[..., :3]
    if a is not None:
        aa = np.asarray(a, dtype=np.float32)
        intensity = np.sum(aa, axis=-1, keepdims=True)
        rgb = intensity * p3
    else:
        rgb = p3
        lo = float(np.nanmin(rgb))
        hi = float(np.nanmax(rgb))
        if hi > lo:
            rgb = (rgb - lo) / (hi - lo)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _frame_from_npz(data: np.lib.npyio.NpzFile, index: int, *, prefer_saved_rgb: bool) -> np.ndarray:
    if prefer_saved_rgb and "rgb" in data.files:
        frame = np.asarray(data["rgb"][index])
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        return frame
    if "P" not in data.files:
        raise ValueError("APF chunk has no P array and no usable saved rgb.")
    p_arr = np.asarray(data["P"])
    a_arr = np.asarray(data["A"]) if "A" in data.files else None
    a_frame = a_arr[index] if a_arr is not None else None
    return _to_rgb_u8(p_arr[index], a_frame)


def _resize_nearest(frame: np.ndarray, size: int) -> np.ndarray:
    frame = np.asarray(frame)
    if int(frame.shape[0]) == int(size) and int(frame.shape[1]) == int(size):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, int(size)).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, int(size)).astype(np.int64)
    return frame[y_idx][:, x_idx]


def _load_source_frame(
    plan_row: dict[str, str],
    *,
    rewrite_prefix: tuple[str, str] | None,
    panel_size: int | None,
    prefer_saved_rgb: bool,
) -> np.ndarray | None:
    raw_apf = str(plan_row.get("source_apf_dir", "")).strip()
    if not raw_apf:
        return None
    apf_dir = _resolve_data_path(raw_apf, rewrite_prefix=rewrite_prefix)
    if not apf_dir.exists():
        return None
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        return None
    target_step = _as_int(plan_row.get("step"))
    best: tuple[int, Path, int] | None = None
    for path, _s0, _s1, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            if "steps" not in data.files:
                continue
            steps = np.asarray(data["steps"], dtype=np.int64)
            if steps.size == 0:
                continue
            if target_step is None:
                candidate_idx = 0
            else:
                candidate_idx = int(np.argmin(np.abs(steps - int(target_step))))
            distance = 0 if target_step is None else int(abs(int(steps[candidate_idx]) - int(target_step)))
            if best is None or distance < best[0]:
                best = (distance, path, candidate_idx)
    if best is None:
        return None
    _distance, path, idx = best
    with np.load(path, allow_pickle=False) as data:
        frame = _frame_from_npz(data, idx, prefer_saved_rgb=prefer_saved_rgb)
    if panel_size is not None:
        frame = _resize_nearest(frame, int(panel_size))
    return frame


def _load_branch_frames(
    branch_dir: Path,
    *,
    max_frames: int,
    snapshot_stride: int,
    panel_size: int | None,
    prefer_saved_rgb: bool,
) -> list[np.ndarray]:
    apf_dir = branch_dir / "apf_logs"
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")

    frames: list[np.ndarray] = []
    seen = 0
    for path, _s0, _s1, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            if "P" not in data.files and not (prefer_saved_rgb and "rgb" in data.files):
                continue
            rgb_arr = np.asarray(data["rgb"]) if prefer_saved_rgb and "rgb" in data.files else None
            p_arr = np.asarray(data["P"]) if "P" in data.files else None
            a_arr = np.asarray(data["A"]) if "A" in data.files else None
            n = int(rgb_arr.shape[0] if rgb_arr is not None else p_arr.shape[0])
            for i in range(n):
                if seen % max(1, int(snapshot_stride)) != 0:
                    seen += 1
                    continue
                seen += 1
                frame = _frame_from_npz(data, i, prefer_saved_rgb=prefer_saved_rgb)
                if panel_size is not None:
                    frame = _resize_nearest(frame, int(panel_size))
                frames.append(frame)
                if len(frames) >= int(max_frames):
                    return frames
    return frames


def _compose_grid(frames: list[np.ndarray], *, gap: int) -> np.ndarray:
    if len(frames) != 4:
        raise ValueError(f"Expected exactly 4 frames, got {len(frames)}")
    h = min(int(frame.shape[0]) for frame in frames)
    w = min(int(frame.shape[1]) for frame in frames)
    trimmed = [frame[:h, :w, :3] for frame in frames]
    gap = max(0, int(gap))
    out = np.zeros((2 * h + gap, 2 * w + gap, 3), dtype=np.uint8)
    out[:h, :w] = trimmed[0]
    out[:h, w + gap : w + gap + w] = trimmed[1]
    out[h + gap : h + gap + h, :w] = trimmed[2]
    out[h + gap : h + gap + h, w + gap : w + gap + w] = trimmed[3]
    return out


def _write_mp4(branch_frames: list[list[np.ndarray]], output: Path, *, fps: float, gap: int, loop_hold_frames: int) -> int:
    n_frames = min(len(frames) for frames in branch_frames)
    if n_frames <= 0:
        raise RuntimeError("Selected branches contain no renderable frames.")
    output.parent.mkdir(parents=True, exist_ok=True)
    first_grid = _compose_grid([frames[0] for frames in branch_frames], gap=gap)
    writer = _Mp4Writer(output, fps=float(fps), first_frame=first_grid)
    written = 0
    try:
        writer.append_data(first_grid)
        written += 1
        for i in range(n_frames):
            if i == 0:
                continue
            grid = _compose_grid([frames[i] for frames in branch_frames], gap=gap)
            writer.append_data(grid)
            written += 1
        if loop_hold_frames > 0:
            grid = _compose_grid([frames[-1] for frames in branch_frames], gap=gap)
            for _ in range(int(loop_hold_frames)):
                writer.append_data(grid)
                written += 1
    finally:
        writer.close()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Render a 2x2 video of four Flow-Lenia C2 branch continuations selected from high "
            "Delta-H/high branch-divergence rows."
        )
    )
    ap.add_argument("--config", type=Path, default=Path("experiments/paper_suite/config.yaml"))
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--scores", type=Path, default=None)
    ap.add_argument("--branch-plan", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metadata-out", type=Path, default=None)
    ap.add_argument("--metric-suffix", choices=["clip_chamfer", "apf"], default="clip_chamfer")
    ap.add_argument("--min-delta-h-quantile", type=float, default=0.75)
    ap.add_argument("--traj-id", default=None)
    ap.add_argument("--pair-id", type=int, default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--snapshot-stride", type=int, default=1)
    ap.add_argument("--panel-size", type=int, default=256)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument("--loop-hold-frames", type=int, default=12)
    ap.add_argument("--prefer-saved-rgb", action="store_true")
    ap.add_argument(
        "--no-prepend-source-frame",
        action="store_true",
        help="Do not prepend the shared source APF frame before the four perturbed branches.",
    )
    ap.add_argument(
        "--rewrite-prefix",
        default=None,
        help="Rewrite stored absolute paths, e.g. /old/repo=/home/coder/project.",
    )
    args = ap.parse_args()

    if int(args.n_branches) != 4:
        raise ValueError("This renderer currently expects --n-branches 4 for a 2x2 grid.")

    output_root = args.output_root
    if output_root is None:
        output_root = _default_output_root(args.config)
    elif not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()

    c2_dir = output_root / "c2_branching"
    if args.scores is not None:
        scores_path = args.scores if args.scores.is_absolute() else (_REPO_ROOT / args.scores)
    else:
        suffix = "" if args.metric_suffix == "apf" else "_clip_chamfer"
        scores_path = _first_existing([c2_dir / f"branching_scores{suffix}.csv", c2_dir / "branching_scores.csv"])
    if args.branch_plan is not None:
        plan_path = args.branch_plan if args.branch_plan.is_absolute() else (_REPO_ROOT / args.branch_plan)
    else:
        plan_path = c2_dir / "branch_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing branch plan: {plan_path}")

    out_path = args.out
    if out_path is None:
        out_path = output_root / "figures/c2_flowlenia_branch_divergence_2x2.mp4"
    elif not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    meta_path = args.metadata_out
    if meta_path is None:
        meta_path = out_path.with_suffix(".json")
    elif not meta_path.is_absolute():
        meta_path = (_REPO_ROOT / meta_path).resolve()

    score_rows = _read_csv(scores_path)
    plan_rows = _read_csv(plan_path)
    selected = _select_score_row(
        score_rows,
        plan_rows=plan_rows,
        min_branches=int(args.n_branches),
        min_delta_h_quantile=float(args.min_delta_h_quantile),
        traj_id=args.traj_id,
        pair_id=args.pair_id,
        condition=args.condition,
    )
    matching = _matching_plan_rows(plan_rows, selected)
    if len(matching) < 4:
        raise RuntimeError(
            f"Selected point has only {len(matching)} branch rows in {plan_path}; need at least 4."
        )

    rewrite_prefix = _parse_rewrite_prefix(args.rewrite_prefix)
    selected_branches = matching[:4]
    branch_dirs = [_resolve_data_path(str(row["branch_dir"]), rewrite_prefix=rewrite_prefix) for row in selected_branches]
    branch_frames = [
        _load_branch_frames(
            branch_dir,
            max_frames=int(args.max_frames),
            snapshot_stride=int(args.snapshot_stride),
            panel_size=int(args.panel_size) if args.panel_size and args.panel_size > 0 else None,
            prefer_saved_rgb=bool(args.prefer_saved_rgb),
        )
        for branch_dir in branch_dirs
    ]
    source_frame = None
    if not args.no_prepend_source_frame:
        source_frame = _load_source_frame(
            selected_branches[0],
            rewrite_prefix=rewrite_prefix,
            panel_size=int(args.panel_size) if args.panel_size and args.panel_size > 0 else None,
            prefer_saved_rgb=bool(args.prefer_saved_rgb),
        )
        if source_frame is not None:
            branch_frames = [[source_frame] + frames for frames in branch_frames]
    written = _write_mp4(
        branch_frames,
        out_path,
        fps=float(args.fps),
        gap=int(args.gap),
        loop_hold_frames=int(args.loop_hold_frames),
    )

    payload = {
        "output": str(out_path),
        "frames_written": int(written),
        "scores_path": str(scores_path),
        "branch_plan": str(plan_path),
        "selection_rule": "max branching_score among rows with delta_h >= quantile threshold unless filters are supplied",
        "min_delta_h_quantile": float(args.min_delta_h_quantile),
        "selected_score_row": selected,
        "selected_branch_rows": selected_branches,
        "branch_dirs": [str(path) for path in branch_dirs],
        "prepended_source_frame": source_frame is not None,
        "per_branch_frames": [len(frames) for frames in branch_frames],
        "fps": float(args.fps),
        "panel_size": int(args.panel_size),
        "snapshot_stride": int(args.snapshot_stride),
    }
    write_json(meta_path, payload)
    print(
        "Rendered C2 branch divergence grid "
        f"traj={selected.get('traj_id')} pair={selected.get('pair_id')} condition={selected.get('condition')} "
        f"delta_h={_as_float(selected.get('delta_h')):.6g} "
        f"branching_score={_as_float(selected.get('branching_score')):.6g} "
        f"frames={written} output={out_path}"
    )
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
