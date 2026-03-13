import argparse
import os
import re
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


_NPZ_PATTERN = re.compile(
    r"P_steps_(\d+)_(\d+)__secs_([0-9.]+)_([0-9.]+)__idx_(\d+)\.npz$"
)


def _list_chunks(base_dir: str) -> List[Tuple[str, int, int, int]]:
    chunks: List[Tuple[str, int, int, int]] = []
    for fn in os.listdir(base_dir):
        m = _NPZ_PATTERN.match(fn)
        if not m:
            continue
        s0, s1, _t0, _t1, idx = m.groups()
        chunks.append((os.path.join(base_dir, fn), int(s0), int(s1), int(idx)))
    chunks.sort(key=lambda x: (x[1], x[3]))
    return chunks


def _overlaps(a0: int, a1: int, b0: Optional[int], b1: Optional[int]) -> bool:
    lo = a0 if b0 is None else int(b0)
    hi = a1 if b1 is None else int(b1)
    return not (a1 < lo or hi < a0)


def _to_rgb_u8_from_ap(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32, copy=False)
    a = a.astype(np.float32, copy=False)
    if p.shape[-1] >= 3:
        p3 = p[..., :3]
    else:
        reps = int(np.ceil(3 / p.shape[-1]))
        p3 = np.tile(p, (1, 1, reps))[..., :3]
    a_sum = np.sum(a, axis=-1, keepdims=True)
    rgb = np.clip(a_sum * p3, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _iter_frames(
    base_dir: str,
    start_step: Optional[int],
    end_step: Optional[int],
    snapshot_stride: int,
    prefer_saved_rgb: bool,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    chunks = _list_chunks(base_dir)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {base_dir}")

    snapshot_counter = 0
    for path, s0, s1, _idx in chunks:
        if not _overlaps(s0, s1, start_step, end_step):
            continue

        with np.load(path) as data:
            if "lagrangian_xy" not in data.files:
                raise ValueError(
                    f"Chunk {os.path.basename(path)} does not contain 'lagrangian_xy'. "
                    "Run simulate_save_apf with save_lagrangian=true."
                )
            if "steps" not in data.files:
                raise ValueError(f"Chunk {os.path.basename(path)} does not contain 'steps'.")

            steps = np.asarray(data["steps"], dtype=np.int64)
            lag = np.asarray(data["lagrangian_xy"], dtype=np.float32)
            has_rgb = "rgb" in data.files
            has_a = "A" in data.files
            has_p = "P" in data.files

            if prefer_saved_rgb and not has_rgb and (not has_a or not has_p):
                raise ValueError(
                    f"Chunk {os.path.basename(path)} has no 'rgb' and missing 'A'/'P' for fallback rendering."
                )
            if (not prefer_saved_rgb) and (not has_a or not has_p):
                raise ValueError(
                    f"Chunk {os.path.basename(path)} requires 'A' and 'P' for rendering, but one is missing."
                )

            rgb_arr = np.asarray(data["rgb"]) if has_rgb and prefer_saved_rgb else None
            a_arr = np.asarray(data["A"]) if rgb_arr is None else None
            p_arr = np.asarray(data["P"]) if rgb_arr is None else None

            for i, step in enumerate(steps):
                step_i = int(step)
                if start_step is not None and step_i < int(start_step):
                    continue
                if end_step is not None and step_i > int(end_step):
                    continue
                if snapshot_counter % snapshot_stride != 0:
                    snapshot_counter += 1
                    continue
                snapshot_counter += 1

                if rgb_arr is not None:
                    frame = rgb_arr[i]
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    frame = _to_rgb_u8_from_ap(p_arr[i], a_arr[i])

                yield step_i, frame, lag[i]


def _make_distinct_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.arange(n) * 179 // max(1, n)).astype(np.uint8)
    hsv[:, 0, 1] = 220
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]
    return bgr


def _xy_to_cv(points_yx: np.ndarray, h: int, w: int) -> np.ndarray:
    y = np.clip(np.rint(points_yx[:, 0] - 0.5), 0, h - 1).astype(np.int32)
    x = np.clip(np.rint(points_yx[:, 1] - 0.5), 0, w - 1).astype(np.int32)
    return np.stack([x, y], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Render Lagrangian trajectories over snapshot video.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with P_steps_*.npz chunks.")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path.")
    parser.add_argument("--start_step", type=int, default=None, help="Inclusive start step.")
    parser.add_argument("--end_step", type=int, default=None, help="Inclusive end step.")
    parser.add_argument(
        "--snapshot_stride",
        type=int,
        default=1,
        help="Use every N-th logged snapshot (not simulation step).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. If omitted, uses logged fps.")
    parser.add_argument("--codec", type=str, default="mp4v", help="OpenCV fourcc code.")
    parser.add_argument("--trail_length", type=int, default=40, help="Number of past positions to keep.")
    parser.add_argument("--point_radius", type=int, default=2, help="Tracked point radius in pixels.")
    parser.add_argument("--line_thickness", type=int, default=1, help="Trajectory line thickness.")
    parser.add_argument("--draw_points", action="store_true", help="Draw current points as circles.")
    parser.add_argument("--draw_step_text", action="store_true", help="Overlay current step text.")
    parser.add_argument("--prefer_saved_rgb", action="store_true", help="Use saved rgb if available.")
    parser.add_argument("--particle_stride", type=int, default=1, help="Draw every N-th particle id.")
    parser.add_argument(
        "--max_particles",
        type=int,
        default=None,
        help="Cap number of particles rendered after striding.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    chunks = _list_chunks(input_dir)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in: {input_dir}")

    with np.load(chunks[0][0]) as first_chunk:
        logged_fps = float(np.asarray(first_chunk["fps"]).item()) if "fps" in first_chunk.files else 25.0
    out_fps = float(args.fps) if args.fps is not None else logged_fps

    output = args.output
    if output is None:
        output = os.path.join(input_dir, "lagrangian_overlay.mp4")
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    snapshot_stride = max(1, int(args.snapshot_stride))
    trail_length = max(1, int(args.trail_length))
    particle_stride = max(1, int(args.particle_stride))
    point_radius = max(1, int(args.point_radius))
    line_thickness = max(1, int(args.line_thickness))

    frame_iter = _iter_frames(
        input_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        snapshot_stride=snapshot_stride,
        prefer_saved_rgb=bool(args.prefer_saved_rgb),
    )

    first = next(frame_iter, None)
    if first is None:
        raise ValueError("No frames matched the selected range.")
    first_step, first_frame, first_xy = first

    h, w = first_frame.shape[:2]
    if first_xy.ndim != 2 or first_xy.shape[-1] != 2:
        raise ValueError(f"Expected lagrangian_xy shape (N,2), got {first_xy.shape}.")

    keep_ids = np.arange(first_xy.shape[0], dtype=np.int32)[::particle_stride]
    if args.max_particles is not None:
        keep_ids = keep_ids[: max(0, int(args.max_particles))]
    if keep_ids.size == 0:
        raise ValueError("No particles selected for drawing. Check particle_stride/max_particles.")

    colors = _make_distinct_colors(keep_ids.size)
    history: List[np.ndarray] = []

    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4].ljust(4, " "))
    writer = cv2.VideoWriter(output, fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}.")

    def draw_frame(step: int, frame_rgb: np.ndarray, xy_all: np.ndarray) -> np.ndarray:
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        xy_sel = np.asarray(xy_all, dtype=np.float32)[keep_ids]
        history.append(xy_sel.copy())
        if len(history) > trail_length:
            del history[0]

        if len(history) > 1:
            hist = np.stack(history, axis=0)  # (L,N,2)
            for i in range(hist.shape[1]):
                poly = _xy_to_cv(hist[:, i, :], h, w).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], isClosed=False, color=tuple(int(c) for c in colors[i]), thickness=line_thickness)

        if args.draw_points:
            pts = _xy_to_cv(xy_sel, h, w)
            for i, pt in enumerate(pts):
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), point_radius, tuple(int(c) for c in colors[i]), -1)

        if args.draw_step_text:
            cv2.putText(
                canvas,
                f"step={int(step)}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    from tqdm import tqdm
    written = 0
    try:
        writer.write(draw_frame(first_step, first_frame, first_xy))
        written += 1
        for step, frame, xy in tqdm(frame_iter):
            writer.write(draw_frame(step, frame, xy))
            written += 1
            if written > 1000:
                break
    finally:
        writer.release()

    print(f"Wrote {written} frames to {output} at {out_fps:.3f} fps.")


if __name__ == "__main__":
    main()
