import argparse
import os
import re
from typing import Iterator, Optional, Tuple

import imageio
import numpy as np


_NPZ_PATTERN = re.compile(
    r"P_steps_(\d+)_(\d+)__secs_([0-9.]+)_([0-9.]+)__idx_(\d+)\.npz$"
)


def _list_chunks(base_dir: str) -> list[Tuple[str, int, int, int]]:
    chunks: list[Tuple[str, int, int, int]] = []
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


def _to_rgb_u8_from_ap(p: np.ndarray, a: Optional[np.ndarray]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.shape[-1] >= 3:
        p3 = p[..., :3]
    else:
        reps = int(np.ceil(3 / p.shape[-1]))
        p3 = np.tile(p, (1, 1, reps))[..., :3]

    if a is None:
        mn = float(np.min(p3))
        mx = float(np.max(p3))
        if mx <= mn:
            rgb = np.zeros_like(p3, dtype=np.float32)
        else:
            rgb = (p3 - mn) / (mx - mn + 1e-8)
    else:
        a = np.asarray(a, dtype=np.float32)
        inten = np.sum(a, axis=-1, keepdims=True)
        rgb = np.clip(inten * p3, 0.0, 1.0)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _iter_frames(
    base_dir: str,
    *,
    start_step: Optional[int],
    end_step: Optional[int],
    snapshot_stride: int,
    prefer_saved_rgb: bool,
) -> Iterator[np.ndarray]:
    chunks = _list_chunks(base_dir)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {base_dir}")

    snapshot_counter = 0
    for path, s0, s1, _idx in chunks:
        if not _overlaps(s0, s1, start_step, end_step):
            continue

        with np.load(path) as data:
            if "steps" not in data.files:
                raise ValueError(f"Chunk {os.path.basename(path)} has no 'steps'.")
            if "P" not in data.files:
                raise ValueError(f"Chunk {os.path.basename(path)} has no 'P'.")

            steps = np.asarray(data["steps"], dtype=np.int64)
            p_arr = np.asarray(data["P"])
            rgb_arr = np.asarray(data["rgb"]) if (prefer_saved_rgb and "rgb" in data.files) else None
            a_arr = np.asarray(data["A"]) if "A" in data.files else None

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
                    frame = np.asarray(rgb_arr[i])
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    a_frame = a_arr[i] if a_arr is not None else None
                    frame = _to_rgb_u8_from_ap(p_arr[i], a_frame)
                yield frame


def main():
    ap = argparse.ArgumentParser(description="Render mp4 video from APF npz logs (P_steps_*.npz).")
    ap.add_argument("--input_dir", required=True, help="Directory with P_steps_*.npz chunks.")
    ap.add_argument("--output", default=None, help="Output mp4 path.")
    ap.add_argument("--start_step", type=int, default=None, help="Inclusive start simulation step.")
    ap.add_argument("--end_step", type=int, default=None, help="Inclusive end simulation step.")
    ap.add_argument("--snapshot_stride", type=int, default=1, help="Use every N-th logged snapshot.")
    ap.add_argument("--fps", type=float, default=None, help="Output fps. If omitted, uses logged fps.")
    ap.add_argument("--codec", type=str, default="libx264", help="Output codec for imageio writer.")
    ap.add_argument("--macro_block_size", type=int, default=1, help="Writer macro block size.")
    ap.add_argument("--prefer_saved_rgb", action="store_true", help="Use saved rgb if present in chunks.")
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    chunks = _list_chunks(input_dir)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {input_dir}")

    with np.load(chunks[0][0]) as first:
        logged_fps = float(np.asarray(first["fps"]).item()) if "fps" in first.files else 25.0
    out_fps = float(args.fps) if args.fps is not None else logged_fps

    output = args.output
    if output is None:
        output = os.path.join(input_dir, "apf_video.mp4")
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    snapshot_stride = max(1, int(args.snapshot_stride))
    frame_iter = _iter_frames(
        input_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        snapshot_stride=snapshot_stride,
        prefer_saved_rgb=bool(args.prefer_saved_rgb),
    )

    writer = imageio.get_writer(
        output,
        fps=out_fps,
        codec=str(args.codec),
        macro_block_size=int(args.macro_block_size),
    )
    written = 0
    try:
        for frame in frame_iter:
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()

    if written <= 0:
        raise RuntimeError("No frames matched the requested range.")
    print(f"Wrote {written} frames to {output} at {out_fps:.3f} fps.")


if __name__ == "__main__":
    main()
