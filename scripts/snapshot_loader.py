import os
import re
import numpy as np
from typing import Optional, Tuple, List
import imageio.v3 as iio
import imageio.v3 as iio

_PATTERN = re.compile(
    r"P_steps_(\d+)_(\d+)__secs_([0-9.]+)_([0-9.]+)__idx_(\d+)\.npz$"
)


def _parse_file(path: str):
    name = os.path.basename(path)
    m = _PATTERN.match(name)
    if not m:
        return None
    s0, s1, t0, t1, idx = m.groups()
    return dict(
        path=path,
        start_step=int(s0),
        end_step=int(s1),
        start_sec=float(t0),
        end_sec=float(t1),
        idx=int(idx),
    )


def list_snapshot_files(base_dir: str):
    entries = []
    for fn in os.listdir(base_dir):
        p = _parse_file(os.path.join(base_dir, fn))
        if p is not None:
            entries.append(p)
    return sorted(entries, key=lambda d: (d["start_step"], d["idx"]))


def _overlaps(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def render_snapshots_to_video(
    P: np.ndarray,
    out_path: str,
    fps: int = 30,
    per_frame_norm: bool = False,
) -> None:
    """
    Render a sequence of P snapshots to an RGB video using the first 3 channels.

    Args:
        P: array of shape (T, H, W, C) containing snapshots.
        out_path: output video path (e.g., .mp4).
        fps: frames per second.
        per_frame_norm: if True, normalize each frame independently; otherwise
                        normalize using global min/max over the provided sequence.
    """
    if P.ndim != 4:
        raise ValueError(f"P should have shape (T, H, W, C); got {P.shape}")

    # Prepare global normalization if needed
    if per_frame_norm:
        global_min = None
        global_max = None
    else:
        p3_all = P[..., :3] if P.shape[-1] >= 3 else np.tile(P, (1, 1, 1, int(np.ceil(3 / P.shape[-1]))))[..., :3]
        global_min = float(np.min(p3_all))
        global_max = float(np.max(p3_all))

    frames = []
    for i in range(P.shape[0]):
        p = P[i]
        if p.shape[-1] < 3:
            reps = (1, 1, int(np.ceil(3 / p.shape[-1])))
            p3 = np.tile(p, reps)[..., :3]
        else:
            p3 = p[..., :3]
        if per_frame_norm:
            mn = float(np.min(p3))
            mx = float(np.max(p3))
        else:
            mn = global_min
            mx = global_max
        if mx is None or mx <= mn:
            rgb = np.zeros((*p3.shape[:2], 3), dtype=np.float32)
        else:
            rgb = (p3 - mn) / (mx - mn + 1e-8)
        frames.append((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps, codec="libx264")


def load_snapshots(
    base_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load P (and optional A) snapshots across chunk files for the requested step or second range.
    Provide either step range or sec range. Returns (steps, P, A) sorted by step. A is None if not saved.
    """
    files = list_snapshot_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {base_dir}")

    # Determine overlap criteria
    use_steps = (start_step is not None) or (end_step is not None)
    use_secs = (start_sec is not None) or (end_sec is not None)
    if not use_steps and not use_secs:
        use_steps = True
        start_step, end_step = files[0]["start_step"], files[-1]["end_step"]

    step_list: List[int] = []
    P_list: List[np.ndarray] = []
    A_list: List[np.ndarray] = []

    for entry in files:
        if use_steps:
            ss = entry["start_step"]
            ee = entry["end_step"]
            if not _overlaps(ss, ee, start_step if start_step is not None else ss, end_step if end_step is not None else ee):
                continue
        if use_secs:
            ts = entry["start_sec"]
            te = entry["end_sec"]
            if not _overlaps(ts, te, start_sec if start_sec is not None else ts, end_sec if end_sec is not None else te):
                continue
        data = np.load(entry["path"])
        steps = np.asarray(data["steps"])
        P = np.asarray(data["P"])
        A = np.asarray(data["A"]) if "A" in data.files else None

        # Filter within requested range
        if use_steps:
            mask = np.ones_like(steps, dtype=bool)
            if start_step is not None:
                mask &= steps >= start_step
            if end_step is not None:
                mask &= steps <= end_step
        elif use_secs:
            fps = float(data["fps"])
            secs = steps / fps
            mask = np.ones_like(secs, dtype=bool)
            if start_sec is not None:
                mask &= secs >= start_sec
            if end_sec is not None:
                mask &= secs <= end_sec
        else:
            mask = np.ones_like(steps, dtype=bool)

        if not np.any(mask):
            continue
        step_list.append(steps[mask])
        P_list.append(P[mask])
        if A is not None:
            A_list.append(A[mask])

    if not step_list:
        raise ValueError("No snapshots matched the requested range.")

    steps_all = np.concatenate(step_list, axis=0)
    P_all = np.concatenate(P_list, axis=0)
    A_all = np.concatenate(A_list, axis=0) if A_list else None
    order = np.argsort(steps_all)
    if A_all is not None:
        return steps_all[order], P_all[order], A_all[order]
    return steps_all[order], P_all[order], None


def render_snapshots_to_video(
    P: np.ndarray,
    out_path: str,
    fps: int = 30,
    per_frame_norm: bool = False,
    A: Optional[np.ndarray] = None,
) -> None:
    """
    Render a sequence of snapshots to an RGB video.

    If A is provided, mimic Pcolor rendering: rgb = clip(sum(A) * P[:3], 0, 1).
    Otherwise, normalize P[:3] for visualization.
    """
    if P.ndim != 4:
        raise ValueError(f"P should have shape (T, H, W, C); got {P.shape}")

    if A is not None and A.shape[0] != P.shape[0]:
        raise ValueError(f"A and P length mismatch: {A.shape[0]} vs {P.shape[0]}")

    frames = []
    if A is None:
        if per_frame_norm:
            global_min = None
            global_max = None
        else:
            p3_all = P[..., :3] if P.shape[-1] >= 3 else np.tile(P, (1, 1, 1, int(np.ceil(3 / P.shape[-1]))))[..., :3]
            global_min = float(np.min(p3_all))
            global_max = float(np.max(p3_all))
    else:
        # When A is present, we will clip after multiplying; no global scaling needed unless per_frame_norm True
        global_min = None
        global_max = None

    for i in range(P.shape[0]):
        p = P[i]
        if p.shape[-1] < 3:
            reps = (1, 1, int(np.ceil(3 / p.shape[-1])))
            p3 = np.tile(p, reps)[..., :3]
        else:
            p3 = p[..., :3]

        if A is not None:
            a_sum = np.sum(A[i], axis=-1, keepdims=True)
            rgb = np.clip(a_sum * p3, 0.0, 1.0)
            if per_frame_norm:
                mn = float(rgb.min())
                mx = float(rgb.max())
                if mx > mn:
                    rgb = (rgb - mn) / (mx - mn + 1e-8)
        else:
            if per_frame_norm:
                mn = float(np.min(p3))
                mx = float(np.max(p3))
            else:
                mn = global_min
                mx = global_max
            if mx is None or mx <= mn:
                rgb = np.zeros((*p3.shape[:2], 3), dtype=np.float32)
            else:
                rgb = (p3 - mn) / (mx - mn + 1e-8)
        frames.append((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps, codec="libx264")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load P snapshots over a step/second range.")
    ap.add_argument("--base_dir", required=True, help="Directory containing P snapshot chunk files.")
    ap.add_argument("--start_step", type=int, default=None, help="Start step (optional).")
    ap.add_argument("--end_step", type=int, default=None, help="End step (optional).")
    ap.add_argument("--start_sec", type=float, default=None, help="Start time in seconds (optional).")
    ap.add_argument("--end_sec", type=float, default=None, help="End time in seconds (optional).")
    args = ap.parse_args()

    steps, P = load_snapshots(
        args.base_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    print(f"Loaded {P.shape[0]} snapshots spanning steps [{steps.min()}, {steps.max()}]. Shape per snapshot: {P.shape[1:]}")
