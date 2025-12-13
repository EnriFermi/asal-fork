import os
import re
import numpy as np
from typing import Optional, Tuple, List

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


def load_snapshots(
    base_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load P snapshots across chunk files for the requested step or second range.
    Provide either step range or sec range. Returns (steps, P) sorted by step.
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

    if not step_list:
        raise ValueError("No snapshots matched the requested range.")

    steps_all = np.concatenate(step_list, axis=0)
    P_all = np.concatenate(P_list, axis=0)
    order = np.argsort(steps_all)
    return steps_all[order], P_all[order]


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
