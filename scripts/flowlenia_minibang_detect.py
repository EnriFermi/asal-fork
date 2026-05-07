from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import (
    intervals_from_mask,
    load_frame_times,
    merge_intervals,
    resolve_path,
    robust_z,
    step_to_video_sec,
    write_json,
)


def _load_manifest(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {dataset_root}")
    payload = json.loads(path.read_text())
    rows = payload.get("trajectories", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid manifest format in {path}")
    return payload


def _as_float_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    return np.asarray(data[key], dtype=np.float64)


def _interval_score(values_z: np.ndarray, i0: int, i1: int) -> float:
    if values_z.size == 0:
        return 0.0
    i0 = max(0, min(int(i0), values_z.size - 1))
    i1 = max(i0, min(int(i1), values_z.size - 1))
    segment = values_z[i0 : i1 + 1]
    return float(np.nanmax(segment)) if segment.size else 0.0


def _range_mask(steps: np.ndarray, start_step: int | None, end_step: int | None) -> np.ndarray:
    mask = np.ones(np.asarray(steps).shape, dtype=bool)
    if start_step is not None:
        mask &= np.asarray(steps) >= int(start_step)
    if end_step is not None:
        mask &= np.asarray(steps) <= int(end_step)
    return mask


def _clip_interval(
    start_step: int,
    end_step: int,
    *,
    detect_start_step: int | None,
    detect_end_step: int | None,
) -> tuple[int, int] | None:
    start = int(start_step)
    end = int(end_step)
    if detect_start_step is not None:
        if end < int(detect_start_step):
            return None
        start = max(start, int(detect_start_step))
    if detect_end_step is not None:
        if start > int(detect_end_step):
            return None
        end = min(end, int(detect_end_step))
    if end < start:
        return None
    return start, end


def detect_for_trajectory(
    row: dict[str, Any],
    *,
    delta_h_z: float,
    delta_h_quantile: float,
    mass_shift_z: float,
    mass_shift_quantile: float,
    merge_gap_steps: int,
    pad_steps: int,
    detect_start_step: int | None,
    detect_end_step: int | None,
) -> list[dict[str, Any]]:
    traj_dir = Path(str(row.get("traj_dir", "")))
    metrics_path = Path(str(row.get("metrics_path", traj_dir / "metrics.npz")))
    if not metrics_path.exists():
        print(f"Warning: metrics file missing for {row.get('traj_id')}: {metrics_path}")
        return []

    frame_times = load_frame_times(traj_dir)
    candidates: list[dict[str, Any]] = []

    with np.load(metrics_path) as data:
        dh = _as_float_array(data, "delta_h_best")
        dh_steps = _as_float_array(data, "delta_h_window_center_steps")
        dh_start = _as_float_array(data, "delta_h_window_start_steps")
        dh_end = _as_float_array(data, "delta_h_window_end_steps")
        if dh is not None and dh_steps is not None and dh.size == dh_steps.size and dh.size > 0:
            dh_z_values = robust_z(dh)
            q_thr = np.nanquantile(dh_z_values, float(delta_h_quantile)) if dh_z_values.size else np.inf
            mask = (dh_z_values >= float(delta_h_z)) | ((dh_z_values >= float(q_thr)) & (dh_z_values > 0.0))
            mask &= _range_mask(dh_steps, detect_start_step, detect_end_step)
            intervals = intervals_from_mask(dh_steps, mask, pad_steps=pad_steps)
            for start, end, i0, i1 in intervals:
                start_step = int(dh_start[i0]) if dh_start is not None and dh_start.size > i0 else int(start)
                end_step = int(dh_end[i1]) if dh_end is not None and dh_end.size > i1 else int(end)
                clipped = _clip_interval(
                    max(0, start_step - pad_steps),
                    max(0, end_step + pad_steps),
                    detect_start_step=detect_start_step,
                    detect_end_step=detect_end_step,
                )
                if clipped is None:
                    continue
                score = _interval_score(dh_z_values, i0, i1)
                candidates.append(
                    dict(
                        traj_id=row.get("traj_id"),
                        start_step=clipped[0],
                        end_step=clipped[1],
                        score=score,
                        delta_h_z_max=score,
                        mass_shift_z_max=0.0,
                        reasons=["delta_h_spike"],
                    )
                )

        tv = _as_float_array(data, "cluster_tv_lag")
        tv_steps = _as_float_array(data, "cluster_steps")
        if tv is not None and tv_steps is not None and tv.size == tv_steps.size and tv.size > 0:
            tv_z_values = robust_z(tv)
            q_thr = np.nanquantile(tv_z_values, float(mass_shift_quantile)) if tv_z_values.size else np.inf
            mask = (tv_z_values >= float(mass_shift_z)) | ((tv_z_values >= float(q_thr)) & (tv_z_values > 0.0))
            mask &= _range_mask(tv_steps, detect_start_step, detect_end_step)
            intervals = intervals_from_mask(tv_steps, mask, pad_steps=pad_steps)
            for start, end, i0, i1 in intervals:
                clipped = _clip_interval(
                    int(start),
                    int(end),
                    detect_start_step=detect_start_step,
                    detect_end_step=detect_end_step,
                )
                if clipped is None:
                    continue
                score = _interval_score(tv_z_values, i0, i1)
                candidates.append(
                    dict(
                        traj_id=row.get("traj_id"),
                        start_step=clipped[0],
                        end_step=clipped[1],
                        score=score,
                        delta_h_z_max=0.0,
                        mass_shift_z_max=score,
                        reasons=["cluster_mass_shift"],
                    )
                )

        entropy = _as_float_array(data, "cluster_entropy_norm")
        entropy_steps = _as_float_array(data, "cluster_steps")
        if entropy is not None and entropy_steps is not None and entropy.size == entropy_steps.size and entropy.size > 2:
            dent = np.zeros_like(entropy)
            dent[1:] = np.abs(np.diff(entropy))
            dent_z_values = robust_z(dent)
            q_thr = np.nanquantile(dent_z_values, float(mass_shift_quantile)) if dent_z_values.size else np.inf
            mask = (dent_z_values >= float(mass_shift_z)) | ((dent_z_values >= float(q_thr)) & (dent_z_values > 0.0))
            mask &= _range_mask(entropy_steps, detect_start_step, detect_end_step)
            intervals = intervals_from_mask(entropy_steps, mask, pad_steps=pad_steps)
            for start, end, i0, i1 in intervals:
                clipped = _clip_interval(
                    int(start),
                    int(end),
                    detect_start_step=detect_start_step,
                    detect_end_step=detect_end_step,
                )
                if clipped is None:
                    continue
                score = _interval_score(dent_z_values, i0, i1)
                candidates.append(
                    dict(
                        traj_id=row.get("traj_id"),
                        start_step=clipped[0],
                        end_step=clipped[1],
                        score=score,
                        delta_h_z_max=0.0,
                        mass_shift_z_max=score,
                        reasons=["cluster_entropy_change"],
                    )
                )

    merged = merge_intervals(candidates, gap_steps=int(merge_gap_steps))
    out: list[dict[str, Any]] = []
    for idx, cand in enumerate(merged):
        start_step = int(cand["start_step"])
        end_step = int(cand["end_step"])
        start_video_sec = step_to_video_sec(start_step, frame_times)
        end_video_sec = step_to_video_sec(end_step, frame_times)
        out.append(
            dict(
                traj_id=row.get("traj_id"),
                candidate_id=f"{row.get('traj_id')}_cand_{idx:03d}",
                video_path=row.get("video_path", str(traj_dir / "video.mp4")),
                metrics_path=str(metrics_path),
                optimization_iter=row.get("iter", ""),
                saturation_T=row.get("saturation_T", ""),
                source=row.get("source", ""),
                loss=row.get("loss", ""),
                start_step=start_step,
                end_step=end_step,
                start_video_sec=start_video_sec,
                end_video_sec=end_video_sec,
                duration_video_sec=(end_video_sec - start_video_sec)
                if np.isfinite(start_video_sec) and np.isfinite(end_video_sec)
                else float("nan"),
                score=float(cand.get("score", 0.0)),
                delta_h_z_max=float(cand.get("delta_h_z_max", 0.0)),
                mass_shift_z_max=float(cand.get("mass_shift_z_max", 0.0)),
                reasons=";".join(cand.get("reasons", [])),
            )
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = [
            "traj_id",
            "candidate_id",
            "video_path",
            "start_video_sec",
            "end_video_sec",
            "start_step",
            "end_step",
            "score",
            "delta_h_z_max",
            "mass_shift_z_max",
            "reasons",
            "optimization_iter",
            "saturation_T",
            "source",
            "loss",
            "metrics_path",
        ]
    else:
        fieldnames = [
            "traj_id",
            "candidate_id",
            "video_path",
            "start_video_sec",
            "end_video_sec",
            "start_step",
            "end_step",
            "score",
            "delta_h_z_max",
            "mass_shift_z_max",
            "reasons",
        ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_sec(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(val):
        return "nan"
    return f"{val:.2f}"


def _write_markdown(path: Path, rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> None:
    by_traj: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_traj.setdefault(str(row["traj_id"]), []).append(row)

    lines: list[str] = []
    lines.append("# Minibang Candidate Review")
    lines.append("")
    lines.append("Times are video seconds, so they can be pasted directly into a player timeline.")
    lines.append("")
    for manifest_row in manifest_rows:
        traj_id = str(manifest_row.get("traj_id"))
        video_path = manifest_row.get("video_path", "")
        cand_rows = by_traj.get(traj_id, [])
        lines.append(f"## {traj_id}")
        lines.append("")
        lines.append(f"- video: `{video_path}`")
        lines.append(f"- optimization_iter: `{manifest_row.get('iter', '')}`")
        lines.append(f"- saturation_T: `{manifest_row.get('saturation_T', '')}`")
        lines.append(f"- candidates: `{len(cand_rows)}`")
        lines.append("")
        if cand_rows:
            lines.append("| candidate | video start | video end | steps | score | reasons |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for cand in cand_rows:
                lines.append(
                    "| {candidate_id} | {start} | {end} | {s0}-{s1} | {score:.2f} | {reasons} |".format(
                        candidate_id=cand["candidate_id"],
                        start=_fmt_sec(cand["start_video_sec"]),
                        end=_fmt_sec(cand["end_video_sec"]),
                        s0=int(cand["start_step"]),
                        s1=int(cand["end_step"]),
                        score=float(cand["score"]),
                        reasons=cand["reasons"],
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect over-inclusive minibang candidates from flowlenia_minibang_simulate outputs."
    )
    parser.add_argument("dataset_root", help="Directory containing manifest.json and traj_XXXXX folders.")
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/detections.")
    parser.add_argument("--delta-h-z", type=float, default=2.5, help="Robust z threshold for deltaH spikes.")
    parser.add_argument(
        "--delta-h-quantile",
        type=float,
        default=0.90,
        help="Also require at least this within-trajectory deltaH robust-z quantile.",
    )
    parser.add_argument("--mass-shift-z", type=float, default=2.0, help="Robust z threshold for cluster shifts.")
    parser.add_argument(
        "--mass-shift-quantile",
        type=float,
        default=0.88,
        help="Also require at least this within-trajectory mass-shift robust-z quantile.",
    )
    parser.add_argument("--merge-gap-steps", type=int, default=5000, help="Merge candidate intervals separated by this gap.")
    parser.add_argument("--pad-steps", type=int, default=1000, help="Pad each raw candidate interval.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Ignore candidates before this simulation step. Defaults to manifest.detect_start_step.",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Ignore candidates after this simulation step. Defaults to manifest.detect_end_step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    output_dir = resolve_path(args.output_dir, dataset_root) if args.output_dir else dataset_root / "detections"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(dataset_root)
    manifest_rows = manifest.get("trajectories", [])
    detect_start_step = args.start_step
    if detect_start_step is None and manifest.get("detect_start_step", None) is not None:
        detect_start_step = int(manifest["detect_start_step"])
    detect_end_step = args.end_step
    if detect_end_step is None and manifest.get("detect_end_step", None) is not None:
        detect_end_step = int(manifest["detect_end_step"])
    all_candidates: list[dict[str, Any]] = []
    for row in manifest_rows:
        all_candidates.extend(
            detect_for_trajectory(
                row,
                delta_h_z=float(args.delta_h_z),
                delta_h_quantile=float(args.delta_h_quantile),
                mass_shift_z=float(args.mass_shift_z),
                mass_shift_quantile=float(args.mass_shift_quantile),
                merge_gap_steps=int(args.merge_gap_steps),
                pad_steps=int(args.pad_steps),
                detect_start_step=detect_start_step,
                detect_end_step=detect_end_step,
            )
        )

    all_candidates.sort(key=lambda r: (str(r["traj_id"]), int(r["start_step"])))
    _write_csv(output_dir / "minibang_candidates.csv", all_candidates)
    write_json(output_dir / "minibang_candidates.json", all_candidates)
    _write_markdown(output_dir / "minibang_candidates.md", all_candidates, manifest_rows)
    print(f"Wrote {len(all_candidates)} candidates to {output_dir}")


if __name__ == "__main__":
    main()
