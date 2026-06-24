from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _run_dirs(root: Path, runs: str | None, max_runs: int | None) -> list[Path]:
    if runs:
        out = [root / name.strip() for name in str(runs).split(",") if name.strip()]
    else:
        out = [p for p in sorted(root.glob("run_*")) if p.is_dir()]
    out = [p for p in out if (p / "best.pkl").exists() and (p / "pop_traj.pkl").exists()]
    if max_runs is not None:
        out = out[: int(max_runs)]
    if not out:
        raise FileNotFoundError(f"No run_*/best.pkl+pop_traj.pkl found under {root}")
    return out


def _summary_row(run_dir: Path, label: str, summary: dict[str, Any]) -> dict[str, Any]:
    per = [float(x) for x in summary.get("per_rep_score", [])]
    row: dict[str, Any] = {
        "run": run_dir.name,
        "label": label,
        "range_start_steps": summary.get("metric_range_start_steps"),
        "range_end_steps": summary.get("metric_range_end_steps"),
        "n_windows": summary.get("metric_n_windows"),
        "optimizer_mean_score": summary.get("optimizer_mean_score"),
        "min_rep_idx": summary.get("min_rep_idx"),
        "max_rep_idx": summary.get("max_rep_idx"),
        "output_dir": summary.get("output_dir"),
        "maps_npz": summary.get("metric_maps_npz"),
        "video": summary.get("all_reps_video_path") or summary.get("video_path"),
        "delta_h_maps_png": summary.get("all_reps_delta_h_maps_png") or summary.get("delta_h_map_png"),
        "processed_maps_png": summary.get("all_reps_delta_h_processed_maps_png") or summary.get("delta_h_processed_map_png"),
        "mspd_by_tau_png": summary.get("all_reps_mspd_by_tau_png") or summary.get("mspd_by_tau_png"),
    }
    for i, value in enumerate(per):
        row[f"rep_{i}_score"] = value
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render exact Flow-Lenia optimizer eval diagnostics for many optimization runs."
    )
    parser.add_argument("optimization_root", help="Directory containing run_000..run_008.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", default=None, help="Comma-separated run names; default all run_*.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--extra-range-start-steps", type=int, default=10000)
    parser.add_argument("--range-end-steps", type=int, default=300000)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--video-stride-steps", type=int, default=50)
    parser.add_argument("--video-max-steps", type=int, default=300000)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--video-resize-method", default="linear")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-extra-range",
        action="store_true",
        help="Only build config-range maps/video; useful for quick reruns.",
    )
    args = parser.parse_args(argv)

    opt_root = Path(args.optimization_root)
    out_root = Path(args.output_dir)
    run_dirs = _run_dirs(opt_root, args.runs, args.max_runs)
    renderer = Path(__file__).resolve().parent / "render_flowlenia_optimizer_eval_diagnostic.py"
    rows: list[dict[str, Any]] = []

    for i_run, run_dir in enumerate(run_dirs, start=1):
        print(f"[{i_run}/{len(run_dirs)}] {run_dir.name}", flush=True)
        base_out = out_root / run_dir.name / "range_config_50000_300000"
        base_summary_path = base_out / "summary.json"
        if args.skip_existing and base_summary_path.exists():
            print(f"  exists: {base_summary_path}", flush=True)
        else:
            cmd = [
                sys.executable,
                str(renderer),
                str(run_dir),
                "--source-root",
                str(args.source_root),
                "--output-dir",
                str(base_out),
                "--rep-index",
                "all",
                "--img-size",
                str(args.img_size),
                "--video-resize-method",
                str(args.video_resize_method),
                "--video-stride-steps",
                str(args.video_stride_steps),
                "--video-max-steps",
                str(args.video_max_steps),
                "--fps",
                str(args.fps),
                "--codec",
                str(args.codec),
                "--frame-batch-size",
                str(args.frame_batch_size),
            ]
            _run_command(cmd)
        rows.append(_summary_row(run_dir, "range_config_50000_300000", _load_summary(base_summary_path)))
        _write_csv(out_root / "batch_summary.csv", rows)

        if args.skip_extra_range:
            continue
        extra_label = f"range_{int(args.extra_range_start_steps)}_{int(args.range_end_steps)}"
        extra_out = out_root / run_dir.name / extra_label
        extra_summary_path = extra_out / "summary.json"
        if args.skip_existing and extra_summary_path.exists():
            print(f"  exists: {extra_summary_path}", flush=True)
        else:
            cmd = [
                sys.executable,
                str(renderer),
                str(run_dir),
                "--source-root",
                str(args.source_root),
                "--output-dir",
                str(extra_out),
                "--rep-index",
                "all",
                "--img-size",
                str(args.img_size),
                "--video-resize-method",
                str(args.video_resize_method),
                "--video-stride-steps",
                str(args.video_stride_steps),
                "--video-max-steps",
                str(args.video_max_steps),
                "--fps",
                str(args.fps),
                "--codec",
                str(args.codec),
                "--frame-batch-size",
                str(args.frame_batch_size),
                "--metric-range-start-steps",
                str(args.extra_range_start_steps),
                "--metric-range-end-steps",
                str(args.range_end_steps),
                "--skip-video",
            ]
            _run_command(cmd)
        rows.append(_summary_row(run_dir, extra_label, _load_summary(extra_summary_path)))
        _write_csv(out_root / "batch_summary.csv", rows)

    payload = {
        "optimization_root": str(opt_root),
        "source_root": str(args.source_root),
        "output_dir": str(out_root),
        "n_runs": len(run_dirs),
        "summary_csv": str(out_root / "batch_summary.csv"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "batch_summary.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
