from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from flowlenia_minibang_common import load_config, resolve_path


def _load_manifest_rows(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("trajectories", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid manifest format: {manifest_path}")
        return manifest, rows

    rows = []
    for traj_dir in sorted(dataset_root.glob("traj_*")):
        if traj_dir.is_dir():
            rows.append(
                {
                    "traj_id": traj_dir.name,
                    "traj_dir": str(traj_dir),
                    "metrics_path": str(traj_dir / "metrics.npz"),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No manifest.json or traj_* dirs found in {dataset_root}")
    return {}, rows


def _candidate_paths(*paths: Any) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if raw is None or raw == "":
            continue
        path = Path(str(raw))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _traj_id(row: dict[str, Any]) -> str:
    if row.get("traj_id", None):
        return str(row["traj_id"])
    if row.get("traj_dir", None):
        return Path(str(row["traj_dir"])).name
    return f"traj_{int(row.get('selection_idx', 0)):05d}"


def _local_traj_dir(dataset_root: Path, row: dict[str, Any]) -> Path:
    traj_id = _traj_id(row)
    local = dataset_root / traj_id
    if local.exists():
        return local
    raw = row.get("traj_dir", None)
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
        mapped = dataset_root / path.name
        if mapped.exists():
            return mapped
    return local


def _metrics_path(dataset_root: Path, row: dict[str, Any]) -> tuple[Path | None, Path]:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        row.get("metrics_path", None),
        traj_dir / "metrics.npz",
        Path(str(row["traj_dir"])) / "metrics.npz" if row.get("traj_dir", None) else None,
    )
    existing = _first_existing(candidates)
    preferred = traj_dir / "metrics.npz"
    return existing, preferred


def _apf_dir(dataset_root: Path, row: dict[str, Any]) -> Path | None:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        row.get("apf_dir", None),
        traj_dir / "apf_logs",
        Path(str(row["traj_dir"])) / "apf_logs" if row.get("traj_dir", None) else None,
    )
    return _first_existing(candidates)


def _config_path(dataset_root: Path, row: dict[str, Any], manifest: dict[str, Any]) -> Path | None:
    traj_dir = _local_traj_dir(dataset_root, row)
    candidates = _candidate_paths(
        traj_dir / "config.yaml",
        Path(str(row["traj_dir"])) / "config.yaml" if row.get("traj_dir", None) else None,
        manifest.get("config_path", None),
    )
    return _first_existing(candidates)


def _npz_has_delta_h(path: Path) -> bool:
    try:
        with np.load(path) as data:
            return "delta_h_best" in data.files
    except Exception:
        return False


def _compute_delta_h_if_needed(
    *,
    dataset_root: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    metrics_path: Path | None,
    preferred_metrics_path: Path,
    metrics_seed: int,
    overwrite: bool,
) -> Path | None:
    if metrics_path is not None and _npz_has_delta_h(metrics_path) and not overwrite:
        return metrics_path

    apf_dir = _apf_dir(dataset_root, row)
    config_path = _config_path(dataset_root, row, manifest)
    if apf_dir is None or config_path is None:
        return metrics_path

    from flowlenia_minibang_simulate import _compute_delta_h_metrics

    _cfg, flat = load_config(config_path)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    if not isinstance(flat_args, dict):
        raise ValueError(f"Could not flatten config: {config_path}")

    selection_idx = int(row.get("selection_idx", row.get("traj_selection_idx", 0)))
    seed = int(metrics_seed) + selection_idx + 31
    computed = _compute_delta_h_metrics(apf_dir, flat_args, seed=seed)

    out_path = metrics_path if metrics_path is not None else preferred_metrics_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Any] = {}
    if out_path.exists() and not overwrite:
        with np.load(out_path) as old:
            merged.update({key: old[key] for key in old.files})
    merged.update(computed)
    np.savez_compressed(out_path, **merged)
    return out_path


def _load_delta_h(metrics_path: Path) -> dict[str, Any]:
    with np.load(metrics_path) as data:
        if "delta_h_best" not in data.files:
            raise KeyError(f"{metrics_path} does not contain delta_h_best")
        y = np.asarray(data["delta_h_best"], dtype=np.float64).reshape(-1)
        if "delta_h_window_center_steps" in data.files:
            x = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        elif "delta_h_window_start_steps" in data.files and "delta_h_window_end_steps" in data.files:
            s0 = np.asarray(data["delta_h_window_start_steps"], dtype=np.float64).reshape(-1)
            s1 = np.asarray(data["delta_h_window_end_steps"], dtype=np.float64).reshape(-1)
            x = 0.5 * (s0 + s1)
        else:
            x = np.arange(y.size, dtype=np.float64)
        if x.size != y.size:
            x = np.arange(y.size, dtype=np.float64)
        selected_tau = None
        if "delta_h_selected_tau_steps" in data.files:
            selected_tau = int(np.asarray(data["delta_h_selected_tau_steps"]).reshape(-1)[0])
    return {"steps": x, "delta_h": y, "selected_tau": selected_tau}


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _plot_one(
    *,
    plt: Any,
    row: dict[str, Any],
    series: dict[str, Any],
    out_path: Path,
    detect_start_step: int | None,
    detect_end_step: int | None,
    yscale: str,
) -> dict[str, Any]:
    traj_id = _traj_id(row)
    steps = np.asarray(series["steps"], dtype=np.float64)
    dh = np.asarray(series["delta_h"], dtype=np.float64)
    finite = np.isfinite(steps) & np.isfinite(dh)
    steps_f = steps[finite]
    dh_f = dh[finite]
    if dh_f.size == 0:
        raise ValueError(f"No finite deltaH points for {traj_id}")

    max_i = int(np.nanargmax(dh_f))
    max_step = float(steps_f[max_i])
    max_dh = float(dh_f[max_i])
    mean_dh = float(np.nanmean(dh_f))

    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.plot(steps_f, dh_f, color="#1f77b4", linewidth=1.8)
    ax.scatter([max_step], [max_dh], color="#d62728", s=34, zorder=3, label=f"max {max_dh:.4g}")
    if detect_start_step is not None:
        ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if detect_end_step is not None:
        ax.axvline(int(detect_end_step), color="#666666", linestyle=":", linewidth=1.0, alpha=0.65)
    if yscale == "log" and np.nanmin(dh_f) > 0.0:
        ax.set_yscale("log")

    subtitle = []
    for key, label in (("loss", "loss"), ("iter", "iter"), ("saturation_T", "T")):
        if row.get(key, None) not in (None, ""):
            value = row[key]
            if isinstance(value, float):
                subtitle.append(f"{label}={value:.4g}")
            else:
                subtitle.append(f"{label}={value}")
    if series.get("selected_tau", None) is not None:
        subtitle.append(f"tau={series['selected_tau']}")

    ax.set_title(f"{traj_id} deltaH" + (f" ({', '.join(subtitle)})" if subtitle else ""))
    ax.set_xlabel("simulation step")
    ax.set_ylabel("deltaH best")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "traj_id": traj_id,
        "status": "ok",
        "plot_path": str(out_path),
        "n_points": int(dh_f.size),
        "step_min": float(np.nanmin(steps_f)),
        "step_max": float(np.nanmax(steps_f)),
        "delta_h_max": max_dh,
        "delta_h_max_step": max_step,
        "delta_h_mean": mean_dh,
        "delta_h_selected_tau_steps": series.get("selected_tau", ""),
    }


def _plot_grid(
    *,
    plt: Any,
    plotted: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: Path,
    detect_start_step: int | None,
    yscale: str,
) -> None:
    if not plotted:
        return
    cols = 4
    rows_n = int(math.ceil(len(plotted) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.7 * rows_n), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (row, series) in zip(axes_arr, plotted):
        steps = np.asarray(series["steps"], dtype=np.float64)
        dh = np.asarray(series["delta_h"], dtype=np.float64)
        finite = np.isfinite(steps) & np.isfinite(dh)
        steps_f = steps[finite]
        dh_f = dh[finite]
        ax.plot(steps_f, dh_f, color="#1f77b4", linewidth=1.1)
        if dh_f.size:
            max_i = int(np.nanargmax(dh_f))
            ax.scatter([steps_f[max_i]], [dh_f[max_i]], color="#d62728", s=12, zorder=3)
        if detect_start_step is not None:
            ax.axvline(int(detect_start_step), color="#666666", linestyle="--", linewidth=0.8, alpha=0.55)
        if yscale == "log" and dh_f.size and np.nanmin(dh_f) > 0.0:
            ax.set_yscale("log")
        ax.set_title(_traj_id(row), fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(plotted) :]:
        ax.axis("off")
    fig.suptitle("deltaH by trajectory", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "traj_id",
        "status",
        "plot_path",
        "metrics_path",
        "n_points",
        "step_min",
        "step_max",
        "delta_h_max",
        "delta_h_max_step",
        "delta_h_mean",
        "delta_h_selected_tau_steps",
        "loss",
        "iter",
        "saturation_T",
        "source",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_index(path: Path, rows: list[dict[str, Any]], grid_path: Path) -> None:
    lines = ["# DeltaH Plots", ""]
    if grid_path.exists():
        lines.extend([f"![overview]({grid_path.name})", ""])
    lines.append("| traj | status | max step | max deltaH | mean deltaH | plot |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in rows:
        plot = Path(str(row.get("plot_path", ""))).name if row.get("plot_path", "") else ""
        plot_link = f"[png]({plot})" if plot else ""
        lines.append(
            "| {traj} | {status} | {step} | {maxv} | {meanv} | {plot} |".format(
                traj=row.get("traj_id", ""),
                status=row.get("status", ""),
                step=_fmt_num(row.get("delta_h_max_step", "")),
                maxv=_fmt_num(row.get("delta_h_max", "")),
                meanv=_fmt_num(row.get("delta_h_mean", "")),
                plot=plot_link,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt_num(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot deltaH curves for FlowLenia minibang trajectory datasets.")
    parser.add_argument("dataset_root", help="Dataset root with manifest.json or traj_* directories.")
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/delta_h_plots.")
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="If metrics.npz is missing delta_h_best, compute deltaH from apf_logs/*.npz first.",
    )
    parser.add_argument(
        "--overwrite-computed",
        action="store_true",
        help="With --compute-missing, recompute deltaH even if metrics.npz already has delta_h_best.",
    )
    parser.add_argument("--metrics-seed", type=int, default=12345, help="Base seed for recomputing deltaH.")
    parser.add_argument("--start-step", type=int, default=None, help="Optional vertical marker. Defaults to manifest.")
    parser.add_argument("--end-step", type=int, default=None, help="Optional vertical marker. Defaults to manifest.")
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    output_dir = resolve_path(args.output_dir, dataset_root) if args.output_dir else dataset_root / "delta_h_plots"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl_cache = Path(tempfile.gettempdir()) / "flowlenia_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, rows = _load_manifest_rows(dataset_root)
    detect_start_step = args.start_step if args.start_step is not None else _as_optional_int(manifest.get("detect_start_step"))
    detect_end_step = args.end_step if args.end_step is not None else _as_optional_int(manifest.get("detect_end_step"))

    summary_rows: list[dict[str, Any]] = []
    plotted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        traj_id = _traj_id(row)
        metrics_path, preferred_metrics_path = _metrics_path(dataset_root, row)
        try:
            if args.compute_missing:
                metrics_path = _compute_delta_h_if_needed(
                    dataset_root=dataset_root,
                    row=row,
                    manifest=manifest,
                    metrics_path=metrics_path,
                    preferred_metrics_path=preferred_metrics_path,
                    metrics_seed=int(args.metrics_seed),
                    overwrite=bool(args.overwrite_computed),
                )
            if metrics_path is None or not metrics_path.exists():
                raise FileNotFoundError("metrics.npz not found")
            series = _load_delta_h(metrics_path)
            plot_path = output_dir / f"{traj_id}_delta_h.png"
            summary = _plot_one(
                plt=plt,
                row=row,
                series=series,
                out_path=plot_path,
                detect_start_step=detect_start_step,
                detect_end_step=detect_end_step,
                yscale=str(args.yscale),
            )
            summary.update(
                metrics_path=str(metrics_path),
                loss=row.get("loss", ""),
                iter=row.get("iter", ""),
                saturation_T=row.get("saturation_T", ""),
                source=row.get("source", ""),
                message="",
            )
            summary_rows.append(summary)
            plotted.append((row, series))
        except Exception as exc:
            summary_rows.append(
                {
                    "traj_id": traj_id,
                    "status": "missing_or_failed",
                    "metrics_path": str(metrics_path or preferred_metrics_path),
                    "loss": row.get("loss", ""),
                    "iter": row.get("iter", ""),
                    "saturation_T": row.get("saturation_T", ""),
                    "source": row.get("source", ""),
                    "message": str(exc),
                }
            )
            print(f"[{traj_id}] skipped: {exc}")

    grid_path = output_dir / "delta_h_grid.png"
    _plot_grid(
        plt=plt,
        plotted=plotted,
        out_path=grid_path,
        detect_start_step=detect_start_step,
        yscale=str(args.yscale),
    )
    _write_csv(output_dir / "delta_h_summary.csv", summary_rows)
    _write_index(output_dir / "index.md", summary_rows, grid_path)
    print(f"Wrote {len(plotted)} deltaH plots to {output_dir}")
    if len(plotted) != len(rows):
        print(f"Skipped {len(rows) - len(plotted)} trajectories; see delta_h_summary.csv")


if __name__ == "__main__":
    main()
