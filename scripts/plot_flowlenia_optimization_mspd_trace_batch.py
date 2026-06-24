from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any


def _np():
    import numpy as np

    return np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _parse_root(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty root label in {spec!r}")
    return label, Path(path)


def _run_dirs(root: Path) -> dict[str, Path]:
    if (root / "pop_traj.pkl").exists():
        return {root.name: root}
    out = {p.name: p for p in sorted(root.glob("run_*")) if (p / "pop_traj.pkl").exists()}
    if not out:
        raise FileNotFoundError(f"No pop_traj.pkl or run_*/pop_traj.pkl found under {root}")
    return out


def _score_array(pop: dict[str, Any]):
    np = _np()
    if "objective_score" in pop:
        return np.asarray(pop["objective_score"], dtype=np.float64), "objective_score"
    if "objective_loss" in pop:
        return -np.asarray(pop["objective_loss"], dtype=np.float64), "-objective_loss"
    if "score_by_seed" in pop:
        return np.nanmean(np.asarray(pop["score_by_seed"], dtype=np.float64), axis=2), "mean(score_by_seed)"
    if "loss_by_seed" in pop:
        return -np.nanmean(np.asarray(pop["loss_by_seed"], dtype=np.float64), axis=2), "-mean(loss_by_seed)"
    if "loss" in pop:
        loss_kind = str(pop.get("loss_kind", "objective_loss"))
        if "rank" in loss_kind:
            raise KeyError(
                "pop_traj.pkl contains rank selection_fitness but no objective_score/objective_loss. "
                "Cannot plot MSPD from rank-only loss."
            )
        return -np.asarray(pop["loss"], dtype=np.float64), "-loss"
    raise KeyError("pop_traj.pkl must contain objective_score, objective_loss, score_by_seed, loss_by_seed, or loss.")


def _trace(run_dir: Path) -> dict[str, Any]:
    np = _np()
    pop = _load_pickle(run_dir / "pop_traj.pkl")
    score, source = _score_array(pop)
    if score.ndim != 2:
        raise ValueError(f"{run_dir}: expected score shape (iters, pop_size), got {score.shape}")
    it = np.arange(score.shape[0], dtype=np.int32)
    return {
        "run_dir": run_dir,
        "protocol": str(pop.get("selection_protocol", "mean_loss")),
        "source": str(source),
        "iter": it,
        "mean": np.nanmean(score, axis=1),
        "median": np.nanmedian(score, axis=1),
        "best": np.nanmax(score, axis=1),
        "std": np.nanstd(score, axis=1),
        "pop_size": int(score.shape[1]),
    }


def _plot_single(
    *,
    label: str,
    run_name: str,
    trace: dict[str, Any],
    output: Path,
    show_band: bool,
    show_best: bool,
    show_median: bool,
    dpi: int,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    it = trace["iter"]
    mean = trace["mean"]
    median = trace["median"]
    best = trace["best"]
    std = trace["std"]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "label": label,
            "run": run_name,
            "iter": int(i),
            "mean_mspd": float(mean[k]),
            "median_mspd": float(median[k]),
            "best_pop_mspd": float(best[k]),
            "std_pop_mspd": float(std[k]),
            "protocol": trace["protocol"],
            "source": trace["source"],
            "pop_size": int(trace["pop_size"]),
        }
        for k, i in enumerate(it)
    ]
    _write_csv(output.with_suffix(".csv"), rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(it, mean, color="#1f77b4", linewidth=2.0, label="population mean MSPD")
    if show_band:
        ax.fill_between(it, mean - std, mean + std, color="#1f77b4", alpha=0.16, linewidth=0, label="+/- pop std")
    if show_best:
        ax.plot(it, best, color="#d62728", linewidth=1.4, alpha=0.75, label="population best MSPD")
    if show_median:
        ax.plot(it, median, color="#2ca02c", linewidth=1.4, alpha=0.75, label="population median MSPD")
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("MSPD")
    ax.set_title(f"MSPD trace ({run_name}, {trace['protocol']}, source={trace['source']})")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)
    return rows


def _plot_compare(
    *,
    run_name: str,
    traces: dict[str, dict[str, Any]],
    output: Path,
    show_band: bool,
    show_best: bool,
    show_median: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#17becf",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for idx, (label, tr) in enumerate(traces.items()):
        color = colors[idx % len(colors)]
        it = tr["iter"]
        mean = tr["mean"]
        std = tr["std"]
        ax.plot(it, mean, color=color, linewidth=2.0, label=f"{label}: mean")
        if show_band:
            ax.fill_between(it, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        if show_best:
            ax.plot(it, tr["best"], color=color, linewidth=1.0, linestyle="--", alpha=0.62, label=f"{label}: best")
        if show_median:
            ax.plot(it, tr["median"], color=color, linewidth=1.0, linestyle=":", alpha=0.72, label=f"{label}: median")
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("MSPD")
    ax.set_title(f"MSPD protocol comparison ({run_name})")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Flow-Lenia optimization MSPD traces for multiple roots and compare matching run_XXX dirs."
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Label and optimization root as label=path. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--show-band", action="store_true")
    parser.add_argument("--show-best", action="store_true")
    parser.add_argument("--show-median", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    roots = [_parse_root(spec) for spec in args.root]
    out_dir = Path(args.output_dir)
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for label, root in roots:
        run_map = _run_dirs(root)
        by_label[label] = {}
        for run_name, run_dir in run_map.items():
            tr = _trace(run_dir)
            by_label[label][run_name] = tr
            rows = _plot_single(
                label=label,
                run_name=run_name,
                trace=tr,
                output=out_dir / "traces" / label / f"{run_name}_mspd_trace.png",
                show_band=bool(args.show_band),
                show_best=bool(args.show_best),
                show_median=bool(args.show_median),
                dpi=int(args.dpi),
            )
            all_rows.extend(rows)
            summary_rows.append(
                {
                    "label": label,
                    "run": run_name,
                    "run_dir": str(run_dir),
                    "protocol": tr["protocol"],
                    "source": tr["source"],
                    "n_iters": int(len(tr["iter"])),
                    "pop_size": int(tr["pop_size"]),
                    "final_mean_mspd": float(tr["mean"][-1]),
                    "max_mean_mspd": float(tr["mean"].max()),
                    "argmax_mean_iter": int(tr["iter"][int(tr["mean"].argmax())]),
                    "final_best_pop_mspd": float(tr["best"][-1]),
                    "max_best_pop_mspd": float(tr["best"].max()),
                }
            )

    common_runs = sorted(set.intersection(*(set(runs.keys()) for runs in by_label.values())))
    for run_name in common_runs:
        _plot_compare(
            run_name=run_name,
            traces={label: by_label[label][run_name] for label, _root in roots},
            output=out_dir / "comparisons" / f"{run_name}_mspd_protocol_comparison.png",
            show_band=bool(args.show_band),
            show_best=bool(args.show_best),
            show_median=bool(args.show_median),
            dpi=int(args.dpi),
        )

    _write_csv(out_dir / "all_trace_points.csv", all_rows)
    _write_csv(out_dir / "trace_summary.csv", summary_rows)
    print(
        {
            "output_dir": str(out_dir),
            "trace_summary_csv": str(out_dir / "trace_summary.csv"),
            "all_trace_points_csv": str(out_dir / "all_trace_points.csv"),
            "traces_dir": str(out_dir / "traces"),
            "comparisons_dir": str(out_dir / "comparisons"),
            "n_common_runs": len(common_runs),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
