from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _score_array(pop: dict[str, Any]) -> tuple[np.ndarray, str]:
    import numpy as np

    if "objective_score" in pop:
        return np.asarray(pop["objective_score"], dtype=np.float64), "objective_score"
    if "loss" in pop:
        return -np.asarray(pop["loss"], dtype=np.float64), "-loss"
    raise KeyError("pop_traj.pkl must contain either objective_score or loss.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, str]:
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = Path(args.run_dir)
    pop_path = run_dir / "pop_traj.pkl"
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing {pop_path}")

    pop = _load_pickle(pop_path)
    score, source = _score_array(pop)
    if score.ndim != 2:
        raise ValueError(f"Expected score shape (iters, pop_size), got {score.shape}")

    it = np.arange(score.shape[0], dtype=np.int32)
    mean = np.nanmean(score, axis=1)
    median = np.nanmedian(score, axis=1)
    best = np.nanmax(score, axis=1)
    std = np.nanstd(score, axis=1)

    if args.output is None:
        output = run_dir / "figures" / "mean_mspd_by_iteration.png"
    else:
        output = Path(args.output)
    csv_path = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "iter": int(i),
            "mean_mspd": float(mean[k]),
            "median_mspd": float(median[k]),
            "best_pop_mspd": float(best[k]),
            "std_pop_mspd": float(std[k]),
        }
        for k, i in enumerate(it)
    ]
    _write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(it, mean, color="#1f77b4", linewidth=2.0, label="population mean MSPD")
    if args.show_band:
        ax.fill_between(it, mean - std, mean + std, color="#1f77b4", alpha=0.16, linewidth=0, label="+/- pop std")
    if args.show_best:
        ax.plot(it, best, color="#d62728", linewidth=1.4, alpha=0.75, label="population best MSPD")
    if args.show_median:
        ax.plot(it, median, color="#2ca02c", linewidth=1.4, alpha=0.75, label="population median MSPD")
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("MSPD")
    protocol = pop.get("selection_protocol", "mean_loss")
    ax.set_title(f"MSPD trace ({run_dir.name}, {protocol}, source={source})")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=int(args.dpi))
    plt.close(fig)

    return {"png": str(output), "csv": str(csv_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot population mean MSPD by optimization iteration from pop_traj.pkl.")
    parser.add_argument("run_dir", help="Optimization run directory containing pop_traj.pkl.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show-band", action="store_true", help="Also show +/- population std band.")
    parser.add_argument("--show-best", action="store_true", help="Also show best population MSPD per iteration.")
    parser.add_argument("--show-median", action="store_true", help="Also show median population MSPD per iteration.")
    args = parser.parse_args()
    paths = run(args)
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
