from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _np():
    import numpy as np

    return np


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _run_dirs(root: Path, run_name: str | None, max_runs: int | None) -> list[Path]:
    if run_name:
        run_dir = root / run_name
        if not (run_dir / "pop_traj.pkl").exists():
            raise FileNotFoundError(f"Missing {run_dir / 'pop_traj.pkl'}")
        return [run_dir]
    dirs = [p for p in sorted(root.glob("run_*")) if (p / "pop_traj.pkl").exists()]
    if max_runs is not None:
        dirs = dirs[: int(max_runs)]
    if not dirs:
        raise FileNotFoundError(f"No run_*/pop_traj.pkl found under {root}")
    return dirs


def _score_matrix(pop: dict[str, Any]) -> tuple[np.ndarray, str]:
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
                "pop_traj.pkl has rank-style loss but no objective_score/objective_loss/score_by_seed. "
                "Cannot recover MSPD from this file."
            )
        return -np.asarray(pop["loss"], dtype=np.float64), "-loss"
    raise KeyError("pop_traj.pkl must contain objective_score, objective_loss, score_by_seed, loss_by_seed, or loss.")


def _percentile_rank(values_sorted: np.ndarray, value: float) -> float:
    np = _np()
    if values_sorted.size <= 1:
        return 1.0
    # Right side treats ties optimistically; these scores are float and ties are rare.
    return float(np.searchsorted(values_sorted, value, side="right") / values_sorted.size)


def _tau_value(pop: dict[str, Any], key: str, i_iter: int, pop_idx: int) -> Any:
    np = _np()
    if key not in pop:
        return ""
    arr = np.asarray(pop[key])
    if arr.shape[:2] != (np.asarray(pop["params"]).shape[0], np.asarray(pop["params"]).shape[1]):
        return ""
    val = arr[i_iter, pop_idx]
    if np.issubdtype(arr.dtype, np.integer):
        return int(val)
    try:
        return float(val)
    except Exception:
        return str(val)


def _cutoff(flat_scores: np.ndarray, *, q: float, iqr_multiplier: float, mode: str) -> dict[str, float]:
    np = _np()
    q25, q50, q75, q95, q99, q_user = np.nanquantile(flat_scores, [0.25, 0.50, 0.75, 0.95, 0.99, q])
    iqr = float(q75 - q25)
    tukey = float(q75 + iqr_multiplier * iqr) if np.isfinite(iqr) else float(q_user)
    if mode == "quantile":
        cutoff = float(q_user)
    elif mode == "tukey":
        cutoff = tukey
    elif mode == "min":
        cutoff = float(min(q_user, tukey))
    else:
        raise ValueError(f"Unknown cutoff mode {mode!r}.")
    return {
        "marginal_q25": float(q25),
        "marginal_q50": float(q50),
        "marginal_q75": float(q75),
        "marginal_q95": float(q95),
        "marginal_q99": float(q99),
        "marginal_q_user": float(q_user),
        "marginal_upper_fence": float(tukey),
        "marginal_outlier_cutoff": float(cutoff),
    }


def _candidate_rows_for_run(
    run_dir: Path,
    *,
    top_per_step_frac: float,
    marginal_max_quantile: float,
    iqr_multiplier: float,
    cutoff_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    np = _np()
    pop = _load_pickle(run_dir / "pop_traj.pkl")
    if not isinstance(pop, dict) or "params" not in pop:
        raise ValueError(f"{run_dir / 'pop_traj.pkl'} must contain a dict with params.")
    params = np.asarray(pop["params"])
    score, score_source = _score_matrix(pop)
    if score.ndim != 2:
        raise ValueError(f"{run_dir}: expected score shape (n_iters, pop_size), got {score.shape}")
    if params.shape[:2] != score.shape:
        raise ValueError(f"{run_dir}: params shape {params.shape[:2]} does not match score shape {score.shape}")

    finite = np.isfinite(score)
    flat_scores = score[finite]
    if flat_scores.size == 0:
        raise ValueError(f"{run_dir}: no finite MSPD scores.")
    sorted_flat = np.sort(flat_scores)
    cutoff_info = _cutoff(
        flat_scores,
        q=float(marginal_max_quantile),
        iqr_multiplier=float(iqr_multiplier),
        mode=str(cutoff_mode),
    )
    pop_size = int(score.shape[1])
    top_k = max(1, int(math.ceil(pop_size * float(top_per_step_frac))))

    rows: list[dict[str, Any]] = []
    for i_iter in range(score.shape[0]):
        step_scores = np.asarray(score[i_iter], dtype=np.float64)
        step_finite = np.isfinite(step_scores)
        if not np.any(step_finite):
            continue
        finite_step_scores = step_scores[step_finite]
        order = np.argsort(-step_scores, kind="mergesort")
        ranks = np.full(pop_size, -1, dtype=np.int32)
        rank = 1
        for idx in order.tolist():
            if np.isfinite(step_scores[idx]):
                ranks[idx] = rank
                rank += 1
        step_mean = float(np.nanmean(finite_step_scores))
        step_std = float(np.nanstd(finite_step_scores))
        step_q25, step_median, step_q75 = np.nanquantile(finite_step_scores, [0.25, 0.5, 0.75])
        step_iqr = float(step_q75 - step_q25)
        for pop_idx in range(pop_size):
            val = float(step_scores[pop_idx])
            if not np.isfinite(val):
                continue
            per_step_rank_1 = int(ranks[pop_idx])
            per_step_percentile = (
                1.0
                if pop_size <= 1
                else float((pop_size - per_step_rank_1) / max(pop_size - 1, 1))
            )
            marginal_percentile = _percentile_rank(sorted_flat, val)
            per_step_z = float((val - step_mean) / step_std) if step_std > 0 else 0.0
            per_step_robust_z = float((val - step_median) / step_iqr) if step_iqr > 0 else 0.0
            is_per_step_top = per_step_rank_1 <= top_k
            is_marginal_outlier = val > cutoff_info["marginal_outlier_cutoff"]
            passes_filter = bool(is_per_step_top and not is_marginal_outlier)
            row = {
                "run": run_dir.name,
                "run_dir": str(run_dir),
                "iter": int(i_iter),
                "pop_idx": int(pop_idx),
                "score_mspd": val,
                "objective_loss": -val,
                "score_source": score_source,
                "per_step_rank_1": per_step_rank_1,
                "per_step_top_k": int(top_k),
                "per_step_percentile": per_step_percentile,
                "per_step_mean": step_mean,
                "per_step_std": step_std,
                "per_step_median": float(step_median),
                "per_step_iqr": step_iqr,
                "per_step_z": per_step_z,
                "per_step_robust_z": per_step_robust_z,
                "marginal_percentile": marginal_percentile,
                "is_per_step_top": int(is_per_step_top),
                "is_marginal_outlier": int(is_marginal_outlier),
                "passes_filter": int(passes_filter),
                "selection_score": val if passes_filter else float("-inf"),
                "tau_selector_raw": _tau_value(pop, "tau_selector_raw", i_iter, pop_idx),
                "tau_idx": _tau_value(pop, "tau_idx", i_iter, pop_idx),
                "tau_steps": _tau_value(pop, "tau_steps", i_iter, pop_idx),
                "tau_frames": _tau_value(pop, "tau_frames", i_iter, pop_idx),
            }
            row.update(cutoff_info)
            rows.append(row)

    passing = [r for r in rows if int(r["passes_filter"]) == 1]
    if passing:
        selected = max(passing, key=lambda r: (float(r["score_mspd"]), float(r["per_step_percentile"])))
        selection_status = "per_step_top_non_marginal_outlier"
    else:
        non_outliers = [r for r in rows if int(r["is_marginal_outlier"]) == 0]
        if non_outliers:
            selected = max(non_outliers, key=lambda r: (float(r["per_step_percentile"]), float(r["score_mspd"])))
            selection_status = "fallback_non_marginal_outlier_best_per_step_percentile"
        else:
            selected = max(rows, key=lambda r: (float(r["per_step_percentile"]), -float(r["marginal_percentile"])))
            selection_status = "fallback_all_scores_marginal_outliers"
    selected = dict(selected)
    selected["selection_status"] = selection_status

    raw_best = max(rows, key=lambda r: float(r["score_mspd"]))
    summary = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "n_iters": int(score.shape[0]),
        "pop_size": int(score.shape[1]),
        "n_scores": int(flat_scores.size),
        "score_source": score_source,
        "top_per_step_frac": float(top_per_step_frac),
        "per_step_top_k": int(top_k),
        "marginal_max_quantile": float(marginal_max_quantile),
        "iqr_multiplier": float(iqr_multiplier),
        "cutoff_mode": str(cutoff_mode),
        "n_passing": int(len(passing)),
        "selected_iter": int(selected["iter"]),
        "selected_pop_idx": int(selected["pop_idx"]),
        "selected_score_mspd": float(selected["score_mspd"]),
        "selected_marginal_percentile": float(selected["marginal_percentile"]),
        "selected_per_step_rank_1": int(selected["per_step_rank_1"]),
        "raw_best_iter": int(raw_best["iter"]),
        "raw_best_pop_idx": int(raw_best["pop_idx"]),
        "raw_best_score_mspd": float(raw_best["score_mspd"]),
        "raw_best_marginal_percentile": float(raw_best["marginal_percentile"]),
        "raw_best_is_marginal_outlier": int(raw_best["is_marginal_outlier"]),
    }
    summary.update(cutoff_info)
    return rows, selected, summary, pop


def _plot_run(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    np = _np()
    scores = np.asarray([float(r["score_mspd"]) for r in rows], dtype=np.float64)
    iters = np.asarray([int(r["iter"]) for r in rows], dtype=np.int32)
    ranks = np.asarray([int(r["per_step_rank_1"]) for r in rows], dtype=np.int32)
    passing = np.asarray([int(r["passes_filter"]) == 1 for r in rows], dtype=bool)
    outlier = np.asarray([int(r["is_marginal_outlier"]) == 1 for r in rows], dtype=bool)
    cutoff = float(selected["marginal_outlier_cutoff"])
    q95 = float(selected["marginal_q95"])
    q99 = float(selected["marginal_q99"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.scatter(iters[~outlier], scores[~outlier], s=16, alpha=0.42, color="#4C78A8", linewidths=0, label="population")
    if np.any(outlier):
        ax.scatter(iters[outlier], scores[outlier], s=18, alpha=0.55, color="#B279A2", linewidths=0, label="marginal outlier")
    if np.any(passing):
        ax.scatter(iters[passing], scores[passing], s=28, alpha=0.85, facecolors="none", edgecolors="#F58518", linewidths=1.0, label="eligible")
    ax.scatter(
        [int(selected["iter"])],
        [float(selected["score_mspd"])],
        s=86,
        marker="*",
        color="#E45756",
        edgecolors="black",
        linewidths=0.6,
        label="selected",
        zorder=5,
    )
    ax.axhline(cutoff, color="#E45756", linewidth=1.2, linestyle="--", alpha=0.8, label="marginal cutoff")
    ax.set_title(f"{run_dir.name}: per-step population scores")
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("MSPD")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.hist(scores, bins=40, color="#72B7B2", alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.axvline(q95, color="#54A24B", linewidth=1.1, linestyle=":", label="q95")
    ax.axvline(q99, color="#B279A2", linewidth=1.1, linestyle=":", label="q99")
    ax.axvline(cutoff, color="#E45756", linewidth=1.4, linestyle="--", label="outlier cutoff")
    ax.axvline(float(selected["score_mspd"]), color="black", linewidth=1.4, label="selected")
    raw_best = float(np.nanmax(scores))
    ax.axvline(raw_best, color="#F58518", linewidth=1.1, linestyle="-.", label="raw max")
    ax.set_title("marginal MSPD distribution")
    ax.set_xlabel("MSPD")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"selected iter={int(selected['iter'])}, pop={int(selected['pop_idx'])}, "
        f"rank={int(selected['per_step_rank_1'])}, MSPD={float(selected['score_mspd']):.6g}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)


def _plot_summary(selected_rows: list[dict[str, Any]], output_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    np = _np()
    if not selected_rows:
        return
    runs = [str(r["run"]) for r in selected_rows]
    selected_scores = np.asarray([float(r["score_mspd"]) for r in selected_rows], dtype=np.float64)
    raw_scores = np.asarray([float(r["raw_best_score_mspd"]) for r in selected_rows], dtype=np.float64)
    cutoff = np.asarray([float(r["marginal_outlier_cutoff"]) for r in selected_rows], dtype=np.float64)
    x = np.arange(len(runs))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.plot(x, raw_scores, marker="o", linewidth=1.3, color="#B279A2", label="raw marginal max")
    ax.plot(x, selected_scores, marker="o", linewidth=1.8, color="#E45756", label="selected robust candidate")
    ax.plot(x, cutoff, marker=".", linewidth=1.0, linestyle="--", color="#4C78A8", label="marginal cutoff")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=35, ha="right")
    ax.set_ylabel("MSPD")
    ax.set_title("Robust MSPD candidate selection by run")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)


def _export_selected_checkpoint(
    *,
    run_dir: Path,
    pop: dict[str, Any],
    selected: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    np = _np()
    params = np.asarray(pop["params"], dtype=np.float32)
    i_iter = int(selected["iter"])
    pop_idx = int(selected["pop_idx"])
    selected_params = np.asarray(params[i_iter, pop_idx], dtype=np.float32)
    selected_dir = output_root / run_dir.name
    selected_dir.mkdir(parents=True, exist_ok=True)

    loss = np.asarray(-float(selected["score_mspd"]), dtype=np.float32)
    _save_pickle(selected_dir / "best.pkl", (selected_params, loss))
    np.save(selected_dir / "params.npy", selected_params)

    cfg_path = run_dir / "optimization_config.yaml"
    if cfg_path.exists():
        shutil.copy2(cfg_path, selected_dir / "optimization_config.yaml")

    tau_payload: dict[str, Any] = {}
    for key in ("tau_selector_raw", "tau_idx", "tau_steps", "tau_frames"):
        val = selected.get(key, "")
        if val != "":
            tau_payload[key] = val
    if tau_payload:
        _write_json(selected_dir / "best_tau.json", tau_payload)

    meta = dict(selected)
    meta["source_run_dir"] = str(run_dir)
    meta["exported_best_pkl"] = str(selected_dir / "best.pkl")
    meta["exported_params_npy"] = str(selected_dir / "params.npy")
    _write_json(selected_dir / "selected_candidate.json", meta)
    return {
        "selected_checkpoint_dir": str(selected_dir),
        "selected_best_pkl": str(selected_dir / "best.pkl"),
        "selected_params_npy": str(selected_dir / "params.npy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select robust Flow-Lenia MSPD optimization candidates: high within-generation "
            "MSPD, but not in the high tail of the marginal MSPD distribution."
        )
    )
    parser.add_argument(
        "optimization_root",
        help="Root with run_*/pop_traj.pkl, e.g. experiments/paper_check_flow_lenia/checkpoints_lockheed_1/optimization",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results/flowlenia_lockheed_1_robust_mspd_candidate_selection",
        help="Directory for CSVs, plots, and selected checkpoint exports.",
    )
    parser.add_argument("--run", default=None, help="Only process one run name, e.g. run_004.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--top-per-step-frac", type=float, default=0.25)
    parser.add_argument("--marginal-max-quantile", type=float, default=0.99)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--cutoff-mode", choices=["quantile", "tukey", "min"], default="min")
    parser.add_argument("--top-candidates-per-run", type=int, default=25)
    parser.add_argument("--no-export-checkpoints", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    opt_root = Path(args.optimization_root)
    out_dir = Path(args.output_dir)
    run_dirs = _run_dirs(opt_root, args.run, args.max_runs)

    all_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        rows, selected, summary, pop = _candidate_rows_for_run(
            run_dir,
            top_per_step_frac=float(args.top_per_step_frac),
            marginal_max_quantile=float(args.marginal_max_quantile),
            iqr_multiplier=float(args.iqr_multiplier),
            cutoff_mode=str(args.cutoff_mode),
        )
        selected_with_summary = dict(selected)
        selected_with_summary.update(
            {
                "raw_best_score_mspd": summary["raw_best_score_mspd"],
                "raw_best_iter": summary["raw_best_iter"],
                "raw_best_pop_idx": summary["raw_best_pop_idx"],
                "raw_best_marginal_percentile": summary["raw_best_marginal_percentile"],
                "raw_best_is_marginal_outlier": summary["raw_best_is_marginal_outlier"],
                "n_passing": summary["n_passing"],
            }
        )
        if not args.no_export_checkpoints:
            selected_with_summary.update(
                _export_selected_checkpoint(
                    run_dir=run_dir,
                    pop=pop,
                    selected=selected_with_summary,
                    output_root=out_dir / "selected_checkpoints",
                )
            )

        eligible_sorted = sorted(
            rows,
            key=lambda r: (
                int(r["passes_filter"]),
                float(r["score_mspd"]),
                float(r["per_step_percentile"]),
            ),
            reverse=True,
        )
        top_rows.extend(eligible_sorted[: int(args.top_candidates_per_run)])
        all_rows.extend(rows)
        selected_rows.append(selected_with_summary)
        summary_rows.append(summary)

        _plot_run(
            run_dir=run_dir,
            rows=rows,
            selected=selected_with_summary,
            output_path=out_dir / "figures" / f"{run_dir.name}_mspd_candidate_distributions.png",
            dpi=int(args.dpi),
        )

    _write_csv(out_dir / "candidate_scores.csv", all_rows)
    _write_csv(out_dir / "top_candidates.csv", top_rows)
    _write_csv(out_dir / "selected_candidates.csv", selected_rows)
    _write_csv(out_dir / "run_summaries.csv", summary_rows)
    _write_json(
        out_dir / "summary.json",
        {
            "optimization_root": str(opt_root),
            "output_dir": str(out_dir),
            "n_runs": len(run_dirs),
            "top_per_step_frac": float(args.top_per_step_frac),
            "marginal_max_quantile": float(args.marginal_max_quantile),
            "iqr_multiplier": float(args.iqr_multiplier),
            "cutoff_mode": str(args.cutoff_mode),
            "selected_candidates_csv": str(out_dir / "selected_candidates.csv"),
            "top_candidates_csv": str(out_dir / "top_candidates.csv"),
            "candidate_scores_csv": str(out_dir / "candidate_scores.csv"),
        },
    )
    _plot_summary(selected_rows, out_dir / "figures" / "selected_candidates_summary.png", dpi=int(args.dpi))

    print(json.dumps({
        "selected_candidates_csv": str(out_dir / "selected_candidates.csv"),
        "top_candidates_csv": str(out_dir / "top_candidates.csv"),
        "candidate_scores_csv": str(out_dir / "candidate_scores.csv"),
        "figures_dir": str(out_dir / "figures"),
        "selected_checkpoints_dir": str(out_dir / "selected_checkpoints"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
