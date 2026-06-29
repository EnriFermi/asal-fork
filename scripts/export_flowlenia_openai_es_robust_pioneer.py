from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def _score_matrix(pop: dict[str, Any]) -> tuple[np.ndarray, str]:
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
                "pop_traj.pkl contains rank-style loss but no objective_score/objective_loss/score_by_seed. "
                "Cannot recover MSPD scores."
            )
        return -np.asarray(pop["loss"], dtype=np.float64), "-loss"
    raise KeyError("pop_traj.pkl must contain objective_score, objective_loss, score_by_seed, loss_by_seed, or loss.")


def _score_by_seed(pop: dict[str, Any], fallback_score: np.ndarray) -> tuple[np.ndarray, str]:
    if "score_by_seed" in pop:
        return np.asarray(pop["score_by_seed"], dtype=np.float64), "score_by_seed"
    if "loss_by_seed" in pop:
        return -np.asarray(pop["loss_by_seed"], dtype=np.float64), "-loss_by_seed"
    return np.asarray(fallback_score, dtype=np.float64)[:, :, None], "score_matrix_as_single_seed"


def _trimmed_mean(values: np.ndarray, trim_frac: float) -> float:
    vals = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if vals.size == 0:
        return float("nan")
    k = int(np.floor(float(trim_frac) * vals.size))
    if vals.size - 2 * k >= 2:
        vals = vals[k : vals.size - k]
    return float(np.nanmean(vals))


def _ewma(values: np.ndarray, beta: float) -> np.ndarray:
    out: list[float] = []
    acc: float | None = None
    for raw in np.asarray(values, dtype=np.float64):
        if not np.isfinite(raw):
            out.append(float("nan"))
            continue
        acc = float(raw) if acc is None else float(beta) * acc + (1.0 - float(beta)) * float(raw)
        out.append(float(acc))
    return np.asarray(out, dtype=np.float64)


def _selected_tau_payload(pop: dict[str, Any], i_iter: int, pop_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("tau_selector_raw", "tau_idx", "tau_steps", "tau_frames"):
        if key not in pop:
            continue
        arr = np.asarray(pop[key])
        if arr.ndim >= 2 and i_iter < arr.shape[0] and pop_idx < arr.shape[1]:
            value = arr[i_iter, pop_idx]
            if np.asarray(value).size == 1:
                out[key] = float(value) if np.asarray(value).dtype.kind == "f" else int(value)
    return out


def _select_robust_pioneer(
    pop: dict[str, Any],
    *,
    lcb_z: float,
    trend_quantile: float,
    ewma_beta: float,
    trim_frac: float,
    min_iter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params = np.asarray(pop.get("params"), dtype=np.float32)
    if params.ndim != 3:
        raise ValueError(f"Expected pop_traj['params'] shape (iters, pop_size, n_params), got {params.shape}.")
    score, score_source = _score_matrix(pop)
    if score.shape != params.shape[:2]:
        raise ValueError(f"Score shape {score.shape} does not match params shape prefix {params.shape[:2]}.")
    seed_score, seed_score_source = _score_by_seed(pop, score)
    if seed_score.shape[:2] != params.shape[:2]:
        raise ValueError(f"Seed score shape {seed_score.shape} does not match params shape prefix {params.shape[:2]}.")

    with np.errstate(invalid="ignore"):
        iter_mean = np.nanmean(score, axis=1)
        iter_median = np.nanmedian(score, axis=1)
        iter_max = np.nanmax(score, axis=1)
        iter_std = np.nanstd(score, axis=1)
    iter_trimmed = np.asarray([_trimmed_mean(row, trim_frac) for row in score], dtype=np.float64)
    iter_trend = _ewma(iter_trimmed, ewma_beta)
    finite_trend = iter_trend[np.isfinite(iter_trend)]
    if finite_trend.size == 0:
        raise ValueError("No finite iteration trend values found.")
    trend_threshold = float(np.nanpercentile(finite_trend, float(trend_quantile)))
    iter_idx = np.arange(score.shape[0], dtype=np.int64)
    trend_gate = (iter_trend >= trend_threshold) & (iter_idx >= int(min_iter))
    if not np.any(trend_gate):
        trend_gate = np.isfinite(iter_trend) & (iter_idx >= int(min_iter))
    if not np.any(trend_gate):
        raise ValueError("No finite candidate iterations after robust trend gate.")

    seed_count = np.sum(np.isfinite(seed_score), axis=2)
    seed_mean = np.nanmean(seed_score, axis=2)
    seed_std = np.nanstd(seed_score, axis=2, ddof=1)
    seed_sem = seed_std / np.sqrt(np.maximum(seed_count, 1))
    seed_sem = np.where(seed_count > 1, seed_sem, 0.0)
    lcb = seed_mean - float(lcb_z) * seed_sem
    lcb = np.where(np.isfinite(lcb), lcb, np.nan)
    lcb[~trend_gate, :] = np.nan
    if not np.any(np.isfinite(lcb)):
        raise ValueError("No finite candidate LCB values after robust trend gate.")

    i_iter, pop_idx = np.unravel_index(np.nanargmax(lcb), lcb.shape)
    i_iter = int(i_iter)
    pop_idx = int(pop_idx)
    selected_seed_scores = np.asarray(seed_score[i_iter, pop_idx], dtype=np.float64)
    selected_score = float(seed_mean[i_iter, pop_idx])

    iter_rows = []
    for i in range(score.shape[0]):
        iter_rows.append(
            {
                "iter": int(i),
                "mean_score_mspd": float(iter_mean[i]),
                "median_score_mspd": float(iter_median[i]),
                "trimmed_mean_score_mspd": float(iter_trimmed[i]),
                "ewma_trimmed_mean_score_mspd": float(iter_trend[i]),
                "max_score_mspd": float(iter_max[i]),
                "std_score_mspd": float(iter_std[i]),
                "trend_gate": int(bool(trend_gate[i])),
            }
        )

    candidate_rows = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            row = {
                "iter": int(i),
                "pop_idx": int(j),
                "score_mspd": float(score[i, j]),
                "seed_mean_mspd": float(seed_mean[i, j]),
                "seed_std_mspd": float(seed_std[i, j]),
                "seed_sem_mspd": float(seed_sem[i, j]),
                "seed_lcb_mspd": float(seed_mean[i, j] - float(lcb_z) * seed_sem[i, j]),
                "seed_min_mspd": float(np.nanmin(seed_score[i, j])),
                "seed_max_mspd": float(np.nanmax(seed_score[i, j])),
                "trend_gate": int(bool(trend_gate[i])),
                "is_selected": int(i == i_iter and j == pop_idx),
            }
            for k, value in enumerate(np.asarray(seed_score[i, j]).reshape(-1)):
                row[f"seed_{k:02d}_mspd"] = float(value)
            candidate_rows.append(row)

    selected = {
        "selection_rule": "robust_pioneer_lcb_in_top_trend",
        "score_source": score_source,
        "seed_score_source": seed_score_source,
        "iter": i_iter,
        "pop_idx": pop_idx,
        "score_mspd": selected_score,
        "loss": -selected_score,
        "seed_lcb_mspd": float(seed_mean[i_iter, pop_idx] - float(lcb_z) * seed_sem[i_iter, pop_idx]),
        "seed_std_mspd": float(seed_std[i_iter, pop_idx]),
        "seed_sem_mspd": float(seed_sem[i_iter, pop_idx]),
        "seed_min_mspd": float(np.nanmin(selected_seed_scores)),
        "seed_max_mspd": float(np.nanmax(selected_seed_scores)),
        "iter_mean_score_mspd": float(iter_mean[i_iter]),
        "iter_median_score_mspd": float(iter_median[i_iter]),
        "iter_trimmed_mean_score_mspd": float(iter_trimmed[i_iter]),
        "iter_ewma_trimmed_mean_score_mspd": float(iter_trend[i_iter]),
        "iter_max_score_mspd": float(iter_max[i_iter]),
        "trend_threshold": trend_threshold,
        "trend_quantile": float(trend_quantile),
        "ewma_beta": float(ewma_beta),
        "trim_frac": float(trim_frac),
        "lcb_z": float(lcb_z),
        "min_iter": int(min_iter),
        "n_iters": int(params.shape[0]),
        "pop_size": int(params.shape[1]),
        "n_params": int(params.shape[2]),
        "seed_scores_mspd": [float(x) for x in selected_seed_scores.reshape(-1)],
        "tau": _selected_tau_payload(pop, i_iter, pop_idx),
        "params": np.asarray(params[i_iter, pop_idx], dtype=np.float32),
    }
    return selected, iter_rows, candidate_rows


def _existing_matches(path: Path, selected: dict[str, Any], run_dir: Path) -> bool:
    meta_path = path / "selected_candidate.json"
    best_path = path / "best.pkl"
    if not best_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return (
        str(meta.get("source_run_dir")) == str(run_dir)
        and int(meta.get("iter", -1)) == int(selected["iter"])
        and int(meta.get("pop_idx", -1)) == int(selected["pop_idx"])
        and str(meta.get("selection_rule")) == str(selected["selection_rule"])
        and abs(float(meta.get("lcb_z", float("nan"))) - float(selected["lcb_z"])) < 1e-12
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    pop_path = run_dir / "pop_traj.pkl"
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing {pop_path}.")
    output_dir = Path(args.output_dir)

    pop = _load_pickle(pop_path)
    selected, iter_rows, candidate_rows = _select_robust_pioneer(
        pop,
        lcb_z=float(args.lcb_z),
        trend_quantile=float(args.trend_quantile),
        ewma_beta=float(args.ewma_beta),
        trim_frac=float(args.trim_frac),
        min_iter=int(args.min_iter),
    )
    params = np.asarray(selected.pop("params"), dtype=np.float32)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        if _existing_matches(output_dir, selected, run_dir):
            summary_path = output_dir / "selected_candidate.json"
            print(summary_path.read_text())
            return json.loads(summary_path.read_text())
        raise FileExistsError(f"{output_dir} already exists and does not match this selection. Use --force or a fresh output dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    loss = np.asarray(float(selected["loss"]), dtype=np.float32)
    _save_pickle(output_dir / "best.pkl", (params, loss))
    np.save(output_dir / "params.npy", params)

    cfg_path = run_dir / "optimization_config.yaml"
    if cfg_path.exists():
        shutil.copy2(cfg_path, output_dir / "optimization_config.yaml")

    if selected["tau"]:
        _write_json(output_dir / "best_tau.json", selected["tau"])

    _write_csv(output_dir / "iteration_scores.csv", iter_rows)
    _write_csv(output_dir / "candidate_scores.csv", candidate_rows)

    meta = dict(selected)
    meta["source_run_dir"] = str(run_dir)
    meta["source_pop_traj"] = str(pop_path)
    meta["exported_checkpoint_dir"] = str(output_dir)
    meta["exported_best_pkl"] = str(output_dir / "best.pkl")
    meta["exported_params_npy"] = str(output_dir / "params.npy")
    _write_json(output_dir / "selected_candidate.json", meta)
    _write_json(
        output_dir / "best_selection.json",
        {
            "best_params_source": "openai_es_robust_pioneer_lcb_in_top_trend",
            "best_pkl_loss_kind": "negative_observed_seed_mean_mspd",
            **{k: v for k, v in meta.items() if k not in {"exported_params_npy"}},
        },
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a robust OpenAI-ES pioneer candidate from pop_traj.pkl.")
    parser.add_argument("run_dir", help="Optimization run directory containing pop_traj.pkl.")
    parser.add_argument("--output-dir", required=True, help="Output checkpoint directory to write best.pkl into.")
    parser.add_argument("--lcb-z", type=float, default=2.0, help="Penalty multiplier for score mean - z * SEM.")
    parser.add_argument("--trend-quantile", type=float, default=90.0, help="Keep only iterations above this EWMA-trend percentile.")
    parser.add_argument("--ewma-beta", type=float, default=0.85, help="EWMA beta for robust population trend.")
    parser.add_argument("--trim-frac", type=float, default=0.125, help="Population trim fraction for per-iteration robust mean.")
    parser.add_argument("--min-iter", type=int, default=0, help="Ignore iterations before this index.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing exported checkpoint directory.")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
