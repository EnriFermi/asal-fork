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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _select_best_mean_iter_median_candidate(pop: dict[str, Any]) -> dict[str, Any]:
    params = np.asarray(pop.get("params"), dtype=np.float32)
    if params.ndim != 3:
        raise ValueError(f"Expected pop_traj['params'] shape (iters, pop_size, n_params), got {params.shape}.")
    score, score_source = _score_matrix(pop)
    if score.shape != params.shape[:2]:
        raise ValueError(f"Score shape {score.shape} does not match params shape prefix {params.shape[:2]}.")

    with np.errstate(invalid="ignore"):
        iter_mean = np.nanmean(score, axis=1)
    if not np.any(np.isfinite(iter_mean)):
        raise ValueError("No finite per-iteration mean scores found in pop_traj.pkl.")
    i_iter = int(np.nanargmax(iter_mean))
    candidate_scores = np.asarray(score[i_iter], dtype=np.float64)
    finite_mask = np.isfinite(candidate_scores)
    if not np.any(finite_mask):
        raise ValueError(f"Best mean-score iteration {i_iter} has no finite candidate scores.")
    finite_indices = np.flatnonzero(finite_mask)
    finite_scores = candidate_scores[finite_indices]
    median_score = float(np.nanmedian(finite_scores))
    local_idx = int(np.argmin(np.abs(finite_scores - median_score)))
    pop_idx = int(finite_indices[local_idx])
    selected_score = float(candidate_scores[pop_idx])

    order = np.argsort(candidate_scores[finite_mask], kind="mergesort")
    finite_sorted_indices = finite_indices[order]
    rank = int(np.where(finite_sorted_indices == pop_idx)[0][0])
    percentile = float(rank / max(1, finite_sorted_indices.size - 1))

    return {
        "selection_rule": "best_mean_iter_median_candidate",
        "score_source": score_source,
        "iter": i_iter,
        "pop_idx": pop_idx,
        "score_mspd": selected_score,
        "loss": -selected_score,
        "iter_mean_score_mspd": float(iter_mean[i_iter]),
        "iter_median_score_mspd": median_score,
        "iter_best_score_mspd": float(np.nanmax(candidate_scores)),
        "iter_worst_score_mspd": float(np.nanmin(candidate_scores)),
        "iter_score_std": float(np.nanstd(candidate_scores)),
        "iter_score_rank_ascending": rank,
        "iter_score_percentile_ascending": percentile,
        "n_iters": int(params.shape[0]),
        "pop_size": int(params.shape[1]),
        "n_params": int(params.shape[2]),
        "params": np.asarray(params[i_iter, pop_idx], dtype=np.float32),
        "candidate_scores": candidate_scores,
        "tau": _selected_tau_payload(pop, i_iter, pop_idx),
    }


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
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    pop_path = run_dir / "pop_traj.pkl"
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing {pop_path}.")
    output_dir = Path(args.output_dir)

    pop = _load_pickle(pop_path)
    selected = _select_best_mean_iter_median_candidate(pop)
    params = np.asarray(selected.pop("params"), dtype=np.float32)
    candidate_scores = np.asarray(selected.pop("candidate_scores"), dtype=np.float64)

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

    rows = [
        {
            "iter": int(selected["iter"]),
            "pop_idx": int(i),
            "score_mspd": float(v),
            "is_selected": int(i == int(selected["pop_idx"])),
        }
        for i, v in enumerate(candidate_scores)
    ]
    _write_csv(output_dir / "candidate_scores_best_mean_iter.csv", rows)

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
            "best_params_source": "openai_es_best_mean_iter_median_candidate",
            "best_pkl_loss_kind": "negative_mspd_candidate_score",
            **{k: v for k, v in meta.items() if k not in {"exported_params_npy"}},
        },
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the median OpenAI-ES perturbation candidate from the best mean-score iteration.")
    parser.add_argument("run_dir", help="Optimization run directory containing pop_traj.pkl.")
    parser.add_argument("--output-dir", required=True, help="Output checkpoint directory to write best.pkl into.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing exported checkpoint directory.")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
