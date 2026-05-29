from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from paper_suite_common import ensure_dir, log_event, resolve_path, write_csv, write_json


TRAJECTORY_CANDIDATES = (
    "trajectory_id",
    "source_trajectory_id",
    "source_traj_id",
    "traj_id",
    "source_id",
    "run_id",
    "trial_uid",
    "trial_id",
    "optimized_run_id",
    "optimized_run_idx",
)
TIME_CANDIDATES = (
    "branch_t",
    "branch_time",
    "time",
    "t",
    "step",
    "branch_step",
    "window_center_step",
    "requested_step",
)
ENERGY_CANDIDATES = (
    "delta_h_energy",
    "branch_energy",
    "delta_h",
    "delta_h_at_branch_time",
    "energy",
    "mspd_energy",
)
DIVERGENCE_CANDIDATES = (
    "branch_divergence",
    "future_branch_divergence",
    "future_divergence",
    "clip_chamfer",
    "clip_chamfer_divergence",
    "branching_score",
    "divergence",
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _resolve_column(headers: list[str], explicit: str | None, candidates: tuple[str, ...], label: str, *, required: bool = True) -> str | None:
    if explicit:
        if explicit in headers:
            return explicit
        lower_to_header = {h.lower(): h for h in headers}
        if explicit.lower() in lower_to_header:
            return lower_to_header[explicit.lower()]
        raise ValueError(f"Requested {label} column {explicit!r} not found. Available columns: {headers}")
    lower_to_header = {h.lower(): h for h in headers}
    for candidate in candidates:
        if candidate.lower() in lower_to_header:
            return lower_to_header[candidate.lower()]
    if required:
        raise ValueError(f"Could not infer {label} column. Pass it explicitly. Available columns: {headers}")
    return None


def _safe_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sample_std(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[finite]
    yy = yy[finite]
    if xx.size < 2:
        return float("nan")
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = int(arr.size)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_arr = arr[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_arr[end] == sorted_arr[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[finite]
    yy = yy[finite]
    if xx.size < 2:
        return float("nan")
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return _pearson(_rankdata_average(xx), _rankdata_average(yy))


def _corr_summary(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(y)
    return {
        "n": int(np.sum(finite)),
        "pearson_r": _pearson(x, y),
        "spearman_r": _spearman(x, y),
    }


def _zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(arr))
    std = _sample_std(arr)
    return (arr - mean) / std, mean, std


def _permutation_pvalue(
    z_energy: np.ndarray,
    z_divergence: np.ndarray,
    group_indices: list[np.ndarray],
    observed: float,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    n_perm = int(n_permutations)
    if n_perm <= 0 or not np.isfinite(observed):
        return {
            "n_permutations": int(max(0, n_perm)),
            "p_value_positive": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    null = np.empty(n_perm, dtype=np.float64)
    for perm_idx in range(n_perm):
        shuffled = np.array(z_energy, copy=True)
        for idx in group_indices:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[perm_idx] = _spearman(shuffled, z_divergence)
    valid = null[np.isfinite(null)]
    if valid.size == 0:
        p = float("nan")
    else:
        p = float((1 + np.sum(valid >= observed)) / (valid.size + 1))
    return {
        "n_permutations": int(n_perm),
        "n_valid_permutations": int(valid.size),
        "p_value_positive": p,
        "null_mean": float(np.mean(valid)) if valid.size else float("nan"),
        "null_std": _sample_std(valid) if valid.size > 1 else float("nan"),
        "null_q05": float(np.quantile(valid, 0.05)) if valid.size else float("nan"),
        "null_q50": float(np.quantile(valid, 0.50)) if valid.size else float("nan"),
        "null_q95": float(np.quantile(valid, 0.95)) if valid.size else float("nan"),
    }


def _finite_records(rows: list[dict[str, str]], trajectory_col: str, energy_col: str, divergence_col: str, time_col: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        traj = str(row.get(trajectory_col, "")).strip()
        energy = _safe_float(row.get(energy_col))
        divergence = _safe_float(row.get(divergence_col))
        if not traj or not np.isfinite(energy) or not np.isfinite(divergence):
            continue
        rec = {
            "row_idx": int(row_idx),
            "trajectory_id": traj,
            "energy": float(energy),
            "divergence": float(divergence),
        }
        if time_col is not None:
            rec["time"] = row.get(time_col, "")
        out.append(rec)
    return out


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    input_path = resolve_path(args.input)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")
    rows, headers = _read_csv(input_path)
    if not rows:
        raise ValueError(f"Input CSV has no rows: {input_path}")

    trajectory_col = _resolve_column(headers, args.trajectory_col, TRAJECTORY_CANDIDATES, "trajectory id")
    energy_col = _resolve_column(headers, args.energy_col, ENERGY_CANDIDATES, "Delta-H energy")
    divergence_col = _resolve_column(headers, args.divergence_col, DIVERGENCE_CANDIDATES, "future divergence")
    time_col = _resolve_column(headers, args.time_col, TIME_CANDIDATES, "branch time", required=False)
    assert trajectory_col is not None and energy_col is not None and divergence_col is not None

    log_event(
        f"C2 within-trajectory association input={input_path} trajectory_col={trajectory_col} "
        f"energy_col={energy_col} divergence_col={divergence_col} time_col={time_col}",
        component="c2-within",
    )

    finite_records = _finite_records(rows, trajectory_col, energy_col, divergence_col, time_col)
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in finite_records:
        groups.setdefault(str(rec["trajectory_id"]), []).append(rec)

    per_traj_rows: list[dict[str, Any]] = []
    retained_records: list[dict[str, Any]] = []
    z_energy_parts: list[np.ndarray] = []
    z_divergence_parts: list[np.ndarray] = []
    group_indices: list[np.ndarray] = []
    offset = 0

    for traj_id in sorted(groups):
        group = groups[traj_id]
        energy = np.asarray([rec["energy"] for rec in group], dtype=np.float64)
        divergence = np.asarray([rec["divergence"] for rec in group], dtype=np.float64)
        n = int(energy.size)
        e_mean = float(np.mean(energy)) if n else float("nan")
        b_mean = float(np.mean(divergence)) if n else float("nan")
        e_std = _sample_std(energy)
        b_std = _sample_std(divergence)
        status = "ok"
        if n < 3:
            status = "dropped_fewer_than_3_valid_branch_states"
        elif not np.isfinite(e_std) or e_std <= 1e-12:
            status = "dropped_zero_energy_variance"
        elif not np.isfinite(b_std) or b_std <= 1e-12:
            status = "dropped_zero_divergence_variance"

        pearson = _pearson(energy, divergence) if status == "ok" else float("nan")
        spearman = _spearman(energy, divergence) if status == "ok" else float("nan")
        per_traj_rows.append(
            {
                "trajectory_id": traj_id,
                "n_branch_states": n,
                "status": status,
                "valid": status == "ok",
                "spearman_rho": spearman,
                "pearson_r": pearson,
                "energy_mean": e_mean,
                "energy_std": e_std,
                "divergence_mean": b_mean,
                "divergence_std": b_std,
            }
        )
        if status != "ok":
            continue
        z_e, _e_mean, _e_std = _zscore(energy)
        z_b, _b_mean, _b_std = _zscore(divergence)
        z_energy_parts.append(z_e)
        z_divergence_parts.append(z_b)
        idx = np.arange(offset, offset + n, dtype=np.int64)
        group_indices.append(idx)
        offset += n
        for rec, ze, zb in zip(group, z_e, z_b):
            retained_records.append({**rec, "z_energy": float(ze), "z_divergence": float(zb)})

    if retained_records:
        raw_energy = np.asarray([rec["energy"] for rec in retained_records], dtype=np.float64)
        raw_divergence = np.asarray([rec["divergence"] for rec in retained_records], dtype=np.float64)
        z_energy = np.concatenate(z_energy_parts) if z_energy_parts else np.asarray([], dtype=np.float64)
        z_divergence = np.concatenate(z_divergence_parts) if z_divergence_parts else np.asarray([], dtype=np.float64)
    else:
        raw_energy = np.asarray([], dtype=np.float64)
        raw_divergence = np.asarray([], dtype=np.float64)
        z_energy = np.asarray([], dtype=np.float64)
        z_divergence = np.asarray([], dtype=np.float64)

    pooled = _corr_summary(raw_energy, raw_divergence)
    within = _corr_summary(z_energy, z_divergence)
    valid_rhos = np.asarray([_safe_float(row.get("spearman_rho")) for row in per_traj_rows if row.get("status") == "ok"], dtype=np.float64)
    valid_rhos = valid_rhos[np.isfinite(valid_rhos)]
    permutation = _permutation_pvalue(
        z_energy,
        z_divergence,
        group_indices,
        within["spearman_r"],
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
    )

    out_csv = resolve_path(args.out_csv)
    out_json = resolve_path(args.out_json)
    if out_csv is None or out_json is None:
        raise ValueError("--out-csv and --out-json are required")
    ensure_dir(out_csv.parent)
    ensure_dir(out_json.parent)
    write_csv(
        out_csv,
        per_traj_rows,
        fieldnames=[
            "trajectory_id",
            "n_branch_states",
            "status",
            "valid",
            "spearman_rho",
            "pearson_r",
            "energy_mean",
            "energy_std",
            "divergence_mean",
            "divergence_std",
        ],
    )

    summary = {
        "input": str(input_path),
        "columns": {
            "trajectory": trajectory_col,
            "time": time_col or "",
            "energy": energy_col,
            "divergence": divergence_col,
        },
        "sample_sizes": {
            "n_input_rows": len(rows),
            "n_finite_rows": len(finite_records),
            "n_total_trajectories": len(groups),
            "n_valid_trajectories": int(sum(1 for row in per_traj_rows if row.get("status") == "ok")),
            "n_dropped_trajectories": int(sum(1 for row in per_traj_rows if row.get("status") != "ok")),
            "n_retained_branch_states": len(retained_records),
        },
        "pooled_raw": pooled,
        "within_trajectory_zscored": within,
        "within_trajectory_spearman_permutation_null": permutation,
        "per_trajectory_spearman_summary": {
            "n_valid": int(valid_rhos.size),
            "mean": float(np.mean(valid_rhos)) if valid_rhos.size else float("nan"),
            "median": float(np.median(valid_rhos)) if valid_rhos.size else float("nan"),
            "std": _sample_std(valid_rhos) if valid_rhos.size > 1 else float("nan"),
            "min": float(np.min(valid_rhos)) if valid_rhos.size else float("nan"),
            "max": float(np.max(valid_rhos)) if valid_rhos.size else float("nan"),
        },
        "outputs": {
            "per_trajectory_csv": str(out_csv),
            "summary_json": str(out_json),
        },
    }
    write_json(out_json, _json_clean(summary))
    log_event(
        f"C2 within-trajectory association done valid_trajectories={summary['sample_sizes']['n_valid_trajectories']} "
        f"retained_states={summary['sample_sizes']['n_retained_branch_states']} "
        f"within_spearman={within['spearman_r']:.6g} perm_p={permutation['p_value_positive']:.6g} "
        f"out_json={out_json} out_csv={out_csv}",
        component="c2-within",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Posthoc C2 within-trajectory Delta-H / future divergence association.")
    parser.add_argument("--input", required=True, help="Branch table CSV with one row per branch state.")
    parser.add_argument("--trajectory-col", default=None, help="Source trajectory id column. Autodetected if omitted.")
    parser.add_argument("--time-col", default=None, help="Optional branch time column. Autodetected for metadata if omitted.")
    parser.add_argument("--energy-col", default=None, help="Delta-H branch energy column. Autodetected if omitted.")
    parser.add_argument("--divergence-col", default=None, help="Future branch divergence column. Autodetected if omitted.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12_345)
    args = parser.parse_args(argv)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
