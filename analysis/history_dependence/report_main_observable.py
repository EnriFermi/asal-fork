from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import ensure_dir, progress_bar, save_dataframe, write_json


PAIR_CLASS_NAMES = ("free-free", "wall-wall", "free-wall")
DEFAULT_TRIAL_METRICS = [
    "delta_h_dist_tau3000_ks",
    "delta_h_dist_tau3000_energy",
    "absdiff_mean_speed",
    "absdiff_speed_std",
    "absdiff_spatial_spread",
]


def _pair_class_from_masks(is_free_a: np.ndarray, is_free_b: np.ndarray) -> np.ndarray:
    pair_class = np.full(is_free_a.shape, "free-wall", dtype=object)
    pair_class[is_free_a & is_free_b] = "free-free"
    pair_class[(~is_free_a) & (~is_free_b)] = "wall-wall"
    return pair_class.astype(str)


def _pair_mask(pair_classes: np.ndarray, class_name: str) -> np.ndarray:
    class_name = str(class_name).strip().lower()
    if class_name not in PAIR_CLASS_NAMES:
        raise ValueError(f"Unsupported pair class {class_name!r}; expected one of {PAIR_CLASS_NAMES}.")
    return pair_classes == class_name


def _extract_upper_triangle(
    runs: pd.DataFrame,
    distance_matrix_or_function: pd.DataFrame | np.ndarray | Callable[[pd.Series, pd.Series], float],
) -> pd.DataFrame:
    ordered = runs.reset_index(drop=True).copy()
    n_runs = int(ordered.shape[0])
    ii, jj = np.triu_indices(n_runs, k=1)

    if isinstance(distance_matrix_or_function, pd.DataFrame):
        matrix = distance_matrix_or_function.loc[ordered["run_id"], ordered["run_id"]].to_numpy(dtype=np.float64)
        values = matrix[ii, jj]
    elif callable(distance_matrix_or_function):
        values = np.asarray(
            [float(distance_matrix_or_function(ordered.iloc[i], ordered.iloc[j])) for i, j in zip(ii, jj)],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(distance_matrix_or_function, dtype=np.float64)
        if matrix.shape != (n_runs, n_runs):
            raise ValueError(f"Distance matrix shape must match number of runs, got {matrix.shape} for n_runs={n_runs}.")
        values = matrix[ii, jj]

    cond = ordered["condition"].astype(str).str.lower().to_numpy()
    is_free = cond == "free"
    payload: dict[str, Any] = {
        "run_a": ordered.iloc[ii]["run_id"].to_numpy(),
        "run_b": ordered.iloc[jj]["run_id"].to_numpy(),
        "condition_a": cond[ii],
        "condition_b": cond[jj],
        "pair_type": _pair_class_from_masks(is_free[ii], is_free[jj]),
        "distance": values,
    }
    if "pair_group_id" in ordered.columns:
        payload["pair_group_a"] = ordered.iloc[ii]["pair_group_id"].astype(str).to_numpy()
        payload["pair_group_b"] = ordered.iloc[jj]["pair_group_id"].astype(str).to_numpy()
        payload["same_pair_group"] = payload["pair_group_a"] == payload["pair_group_b"]
    if "trial_idx" in ordered.columns:
        payload["trial_idx_a"] = ordered.iloc[ii]["trial_idx"].to_numpy()
        payload["trial_idx_b"] = ordered.iloc[jj]["trial_idx"].to_numpy()
    if "variant" in ordered.columns:
        payload["variant_a"] = ordered.iloc[ii]["variant"].astype(str).to_numpy()
        payload["variant_b"] = ordered.iloc[jj]["variant"].astype(str).to_numpy()
    return pd.DataFrame(payload)


def _u_from_values(class_a_values: np.ndarray, class_b_values: np.ndarray, *, tie_tol: float = 0.0) -> tuple[float, float, float]:
    a = np.asarray(class_a_values, dtype=np.float64)
    b = np.asarray(class_b_values, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan
    diff = a[:, None] - b[None, :]
    if tie_tol > 0:
        gt = diff > tie_tol
        eq = np.abs(diff) <= tie_tol
    else:
        gt = diff > 0.0
        eq = diff == 0.0
    a_strict = float(np.mean(gt))
    a_tie = float(np.mean(gt + 0.5 * eq))
    u_stat = float(np.sum(gt + 0.5 * eq))
    return a_strict, a_tie, u_stat


def _summarize_class_values(class_a_values: np.ndarray, class_b_values: np.ndarray, *, tie_tol: float = 0.0) -> dict[str, float]:
    a = np.asarray(class_a_values, dtype=np.float64)
    b = np.asarray(class_b_values, dtype=np.float64)
    a_strict, a_tie, u_stat = _u_from_values(a, b, tie_tol=tie_tol)
    return {
        "n_class_a": int(a.size),
        "n_class_b": int(b.size),
        "mean_class_a": float(np.mean(a)),
        "mean_class_b": float(np.mean(b)),
        "delta_mean": float(np.mean(a) - np.mean(b)),
        "ratio_mean": float(np.mean(a) / max(np.mean(b), 1e-12)),
        "A": float(a_strict),
        "A_tie": float(a_tie),
        "u_statistic": float(u_stat),
    }


def _bootstrap_a_tie(
    runs: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    class_a: str,
    class_b: str,
    tie_tol: float,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    show_progress: bool,
) -> tuple[float, float, np.ndarray]:
    order = matrix.index.tolist()
    idx_map = {run_id: idx for idx, run_id in enumerate(order)}
    arr = matrix.loc[order, order].to_numpy(dtype=np.float64)

    free_ids = runs.loc[runs["condition"] == "free", "run_id"].astype(str).tolist()
    wall_ids = runs.loc[runs["condition"] == "wall", "run_id"].astype(str).tolist()
    if int(n_bootstrap) <= 0 or len(free_ids) < 2 or len(wall_ids) < 1:
        return np.nan, np.nan, np.asarray([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    stats = np.zeros(int(n_bootstrap), dtype=np.float64)
    with progress_bar(total=int(n_bootstrap), desc="Bootstrap A_tie", enabled=show_progress, leave=False) as pbar:
        for rep in range(int(n_bootstrap)):
            free_boot = [free_ids[idx] for idx in rng.integers(0, len(free_ids), size=len(free_ids))]
            wall_boot = [wall_ids[idx] for idx in rng.integers(0, len(wall_ids), size=len(wall_ids))]
            sample_ids = free_boot + wall_boot
            sample_cond = np.asarray(["free"] * len(free_boot) + ["wall"] * len(wall_boot), dtype=object)
            sample_idx = np.asarray([idx_map[run_id] for run_id in sample_ids], dtype=np.int32)
            sample_arr = arr[np.ix_(sample_idx, sample_idx)]
            ii, jj = np.triu_indices(sample_arr.shape[0], k=1)
            pair_classes = _pair_class_from_masks(sample_cond[ii] == "free", sample_cond[jj] == "free")
            distances = sample_arr[ii, jj]
            class_a_values = distances[_pair_mask(pair_classes, class_a)]
            class_b_values = distances[_pair_mask(pair_classes, class_b)]
            _, a_tie, _ = _u_from_values(class_a_values, class_b_values, tie_tol=tie_tol)
            stats[rep] = a_tie
            pbar.update(1)

    alpha = (1.0 - float(ci_level)) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    return low, high, stats


def _permutation_superiority_pvalue(
    runs: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    class_a: str,
    class_b: str,
    tie_tol: float,
    max_exact: int,
    n_samples: int,
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    order = matrix.index.tolist()
    cond = runs.set_index("run_id").loc[order, "condition"].astype(str).str.lower().to_numpy()
    n_total = len(order)
    n_wall = int(np.sum(cond == "wall"))
    n_free = int(np.sum(cond == "free"))
    if n_wall <= 0 or n_free <= 1 or (int(max_exact) < 1 and int(n_samples) < 1):
        return {"pvalue_greater": np.nan, "permutation_mode": "invalid", "n_permutations": 0}

    arr = matrix.loc[order, order].to_numpy(dtype=np.float64)
    ii, jj = np.triu_indices(n_total, k=1)
    distances = arr[ii, jj]

    def _stat_from_labels(labels: np.ndarray) -> tuple[float, float]:
        pair_classes = _pair_class_from_masks(labels[ii] == "free", labels[jj] == "free")
        class_a_values = distances[_pair_mask(pair_classes, class_a)]
        class_b_values = distances[_pair_mask(pair_classes, class_b)]
        _, a_tie, u_stat = _u_from_values(class_a_values, class_b_values, tie_tol=tie_tol)
        return a_tie, u_stat

    observed_a_tie, observed_u = _stat_from_labels(cond)
    total_perms = int(math.comb(n_total, n_wall))
    rng = np.random.default_rng(seed)
    sampled_stats = []

    if total_perms <= int(max_exact):
        mode = "exact"
        iterable = itertools.combinations(range(n_total), n_wall)
        progress_total = total_perms
    else:
        mode = "monte_carlo"
        iterable = range(int(n_samples))
        progress_total = int(n_samples)

    with progress_bar(total=progress_total, desc="Main observable permutations", enabled=show_progress, leave=False) as pbar:
        if mode == "exact":
            for wall_idx in iterable:
                labels = np.full(n_total, "free", dtype=object)
                labels[list(wall_idx)] = "wall"
                stat, _ = _stat_from_labels(labels)
                sampled_stats.append(stat)
                pbar.update(1)
        else:
            for _ in iterable:
                wall_idx = rng.choice(n_total, size=n_wall, replace=False)
                labels = np.full(n_total, "free", dtype=object)
                labels[wall_idx] = "wall"
                stat, _ = _stat_from_labels(labels)
                sampled_stats.append(stat)
                pbar.update(1)

    stat_arr = np.asarray(sampled_stats, dtype=np.float64)
    pvalue = float((1.0 + np.sum(stat_arr >= observed_a_tie)) / (1.0 + stat_arr.size))
    return {
        "observed_a_tie": float(observed_a_tie),
        "u_statistic": float(observed_u),
        "pvalue_greater": float(pvalue),
        "permutation_mode": mode,
        "n_permutations": int(stat_arr.size),
        "n_free": int(n_free),
        "n_wall": int(n_wall),
    }


def compute_pair_class_superiority(
    run_groups: pd.DataFrame,
    distance_matrix_or_function: pd.DataFrame | np.ndarray | Callable[[pd.Series, pd.Series], float],
    class_a: str = "free-wall",
    class_b: str = "free-free",
    *,
    distance_name: str = "distance",
    tie_tol: float = 0.0,
) -> dict[str, Any]:
    if "run_id" not in run_groups.columns or "condition" not in run_groups.columns:
        raise ValueError("run_groups must contain columns 'run_id' and 'condition'.")

    raw = _extract_upper_triangle(run_groups, distance_matrix_or_function)
    raw = raw.rename(columns={"distance": distance_name})
    raw["selected_role"] = "other"
    raw.loc[raw["pair_type"] == class_a, "selected_role"] = "class_a"
    raw.loc[raw["pair_type"] == class_b, "selected_role"] = "class_b"

    class_a_values = raw.loc[raw["pair_type"] == class_a, distance_name].to_numpy(dtype=np.float64)
    class_b_values = raw.loc[raw["pair_type"] == class_b, distance_name].to_numpy(dtype=np.float64)
    if class_a_values.size == 0 or class_b_values.size == 0:
        raise ValueError(
            f"Need non-empty pair classes for class_a={class_a!r} and class_b={class_b!r}; "
            f"got sizes {class_a_values.size} and {class_b_values.size}."
        )

    stats = _summarize_class_values(class_a_values, class_b_values, tie_tol=tie_tol)
    stats.update(
        {
            "distance_name": distance_name,
            "class_a": class_a,
            "class_b": class_b,
            "mean_free_wall": float(np.mean(class_a_values)) if class_a == "free-wall" else stats["mean_class_a"],
            "mean_free_free": float(np.mean(class_b_values)) if class_b == "free-free" else stats["mean_class_b"],
        }
    )
    return {
        "raw_pairwise": raw,
        "class_a_values": class_a_values,
        "class_b_values": class_b_values,
        "summary": stats,
    }


def _pick_distance_name(matrices: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> str:
    main_cfg = dict(cfg.get("main_observable", {}))
    requested = main_cfg.get("distance_name")
    if requested:
        requested = str(requested)
        if requested not in matrices:
            raise KeyError(f"Requested main_observable.distance_name={requested!r} not found in matrices.")
        return requested

    preferred = [
        "embedding_cloud_chamfer_cosine",
        "embedding_synced_cosine",
        "embedding_synced_euclidean",
    ]
    for name in preferred:
        if name in matrices:
            return name
    if matrices:
        return sorted(matrices.keys())[0]
    raise ValueError("No distance matrices available for main observable reporting.")


def _variant_sort_key(value: str) -> tuple[int, str]:
    label = str(value).strip().lower()
    order = {"control_a": 0, "control_b": 1, "a": 0, "b": 1}
    return order.get(label, 99), label


def _build_trial_effect_table(
    runs: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    effect_mode: str = "mean_controls",
    anchor_variant: str = "control_a",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {"run_id", "condition", "pair_group_id"}
    if not required.issubset(runs.columns):
        return pd.DataFrame(), [{"reason": f"missing columns: {sorted(required - set(runs.columns))}"}]

    order = runs.copy()
    if "trial_idx" in order.columns:
        order = order.sort_values(["trial_idx", "pair_group_id", "condition", "variant", "run_id"])
    else:
        order = order.sort_values(["pair_group_id", "condition", "variant", "run_id"])

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pair_group_id, group in order.groupby("pair_group_id", sort=False):
        group = group.reset_index(drop=True)
        free = group.loc[group["condition"].astype(str).str.lower() == "free"].copy()
        wall = group.loc[group["condition"].astype(str).str.lower() == "wall"].copy()
        if free.shape[0] != 2 or wall.shape[0] != 1:
            skipped.append(
                {
                    "pair_group_id": str(pair_group_id),
                    "n_free": int(free.shape[0]),
                    "n_wall": int(wall.shape[0]),
                    "reason": "expected exactly 2 free runs and 1 wall run",
                }
            )
            continue

        free = free.sort_values(by=["variant", "run_id"], key=lambda s: s.map(lambda v: _variant_sort_key(str(v))))
        wall_row = wall.iloc[0]
        free_a = free.iloc[0]
        free_b = free.iloc[1]

        d_free_free = float(matrix.loc[free_a["run_id"], free_b["run_id"]])
        d_free_wall_a = float(matrix.loc[free_a["run_id"], wall_row["run_id"]])
        d_free_wall_b = float(matrix.loc[free_b["run_id"], wall_row["run_id"]])
        d_free_wall_mean = 0.5 * (d_free_wall_a + d_free_wall_b)

        mode = str(effect_mode).strip().lower()
        anchor_name = str(anchor_variant).strip().lower()
        anchor_row = None
        reference_row = None
        if mode == "anchor":
            for _, candidate in free.iterrows():
                if str(candidate.get("variant", "")).strip().lower() == anchor_name:
                    anchor_row = candidate
                    break
            if anchor_row is None:
                anchor_row = free.iloc[0]
            reference_pool = free.loc[free["run_id"] != anchor_row["run_id"]]
            reference_row = reference_pool.iloc[0]
            d_anchor_wall = float(matrix.loc[anchor_row["run_id"], wall_row["run_id"]])
            delta_trial = float(d_anchor_wall - d_free_free)
        else:
            d_anchor_wall = d_free_wall_a
            delta_trial = float(d_free_wall_mean - d_free_free)
        anchor_actual = free_a if anchor_row is None else anchor_row
        reference_actual = free_b if reference_row is None else reference_row
        rows.append(
            {
                "pair_group_id": str(pair_group_id),
                "trial_id": str(pair_group_id),
                "trial_idx": None if "trial_idx" not in group.columns else int(group["trial_idx"].iloc[0]),
                "free_a_run_id": str(free_a["run_id"]),
                "free_b_run_id": str(free_b["run_id"]),
                "wall_run_id": str(wall_row["run_id"]),
                "free_a_variant": str(free_a.get("variant", "free_a")),
                "free_b_variant": str(free_b.get("variant", "free_b")),
                "wall_variant": str(wall_row.get("variant", "wall")),
                "effect_mode": mode,
                "anchor_variant_requested": anchor_variant,
                "anchor_run_id": str(anchor_actual["run_id"]),
                "anchor_variant": str(anchor_actual.get("variant", "free_a")),
                "reference_run_id": str(reference_actual["run_id"]),
                "reference_variant": str(reference_actual.get("variant", "free_b")),
                "d_free_free": d_free_free,
                "d_anchor_wall": float(d_anchor_wall),
                "d_free_wall_a": d_free_wall_a,
                "d_free_wall_b": d_free_wall_b,
                "d_free_wall_mean": d_free_wall_mean,
                "delta_trial": float(delta_trial),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        if "trial_idx" in frame.columns:
            frame = frame.sort_values(["trial_idx", "pair_group_id"]).reset_index(drop=True)
        else:
            frame = frame.sort_values(["pair_group_id"]).reset_index(drop=True)
    return frame, skipped


def _sign_flip_pvalue(
    delta: np.ndarray,
    *,
    max_exact: int,
    n_samples: int,
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 1:
        return {"pvalue_greater": np.nan, "mode": "invalid", "n_permutations": 0}

    observed = float(np.mean(values))
    total = int(1 << n)
    threshold = observed - 1e-15

    if total <= int(max_exact):
        mode = "exact"
        exceed = 0
        with progress_bar(total=total, desc="Trial sign-flip", enabled=show_progress, leave=False) as pbar:
            for signs in itertools.product((-1.0, 1.0), repeat=n):
                stat = float(np.mean(values * np.asarray(signs, dtype=np.float64)))
                if stat >= threshold:
                    exceed += 1
                pbar.update(1)
        pvalue = float(exceed / total)
        n_perm = total
    else:
        mode = "monte_carlo"
        rng = np.random.default_rng(seed)
        exceed = 0
        with progress_bar(total=int(n_samples), desc="Trial sign-flip", enabled=show_progress, leave=False) as pbar:
            for _ in range(int(n_samples)):
                signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=n, replace=True)
                stat = float(np.mean(values * signs))
                if stat >= threshold:
                    exceed += 1
                pbar.update(1)
        n_perm = int(n_samples)
        pvalue = float((1.0 + exceed) / (1.0 + n_perm))

    return {
        "observed_statistic": observed,
        "pvalue_greater": pvalue,
        "mode": mode,
        "n_permutations": int(n_perm),
        "n_trials": int(n),
    }


def _sign_test_pvalue(delta: np.ndarray, *, zero_tol: float = 1e-12) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    values = values[np.isfinite(values)]
    mask = np.abs(values) > float(zero_tol)
    nz = values[mask]
    n = int(nz.size)
    if n < 1:
        return {
            "pvalue_greater": 1.0,
            "n_nonzero": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": int(values.size),
        }

    n_positive = int(np.sum(nz > 0.0))
    tail = sum(math.comb(n, k) for k in range(n_positive, n + 1))
    pvalue = float(tail / (2 ** n))
    return {
        "pvalue_greater": pvalue,
        "n_nonzero": int(n),
        "n_positive": int(n_positive),
        "n_negative": int(np.sum(nz < 0.0)),
        "n_zero": int(values.size - n),
    }


def _bootstrap_delta_ci(
    delta: np.ndarray,
    *,
    stat: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    show_progress: bool,
) -> tuple[float, float, np.ndarray]:
    values = np.asarray(delta, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 1 or int(n_bootstrap) <= 0:
        return np.nan, np.nan, np.asarray([], dtype=np.float64)

    if stat == "median":
        reducer = np.median
    else:
        reducer = np.mean

    rng = np.random.default_rng(seed)
    reps = np.zeros(int(n_bootstrap), dtype=np.float64)
    with progress_bar(total=int(n_bootstrap), desc=f"Bootstrap {stat}", enabled=show_progress, leave=False) as pbar:
        for idx in range(int(n_bootstrap)):
            sample = values[rng.integers(0, n, size=n)]
            reps[idx] = float(reducer(sample))
            pbar.update(1)

    alpha = (1.0 - float(ci_level)) / 2.0
    return float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha)), reps


def _trial_summary_row(
    observable: str,
    trial_df: pd.DataFrame,
    sign_flip: dict[str, Any],
    sign_test: dict[str, Any],
    mean_ci: tuple[float, float],
    median_ci: tuple[float, float],
) -> dict[str, Any]:
    delta = trial_df["delta_trial"].to_numpy(dtype=np.float64)
    return {
        "observable": observable,
        "effect_mode": str(trial_df["effect_mode"].iloc[0]),
        "anchor_variant": str(trial_df["anchor_variant"].iloc[0]),
        "n_trials": int(delta.size),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "bootstrap_CI_low": float(mean_ci[0]),
        "bootstrap_CI_high": float(mean_ci[1]),
        "median_bootstrap_CI_low": float(median_ci[0]),
        "median_bootstrap_CI_high": float(median_ci[1]),
        "exact_pvalue_greater": float(sign_flip["pvalue_greater"]),
        "sign_flip_mode": str(sign_flip["mode"]),
        "sign_flip_permutations": int(sign_flip["n_permutations"]),
        "sign_test_pvalue_greater": float(sign_test["pvalue_greater"]),
        "n_positive_trials": int(np.sum(delta > 0.0)),
        "n_negative_trials": int(np.sum(delta < 0.0)),
        "n_zero_trials": int(np.sum(delta == 0.0)),
    }


def _summary_row(main_summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "distance_name": main_summary["distance_name"],
        "mean_free_free": float(main_summary["mean_free_free"]),
        "mean_free_wall": float(main_summary["mean_free_wall"]),
        "delta_mean": float(main_summary["delta_mean"]),
        "ratio_mean": float(main_summary["ratio_mean"]),
        "A_tie": float(main_summary["A_tie"]),
        "descriptive_CI_low": float(main_summary["descriptive_CI_low"]),
        "descriptive_CI_high": float(main_summary["descriptive_CI_high"]),
        "descriptive_pvalue_greater": float(main_summary["descriptive_pvalue_greater"]),
    }
    if "mean_delta" in main_summary:
        row.update(
            {
                "trial_effect_mode": str(main_summary.get("trial_effect_mode", "")),
                "trial_anchor_variant": str(main_summary.get("trial_anchor_variant", "")),
                "n_trials": int(main_summary["n_trials"]),
                "mean_delta": float(main_summary["mean_delta"]),
                "median_delta": float(main_summary["median_delta"]),
                "bootstrap_CI_low": float(main_summary["bootstrap_CI_low"]),
                "bootstrap_CI_high": float(main_summary["bootstrap_CI_high"]),
                "exact_pvalue_greater": float(main_summary["exact_pvalue_greater"]),
                "sign_test_pvalue_greater": float(main_summary["sign_test_pvalue_greater"]),
            }
        )
    return row


def _report_string(summary: dict[str, Any]) -> str:
    if "mean_delta" in summary:
        effect_clause = "wall runs are farther from their paired controls than controls are from each other"
        if str(summary.get("trial_effect_mode", "")).strip().lower() == "anchor":
            anchor_variant = str(summary.get("trial_anchor_variant", "control_a"))
            effect_clause = (
                f"wall runs are farther from the matched {anchor_variant} control than that control is from its paired "
                f"free control"
            )
        return (
            f"For distance {summary['distance_name']}, across {summary['n_trials']} matched trials the mean trial-level "
            f"effect was mean_delta = {summary['mean_delta']:.3f} and median_delta = {summary['median_delta']:.3f}, "
            f"with 95% bootstrap CI [{summary['bootstrap_CI_low']:.3f}, {summary['bootstrap_CI_high']:.3f}] and "
            f"one-sided sign-flip p = {summary['exact_pvalue_greater']:.3g}. "
            f"This is consistent with frustration-like history dependence: {effect_clause}. Secondary pooled-pair evidence gives A_tie = "
            f"{summary['A_tie']:.3f} for free-wall versus free-free distances."
        )
    return (
        f"For distance {summary['distance_name']}, the pooled-pair superiority statistic is A_tie = "
        f"{summary['A_tie']:.3f}, 95% CI [{summary['descriptive_CI_low']:.3f}, {summary['descriptive_CI_high']:.3f}], "
        f"p = {summary['descriptive_pvalue_greater']:.3g}. "
        f"This is descriptive evidence consistent with frustration-like history dependence."
    )


def plot_superiority_strip(
    raw_pairwise: pd.DataFrame,
    value_col: str,
    path: str | Path,
    *,
    class_a: str,
    class_b: str,
    center: str = "mean",
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    order = [class_b, class_a]
    colors = {class_b: "#4C72B0", class_a: "#C44E52"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)
    rng = np.random.default_rng(0)
    for pos, pair_name in enumerate(order):
        vals = raw_pairwise.loc[raw_pairwise["pair_type"] == pair_name, value_col].dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=vals.size)
        ax.scatter(
            np.full(vals.size, pos) + jitter,
            vals,
            s=42,
            alpha=0.85,
            color=colors[pair_name],
            edgecolor="white",
            linewidth=0.5,
        )
        center_val = float(np.median(vals)) if center == "median" else float(np.mean(vals))
        ax.hlines(center_val, pos - 0.22, pos + 0.22, color="black", linewidth=2.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col}: {class_b} vs {class_a}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_superiority_cdf(
    class_b_values: np.ndarray,
    class_a_values: np.ndarray,
    path: str | Path,
    *,
    class_a: str,
    class_b: str,
    distance_name: str,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=dpi)
    for values, label, color in (
        (class_b_values, class_b, "#4C72B0"),
        (class_a_values, class_a, "#C44E52"),
    ):
        vals = np.sort(np.asarray(values, dtype=np.float64))
        y = np.arange(1, vals.size + 1, dtype=np.float64) / vals.size
        ax.step(vals, y, where="post", linewidth=2.2, label=label, color=color)
    ax.set_xlabel(distance_name)
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"Descriptive ECDF: {class_a} vs {class_b}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_trial_effects(trial_df: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180) -> None:
    import matplotlib.pyplot as plt

    ordered = trial_df.sort_values("delta_trial").reset_index(drop=True)
    y = np.arange(ordered.shape[0])
    colors = np.where(ordered["delta_trial"].to_numpy(dtype=np.float64) >= 0.0, "#C44E52", "#4C72B0")

    fig_h = max(3.2, 0.45 * ordered.shape[0] + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=dpi)
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.hlines(y=y, xmin=0.0, xmax=ordered["delta_trial"], color=colors, linewidth=2.2, alpha=0.9)
    ax.scatter(ordered["delta_trial"], y, s=60, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    labels = ordered["trial_id"].astype(str).tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("delta_trial")
    ax.set_ylabel("matched trial")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_trial_ecdf(trial_df: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180) -> None:
    import matplotlib.pyplot as plt

    vals = np.sort(trial_df["delta_trial"].to_numpy(dtype=np.float64))
    y = np.arange(1, vals.size + 1, dtype=np.float64) / vals.size

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=dpi)
    ax.step(vals, y, where="post", linewidth=2.4, color="#C44E52")
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("delta_trial")
    ax.set_ylabel("empirical CDF across trials")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_main_observable_report(
    runs: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    output_dir: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    main_cfg = dict(cfg.get("main_observable", {}))
    if not bool(main_cfg.get("enabled", True)):
        return {}

    show_progress = bool(dict(cfg.get("progress", {})).get("enabled", True))
    distance_name = _pick_distance_name(matrices, cfg)
    class_a = str(main_cfg.get("class_a", "free-wall"))
    class_b = str(main_cfg.get("class_b", "free-free"))
    tie_tol = float(main_cfg.get("tie_tolerance", 0.0))
    bootstrap_reps = int(main_cfg.get("bootstrap_reps", 2000))
    bootstrap_seed = int(main_cfg.get("bootstrap_seed", 0))
    ci_level = float(main_cfg.get("ci_level", 0.95))
    permutation_max_exact = int(main_cfg.get("permutation_max_exact", 200000))
    permutation_samples = int(main_cfg.get("permutation_samples", 20000))
    permutation_seed = int(main_cfg.get("permutation_seed", 0))
    dpi = int(main_cfg.get("figure_dpi", dict(cfg.get("reporting", {})).get("figure_dpi", 180)))
    include_matrix = bool(main_cfg.get("include_matrix_figure", True))
    trial_effect_mode = str(main_cfg.get("trial_effect_mode", "mean_controls"))
    trial_anchor_variant = str(main_cfg.get("trial_anchor_variant", "control_a"))

    matrix = matrices[distance_name]
    descriptive = compute_pair_class_superiority(
        runs,
        matrix,
        class_a=class_a,
        class_b=class_b,
        distance_name=distance_name,
        tie_tol=tie_tol,
    )
    perm = _permutation_superiority_pvalue(
        runs,
        matrix,
        class_a=class_a,
        class_b=class_b,
        tie_tol=tie_tol,
        max_exact=permutation_max_exact,
        n_samples=permutation_samples,
        seed=permutation_seed,
        show_progress=show_progress,
    )
    ci_low, ci_high, bootstrap_stats = _bootstrap_a_tie(
        runs,
        matrix,
        class_a=class_a,
        class_b=class_b,
        tie_tol=tie_tol,
        n_bootstrap=bootstrap_reps,
        ci_level=ci_level,
        seed=bootstrap_seed,
        show_progress=show_progress,
    )
    descriptive_summary = dict(descriptive["summary"])
    descriptive_summary.update(
        {
            "descriptive_CI_low": float(ci_low),
            "descriptive_CI_high": float(ci_high),
            "descriptive_pvalue_greater": float(perm["pvalue_greater"]),
        }
    )

    trial_df, skipped = _build_trial_effect_table(
        runs,
        matrix,
        effect_mode=trial_effect_mode,
        anchor_variant=trial_anchor_variant,
    )
    summary = dict(descriptive_summary)
    if not trial_df.empty:
        sign_flip = _sign_flip_pvalue(
            trial_df["delta_trial"].to_numpy(dtype=np.float64),
            max_exact=permutation_max_exact,
            n_samples=permutation_samples,
            seed=permutation_seed,
            show_progress=show_progress,
        )
        sign_test = _sign_test_pvalue(trial_df["delta_trial"].to_numpy(dtype=np.float64), zero_tol=max(tie_tol, 1e-12))
        mean_ci_low, mean_ci_high, mean_boot = _bootstrap_delta_ci(
            trial_df["delta_trial"].to_numpy(dtype=np.float64),
            stat="mean",
            n_bootstrap=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed,
            show_progress=show_progress,
        )
        median_ci_low, median_ci_high, median_boot = _bootstrap_delta_ci(
            trial_df["delta_trial"].to_numpy(dtype=np.float64),
            stat="median",
            n_bootstrap=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed + 1,
            show_progress=show_progress,
        )
        summary.update(
            {
                "trial_effect_mode": trial_effect_mode,
                "trial_anchor_variant": trial_anchor_variant,
                "n_trials": int(trial_df.shape[0]),
                "mean_delta": float(np.mean(trial_df["delta_trial"])),
                "median_delta": float(np.median(trial_df["delta_trial"])),
                "bootstrap_CI_low": float(mean_ci_low),
                "bootstrap_CI_high": float(mean_ci_high),
                "exact_pvalue_greater": float(sign_flip["pvalue_greater"]),
                "sign_test_pvalue_greater": float(sign_test["pvalue_greater"]),
                "sign_flip_mode": str(sign_flip["mode"]),
                "sign_flip_permutations": int(sign_flip["n_permutations"]),
            }
        )
    else:
        mean_boot = np.asarray([], dtype=np.float64)
        median_boot = np.asarray([], dtype=np.float64)
        sign_flip = {}
        sign_test = {}

    report = {
        "distance_name": distance_name,
        "summary": summary,
        "raw_pairwise": descriptive["raw_pairwise"],
        "class_a_values": descriptive["class_a_values"],
        "class_b_values": descriptive["class_b_values"],
        "trial_effects": trial_df,
        "skipped_groups": skipped,
        "bootstrap_distribution": bootstrap_stats,
        "trial_bootstrap_mean": mean_boot,
        "trial_bootstrap_median": median_boot,
        "permutation": perm,
        "sign_flip": sign_flip,
        "sign_test": sign_test,
    }
    report["text_summary"] = _report_string(summary)

    base_dir = ensure_dir(Path(output_dir) / "main_observable")
    tables_dir = ensure_dir(base_dir / "tables")
    figures_dir = ensure_dir(base_dir / "figures")

    raw_pairwise = report["raw_pairwise"].copy()
    raw_pairwise = raw_pairwise.rename(columns={distance_name: "distance"})
    raw_pairwise.insert(raw_pairwise.columns.get_loc("selected_role") + 1, "distance_name", distance_name)
    save_dataframe(tables_dir / "raw_pairwise_values.csv", raw_pairwise)
    selected_raw = raw_pairwise.loc[raw_pairwise["selected_role"].isin(["class_a", "class_b"])].reset_index(drop=True)
    save_dataframe(tables_dir / "selected_pairwise_values.csv", selected_raw)
    if not trial_df.empty:
        save_dataframe(tables_dir / "trial_level_effects.csv", trial_df)
    if skipped:
        save_dataframe(tables_dir / "trial_level_skipped_groups.csv", pd.DataFrame(skipped))

    summary_df = pd.DataFrame([_summary_row(summary)])
    save_dataframe(tables_dir / "main_observable_summary.csv", summary_df)
    (base_dir / "main_observable_report.txt").write_text(report["text_summary"])
    write_json(
        base_dir / "main_observable_metadata.json",
        {
            "distance_name": distance_name,
            "class_a": class_a,
            "class_b": class_b,
            "tie_tolerance": tie_tol,
            "bootstrap_reps": bootstrap_reps,
            "bootstrap_seed": bootstrap_seed,
            "ci_level": ci_level,
            "permutation_max_exact": permutation_max_exact,
            "permutation_samples": permutation_samples,
            "permutation_seed": permutation_seed,
            "trial_effect_mode": trial_effect_mode,
            "trial_anchor_variant": trial_anchor_variant,
            "skipped_groups": skipped,
        },
    )

    strip_path = figures_dir / "pair_class_strip.png"
    plot_superiority_strip(
        descriptive["raw_pairwise"],
        distance_name,
        strip_path,
        class_a=class_a,
        class_b=class_b,
        center=str(main_cfg.get("strip_center", "mean")),
        dpi=dpi,
    )
    cdf_path = figures_dir / "pair_class_cdf.png"
    plot_superiority_cdf(
        descriptive["class_b_values"],
        descriptive["class_a_values"],
        cdf_path,
        class_a=class_a,
        class_b=class_b,
        distance_name=distance_name,
        dpi=dpi,
    )

    figure_paths = {
        "pair_class_strip": str(strip_path),
        "pair_class_cdf": str(cdf_path),
    }
    if not trial_df.empty:
        trial_lollipop_path = figures_dir / "trial_level_effects.png"
        plot_trial_effects(trial_df, trial_lollipop_path, title=f"Matched trial effects: {distance_name}", dpi=dpi)
        figure_paths["trial_level_effects"] = str(trial_lollipop_path)
        trial_ecdf_path = figures_dir / "trial_level_ecdf.png"
        plot_trial_ecdf(trial_df, trial_ecdf_path, title=f"Trial-level ECDF: {distance_name}", dpi=dpi)
        figure_paths["trial_level_ecdf"] = str(trial_ecdf_path)

    if include_matrix:
        from .reporting import plot_distance_matrix

        matrix_path = figures_dir / "pairwise_distance_matrix.png"
        ordered_runs = runs.sort_values(["condition", "run_id"]).reset_index(drop=True)
        plot_distance_matrix(
            matrix.loc[ordered_runs["run_id"], ordered_runs["run_id"]],
            ordered_runs,
            matrix_path,
            title=f"{distance_name} distance matrix",
            dpi=dpi,
        )
        figure_paths["pairwise_distance_matrix"] = str(matrix_path)

    report["paths"] = {
        "base_dir": str(base_dir),
        "raw_pairwise_csv": str(tables_dir / "raw_pairwise_values.csv"),
        "selected_pairwise_csv": str(tables_dir / "selected_pairwise_values.csv"),
        "trial_level_csv": None if trial_df.empty else str(tables_dir / "trial_level_effects.csv"),
        "summary_csv": str(tables_dir / "main_observable_summary.csv"),
        "report_txt": str(base_dir / "main_observable_report.txt"),
        "figure_paths": figure_paths,
    }
    return report


def generate_trial_level_metric_tests(
    runs: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    output_dir: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    trial_cfg = dict(cfg.get("trial_level_tests", {}))
    if not bool(trial_cfg.get("enabled", True)):
        return {}

    progress_cfg = dict(cfg.get("progress", {}))
    show_progress = bool(progress_cfg.get("enabled", True))
    metrics = [str(x) for x in trial_cfg.get("metrics", DEFAULT_TRIAL_METRICS)]
    metrics = [name for name in metrics if name in matrices]
    if not metrics:
        return {}

    bootstrap_reps = int(trial_cfg.get("bootstrap_reps", 2000))
    bootstrap_seed = int(trial_cfg.get("bootstrap_seed", 0))
    ci_level = float(trial_cfg.get("ci_level", 0.95))
    permutation_max_exact = int(trial_cfg.get("permutation_max_exact", 200000))
    permutation_samples = int(trial_cfg.get("permutation_samples", 20000))
    permutation_seed = int(trial_cfg.get("permutation_seed", 0))
    zero_tolerance = float(trial_cfg.get("zero_tolerance", 1e-12))
    dpi = int(trial_cfg.get("figure_dpi", dict(cfg.get("reporting", {})).get("figure_dpi", 180)))
    effect_mode = str(trial_cfg.get("effect_mode", "anchor"))
    anchor_variant = str(trial_cfg.get("anchor_variant", "control_a"))

    base_dir = ensure_dir(Path(output_dir) / "trial_level_tests")
    tables_dir = ensure_dir(base_dir / "tables")
    figures_dir = ensure_dir(base_dir / "figures")

    summary_rows = []
    trial_frames = []
    skipped_rows = []
    figure_paths: dict[str, str] = {}

    with progress_bar(total=len(metrics), desc="Trial-level tests", enabled=show_progress, leave=False) as pbar:
        for metric_idx, metric_name in enumerate(metrics):
            matrix = matrices[metric_name]
            trial_df, skipped = _build_trial_effect_table(
                runs,
                matrix,
                effect_mode=effect_mode,
                anchor_variant=anchor_variant,
            )
            skipped_rows.extend(
                [{"observable": metric_name, **row} for row in skipped]
            )
            if trial_df.empty:
                pbar.update(1)
                continue

            delta = trial_df["delta_trial"].to_numpy(dtype=np.float64)
            sign_flip = _sign_flip_pvalue(
                delta,
                max_exact=permutation_max_exact,
                n_samples=permutation_samples,
                seed=permutation_seed + metric_idx,
                show_progress=show_progress,
            )
            sign_test = _sign_test_pvalue(delta, zero_tol=zero_tolerance)
            mean_ci_low, mean_ci_high, _ = _bootstrap_delta_ci(
                delta,
                stat="mean",
                n_bootstrap=bootstrap_reps,
                ci_level=ci_level,
                seed=bootstrap_seed + metric_idx,
                show_progress=show_progress,
            )
            median_ci_low, median_ci_high, _ = _bootstrap_delta_ci(
                delta,
                stat="median",
                n_bootstrap=bootstrap_reps,
                ci_level=ci_level,
                seed=bootstrap_seed + 1000 + metric_idx,
                show_progress=show_progress,
            )
            summary_rows.append(
                _trial_summary_row(
                    metric_name,
                    trial_df,
                    sign_flip,
                    sign_test,
                    (mean_ci_low, mean_ci_high),
                    (median_ci_low, median_ci_high),
                )
            )
            trial_out = trial_df.copy()
            trial_out.insert(0, "observable", metric_name)
            trial_frames.append(trial_out)

            safe_name = metric_name.replace("/", "__")
            lollipop_path = figures_dir / f"{safe_name}_trial_lollipop.png"
            plot_trial_effects(trial_df, lollipop_path, title=f"Matched trial effects: {metric_name}", dpi=dpi)
            ecdf_path = figures_dir / f"{safe_name}_trial_ecdf.png"
            plot_trial_ecdf(trial_df, ecdf_path, title=f"Trial-level ECDF: {metric_name}", dpi=dpi)
            figure_paths[f"{metric_name}__trial_lollipop"] = str(lollipop_path)
            figure_paths[f"{metric_name}__trial_ecdf"] = str(ecdf_path)
            pbar.update(1)

    summary_df = pd.DataFrame(summary_rows)
    trials_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)

    if not summary_df.empty:
        save_dataframe(tables_dir / "trial_level_summary.csv", summary_df)
    if not trials_df.empty:
        save_dataframe(tables_dir / "trial_level_effects.csv", trials_df)
    if not skipped_df.empty:
        save_dataframe(tables_dir / "trial_level_skipped_groups.csv", skipped_df)

    report_lines = []
    for row in summary_rows:
        effect_note = ""
        if str(row.get("effect_mode", "")).strip().lower() == "anchor":
            effect_note = f", anchor={row.get('anchor_variant', 'control_a')}"
        report_lines.append(
            f"{row['observable']}: mean_delta={row['mean_delta']:.3f}, median_delta={row['median_delta']:.3f}, "
            f"95% CI [{row['bootstrap_CI_low']:.3f}, {row['bootstrap_CI_high']:.3f}], "
            f"sign-flip p={row['exact_pvalue_greater']:.3g}, sign-test p={row['sign_test_pvalue_greater']:.3g}"
            f"{effect_note}."
        )
    report_text = "\n".join(report_lines)
    if report_text:
        (base_dir / "trial_level_report.txt").write_text(report_text)

    write_json(
        base_dir / "trial_level_metadata.json",
        {
            "metrics_requested": [str(x) for x in trial_cfg.get("metrics", DEFAULT_TRIAL_METRICS)],
            "metrics_used": metrics,
            "bootstrap_reps": bootstrap_reps,
            "bootstrap_seed": bootstrap_seed,
            "ci_level": ci_level,
            "permutation_max_exact": permutation_max_exact,
            "permutation_samples": permutation_samples,
            "permutation_seed": permutation_seed,
            "zero_tolerance": zero_tolerance,
            "effect_mode": effect_mode,
            "anchor_variant": anchor_variant,
        },
    )

    return {
        "summary": summary_df,
        "trial_effects": trials_df,
        "skipped_groups": skipped_df,
        "report_text": report_text,
        "paths": {
            "base_dir": str(base_dir),
            "summary_csv": None if summary_df.empty else str(tables_dir / "trial_level_summary.csv"),
            "trial_effects_csv": None if trials_df.empty else str(tables_dir / "trial_level_effects.csv"),
            "report_txt": None if not report_text else str(base_dir / "trial_level_report.txt"),
            "figure_paths": figure_paths,
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Main pair-class and trial-level reports for history-dependence analysis.")
    parser.add_argument("config", help="Path to analysis YAML config.")
    args = parser.parse_args(argv)

    from .pipeline import load_analysis_config, run_analysis

    cfg = load_analysis_config(args.config)
    result = run_analysis(cfg)
    report = result.get("main_observable", {})
    if report:
        print(report["text_summary"])
        print(f"Saved main observable outputs to {report['paths']['base_dir']}")
    trial_tests = result.get("trial_level_tests", {})
    if trial_tests and trial_tests.get("report_text"):
        print(trial_tests["report_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
