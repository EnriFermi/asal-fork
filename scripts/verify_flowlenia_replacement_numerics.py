from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = [5000, 10000, 15000, 20000, 30000]
RTOL = 1e-12
ATOL = 1e-15


def _assert_close(
    actual: pd.Series | np.ndarray,
    expected: pd.Series | np.ndarray,
    label: str,
) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.allclose(
        left,
        right,
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    ):
        max_error = (
            float(np.nanmax(np.abs(left - right)))
            if left.shape == right.shape
            else float("nan")
        )
        raise AssertionError(
            f"{label} mismatch: {left.shape=} {right.shape=} {max_error=}"
        )


def _verify_c1(data_root: Path) -> dict[str, object]:
    scores = pd.read_csv(data_root / "c1_rollout_scores.csv")
    summary = pd.read_csv(data_root / "c1_run_summary.csv").sort_values("run_idx")
    if sorted(scores["run_idx"].astype(int).unique()) != list(range(10)):
        raise AssertionError("C1 rollout table does not contain runs 000--009.")

    rows = []
    for run_idx, group in scores.groupby("run_idx", sort=True):
        optimized = group.loc[
            group["candidate_kind"] == "optimized",
            "eval_score_mspd",
        ].to_numpy(float)
        random = group.loc[
            group["candidate_kind"] == "random",
            "eval_score_mspd",
        ].to_numpy(float)
        if optimized.size != 4 or random.size != 12:
            raise AssertionError(
                f"C1 run {int(run_idx):03d}: expected 4/12 rollout rows."
            )
        optimized_median = float(np.median(optimized))
        random_median = float(np.median(random))
        rows.append(
            {
                "run_idx": int(run_idx),
                "optimized_median": optimized_median,
                "random_median": random_median,
                "contrast": optimized_median - random_median,
            }
        )
    derived = pd.DataFrame(rows).sort_values("run_idx")
    _assert_close(
        derived["optimized_median"],
        summary["eval_score_mspd__optimized_median"],
        "C1 optimized medians",
    )
    _assert_close(
        derived["random_median"],
        summary["eval_score_mspd__random_median"],
        "C1 random medians",
    )
    _assert_close(
        derived["contrast"],
        summary["delta_vs_random_median"],
        "C1 contrasts",
    )
    n_positive = int((derived["contrast"] > 0.0).sum())
    if n_positive != 9:
        raise AssertionError(f"C1 expected 9/10 positive, found {n_positive}/10.")

    tau = pd.read_csv(data_root / "c1_tau_profiles_long.csv")
    tau_summary = pd.read_csv(
        data_root / "c1_tau_profiles_summary.csv"
    ).sort_values(["split", "candidate_kind", "tau_steps"])
    if len(tau) != 3200:
        raise AssertionError(f"C1 tau table expected 3200 rows, found {len(tau)}.")
    derived_tau = (
        tau.groupby(["split", "candidate_kind", "tau_steps"], sort=True)["mspd"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="size",
        )
        .reset_index()
        .sort_values(["split", "candidate_kind", "tau_steps"])
    )
    for column in ("median", "q25", "q75", "n"):
        _assert_close(
            derived_tau[column],
            tau_summary[column],
            f"C1 tau {column}",
        )
    return {
        "runs": 10,
        "rollout_rows": int(len(scores)),
        "positive_contrasts": n_positive,
        "tau_profile_rows": int(len(tau)),
    }


def _verify_c2(data_root: Path) -> dict[str, object]:
    pair_details = pd.read_csv(
        data_root / "c2_rng_only_pair_details_t0_excluded.csv"
    )
    scores = pd.read_csv(data_root / "c2_rng_only_scores_t0_excluded.csv")
    matched = pd.read_csv(data_root / "c2_rng_only_matched_high_low.csv")
    run_horizon = pd.read_csv(
        data_root / "c2_rng_only_run_horizon.csv"
    ).sort_values(["run_idx", "horizon_steps"])
    run_aggregate = pd.read_csv(
        data_root / "c2_rng_only_run_aggregate.csv"
    ).sort_values("run_idx")

    keys = ["run_idx", "traj_id", "condition", "pair_id", "horizon_steps"]
    pair_medians = (
        pair_details.groupby(keys, sort=True)["pairwise_future_clip_chamfer"]
        .median()
        .rename("derived")
        .reset_index()
    )
    score_check = scores.merge(pair_medians, on=keys, validate="one_to_one")
    _assert_close(
        score_check["future_clip_chamfer"],
        score_check["derived"],
        "C2 state score from three branch-pair distances",
    )

    derived_matched = scores.pivot(
        index=["run_idx", "traj_id", "horizon_steps", "pair_id"],
        columns="condition",
        values="future_clip_chamfer",
    ).reset_index()
    derived_matched["high_minus_low"] = (
        derived_matched["high"] - derived_matched["low"]
    )
    match_keys = ["run_idx", "traj_id", "horizon_steps", "pair_id"]
    matched_check = matched.merge(
        derived_matched,
        on=match_keys,
        suffixes=("_stored", "_derived"),
        validate="one_to_one",
    )
    for column in ("high", "mid", "low", "high_minus_low"):
        _assert_close(
            matched_check[f"{column}_stored"],
            matched_check[f"{column}_derived"],
            f"C2 matched {column}",
        )

    derived_run_horizon = (
        matched.groupby(["run_idx", "horizon_steps"], sort=True)
        .agg(
            mean_high=("high", "mean"),
            mean_low=("low", "mean"),
            mean_high_minus_low=("high_minus_low", "mean"),
            median_high_minus_low=("high_minus_low", "median"),
            n_matched_pairs=("high_minus_low", "size"),
        )
        .reset_index()
        .sort_values(["run_idx", "horizon_steps"])
    )
    derived_run_aggregate = (
        matched.groupby("run_idx", sort=True)
        .agg(
            mean_high=("high", "mean"),
            mean_low=("low", "mean"),
            mean_high_minus_low=("high_minus_low", "mean"),
            median_high_minus_low=("high_minus_low", "median"),
            n_matched_pairs=("high_minus_low", "size"),
        )
        .reset_index()
        .sort_values("run_idx")
    )
    aggregate_columns = [
        "mean_high",
        "mean_low",
        "mean_high_minus_low",
        "median_high_minus_low",
        "n_matched_pairs",
    ]
    for column in aggregate_columns:
        _assert_close(
            derived_run_horizon[column],
            run_horizon[column],
            f"C2 run-horizon {column}",
        )
        _assert_close(
            derived_run_aggregate[column],
            run_aggregate[column],
            f"C2 run aggregate {column}",
        )
    if sorted(run_horizon["horizon_steps"].astype(int).unique()) != HORIZONS:
        raise AssertionError("C2 horizon grid does not match 5k--30k.")
    n_positive = int((run_aggregate["mean_high_minus_low"] > 0.0).sum())
    if n_positive != 10:
        raise AssertionError(f"C2 expected 10/10 positive, found {n_positive}/10.")

    points = pd.read_csv(data_root / "c2_branch_selection_points.csv")
    point_counts = points.groupby(["run_idx", "condition"]).size()
    if set(points["condition"]) != {"high", "mid", "low"}:
        raise AssertionError("C2 branch-selection strata are incomplete.")
    if len(points) != 150 or not bool((point_counts == 5).all()):
        raise AssertionError("C2 branch selection must have 5 points per stratum/run.")
    heatmaps = pd.read_csv(data_root / "c2_branch_selection_heatmaps.csv")
    if len(heatmaps) != 4700:
        raise AssertionError(
            f"C2 heatmap table expected 4700 cells, found {len(heatmaps)}."
        )
    shape_check = heatmaps.groupby("run_idx").agg(
        n_tau=("tau_idx", "nunique"),
        n_windows=("window_idx", "nunique"),
        n_cells=("processed_delta_h", "size"),
    )
    if not bool(
        (
            (shape_check["n_tau"] == 10)
            & (shape_check["n_windows"] == 47)
            & (shape_check["n_cells"] == 470)
        ).all()
    ):
        raise AssertionError("C2 branch-selection heatmap shapes are inconsistent.")

    return {
        "runs": 10,
        "branch_pair_rows": int(len(pair_details)),
        "state_horizon_scores": int(len(scores)),
        "positive_run_aggregates": n_positive,
        "horizons": HORIZONS,
        "selected_branch_points": int(len(points)),
        "heatmap_cells": int(len(heatmaps)),
    }


def _verify_c5(data_root: Path) -> dict[str, object]:
    points = pd.read_csv(data_root / "c5_5000_point_metrics.csv")
    candidates = pd.read_csv(
        data_root / "c5_5000_candidate_summary.csv"
    ).sort_values("candidate_id")
    runs = pd.read_csv(data_root / "c5_5000_run_summary.csv").sort_values(
        "run_idx"
    )
    if set(points["horizon_steps"].astype(int)) != {5000}:
        raise AssertionError("C5 point table is not restricted to the 5k horizon.")

    derived_excess = (
        points["paired_same_seed_clip_post_release"]
        - points["free_within_clip_post_release"]
    )
    _assert_close(
        derived_excess,
        points["excess_clip_post_release"],
        "C5 point metric",
    )
    candidate_medians = (
        points.groupby("candidate_id", sort=True)["excess_clip_post_release"]
        .median()
        .rename("derived")
        .reset_index()
        .sort_values("candidate_id")
    )
    candidate_check = candidates.merge(
        candidate_medians,
        on="candidate_id",
        validate="one_to_one",
    )
    _assert_close(
        candidate_check["excess_clip_post_release"],
        candidate_check["derived"],
        "C5 candidate medians",
    )
    if len(points) != 600 or len(candidates) != 40:
        raise AssertionError("C5 expected 600 point rows and 40 candidate rows.")

    derived_rows = []
    for run_idx, group in candidates.groupby("run_idx", sort=True):
        optimized = group.loc[
            group["candidate_kind"] == "optimized",
            "excess_clip_post_release",
        ].to_numpy(float)
        random = group.loc[
            group["candidate_kind"] == "random",
            "excess_clip_post_release",
        ].to_numpy(float)
        if optimized.size != 1 or random.size != 3:
            raise AssertionError(
                f"C5 run {int(run_idx):03d}: expected 1 optimized and 3 random candidates."
            )
        random_median = float(np.median(random))
        derived_rows.append(
            {
                "run_idx": int(run_idx),
                "optimized": float(optimized[0]),
                "random_median": random_median,
                "contrast": float(optimized[0] - random_median),
            }
        )
    derived_runs = pd.DataFrame(derived_rows).sort_values("run_idx")
    _assert_close(
        derived_runs["optimized"],
        runs["opt_excess_clip_post_release"],
        "C5 optimized candidate values",
    )
    _assert_close(
        derived_runs["random_median"],
        runs["random_median_excess_clip_post_release"],
        "C5 random medians",
    )
    _assert_close(
        derived_runs["contrast"],
        runs["contrast_excess_clip_post_release"],
        "C5 run contrasts",
    )
    n_positive = int((derived_runs["contrast"] > 0.0).sum())
    if n_positive != 8:
        raise AssertionError(f"C5 expected 8/10 positive, found {n_positive}/10.")
    return {
        "horizon_steps": 5000,
        "horizon_role": "sensitivity",
        "point_rows": int(len(points)),
        "candidate_rows": int(len(candidates)),
        "runs": 10,
        "positive_contrasts": n_positive,
    }


def run(data_root: Path) -> dict[str, object]:
    return {
        "status": "passed",
        "c1": _verify_c1(data_root),
        "c2": _verify_c2(data_root),
        "c5": _verify_c5(data_root),
        "branch_selection_marker_y_semantics": (
            "fixed display lanes for high/mid/low; not selected tau values"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute the numerical aggregation chain behind "
            "the five canonical Flow-Lenia replacement figures."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    result = run(args.data_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
