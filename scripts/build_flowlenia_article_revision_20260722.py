from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
C1_ROOT = REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper"
)
C2_C5_ROOT = REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper"
)
C2_ROOT = C2_C5_ROOT / "c2_noise_horizon_sweep/full"
C2_SOURCE_METRICS_ROOT = C2_C5_ROOT / "c2_source_metrics"
C5_ROOT = (
    C2_C5_ROOT
    / "flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis/article_revision_20260722"

HORIZONS = (5000, 10000, 15000, 20000, 30000)
SNAPSHOT_INTERVAL = 50
FRAMES_PER_HORIZON = 8
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260722

OPT_COLOR = "#1769AA"
RANDOM_COLOR = "#9AA0A6"
POSITIVE_COLOR = "#287D8E"
NEGATIVE_COLOR = "#B8574D"
MID_COLOR = "#D99000"
LOW_COLOR = "#377EB8"
HIGH_COLOR = "#C23B45"


def _matplotlib() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "flowlenia_article_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _save_figure(fig: Any, figures_root: Path, name: str) -> list[Path]:
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 240}),
        (".pdf", {}),
    ):
        path = figures_root / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def _sign_test_greater(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array) & (array != 0.0)]
    n = int(array.size)
    k = int(np.sum(array > 0.0))
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n))


def _bootstrap_median_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_REPS, array.size))
    medians = np.median(array[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_greater(values: Iterable[float]) -> float:
    from scipy.stats import wilcoxon

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(
        wilcoxon(
            array,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _jitter(count: int, width: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    return np.linspace(-width, width, count, dtype=np.float64)


def _build_c1(data_root: Path, figures_root: Path) -> dict[str, Any]:
    plt = _matplotlib()
    scores_path = C1_ROOT / "flow_lenia/checkpoint_scores.csv"
    contrasts_path = C1_ROOT / "flow_lenia/group_contrasts.csv"
    scores = pd.read_csv(scores_path)
    contrasts = pd.read_csv(contrasts_path).sort_values("optimized_run_idx")

    expected_runs = list(range(10))
    if contrasts["optimized_run_idx"].astype(int).tolist() != expected_runs:
        raise ValueError("C1 must contain exactly runs 000--009.")
    positive_count = int((contrasts["delta_vs_random_median"] > 0.0).sum())
    if positive_count != 9:
        raise ValueError(f"Expected the audited C1 result 9/10, found {positive_count}/10.")

    c1_table = contrasts.rename(columns={"optimized_run_idx": "run_idx"}).copy()
    c1_table.to_csv(data_root / "c1_run_summary.csv", index=False)
    c1_rollout_columns = [
        "optimized_run_idx",
        "trial_uid",
        "candidate_kind",
        "candidate_idx",
        "rollout_seed_idx",
        "run_seed",
        "train_tau_steps",
        "eval_score_mspd",
        "optimizer_reference_train_mspd_exact_match",
    ]
    c1_rollouts = scores[c1_rollout_columns].rename(
        columns={"optimized_run_idx": "run_idx"}
    )
    c1_rollouts.to_csv(data_root / "c1_rollout_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.15, 3.55), constrained_layout=True)
    for run_idx in expected_runs:
        group = scores[scores["optimized_run_idx"].astype(int) == run_idx]
        optimized = (
            group[group["candidate_kind"] == "optimized"]["eval_score_mspd"]
            .astype(float)
            .to_numpy()
            * 1000.0
        )
        random = (
            group[group["candidate_kind"] == "random"]["eval_score_mspd"]
            .astype(float)
            .to_numpy()
            * 1000.0
        )
        if optimized.size != 4 or random.size != 12:
            raise ValueError(
                f"C1 run {run_idx:03d}: expected 4 optimized and 12 random "
                f"rollout scores, found {optimized.size} and {random.size}."
            )
        ax.scatter(
            run_idx - 0.12 + _jitter(random.size, 0.07),
            np.sort(random),
            s=18,
            color=RANDOM_COLOR,
            alpha=0.72,
            edgecolors="none",
            zorder=2,
        )
        ax.scatter(
            run_idx + 0.12 + _jitter(optimized.size, 0.045),
            np.sort(optimized),
            s=28,
            color=OPT_COLOR,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        random_median = float(np.median(random))
        optimized_median = float(np.median(optimized))
        ax.plot(
            [run_idx - 0.25, run_idx - 0.01],
            [random_median, random_median],
            color="#202124",
            linewidth=1.7,
            zorder=4,
        )
        ax.scatter(
            [run_idx + 0.12],
            [optimized_median],
            marker="D",
            s=35,
            color=OPT_COLOR,
            edgecolors="#202124",
            linewidths=0.45,
            zorder=4,
        )
        ax.plot(
            [run_idx - 0.01, run_idx + 0.12],
            [random_median, optimized_median],
            color=POSITIVE_COLOR if optimized_median > random_median else NEGATIVE_COLOR,
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )

    from matplotlib.lines import Line2D

    legend = [
        Line2D([], [], marker="o", linestyle="none", color=RANDOM_COLOR, label="random rollouts"),
        Line2D([], [], marker="o", linestyle="none", color=OPT_COLOR, label="optimized rollouts"),
        Line2D([], [], color="#202124", linewidth=1.7, label="random median"),
        Line2D([], [], marker="D", linestyle="none", color=OPT_COLOR, markeredgecolor="#202124", label="optimized median"),
    ]
    ax.legend(handles=legend, ncol=2, frameon=False, loc="upper left")
    ax.axhline(0.0, color="#DADCE0", linewidth=0.7, zorder=0)
    ax.set_xticks(expected_runs)
    ax.set_xticklabels([f"{idx:03d}" for idx in expected_runs])
    ax.set_xlabel("optimization run")
    ax.set_ylabel(r"optimizer-matched MSPD ($\times 10^{-3}$)")
    ax.set_title("Flow-Lenia C1: optimized and matched-random rollout scores")
    ax.text(
        0.99,
        0.96,
        "9/10 positive matched contrasts",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    _save_figure(fig, figures_root, "flow_c1_paired_raw_paper")
    plt.close(fig)

    tau_rows: list[dict[str, Any]] = []
    for row in scores.itertuples(index=False):
        maps_path = Path(str(row.maps_path))
        with np.load(maps_path, allow_pickle=False) as data:
            tau_steps = np.asarray(data["tau_steps"], dtype=np.int64)
            selection = np.asarray(data["selection_score_by_tau"], dtype=np.float64)
            evaluation = np.asarray(data["eval_score_by_tau"], dtype=np.float64)
        if not (tau_steps.size == selection.size == evaluation.size == 10):
            raise ValueError(f"Unexpected C1 lag profile in {maps_path}.")
        for tau, select_value, eval_value in zip(
            tau_steps, selection, evaluation
        ):
            common = {
                "run_idx": int(row.optimized_run_idx),
                "trial_uid": str(row.trial_uid),
                "candidate_kind": str(row.candidate_kind),
                "candidate_idx": int(row.candidate_idx),
                "rollout_seed_idx": int(row.rollout_seed_idx),
                "tau_steps": int(tau),
                "train_tau_steps": int(row.train_tau_steps),
            }
            tau_rows.append({**common, "split": "selection", "mspd": float(select_value)})
            tau_rows.append({**common, "split": "evaluation", "mspd": float(eval_value)})
    tau_long = pd.DataFrame(tau_rows)
    tau_long.to_csv(data_root / "c1_tau_profiles_long.csv", index=False)
    tau_summary = (
        tau_long.groupby(["split", "candidate_kind", "tau_steps"], sort=True)["mspd"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n="size",
        )
        .reset_index()
    )
    tau_summary.to_csv(data_root / "c1_tau_profiles_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.05), sharey=True, constrained_layout=True)
    for ax, split, title in zip(
        axes,
        ("selection", "evaluation"),
        ("interleaved split A", "interleaved split B"),
    ):
        for kind, color, label in (
            ("random", RANDOM_COLOR, "random"),
            ("optimized", OPT_COLOR, "optimized"),
        ):
            values = tau_summary[
                (tau_summary["split"] == split)
                & (tau_summary["candidate_kind"] == kind)
            ].sort_values("tau_steps")
            x = values["tau_steps"].to_numpy(dtype=float) / 1000.0
            median = values["median"].to_numpy(dtype=float) * 1000.0
            q25 = values["q25"].to_numpy(dtype=float) * 1000.0
            q75 = values["q75"].to_numpy(dtype=float) * 1000.0
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, median, color=color, linewidth=1.8, marker="o", markersize=3.2, label=label)
        ax.set_title(title)
        ax.set_xlabel(r"lag $\tau$ (thousand steps)")
        ax.set_xticks(np.arange(1, 11, 1))
        ax.grid(axis="y", color="#ECEFF1", linewidth=0.7)
    axes[0].set_ylabel(r"MSPD ($\times 10^{-3}$)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Flow-Lenia C1 lag-profile diagnostic", fontsize=10.5)
    _save_figure(fig, figures_root, "flow_c1_tau_profiles")
    plt.close(fig)

    values = contrasts["delta_vs_random_median"].to_numpy(dtype=float)
    return {
        "n_runs": int(values.size),
        "n_positive": positive_count,
        "median_contrast": float(np.median(values)),
        "mean_contrast": float(np.mean(values)),
        "sign_test_greater_p": _sign_test_greater(values),
        "score_unit": "eval_score_mspd at optimization-selected train tau",
        "input_scores": str(scores_path),
        "input_contrasts": str(contrasts_path),
    }


def _sample_offsets(horizon: int) -> list[int]:
    if horizon <= 0 or horizon % SNAPSHOT_INTERVAL != 0:
        raise ValueError(f"Invalid horizon {horizon}.")
    capture_count = horizon // SNAPSHOT_INTERVAL + 1
    indices = np.linspace(0, capture_count - 1, FRAMES_PER_HORIZON).astype(np.int64)
    offsets = (indices * SNAPSHOT_INTERVAL).astype(np.int64).tolist()
    if offsets[0] != 0 or offsets[-1] != horizon or len(offsets) != FRAMES_PER_HORIZON:
        raise RuntimeError(f"Invalid frame schedule for horizon {horizon}: {offsets}")
    return [int(value) for value in offsets]


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12, None)


def _embedding_chamfer_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Match paper_suite_c2_branching._embedding_chamfer_cosine exactly."""
    left_norm = _normalize_embeddings(left)
    right_norm = _normalize_embeddings(right)
    distances = 1.0 - left_norm @ right_norm.T
    return float(
        0.5
        * (
            np.mean(np.min(distances, axis=1))
            + np.mean(np.min(distances, axis=0))
        )
    )


def _c2_cache_path(cache_root: Path, branch_dir: Path) -> Path:
    digest = hashlib.sha256(str(branch_dir.resolve()).encode("utf-8")).hexdigest()[:24]
    return cache_root / f"{digest}.npz"


def _build_c2(data_root: Path, figures_root: Path) -> dict[str, Any]:
    plt = _matplotlib()
    plan_path = C2_ROOT / "sweep_plan.csv"
    plan = pd.read_csv(plan_path)
    plan = plan[np.isclose(plan["strength"].astype(float), 0.0)].copy()
    if len(plan) != 450:
        raise ValueError(f"Expected 450 RNG-only C2 branch rows, found {len(plan)}.")
    if sorted(plan["run_idx"].astype(int).unique().tolist()) != list(range(10)):
        raise ValueError("C2 RNG-only plan must contain runs 000--009.")

    cache_root = C2_ROOT / "clip_union_cache"
    embedding_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cache_audit_rows: list[dict[str, Any]] = []
    for row in plan.itertuples(index=False):
        branch_dir = Path(str(row.branch_dir))
        cache_path = _c2_cache_path(cache_root, branch_dir)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing C2 embedding cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            embeddings = np.asarray(data["z"], dtype=np.float32)
            cached_branch_dir = str(np.asarray(data["branch_dir"]).item())
            source_signature = str(np.asarray(data["source_signature"]).item())
            cache_version = str(np.asarray(data["cache_version"]).item())
        if Path(cached_branch_dir).resolve() != branch_dir.resolve():
            raise ValueError(f"C2 cache points to another branch: {cache_path}")
        if embeddings.shape != (steps.size, 512):
            raise ValueError(f"Unexpected C2 embedding shape {embeddings.shape}: {cache_path}")
        embedding_index[str(row.branch_dir)] = (steps, embeddings)
        cache_audit_rows.append(
            {
                "run_idx": int(row.run_idx),
                "condition": str(row.condition),
                "pair_id": int(row.pair_id),
                "branch_id": int(row.branch_id),
                "branch_dir": str(branch_dir),
                "cache_path": str(cache_path),
                "cache_sha256": _sha256(cache_path),
                "source_signature": source_signature,
                "cache_version": cache_version,
                "n_embeddings": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
            }
        )
    pd.DataFrame(cache_audit_rows).to_csv(data_root / "c2_embedding_cache_audit.csv", index=False)

    score_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    group_columns = ["run_idx", "traj_id", "condition", "pair_id"]
    for key, group in plan.groupby(group_columns, sort=True):
        group = group.sort_values("branch_id")
        if len(group) != 3:
            raise ValueError(f"C2 state {key} has {len(group)} branches, expected 3.")
        branch_records = list(group.itertuples(index=False))
        for horizon in HORIZONS:
            all_offsets = _sample_offsets(horizon)
            future_offsets = all_offsets[1:]
            branch_embeddings: list[np.ndarray] = []
            for branch in branch_records:
                steps, embeddings = embedding_index[str(branch.branch_dir)]
                target_steps = np.asarray(
                    [int(branch.step) + offset for offset in future_offsets],
                    dtype=np.int64,
                )
                indices = []
                for target_step in target_steps:
                    hits = np.flatnonzero(steps == int(target_step))
                    if hits.size != 1:
                        raise ValueError(
                            f"C2 cache {branch.branch_dir} has {hits.size} embeddings "
                            f"for step {target_step}."
                        )
                    indices.append(int(hits[0]))
                branch_embeddings.append(embeddings[np.asarray(indices, dtype=np.int64)])

            pair_values: list[float] = []
            for left_idx, right_idx in combinations(range(3), 2):
                value = _embedding_chamfer_cosine(
                    branch_embeddings[left_idx], branch_embeddings[right_idx]
                )
                pair_values.append(value)
                pair_rows.append(
                    {
                        "run_idx": int(key[0]),
                        "traj_id": str(key[1]),
                        "condition": str(key[2]),
                        "pair_id": int(key[3]),
                        "horizon_steps": int(horizon),
                        "branch_id_i": int(branch_records[left_idx].branch_id),
                        "branch_id_j": int(branch_records[right_idx].branch_id),
                        "pairwise_future_clip_chamfer": value,
                        "frame_offsets": ",".join(str(value) for value in future_offsets),
                    }
                )
            score_rows.append(
                {
                    "run_idx": int(key[0]),
                    "traj_id": str(key[1]),
                    "condition": str(key[2]),
                    "pair_id": int(key[3]),
                    "step": int(branch_records[0].step),
                    "delta_h": float(branch_records[0].delta_h),
                    "horizon_steps": int(horizon),
                    "future_clip_chamfer": float(np.median(pair_values)),
                    "branch_pair_std": float(np.std(pair_values, ddof=1)),
                    "n_branches": 3,
                    "n_branch_pairs": 3,
                    "n_future_frames": len(future_offsets),
                    "frame_offsets": ",".join(str(value) for value in future_offsets),
                    "branch_point_frame_excluded": True,
                    "external_state_noise": 0.0,
                }
            )

    scores = pd.DataFrame(score_rows)
    pair_details = pd.DataFrame(pair_rows)
    if len(scores) != 750 or len(pair_details) != 2250:
        raise ValueError(
            f"Unexpected C2 output size: {len(scores)} scores, "
            f"{len(pair_details)} branch pairs."
        )
    scores.to_csv(data_root / "c2_rng_only_scores_t0_excluded.csv", index=False)
    pair_details.to_csv(data_root / "c2_rng_only_pair_details_t0_excluded.csv", index=False)

    condition_summary = (
        scores.groupby(["run_idx", "horizon_steps", "condition"], sort=True)[
            "future_clip_chamfer"
        ]
        .mean()
        .rename("mean_future_clip_chamfer")
        .reset_index()
    )
    condition_summary.to_csv(data_root / "c2_rng_only_run_condition_horizon.csv", index=False)

    matched = scores.pivot(
        index=["run_idx", "traj_id", "horizon_steps", "pair_id"],
        columns="condition",
        values="future_clip_chamfer",
    ).reset_index()
    for condition in ("high", "mid", "low"):
        if condition not in matched:
            raise ValueError(f"C2 matched table is missing {condition} scores.")
    matched["high_minus_low"] = matched["high"] - matched["low"]
    matched.to_csv(data_root / "c2_rng_only_matched_high_low.csv", index=False)

    run_horizon = (
        matched.groupby(["run_idx", "horizon_steps"], sort=True)
        .agg(
            mean_high=("high", "mean"),
            mean_low=("low", "mean"),
            mean_high_minus_low=("high_minus_low", "mean"),
            median_high_minus_low=("high_minus_low", "median"),
            n_matched_pairs=("high_minus_low", "size"),
        )
        .reset_index()
    )
    run_horizon.to_csv(data_root / "c2_rng_only_run_horizon.csv", index=False)
    run_aggregate = (
        matched.groupby("run_idx", sort=True)
        .agg(
            mean_high=("high", "mean"),
            mean_low=("low", "mean"),
            mean_high_minus_low=("high_minus_low", "mean"),
            median_high_minus_low=("high_minus_low", "median"),
            n_matched_pairs=("high_minus_low", "size"),
        )
        .reset_index()
    )
    run_aggregate.to_csv(data_root / "c2_rng_only_run_aggregate.csv", index=False)

    effects = run_aggregate["mean_high_minus_low"].to_numpy(dtype=float)
    positive_count = int(np.sum(effects > 0.0))
    if positive_count != 10:
        raise ValueError(f"Expected the audited C2 result 10/10, found {positive_count}/10.")
    ci_low, ci_high = _bootstrap_median_ci(effects)
    horizon_rows = []
    for horizon, frame in run_horizon.groupby("horizon_steps", sort=True):
        values = frame["mean_high_minus_low"].to_numpy(dtype=float)
        horizon_rows.append(
            {
                "horizon_steps": int(horizon),
                "n_runs": int(values.size),
                "n_positive": int(np.sum(values > 0.0)),
                "median_contrast": float(np.median(values)),
                "mean_contrast": float(np.mean(values)),
                "sign_test_greater_p": _sign_test_greater(values),
            }
        )
    pd.DataFrame(horizon_rows).to_csv(data_root / "c2_rng_only_horizon_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45), constrained_layout=True)
    left, right = axes
    x = np.asarray(HORIZONS, dtype=np.float64) / 1000.0
    pivot = run_horizon.pivot(
        index="run_idx", columns="horizon_steps", values="mean_high_minus_low"
    ).reindex(columns=HORIZONS)
    for run_idx, values in pivot.iterrows():
        left.plot(
            x,
            values.to_numpy(dtype=float) * 1000.0,
            color="#B0B6BB",
            linewidth=0.8,
            alpha=0.7,
            marker="o",
            markersize=2.4,
        )
    median_by_horizon = pivot.median(axis=0).to_numpy(dtype=float) * 1000.0
    left.plot(
        x,
        median_by_horizon,
        color=POSITIVE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        label="median across runs",
    )
    left.axhline(0.0, color="#202124", linewidth=0.8)
    left.set_xticks(x)
    left.set_xlabel("continuation horizon (thousand steps)")
    left.set_ylabel(r"high $-$ low CLIP--Chamfer ($\times 10^{-3}$)")
    left.set_title("A. Horizon dependence")
    left.legend(frameon=False, loc="upper left")

    run_ids = run_aggregate["run_idx"].astype(int).to_numpy()
    scaled_effects = effects * 1000.0
    colors = [POSITIVE_COLOR if value > 0.0 else NEGATIVE_COLOR for value in effects]
    right.bar(run_ids, scaled_effects, color=colors, width=0.72)
    right.axhline(0.0, color="#202124", linewidth=0.8)
    right.set_xticks(run_ids)
    right.set_xticklabels([f"{value:03d}" for value in run_ids])
    right.set_xlabel("source optimization run")
    right.set_ylabel(r"mean high $-$ low CLIP--Chamfer ($\times 10^{-3}$)")
    right.set_title("B. Run-level aggregate")
    right.text(
        0.98,
        0.95,
        "10/10 positive",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )
    fig.suptitle("Flow-Lenia C2: intrinsic RNG-only future divergence", fontsize=10.5)
    _save_figure(fig, figures_root, "flow_c2_pooled_and_within_paper")
    plt.close(fig)

    _build_c2_selection_preview(plan, data_root, figures_root)

    return {
        "n_runs": int(effects.size),
        "n_positive": positive_count,
        "median_contrast": float(np.median(effects)),
        "mean_contrast": float(np.mean(effects)),
        "bootstrap_median_ci_95": [ci_low, ci_high],
        "sign_test_greater_p": _sign_test_greater(effects),
        "wilcoxon_greater_p": _wilcoxon_greater(effects),
        "aggregation": (
            "per-run mean of 25 matched high-minus-low values "
            "(5 branch ranks x 5 horizons)"
        ),
        "metric": "symmetric mean cosine Chamfer (0.5 times the two directed means)",
        "branch_point_frame_excluded": True,
        "future_frames_per_horizon": 7,
        "external_state_noise": 0.0,
        "input_plan": str(plan_path),
    }


def _run_idx_from_name(path: Path) -> int:
    match = re.search(r"run_(\d{3})_seed_000", path.name)
    if match is None:
        raise ValueError(f"Could not parse run index from {path.name}.")
    return int(match.group(1))


def _build_c2_selection_preview(
    plan: pd.DataFrame,
    data_root: Path,
    figures_root: Path,
) -> None:
    plt = _matplotlib()
    unique_points = (
        plan.drop_duplicates(["run_idx", "condition", "pair_id"])[
            ["run_idx", "traj_id", "condition", "pair_id", "step", "delta_h"]
        ]
        .sort_values(["run_idx", "condition", "pair_id"])
        .reset_index(drop=True)
    )
    if len(unique_points) != 150:
        raise ValueError(f"Expected 150 C2 selected branch points, found {len(unique_points)}.")
    unique_points.to_csv(data_root / "c2_branch_selection_points.csv", index=False)

    metric_paths: dict[int, Path] = {}
    for path in C2_SOURCE_METRICS_ROOT.glob("*/metrics.npz"):
        metric_paths[_run_idx_from_name(path.parent)] = path
    if sorted(metric_paths) != list(range(10)):
        raise ValueError(f"Expected ten C2 source metric files, found {sorted(metric_paths)}.")

    panels: list[dict[str, Any]] = []
    finite_values = []
    heatmap_rows: list[dict[str, Any]] = []
    for run_idx in range(10):
        with np.load(metric_paths[run_idx], allow_pickle=False) as data:
            heatmap = np.asarray(data["delta_h_map"], dtype=np.float64)
            tau_steps = np.asarray(data["delta_h_tau_steps"], dtype=np.int64)
            centers = np.asarray(data["delta_h_window_center_steps"], dtype=np.int64)
        if heatmap.shape != (tau_steps.size, centers.size):
            raise ValueError(f"Unexpected C2 heatmap shape in {metric_paths[run_idx]}.")
        heatmap = np.maximum(heatmap, 0.0)
        finite_values.append(heatmap[np.isfinite(heatmap)])
        for tau_idx, tau_step in enumerate(tau_steps):
            for window_idx, center_step in enumerate(centers):
                heatmap_rows.append(
                    {
                        "run_idx": run_idx,
                        "tau_idx": tau_idx,
                        "tau_steps": int(tau_step),
                        "window_idx": window_idx,
                        "window_center_steps": int(center_step),
                        "processed_delta_h": float(heatmap[tau_idx, window_idx]),
                    }
                )
        panels.append(
            {
                "run_idx": run_idx,
                "heatmap": heatmap,
                "tau_steps": tau_steps,
                "centers": centers,
            }
        )
    pd.DataFrame(heatmap_rows).to_csv(
        data_root / "c2_branch_selection_heatmaps.csv",
        index=False,
    )
    vmax = float(np.quantile(np.concatenate(finite_values), 0.99))

    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.35), constrained_layout=True)
    last_image = None
    marker_meta = {
        "high": (HIGH_COLOR, "v", 8.9),
        "mid": (MID_COLOR, "o", 4.5),
        "low": (LOW_COLOR, "^", 0.1),
    }
    for panel, ax in zip(panels, axes.flat):
        run_idx = int(panel["run_idx"])
        heatmap = panel["heatmap"]
        tau_steps = panel["tau_steps"]
        centers = panel["centers"]
        last_image = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        points = unique_points[unique_points["run_idx"].astype(int) == run_idx]
        for point in points.itertuples(index=False):
            color, marker, y = marker_meta[str(point.condition)]
            window_idx = int(np.argmin(np.abs(centers - int(point.step))))
            ax.axvline(window_idx, color=color, linewidth=0.7, alpha=0.65)
            ax.scatter(
                [window_idx],
                [y],
                marker=marker,
                s=22,
                color=color,
                edgecolors="white",
                linewidths=0.3,
                zorder=4,
            )
        x_ticks = np.linspace(0, centers.size - 1, 4).astype(int)
        y_ticks = np.asarray([0, 3, 6, 9], dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{centers[idx] / 1000:.0f}" for idx in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tau_steps[idx] / 1000:.0f}" for idx in y_ticks])
        ax.set_title(f"opt_{run_idx:03d}")
        if run_idx >= 5:
            ax.set_xlabel("branch time (k steps)")
        if run_idx % 5 == 0:
            ax.set_ylabel(r"lag $\tau$ (k steps)")
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            shrink=0.78,
            pad=0.012,
            label=r"processed $\Delta H$",
        )
    from matplotlib.lines import Line2D

    legend = [
        Line2D([], [], marker="v", linestyle="none", color=HIGH_COLOR, label="high"),
        Line2D([], [], marker="o", linestyle="none", color=MID_COLOR, label="mid"),
        Line2D([], [], marker="^", linestyle="none", color=LOW_COLOR, label="low"),
    ]
    fig.legend(
        handles=legend,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
    )
    _save_figure(fig, figures_root, "c2_branching_branch_selection_preview")
    plt.close(fig)


def _build_c5(data_root: Path, figures_root: Path) -> dict[str, Any]:
    plt = _matplotlib()
    run_summary_path = C5_ROOT / "run_summary.csv"
    full = pd.read_csv(run_summary_path)
    table = full[full["horizon_steps"].astype(int) == 5000].copy()
    table = table.sort_values("run_idx").reset_index(drop=True)
    if table["run_idx"].astype(int).tolist() != list(range(10)):
        raise ValueError("C5 5k table must contain runs 000--009.")
    values = table["contrast_excess_clip_post_release"].to_numpy(dtype=float)
    positive_count = int(np.sum(values > 0.0))
    if positive_count != 8:
        raise ValueError(f"Expected the audited C5 result 8/10, found {positive_count}/10.")
    table.to_csv(data_root / "c5_5000_run_summary.csv", index=False)

    point_columns = [
        "run_idx",
        "trial_idx",
        "candidate_id",
        "candidate_kind",
        "candidate_idx",
        "condition",
        "point_id",
        "pair_id",
        "step",
        "delta_h",
        "horizon_steps",
        "release_step",
        "paired_same_seed_clip_post_release",
        "free_within_clip_post_release",
        "excess_clip_post_release",
    ]
    points = pd.read_csv(C5_ROOT / "point_metrics.csv", usecols=point_columns)
    points = points[points["horizon_steps"].astype(int) == 5000].copy()
    if len(points) != 600:
        raise ValueError(f"Expected 600 C5 5k point rows, found {len(points)}.")
    points.to_csv(data_root / "c5_5000_point_metrics.csv", index=False)

    candidate_columns = [
        "run_idx",
        "trial_idx",
        "candidate_id",
        "candidate_kind",
        "candidate_idx",
        "horizon_steps",
        "n_points",
        "paired_same_seed_clip_post_release",
        "free_within_clip_post_release",
        "excess_clip_post_release",
    ]
    candidates = pd.read_csv(
        C5_ROOT / "candidate_summary.csv",
        usecols=candidate_columns,
    )
    candidates = candidates[candidates["horizon_steps"].astype(int) == 5000].copy()
    if len(candidates) != 40:
        raise ValueError(f"Expected 40 C5 5k candidate rows, found {len(candidates)}.")
    candidates.to_csv(data_root / "c5_5000_candidate_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.15, 3.2), constrained_layout=True)
    run_ids = table["run_idx"].astype(int).to_numpy()
    scaled = values * 1000.0
    colors = [POSITIVE_COLOR if value > 0.0 else NEGATIVE_COLOR for value in values]
    ax.bar(run_ids, scaled, color=colors, width=0.72)
    ax.axhline(0.0, color="#202124", linewidth=0.8)
    ax.set_xticks(run_ids)
    ax.set_xticklabels([f"{value:03d}" for value in run_ids])
    ax.set_xlabel("optimization run")
    ax.set_ylabel(r"optimized $-$ random-median frustration ($\times 10^{-3}$)")
    ax.set_title(
        "Flow-Lenia C5: post-release excess CLIP--Chamfer contrast\n"
        "5k sensitivity horizon"
    )
    ax.text(
        0.02,
        0.95,
        "8/10 positive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    _save_figure(fig, figures_root, "flow_c5_frustration_paper")
    plt.close(fig)

    ci_low, ci_high = _bootstrap_median_ci(values)
    return {
        "horizon_steps": 5000,
        "horizon_role": "sensitivity",
        "n_runs": int(values.size),
        "n_positive": positive_count,
        "median_contrast": float(np.median(values)),
        "mean_contrast": float(np.mean(values)),
        "bootstrap_median_ci_95": [ci_low, ci_high],
        "sign_test_greater_p": _sign_test_greater(values),
        "wilcoxon_greater_p": _wilcoxon_greater(values),
        "metric": "contrast_excess_clip_post_release",
        "external_state_noise": 0.0,
        "input_run_summary": str(run_summary_path),
    }


def _copy_compatibility_aliases(figures_root: Path) -> None:
    aliases = {
        "flow_c1_paired_raw_paper.png": "flow_c1_paired_raw_clean.png",
        "flow_c2_pooled_and_within_paper.png": "flow_c2_clip_chamfer_association.png",
        "flow_c5_frustration_paper.png": "flow_c5_frustration_clean.png",
    }
    for source_name, target_name in aliases.items():
        shutil.copy2(figures_root / source_name, figures_root / target_name)


def run(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    figures_root = output_root / "figures"
    data_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    c2_summary = _build_c2(data_root, figures_root)
    c2_audit_path = data_root / "c2_branch_point_equality_audit.json"
    if not c2_audit_path.exists():
        raise FileNotFoundError(
            "Missing C2 branch-point audit. Run "
            "`python scripts/audit_flowlenia_c2_branch_points.py` first."
        )
    c2_audit = json.loads(c2_audit_path.read_text())
    if (
        not bool(c2_audit.get("all_start_states_bit_exact"))
        or int(c2_audit.get("n_branches", -1)) != 450
        or float(c2_audit.get("max_a_abs_diff", float("inf"))) != 0.0
        or float(c2_audit.get("max_p_abs_diff", float("inf"))) != 0.0
    ):
        raise ValueError("C2 branch-point equality audit did not pass.")
    c2_summary["branch_point_equality_audit"] = c2_audit

    summary = {
        "build": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "article_source_modified": False,
        },
        "c1": _build_c1(data_root, figures_root),
        "c2": c2_summary,
        "c5": _build_c5(data_root, figures_root),
    }
    _copy_compatibility_aliases(figures_root)
    _write_json(data_root / "flowlenia_revision_summary.json", summary)

    manifest_rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        manifest_rows.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(data_root / "artifact_manifest.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Flow-Lenia article replacement figures and audited tables from "
            "the completed C1/C2/C5 artifacts without rerunning simulations."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    result = run(output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
