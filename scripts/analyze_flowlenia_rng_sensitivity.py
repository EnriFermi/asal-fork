from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

import flowlenia_rng_sensitivity_experiment as sim


ANALYSIS_VERSION = "flowlenia-rng-sensitivity-analysis-v1"
METRICS = (
    "a_relative_l1",
    "p_mass_weighted_l1",
    "render_l1",
    "flow_relative_l1",
    "mass_relative",
)
METRIC_LABELS = {
    "a_relative_l1": "Mass-normalized A divergence",
    "p_mass_weighted_l1": "Mass-weighted P divergence",
    "render_l1": "Rendered RGB divergence",
    "flow_relative_l1": "Normalized flow-field divergence",
    "mass_relative": "Relative total-mass divergence",
}
OPT_COLOR = "#D55E00"
RANDOM_COLOR = "#4C78A8"
SHARED_COLOR = "#009E73"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _batch_path(root: Path, candidate_id: str, batch_idx: int) -> Path:
    return sim._batch_output_path(root, candidate_id, batch_idx)


def _load_manifests(root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    protocol = sim.load_protocol(root)
    candidates = pd.read_csv(root / "candidates.csv")
    contexts = pd.read_csv(root / "contexts.csv", keep_default_na=False)
    if len(candidates) != 40 or candidates["candidate_kind"].value_counts().to_dict() != {
        "random": 30,
        "optimized": 10,
    }:
        raise RuntimeError("Candidate manifest does not contain 10 optimized + 30 random")
    if len(contexts) != 40 * sim.N_CONTEXTS:
        raise RuntimeError("Context manifest has an unexpected number of rows")
    return protocol, candidates, contexts


def _validate_all_outputs(
    root: Path,
    protocol: dict[str, Any],
    candidates: pd.DataFrame,
    contexts: pd.DataFrame,
) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    max_duplicate = 0.0
    max_t0 = 0.0
    elapsed = 0.0
    shared_hashes: dict[int, set[str]] = {
        idx: set() for idx in range(20, 24)
    }
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        rows = contexts[contexts["candidate_id"] == candidate_id].sort_values(
            "context_idx"
        )
        for batch_idx in range(2):
            batch_rows = rows.iloc[
                batch_idx * sim.CONTEXTS_PER_BATCH :
                (batch_idx + 1) * sim.CONTEXTS_PER_BATCH
            ]
            expected = batch_rows["context_idx"].astype(int).tolist()
            path = _batch_path(root, candidate_id, batch_idx)
            if not path.exists():
                missing.append(str(path))
                continue
            candidate_str = {key: str(value) for key, value in candidate.items()}
            if not sim._validate_batch_output(
                path,
                candidate=candidate_str,
                protocol=protocol,
                expected_context_indices=expected,
            ):
                invalid.append(str(path))
                continue
            with np.load(path, allow_pickle=False) as data:
                elapsed += float(data["elapsed_seconds"])
                max_duplicate = max(
                    max_duplicate,
                    *(
                        float(np.max(data[key]))
                        for key in (
                            "duplicate_a_max_abs",
                            "duplicate_p_max_abs",
                            "duplicate_f_max_abs",
                            "duplicate_rng_max_abs",
                        )
                    ),
                )
                max_t0 = max(
                    max_t0,
                    *(float(np.max(data[key][:, 0])) for key in METRICS),
                )
                for context_idx, state_hash in zip(
                    np.asarray(data["context_indices"], dtype=int),
                    np.asarray(data["source_state_hashes"]).astype(str),
                    strict=True,
                ):
                    if int(context_idx) >= 20:
                        shared_hashes[int(context_idx)].add(str(state_hash))
    shared_counts = {str(key): len(value) for key, value in shared_hashes.items()}
    status = (
        "passed"
        if not missing
        and not invalid
        and max_duplicate == 0.0
        and max_t0 == 0.0
        and all(count == 1 for count in shared_counts.values())
        else "failed"
    )
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "missing_outputs": missing,
        "invalid_outputs": invalid,
        "valid_batch_count": 80 - len(missing) - len(invalid),
        "expected_batch_count": 80,
        "max_duplicate_distance": max_duplicate,
        "max_t0_pair_distance": max_t0,
        "shared_state_hash_counts_by_context": shared_counts,
        "shared_states_identical_across_all_candidates": all(
            count == 1 for count in shared_counts.values()
        ),
        "summed_gpu_batch_seconds": elapsed,
    }
    _write_json(root / "analysis_input_audit.json", audit)
    if status != "passed":
        raise RuntimeError(f"Simulation input audit failed: {audit}")
    return audit


def _normalized_auc(steps: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapz(values, steps) / float(steps[-1] - steps[0]))


def build_tables(root: Path) -> dict[str, Any]:
    protocol, candidates, contexts = _load_manifests(root)
    input_audit = _validate_all_outputs(root, protocol, candidates, contexts)
    candidate_lookup = candidates.set_index("candidate_id").to_dict("index")
    context_lookup = contexts.set_index(["candidate_id", "context_idx"]).to_dict(
        "index"
    )

    context_curve_rows: list[dict[str, Any]] = []
    context_score_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        for batch_idx in range(2):
            path = _batch_path(root, candidate_id, batch_idx)
            with np.load(path, allow_pickle=False) as data:
                steps = np.asarray(data["metric_steps"], dtype=np.int32)
                context_indices = np.asarray(data["context_indices"], dtype=np.int32)
                pair_medians = {
                    metric: np.median(np.asarray(data[metric], dtype=np.float64), axis=2)
                    for metric in METRICS
                }
                for local_idx, context_idx in enumerate(context_indices):
                    context_meta = context_lookup[(candidate_id, int(context_idx))]
                    base = {
                        "candidate_id": candidate_id,
                        "candidate_kind": candidate["candidate_kind"],
                        "run_idx": int(candidate["run_idx"]),
                        "candidate_idx": int(candidate["candidate_idx"]),
                        "context_idx": int(context_idx),
                        "arm": context_meta["arm"],
                        "rollout_seed_idx": int(context_meta["rollout_seed_idx"]),
                        "anchor_idx": int(context_meta["anchor_idx"]),
                        "source_step": int(context_meta["source_step"]),
                    }
                    for step_idx, step in enumerate(steps):
                        context_curve_rows.append(
                            {
                                **base,
                                "distance_steps": int(step),
                                **{
                                    metric: float(pair_medians[metric][local_idx, step_idx])
                                    for metric in METRICS
                                },
                            }
                        )
                    context_score_rows.append(
                        {
                            **base,
                            **{
                                f"{metric}_auc": _normalized_auc(
                                    steps, pair_medians[metric][local_idx]
                                )
                                for metric in METRICS
                            },
                            **{
                                f"{metric}_final": float(pair_medians[metric][local_idx, -1])
                                for metric in METRICS
                            },
                        }
                    )

    context_curves = pd.DataFrame(context_curve_rows)
    context_scores = pd.DataFrame(context_score_rows)
    candidate_curves = (
        context_curves.groupby(
            ["candidate_id", "candidate_kind", "run_idx", "candidate_idx", "arm", "distance_steps"],
            as_index=False,
        )[list(METRICS)]
        .mean()
        .sort_values(["candidate_id", "arm", "distance_steps"])
    )

    candidate_score_rows: list[dict[str, Any]] = []
    for (candidate_id, arm), frame in candidate_curves.groupby(
        ["candidate_id", "arm"], sort=False
    ):
        frame = frame.sort_values("distance_steps")
        steps = frame["distance_steps"].to_numpy(dtype=np.float64)
        candidate = candidate_lookup[candidate_id]
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_kind": candidate["candidate_kind"],
            "run_idx": int(candidate["run_idx"]),
            "candidate_idx": int(candidate["candidate_idx"]),
            "arm": arm,
            "n_contexts": 20 if arm == "trajectory" else 4,
            "c1_mspd_mean": float(candidate["c1_mspd_mean"]),
            "c1_mspd_median": float(candidate["c1_mspd_median"]),
        }
        early = steps <= 1_000
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_auc"] = _normalized_auc(steps, values)
            row[f"{metric}_early_1k_auc"] = _normalized_auc(
                steps[early], values[early]
            )
            row[f"{metric}_final"] = float(values[-1])
            row[f"{metric}_at_50"] = float(values[np.flatnonzero(steps == 50)[0]])
            row[f"{metric}_at_1000"] = float(values[np.flatnonzero(steps == 1_000)[0]])
        a_values = frame["a_relative_l1"].to_numpy(dtype=np.float64)
        threshold_hits = np.flatnonzero(a_values >= 0.5)
        row["a_relative_l1_time_to_0p5"] = (
            float(steps[threshold_hits[0]])
            if threshold_hits.size
            else float(sim.HORIZON_STEPS + sim.STEP_CHUNK)
        )
        candidate_score_rows.append(row)
    candidate_scores = pd.DataFrame(candidate_score_rows)

    _atomic_csv(context_curves, root / "context_divergence_curves.csv")
    _atomic_csv(context_scores, root / "context_sensitivity_scores.csv")
    _atomic_csv(candidate_curves, root / "candidate_divergence_curves.csv")
    _atomic_csv(candidate_scores, root / "candidate_sensitivity_scores.csv")

    statistics = run_statistics(root, candidate_scores, candidate_curves)
    correlation = run_correlations(root, candidate_scores)
    return {
        "input_audit": input_audit,
        "statistics": statistics,
        "correlation": correlation,
        "n_context_curve_rows": len(context_curves),
        "n_candidate_curve_rows": len(candidate_curves),
    }


def _bootstrap_difference(
    optimized: np.ndarray,
    random: np.ndarray,
    *,
    statistic: str,
    reps: int = 50_000,
    seed: int = 91_337,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    reducer = np.mean if statistic == "mean" else np.median
    chunk = 2_000
    for start in range(0, reps, chunk):
        n = min(chunk, reps - start)
        opt_sample = optimized[rng.integers(0, len(optimized), size=(n, len(optimized)))]
        rand_sample = random[rng.integers(0, len(random), size=(n, len(random)))]
        values[start : start + n] = reducer(opt_sample, axis=1) - reducer(
            rand_sample, axis=1
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _permutation_p_greater(
    optimized: np.ndarray,
    random: np.ndarray,
    *,
    reps: int = 200_000,
    seed: int = 78_331,
) -> float:
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([optimized, random])
    observed = float(np.mean(optimized) - np.mean(random))
    exceed = 0
    for _ in range(reps):
        perm = rng.permutation(len(pooled))
        diff = float(
            np.mean(pooled[perm[: len(optimized)]])
            - np.mean(pooled[perm[len(optimized) :]])
        )
        exceed += diff >= observed
    return float((exceed + 1) / (reps + 1))


def _group_test(
    frame: pd.DataFrame,
    column: str,
    *,
    permutation: bool,
) -> dict[str, Any]:
    from scipy import stats

    optimized = frame.loc[
        frame["candidate_kind"] == "optimized", column
    ].to_numpy(dtype=np.float64)
    random = frame.loc[frame["candidate_kind"] == "random", column].to_numpy(
        dtype=np.float64
    )
    if len(optimized) != 10 or len(random) != 30:
        raise RuntimeError(f"Unexpected group sizes for {column}")
    u, p_mw = stats.mannwhitneyu(
        optimized, random, alternative="greater", method="exact"
    )
    welch = stats.ttest_ind(optimized, random, equal_var=False, alternative="greater")
    mean_ci = _bootstrap_difference(optimized, random, statistic="mean")
    median_ci = _bootstrap_difference(
        optimized, random, statistic="median", seed=91_338
    )
    return {
        "column": column,
        "n_optimized": len(optimized),
        "n_random": len(random),
        "optimized_mean": float(np.mean(optimized)),
        "random_mean": float(np.mean(random)),
        "mean_difference": float(np.mean(optimized) - np.mean(random)),
        "mean_difference_ci95_low": mean_ci[0],
        "mean_difference_ci95_high": mean_ci[1],
        "optimized_median": float(np.median(optimized)),
        "random_median": float(np.median(random)),
        "median_difference": float(np.median(optimized) - np.median(random)),
        "median_difference_ci95_low": median_ci[0],
        "median_difference_ci95_high": median_ci[1],
        "mann_whitney_u": float(u),
        "mann_whitney_exact_one_sided_p": float(p_mw),
        "rank_biserial": float(2.0 * u / (len(optimized) * len(random)) - 1.0),
        "welch_t": float(welch.statistic),
        "welch_one_sided_p": float(welch.pvalue),
        "permutation_mean_one_sided_p": (
            _permutation_p_greater(optimized, random) if permutation else None
        ),
    }


def run_statistics(
    root: Path,
    candidate_scores: pd.DataFrame,
    candidate_curves: pd.DataFrame,
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for arm in ("trajectory", "shared_state"):
        frame = candidate_scores[candidate_scores["arm"] == arm]
        for metric in METRICS:
            for suffix in ("auc", "early_1k_auc", "final"):
                column = f"{metric}_{suffix}"
                test = _group_test(
                    frame,
                    column,
                    permutation=(
                        arm == sim.PRIMARY_ARM
                        and metric == sim.PRIMARY_METRIC
                        and suffix == "auc"
                    ),
                )
                test.update(
                    {
                        "arm": arm,
                        "metric": metric,
                        "endpoint": suffix,
                        "primary": bool(
                            arm == sim.PRIMARY_ARM
                            and metric == sim.PRIMARY_METRIC
                            and suffix == "auc"
                        ),
                    }
                )
                tests.append(test)

    horizon_rows: list[dict[str, Any]] = []
    for arm in ("trajectory", "shared_state"):
        for step in sim.HORIZON_SUMMARY_STEPS:
            frame = candidate_curves[
                (candidate_curves["arm"] == arm)
                & (candidate_curves["distance_steps"] == step)
            ]
            test = _group_test(frame, sim.PRIMARY_METRIC, permutation=False)
            horizon_rows.append(
                {
                    "arm": arm,
                    "distance_steps": int(step),
                    **test,
                }
            )
    tests_frame = pd.DataFrame(tests)
    horizons_frame = pd.DataFrame(horizon_rows)
    _atomic_csv(tests_frame, root / "group_statistical_tests.csv")
    _atomic_csv(horizons_frame, root / "horizon_statistical_tests.csv")
    primary = next(row for row in tests if row["primary"])
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "primary_endpoint": (
            "trajectory-arm candidate mean over 20 contexts of normalized "
            "0..10k A-relative-L1 AUC"
        ),
        "primary_test": primary,
        "all_tests_are_secondary_except_primary": True,
        "n_tests_reported": len(tests),
        "interpretation_rule": (
            "Optimization-increased sensitivity is supported only if the "
            "pre-registered one-sided primary p-value is below 0.05 and the "
            "effect direction is positive."
        ),
    }
    _write_json(root / "statistical_summary.json", summary)
    return summary


def run_correlations(root: Path, candidate_scores: pd.DataFrame) -> dict[str, Any]:
    from scipy import stats

    rows: list[dict[str, Any]] = []
    for arm in ("trajectory", "shared_state"):
        frame = candidate_scores[candidate_scores["arm"] == arm]
        for subset_name, subset in (
            ("all", frame),
            ("optimized", frame[frame["candidate_kind"] == "optimized"]),
            ("random", frame[frame["candidate_kind"] == "random"]),
        ):
            x = subset["c1_mspd_mean"].to_numpy(dtype=np.float64)
            y = subset[f"{sim.PRIMARY_METRIC}_auc"].to_numpy(dtype=np.float64)
            spearman = stats.spearmanr(x, y)
            kendall = stats.kendalltau(x, y)
            rows.append(
                {
                    "arm": arm,
                    "subset": subset_name,
                    "n": len(subset),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_two_sided": float(spearman.pvalue),
                    "kendall_tau": float(kendall.statistic),
                    "kendall_p_two_sided": float(kendall.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, root / "mspd_sensitivity_correlations.csv")
    summary = {"analysis_version": ANALYSIS_VERSION, "tests": rows}
    _write_json(root / "correlation_summary.json", summary)
    return summary


def _bootstrap_curve(values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reps = 20_000
    result = np.empty((reps, values.shape[1]), dtype=np.float32)
    chunk = 1_000
    for start in range(0, reps, chunk):
        n = min(chunk, reps - start)
        indices = rng.integers(0, values.shape[0], size=(n, values.shape[0]))
        result[start : start + n] = np.mean(values[indices], axis=1)
    low, high = np.quantile(result, [0.025, 0.975], axis=0)
    return low, high


def _save_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def make_plots(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate_curves = pd.read_csv(root / "candidate_divergence_curves.csv")
    candidate_scores = pd.read_csv(root / "candidate_sensitivity_scores.csv")
    context_scores = pd.read_csv(root / "context_sensitivity_scores.csv")
    stats_summary = json.loads((root / "statistical_summary.json").read_text())
    figure_dir = root / "figures"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for row_idx, arm in enumerate(("trajectory", "shared_state")):
        arm_frame = candidate_curves[candidate_curves["arm"] == arm]
        for col_idx, max_step in enumerate((1_000, sim.HORIZON_STEPS)):
            ax = axes[row_idx, col_idx]
            for kind, color, label, seed in (
                ("random", RANDOM_COLOR, "Random", 41 + row_idx),
                ("optimized", OPT_COLOR, "Optimized", 61 + row_idx),
            ):
                frame = arm_frame[arm_frame["candidate_kind"] == kind]
                pivot = frame.pivot(
                    index="candidate_id", columns="distance_steps", values=sim.PRIMARY_METRIC
                ).sort_index(axis=1)
                steps = pivot.columns.to_numpy(dtype=float)
                values = pivot.to_numpy(dtype=float)
                mean = np.mean(values, axis=0)
                low, high = _bootstrap_curve(values, seed)
                mask = steps <= max_step
                ax.plot(steps[mask], mean[mask], color=color, lw=2, label=label)
                ax.fill_between(steps[mask], low[mask], high[mask], color=color, alpha=0.18)
            ax.set_xlim(0, max_step)
            ax.set_xlabel("Steps after RNG fork")
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[sim.PRIMARY_METRIC])
            ax.set_title(
                f"{'Visited C1 states' if arm == 'trajectory' else 'Shared initial states'}: "
                f"0-{max_step // 1000 if max_step >= 1000 else max_step}k"
            )
            ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "rng_divergence_curves")
    plt.close(fig)

    primary = candidate_scores[candidate_scores["arm"] == "trajectory"].copy()
    score_col = f"{sim.PRIMARY_METRIC}_auc"
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    rng = np.random.default_rng(1_903)
    for x, (kind, color, label) in enumerate(
        (("random", RANDOM_COLOR, "Random"), ("optimized", OPT_COLOR, "Optimized"))
    ):
        values = primary.loc[primary["candidate_kind"] == kind, score_col].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=38, color=color, alpha=0.85, label=label)
        ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="black", lw=2)
    test = stats_summary["primary_test"]
    ax.set_xticks([0, 1], ["Random\n(n=30)", "Optimized\n(n=10)"])
    ax.set_ylabel("Candidate RNG-sensitivity score\n(normalized A-divergence AUC)")
    ax.set_title(
        f"Exact one-sided Mann-Whitney p={test['mann_whitney_exact_one_sided_p']:.4g}; "
        f"r={test['rank_biserial']:+.2f}"
    )
    fig.tight_layout()
    _save_figure(fig, figure_dir / "rng_sensitivity_all_candidates")
    plt.close(fig)

    ordered_candidates = primary.sort_values(
        ["candidate_kind", "run_idx", "candidate_idx"],
        ascending=[False, True, True],
    )["candidate_id"].tolist()
    heat = context_scores.pivot(
        index="candidate_id", columns="context_idx", values=f"{sim.PRIMARY_METRIC}_auc"
    ).loc[ordered_candidates]
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(heat)), heat.index, fontsize=7)
    labels = [
        (f"s{idx // 5}@{sim.SOURCE_STEPS[idx % 5] // 1000}k" if idx < 20 else f"shared{idx - 20}")
        for idx in heat.columns.astype(int)
    ]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Fork context")
    ax.set_title("RNG sensitivity by candidate and fork state")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Normalized A-divergence AUC")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "rng_sensitivity_context_heatmap")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, arm in zip(axes, ("trajectory", "shared_state"), strict=True):
        frame = candidate_scores[candidate_scores["arm"] == arm]
        for kind, color, label in (
            ("random", RANDOM_COLOR, "Random"),
            ("optimized", OPT_COLOR, "Optimized"),
        ):
            subset = frame[frame["candidate_kind"] == kind]
            ax.scatter(
                subset["c1_mspd_mean"], subset[score_col], color=color, s=35, alpha=0.85, label=label
            )
        ax.set_xscale("log")
        ax.set_xlabel("Mean C1 MSPD")
        ax.set_ylabel("RNG-sensitivity score")
        ax.set_title("Visited C1 states" if arm == "trajectory" else "Shared initial states")
        ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "c1_mspd_vs_rng_sensitivity")
    plt.close(fig)

    horizons = pd.read_csv(root / "horizon_statistical_tests.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for arm, color, label in (
        ("trajectory", OPT_COLOR, "Visited C1 states"),
        ("shared_state", SHARED_COLOR, "Shared initial states"),
    ):
        frame = horizons[horizons["arm"] == arm]
        ax.plot(
            frame["distance_steps"], frame["mean_difference"], marker="o", color=color, label=label
        )
        ax.fill_between(
            frame["distance_steps"],
            frame["mean_difference_ci95_low"],
            frame["mean_difference_ci95_high"],
            color=color,
            alpha=0.16,
        )
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("Steps after RNG fork")
    ax.set_ylabel("Optimized - random mean A divergence")
    ax.set_title("Optimization effect across intermediate horizons")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "optimization_effect_by_horizon")
    plt.close(fig)


def _render_reference(context: pd.Series, visual_steps: np.ndarray) -> np.ndarray:
    path = Path(str(context["source_chunk_path"]))
    source_step = int(context["source_step"])
    wanted = source_step + visual_steps
    with np.load(path, allow_pickle=False) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        indices = []
        for step in wanted:
            hit = np.flatnonzero(steps == int(step))
            if hit.size != 1:
                raise RuntimeError(f"Reference step {step} not found in {path}")
            indices.append(int(hit[0]))
        A = np.asarray(data["A"][indices], dtype=np.float32)
        P = np.asarray(data["P"][indices], dtype=np.float32)
    rgb = np.clip(A.sum(axis=-1, keepdims=True) * P[..., :3], 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def _font(size: int):
    from PIL import ImageFont

    paths = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _video_grid_frame(
    frames: list[np.ndarray],
    labels: list[str],
    *,
    title: str,
    tile_size: int = 256,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    cols, rows = 4, 3
    title_h = 36
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black", font=_font(16))
    for idx, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        x = (idx % cols) * tile_size
        y = title_h + (idx // cols) * tile_size
        tile = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        draw.rectangle((x + 4, y + 4, x + 142, y + 28), fill=(0, 0, 0))
        draw.text((x + 9, y + 7), label, fill="white", font=_font(14))
    return np.asarray(canvas, dtype=np.uint8)


def make_videos(
    root: Path,
    *,
    fps: int = 12,
    candidate_ids: set[str] | None = None,
) -> None:
    import imageio.v2 as imageio

    _protocol, candidates, contexts = _load_manifests(root)
    video_dir = root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    if candidate_ids is not None:
        unknown = candidate_ids.difference(candidates["candidate_id"].astype(str))
        if unknown:
            raise RuntimeError(f"Unknown video candidates: {sorted(unknown)}")
        candidates = candidates[candidates["candidate_id"].isin(candidate_ids)]
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        for batch_idx, arm, visual_context_idx in (
            (0, "trajectory", 2),
            (1, "shared_state", 20),
        ):
            output = video_dir / f"{candidate_id}_{arm}.mp4"
            if output.exists() and output.stat().st_size > 10_000:
                continue
            batch_path = _batch_path(root, candidate_id, batch_idx)
            with np.load(batch_path, allow_pickle=False) as data:
                visual_steps = np.asarray(data["visual_steps"], dtype=np.int32)
                visual_rgb = np.asarray(data["visual_rgb"], dtype=np.uint8)
                stored_context_idx = int(np.asarray(data["visual_context_idx"]).item())
            if stored_context_idx != visual_context_idx:
                raise RuntimeError(f"Unexpected visual context in {batch_path}")
            context = contexts[
                (contexts["candidate_id"] == candidate_id)
                & (contexts["context_idx"].astype(int) == visual_context_idx)
            ].iloc[0]
            reference = (
                _render_reference(context, visual_steps)
                if arm == "trajectory"
                else None
            )
            if reference is not None and not np.array_equal(reference[0], visual_rgb[0, 0]):
                raise RuntimeError(f"Fork origin does not match C1 reference for {candidate_id}")
            writer = imageio.get_writer(
                output,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                quality=8,
            )
            try:
                for time_idx, distance in enumerate(visual_steps):
                    frames: list[np.ndarray] = []
                    labels: list[str] = []
                    if reference is not None:
                        frames.append(reference[time_idx])
                        labels.append("C1 reference")
                    for branch_idx in range(sim.N_UNIQUE_BRANCHES):
                        frames.append(visual_rgb[branch_idx, time_idx])
                        labels.append(f"RNG {branch_idx}")
                    frames.append(visual_rgb[sim.DUPLICATE_BRANCH, time_idx])
                    labels.append("RNG 0 duplicate")
                    frame = _video_grid_frame(
                        frames,
                        labels,
                        title=(
                            f"{candidate_id} | {arm} | "
                            f"{int(distance):,} steps after fork"
                        ),
                    )
                    repeats = 6 if time_idx in (0, len(visual_steps) - 1) else 1
                    for _ in range(repeats):
                        writer.append_data(frame)
            finally:
                writer.close()


def write_report(root: Path) -> None:
    stats_summary = json.loads((root / "statistical_summary.json").read_text())
    correlations = pd.read_csv(root / "mspd_sensitivity_correlations.csv")
    primary = stats_summary["primary_test"]
    all_corr = correlations[
        (correlations["arm"] == "trajectory") & (correlations["subset"] == "all")
    ].iloc[0]
    supported = (
        primary["mean_difference"] > 0
        and primary["mann_whitney_exact_one_sided_p"] < 0.05
    )
    text = f"""# FlowLenia RNG-sensitivity experiment

## Protocol

- 10 optimized and 30 independently sampled existing random parameter candidates.
- 20 visited C1 states per candidate: four rollout seeds x steps 50k, 100k, 150k, 200k, 250k.
- Four additional physical states shared exactly across all 40 candidates.
- Nine unique continuation RNG streams and one bit-exact duplicate control.
- No state perturbation or external noise; horizon 10,000; metrics every 50 steps.

## Primary result

- Optimized mean AUC: `{primary['optimized_mean']:.8g}`
- Random mean AUC: `{primary['random_mean']:.8g}`
- Mean difference: `{primary['mean_difference']:+.8g}` (95% bootstrap CI `{primary['mean_difference_ci95_low']:+.8g}`, `{primary['mean_difference_ci95_high']:+.8g}`)
- Exact one-sided Mann-Whitney: `U={primary['mann_whitney_u']:.1f}`, `p={primary['mann_whitney_exact_one_sided_p']:.8g}`
- Rank-biserial effect: `{primary['rank_biserial']:+.4f}`
- Pre-registered optimization-increased sensitivity hypothesis supported: **{str(supported).lower()}**

## MSPD association

- Across all 40 candidates, Kendall tau between mean C1 MSPD and trajectory sensitivity: `{all_corr['kendall_tau']:+.4f}` (`p={all_corr['kendall_p_two_sided']:.8g}`).
- Spearman rho: `{all_corr['spearman_rho']:+.4f}` (`p={all_corr['spearman_p_two_sided']:.8g}`).

## Controls

- Every RNG branch starts from a bit-exact common state.
- Branch 9 duplicates branch 0 and remains bit-exact through 10,000 steps.
- Shared physical states are hash-identical across all 40 candidates.

The experiment establishes stochastic/RNG-induced trajectory sensitivity. It is not, by itself, a deterministic Lyapunov-chaos test.
"""
    (root / "RNG_SENSITIVITY_REPORT.md").write_text(text, encoding="utf-8")


def write_runtime_environment(root: Path) -> dict[str, Any]:
    import jax
    import jaxlib

    def command_text(command: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    environment = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "ptxas_version": command_text(["ptxas", "--version"]),
        "nvidia_smi": command_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    _write_json(root / "runtime_environment.json", environment)
    return environment


def final_audit(root: Path) -> dict[str, Any]:
    protocol, candidates, contexts = _load_manifests(root)
    input_audit = _validate_all_outputs(root, protocol, candidates, contexts)
    required = [
        root / "context_divergence_curves.csv",
        root / "context_sensitivity_scores.csv",
        root / "candidate_divergence_curves.csv",
        root / "candidate_sensitivity_scores.csv",
        root / "group_statistical_tests.csv",
        root / "horizon_statistical_tests.csv",
        root / "mspd_sensitivity_correlations.csv",
        root / "statistical_summary.json",
        root / "RNG_SENSITIVITY_REPORT.md",
        root / "runtime_environment.json",
    ]
    figure_bases = (
        "rng_divergence_curves",
        "rng_sensitivity_all_candidates",
        "rng_sensitivity_context_heatmap",
        "c1_mspd_vs_rng_sensitivity",
        "optimization_effect_by_horizon",
    )
    for name in figure_bases:
        required.extend([root / "figures" / f"{name}.png", root / "figures" / f"{name}.pdf"])
    required.extend(
        root / "videos" / f"{candidate_id}_{arm}.mp4"
        for candidate_id in candidates["candidate_id"]
        for arm in ("trajectory", "shared_state")
    )
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    audit = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete" if not missing and input_audit["status"] == "passed" else "failed",
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol["code_bundle_sha256"],
        "simulation_batches": input_audit["valid_batch_count"],
        "metric_timepoints": len(sim.METRIC_STEPS),
        "candidate_count": len(candidates),
        "video_count": sum(path.suffix == ".mp4" for path in required),
        "missing_artifacts": missing,
    }
    _write_json(root / "completion_audit.json", audit)
    if audit["status"] != "complete":
        raise RuntimeError(f"Completion audit failed: {audit}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FlowLenia RNG sensitivity.")
    parser.add_argument(
        "stage", choices=("analyze", "plots", "videos", "audit", "all")
    )
    parser.add_argument("--output-root", default=str(sim.DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--candidate-id", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    root = sim._resolve(cli.output_root)
    if cli.stage in ("analyze", "all"):
        summary = build_tables(root)
        print(json.dumps(summary["statistics"]["primary_test"], indent=2), flush=True)
    if cli.stage in ("plots", "all"):
        make_plots(root)
    if cli.stage in ("videos", "all"):
        make_videos(
            root,
            fps=int(cli.video_fps),
            candidate_ids=set(cli.candidate_id) if cli.candidate_id else None,
        )
    if cli.stage in ("audit", "all"):
        write_report(root)
        write_runtime_environment(root)
        audit = final_audit(root)
        print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
