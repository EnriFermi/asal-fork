from __future__ import annotations

import argparse
import hashlib
import json
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

import flowlenia_rng_sensitivity_clip_simulation as sim
import flowlenia_rng_sensitivity_experiment as base


ANALYSIS_VERSION = "flowlenia-rng-sensitivity-clip-chamfer-analysis-v1"
DEFAULT_OUTPUT_ROOT = sim.DEFAULT_OUTPUT_ROOT
PAIR_COLUMNS = (
    "candidate_id",
    "candidate_kind",
    "run_idx",
    "candidate_idx",
    "context_idx",
    "arm",
    "rollout_seed_idx",
    "anchor_idx",
    "source_step",
    "horizon_steps",
    "pair_idx",
    "branch_left",
    "branch_right",
    "clip_chamfer",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return value.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    tmp.replace(path)


def _analysis_fingerprint() -> dict[str, Any]:
    files = {
        str(path.relative_to(_REPO_ROOT)): _sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            (_REPO_ROOT / "scripts/paper_suite_c2_branching.py").resolve(),
        )
    }
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"analysis_version": ANALYSIS_VERSION, "files": files, "identity_sha256": identity}


def _batch_path(root: Path, candidate_id: str, batch_idx: int) -> Path:
    return root / "simulation" / candidate_id / f"batch_{batch_idx:02d}.npz"


def _pairwise_chamfer(
    embeddings: np.ndarray,
    capture_steps: np.ndarray,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    values /= np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12, None)
    step_to_index = {int(step): index for index, step in enumerate(capture_steps)}
    result = np.empty(
        (
            values.shape[0],
            len(sim.HORIZONS),
            base.N_PAIRS,
        ),
        dtype=np.float64,
    )
    for horizon_idx, horizon in enumerate(sim.HORIZONS):
        frame_indices = np.asarray(
            [step_to_index[step] for step in sim.HORIZON_OFFSETS[horizon]],
            dtype=np.int64,
        )
        cloud = values[:, :, frame_indices, :]
        left = cloud[:, base.PAIR_LEFT]
        right = cloud[:, base.PAIR_RIGHT]
        distance = 1.0 - np.einsum(
            "cpid,cpjd->cpij", left, right, optimize=True
        )
        left_to_right = np.mean(np.min(distance, axis=3), axis=2)
        right_to_left = np.mean(np.min(distance, axis=2), axis=2)
        result[:, horizon_idx] = 0.5 * (left_to_right + right_to_left)
    return result


def _verify_chamfer_implementation(
    embeddings: np.ndarray,
    capture_steps: np.ndarray,
    pair_values: np.ndarray,
) -> float:
    step_to_index = {int(step): index for index, step in enumerate(capture_steps)}
    checks = []
    for context_idx, horizon_idx, pair_idx in (
        (0, 0, 0),
        (min(3, embeddings.shape[0] - 1), 4, 17),
        (embeddings.shape[0] - 1, len(sim.HORIZONS) - 1, base.N_PAIRS - 1),
    ):
        horizon = sim.HORIZONS[horizon_idx]
        indices = [step_to_index[step] for step in sim.HORIZON_OFFSETS[horizon]]
        expected = sim._chamfer(
            embeddings[context_idx, base.PAIR_LEFT[pair_idx], indices],
            embeddings[context_idx, base.PAIR_RIGHT[pair_idx], indices],
        )
        checks.append(abs(expected - pair_values[context_idx, horizon_idx, pair_idx]))
    return float(max(checks))


def build_tables(root: Path, *, require_complete: bool) -> dict[str, Any]:
    simulation_audit = sim.audit(root, require_complete=require_complete)
    protocol = sim.load_protocol(root)
    candidates = pd.read_csv(root / "candidates.csv")
    contexts = pd.read_csv(root / "contexts.csv")
    context_lookup = contexts.set_index(["candidate_id", "context_idx"])
    candidate_lookup = candidates.set_index("candidate_id")

    pair_rows: list[tuple[Any, ...]] = []
    implementation_max_abs = 0.0
    processed_batches = 0
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        for batch_idx in range(2):
            path = _batch_path(root, candidate_id, batch_idx)
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            with np.load(path, allow_pickle=False) as data:
                context_indices = np.asarray(data["context_indices"], dtype=np.int32)
                capture_steps = np.asarray(data["capture_steps"], dtype=np.int32)
                embeddings = np.asarray(data["clip_embeddings"], dtype=np.float32)
            pair_values = _pairwise_chamfer(embeddings, capture_steps)
            implementation_max_abs = max(
                implementation_max_abs,
                _verify_chamfer_implementation(
                    embeddings, capture_steps, pair_values
                ),
            )
            for local_idx, context_idx in enumerate(context_indices):
                meta = context_lookup.loc[(candidate_id, int(context_idx))]
                for horizon_idx, horizon in enumerate(sim.HORIZONS):
                    for pair_idx, (left, right) in enumerate(
                        zip(base.PAIR_LEFT, base.PAIR_RIGHT, strict=True)
                    ):
                        pair_rows.append(
                            (
                                candidate_id,
                                str(candidate["candidate_kind"]),
                                int(candidate["run_idx"]),
                                int(candidate["candidate_idx"]),
                                int(context_idx),
                                str(meta["arm"]),
                                int(meta["rollout_seed_idx"]),
                                int(meta["anchor_idx"]),
                                int(meta["source_step"]),
                                int(horizon),
                                int(pair_idx),
                                int(left),
                                int(right),
                                float(pair_values[local_idx, horizon_idx, pair_idx]),
                            )
                        )
            processed_batches += 1

    pairs = pd.DataFrame.from_records(pair_rows, columns=PAIR_COLUMNS)
    if pairs.empty:
        raise RuntimeError("No CLIP simulation outputs available")
    context_scores = (
        pairs.groupby(
            [
                "candidate_id",
                "candidate_kind",
                "run_idx",
                "candidate_idx",
                "context_idx",
                "arm",
                "rollout_seed_idx",
                "anchor_idx",
                "source_step",
                "horizon_steps",
            ],
            as_index=False,
        )["clip_chamfer"]
        .median()
        .sort_values(["candidate_id", "context_idx", "horizon_steps"])
    )
    candidate_horizons = (
        context_scores.groupby(
            [
                "candidate_id",
                "candidate_kind",
                "run_idx",
                "candidate_idx",
                "arm",
                "horizon_steps",
            ],
            as_index=False,
        )["clip_chamfer"]
        .mean()
        .sort_values(["candidate_id", "arm", "horizon_steps"])
    )
    candidate_horizons["c1_mspd_mean"] = candidate_horizons["candidate_id"].map(
        candidate_lookup["c1_mspd_mean"]
    )
    candidate_horizons["c1_mspd_median"] = candidate_horizons["candidate_id"].map(
        candidate_lookup["c1_mspd_median"]
    )

    summary_rows: list[dict[str, Any]] = []
    for (candidate_id, arm), frame in candidate_horizons.groupby(
        ["candidate_id", "arm"], sort=True
    ):
        frame = frame.sort_values("horizon_steps")
        candidate = candidate_lookup.loc[candidate_id]
        steps = np.concatenate([[0.0], frame["horizon_steps"].to_numpy(dtype=float)])
        values = np.concatenate([[0.0], frame["clip_chamfer"].to_numpy(dtype=float)])
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": str(candidate["candidate_kind"]),
                "run_idx": int(candidate["run_idx"]),
                "candidate_idx": int(candidate["candidate_idx"]),
                "arm": arm,
                "n_contexts": 20 if arm == "trajectory" else 4,
                "clip_chamfer_10k": float(values[-1]),
                "clip_chamfer_horizon_auc": float(
                    np.trapz(values, steps) / sim.HORIZONS[-1]
                ),
                "c1_mspd_mean": float(candidate["c1_mspd_mean"]),
                "c1_mspd_median": float(candidate["c1_mspd_median"]),
            }
        )
    candidate_summary = pd.DataFrame(summary_rows)

    _write_csv(root / "clip_pairwise_scores.csv", pairs)
    _write_csv(root / "clip_context_scores.csv", context_scores)
    _write_csv(root / "clip_candidate_horizons.csv", candidate_horizons)
    _write_csv(root / "clip_candidate_summary.csv", candidate_summary)
    report = {
        "status": "complete" if processed_batches == 80 else "partial",
        "analysis_version": ANALYSIS_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "processed_batches": processed_batches,
        "pair_rows": len(pairs),
        "context_horizon_rows": len(context_scores),
        "candidate_horizon_rows": len(candidate_horizons),
        "candidate_summary_rows": len(candidate_summary),
        "max_vectorized_vs_reference_chamfer_abs": implementation_max_abs,
        "analysis_fingerprint": _analysis_fingerprint(),
        "simulation_audit_status": simulation_audit["status"],
    }
    _write_json(root / "analysis_table_audit.json", report)
    if implementation_max_abs > 1.0e-12:
        raise RuntimeError(f"Vectorized Chamfer mismatch: {implementation_max_abs}")
    return report


def _bootstrap_difference(
    optimized: np.ndarray,
    random: np.ndarray,
    *,
    reducer: str,
    seed: int,
    reps: int = 50_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    function = np.mean if reducer == "mean" else np.median
    values = np.empty(reps, dtype=np.float64)
    chunk = 2_000
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        opt_sample = optimized[
            rng.integers(0, len(optimized), size=(count, len(optimized)))
        ]
        random_sample = random[
            rng.integers(0, len(random), size=(count, len(random)))
        ]
        values[start : start + count] = function(opt_sample, axis=1) - function(
            random_sample, axis=1
        )
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def _group_test(frame: pd.DataFrame, column: str, *, seed: int) -> dict[str, Any]:
    from scipy import stats

    optimized = frame.loc[
        frame["candidate_kind"] == "optimized", column
    ].to_numpy(dtype=np.float64)
    random = frame.loc[frame["candidate_kind"] == "random", column].to_numpy(
        dtype=np.float64
    )
    if optimized.size != 10 or random.size != 30:
        raise RuntimeError(
            f"Unexpected candidate counts for {column}: {optimized.size}, {random.size}"
        )
    u_value, p_value = stats.mannwhitneyu(
        optimized, random, alternative="greater", method="exact"
    )
    welch = stats.ttest_ind(
        optimized, random, equal_var=False, alternative="greater"
    )
    mean_ci = _bootstrap_difference(
        optimized, random, reducer="mean", seed=seed
    )
    median_ci = _bootstrap_difference(
        optimized, random, reducer="median", seed=seed + 1
    )
    return {
        "column": column,
        "n_optimized": int(optimized.size),
        "n_random": int(random.size),
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
        "mann_whitney_u": float(u_value),
        "mann_whitney_exact_one_sided_p": float(p_value),
        "rank_biserial": float(
            2.0 * u_value / (optimized.size * random.size) - 1.0
        ),
        "welch_t": float(welch.statistic),
        "welch_one_sided_p": float(welch.pvalue),
    }


def run_statistics(root: Path) -> dict[str, Any]:
    candidates = pd.read_csv(root / "clip_candidate_summary.csv")
    horizons = pd.read_csv(root / "clip_candidate_horizons.csv")
    rows: list[dict[str, Any]] = []
    for arm_idx, arm in enumerate(("trajectory", "shared_state")):
        arm_summary = candidates[candidates["arm"] == arm]
        for endpoint_idx, endpoint in enumerate(
            ("clip_chamfer_10k", "clip_chamfer_horizon_auc")
        ):
            test = _group_test(
                arm_summary,
                endpoint,
                seed=70_000 + arm_idx * 100 + endpoint_idx,
            )
            rows.append(
                {
                    "arm": arm,
                    "endpoint": endpoint,
                    "horizon_steps": 10_000 if endpoint == "clip_chamfer_10k" else -1,
                    "primary": arm == "trajectory" and endpoint == "clip_chamfer_10k",
                    **test,
                }
            )
        for horizon_idx, horizon in enumerate(sim.HORIZONS):
            frame = horizons[
                (horizons["arm"] == arm)
                & (horizons["horizon_steps"] == horizon)
            ]
            test = _group_test(
                frame,
                "clip_chamfer",
                seed=71_000 + arm_idx * 1_000 + horizon_idx,
            )
            rows.append(
                {
                    "arm": arm,
                    "endpoint": "clip_chamfer_at_horizon",
                    "horizon_steps": int(horizon),
                    "primary": False,
                    **test,
                }
            )
    table = pd.DataFrame(rows)
    _write_csv(root / "clip_statistical_tests.csv", table)
    primary = table[table["primary"]].iloc[0].to_dict()
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "primary_endpoint": (
            "trajectory-arm candidate mean over 20 contexts of median-pair "
            "symmetric CLIP-Chamfer at 10k"
        ),
        "primary_test": primary,
        "interpretation_rule": (
            "Optimization-increased RNG sensitivity is supported only if the "
            "one-sided exact primary p-value is below 0.05 with positive effect."
        ),
        "supported": bool(
            primary["mann_whitney_exact_one_sided_p"] < 0.05
            and primary["mean_difference"] > 0.0
        ),
        "all_other_tests_secondary": True,
    }
    _write_json(root / "clip_statistical_summary.json", summary)
    return summary


def run_correlations(root: Path) -> dict[str, Any]:
    from scipy import stats

    candidate_summary = pd.read_csv(root / "clip_candidate_summary.csv")
    old_root = Path(sim.load_protocol(root)["source_root"])
    old_scores = pd.read_csv(old_root / "candidate_sensitivity_scores.csv")
    rows: list[dict[str, Any]] = []
    for arm in ("trajectory", "shared_state"):
        frame = candidate_summary[candidate_summary["arm"] == arm]
        for subset_name, subset in (
            ("all", frame),
            ("optimized", frame[frame["candidate_kind"] == "optimized"]),
            ("random", frame[frame["candidate_kind"] == "random"]),
        ):
            for x_column in ("c1_mspd_mean",):
                for y_column in (
                    "clip_chamfer_10k",
                    "clip_chamfer_horizon_auc",
                ):
                    x_value = subset[x_column].to_numpy(dtype=float)
                    y_value = subset[y_column].to_numpy(dtype=float)
                    spearman = stats.spearmanr(x_value, y_value)
                    kendall = stats.kendalltau(x_value, y_value)
                    rows.append(
                        {
                            "arm": arm,
                            "subset": subset_name,
                            "x": x_column,
                            "y": y_column,
                            "n": len(subset),
                            "spearman_rho": float(spearman.statistic),
                            "spearman_p_two_sided": float(spearman.pvalue),
                            "kendall_tau": float(kendall.statistic),
                            "kendall_p_two_sided": float(kendall.pvalue),
                        }
                    )
    correlations = pd.DataFrame(rows)
    _write_csv(root / "clip_correlations.csv", correlations)

    comparison = candidate_summary.merge(
        old_scores[
            [
                "candidate_id",
                "arm",
                "a_relative_l1_auc",
                "a_relative_l1_final",
            ]
        ],
        on=["candidate_id", "arm"],
        validate="one_to_one",
    )
    comparison_rows = []
    for arm in ("trajectory", "shared_state"):
        frame = comparison[comparison["arm"] == arm]
        for old_column, new_column in (
            ("a_relative_l1_auc", "clip_chamfer_horizon_auc"),
            ("a_relative_l1_final", "clip_chamfer_10k"),
        ):
            spearman = stats.spearmanr(frame[old_column], frame[new_column])
            kendall = stats.kendalltau(frame[old_column], frame[new_column])
            comparison_rows.append(
                {
                    "arm": arm,
                    "l1_endpoint": old_column,
                    "clip_endpoint": new_column,
                    "n": len(frame),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p_two_sided": float(spearman.pvalue),
                    "kendall_tau": float(kendall.statistic),
                    "kendall_p_two_sided": float(kendall.pvalue),
                }
            )
    _write_csv(root / "l1_vs_clip_candidate_scores.csv", comparison)
    _write_csv(root / "l1_vs_clip_correlations.csv", pd.DataFrame(comparison_rows))
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "mspd_correlations": rows,
        "l1_vs_clip_correlations": comparison_rows,
    }
    _write_json(root / "clip_correlation_summary.json", summary)
    return summary


def _bootstrap_curve(values: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reps = 20_000
    result = np.empty((reps, values.shape[1]), dtype=np.float32)
    for start in range(0, reps, 1_000):
        count = min(1_000, reps - start)
        indices = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        result[start : start + count] = np.mean(values[indices], axis=1)
    low, high = np.quantile(result, [0.025, 0.975], axis=0)
    return low, high


def _save_figure(fig: Any, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def make_plots(root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = {"random": "#4C78A8", "optimized": "#D95F02"}
    labels = {"random": "Random", "optimized": "Optimized"}
    horizons = pd.read_csv(root / "clip_candidate_horizons.csv")
    summary = pd.read_csv(root / "clip_candidate_summary.csv")
    tests = pd.read_csv(root / "clip_statistical_tests.csv")
    correlations = pd.read_csv(root / "clip_correlations.csv")
    comparison = pd.read_csv(root / "l1_vs_clip_candidate_scores.csv")
    contexts = pd.read_csv(root / "clip_context_scores.csv")
    figure_dir = root / "figures"
    generated: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for axis, arm, title in zip(
        axes,
        ("trajectory", "shared_state"),
        ("Visited C1 states", "Shared initial states"),
        strict=True,
    ):
        frame = horizons[horizons["arm"] == arm]
        for kind_idx, kind in enumerate(("random", "optimized")):
            pivot = frame[frame["candidate_kind"] == kind].pivot(
                index="candidate_id", columns="horizon_steps", values="clip_chamfer"
            )
            pivot = pivot.reindex(columns=sim.HORIZONS)
            values = pivot.to_numpy(dtype=float)
            mean = np.mean(values, axis=0)
            low, high = _bootstrap_curve(values, seed=90_000 + kind_idx)
            x_value = np.asarray(sim.HORIZONS, dtype=float) / 1_000.0
            axis.plot(x_value, mean, color=colors[kind], lw=2.2, label=labels[kind])
            axis.fill_between(x_value, low, high, color=colors[kind], alpha=0.16)
        axis.set_title(title)
        axis.set_xlabel("Continuation horizon (thousand steps)")
        axis.set_ylabel("Median-pair CLIP-Chamfer")
        axis.grid(alpha=0.2)
        axis.legend()
    path = figure_dir / "clip_chamfer_horizon_curves"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    primary = tests[tests["primary"]].iloc[0]
    trajectory = summary[summary["arm"] == "trajectory"]
    fig, axis = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    rng = np.random.default_rng(20260722)
    for x_index, kind in enumerate(("random", "optimized")):
        values = trajectory.loc[
            trajectory["candidate_kind"] == kind, "clip_chamfer_10k"
        ].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axis.scatter(
            x_index + jitter,
            values,
            s=45,
            color=colors[kind],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.hlines(
            np.median(values), x_index - 0.22, x_index + 0.22, color="black", lw=2.2
        )
    axis.set_xticks((0, 1), ("Random\n(n=30)", "Optimized\n(n=10)"))
    axis.set_ylabel("Candidate CLIP-Chamfer at 10k")
    axis.set_title(
        "Exact one-sided Mann-Whitney "
        f"p={primary['mann_whitney_exact_one_sided_p']:.4g}; "
        f"r={primary['rank_biserial']:+.2f}"
    )
    axis.grid(axis="y", alpha=0.2)
    path = figure_dir / "clip_chamfer_all_candidates_10k"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    trajectory = summary[summary["arm"] == "trajectory"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True, constrained_layout=True)
    panel_specs = (
        ("all", trajectory, "All candidates"),
        ("random", trajectory[trajectory["candidate_kind"] == "random"], "Random"),
        (
            "optimized",
            trajectory[trajectory["candidate_kind"] == "optimized"],
            "Optimized",
        ),
    )
    for axis, (subset_name, frame, title) in zip(axes, panel_specs, strict=True):
        for kind in ("random", "optimized"):
            points = frame[frame["candidate_kind"] == kind]
            if points.empty:
                continue
            axis.scatter(
                points["c1_mspd_mean"].to_numpy(dtype=float) * 1_000.0,
                points["clip_chamfer_10k"].to_numpy(dtype=float) * 1_000.0,
                s=42,
                alpha=0.8,
                color=colors[kind],
                edgecolor="white",
                linewidth=0.45,
                label=labels[kind],
            )
        x_value = frame["c1_mspd_mean"].to_numpy(dtype=float) * 1_000.0
        y_value = frame["clip_chamfer_10k"].to_numpy(dtype=float) * 1_000.0
        if np.unique(x_value).size > 1:
            slope, intercept = np.polyfit(x_value, y_value, deg=1)
            line_x = np.linspace(float(np.min(x_value)), float(np.max(x_value)), 100)
            axis.plot(
                line_x,
                slope * line_x + intercept,
                color="#303030",
                linestyle="--",
                linewidth=1.35,
                alpha=0.8,
                label="linear guide",
            )
        statistic = correlations[
            (correlations["arm"] == "trajectory")
            & (correlations["subset"] == subset_name)
            & (correlations["x"] == "c1_mspd_mean")
            & (correlations["y"] == "clip_chamfer_10k")
        ].iloc[0]
        axis.text(
            0.04,
            0.96,
            "Spearman "
            rf"$\rho={float(statistic['spearman_rho']):+.3f}$"
            "\n"
            rf"$p={float(statistic['spearman_p_two_sided']):.3g}$; "
            rf"$\tau={float(statistic['kendall_tau']):+.3f}$"
            "\n"
            rf"$n={int(statistic['n'])}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        axis.set_title(title)
        axis.set_xlabel(r"mean C1 MSPD ($\times 10^{-3}$)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"CLIP--Chamfer sensitivity at 10k ($\times 10^{-3}$)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("C1 MSPD and intrinsic RNG sensitivity", fontsize=12.5)
    path = figure_dir / "clip_vs_c1_mspd_sensitivity"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    fig, axis = plt.subplots(figsize=(7.0, 4.1), constrained_layout=True)
    for arm_idx, (arm, label, color) in enumerate(
        (
            ("trajectory", "Visited C1 states", colors["optimized"]),
            ("shared_state", "Shared initial states", "#009E73"),
        )
    ):
        means = []
        lows = []
        highs = []
        for horizon_idx, horizon in enumerate(sim.HORIZONS):
            frame = horizons[
                (horizons["arm"] == arm)
                & (horizons["horizon_steps"] == horizon)
            ]
            optimized = frame.loc[
                frame["candidate_kind"] == "optimized", "clip_chamfer"
            ].to_numpy(dtype=float)
            random = frame.loc[
                frame["candidate_kind"] == "random", "clip_chamfer"
            ].to_numpy(dtype=float)
            means.append(float(np.mean(optimized) - np.mean(random)))
            low, high = _bootstrap_difference(
                optimized,
                random,
                reducer="mean",
                seed=95_000 + arm_idx * 100 + horizon_idx,
                reps=20_000,
            )
            lows.append(low)
            highs.append(high)
        x_value = np.asarray(sim.HORIZONS, dtype=float) / 1_000.0
        axis.plot(x_value, means, marker="o", color=color, lw=2.0, label=label)
        axis.fill_between(x_value, lows, highs, color=color, alpha=0.15)
    axis.axhline(0.0, color="black", lw=1.0)
    axis.set_xlabel("Continuation horizon (thousand steps)")
    axis.set_ylabel("Optimized - random mean CLIP-Chamfer")
    axis.set_title("Optimization effect across horizons")
    axis.grid(alpha=0.2)
    axis.legend()
    path = figure_dir / "clip_optimization_effect_by_horizon"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    for axis, arm, title in zip(
        axes,
        ("trajectory", "shared_state"),
        ("Visited C1 states", "Shared initial states"),
        strict=True,
    ):
        frame = comparison[comparison["arm"] == arm]
        for kind in ("random", "optimized"):
            subset = frame[frame["candidate_kind"] == kind]
            axis.scatter(
                subset["a_relative_l1_auc"],
                subset["clip_chamfer_horizon_auc"],
                s=36,
                alpha=0.78,
                color=colors[kind],
                label=labels[kind],
            )
        axis.set_title(title)
        axis.set_xlabel("Mass-normalized L1 AUC")
        axis.set_ylabel("CLIP-Chamfer horizon AUC")
        axis.grid(alpha=0.2)
        axis.legend()
    path = figure_dir / "clip_vs_l1_candidate_sensitivity"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    trajectory_contexts = contexts[
        (contexts["arm"] == "trajectory")
        & (contexts["horizon_steps"] == sim.HORIZONS[-1])
    ].copy()
    trajectory_contexts["context_label"] = trajectory_contexts.apply(
        lambda row: f"s{int(row.rollout_seed_idx)}@{int(row.source_step) // 1000}k",
        axis=1,
    )
    matrix = trajectory_contexts.pivot(
        index="candidate_id", columns="context_label", values="clip_chamfer"
    )
    candidate_order = (
        summary[summary["arm"] == "trajectory"]
        .sort_values(["candidate_kind", "run_idx", "candidate_idx"], ascending=[True, True, True])[
            "candidate_id"
        ]
        .tolist()
    )
    context_order = [f"s{seed}@{step // 1000}k" for seed in range(4) for step in base.SOURCE_STEPS]
    matrix = matrix.reindex(index=candidate_order, columns=context_order)
    fig, axis = plt.subplots(figsize=(10.8, 8.8), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(context_order)), context_order, rotation=55, ha="right", fontsize=7)
    axis.set_yticks(range(len(candidate_order)), candidate_order, fontsize=6.5)
    axis.set_xlabel("C1 fork context")
    axis.set_title("CLIP-Chamfer at 10k by candidate and fork state")
    fig.colorbar(image, ax=axis, label="Median-pair CLIP-Chamfer")
    path = figure_dir / "clip_context_heatmap_10k"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    candidate_matrix = horizons[horizons["arm"] == "trajectory"].pivot(
        index="candidate_id", columns="horizon_steps", values="clip_chamfer"
    )
    candidate_matrix = candidate_matrix.reindex(index=candidate_order, columns=sim.HORIZONS)
    fig, axis = plt.subplots(figsize=(7.6, 8.8), constrained_layout=True)
    image = axis.imshow(candidate_matrix.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_xticks(
        range(len(sim.HORIZONS)), [f"{value // 1000}k" for value in sim.HORIZONS]
    )
    axis.set_yticks(range(len(candidate_order)), candidate_order, fontsize=6.5)
    axis.set_xlabel("Continuation horizon")
    axis.set_title("Candidate CLIP-Chamfer dynamics")
    fig.colorbar(image, ax=axis, label="Mean-context CLIP-Chamfer")
    path = figure_dir / "clip_candidate_horizon_heatmap"
    _save_figure(fig, path)
    plt.close(fig)
    generated.append(path.name)

    _write_json(
        root / "figure_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "figures": generated,
            "formats": ["png", "pdf"],
        },
    )
    return generated


def write_report(root: Path) -> Path:
    statistical = json.loads((root / "clip_statistical_summary.json").read_text())
    correlations = pd.read_csv(root / "clip_correlations.csv")
    l1_correlations = pd.read_csv(root / "l1_vs_clip_correlations.csv")
    primary = statistical["primary_test"]
    mspd = correlations[
        (correlations["arm"] == "trajectory")
        & (correlations["subset"] == "all")
        & (correlations["y"] == "clip_chamfer_10k")
    ].iloc[0]
    l1 = l1_correlations[
        (l1_correlations["arm"] == "trajectory")
        & (l1_correlations["clip_endpoint"] == "clip_chamfer_10k")
    ].iloc[0]
    text = f"""# FlowLenia RNG sensitivity with CLIP-Chamfer

## Protocol

- Same 10 optimized and 30 existing random candidates as the audited L1 experiment.
- Same 20 visited C1 states per candidate; four shared-state controls are secondary.
- Same nine unique continuation RNG branches and exact duplicate control; no external noise.
- Horizon 10,000. At each 1k horizon, eight evenly spaced rendered frames are embedded by OpenAI CLIP ViT-B/32.
- Pair distance is symmetric cosine Chamfer between two eight-frame embedding clouds.
- Context score is the median over all 36 branch pairs; candidate score is the mean over 20 C1 contexts.

## Primary result

- Optimized mean CLIP-Chamfer at 10k: `{primary['optimized_mean']:.8g}`
- Random mean: `{primary['random_mean']:.8g}`
- Mean difference: `{primary['mean_difference']:+.8g}` (95% bootstrap CI `{primary['mean_difference_ci95_low']:+.8g}`, `{primary['mean_difference_ci95_high']:+.8g}`)
- Exact one-sided Mann-Whitney: `U={primary['mann_whitney_u']:.1f}`, `p={primary['mann_whitney_exact_one_sided_p']:.8g}`
- Rank-biserial effect: `{primary['rank_biserial']:+.4f}`
- Optimization-increased RNG sensitivity supported under the fixed rule: **{str(statistical['supported']).lower()}**

## Associations

- Mean C1 MSPD versus 10k CLIP sensitivity across all 40 candidates: Spearman rho `{mspd['spearman_rho']:+.4f}` (`p={mspd['spearman_p_two_sided']:.5g}`), Kendall tau `{mspd['kendall_tau']:+.4f}`.
- L1-final versus CLIP-10k sensitivity: Spearman rho `{l1['spearman_rho']:+.4f}` (`p={l1['spearman_p_two_sided']:.5g}`).

## Audits

- Every re-simulated L1 metric array is bit-exact with the prior experiment.
- Exact duplicate and common t=0 embeddings are canonicalized from their bit-exact rendered inputs.
- Pilot batch-vs-single-frame CLIP parity is recorded in `simulation_audit.json`.

CLIP-Chamfer measures perceptual separation between finite stochastic trajectory clouds. It is not a deterministic Lyapunov-exponent estimate.
"""
    path = root / "CLIP_CHAMFER_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_runtime(root: Path) -> dict[str, Any]:
    import jax
    import scipy

    try:
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        nvidia = f"unavailable: {exc}"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "nvidia_smi": nvidia,
    }
    _write_json(root / "analysis_runtime.json", payload)
    return payload


def final_audit(root: Path) -> dict[str, Any]:
    simulation = sim.audit(root, require_complete=True)
    table_audit = json.loads((root / "analysis_table_audit.json").read_text())
    expected_rows = {
        "clip_pairwise_scores.csv": 40 * 24 * len(sim.HORIZONS) * base.N_PAIRS,
        "clip_context_scores.csv": 40 * 24 * len(sim.HORIZONS),
        "clip_candidate_horizons.csv": 40 * 2 * len(sim.HORIZONS),
        "clip_candidate_summary.csv": 40 * 2,
        "clip_statistical_tests.csv": 24,
    }
    table_checks = {}
    for name, expected in expected_rows.items():
        path = root / name
        found = len(pd.read_csv(path)) if path.exists() else -1
        table_checks[name] = {
            "expected": expected,
            "found": found,
            "passed": found == expected,
        }
    figures = json.loads((root / "figure_manifest.json").read_text())["figures"]
    figure_checks = {
        f"{name}.{suffix}": (
            (root / "figures" / f"{name}.{suffix}").exists()
            and (root / "figures" / f"{name}.{suffix}").stat().st_size > 10_000
        )
        for name in figures
        for suffix in ("png", "pdf")
    }
    required = (
        root / "CLIP_CHAMFER_REPORT.md",
        root / "clip_statistical_summary.json",
        root / "clip_correlation_summary.json",
        root / "analysis_runtime.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    passed = (
        simulation["status"] == "complete"
        and table_audit["status"] == "complete"
        and all(item["passed"] for item in table_checks.values())
        and all(figure_checks.values())
        and not missing
    )
    report = {
        "status": "complete" if passed else "failed",
        "analysis_version": ANALYSIS_VERSION,
        "plan_sha256": simulation["plan_sha256"],
        "simulation_code_bundle_sha256": simulation[
            "simulation_code_bundle_sha256"
        ],
        "model_identity_sha256": simulation["model_identity_sha256"],
        "analysis_fingerprint": _analysis_fingerprint(),
        "simulation_batches": simulation["valid_batches"],
        "replay_exact_batches": simulation["replay_exact_batches"],
        "table_checks": table_checks,
        "figure_checks": figure_checks,
        "missing_artifacts": missing,
    }
    _write_json(root / "completion_audit.json", report)
    if not passed:
        raise RuntimeError(f"Final CLIP analysis audit failed: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FlowLenia RNG sensitivity with CLIP-Chamfer."
    )
    parser.add_argument(
        "phase", choices=("tables", "statistics", "plots", "report", "audit", "all")
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve(args.output_root)
    result: Any = None
    if args.phase in {"tables", "all"}:
        result = build_tables(root, require_complete=not args.allow_partial)
    if args.phase in {"statistics", "all"}:
        result = run_statistics(root)
        run_correlations(root)
    if args.phase in {"plots", "all"}:
        result = make_plots(root)
    if args.phase in {"report", "all"}:
        result = str(write_report(root))
        write_runtime(root)
    if args.phase in {"audit", "all"}:
        result = final_audit(root)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
