from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import ensure_dir, progress_bar, save_dataframe, write_json


PAIR_CLASS_NAMES = ("free-free", "wall-wall", "free-wall")


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
            raise ValueError(
                f"Distance matrix shape must match number of runs, got {matrix.shape} for n_runs={n_runs}."
            )
        values = matrix[ii, jj]

    cond = ordered["condition"].astype(str).str.lower().to_numpy()
    is_free = cond == "free"
    pair_classes = _pair_class_from_masks(is_free[ii], is_free[jj])
    raw = pd.DataFrame(
        {
            "run_a": ordered.iloc[ii]["run_id"].to_numpy(),
            "run_b": ordered.iloc[jj]["run_id"].to_numpy(),
            "condition_a": cond[ii],
            "condition_b": cond[jj],
            "pair_class": pair_classes,
            "distance": values,
        }
    )
    return raw


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
    if n_wall <= 0 or n_free <= 1 or int(max_exact) < 1 and int(n_samples) < 1:
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

    raw = _extract_upper_triangle(run_groups[["run_id", "condition"]], distance_matrix_or_function)
    raw = raw.rename(columns={"pair_class": "pair_type", "distance": distance_name})
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


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report["summary"])
    return {
        "distance_name": summary["distance_name"],
        "mean_free_free": float(summary["mean_free_free"]),
        "mean_free_wall": float(summary["mean_free_wall"]),
        "delta_mean": float(summary["delta_mean"]),
        "ratio_mean": float(summary["ratio_mean"]),
        "A_tie": float(summary["A_tie"]),
        "CI_low": float(summary["CI_low"]),
        "CI_high": float(summary["CI_high"]),
        "pvalue_greater": float(summary["pvalue_greater"]),
    }


def _report_string(report: dict[str, Any]) -> str:
    s = report["summary"]
    return (
        f"For embedding distance {s['distance_name']}, the probability that a random {s['class_a']} pair is farther "
        f"apart than a random {s['class_b']} pair is A_tie = {s['A_tie']:.3f}, 95% CI "
        f"[{s['CI_low']:.3f}, {s['CI_high']:.3f}], p = {s['pvalue_greater']:.3g}. "
        f"This pattern is consistent with frustration-like history dependence: late-time divergence induced by the "
        f"wall protocol exceeds ordinary free-run variability."
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
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=dpi)
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
    ax.set_title(f"Dominance plot: {class_a} vs {class_b}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
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

    matrix = matrices[distance_name]
    report = compute_pair_class_superiority(
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

    report["summary"].update(
        {
            "CI_low": float(ci_low),
            "CI_high": float(ci_high),
            "pvalue_greater": float(perm["pvalue_greater"]),
            "permutation_mode": perm["permutation_mode"],
            "n_permutations": int(perm["n_permutations"]),
            "u_statistic": float(perm["u_statistic"]),
            "observed_a_tie": float(perm["observed_a_tie"]),
        }
    )
    report["bootstrap_distribution"] = bootstrap_stats
    report["permutation"] = perm
    report["text_summary"] = _report_string(report)

    base_dir = ensure_dir(Path(output_dir) / "main_observable")
    tables_dir = ensure_dir(base_dir / "tables")
    figures_dir = ensure_dir(base_dir / "figures")

    raw_pairwise = report["raw_pairwise"].copy()
    raw_pairwise = raw_pairwise.rename(columns={distance_name: "distance"})
    raw_pairwise.insert(raw_pairwise.columns.get_loc("selected_role") + 1, "distance_name", distance_name)
    save_dataframe(tables_dir / "raw_pairwise_values.csv", raw_pairwise)

    selected_raw = raw_pairwise.loc[raw_pairwise["selected_role"].isin(["class_a", "class_b"])].reset_index(drop=True)
    save_dataframe(tables_dir / "selected_pairwise_values.csv", selected_raw)

    summary_df = pd.DataFrame([_summary_row(report)])
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
        },
    )

    strip_path = figures_dir / "pair_class_strip.png"
    plot_superiority_strip(
        report["raw_pairwise"],
        distance_name,
        strip_path,
        class_a=class_a,
        class_b=class_b,
        center=str(main_cfg.get("strip_center", "mean")),
        dpi=dpi,
    )
    cdf_path = figures_dir / "pair_class_cdf.png"
    plot_superiority_cdf(
        report["class_b_values"],
        report["class_a_values"],
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
        "summary_csv": str(tables_dir / "main_observable_summary.csv"),
        "report_txt": str(base_dir / "main_observable_report.txt"),
        "figure_paths": figure_paths,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Main pair-class superiority report for history-dependence analysis.")
    parser.add_argument("config", help="Path to analysis YAML config.")
    args = parser.parse_args(argv)

    from .pipeline import load_analysis_config, run_analysis

    cfg = load_analysis_config(args.config)
    result = run_analysis(cfg)
    report = result.get("main_observable", {})
    if not report:
        print("Main observable reporting is disabled.")
        return 0
    print(report["text_summary"])
    print(f"Saved main observable outputs to {report['paths']['base_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
