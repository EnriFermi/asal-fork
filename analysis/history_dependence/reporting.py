from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAIR_ORDER = ["free-free", "wall-wall", "free-wall"]
PAIR_COLORS = {
    "free-free": "#4C72B0",
    "wall-wall": "#55A868",
    "free-wall": "#C44E52",
}


def summarize_pairwise_values(pair_df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    rows = []
    if pair_df.empty:
        return pd.DataFrame()
    for column in value_columns:
        if column not in pair_df.columns:
            continue
        for pair_name in PAIR_ORDER:
            vals = pair_df.loc[pair_df["pair_type"] == pair_name, column].dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "observable": column,
                    "pair_type": pair_name,
                    "n_pairs": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1) if vals.size > 1 else 0.0),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    return pd.DataFrame(rows)


def compute_effect_sizes(pair_df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    rows = []
    if pair_df.empty:
        return pd.DataFrame()
    for column in value_columns:
        if column not in pair_df.columns:
            continue
        ff = pair_df.loc[pair_df["pair_type"] == "free-free", column].dropna().to_numpy(dtype=np.float64)
        fw = pair_df.loc[pair_df["pair_type"] == "free-wall", column].dropna().to_numpy(dtype=np.float64)
        ww = pair_df.loc[pair_df["pair_type"] == "wall-wall", column].dropna().to_numpy(dtype=np.float64)
        if ff.size == 0 or fw.size == 0:
            continue
        rows.append(
            {
                "observable": column,
                "mean_free_free": float(np.mean(ff)),
                "mean_wall_wall": float(np.mean(ww)) if ww.size else np.nan,
                "mean_free_wall": float(np.mean(fw)),
                "effect_free_wall_minus_free_free": float(np.mean(fw) - np.mean(ff)),
                "ratio_free_wall_over_free_free": float(np.mean(fw) / max(np.mean(ff), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _mean_pair_distance(arr: np.ndarray, labels: list[str], pair_name: str) -> float:
    vals = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[1]):
            pair = "-".join(sorted((labels[i], labels[j])))
            if pair == pair_name:
                vals.append(arr[i, j])
    return float(np.mean(vals)) if vals else np.nan


def permutation_test_from_matrix(
    matrix: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    observable: str,
    max_exact: int = 200_000,
    n_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    order = matrix.index.tolist()
    cond_map = runs.set_index("run_id")["condition"].to_dict()
    labels = [str(cond_map[x]) for x in order]
    n_total = len(labels)
    n_wall = sum(lbl == "wall" for lbl in labels)
    if n_wall <= 0 or n_wall >= n_total:
        return {"observable": observable, "pvalue_greater": np.nan, "mode": "invalid"}

    arr = np.asarray(matrix.to_numpy(dtype=np.float64))
    observed = _mean_pair_distance(arr, labels, "free-wall") - _mean_pair_distance(arr, labels, "free-free")
    combos_total = int(math.comb(n_total, n_wall))

    rng = np.random.default_rng(seed)
    stats = []
    if combos_total <= max_exact:
        iterable = itertools.combinations(range(n_total), n_wall)
        mode = "exact"
        for wall_idx in iterable:
            perm = ["free"] * n_total
            for idx in wall_idx:
                perm[idx] = "wall"
            stat = _mean_pair_distance(arr, perm, "free-wall") - _mean_pair_distance(arr, perm, "free-free")
            stats.append(stat)
    else:
        mode = "monte_carlo"
        for _ in range(int(n_samples)):
            perm_idx = rng.choice(n_total, size=n_wall, replace=False)
            perm = ["free"] * n_total
            for idx in perm_idx:
                perm[idx] = "wall"
            stat = _mean_pair_distance(arr, perm, "free-wall") - _mean_pair_distance(arr, perm, "free-free")
            stats.append(stat)

    stat_arr = np.asarray(stats, dtype=np.float64)
    pvalue = float((1.0 + np.sum(stat_arr >= observed)) / (1.0 + stat_arr.size))
    return {
        "observable": observable,
        "observed_statistic": float(observed),
        "pvalue_greater": pvalue,
        "n_permutations": int(stat_arr.size),
        "mode": mode,
        "n_runs": int(n_total),
        "n_wall": int(n_wall),
        "n_free": int(n_total - n_wall),
    }


def compute_concordance(pair_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from scipy import stats as scipy_stats

    rows = []
    use_cols = [col for col in columns if col in pair_df.columns]
    for idx_a, col_a in enumerate(use_cols):
        for col_b in use_cols[idx_a + 1:]:
            valid = pair_df[[col_a, col_b]].dropna()
            if valid.shape[0] < 3:
                continue
            rho, pvalue = scipy_stats.spearmanr(valid[col_a], valid[col_b])
            rows.append(
                {
                    "observable_a": col_a,
                    "observable_b": col_b,
                    "n_pairs": int(valid.shape[0]),
                    "spearman_rho": float(rho),
                    "spearman_pvalue": float(pvalue),
                }
            )
    return pd.DataFrame(rows)


def plot_distance_matrix(matrix: pd.DataFrame, runs: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180) -> None:
    import matplotlib.pyplot as plt

    order = matrix.index.tolist()
    cond_map = runs.set_index("run_id")["condition"].to_dict()
    tick_labels = [f"{run_id}\n{cond_map.get(run_id, '?')}" for run_id in order]

    fig, ax = plt.subplots(figsize=(0.65 * len(order) + 2.5, 0.65 * len(order) + 2.0), dpi=dpi)
    im = ax.imshow(matrix.to_numpy(dtype=np.float64), cmap="magma")
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_strip(pair_df: pd.DataFrame, value_col: str, path: str | Path, *, title: str, dpi: int = 180) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=dpi)
    rng = np.random.default_rng(0)
    for pos, pair_name in enumerate(PAIR_ORDER):
        vals = pair_df.loc[pair_df["pair_type"] == pair_name, value_col].dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(
            np.full(vals.size, pos, dtype=np.float64) + jitter,
            vals,
            s=42,
            alpha=0.85,
            color=PAIR_COLORS[pair_name],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hlines(np.mean(vals), pos - 0.22, pos + 0.22, color="black", linewidth=2.0)
    ax.set_xticks(range(len(PAIR_ORDER)))
    ax.set_xticklabels(PAIR_ORDER)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_h_examples(
    per_run: dict[str, dict[str, Any]],
    runs: pd.DataFrame,
    path: str | Path,
    *,
    n_per_condition: int = 2,
    run_ids: list[str] | None = None,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    if run_ids is None:
        chosen = []
        for condition in ("free", "wall"):
            subset = runs.loc[runs["condition"] == condition, "run_id"].tolist()[:n_per_condition]
            chosen.extend(subset)
    else:
        chosen = run_ids
    chosen = [run_id for run_id in chosen if run_id in per_run]
    if not chosen:
        return

    n_cols = min(2, len(chosen))
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.8 * n_rows), dpi=dpi, squeeze=False)
    axes_flat = axes.ravel()
    for ax, run_id in zip(axes_flat, chosen):
        data = per_run[run_id]
        h_map = np.asarray(data["delta_h_map"], dtype=np.float64)
        tau_steps = np.asarray(data["tau_steps"], dtype=np.int32)
        starts = np.asarray(data["window_start_steps"], dtype=np.int32)
        im = ax.imshow(
            h_map,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[int(starts[0]), int(starts[-1]), int(tau_steps[0]), int(tau_steps[-1])],
        )
        ax.set_title(f"{run_id}")
        ax.set_xlabel("window start (steps)")
        ax.set_ylabel("tau (steps)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[len(chosen):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(pair_df: pd.DataFrame, x_col: str, y_col: str, path: str | Path, *, title: str, dpi: int = 180) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    valid = pair_df[[x_col, y_col, "pair_type"]].dropna()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=dpi)
    for pair_name in PAIR_ORDER:
        sub = valid.loc[valid["pair_type"] == pair_name]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=46,
            alpha=0.85,
            color=PAIR_COLORS[pair_name],
            label=pair_name,
            edgecolor="white",
            linewidth=0.5,
        )
    rho, pvalue = scipy_stats.spearmanr(valid[x_col], valid[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}, p={pvalue:.3g}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_table(frame: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180, round_digits: int = 4) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    table_df = frame.copy()
    for column in table_df.columns:
        if np.issubdtype(table_df[column].dtype, np.number):
            table_df[column] = table_df[column].map(lambda x: f"{x:.{round_digits}f}")

    fig_h = max(1.8, 0.42 * (len(table_df) + 2))
    fig, ax = plt.subplots(figsize=(11.0, fig_h), dpi=dpi)
    ax.axis("off")
    ax.set_title(title, loc="left", pad=12.0)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_frame_panel(
    runs: pd.DataFrame,
    path: str | Path,
    *,
    n_per_condition: int = 2,
    dpi: int = 180,
) -> bool:
    from PIL import Image
    import matplotlib.pyplot as plt

    if "frame_path" not in runs.columns:
        return False
    chosen = []
    for condition in ("free", "wall"):
        subset = runs.loc[(runs["condition"] == condition) & runs["frame_path"].notna()]
        subset = subset.loc[subset["frame_path"].map(lambda x: Path(str(x)).exists())]
        chosen.extend(subset.head(n_per_condition).to_dict(orient="records"))
    if not chosen:
        return False

    n_cols = min(2, len(chosen))
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.4 * n_rows), dpi=dpi, squeeze=False)
    for ax, row in zip(axes.ravel(), chosen):
        img = Image.open(row["frame_path"])
        ax.imshow(img)
        ax.set_title(f"{row['run_id']} ({row['condition']})")
        ax.axis("off")
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True
