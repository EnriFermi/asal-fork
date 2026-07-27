#!/usr/bin/env python3
"""Plot Flow-Lenia OpenAI-ES population history as PCA swarms."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_SOURCE_ROOT = Path(
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization"
)
DEFAULT_SELECTED_ROOT = Path(
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/optimization"
)
DEFAULT_OUTPUT_ROOT = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
    "figures/pca_swarm"
)


@dataclass(frozen=True)
class RunHistory:
    run_idx: int
    run_name: str
    pop_traj_path: Path
    params: np.ndarray
    score: np.ndarray
    tau_steps: np.ndarray | None
    selected_iter: int | None
    selected_pop_idx: int | None
    selected_score: float | None
    selected_tau_steps: int | None


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected dict, got {type(data)!r}")
    return data


def _run_idx_from_name(name: str) -> int:
    if not name.startswith("run_"):
        raise ValueError(f"Cannot parse run index from {name!r}")
    return int(name.split("_", 1)[1])


def _load_selected(selected_root: Path, run_name: str) -> dict[str, Any] | None:
    path = selected_root / run_name / "selected_candidate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_histories(source_root: Path, selected_root: Path) -> list[RunHistory]:
    histories: list[RunHistory] = []
    for pop_path in sorted(source_root.glob("run_*/pop_traj.pkl")):
        run_dir = pop_path.parent
        run_name = run_dir.name
        data = _load_pkl(pop_path)
        params = np.asarray(data["params"], dtype=np.float64)
        if params.ndim != 3:
            raise ValueError(f"{pop_path}: expected params [iter,pop,dim], got {params.shape}")

        if "objective_score" in data:
            score = np.asarray(data["objective_score"], dtype=np.float64)
        elif "score_by_seed" in data:
            score = np.nanmean(np.asarray(data["score_by_seed"], dtype=np.float64), axis=-1)
        elif "loss" in data:
            score = -np.asarray(data["loss"], dtype=np.float64)
        else:
            score = np.full(params.shape[:2], np.nan, dtype=np.float64)
        if score.shape != params.shape[:2]:
            raise ValueError(f"{pop_path}: score shape {score.shape} does not match params {params.shape[:2]}")

        tau_steps = None
        if "tau_steps" in data:
            tau_steps = np.asarray(data["tau_steps"])
            if tau_steps.shape != params.shape[:2]:
                raise ValueError(f"{pop_path}: tau_steps shape {tau_steps.shape} does not match params {params.shape[:2]}")

        selected = _load_selected(selected_root, run_name) or {}
        selected_tau = selected.get("tau") or {}
        histories.append(
            RunHistory(
                run_idx=_run_idx_from_name(run_name),
                run_name=run_name,
                pop_traj_path=pop_path.resolve(),
                params=params,
                score=score,
                tau_steps=tau_steps,
                selected_iter=int(selected["iter"]) if "iter" in selected else None,
                selected_pop_idx=int(selected["pop_idx"]) if "pop_idx" in selected else None,
                selected_score=float(selected["score_mspd"]) if "score_mspd" in selected else None,
                selected_tau_steps=int(selected_tau["tau_steps"]) if "tau_steps" in selected_tau else None,
            )
        )
    if not histories:
        raise FileNotFoundError(f"No run_*/pop_traj.pkl files found under {source_root}")
    return histories


def _fit_transform(points: np.ndarray, standardize: bool) -> tuple[np.ndarray, dict[str, Any], Any]:
    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(points)
    else:
        scaler = None
        x = points
    pca = PCA(n_components=2)
    xy = pca.fit_transform(x)
    meta = {
        "standardize": bool(standardize),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    return xy, meta, (scaler, pca)


def _transform(points: np.ndarray, model: Any, standardize: bool) -> np.ndarray:
    scaler, pca = model
    x = scaler.transform(points) if standardize else points
    return pca.transform(x)


def _flatten_run(history: RunHistory) -> pd.DataFrame:
    n_iter, pop_size, n_dim = history.params.shape
    score = history.score.reshape(-1)
    rows = pd.DataFrame(
        {
            "run_idx": history.run_idx,
            "run_name": history.run_name,
            "iter": np.repeat(np.arange(n_iter), pop_size),
            "pop_idx": np.tile(np.arange(pop_size), n_iter),
            "score_mspd": score,
        }
    )
    if history.tau_steps is not None:
        rows["tau_steps"] = history.tau_steps.reshape(-1)
    else:
        rows["tau_steps"] = np.nan
    rows["is_selected"] = False
    if history.selected_iter is not None and history.selected_pop_idx is not None:
        rows.loc[
            (rows["iter"] == history.selected_iter) & (rows["pop_idx"] == history.selected_pop_idx),
            "is_selected",
        ] = True
    return rows


def _center_xy(history: RunHistory, model: Any, standardize: bool) -> np.ndarray:
    centers = np.nanmean(history.params, axis=1)
    return _transform(centers, model, standardize)


def _set_equalish_limits(ax: plt.Axes, xy: np.ndarray) -> None:
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return
    lo = np.nanpercentile(xy[finite], 1, axis=0)
    hi = np.nanpercentile(xy[finite], 99, axis=0)
    pad = np.maximum((hi - lo) * 0.12, 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def _plot_run_panel(
    ax: plt.Axes,
    history: RunHistory,
    rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    model: Any,
    standardize: bool,
    title: str,
    color_by: str,
) -> None:
    if color_by == "score":
        c = rows["score_mspd"].to_numpy()
        label = "MSPD"
        cmap = "magma"
    else:
        c = rows["iter"].to_numpy()
        label = "iter"
        cmap = "viridis"

    sc = ax.scatter(
        rows[x_col],
        rows[y_col],
        c=c,
        cmap=cmap,
        s=14,
        alpha=0.62,
        linewidths=0,
    )

    # Slot paths are only a visual aid: OpenAI-ES pop_idx is a perturbation slot, not a persistent lineage.
    for pop_idx, slot_rows in rows.groupby("pop_idx", sort=True):
        slot_rows = slot_rows.sort_values("iter")
        ax.plot(slot_rows[x_col], slot_rows[y_col], color="0.15", alpha=0.10, linewidth=0.7)

    center = _center_xy(history, model, standardize)
    ax.plot(center[:, 0], center[:, 1], color="black", linewidth=1.8, alpha=0.85, label="population mean")
    marker_idx = np.arange(0, center.shape[0], max(1, center.shape[0] // 10))
    ax.scatter(center[marker_idx, 0], center[marker_idx, 1], color="black", s=16, alpha=0.9, linewidths=0)

    selected_rows = rows[rows["is_selected"]]
    if len(selected_rows) == 1:
        sel = selected_rows.iloc[0]
        ax.scatter(
            [sel[x_col]],
            [sel[y_col]],
            marker="*",
            s=190,
            color="#e31a1c",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="selected",
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="0.88", linewidth=0.6)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    ax.legend(loc="best", fontsize=8, frameon=False)


def _plot_per_run(
    out_path: Path,
    history: RunHistory,
    rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    model: Any,
    standardize: bool,
    projection_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    title_base = f"{history.run_name} {projection_label}"
    _plot_run_panel(
        axes[0],
        history,
        rows,
        x_col,
        y_col,
        model,
        standardize,
        f"{title_base}: by iteration",
        color_by="iter",
    )
    _plot_run_panel(
        axes[1],
        history,
        rows,
        x_col,
        y_col,
        model,
        standardize,
        f"{title_base}: by MSPD",
        color_by="score",
    )
    fig.suptitle(
        f"Flow-Lenia OpenAI-ES PCA swarm: {history.run_name}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_global_grid(
    out_path: Path,
    histories: list[RunHistory],
    rows_all: pd.DataFrame,
    x_col: str,
    y_col: str,
    model: Any,
    standardize: bool,
) -> None:
    n = len(histories)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, history in zip(axes_arr, histories):
        rows = rows_all[rows_all["run_name"] == history.run_name].copy()
        _plot_run_panel(
            ax,
            history,
            rows,
            x_col,
            y_col,
            model,
            standardize,
            f"{history.run_name}",
            color_by="iter",
        )
        _set_equalish_limits(ax, rows[[x_col, y_col]].to_numpy())
    for ax in axes_arr[len(histories) :]:
        ax.axis("off")
    fig.suptitle("Flow-Lenia OpenAI-ES PCA swarm, global projection", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-pca", action="store_true", help="Use raw parameter coordinates instead of z-scored PCA.")
    args = parser.parse_args()

    standardize = not args.raw_pca
    histories = _load_histories(args.source_root, args.selected_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_points = np.concatenate([h.params.reshape(-1, h.params.shape[-1]) for h in histories], axis=0)
    global_xy, global_meta, global_model = _fit_transform(all_points, standardize=standardize)

    rows_parts = []
    offset = 0
    local_meta: dict[str, Any] = {}
    for history in histories:
        n_points = history.params.shape[0] * history.params.shape[1]
        rows = _flatten_run(history)
        rows["pc1_global"] = global_xy[offset : offset + n_points, 0]
        rows["pc2_global"] = global_xy[offset : offset + n_points, 1]
        offset += n_points

        local_points = history.params.reshape(-1, history.params.shape[-1])
        local_xy, meta, local_model = _fit_transform(local_points, standardize=standardize)
        rows["pc1_local"] = local_xy[:, 0]
        rows["pc2_local"] = local_xy[:, 1]
        local_meta[history.run_name] = {**meta, "pop_traj_path": str(history.pop_traj_path)}

        _plot_per_run(
            args.output_root / f"pca_swarm_{history.run_name}_global.png",
            history,
            rows,
            "pc1_global",
            "pc2_global",
            global_model,
            standardize,
            "global PCA",
        )
        _plot_per_run(
            args.output_root / f"pca_swarm_{history.run_name}_local.png",
            history,
            rows,
            "pc1_local",
            "pc2_local",
            local_model,
            standardize,
            "local PCA",
        )
        rows_parts.append(rows)

    rows_all = pd.concat(rows_parts, ignore_index=True)
    rows_all.to_csv(args.output_root / "pca_swarm_points.csv", index=False)
    _plot_global_grid(
        args.output_root / "pca_swarm_global_grid.png",
        histories,
        rows_all,
        "pc1_global",
        "pc2_global",
        global_model,
        standardize,
    )

    selected_rows = rows_all[rows_all["is_selected"]].copy()
    selected_rows.to_csv(args.output_root / "pca_swarm_selected_candidates.csv", index=False)
    summary = {
        "source_root": str(args.source_root.resolve()),
        "selected_root": str(args.selected_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "n_runs": len(histories),
        "runs": [h.run_name for h in histories],
        "n_points": int(len(rows_all)),
        "standardize": bool(standardize),
        "global_pca": global_meta,
        "local_pca": local_meta,
        "selected_candidates": selected_rows[
            [
                "run_name",
                "iter",
                "pop_idx",
                "score_mspd",
                "tau_steps",
                "pc1_global",
                "pc2_global",
                "pc1_local",
                "pc2_local",
            ]
        ].to_dict(orient="records"),
    }
    (args.output_root / "pca_swarm_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote PCA swarm outputs to {args.output_root}")


if __name__ == "__main__":
    main()
