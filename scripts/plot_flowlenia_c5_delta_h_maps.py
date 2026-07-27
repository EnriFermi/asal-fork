#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The existing paper-suite C5 table was evaluated on JAX CPU. Keep the map
# backend identical so cached scalar scores reproduce that table exactly.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import numpy as np
import pandas as pd

from analysis.history_dependence.paper_check_metric_stats import (
    _build_metric_cfg,
    _load_source_metric_summary,
    _paper_check_metric_seed_base,
)
from analysis.history_dependence.pipeline import load_analysis_config
from analysis.history_dependence.trajectory_metrics import compute_delta_h_summary


_DEFAULT_SUITE_ROOT = (
    _REPO_ROOT
    / "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper"
)
_DEFAULT_FLOW_ROOT = _DEFAULT_SUITE_ROOT / "flow_lenia"
_BRANCHES = (
    ("control_a", "xy_control_a", 0, "Control A (no walls)"),
    ("control_b", "xy_control_b", 1, "Control B (no walls)"),
    ("walls", "xy_walls", 2, "Walls"),
)
_PROTOCOL_VERSION = "flowlenia-c5-delta-h-maps-v2"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _REPO_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _cache_key(
    *,
    source_sha256: str,
    metric_cfg: dict[str, Any],
    metric_seed: int,
    metric_fold_in: int,
    branch: str,
    runtime_fingerprint: dict[str, str],
) -> str:
    payload = {
        "protocol_version": _PROTOCOL_VERSION,
        "source_sha256": source_sha256,
        "metric_cfg": metric_cfg,
        "metric_seed": int(metric_seed),
        "metric_fold_in": int(metric_fold_in),
        "branch": branch,
        "runtime_fingerprint": runtime_fingerprint,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _scalar(data: Any, key: str, default: int = 0) -> int:
    if key not in data.files:
        return int(default)
    return int(np.asarray(data[key]).item())


def _save_cache(
    path: Path,
    *,
    summary: dict[str, Any],
    cache_key: str,
    source_path: Path,
    source_sha256: str,
    metric_cfg: dict[str, Any],
    metric_seed: int,
    metric_fold_in: int,
    branch: str,
    trajectory_start_steps: int,
    runtime_fingerprint: dict[str, str],
) -> None:
    starts = np.asarray(summary["window_start_steps"], dtype=np.int64)
    payload = {
        "delta_h_map": np.asarray(summary["delta_h_map"], dtype=np.float32),
        "delta_h_best": np.asarray(summary["delta_h_best"], dtype=np.float32),
        "score_by_tau": np.asarray(summary["score_by_tau"], dtype=np.float32),
        "amp_by_tau": np.asarray(summary["amp_by_tau"], dtype=np.float32),
        "msc_by_tau": np.asarray(summary["msc_by_tau"], dtype=np.float32),
        "tau_frames": np.asarray(summary["tau_frames"], dtype=np.int32),
        "tau_steps": np.asarray(summary["tau_steps"], dtype=np.int32),
        "window_start_frames": np.asarray(summary["window_start_frames"], dtype=np.int32),
        "window_start_steps": starts.astype(np.int32),
        "window_start_absolute_steps": (starts + trajectory_start_steps).astype(np.int64),
        "tau_best_idx": np.asarray(summary["tau_best_idx"], dtype=np.int32),
        "tau_best_frames": np.asarray(summary["tau_best_frames"], dtype=np.int32),
        "tau_best_steps": np.asarray(summary["tau_best_steps"], dtype=np.int32),
        "score_scalar": np.asarray(summary["score_scalar"], dtype=np.float64),
        "amp_scalar": np.asarray(summary["amp_scalar"], dtype=np.float64),
        "msc_scalar": np.asarray(summary["msc_scalar"], dtype=np.float64),
        "cache_key": np.asarray(cache_key),
        "protocol_version": np.asarray(_PROTOCOL_VERSION),
        "source_path": np.asarray(str(source_path)),
        "source_sha256": np.asarray(source_sha256),
        "metric_cfg_json": np.asarray(_canonical_json(metric_cfg)),
        "metric_seed": np.asarray(metric_seed, dtype=np.int64),
        "metric_fold_in": np.asarray(metric_fold_in, dtype=np.int32),
        "branch": np.asarray(branch),
        "trajectory_start_steps": np.asarray(trajectory_start_steps, dtype=np.int64),
        "runtime_fingerprint_json": np.asarray(_canonical_json(runtime_fingerprint)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_valid_cache(path: Path, expected_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["cache_key"]).item()) != expected_key:
                return None
            return {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError, KeyError):
        return None


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError("Cannot construct heatmap edges for an empty coordinate axis.")
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    if not np.all(np.diff(values) > 0):
        raise ValueError(f"Heatmap coordinates must increase strictly: {values.tolist()}")
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (middle[0] - values[0])],
            middle,
            [values[-1] + (values[-1] - middle[-1])],
        )
    )


def _candidate_label(row: pd.Series) -> str:
    if str(row["candidate_kind_canon"]) == "optimized":
        return "optimized"
    return str(row["candidate_label"])


def _finite_limit(arrays: list[np.ndarray], percentile: float | None = None) -> float:
    finite = np.concatenate(
        [np.asarray(array, dtype=np.float64)[np.isfinite(array)] for array in arrays]
    )
    if finite.size < 1:
        return 1.0
    value = float(np.max(finite) if percentile is None else np.percentile(finite, percentile))
    return max(value, 1e-12)


def _draw_map(
    ax: Any,
    data: dict[str, Any],
    *,
    title: str,
    vmax: float,
    fixed_tau_steps: int,
    show_y_label: bool,
) -> Any:
    delta_h = np.asarray(data["delta_h_map"], dtype=np.float64)
    tau_steps = np.asarray(data["tau_steps"], dtype=np.float64)
    starts = np.asarray(data["window_start_absolute_steps"], dtype=np.float64)
    if delta_h.shape != (tau_steps.size, starts.size):
        raise ValueError(
            f"Map/coordinate mismatch for {title}: map={delta_h.shape}, "
            f"tau={tau_steps.size}, starts={starts.size}"
        )

    image = ax.pcolormesh(
        _coordinate_edges(starts / 1_000_000.0),
        _coordinate_edges(tau_steps / 1_000.0),
        delta_h,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )
    best_tau = int(np.asarray(data["tau_best_steps"]).item())
    ax.axhline(best_tau / 1_000.0, color="white", linewidth=1.15)
    ax.axhline(fixed_tau_steps / 1_000.0, color="#ff9f1c", linewidth=1.05, linestyle="--")
    score = float(np.asarray(data["score_scalar"]).item())
    ax.set_title(f"{title}\nscore={score:.4g}, argmax tau={best_tau:,}", fontsize=9)
    ax.set_xlabel("simulation step (millions)")
    if show_y_label:
        ax.set_ylabel("tau (thousand steps)")
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    return image


def _plot_trial(
    row: pd.Series,
    maps: dict[str, dict[str, Any]],
    *,
    output: Path,
    fixed_tau_steps: int,
    dpi: int,
) -> None:
    vmax = _finite_limit([maps[name]["delta_h_map"] for name, *_ in _BRANCHES])
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.55),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for col, (name, _key, _fold, title) in enumerate(_BRANCHES):
        image = _draw_map(
            axes[0, col],
            maps[name],
            title=title,
            vmax=vmax,
            fixed_tau_steps=fixed_tau_steps,
            show_y_label=col == 0,
        )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.84, pad=0.012, label="Delta-H")
    run_idx = int(row["optimized_run_idx"])
    trial_idx = int(row["trial_idx"])
    fig.suptitle(
        f"Flow-Lenia C5: opt_{run_idx:03d}, {_candidate_label(row)}, trial_{trial_idx:05d}\n"
        f"white: argmax tau; orange dashed: fixed distribution tau={fixed_tau_steps:,}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_run(
    run_idx: int,
    rows: pd.DataFrame,
    maps_by_trial: dict[int, dict[str, dict[str, Any]]],
    *,
    output: Path,
    fixed_tau_steps: int,
    dpi: int,
    percentile: float,
) -> None:
    rows = rows.assign(
        _candidate_order=np.where(rows["candidate_kind_canon"] == "optimized", 0, 1)
    ).sort_values(["_candidate_order", "candidate_idx"])
    arrays = [
        maps_by_trial[int(row["trial_idx"])][name]["delta_h_map"]
        for _, row in rows.iterrows()
        for name, *_ in _BRANCHES
    ]
    vmax = _finite_limit(arrays, percentile=percentile)
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.0, 2.55 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_idx, (_, row) in enumerate(rows.iterrows()):
        trial_idx = int(row["trial_idx"])
        for col, (name, _key, _fold, condition_title) in enumerate(_BRANCHES):
            title = condition_title if row_idx == 0 else ""
            image = _draw_map(
                axes[row_idx, col],
                maps_by_trial[trial_idx][name],
                title=title,
                vmax=vmax,
                fixed_tau_steps=fixed_tau_steps,
                show_y_label=col == 0,
            )
            if col == 0:
                axes[row_idx, col].text(
                    -0.23,
                    0.5,
                    f"{_candidate_label(row)}\ntrial_{trial_idx:05d}",
                    transform=axes[row_idx, col].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, pad=0.012, label="Delta-H")
    fig.suptitle(
        f"Flow-Lenia C5 opt_{run_idx:03d}: controls and walls "
        f"(shared color scale, p{percentile:g} clipped)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute and plot Flow-Lenia C5 Delta-H maps from cached frustration "
            "trajectories without running simulations."
        )
    )
    parser.add_argument(
        "--trial-metrics",
        type=Path,
        default=_DEFAULT_FLOW_ROOT / "frustration_trial_metrics.csv",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=_DEFAULT_FLOW_ROOT / "generated_history_analysis_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_FLOW_ROOT / "c5_delta_h_maps",
    )
    parser.add_argument(
        "--trial",
        type=int,
        action="append",
        help="Only process this trial index; may be supplied more than once.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore valid map caches.")
    parser.add_argument("--cache-only", action="store_true", help="Skip figure rendering.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--run-scale-percentile",
        type=float,
        default=99.5,
        help="Shared per-run color maximum percentile; per-trial plots use the exact maximum.",
    )
    parser.add_argument(
        "--score-parity-atol",
        type=float,
        default=1e-12,
        help="Maximum allowed absolute difference from the existing C5 scalar score.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics_path = _resolve(args.trial_metrics)
    analysis_config_path = _resolve(args.analysis_config)
    output_dir = _resolve(args.output_dir)
    rows = pd.read_csv(metrics_path)
    if args.trial:
        requested = {int(value) for value in args.trial}
        rows = rows.loc[rows["trial_idx"].astype(int).isin(requested)].copy()
        missing = requested - set(rows["trial_idx"].astype(int))
        if missing:
            raise ValueError(f"Requested trial indices are absent: {sorted(missing)}")
    if rows.empty:
        raise ValueError(f"No trials selected from {metrics_path}")

    analysis_cfg = load_analysis_config(analysis_config_path)
    traj_cfg = dict(analysis_cfg["trajectories"])
    fixed_tau_steps = int(
        traj_cfg.get("fixed_tau_distribution_steps", traj_cfg.get("metric_tau_steps", 3000))
    )
    frustration_roots = {_resolve(value) for value in rows["frustration_root"].astype(str)}
    if len(frustration_roots) != 1:
        raise ValueError(f"Expected one frustration root, found {sorted(map(str, frustration_roots))}")
    frustration_root = next(iter(frustration_roots))
    source_metric_summary = _load_source_metric_summary(frustration_root)
    runtime_fingerprint = {
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jax.lib.__version__),
        "jax_backend": str(jax.default_backend()),
    }

    cache_dir = output_dir / "cache"
    maps_by_trial: dict[int, dict[str, dict[str, Any]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    total_maps = len(rows) * len(_BRANCHES)
    completed = 0

    for _, row in rows.sort_values("trial_idx").iterrows():
        trial_idx = int(row["trial_idx"])
        source_path = Path(str(row["lagrangian_path"]))
        if not source_path.is_absolute():
            source_path = frustration_root / source_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha256 = _sha256(source_path)
        metric_cfg = _build_metric_cfg(
            analysis_cfg,
            lagrangian_path=source_path,
            source_metric_summary=source_metric_summary,
        )
        metric_seed = _paper_check_metric_seed_base(dict(row))
        if metric_seed is None:
            raise ValueError(f"Could not derive metric RNG seed for trial {trial_idx}")

        trial_maps: dict[str, dict[str, Any]] = {}
        with np.load(source_path, allow_pickle=False) as trajectory:
            trajectory_start_steps = _scalar(trajectory, "trajectory_start_steps")
            for branch, xy_key, fold_in, _title in _BRANCHES:
                expected_key = _cache_key(
                    source_sha256=source_sha256,
                    metric_cfg=metric_cfg,
                    metric_seed=metric_seed,
                    metric_fold_in=fold_in,
                    branch=branch,
                    runtime_fingerprint=runtime_fingerprint,
                )
                cache_path = cache_dir / f"trial_{trial_idx:05d}_{branch}_delta_h.npz"
                cached = None if args.force else _load_valid_cache(cache_path, expected_key)
                cache_status = "hit"
                if cached is None:
                    cache_status = "computed"
                    summary = compute_delta_h_summary(
                        np.asarray(trajectory[xy_key], dtype=np.float64),
                        metric_cfg,
                        metric_rng_seed=metric_seed,
                        metric_rng_fold_in=fold_in,
                    )
                    _save_cache(
                        cache_path,
                        summary=summary,
                        cache_key=expected_key,
                        source_path=source_path,
                        source_sha256=source_sha256,
                        metric_cfg=metric_cfg,
                        metric_seed=metric_seed,
                        metric_fold_in=fold_in,
                        branch=branch,
                        trajectory_start_steps=trajectory_start_steps,
                        runtime_fingerprint=runtime_fingerprint,
                    )
                    cached = _load_valid_cache(cache_path, expected_key)
                    if cached is None:
                        raise RuntimeError(f"Could not validate newly written cache: {cache_path}")

                delta_h = np.asarray(cached["delta_h_map"], dtype=np.float64)
                if delta_h.ndim != 2 or not np.all(np.isfinite(delta_h)):
                    raise ValueError(f"Invalid Delta-H map in {cache_path}: shape={delta_h.shape}")
                score_scalar = float(np.asarray(cached["score_scalar"]).item())
                expected_score_column = f"msc_score_{branch}"
                expected_score = float(row[expected_score_column])
                score_abs_diff = abs(score_scalar - expected_score)
                if score_abs_diff > float(args.score_parity_atol):
                    raise ValueError(
                        f"C5 score parity failed for trial_{trial_idx:05d} {branch}: "
                        f"recomputed={score_scalar:.17g}, table={expected_score:.17g}, "
                        f"abs_diff={score_abs_diff:.3g}, atol={args.score_parity_atol:.3g}, "
                        f"backend={runtime_fingerprint['jax_backend']}"
                    )
                trial_maps[branch] = cached
                completed += 1
                print(
                    f"[{completed:03d}/{total_maps:03d}] trial_{trial_idx:05d} "
                    f"{branch}: {cache_status} shape={delta_h.shape}",
                    flush=True,
                )
                manifest_rows.append(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "trial_idx": trial_idx,
                        "optimized_run_idx": int(row["optimized_run_idx"]),
                        "candidate_kind": str(row["candidate_kind_canon"]),
                        "candidate_idx": int(row["candidate_idx"]),
                        "candidate_label": _candidate_label(row),
                        "branch": branch,
                        "condition": "walls" if branch == "walls" else "no_walls",
                        "metric_seed": int(metric_seed),
                        "metric_fold_in": int(fold_in),
                        "source_path": str(source_path),
                        "source_sha256": source_sha256,
                        "cache_path": str(cache_path),
                        "cache_key": expected_key,
                        "cache_status": cache_status,
                        "jax_backend": runtime_fingerprint["jax_backend"],
                        "jax_version": runtime_fingerprint["jax_version"],
                        "jaxlib_version": runtime_fingerprint["jaxlib_version"],
                        "map_tau_count": int(delta_h.shape[0]),
                        "map_window_count": int(delta_h.shape[1]),
                        "delta_h_min": float(np.min(delta_h)),
                        "delta_h_mean": float(np.mean(delta_h)),
                        "delta_h_max": float(np.max(delta_h)),
                        "score_scalar": score_scalar,
                        "expected_score_scalar": expected_score,
                        "score_abs_diff": score_abs_diff,
                        "tau_best_steps": int(np.asarray(cached["tau_best_steps"]).item()),
                    }
                )
        maps_by_trial[trial_idx] = trial_maps

    manifest = pd.DataFrame(manifest_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    metric_cfg_values = {
        str(np.asarray(maps_by_trial[trial_idx]["control_a"]["metric_cfg_json"]).item())
        for trial_idx in maps_by_trial
    }
    if len(metric_cfg_values) != 1:
        raise ValueError(f"Metric config differs across selected trials: {len(metric_cfg_values)} configs")
    (output_dir / "metric_config.json").write_text(
        json.dumps(json.loads(next(iter(metric_cfg_values))), indent=2, sort_keys=True) + "\n"
    )

    trial_figure_paths: list[str] = []
    run_figure_paths: list[str] = []
    if not args.cache_only:
        for _, row in rows.sort_values("trial_idx").iterrows():
            trial_idx = int(row["trial_idx"])
            run_idx = int(row["optimized_run_idx"])
            candidate = _candidate_label(row)
            output = (
                output_dir
                / "per_trial"
                / f"opt_{run_idx:03d}"
                / f"trial_{trial_idx:05d}_{candidate}_delta_h.png"
            )
            _plot_trial(
                row,
                maps_by_trial[trial_idx],
                output=output,
                fixed_tau_steps=fixed_tau_steps,
                dpi=args.dpi,
            )
            trial_figure_paths.append(str(output))

        for run_idx, group in rows.groupby("optimized_run_idx", sort=True):
            expected_candidates = {"optimized", "random_000", "random_001", "random_002"}
            actual_candidates = {_candidate_label(row) for _, row in group.iterrows()}
            if actual_candidates != expected_candidates:
                print(
                    f"[plots] opt_{int(run_idx):03d}: skipping run sheet for partial candidate "
                    f"set {sorted(actual_candidates)}",
                    flush=True,
                )
                continue
            output = output_dir / "per_run" / f"opt_{int(run_idx):03d}_delta_h.png"
            _plot_run(
                int(run_idx),
                group,
                maps_by_trial,
                output=output,
                fixed_tau_steps=fixed_tau_steps,
                dpi=args.dpi,
                percentile=float(args.run_scale_percentile),
            )
            run_figure_paths.append(str(output))

    summary = {
        "protocol_version": _PROTOCOL_VERSION,
        "status": "complete",
        "source_trial_metrics": str(metrics_path),
        "analysis_config": str(analysis_config_path),
        "frustration_root": str(frustration_root),
        "trial_count": int(rows["trial_idx"].nunique()),
        "map_count": int(len(manifest)),
        "no_walls_map_count": int((manifest["condition"] == "no_walls").sum()),
        "walls_map_count": int((manifest["condition"] == "walls").sum()),
        "cache_hits": int((manifest["cache_status"] == "hit").sum()),
        "maps_computed": int((manifest["cache_status"] == "computed").sum()),
        "runtime_fingerprint": runtime_fingerprint,
        "score_parity_atol": float(args.score_parity_atol),
        "max_score_abs_diff": float(manifest["score_abs_diff"].max()),
        "all_scores_match_existing_c5_table": bool(
            (manifest["score_abs_diff"] <= float(args.score_parity_atol)).all()
        ),
        "all_maps_finite": bool(
            np.isfinite(
                manifest[["delta_h_min", "delta_h_mean", "delta_h_max"]].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "map_shapes": sorted(
            {
                f"{int(row.map_tau_count)}x{int(row.map_window_count)}"
                for row in manifest.itertuples()
            }
        ),
        "fixed_tau_distribution_steps": fixed_tau_steps,
        "trial_figure_count": len(trial_figure_paths),
        "run_figure_count": len(run_figure_paths),
        "trial_figures": trial_figure_paths,
        "run_figures": run_figure_paths,
        "manifest": str(output_dir / "manifest.csv"),
        "metric_config": str(output_dir / "metric_config.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status",
        "trial_count",
        "map_count",
        "no_walls_map_count",
        "walls_map_count",
        "cache_hits",
        "maps_computed",
        "map_shapes",
        "trial_figure_count",
        "run_figure_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
