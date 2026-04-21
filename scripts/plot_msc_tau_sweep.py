from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config


def _resolve_path(path_like: str | None, root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return root / path


def _ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_config(config_path: Path) -> tuple[Any, Any]:
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key, default=None: os.getenv(key, default))
    cfg = OmegaConf.load(str(config_path))
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("metric", {}),
        cfg.get("plot", {}),
    )
    return cfg, flat


def _load_params(run_cfg: dict[str, Any], project_root: Path) -> jax.Array:
    params_path = _resolve_path(run_cfg.get("params_path"), project_root)
    if params_path is not None:
        if not params_path.exists():
            raise FileNotFoundError(f"params_path does not exist: {params_path}")
        if params_path.suffix == ".npy":
            params_obj = np.load(params_path)
        elif params_path.suffix == ".npz":
            with np.load(params_path, allow_pickle=False) as data:
                if "params" not in data.files:
                    raise ValueError(f"{params_path} does not contain array 'params'.")
                params_obj = data["params"]
        elif params_path.suffix == ".pkl":
            params_obj = util.load_pkl(str(params_path.parent), params_path.stem)
        else:
            raise ValueError(f"Unsupported params_path suffix for {params_path}. Use .pkl, .npy, or .npz.")
    else:
        checkpoint_dir = _resolve_path(run_cfg.get("checkpoint_dir"), project_root)
        if checkpoint_dir is None:
            raise ValueError("Each run must define either checkpoint_dir or params_path.")
        params_name = str(run_cfg.get("params_name", "best"))
        params_obj = util.load_pkl(str(checkpoint_dir), params_name)
        if params_obj is None:
            raise FileNotFoundError(f"{params_name}.pkl not found in {checkpoint_dir}")
    params = params_obj[0] if isinstance(params_obj, tuple) else params_obj
    return jnp.asarray(np.asarray(params, dtype=np.float32))


def _extract_state_positions(state) -> jax.Array:
    if isinstance(state, dict):
        if "x" not in state:
            raise ValueError(
                "state_x trajectory source requires a state dict containing key 'x'."
            )
        return jnp.asarray(state["x"], dtype=jnp.float32)
    arr = jnp.asarray(state, dtype=jnp.float32)
    if arr.ndim < 2 or int(arr.shape[-1]) != 2:
        raise ValueError(
            "state_x trajectory source requires a state dict containing key 'x' "
            "or an array with shape (..., 2)."
        )
    return arr


def _unwrap_sampled_xy_np(
    xy_seq: np.ndarray,
    *,
    domain_y: float,
    domain_x: float,
) -> np.ndarray:
    xy = np.asarray(xy_seq, dtype=np.float32)
    if xy.shape[0] <= 1:
        return xy
    dxy = xy[1:] - xy[:-1]
    if domain_y > 0:
        dxy[..., 0] = (dxy[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
    if domain_x > 0:
        dxy[..., 1] = (dxy[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
    increments = np.cumsum(dxy, axis=0, dtype=np.float32)
    return np.concatenate((xy[:1], xy[:1] + increments), axis=0)


def _build_state_x_rollout(substrate, *, rollout_steps: int, sample_every_steps: int):
    time_sampling = int(rollout_steps) // int(sample_every_steps)

    def rollout(rng, params):
        k_state, k_scan = jax.random.split(rng)
        s0 = substrate.init_state(k_state, params)
        _extract_state_positions(s0)

        def step_fn(state, key_step):
            state_next = substrate.step_state(key_step, state, params)
            return state_next, None

        def chunk_fn(state, key_chunk):
            state_next, _ = jax.lax.scan(
                step_fn,
                state,
                jax.random.split(key_chunk, sample_every_steps),
            )
            return state_next, _extract_state_positions(state_next)

        _, xy_seq = jax.lax.scan(
            chunk_fn,
            s0,
            jax.random.split(k_scan, time_sampling),
        )
        return xy_seq

    return jax.jit(rollout)


def _build_metric_cfg(flat_args: Any, substrate) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_space_defaults = util.metric_periodic_space_defaults(substrate.substrate if hasattr(substrate, "substrate") else substrate)
    if (not hasattr(flat_args, "metric_periodic")) or getattr(flat_args, "metric_periodic", None) is None:
        flat_args.metric_periodic = bool(metric_space_defaults["periodic"])
    if (not hasattr(flat_args, "metric_domain_y")) or getattr(flat_args, "metric_domain_y", None) is None:
        flat_args.metric_domain_y = float(metric_space_defaults["domain_y"])
    if (not hasattr(flat_args, "metric_domain_x")) or getattr(flat_args, "metric_domain_x", None) is None:
        flat_args.metric_domain_x = float(metric_space_defaults["domain_x"])
    metric_cfg = resolve_metric_config(flat_args)
    positions_unwrapped = bool(
        str(getattr(flat_args, "metric_trajectory_source", "state_x")).strip().lower() == "state_x"
        and bool(metric_space_defaults["periodic"])
        and bool(getattr(flat_args, "metric_unwrap_state_x", True))
    )
    metric_cfg["positions_unwrapped"] = positions_unwrapped
    return metric_cfg, metric_space_defaults


def _score_trajectory(
    metric_eval,
    metric_cfg: dict[str, Any],
    xy_seq: np.ndarray,
    *,
    metric_seed: int,
) -> dict[str, Any]:
    rng = jax.random.PRNGKey(int(metric_seed))
    _, info = metric_eval(rng, jnp.asarray(xy_seq, dtype=jnp.float32))
    info = jax.device_get(info)
    return {
        "score_by_tau": np.asarray(info["score_by_tau"], dtype=np.float64),
        "amp_by_tau": np.asarray(info["amp_by_tau"], dtype=np.float64),
        "msc_by_tau": np.asarray(info["msc_by_tau"], dtype=np.float64),
        "tau_steps": np.asarray(info["tau_steps"], dtype=np.int32),
        "tau_frames": np.asarray(info["tau_frames"], dtype=np.int32),
        "tau_selected_idx": int(np.asarray(info["tau_selected_idx"]).item()),
        "tau_best_steps": int(np.asarray(info["tau_best_steps"]).item()),
        "tau_best_frames": int(np.asarray(info["tau_best_frames"]).item()),
        "score_scalar": float(np.asarray(info["score"]).item()),
        "amp_scalar": float(np.asarray(info["amp"]).item()),
        "msc_scalar": float(np.asarray(info["msc"]).item()),
    }


def _save_long_tables(per_run_rows: list[dict[str, Any]], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    for row in per_run_rows:
        for tau_steps, tau_frames, score, amp, msc in zip(
            row["tau_steps"],
            row["tau_frames"],
            row["score_by_tau"],
            row["amp_by_tau"],
            row["msc_by_tau"],
        ):
            long_rows.append(
                {
                    "run_id": row["run_id"],
                    "group": row["group"],
                    "label": row["label"],
                    "seed": row["seed"],
                    "tau_steps": int(tau_steps),
                    "tau_frames": int(tau_frames),
                    "score": float(score),
                    "amp": float(amp),
                    "msc": float(msc),
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "tau_sweep_per_run.csv", index=False)

    if long_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            long_df.groupby(["group", "tau_steps", "tau_frames"], dropna=False)
            .agg(
                n_runs=("run_id", "nunique"),
                score_mean=("score", "mean"),
                score_std=("score", "std"),
                score_median=("score", "median"),
                amp_mean=("amp", "mean"),
                amp_std=("amp", "std"),
                amp_median=("amp", "median"),
                msc_mean=("msc", "mean"),
                msc_std=("msc", "std"),
                msc_median=("msc", "median"),
            )
            .reset_index()
        )
    summary_df.to_csv(out_dir / "tau_sweep_group_summary.csv", index=False)
    return long_df, summary_df


def _plot_tau_sweep(
    long_df: pd.DataFrame,
    *,
    out_path: Path,
    value_key: str,
    group_order: list[str],
    colors: dict[str, str],
    title: str,
    ylabel: str,
    show_individual: bool,
    individual_alpha: float,
    summary_stat: str,
    error_band: str,
    y_scale: str,
    log_floor: float,
) -> None:
    if long_df.empty:
        raise ValueError("No tau-sweep rows available for plotting.")

    value_col = {
        "score_by_tau": "score",
        "score": "score",
        "msc_by_tau": "msc",
        "msc": "msc",
        "amp_by_tau": "amp",
        "amp": "amp",
    }.get(str(value_key).strip().lower())
    if value_col is None:
        raise ValueError("plot.value_key must be one of score_by_tau, msc_by_tau, amp_by_tau.")

    summary_stat = str(summary_stat).strip().lower()
    if summary_stat not in {"mean", "median"}:
        raise ValueError("plot.summary_stat must be 'mean' or 'median'.")
    error_band = str(error_band).strip().lower()
    if error_band not in {"none", "std", "sem"}:
        raise ValueError("plot.error_band must be one of 'none', 'std', 'sem'.")
    y_scale = str(y_scale).strip().lower()
    if y_scale not in {"linear", "log"}:
        raise ValueError("plot.y_scale must be one of 'linear', 'log'.")
    log_floor = float(log_floor)
    if log_floor <= 0.0:
        raise ValueError("plot.log_floor must be > 0.")

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)

    for group in group_order:
        sub = long_df.loc[long_df["group"] == group].copy()
        if sub.empty:
            continue
        color = colors.get(group, None)
        if show_individual:
            for run_id, run_df in sub.groupby("run_id", sort=False):
                run_df = run_df.sort_values("tau_steps")
                y_run = run_df[value_col].to_numpy(dtype=np.float64)
                if y_scale == "log":
                    y_run = np.maximum(y_run, log_floor)
                ax.plot(
                    run_df["tau_steps"].to_numpy(dtype=np.float64),
                    y_run,
                    color=color,
                    alpha=float(individual_alpha),
                    linewidth=1.2,
                )
        grouped = (
            sub.groupby("tau_steps", dropna=False)[value_col]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
            .sort_values("tau_steps")
        )
        y = grouped[summary_stat].to_numpy(dtype=np.float64)
        x = grouped["tau_steps"].to_numpy(dtype=np.float64)
        if y_scale == "log":
            y = np.maximum(y, log_floor)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=5.5,
            color=color,
            label=group,
        )
        if error_band != "none":
            if error_band == "std":
                err = grouped["std"].fillna(0.0).to_numpy(dtype=np.float64)
            else:
                denom = np.sqrt(np.maximum(grouped["count"].to_numpy(dtype=np.float64), 1.0))
                err = grouped["std"].fillna(0.0).to_numpy(dtype=np.float64) / denom
            y_lo = y - err
            y_hi = y + err
            if y_scale == "log":
                y_lo = np.maximum(y_lo, log_floor)
                y_hi = np.maximum(y_hi, log_floor)
            ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15)

    ax.set_xlabel("tau (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(y_scale)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/plot_msc_tau_sweep.py <config.yaml>")

    project_root = _REPO_ROOT
    config_path = Path(sys.argv[1]).resolve()
    cfg, flat = _load_config(config_path)

    out_dir = _ensure_dir(_resolve_path(str(cfg.get("meta", {}).get("output_dir")), project_root))
    traj_dir = _ensure_dir(out_dir / "trajectories")
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    runs_cfg = OmegaConf.to_container(cfg.get("runs", []), resolve=True)
    if not runs_cfg:
        raise ValueError("Config must define a non-empty runs list.")

    flat_args = SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))
    if str(getattr(flat_args, "metric_trajectory_source", "state_x")).strip().lower() != "state_x":
        raise ValueError("This script currently supports only simulation.metric_trajectory_source='state_x'.")
    substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate(
            flat_args.substrate,
            **util.substrate_kwargs_from_args(flat_args),
        )
    )

    metric_cfg, metric_space_defaults = _build_metric_cfg(flat_args, substrate)
    sample_every_steps = int(metric_cfg["sample_every_steps"])
    rollout_steps = int(metric_cfg["rollout_steps"])
    rollout_fn = _build_state_x_rollout(
        substrate,
        rollout_steps=rollout_steps,
        sample_every_steps=sample_every_steps,
    )
    metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))

    per_run_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    default_seed = int(cfg.get("meta", {}).get("seed", 0))
    metric_seed_base = int(cfg.get("meta", {}).get("metric_seed_base", 123_000))
    save_trajectories = bool(cfg.get("meta", {}).get("save_trajectories", True))
    reuse_saved_trajectories = bool(cfg.get("meta", {}).get("reuse_saved_trajectories", True))

    for idx, run_cfg in enumerate(runs_cfg):
        run_id = str(run_cfg.get("run_id") or f"run_{idx:03d}")
        label = str(run_cfg.get("label", run_id))
        group = str(run_cfg.get("group", "default"))
        run_seed = int(run_cfg.get("seed", default_seed))
        metric_seed = int(run_cfg.get("metric_seed", metric_seed_base + idx))

        trajectory_path = _resolve_path(run_cfg.get("trajectory_path"), project_root)
        auto_trajectory_path = traj_dir / f"{run_id}.npz"
        xy_seq: np.ndarray
        if trajectory_path is None and reuse_saved_trajectories and auto_trajectory_path.exists():
            trajectory_path = auto_trajectory_path
        if trajectory_path is not None and trajectory_path.exists():
            trajectory_key = str(run_cfg.get("trajectory_key", "xy"))
            with np.load(trajectory_path, allow_pickle=False) as data:
                if trajectory_key not in data.files:
                    raise KeyError(
                        f"{trajectory_path} does not contain {trajectory_key!r}; available keys: {sorted(data.files)}"
                    )
                xy_seq = np.asarray(data[trajectory_key], dtype=np.float32)
        else:
            params = _load_params(run_cfg, project_root)
            rng = jax.random.PRNGKey(run_seed)
            xy_seq = np.asarray(jax.device_get(rollout_fn(rng, params)), dtype=np.float32)
            if metric_cfg.get("positions_unwrapped", False):
                xy_seq = _unwrap_sampled_xy_np(
                    xy_seq,
                    domain_y=float(metric_space_defaults["domain_y"]),
                    domain_x=float(metric_space_defaults["domain_x"]),
                )
            if save_trajectories:
                trajectory_path = auto_trajectory_path
                np.savez_compressed(
                    trajectory_path,
                    xy=xy_seq,
                    sample_every_steps=np.asarray(sample_every_steps, dtype=np.int32),
                    rollout_steps=np.asarray(rollout_steps, dtype=np.int32),
                    seed=np.asarray(run_seed, dtype=np.int64),
                )

        if xy_seq.ndim != 3 or int(xy_seq.shape[-1]) != 2:
            raise ValueError(f"Trajectory for {run_id} must have shape (T, N, 2), got {xy_seq.shape}.")

        sweep = _score_trajectory(
            metric_eval,
            metric_cfg,
            xy_seq,
            metric_seed=metric_seed,
        )
        per_run_rows.append(
            {
                "run_id": run_id,
                "group": group,
                "label": label,
                "seed": run_seed,
                **sweep,
            }
        )
        manifest_rows.append(
            {
                "run_id": run_id,
                "group": group,
                "label": label,
                "seed": run_seed,
                "metric_seed": metric_seed,
                "trajectory_path": None if trajectory_path is None else str(trajectory_path),
                "checkpoint_dir": run_cfg.get("checkpoint_dir"),
                "params_path": run_cfg.get("params_path"),
                "params_name": run_cfg.get("params_name", "best"),
            }
        )

    long_df, summary_df = _save_long_tables(per_run_rows, out_dir)
    _write_json(out_dir / "resolved_run_manifest.json", manifest_rows)

    plot_cfg = OmegaConf.to_container(cfg.get("plot", {}), resolve=True)
    group_order = [str(x) for x in plot_cfg.get("group_order", [])]
    if not group_order:
        group_order = list(dict.fromkeys(long_df["group"].astype(str).tolist()))
    colors = {str(k): str(v) for k, v in dict(plot_cfg.get("colors", {})).items()}
    value_key = str(plot_cfg.get("value_key", "score_by_tau"))
    ylabel = str(plot_cfg.get("ylabel", "MSC score"))
    title = str(plot_cfg.get("title", "MSC tau sweep"))
    show_individual = bool(plot_cfg.get("show_individual", True))
    individual_alpha = float(plot_cfg.get("individual_alpha", 0.2))
    summary_stat = str(plot_cfg.get("summary_stat", "mean"))
    error_band = str(plot_cfg.get("error_band", "std"))
    y_scale = str(plot_cfg.get("y_scale", "linear"))
    log_floor = float(plot_cfg.get("log_floor", 1.0e-12))

    _plot_tau_sweep(
        long_df,
        out_path=out_dir / "tau_sweep_plot.png",
        value_key=value_key,
        group_order=group_order,
        colors=colors,
        title=title,
        ylabel=ylabel,
        show_individual=show_individual,
        individual_alpha=individual_alpha,
        summary_stat=summary_stat,
        error_band=error_band,
        y_scale=y_scale,
        log_floor=log_floor,
    )

    summary_payload = {
        "n_runs": int(len(per_run_rows)),
        "groups": sorted({str(row["group"]) for row in per_run_rows}),
        "tau_steps": []
        if summary_df.empty
        else sorted({int(x) for x in summary_df["tau_steps"].dropna().astype(int).tolist()}),
        "plot_path": str(out_dir / "tau_sweep_plot.png"),
        "per_run_csv": str(out_dir / "tau_sweep_per_run.csv"),
        "group_summary_csv": str(out_dir / "tau_sweep_group_summary.csv"),
    }
    _write_json(out_dir / "summary.json", summary_payload)
    print(f"Saved tau sweep plot to {out_dir / 'tau_sweep_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
