from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config
from paper_suite_common import ensure_dir, load_config, log_event, resolve_path, to_plain, write_json


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return obj.get(key, default)
    except Exception:
        return getattr(obj, key, default)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_best_checkpoint(run_dir: Path) -> tuple[np.ndarray, float]:
    obj = _load_pickle(run_dir / "best.pkl")
    if isinstance(obj, tuple) and len(obj) == 2:
        params, loss = obj
    else:
        params, loss = obj, np.nan
    return np.asarray(params, dtype=np.float32).reshape(-1), float(np.asarray(loss).reshape(-1)[0])


def _load_best_tau(run_dir: Path) -> dict[str, float | int] | None:
    path = run_dir / "best_tau.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    out: dict[str, float | int] = {}
    for key in ("tau_selector_raw", "tau_idx", "tau_frames", "tau_steps"):
        if key in payload:
            val = payload[key]
            out[key] = float(val) if key == "tau_selector_raw" else int(val)
    return out


def _init_lagrangian_points_jax(
    A0: jax.Array,
    *,
    n_particles: int,
    init_mode: str,
    border: str,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    sx = int(A0.shape[0])
    sy = int(A0.shape[1])
    init_mode = str(init_mode).strip().lower()
    if init_mode == "uniform":
        k0, k1 = jax.random.split(key)
        y = jax.random.uniform(k0, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(k1, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif init_mode == "mass":
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(total > 0.0, flat / jnp.maximum(total, 1e-12), jnp.ones_like(flat) / flat.size)
        k_idx, k_jit = jax.random.split(key)
        idx = jax.random.choice(k_idx, flat.size, shape=(n_particles,), replace=True, p=probs)
        iy = idx // sy
        ix = idx % sy
        jitter = jax.random.uniform(k_jit, (n_particles, 2), minval=-0.49, maxval=0.49)
        pts = jnp.stack((iy.astype(jnp.float32) + 0.5, ix.astype(jnp.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unknown metric_lagrangian_init_mode={init_mode!r}. Use 'mass' or 'uniform'.")

    if border == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
        pts = jnp.stack((y, x), axis=-1)
    else:
        lo = float(sigma)
        hi_y = float(sx - sigma)
        hi_x = float(sy - sigma)
        y = jnp.clip(pts[:, 0], lo, hi_y)
        x = jnp.clip(pts[:, 1], lo, hi_x)
        pts = jnp.stack((y, x), axis=-1)
    return pts.astype(jnp.float32)


def _flat_optimization_config(path: Path) -> SimpleNamespace:
    cfg = OmegaConf.load(path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    return SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))


def _checkpoint_root_from_suite(cfg: Any) -> Path:
    section = _get(_get(cfg.get("simulation", {}), "flow_lenia_arun_lagrangian_apf", {}), "optimized_checkpoint_roots", None)
    roots = list(section or [])
    if not roots:
        raise ValueError("No simulation.flow_lenia_arun_lagrangian_apf.optimized_checkpoint_roots configured.")
    root = resolve_path(roots[0])
    if root is None or not root.exists():
        raise FileNotFoundError(f"Optimization checkpoint root not found: {root}")
    return root


def _run_dirs(root: Path, max_runs: int | None) -> list[Path]:
    dirs = [p for p in sorted(root.glob("run_*")) if p.is_dir() and (p / "best.pkl").exists()]
    if max_runs is not None:
        dirs = dirs[: int(max_runs)]
    if not dirs:
        raise FileNotFoundError(f"No run_*/best.pkl checkpoints under {root}")
    return dirs


def _build_rescorer(args: SimpleNamespace):
    base_substrate = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    if getattr(args, "rollout_steps", None) is None:
        args.rollout_steps = substrate.rollout_steps

    metric_space_defaults = util.metric_periodic_space_defaults(base_substrate)
    if getattr(args, "metric_periodic", None) is None:
        args.metric_periodic = bool(metric_space_defaults["periodic"])
    if getattr(args, "metric_domain_y", None) is None:
        args.metric_domain_y = float(metric_space_defaults["domain_y"])
    if getattr(args, "metric_domain_x", None) is None:
        args.metric_domain_x = float(metric_space_defaults["domain_x"])

    metric_cfg = resolve_metric_config(args)
    metric_loss_fn = make_metric_loss_fn(metric_cfg, include_maps=True)
    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))

    if str(getattr(args, "metric_trajectory_source", "lagrangian")).strip().lower() != "lagrangian":
        raise ValueError("This rescorer currently supports metric_trajectory_source='lagrangian' only.")

    def rollout_metric_xy(rng, params):
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng, 4)
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("State does not contain F. Flow-Lenia debug_return_F must be enabled.")
        rt = substrate.RT
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n_particles,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)

        def step_fn(state, key_step):
            st, pts, ch = state
            st = substrate.step_state(key_step, st, params)
            lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
            pts, ch = rt.advect_particles(
                points=pts,
                F=st["F"],
                A=st["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=ch,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return (st, pts, ch), None

        def chunk_fn(state, key_chunk):
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, chunk_steps))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, time_sampling),
        )
        return xy_seq

    @jax.jit
    def eval_one(rng, params, tau_raw):
        rng_roll, rng_metric = jax.random.split(rng)
        xy = rollout_metric_xy(rng_roll, params)
        loss, info = metric_loss_fn(rng_metric, xy, tau_selector=tau_raw)
        score_by_tau = info["score_by_tau"]
        best_idx = jnp.argmax(score_by_tau)
        return {
            "loss": loss,
            "score": info["score"],
            "tau_selected_idx": info["tau_selected_idx"],
            "tau_selected_steps": info["tau_best_steps"],
            "score_by_tau": score_by_tau,
            "max_tau_idx": best_idx,
            "max_tau_steps": info["tau_steps"][best_idx],
            "max_tau_score": score_by_tau[best_idx],
        }

    return eval_one, metric_cfg


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def run(
    config_path: str | Path,
    *,
    optimization_config: str | Path,
    output_dir: str | Path | None,
    force: bool,
    max_runs: int | None,
    n_reps: int | None,
    seed_base: int,
) -> dict[str, Any]:
    suite_cfg, _ = load_config(config_path)
    opt_root = _checkpoint_root_from_suite(suite_cfg)
    opt_cfg_path = resolve_path(optimization_config)
    if opt_cfg_path is None or not opt_cfg_path.exists():
        raise FileNotFoundError(f"Optimization config not found: {opt_cfg_path}")
    flat = _flat_optimization_config(opt_cfg_path)
    reps = int(n_reps if n_reps is not None else getattr(flat, "bs", 4))
    if reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {reps}")

    if output_dir is None:
        root = resolve_path(_get(suite_cfg.get("meta", {}), "output_root", "analysis/results/paper_suite"))
        out_dir = ensure_dir(Path(root) / "flow_lenia")
    else:
        out_dir = ensure_dir(resolve_path(output_dir) or Path(output_dir))
    agg_path = out_dir / "train_objective_rescore.csv"
    reps_path = out_dir / "train_objective_rescore_reps.csv"
    summary_path = out_dir / "train_objective_rescore_summary.json"
    if not force and agg_path.exists() and reps_path.exists():
        return {"status": "exists", "aggregate": str(agg_path), "reps": str(reps_path)}

    eval_one, metric_cfg = _build_rescorer(flat)
    run_dirs = _run_dirs(opt_root, max_runs)
    rep_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    log_event(
        f"Flow-Lenia train objective rescore start n_runs={len(run_dirs)} n_reps={reps} opt_root={opt_root}",
        component="train-rescore",
    )
    for run_dir in tqdm(run_dirs, desc="train-objective-rescore"):
        params_np, saved_loss = _load_best_checkpoint(run_dir)
        tau_info = _load_best_tau(run_dir) or {}
        tau_raw = float(tau_info.get("tau_selector_raw", 0.0))
        params = jnp.asarray(params_np, dtype=jnp.float32)
        train_scores: list[float] = []
        max_scores: list[float] = []
        for rep in range(reps):
            seed = int(seed_base + 1009 * int(run_dir.name.split("_")[-1]) + rep)
            out = eval_one(jax.random.PRNGKey(seed), params, jnp.asarray(tau_raw, dtype=jnp.float32))
            out_np = {key: np.asarray(jax.device_get(value)) for key, value in out.items()}
            train_score = float(np.asarray(out_np["score"]).reshape(-1)[0])
            max_score = float(np.asarray(out_np["max_tau_score"]).reshape(-1)[0])
            train_scores.append(train_score)
            max_scores.append(max_score)
            rep_rows.append(
                {
                    "run": run_dir.name,
                    "rep": rep,
                    "seed": seed,
                    "saved_best_loss": saved_loss,
                    "saved_best_mspd": -saved_loss if np.isfinite(saved_loss) else np.nan,
                    "best_tau_steps_json": tau_info.get("tau_steps", np.nan),
                    "best_tau_selector_raw": tau_raw,
                    "rescore_train_tau_mspd": train_score,
                    "rescore_selected_tau_idx": int(np.asarray(out_np["tau_selected_idx"]).reshape(-1)[0]),
                    "rescore_selected_tau_steps": int(np.asarray(out_np["tau_selected_steps"]).reshape(-1)[0]),
                    "rescore_max_tau_mspd": max_score,
                    "rescore_max_tau_idx": int(np.asarray(out_np["max_tau_idx"]).reshape(-1)[0]),
                    "rescore_max_tau_steps": int(np.asarray(out_np["max_tau_steps"]).reshape(-1)[0]),
                }
            )
        train_arr = np.asarray(train_scores, dtype=np.float64)
        max_arr = np.asarray(max_scores, dtype=np.float64)
        saved_mspd = -saved_loss if np.isfinite(saved_loss) else np.nan
        agg_rows.append(
            {
                "run": run_dir.name,
                "saved_best_loss": saved_loss,
                "saved_best_mspd": saved_mspd,
                "best_tau_steps_json": tau_info.get("tau_steps", np.nan),
                "best_tau_selector_raw": tau_raw,
                "n_reps": reps,
                "rescore_train_tau_mean": float(np.nanmean(train_arr)),
                "rescore_train_tau_median": float(np.nanmedian(train_arr)),
                "rescore_train_tau_max": float(np.nanmax(train_arr)),
                "rescore_train_tau_std": float(np.nanstd(train_arr)),
                "rescore_max_tau_mean": float(np.nanmean(max_arr)),
                "rescore_max_tau_median": float(np.nanmedian(max_arr)),
                "rescore_max_tau_max": float(np.nanmax(max_arr)),
                "rescore_max_tau_std": float(np.nanstd(max_arr)),
                "saved_minus_rescore_train_mean": float(saved_mspd - np.nanmean(train_arr)) if np.isfinite(saved_mspd) else np.nan,
                "saved_minus_rescore_max_mean": float(saved_mspd - np.nanmean(max_arr)) if np.isfinite(saved_mspd) else np.nan,
            }
        )

    _write_csv(reps_path, rep_rows)
    _write_csv(agg_path, agg_rows)
    summary = {
        "status": "ok",
        "optimization_root": str(opt_root),
        "optimization_config": str(opt_cfg_path),
        "n_runs": len(run_dirs),
        "n_reps": reps,
        "seed_base": int(seed_base),
        "metric_summary": metric_summary(metric_cfg),
        "aggregate": str(agg_path),
        "reps": str(reps_path),
    }
    write_json(summary_path, to_plain(summary))
    log_event(f"Flow-Lenia train objective rescore done aggregate={agg_path}", component="train-rescore")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Directly rescore Flow-Lenia best.pkl with the train MSPD objective.")
    parser.add_argument("config", help="paper-suite config")
    parser.add_argument(
        "--optimization-config",
        default="experiments/paper_check_flow_lenia/optimization/config_longrun_check_fix.yaml",
        help="optimization base config used for training",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--n-reps", type=int, default=None, help="fresh stochastic reps per checkpoint; default=optimization.bs")
    parser.add_argument("--seed-base", type=int, default=12345000)
    args = parser.parse_args(argv)
    result = run(
        args.config,
        optimization_config=args.optimization_config,
        output_dir=args.output_dir,
        force=args.force,
        max_runs=args.max_runs,
        n_reps=args.n_reps,
        seed_base=args.seed_base,
    )
    print(to_plain(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
