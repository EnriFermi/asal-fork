from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import split
from omegaconf import OmegaConf

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _path_from_repo(raw: Any) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _load_flat_config(path: Path, *, legacy_sigma_collision: bool = False) -> Any:
    cfg = OmegaConf.load(path)
    substrate_cfg = cfg.get("substrate", {})
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        substrate_cfg,
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    if (
        not legacy_sigma_collision
        and str(substrate_cfg.get("substrate", "")).strip().lower() == "lenia_flow"
    ):
        flow_sigma = substrate_cfg.get("flow_sigma", substrate_cfg.get("sigma", None))
        if flow_sigma is not None and flat.get("flow_sigma", None) is None:
            flat.flow_sigma = flow_sigma
    return flat


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
        k0, k1 = split(key)
        y = jax.random.uniform(k0, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(k1, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif init_mode == "mass":
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(total > 0.0, flat / jnp.maximum(total, 1e-12), jnp.ones_like(flat) / flat.size)
        k_idx, k_jit = split(key)
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


def _build_exact_eval(args: Any):
    base_substrate = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    metric_space_defaults = util.metric_periodic_space_defaults(base_substrate)
    if (not hasattr(args, "metric_periodic")) or (getattr(args, "metric_periodic", None) is None):
        args.metric_periodic = bool(metric_space_defaults["periodic"])
    if (not hasattr(args, "metric_domain_y")) or (getattr(args, "metric_domain_y", None) is None):
        args.metric_domain_y = float(metric_space_defaults["domain_y"])
    if (not hasattr(args, "metric_domain_x")) or (getattr(args, "metric_domain_x", None) is None):
        args.metric_domain_x = float(metric_space_defaults["domain_x"])

    metric_cfg = resolve_metric_config(args)
    optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
    positions_unwrapped = False
    metric_cfg["positions_unwrapped"] = positions_unwrapped
    metric_loss_fn = make_metric_loss_fn(metric_cfg)

    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    substrate_param_dims = int(substrate.n_params)
    log_clip_evolution = bool(getattr(args, "log_clip_evolution", True))
    if log_clip_evolution:
        raise ValueError("This exact rescore script currently supports log_clip_evolution=false only.")

    lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))

    if str(getattr(args, "metric_trajectory_source", "lagrangian")).strip().lower() != "lagrangian":
        raise ValueError("This exact rescore script supports metric_trajectory_source='lagrangian' only.")

    def split_candidate_params(params_full):
        params_sub = params_full[:substrate_param_dims]
        tau_selector = params_full[substrate_param_dims] if optimize_tau else None
        return params_sub, tau_selector

    def rollout_metric_xy_and_aux(rng, params):
        k_state, k_pts, k_ch, k_scan = split(rng, 4)
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("State does not contain F. Flow-Lenia debug_return_F must be enabled.")
        if not hasattr(substrate, "RT"):
            raise ValueError("Substrate does not provide RT for lagrangian advection.")
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
            state_next, _ = jax.lax.scan(step_fn, state, split(key_chunk, chunk_steps))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            split(k_scan, time_sampling),
        )
        return xy_seq, {}

    def calc_loss(rng, params_full):
        params, tau_selector = split_candidate_params(params_full)
        rng_roll, rng_metric = split(rng)
        xy_seq, aux_dict = rollout_metric_xy_and_aux(rng_roll, params)
        if optimize_tau:
            msc_loss, msc_dict = metric_loss_fn(rng_metric, xy_seq, tau_selector=tau_selector)
        else:
            msc_loss, msc_dict = metric_loss_fn(rng_metric, xy_seq)
        if aux_dict:
            msc_dict = dict(msc_dict, **aux_dict)
        return msc_loss, msc_dict

    calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))

    @jax.jit
    def eval_chunk_seed_block(params_chunk, seed_keys):
        return calc_loss_vv(seed_keys, params_chunk)

    return eval_chunk_seed_block, metric_cfg, substrate_param_dims, optimize_tau


def _params_full_batch(pop: dict[str, Any], *, i_iter: int, substrate_param_dims: int, optimize_tau: bool) -> np.ndarray:
    params = np.asarray(pop["params"], dtype=np.float32)[int(i_iter)]
    if not optimize_tau:
        return params
    if params.shape[1] == substrate_param_dims + 1:
        return params
    if params.shape[1] != substrate_param_dims:
        raise ValueError(
            f"pop params dim {params.shape[1]} is neither substrate_dim={substrate_param_dims} "
            f"nor full_dim={substrate_param_dims + 1}"
        )
    if "tau_selector_raw" not in pop:
        raise KeyError("Optimization used trainable tau but pop_traj has no tau_selector_raw.")
    tau = np.asarray(pop["tau_selector_raw"], dtype=np.float32)[int(i_iter)]
    return np.concatenate((params, tau[:, None]), axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exactly re-evaluate one selected Flow-Lenia OpenAI-ES population block against stored score_by_seed."
    )
    parser.add_argument(
        "--selected-run-dir",
        default=(
            "experiments/paper_check_flow_lenia/"
            "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
            "optimization/run_000"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--legacy-sigma-collision",
        action="store_true",
        help=(
            "Emulate main_opt_msc.py before the 2026-07-06 flow_sigma namespace fix: "
            "optimization.sigma overwrites substrate.sigma in the flat config."
        ),
    )
    args = parser.parse_args()

    selected_run_dir = _path_from_repo(args.selected_run_dir)
    selected_path = selected_run_dir / "selected_candidate.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}")
    selected = json.loads(selected_path.read_text())
    pop_path = _path_from_repo(selected["source_pop_traj"])
    source_run_dir = _path_from_repo(selected.get("source_run_dir", pop_path.parent))
    opt_config_path = source_run_dir / "optimization_config.yaml"
    if not opt_config_path.exists():
        raise FileNotFoundError(f"Missing optimization config: {opt_config_path}")

    pop = _load_pickle(pop_path)
    i_iter = int(selected["iter"])
    pop_idx = int(selected["pop_idx"])
    seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)[i_iter]
    stored_score_by_seed = np.asarray(pop["score_by_seed"], dtype=np.float64)[i_iter, pop_idx]
    stored_loss_by_seed = -stored_score_by_seed

    flat = _load_flat_config(opt_config_path, legacy_sigma_collision=bool(args.legacy_sigma_collision))
    eval_chunk_seed_block, metric_cfg, substrate_param_dims, optimize_tau = _build_exact_eval(flat)
    params_full = _params_full_batch(
        pop,
        i_iter=i_iter,
        substrate_param_dims=substrate_param_dims,
        optimize_tau=optimize_tau,
    )
    print(
        "Exact optimizer rescore start: "
        f"selected_run={selected_run_dir.name} source_run={source_run_dir.name} "
        f"iter={i_iter} pop_idx={pop_idx} params_full_shape={params_full.shape} "
        f"seed_keys_shape={seed_keys.shape}"
    )
    loss_by_seed, loss_dict_by_seed = eval_chunk_seed_block(
        jnp.asarray(params_full, dtype=jnp.float32),
        jnp.asarray(seed_keys, dtype=jnp.uint32),
    )
    loss_np = np.asarray(jax.device_get(loss_by_seed), dtype=np.float64)
    score_np = -loss_np
    selected_score = score_np[pop_idx]
    selected_loss = loss_np[pop_idx]
    score_diff = selected_score - stored_score_by_seed
    loss_diff = selected_loss - stored_loss_by_seed

    info_np = jax.device_get(loss_dict_by_seed)
    selected_info: dict[str, Any] = {}
    if isinstance(info_np, dict):
        for key in (
            "score",
            "msc",
            "amp",
            "tau_selected_idx",
            "tau_best_steps",
            "score_tau_max",
            "score_tau_mean",
            "score_tau_min",
        ):
            if key in info_np:
                selected_info[key] = [
                    float(x) for x in np.asarray(info_np[key][pop_idx], dtype=np.float64).reshape(-1)
                ]

    payload = {
        "status": "ok",
        "selected_run_dir": str(selected_run_dir),
        "source_run_dir": str(source_run_dir),
        "source_pop_traj": str(pop_path),
        "optimization_config": str(opt_config_path),
        "iter": i_iter,
        "pop_idx": pop_idx,
        "params_full_shape": [int(x) for x in params_full.shape],
        "substrate_param_dims": int(substrate_param_dims),
        "optimize_tau": bool(optimize_tau),
        "legacy_sigma_collision": bool(args.legacy_sigma_collision),
        "resolved_args_sigma": float(flat.get("sigma")),
        "resolved_args_flow_sigma": (
            None if flat.get("flow_sigma", None) is None else float(flat.get("flow_sigma"))
        ),
        "seed_keys": [[int(v) for v in key] for key in seed_keys.reshape((-1, 2))],
        "stored_score_by_seed": [float(x) for x in stored_score_by_seed.reshape(-1)],
        "recomputed_score_by_seed": [float(x) for x in selected_score.reshape(-1)],
        "score_diff_by_seed": [float(x) for x in score_diff.reshape(-1)],
        "score_max_abs_diff": float(np.nanmax(np.abs(score_diff))),
        "score_mean_abs_diff": float(np.nanmean(np.abs(score_diff))),
        "stored_loss_by_seed": [float(x) for x in stored_loss_by_seed.reshape(-1)],
        "recomputed_loss_by_seed": [float(x) for x in selected_loss.reshape(-1)],
        "loss_diff_by_seed": [float(x) for x in loss_diff.reshape(-1)],
        "loss_max_abs_diff": float(np.nanmax(np.abs(loss_diff))),
        "stored_score_mean": float(np.nanmean(stored_score_by_seed)),
        "recomputed_score_mean": float(np.nanmean(selected_score)),
        "score_mean_diff": float(np.nanmean(selected_score) - np.nanmean(stored_score_by_seed)),
        "selected_info_by_seed": selected_info,
        "metric_summary": metric_summary(metric_cfg),
    }
    out_path = _path_from_repo(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
