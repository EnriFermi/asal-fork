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

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config
from flowlenia_exact_optimizer_rescore_one import (
    _init_lagrangian_points_jax,
    _load_flat_config,
    _load_pickle,
    _params_full_batch,
    _path_from_repo,
)


def _build_score_fn(args: Any):
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
    metric_loss_fn = make_metric_loss_fn(metric_cfg)
    optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
    substrate_param_dims = int(substrate.n_params)
    if bool(getattr(args, "log_clip_evolution", True)):
        raise ValueError("This diagnostic supports log_clip_evolution=false only.")
    if str(getattr(args, "metric_trajectory_source", "lagrangian")).strip().lower() != "lagrangian":
        raise ValueError("This diagnostic supports metric_trajectory_source='lagrangian' only.")

    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))

    def split_candidate_params(params_full):
        params_sub = params_full[:substrate_param_dims]
        tau_selector = params_full[substrate_param_dims] if optimize_tau else None
        return params_sub, tau_selector

    def eval_key_parts(eval_key):
        rng_roll, rng_metric = split(eval_key)
        k_state, k_pts, k_ch, k_scan = split(rng_roll, 4)
        return k_state, k_pts, k_ch, k_scan, rng_metric

    def rollout_from_parts(k_state, k_pts, k_ch, k_scan, params):
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("Flow-Lenia state has no F; debug_return_F must be enabled.")
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
        return xy_seq

    @jax.jit
    def score_from_parts(k_state, k_pts, k_ch, k_scan, rng_metric, params_full):
        params, tau_selector = split_candidate_params(params_full)
        xy_seq = rollout_from_parts(k_state, k_pts, k_ch, k_scan, params)
        if optimize_tau:
            loss, info = metric_loss_fn(rng_metric, xy_seq, tau_selector=tau_selector)
        else:
            loss, info = metric_loss_fn(rng_metric, xy_seq)
        return -loss, info

    @jax.jit
    def initial_ap_diff(k_state_a, k_state_b, params_full):
        params, _tau_selector = split_candidate_params(params_full)
        sa = substrate.init_state(k_state_a, params)
        sb = substrate.init_state(k_state_b, params)
        return (
            jnp.max(jnp.abs(sa["A"] - sb["A"])),
            jnp.max(jnp.abs(sa["P"] - sb["P"])),
        )

    return score_from_parts, initial_ap_diff, eval_key_parts, substrate_param_dims, optimize_tau


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score one Flow-Lenia selected candidate with same A/P init but changed non-init stochastic keys."
    )
    parser.add_argument(
        "--selected-run-dir",
        default=(
            "experiments/paper_check_flow_lenia/"
            "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
            "optimization/run_000"
        ),
    )
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--alt-seed", type=int, default=991337)
    parser.add_argument(
        "--variant",
        choices=[
            "same_init_alt_rest",
            "same_init_same_lag_init_alt_scan_metric",
            "same_init_same_lag_init_alt_scan_same_metric",
            "same_init_same_rollout_alt_metric",
        ],
        default="same_init_alt_rest",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    selected_run_dir = _path_from_repo(args.selected_run_dir)
    selected = json.loads((selected_run_dir / "selected_candidate.json").read_text())
    pop_path = _path_from_repo(selected["source_pop_traj"])
    source_run_dir = _path_from_repo(selected.get("source_run_dir", pop_path.parent))
    pop = _load_pickle(pop_path)

    i_iter = int(selected["iter"])
    pop_idx = int(selected["pop_idx"])
    seed_idx = int(args.seed_idx)
    stored_score = float(np.asarray(pop["score_by_seed"], dtype=np.float64)[i_iter, pop_idx, seed_idx])
    opt_scores = np.asarray(pop["score_by_seed"], dtype=np.float64)[i_iter, pop_idx]
    opt_mean = float(np.mean(opt_scores))

    flat = _load_flat_config(source_run_dir / "optimization_config.yaml", legacy_sigma_collision=True)
    score_from_parts, initial_ap_diff, key_parts, substrate_param_dims, optimize_tau = _build_score_fn(flat)
    params_full_all = _params_full_batch(
        pop,
        i_iter=i_iter,
        substrate_param_dims=substrate_param_dims,
        optimize_tau=optimize_tau,
    )
    params_full = jnp.asarray(params_full_all[pop_idx], dtype=jnp.float32)

    exact_key = jnp.asarray(np.asarray(pop["seed_keys"], dtype=np.uint32)[i_iter, seed_idx], dtype=jnp.uint32)
    alt_key = jax.random.PRNGKey(int(args.alt_seed))
    exact = key_parts(exact_key)
    alt = key_parts(alt_key)

    if args.variant == "same_init_alt_rest":
        parts = (exact[0], alt[1], alt[2], alt[3], alt[4])
    elif args.variant == "same_init_same_lag_init_alt_scan_metric":
        parts = (exact[0], exact[1], exact[2], alt[3], alt[4])
    elif args.variant == "same_init_same_lag_init_alt_scan_same_metric":
        parts = (exact[0], exact[1], exact[2], alt[3], exact[4])
    else:
        parts = (exact[0], exact[1], exact[2], exact[3], alt[4])

    score, info = score_from_parts(*parts, params_full)
    score_f = float(np.asarray(jax.device_get(score), dtype=np.float64))
    ap_a_diff, ap_p_diff = initial_ap_diff(exact[0], parts[0], params_full)
    info_np = jax.device_get(info)
    info_small: dict[str, float] = {}
    if isinstance(info_np, dict):
        for key in ("score", "msc", "amp", "tau_selected_idx", "tau_best_steps"):
            if key in info_np:
                info_small[key] = float(np.asarray(info_np[key], dtype=np.float64))

    payload = {
        "status": "ok",
        "selected_run_dir": str(selected_run_dir),
        "source_run_dir": str(source_run_dir),
        "iter": i_iter,
        "pop_idx": pop_idx,
        "seed_idx": seed_idx,
        "variant": args.variant,
        "alt_seed": int(args.alt_seed),
        "resolved_args_sigma": float(flat.get("sigma")),
        "resolved_args_flow_sigma": None if flat.get("flow_sigma", None) is None else float(flat.get("flow_sigma")),
        "initial_A_max_abs_diff": float(np.asarray(jax.device_get(ap_a_diff), dtype=np.float64)),
        "initial_P_max_abs_diff": float(np.asarray(jax.device_get(ap_p_diff), dtype=np.float64)),
        "opt_seed_score": stored_score,
        "opt_mean_score": opt_mean,
        "same_init_score": score_f,
        "diff_vs_opt_seed": score_f - stored_score,
        "abs_diff_vs_opt_seed": abs(score_f - stored_score),
        "diff_vs_opt_mean": score_f - opt_mean,
        "abs_diff_vs_opt_mean": abs(score_f - opt_mean),
        "opt_score_by_seed": [float(x) for x in opt_scores],
        "info": info_small,
    }
    out_path = _path_from_repo(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
