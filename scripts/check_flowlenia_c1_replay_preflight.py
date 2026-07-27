from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from paper_suite_common import as_list, load_config, resolve_path, write_json


DEFAULT_CONFIG = (
    "experiments/paper_suite/"
    "config_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random.yaml"
)
DEFAULT_SELECTED_ROOT = (
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/optimization"
)
DEFAULT_SCORES_CSV = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
    "flow_lenia/checkpoint_scores.csv"
)


_MISSING = object()


def _get(cfg: Any, key: Any, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(key, (tuple, list)):
        cur = cfg
        for part in key:
            cur = _get(cur, part, _MISSING)
            if cur is _MISSING:
                return default
        return cur
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default) if isinstance(key, str) else default


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _array_sha256(arr: Any) -> str:
    data = np.asarray(arr, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _selected_checkpoint_audit(run_dir: Path, params: np.ndarray) -> dict[str, Any]:
    meta_path = run_dir / "selected_candidate.json"
    out: dict[str, Any] = {
        "selected_run_dir": str(run_dir),
        "best_params_sha256": _array_sha256(params),
        "has_selected_candidate_json": bool(meta_path.exists()),
    }
    if not meta_path.exists():
        return out

    meta = json.loads(meta_path.read_text())
    out.update(
        {
            "source_run_dir": str(meta.get("source_run_dir", "")),
            "source_pop_traj": str(meta.get("source_pop_traj", "")),
            "selected_iter": int(meta.get("iter", -1)),
            "selected_pop_idx": int(meta.get("pop_idx", -1)),
            "selected_score_mspd": float(meta.get("score_mspd", float("nan"))),
            "selected_seed_scores_mspd": [float(x) for x in meta.get("seed_scores_mspd", [])],
            "selected_tau": meta.get("tau", {}),
            "selection_rule": str(meta.get("selection_rule", "")),
        }
    )

    pop_path = Path(str(meta.get("source_pop_traj", "")))
    if not pop_path.is_absolute():
        pop_path = _REPO_ROOT / pop_path
    out["source_pop_traj_resolved"] = str(pop_path)
    out["source_pop_traj_exists"] = bool(pop_path.exists())
    if not pop_path.exists():
        return out

    pop = _load_pickle(pop_path)
    i_iter = int(meta.get("iter", -1))
    pop_idx = int(meta.get("pop_idx", -1))
    pop_params = np.asarray(pop.get("params"), dtype=np.float32)
    if pop_params.ndim != 3 or i_iter < 0 or pop_idx < 0:
        out["pop_param_check_error"] = f"bad pop params shape/index: shape={pop_params.shape}, iter={i_iter}, pop_idx={pop_idx}"
        return out
    if i_iter >= pop_params.shape[0] or pop_idx >= pop_params.shape[1]:
        out["pop_param_check_error"] = f"selected index outside pop params shape={pop_params.shape}"
        return out

    expected = np.asarray(pop_params[i_iter, pop_idx], dtype=np.float32).reshape(-1)
    diff = np.asarray(params, dtype=np.float32).reshape(-1) - expected
    out.update(
        {
            "pop_params_shape": list(pop_params.shape),
            "pop_selected_params_sha256": _array_sha256(expected),
            "params_match_selected_pop": bool(np.nanmax(np.abs(diff)) == 0.0),
            "params_vs_pop_max_abs_diff": float(np.nanmax(np.abs(diff))),
            "params_vs_pop_mean_abs_diff": float(np.nanmean(np.abs(diff))),
        }
    )
    for key in ("score_by_seed", "objective_score", "tau_steps"):
        if key not in pop:
            continue
        arr = np.asarray(pop[key])
        if arr.ndim >= 2 and i_iter < arr.shape[0] and pop_idx < arr.shape[1]:
            value = arr[i_iter, pop_idx]
            out[f"pop_selected_{key}"] = (
                [float(x) for x in np.asarray(value).reshape(-1)]
                if np.asarray(value).size > 1
                else float(np.asarray(value).reshape(-1)[0])
            )
    return out


def _flat_optimization_config(path: Path, *, legacy_sigma_collision: bool = False) -> Any:
    cfg = OmegaConf.load(path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    substrate_cfg = cfg.get("substrate", {})
    if legacy_sigma_collision:
        if flat.get("flow_sigma", None) is not None:
            flat.flow_sigma = None
    elif str(substrate_cfg.get("substrate", "")).strip().lower() == "lenia_flow":
        flow_sigma = substrate_cfg.get("flow_sigma", substrate_cfg.get("sigma", None))
        if flow_sigma is not None:
            # main_opt_msc reconstructs this in memory so the optimizer search
            # radius (`optimization.sigma`) cannot become Flow-Lenia's physical
            # reintegration sigma.  Archived optimization_config.yaml files may
            # not contain flow_sigma, so the preflight must reconstruct it too.
            flat.flow_sigma = flow_sigma
    return flat


def _make_substrate(args: Any):
    import substrates
    import util

    base = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    if hasattr(base, "debug_return_F"):
        base.debug_return_F = True
    return substrates.FlattenSubstrateParameters(base)


def _optimization_lagrangian_xy(
    *,
    opt_flat: Any,
    params: np.ndarray,
    run_seed: int,
    rollout_steps: int,
    sample_every_steps: int,
    include_initial: bool = False,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    args.rollout_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params, dtype=np.float32))
    rng_roll = _metric_roll_key(args, jax.random.PRNGKey(int(run_seed)))
    k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
    s0 = substrate.init_state(k_state, params_j)
    if "F" not in s0:
        raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(_get(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(_get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))
    lag_noise_model = str(_get(args, "metric_lagrangian_noise_model", "rt_box"))
    lag_diffusion_scale = float(_get(args, "metric_lagrangian_diffusion_scale", 1.0))

    pts0 = _init_lagrangian_points_jax(
        s0["A"],
        n_particles=lag_n,
        init_mode=lag_init_mode,
        border=str(getattr(rt, "border", "wall")),
        sigma=float(getattr(rt, "sigma", 0.0)),
        key=k_pts,
    )
    if lag_channel_mode in ("fixed", "resample"):
        ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
    else:
        ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

    def step_fn(state, key_step):
        st, pts, ch = state
        st = substrate.step_state(key_step, st, params_j)
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
        state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
        return state_next, state_next[1]

    time_sampling = int(rollout_steps) // int(sample_every_steps)
    (_, _, _), xy_seq = jax.lax.scan(
        chunk_fn,
        (s0, pts0, ch0),
        jax.random.split(k_scan, time_sampling),
    )
    if include_initial:
        xy_seq = jnp.concatenate((pts0[None, :, :], xy_seq), axis=0)
    return np.asarray(jax.device_get(xy_seq), dtype=np.float32)


def _optimization_initial_snapshot(
    *,
    opt_flat: Any,
    params: np.ndarray,
    run_seed: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params, dtype=np.float32))
    rng_roll = _metric_roll_key(args, jax.random.PRNGKey(int(run_seed)))
    k_state, k_pts, k_ch, _k_scan = jax.random.split(rng_roll, 4)
    s0 = substrate.init_state(k_state, params_j)
    if "F" not in s0:
        raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))
    pts0 = _init_lagrangian_points_jax(
        s0["A"],
        n_particles=lag_n,
        init_mode=lag_init_mode,
        border=str(getattr(rt, "border", "wall")),
        sigma=float(getattr(rt, "sigma", 0.0)),
        key=k_pts,
    )
    if lag_channel_mode in ("fixed", "resample"):
        ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
    else:
        ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)
    return {
        "A": np.asarray(jax.device_get(s0["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(s0["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(s0["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(pts0), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(ch0), dtype=np.int32),
    }


def _rollout_flat_initial_snapshot(
    *,
    flat_args: dict[str, Any],
    params: np.ndarray,
    run_seed: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(dict(flat_args))
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params, dtype=np.float32))
    rng_roll = _metric_roll_key(args, jax.random.PRNGKey(int(run_seed)))
    k_state, k_pts, k_ch, _k_scan = jax.random.split(rng_roll, 4)
    s0 = substrate.init_state(k_state, params_j)
    if "F" not in s0:
        raise ValueError("Rollout-flat replay requires Flow-Lenia state with F.")
    rt = substrate.RT

    lag_n = int(_get(args, "lagrangian_n_particles", _get(args, "metric_lagrangian_n_particles", 8192)))
    lag_init_mode = str(_get(args, "lagrangian_init_mode", _get(args, "metric_lagrangian_init_mode", "mass")))
    lag_channel_mode = str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample")))
    pts0 = _init_lagrangian_points_jax(
        s0["A"],
        n_particles=lag_n,
        init_mode=lag_init_mode,
        border=str(getattr(rt, "border", "wall")),
        sigma=float(getattr(rt, "sigma", 0.0)),
        key=k_pts,
    )
    if lag_channel_mode in ("fixed", "resample"):
        ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
    else:
        ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)
    return {
        "A": np.asarray(jax.device_get(s0["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(s0["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(s0["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(pts0), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(ch0), dtype=np.int32),
    }


def _optimizer_batch_reference_inputs(run_dir: Path, seed_idx: int) -> dict[str, Any] | None:
    meta_path = run_dir / "selected_candidate.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    pop_path = Path(str(meta.get("source_pop_traj", "")))
    if not pop_path.is_absolute():
        pop_path = _REPO_ROOT / pop_path
    if not pop_path.exists():
        return None
    pop = _load_pickle(pop_path)
    if "params" not in pop or "seed_keys" not in pop:
        return None
    i_iter = int(meta.get("iter", -1))
    pop_idx = int(meta.get("pop_idx", -1))
    params = np.asarray(pop["params"], dtype=np.float32)
    seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)
    if params.ndim != 3 or seed_keys.ndim != 3:
        return None
    if i_iter < 0 or i_iter >= params.shape[0] or pop_idx < 0 or pop_idx >= params.shape[1]:
        return None
    if int(seed_idx) < 0 or int(seed_idx) >= seed_keys.shape[1]:
        return None
    return {
        "iter": i_iter,
        "pop_idx": pop_idx,
        "seed_idx": int(seed_idx),
        "params_batch": np.asarray(params[i_iter], dtype=np.float32),
        "seed_keys": np.asarray(seed_keys[i_iter], dtype=np.uint32),
        "selected_seed_key": [int(x) for x in np.asarray(seed_keys[i_iter, int(seed_idx)]).reshape(-1)],
    }


def _optimization_lagrangian_xy_from_optimizer_batch(
    *,
    opt_flat: Any,
    params_batch: np.ndarray,
    seed_keys: np.ndarray,
    pop_idx: int,
    seed_idx: int,
    rollout_steps: int,
    sample_every_steps: int,
    include_initial: bool = False,
    jit_compile: bool = False,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    args.rollout_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params_batch, dtype=np.float32))
    seed_keys_j = jnp.asarray(np.asarray(seed_keys, dtype=np.uint32))

    # Force RT construction before the vmapped rollout closes over it, matching
    # the batched APF runner and avoiding accidental dependence on trace order.
    _ = substrate.init_state(jax.random.PRNGKey(0), params_j[0])
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(_get(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(_get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))
    lag_noise_model = str(_get(args, "metric_lagrangian_noise_model", "rt_box"))
    lag_diffusion_scale = float(_get(args, "metric_lagrangian_diffusion_scale", 1.0))

    def rollout_one(eval_key, params):
        rng_roll = _metric_roll_key(args, eval_key)
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")

        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

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
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
            return state_next, state_next[1]

        time_sampling = int(rollout_steps) // int(sample_every_steps)
        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, time_sampling),
        )
        if include_initial:
            xy_seq = jnp.concatenate((pts0[None, :, :], xy_seq), axis=0)
        return xy_seq

    # This matches main_opt_msc.calc_loss_vv:
    # vmap(vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0)).
    def rollout_all(seed_keys_in, params_in):
        return jax.vmap(
            lambda params: jax.vmap(lambda key: rollout_one(key, params))(seed_keys_in)
        )(params_in)

    if jit_compile:
        xy_all = jax.jit(rollout_all)(seed_keys_j, params_j)
    else:
        xy_all = rollout_all(seed_keys_j, params_j)
    xy = xy_all[int(pop_idx), int(seed_idx)]
    return np.asarray(jax.device_get(xy), dtype=np.float32)


def _optimization_lagrangian_xy_from_flat_pair_batch(
    *,
    opt_flat: Any,
    params_batch: np.ndarray,
    seed_keys: np.ndarray,
    pop_idx: int,
    seed_idx: int,
    rollout_steps: int,
    sample_every_steps: int,
    jit_compile: bool = False,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    args.rollout_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    substrate = _make_substrate(args)
    params_np = np.asarray(params_batch, dtype=np.float32)
    seed_np = np.asarray(seed_keys, dtype=np.uint32)
    n_pop = int(params_np.shape[0])
    n_seed = int(seed_np.shape[0])
    params_flat = np.repeat(params_np, n_seed, axis=0)
    seed_flat = np.tile(seed_np[None, :, :], (n_pop, 1, 1)).reshape((n_pop * n_seed, 2))
    params_j = jnp.asarray(params_flat)
    seed_keys_j = jnp.asarray(seed_flat)

    _ = substrate.init_state(jax.random.PRNGKey(0), params_j[0])
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(_get(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(_get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))
    lag_noise_model = str(_get(args, "metric_lagrangian_noise_model", "rt_box"))
    lag_diffusion_scale = float(_get(args, "metric_lagrangian_diffusion_scale", 1.0))

    def rollout_one(eval_key, params):
        rng_roll = _metric_roll_key(args, eval_key)
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

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
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
            return state_next, state_next[1]

        time_sampling = int(rollout_steps) // int(sample_every_steps)
        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, time_sampling),
        )
        return xy_seq

    def rollout_all(seed_keys_in, params_in):
        return jax.vmap(rollout_one)(seed_keys_in, params_in)

    if jit_compile:
        xy_all = jax.jit(rollout_all)(seed_keys_j, params_j)
    else:
        xy_all = rollout_all(seed_keys_j, params_j)
    flat_idx = int(pop_idx) * n_seed + int(seed_idx)
    xy = xy_all[flat_idx]
    return np.asarray(jax.device_get(xy), dtype=np.float32)


def _optimization_initial_snapshot_from_optimizer_batch(
    *,
    opt_flat: Any,
    params_batch: np.ndarray,
    seed_keys: np.ndarray,
    pop_idx: int,
    seed_idx: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params_batch, dtype=np.float32))
    seed_keys_j = jnp.asarray(np.asarray(seed_keys, dtype=np.uint32))

    _ = substrate.init_state(jax.random.PRNGKey(0), params_j[0])
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))

    def init_one(eval_key, params):
        rng_roll = _metric_roll_key(args, eval_key)
        k_state, k_pts, k_ch, _k_scan = jax.random.split(rng_roll, 4)
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)
        return {
            "A": s0["A"],
            "P": s0["P"],
            "F": s0["F"],
            "lagrangian_xy": pts0,
            "lagrangian_c": ch0,
        }

    snap_all = jax.vmap(
        lambda params: jax.vmap(lambda key: init_one(key, params))(seed_keys_j)
    )(params_j)
    snap = jax.tree.map(lambda x: x[int(pop_idx), int(seed_idx)], snap_all)
    return {
        "A": np.asarray(jax.device_get(snap["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(snap["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(snap["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(snap["lagrangian_xy"]), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(snap["lagrangian_c"]), dtype=np.int32),
    }


def _load_first_apf_snapshot(apf_dir: Path) -> dict[str, np.ndarray]:
    from flowlenia_minibang_common import list_apf_chunks

    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}.")
    first_path = chunks[0][0]
    with np.load(first_path) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        if steps.size == 0:
            raise ValueError(f"{first_path} has empty steps.")
        idx = int(np.argmin(np.abs(steps - 0)))
        out = {"steps": steps}
        for key in ("A", "P", "F", "lagrangian_xy", "lagrangian_c", "resume_stepper_mode"):
            if key in data.files:
                arr = np.asarray(data[key])
                out[key] = arr[idx]
        out["chunk_path"] = np.asarray(str(first_path))
        return out


def _snapshot_diff_summary(apf: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("A", "P", "F", "lagrangian_xy"):
        if key not in apf or key not in ref:
            out[f"{key}_status"] = "missing"
            continue
        a = np.asarray(apf[key], dtype=np.float32)
        b = np.asarray(ref[key], dtype=np.float32)
        if a.shape != b.shape:
            out[f"{key}_status"] = "shape_mismatch"
            out[f"{key}_apf_shape"] = list(a.shape)
            out[f"{key}_ref_shape"] = list(b.shape)
            continue
        d = a - b
        out[f"{key}_status"] = "ok"
        out[f"{key}_max_abs_diff"] = float(np.nanmax(np.abs(d)))
        out[f"{key}_mean_abs_diff"] = float(np.nanmean(np.abs(d)))
    if "lagrangian_c" in apf and "lagrangian_c" in ref:
        a = np.asarray(apf["lagrangian_c"], dtype=np.int32)
        b = np.asarray(ref["lagrangian_c"], dtype=np.int32)
        out["lagrangian_c_shape"] = list(a.shape)
        out["lagrangian_c_mismatch_frac"] = float(np.mean(a != b)) if a.shape == b.shape else None
    return out


def _xy_diff_summary(apf_xy: np.ndarray, ref_xy: np.ndarray, *, atol: float) -> dict[str, Any]:
    if np.asarray(apf_xy).shape != np.asarray(ref_xy).shape:
        return {
            "status": "shape_mismatch",
            "apf_shape": list(np.asarray(apf_xy).shape),
            "ref_shape": list(np.asarray(ref_xy).shape),
        }
    diff = np.asarray(apf_xy, dtype=np.float32) - np.asarray(ref_xy, dtype=np.float32)
    per_sample_max = np.nanmax(np.abs(diff).reshape((diff.shape[0], -1)), axis=1)
    per_sample_mean = np.nanmean(np.abs(diff).reshape((diff.shape[0], -1)), axis=1)
    first_failed = np.flatnonzero(per_sample_max > float(atol))
    return {
        "status": "ok" if not first_failed.size else "failed",
        "max_abs_xy_diff": float(np.nanmax(np.abs(diff))),
        "mean_abs_xy_diff": float(np.nanmean(np.abs(diff))),
        "per_sample_max_abs_xy_diff": [float(x) for x in per_sample_max.reshape(-1)],
        "per_sample_mean_abs_xy_diff": [float(x) for x in per_sample_mean.reshape(-1)],
        "first_failed_sample_idx": int(first_failed[0]) if first_failed.size else None,
    }


def _xy_pairwise_matrix(arrays: dict[str, np.ndarray], *, atol: float) -> dict[str, Any]:
    names = [name for name, arr in arrays.items() if arr is not None]
    pairs: dict[str, Any] = {}
    closest: dict[str, list[dict[str, Any]]] = {}
    for left_name in names:
        left_rows: list[dict[str, Any]] = []
        for right_name in names:
            if left_name == right_name:
                continue
            summary = _xy_diff_summary(arrays[left_name], arrays[right_name], atol=atol)
            key = f"{left_name}__vs__{right_name}"
            pairs[key] = summary
            if summary.get("status") != "shape_mismatch":
                left_rows.append(
                    {
                        "name": right_name,
                        "status": summary.get("status"),
                        "max_abs_xy_diff": summary.get("max_abs_xy_diff"),
                        "mean_abs_xy_diff": summary.get("mean_abs_xy_diff"),
                    }
                )
        left_rows.sort(key=lambda row: (float(row.get("mean_abs_xy_diff", float("inf"))), float(row.get("max_abs_xy_diff", float("inf")))))
        closest[left_name] = left_rows[:5]
    return {
        "array_names": names,
        "pairs": pairs,
        "closest_by_mean_abs": closest,
    }


def _array_shapes(arrays: dict[str, np.ndarray | None]) -> dict[str, list[int] | None]:
    return {
        name: (None if arr is None else [int(x) for x in np.asarray(arr).shape])
        for name, arr in arrays.items()
    }


def _seed_int_from_prng_key(key: Any) -> int | None:
    arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    if arr.size == 2 and int(arr[0]) == 0:
        return int(arr[1])
    return None


def _metric_roll_key(args: Any, eval_key: Any) -> Any:
    import jax

    if bool(_get(args, "log_clip_evolution", True)):
        rng_roll, _rng_metric, _rng_clip = jax.random.split(eval_key, 3)
    else:
        rng_roll, _rng_metric = jax.random.split(eval_key, 2)
    return rng_roll


def _config_value_diff(left: Any, right: Any, keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in keys:
        lv = _get(left, key, None)
        rv = _get(right, key, None)
        if str(lv) != str(rv):
            out.append({"key": key, "left": None if lv is None else str(lv), "right": None if rv is None else str(rv)})
    return out


def _same_float(left: Any, right: Any, *, atol: float = 1.0e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= float(atol)
    except Exception:
        return str(left) == str(right)


def _split_protocol_config_diffs(
    diffs: list[dict[str, Any]],
    *,
    rollout_flat: Any,
    opt_flat: Any,
    legacy_sigma_collision: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol_diffs: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for row in diffs:
        key = str(row.get("key", ""))
        if key == "sigma":
            rollout_flow_sigma = _get(rollout_flat, "flow_sigma", None)
            opt_flow_sigma = _get(opt_flat, "flow_sigma", None)
            opt_effective_sigma = _get(opt_flat, "sigma", None) if legacy_sigma_collision else opt_flow_sigma
            if (
                rollout_flow_sigma is not None
                and opt_effective_sigma is not None
                and _same_float(rollout_flow_sigma, opt_effective_sigma)
            ):
                ignored_row = dict(row)
                ignored_row["reason"] = (
                    "Flow-Lenia physical sigma matches effective replay sigma; "
                    "legacy optimizer checkpoints used optimization.sigma as physical sigma"
                    if legacy_sigma_collision
                    else
                    "benign Flow-Lenia namespace collision: optimization.sigma is optimizer radius; "
                    "physical substrate sigma is compared through flow_sigma"
                )
                ignored.append(ignored_row)
                continue
        if key == "flow_sigma" and legacy_sigma_collision:
            rollout_flow_sigma = _get(rollout_flat, "flow_sigma", None)
            opt_effective_sigma = _get(opt_flat, "sigma", None)
            if (
                rollout_flow_sigma is not None
                and opt_effective_sigma is not None
                and _same_float(rollout_flow_sigma, opt_effective_sigma)
            ):
                ignored_row = dict(row)
                ignored_row["reason"] = (
                    "legacy optimizer checkpoint had no flow_sigma key; replay flow_sigma "
                    "is set to the historical effective physical sigma"
                )
                ignored.append(ignored_row)
                continue
        protocol_diffs.append(row)
    return protocol_diffs, ignored


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _runtime_source_audit() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "repo_root": str(_REPO_ROOT),
    }
    try:
        import jax
        import jaxlib

        out.update(
            {
                "jax_version": getattr(jax, "__version__", "unknown"),
                "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    except Exception as exc:
        out["jax_audit_error"] = repr(exc)
    try:
        import flowlenia_minibang_simulate as minibang_sim

        out["flowlenia_minibang_simulate_file"] = str(Path(minibang_sim.__file__).resolve())
        out["flowlenia_minibang_simulate_stepper_mode"] = str(
            getattr(minibang_sim, "OPTIMIZATION_METRIC_STEPPER_MODE", "missing")
        )
    except Exception as exc:
        out["flowlenia_minibang_simulate_import_error"] = repr(exc)

    files = [
        Path(__file__).resolve(),
        _REPO_ROOT / "scripts" / "flowlenia_minibang_simulate.py",
        _REPO_ROOT / "scripts" / "main_opt_msc.py",
        _REPO_ROOT / "substrates" / "lenia_flow" / "lenia_flow.py",
        _REPO_ROOT / "substrates" / "lenia_flow" / "reintegration_tracking.py",
    ]
    source_sha256 = {}
    for path in files:
        resolved = Path(path).resolve()
        try:
            label = str(resolved.relative_to(_REPO_ROOT))
        except ValueError:
            label = str(resolved)
        source_sha256[label] = _file_sha256(resolved)
    out["source_sha256"] = source_sha256
    return out


def _first_metric_step_key(rng_roll: Any, *, rollout_steps: int, sample_every_steps: int) -> Any:
    import jax

    n_chunks = int(rollout_steps) // int(sample_every_steps)
    first_chunk_key = jax.random.split(rng_roll, n_chunks)[0]
    return jax.random.split(first_chunk_key, int(sample_every_steps))[0]


def _lagrangian_settings(args: Any, *, prefer_logging_names: bool) -> dict[str, Any]:
    if prefer_logging_names:
        return {
            "n_particles": int(_get(args, "lagrangian_n_particles", _get(args, "metric_lagrangian_n_particles", 8192))),
            "init_mode": str(_get(args, "lagrangian_init_mode", _get(args, "metric_lagrangian_init_mode", "mass"))),
            "flow_channel": int(_get(args, "lagrangian_flow_channel", _get(args, "metric_lagrangian_flow_channel", -1))),
            "flow_reduce": str(_get(args, "lagrangian_flow_reduce", _get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))),
            "channel_mode": str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample"))),
            "noise_model": str(_get(args, "lagrangian_noise_model", _get(args, "metric_lagrangian_noise_model", "rt_box"))),
            "diffusion_scale": float(
                _get(args, "lagrangian_diffusion_scale", _get(args, "metric_lagrangian_diffusion_scale", 1.0))
            ),
        }
    return {
        "n_particles": int(_get(args, "metric_lagrangian_n_particles", 8192)),
        "init_mode": str(_get(args, "metric_lagrangian_init_mode", "mass")),
        "flow_channel": int(_get(args, "metric_lagrangian_flow_channel", -1)),
        "flow_reduce": str(_get(args, "metric_lagrangian_flow_reduce", "mass_weighted")),
        "channel_mode": str(_get(args, "metric_lagrangian_channel_mode", "resample")),
        "noise_model": str(_get(args, "metric_lagrangian_noise_model", "rt_box")),
        "diffusion_scale": float(_get(args, "metric_lagrangian_diffusion_scale", 1.0)),
    }


def _one_step_diagnostic(
    *,
    opt_flat: Any,
    rollout_flat: dict[str, Any],
    params: np.ndarray,
    run_seed: int,
    rollout_steps: int,
    sample_every_steps: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    params_j = jnp.asarray(np.asarray(params, dtype=np.float32))
    opt_args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    opt_args.rollout_steps = int(rollout_steps)
    opt_args.sample_every_steps = int(sample_every_steps)
    rollout_args = OmegaConf.create(dict(rollout_flat))
    rollout_args.rollout_steps = int(rollout_steps)
    rollout_args.sample_every_steps = int(sample_every_steps)

    opt_substrate = _make_substrate(opt_args)
    rollout_substrate = _make_substrate(rollout_args)
    eval_key = jax.random.PRNGKey(int(run_seed))
    opt_rng_roll = _metric_roll_key(opt_args, eval_key)
    # This mirrors flowlenia_minibang_simulate.py. Missing log_clip_evolution keeps
    # the historical APF behavior: two-way split.
    if bool(_get(rollout_args, "log_clip_evolution", False)):
        rollout_rng_roll = jax.random.split(eval_key, 3)[0]
    else:
        rollout_rng_roll = jax.random.split(eval_key, 2)[0]

    opt_k_state, opt_k_pts, opt_k_ch, opt_k_scan = jax.random.split(opt_rng_roll, 4)
    roll_k_state, roll_k_pts, roll_k_ch, roll_k_scan = jax.random.split(rollout_rng_roll, 4)
    opt_step_key = _first_metric_step_key(
        opt_k_scan,
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
    )
    roll_step_key = _first_metric_step_key(
        roll_k_scan,
        rollout_steps=int(rollout_steps),
        sample_every_steps=int(sample_every_steps),
    )

    opt_settings = _lagrangian_settings(opt_args, prefer_logging_names=False)
    roll_settings = _lagrangian_settings(rollout_args, prefer_logging_names=True)

    def init_bundle(substrate, settings, k_state, k_pts, k_ch):
        state0 = substrate.init_state(k_state, params_j)
        rt = substrate.RT
        pts0 = _init_lagrangian_points_jax(
            state0["A"],
            n_particles=int(settings["n_particles"]),
            init_mode=str(settings["init_mode"]),
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if str(settings["channel_mode"]) in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, state0["A"], k_ch)
        else:
            ch0 = jnp.zeros((int(settings["n_particles"]),), dtype=jnp.int32)
        return state0, pts0, ch0

    def advance_one(substrate, settings, state0, pts0, ch0, key_step):
        state1 = substrate.step_state(key_step, state0, params_j)
        lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
        pts1, ch1 = substrate.RT.advect_particles(
            points=pts0,
            F=state1["F"],
            A=state1["A"],
            channel=int(settings["flow_channel"]),
            reduce=str(settings["flow_reduce"]),
            point_channels=ch0,
            channel_mode=str(settings["channel_mode"]),
            key=lag_key,
            noise_model=str(settings["noise_model"]),
            diffusion_scale=float(settings["diffusion_scale"]),
        )
        return state1, pts1, ch1

    opt_state0, opt_pts0, opt_ch0 = init_bundle(opt_substrate, opt_settings, opt_k_state, opt_k_pts, opt_k_ch)
    roll_state0, roll_pts0, roll_ch0 = init_bundle(
        rollout_substrate,
        roll_settings,
        roll_k_state,
        roll_k_pts,
        roll_k_ch,
    )
    opt_state1, opt_pts1, opt_ch1 = advance_one(
        opt_substrate,
        opt_settings,
        opt_state0,
        opt_pts0,
        opt_ch0,
        opt_step_key,
    )
    roll_state1, roll_pts1, roll_ch1 = advance_one(
        rollout_substrate,
        roll_settings,
        roll_state0,
        roll_pts0,
        roll_ch0,
        roll_step_key,
    )

    opt_initial = {
        "A": np.asarray(jax.device_get(opt_state0["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(opt_state0["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(opt_state0["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(opt_pts0), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(opt_ch0), dtype=np.int32),
    }
    roll_initial = {
        "A": np.asarray(jax.device_get(roll_state0["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(roll_state0["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(roll_state0["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(roll_pts0), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(roll_ch0), dtype=np.int32),
    }
    opt_step = {
        "A": np.asarray(jax.device_get(opt_state1["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(opt_state1["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(opt_state1["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(opt_pts1), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(opt_ch1), dtype=np.int32),
    }
    roll_step = {
        "A": np.asarray(jax.device_get(roll_state1["A"]), dtype=np.float32),
        "P": np.asarray(jax.device_get(roll_state1["P"]), dtype=np.float32),
        "F": np.asarray(jax.device_get(roll_state1["F"]), dtype=np.float32),
        "lagrangian_xy": np.asarray(jax.device_get(roll_pts1), dtype=np.float32),
        "lagrangian_c": np.asarray(jax.device_get(roll_ch1), dtype=np.int32),
    }
    return {
        "opt_log_clip_evolution": bool(_get(opt_args, "log_clip_evolution", True)),
        "apf_log_clip_evolution": bool(_get(rollout_args, "log_clip_evolution", False)),
        "opt_lagrangian_settings": opt_settings,
        "apf_lagrangian_settings": roll_settings,
        "initial_diff": _snapshot_diff_summary(roll_initial, opt_initial),
        "after_one_step_diff": _snapshot_diff_summary(roll_step, opt_step),
    }


def _apf_style_first_chunk_variants(
    *,
    rollout_flat: dict[str, Any],
    params: np.ndarray,
    run_seed: int,
    rollout_steps: int,
    sample_every_steps: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(dict(rollout_flat))
    args.rollout_steps = int(rollout_steps)
    args.max_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    args.snapshot_interval = int(sample_every_steps)
    substrate = _make_substrate(args)
    params_batch = jnp.asarray(np.asarray(params, dtype=np.float32).reshape((1, -1)))

    eval_key = jax.random.PRNGKey(int(run_seed))
    if bool(_get(args, "log_clip_evolution", False)):
        rng_roll = jax.random.split(eval_key, 3)[0]
    else:
        rng_roll = jax.random.split(eval_key, 2)[0]
    k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
    init_keys = jnp.stack([k_state], axis=0)
    lag_keys = jnp.stack([k_pts], axis=0)
    ch_keys = jnp.stack([k_ch], axis=0)
    n_scan_chunks = int(rollout_steps) // int(sample_every_steps)
    scan_chunk_keys = jnp.stack([jax.random.split(k_scan, n_scan_chunks)], axis=0)

    _ = substrate.init_state(jax.random.PRNGKey(0), params_batch[0])
    rt = substrate.RT
    states0 = jax.jit(lambda keys, p: jax.vmap(substrate.init_state)(keys, p))(init_keys, params_batch)

    lag_settings = _lagrangian_settings(args, prefer_logging_names=True)

    def init_lag_one(A0, key_pts, key_ch):
        pts = _init_lagrangian_points_jax(
            A0,
            n_particles=int(lag_settings["n_particles"]),
            init_mode=str(lag_settings["init_mode"]),
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=key_pts,
        )
        if str(lag_settings["channel_mode"]) in ("fixed", "resample"):
            ch = rt.sample_point_channels(pts, A0, key_ch)
        else:
            ch = jnp.zeros((int(lag_settings["n_particles"]),), dtype=jnp.int32)
        return pts, ch

    lag_xy0, lag_ch0 = jax.jit(lambda A0, kp, kc: jax.vmap(init_lag_one)(A0, kp, kc))(
        states0["A"],
        lag_keys,
        ch_keys,
    )

    def capture_like_apf(states_in, params_in, lag_xy_in, lag_ch_in):
        img_size = int(_get(args, "img_size", _get(args, "video_img_size", 224)))
        rgb = jax.vmap(lambda st, p: substrate.render_state(st, p, img_size=img_size))(states_in, params_in)
        return (
            states_in["P"],
            states_in["A"],
            states_in["F"],
            rgb,
            lag_xy_in,
            lag_ch_in,
            states_in["t"],
            states_in["mass_cycle_start"],
        )

    capture_jit = jax.jit(capture_like_apf)

    def advance(states_in, lag_xy_in, lag_ch_in, params_in, scan_chunk_keys_in):
        chunk_keys = scan_chunk_keys_in[:, 0]

        def advance_one(key_chunk, st_i, pts_i, ch_i, params_i):
            rngs = jax.random.split(key_chunk, int(sample_every_steps))

            def scan_body(carry, key_i):
                st, pts, ch = carry
                st_next = substrate.step_state(key_i, st, params_i)
                lag_key = jax.random.fold_in(key_i, jnp.uint32(0x4C4147))
                pts_next, ch_next = rt.advect_particles(
                    points=pts,
                    F=st_next["F"],
                    A=st_next["A"],
                    channel=int(lag_settings["flow_channel"]),
                    reduce=str(lag_settings["flow_reduce"]),
                    point_channels=ch,
                    channel_mode=str(lag_settings["channel_mode"]),
                    key=lag_key,
                    noise_model=str(lag_settings["noise_model"]),
                    diffusion_scale=float(lag_settings["diffusion_scale"]),
                )
                return (st_next, pts_next, ch_next), None

            (st_out_i, pts_out_i, ch_out_i), _ = jax.lax.scan(scan_body, (st_i, pts_i, ch_i), rngs)
            return st_out_i, pts_out_i, ch_out_i

        return jax.vmap(advance_one)(chunk_keys, states_in, lag_xy_in, lag_ch_in, params_in)

    advance_jit = jax.jit(advance)
    states1, lag_xy1, lag_ch1 = advance_jit(states0, lag_xy0, lag_ch0, params_batch, scan_chunk_keys)
    _ = jax.device_get(capture_jit(states0, params_batch, lag_xy0, lag_ch0))
    states1_after_capture, lag_xy1_after_capture, lag_ch1_after_capture = advance_jit(
        states0,
        lag_xy0,
        lag_ch0,
        params_batch,
        scan_chunk_keys,
    )

    return {
        "no_capture_xy": np.asarray(jax.device_get(lag_xy1[0]), dtype=np.float32)[None, :, :],
        "with_capture_xy": np.asarray(jax.device_get(lag_xy1_after_capture[0]), dtype=np.float32)[None, :, :],
        "no_capture_A": np.asarray(jax.device_get(states1["A"][0]), dtype=np.float32),
        "with_capture_A": np.asarray(jax.device_get(states1_after_capture["A"][0]), dtype=np.float32),
        "no_capture_ch": np.asarray(jax.device_get(lag_ch1[0]), dtype=np.int32),
        "with_capture_ch": np.asarray(jax.device_get(lag_ch1_after_capture[0]), dtype=np.int32),
    }


def _trace_diff_summary(left: np.ndarray, right: np.ndarray, *, atol: float) -> dict[str, Any]:
    if np.asarray(left).shape != np.asarray(right).shape:
        return {
            "status": "shape_mismatch",
            "left_shape": list(np.asarray(left).shape),
            "right_shape": list(np.asarray(right).shape),
        }
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    flat = np.abs(diff).reshape((diff.shape[0], -1))
    per_step_max = np.nanmax(flat, axis=1)
    per_step_mean = np.nanmean(flat, axis=1)
    failed = np.flatnonzero(per_step_max > float(atol))
    probe_idx = sorted({0, min(1, diff.shape[0] - 1), min(4, diff.shape[0] - 1), min(9, diff.shape[0] - 1), min(24, diff.shape[0] - 1), diff.shape[0] - 1})
    return {
        "status": "ok" if not failed.size else "failed",
        "n_internal_steps": int(diff.shape[0]),
        "first_failed_internal_step": int(failed[0] + 1) if failed.size else None,
        "final_max_abs_xy_diff": float(per_step_max[-1]),
        "final_mean_abs_xy_diff": float(per_step_mean[-1]),
        "max_over_trace": float(np.nanmax(per_step_max)),
        "mean_over_trace": float(np.nanmean(per_step_mean)),
        "probe_steps": [
            {
                "internal_step": int(i + 1),
                "max_abs_xy_diff": float(per_step_max[i]),
                "mean_abs_xy_diff": float(per_step_mean[i]),
            }
            for i in probe_idx
        ],
    }


def _first_chunk_trace_diagnostic(
    *,
    rollout_flat: dict[str, Any],
    params: np.ndarray,
    run_seed: int,
    sample_every_steps: int,
    atol: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(dict(rollout_flat))
    args.rollout_steps = int(sample_every_steps)
    args.max_steps = int(sample_every_steps)
    args.sample_every_steps = int(sample_every_steps)
    args.snapshot_interval = int(sample_every_steps)
    params_1 = jnp.asarray(np.asarray(params, dtype=np.float32))
    params_b = params_1[None, :]
    substrate = _make_substrate(args)
    _ = substrate.init_state(jax.random.PRNGKey(0), params_1)
    rt = substrate.RT
    settings = _lagrangian_settings(args, prefer_logging_names=True)

    eval_key = jax.random.PRNGKey(int(run_seed))
    if bool(_get(args, "log_clip_evolution", False)):
        rng_roll = jax.random.split(eval_key, 3)[0]
    else:
        rng_roll = jax.random.split(eval_key, 2)[0]
    k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
    chunk_key = jax.random.split(k_scan, 1)[0]
    step_keys = jax.random.split(chunk_key, int(sample_every_steps))

    def init_one(params_i):
        state0 = substrate.init_state(k_state, params_i)
        pts0 = _init_lagrangian_points_jax(
            state0["A"],
            n_particles=int(settings["n_particles"]),
            init_mode=str(settings["init_mode"]),
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if str(settings["channel_mode"]) in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, state0["A"], k_ch)
        else:
            ch0 = jnp.zeros((int(settings["n_particles"]),), dtype=jnp.int32)
        return state0, pts0, ch0

    def step_one(params_i, carry, key_i):
        st, pts, ch = carry
        st_next = substrate.step_state(key_i, st, params_i)
        lag_key = jax.random.fold_in(key_i, jnp.uint32(0x4C4147))
        pts_next, ch_next = rt.advect_particles(
            points=pts,
            F=st_next["F"],
            A=st_next["A"],
            channel=int(settings["flow_channel"]),
            reduce=str(settings["flow_reduce"]),
            point_channels=ch,
            channel_mode=str(settings["channel_mode"]),
            key=lag_key,
            noise_model=str(settings["noise_model"]),
            diffusion_scale=float(settings["diffusion_scale"]),
        )
        return (st_next, pts_next, ch_next), pts_next

    def whole_trace(params_i):
        state0, pts0, ch0 = init_one(params_i)
        (_, _, _), pts_seq = jax.lax.scan(lambda carry, key_i: step_one(params_i, carry, key_i), (state0, pts0, ch0), step_keys)
        return pts_seq

    whole_jit = np.asarray(jax.device_get(jax.jit(whole_trace)(params_1)), dtype=np.float32)

    init_jit = jax.jit(lambda p: init_one(p))
    state0, pts0, ch0 = init_jit(params_1)

    def capture_like_apf(st, pts, ch, params_i):
        img_size = int(_get(args, "img_size", _get(args, "video_img_size", 224)))
        rgb = substrate.render_state(st, params_i, img_size=img_size)
        return st["P"], st["A"], st["F"], rgb, pts, ch, st["t"], st["mass_cycle_start"]

    def separate_trace(st, pts, ch, params_i):
        (_, _, _), pts_seq = jax.lax.scan(lambda carry, key_i: step_one(params_i, carry, key_i), (st, pts, ch), step_keys)
        return pts_seq

    separate_jit_fn = jax.jit(separate_trace)
    separate_jit = np.asarray(jax.device_get(separate_jit_fn(state0, pts0, ch0, params_1)), dtype=np.float32)
    _ = jax.device_get(jax.jit(capture_like_apf)(state0, pts0, ch0, params_1))
    separate_after_capture_jit = np.asarray(jax.device_get(separate_jit_fn(state0, pts0, ch0, params_1)), dtype=np.float32)

    # Legacy APF shape: scan over time outside, vmap over batch inside. For B=1
    # it should be close, but keeping it here guards against accidental path use.
    state_b, pts_b, ch_b = jax.jit(lambda p: jax.vmap(init_one)(p))(params_b)

    def legacy_scan_vmap_trace(states_in, pts_in, ch_in, params_in):
        def scan_body(carry, key_i):
            st_b, pts_bi, ch_bi = carry

            def one(params_i, st_i, pts_i, ch_i):
                (st_next, pts_next, ch_next), out = step_one(params_i, (st_i, pts_i, ch_i), key_i)
                return st_next, pts_next, ch_next, out

            st_next, pts_next, ch_next, out = jax.vmap(one)(params_in, st_b, pts_bi, ch_bi)
            return (st_next, pts_next, ch_next), out[0]

        (_, _, _), pts_seq = jax.lax.scan(scan_body, (states_in, pts_in, ch_in), step_keys)
        return pts_seq

    legacy_jit = np.asarray(jax.device_get(jax.jit(legacy_scan_vmap_trace)(state_b, pts_b, ch_b, params_b)), dtype=np.float32)

    traces = {
        "whole_jit": whole_jit,
        "separate_jit": separate_jit,
        "separate_after_capture_jit": separate_after_capture_jit,
        "legacy_scan_vmap_jit": legacy_jit,
    }
    pairs = {}
    names = list(traces)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairs[f"{left}__vs__{right}"] = _trace_diff_summary(traces[left], traces[right], atol=atol)
    finals = {name: traces[name][-1][None, :, :] for name in names}
    return {
        "settings": settings,
        "trace_names": names,
        "pairwise_trace": pairs,
        "final_pairwise": _xy_pairwise_matrix(finals, atol=atol),
    }


def _resolve_section(cfg: Any) -> Any:
    section = _get(_get(cfg, "simulation", {}), "flow_lenia_arun_lagrangian_apf", None)
    if section is None:
        raise ValueError("Missing simulation.flow_lenia_arun_lagrangian_apf in generated config.")
    return section


def _audit_generated_config(config_path: Path) -> dict[str, Any]:
    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    c1 = _get(_get(_get(cfg, "datasets", {}), "flow_lenia", {}), "c1", {})
    rollout_overrides = _get(section, "rollout_overrides", {})
    rollout_logging = _get(rollout_overrides, "logging", {})
    rollout_substrate = _get(rollout_overrides, "substrate", {})
    run_seed_protocol = str(_get(section, "run_seed_protocol", ""))
    metric_seed_protocol = str(_get(c1, "metric_seed_protocol", ""))
    optimized_replay_source = str(_get(c1, "optimized_replay_source", "apf"))
    optimized_replay_legacy_sigma_collision = bool(
        _get(c1, "optimized_replay_legacy_sigma_collision", False)
    )
    legacy_mapping_raw = _get(c1, "optimizer_native_legacy_sigma_collision_by_source_run_idx", None)
    legacy_mapping = (
        OmegaConf.to_container(legacy_mapping_raw, resolve=True)
        if OmegaConf.is_config(legacy_mapping_raw)
        else (legacy_mapping_raw or {})
    )
    metric_log_clip = _get(c1, "metric_log_clip_evolution", None)
    rollout_log_clip = _get(rollout_logging, "log_clip_evolution", None)
    rollout_sigma = _get(rollout_substrate, "sigma", None)
    rollout_flow_sigma = _get(rollout_substrate, "flow_sigma", None)
    out = {
        "config": str(config_path),
        "run_seed_protocol": run_seed_protocol,
        "metric_seed_protocol": metric_seed_protocol,
        "optimized_replay_source": optimized_replay_source,
        "optimized_replay_legacy_sigma_collision": bool(optimized_replay_legacy_sigma_collision),
        "optimizer_native_legacy_sigma_collision_by_source_run_idx": legacy_mapping,
        "metric_log_clip_evolution": None if metric_log_clip is None else bool(metric_log_clip),
        "rollout_log_clip_evolution": None if rollout_log_clip is None else bool(rollout_log_clip),
        "rollout_sigma": None if rollout_sigma is None else float(rollout_sigma),
        "rollout_flow_sigma": None if rollout_flow_sigma is None else float(rollout_flow_sigma),
        "apf_root": str(_get(section, "output_root", "")),
        "rollout_config": str(_get(section, "rollout_config", "")),
        "run_seed_base": int(_get(section, "run_seed_base", -1)),
        "run_seed_mode": str(_get(section, "run_seed_mode", "")),
        "run_seed_rep_stride": int(_get(section, "run_seed_rep_stride", -1)),
        "n_rollout_seeds_per_checkpoint": int(_get(section, "n_rollout_seeds_per_checkpoint", -1)),
    }
    errors = []
    if run_seed_protocol != "optimization_metric":
        errors.append(f"run_seed_protocol={run_seed_protocol!r}, expected 'optimization_metric'")
    if metric_seed_protocol != "optimization_metric":
        errors.append(f"metric_seed_protocol={metric_seed_protocol!r}, expected 'optimization_metric'")
    if optimized_replay_source not in {"optimizer_native", "apf"}:
        errors.append(
            f"optimized_replay_source={optimized_replay_source!r}, expected 'optimizer_native' or 'apf'"
        )
    if optimized_replay_source == "optimizer_native" and not optimized_replay_legacy_sigma_collision and not legacy_mapping:
        errors.append(
            "optimized_replay_legacy_sigma_collision is false; this archived fixed-init OpenAI-ES "
            "campaign requires legacy sigma collision replay to match stored optimization scores"
        )
    if metric_log_clip is None:
        errors.append("missing datasets.flow_lenia.c1.metric_log_clip_evolution")
    if rollout_log_clip is None:
        errors.append("missing simulation.flow_lenia_arun_lagrangian_apf.rollout_overrides.logging.log_clip_evolution")
    if metric_log_clip is not None and rollout_log_clip is not None and bool(metric_log_clip) != bool(rollout_log_clip):
        errors.append(
            "metric_log_clip_evolution and rollout log_clip_evolution disagree: "
            f"{bool(metric_log_clip)!r} vs {bool(rollout_log_clip)!r}"
        )
    if rollout_flow_sigma is None:
        errors.append("missing simulation.flow_lenia_arun_lagrangian_apf.rollout_overrides.substrate.flow_sigma")
    if rollout_sigma is not None and rollout_flow_sigma is not None and float(rollout_sigma) != float(rollout_flow_sigma):
        errors.append(
            "rollout substrate sigma and flow_sigma disagree: "
            f"{float(rollout_sigma)!r} vs {float(rollout_flow_sigma)!r}"
        )
    if out["run_seed_mode"] != "source_run_idx":
        errors.append(f"run_seed_mode={out['run_seed_mode']!r}, expected 'source_run_idx'")
    if out["run_seed_rep_stride"] != 1:
        errors.append(f"run_seed_rep_stride={out['run_seed_rep_stride']!r}, expected 1")
    out["errors"] = errors
    return out


def _mapping_get_bool(raw: Any, key: int, default: bool) -> bool:
    if raw in (None, ""):
        return bool(default)
    values = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    if not isinstance(values, dict):
        return bool(default)
    keys = (key, str(key), f"run_{int(key):03d}", f"{int(key):03d}")
    for candidate in keys:
        if candidate in values:
            return bool(values[candidate])
    return bool(default)


def _legacy_sigma_collision_for_run(config_audit: dict[str, Any], run_idx: int) -> bool:
    return _mapping_get_bool(
        config_audit.get("optimizer_native_legacy_sigma_collision_by_source_run_idx", None),
        int(run_idx),
        bool(config_audit.get("optimized_replay_legacy_sigma_collision", False)),
    )


def _apply_section_rollout_overrides_for_preflight(rollout_cfg: Any, rollout_flat: Any, section: Any) -> tuple[Any, Any]:
    cfg_out = OmegaConf.create(OmegaConf.to_container(rollout_cfg, resolve=False))
    flat_out = OmegaConf.create(OmegaConf.to_container(rollout_flat, resolve=False))
    overrides = _get(section, "rollout_overrides", None)
    if overrides is not None:
        allowed = {"meta", "substrate", "simulation", "logging", "metric", "minibang"}
        for section_name, values in overrides.items():
            name = str(section_name)
            if name not in allowed:
                raise ValueError(f"Unknown rollout_overrides section {name!r}; expected one of {sorted(allowed)}.")
            if cfg_out.get(name, None) is None:
                cfg_out[name] = OmegaConf.create()
            cfg_out[name] = OmegaConf.merge(cfg_out.get(name, {}), values)
            if values is not None:
                flat_out = OmegaConf.merge(flat_out, values)
    run_seed_protocol = _get(section, "run_seed_protocol", None)
    if run_seed_protocol is not None:
        if cfg_out.get("minibang", None) is None:
            cfg_out.minibang = OmegaConf.create()
        cfg_out.minibang.run_seed_protocol = run_seed_protocol
        flat_out.run_seed_protocol = run_seed_protocol
    return cfg_out, flat_out


def _optimization_eval_seed_count(run_dir: Path) -> int | None:
    cfg_path = Path(run_dir) / "optimization_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    opt = cfg.get("optimization", {})
    algorithm = str(opt.get("optimizer_algorithm", opt.get("optimization_algorithm", "cma_es"))).strip().lower()
    if algorithm in {
        "mirrored_openai_es",
        "mirrored_batch_openai_es",
        "openai_es",
        "batch_openai_es",
        "mirrored_es",
        "antithetic_openai_es",
    }:
        raw = opt.get("openai_es_n_seeds", None)
    else:
        raw = opt.get("bs", None)
    return None if raw is None else int(raw)


def _configured_selected_run_dirs(config_path: Path, selected_root: Path) -> list[Path]:
    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    raw_dirs = _get(section, "optimized_checkpoint_dirs", None)
    if raw_dirs not in (None, ""):
        values = OmegaConf.to_container(raw_dirs, resolve=True) if OmegaConf.is_config(raw_dirs) else raw_dirs
        dirs = [_as_repo_path(value) for value in values]
    else:
        dirs = sorted(Path(selected_root).glob("run_*"))
    return sorted([Path(path) for path in dirs if Path(path).is_dir()])


def _audit_seed_count(config_audit: dict[str, Any], run_dirs: list[Path]) -> dict[str, Any]:
    requested = int(config_audit.get("n_rollout_seeds_per_checkpoint", -1))
    rows = []
    errors = []
    for run_dir in sorted(run_dirs):
        expected = _optimization_eval_seed_count(run_dir)
        row = {"run": run_dir.name, "optimization_eval_seed_count": expected, "c1_rollout_seed_count": requested}
        rows.append(row)
        if expected is None:
            errors.append(f"{run_dir}: missing optimization eval seed count")
        elif int(expected) != requested:
            errors.append(f"{run_dir.name}: optimization eval seed count={expected}, C1 rollout seed count={requested}")
    if not rows:
        errors.append("no selected run directories found in generated config")
    return {"status": "ok" if not errors else "failed", "rows": rows, "errors": errors}


def _as_repo_path(path_raw: Any) -> Path:
    path = Path(str(path_raw))
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_best_params(run_dir: Path) -> np.ndarray:
    best_path = run_dir / "best.pkl"
    best = _load_pickle(best_path)
    params = best[0] if isinstance(best, (tuple, list)) else best
    return np.asarray(params, dtype=np.float32).reshape(-1)


def _audit_selected_candidates(run_dirs: list[Path], source_optimization_root: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for run_dir in sorted(run_dirs):
        row: dict[str, Any] = {"run": run_dir.name, "run_dir": str(run_dir)}
        rows.append(row)
        for name in ("best.pkl", "params.npy", "optimization_config.yaml", "selected_candidate.json"):
            exists = bool((run_dir / name).exists())
            row[f"has_{name.replace('.', '_')}"] = exists
            if not exists:
                errors.append(f"{run_dir}: missing {name}")
        if not (run_dir / "best.pkl").exists() or not (run_dir / "selected_candidate.json").exists():
            continue

        try:
            params = _load_best_params(run_dir)
            meta = json.loads((run_dir / "selected_candidate.json").read_text())
            source_pop = _as_repo_path(meta.get("source_pop_traj", ""))
            if not source_pop.exists() and source_optimization_root is not None:
                try:
                    run_idx = int(run_dir.name.split("_", 1)[1])
                    fallback_pop = source_optimization_root / f"run_{run_idx:03d}" / "pop_traj.pkl"
                    if fallback_pop.exists():
                        source_pop = fallback_pop
                        row["source_pop_traj_fallback_used"] = True
                except Exception:
                    pass
            row["source_pop_traj"] = str(source_pop)
            row["source_pop_traj_exists"] = bool(source_pop.exists())
            if not source_pop.exists():
                errors.append(f"{run_dir}: source_pop_traj does not exist: {source_pop}")
                continue
            pop = _load_pickle(source_pop)
            pop_params = np.asarray(pop.get("params"), dtype=np.float32)
            i_iter = int(meta.get("iter", -1))
            pop_idx = int(meta.get("pop_idx", -1))
            row["selected_iter"] = i_iter
            row["selected_pop_idx"] = pop_idx
            row["pop_params_shape"] = list(pop_params.shape)
            if pop_params.ndim != 3:
                errors.append(f"{run_dir}: pop_traj['params'] has shape {pop_params.shape}, expected 3D")
                continue
            if i_iter < 0 or i_iter >= pop_params.shape[0] or pop_idx < 0 or pop_idx >= pop_params.shape[1]:
                errors.append(f"{run_dir}: selected iter/pop_idx outside pop params shape {pop_params.shape}")
                continue
            expected = np.asarray(pop_params[i_iter, pop_idx], dtype=np.float32).reshape(-1)
            diff = np.asarray(params, dtype=np.float32).reshape(-1) - expected
            row["params_vs_selected_pop_max_abs_diff"] = float(np.nanmax(np.abs(diff)))
            if float(row["params_vs_selected_pop_max_abs_diff"]) != 0.0:
                errors.append(
                    f"{run_dir}: best.pkl params differ from source pop selected candidate "
                    f"max_abs_diff={row['params_vs_selected_pop_max_abs_diff']}"
                )
            for key in ("score_by_seed", "seed_keys"):
                if key in pop:
                    arr = np.asarray(pop[key])
                    row[f"pop_{key}_shape"] = list(arr.shape)
        except Exception as exc:
            errors.append(f"{run_dir}: selected candidate audit failed: {exc!r}")
    if not rows:
        errors.append("no selected run directories found in generated config")
    return {"status": "ok" if not errors else "failed", "rows": rows, "errors": errors}


def _audit_optimization_protocol(config_audit: dict[str, Any], run_dirs: list[Path]) -> dict[str, Any]:
    requested_seeds = int(config_audit.get("n_rollout_seeds_per_checkpoint", -1))
    expected_seed_base = int(config_audit.get("run_seed_base", -1))
    run_seed_mode = str(config_audit.get("run_seed_mode", "")).strip().lower()
    expected_log_clip = config_audit.get("rollout_log_clip_evolution", None)
    expected_flow_sigma = config_audit.get("rollout_flow_sigma", None)
    has_legacy_mapping = bool(config_audit.get("optimizer_native_legacy_sigma_collision_by_source_run_idx", None))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for suite_idx, run_dir in enumerate(sorted(run_dirs)):
        try:
            run_idx = int(run_dir.name.split("_", 1)[1])
        except Exception:
            run_idx = -1
        if run_seed_mode == "source_run_idx" and run_idx >= 0:
            expected_fixed_eval_seed_base = int(expected_seed_base + 2 * run_idx)
        elif run_seed_mode == "suite_index":
            expected_fixed_eval_seed_base = int(expected_seed_base + 2 * suite_idx)
        else:
            expected_fixed_eval_seed_base = int(expected_seed_base)
        legacy_sigma_collision = _legacy_sigma_collision_for_run(config_audit, run_idx)
        cfg_path = run_dir / "optimization_config.yaml"
        row: dict[str, Any] = {"run": run_dir.name, "optimization_config": str(cfg_path)}
        rows.append(row)
        if not cfg_path.exists():
            errors.append(f"{run_dir}: missing optimization_config.yaml")
            continue
        try:
            cfg = OmegaConf.load(cfg_path)
            opt = cfg.get("optimization", {})
            substrate = cfg.get("substrate", {})
            flat = _flat_optimization_config(cfg_path, legacy_sigma_collision=legacy_sigma_collision)
            algorithm = str(opt.get("optimizer_algorithm", opt.get("optimization_algorithm", ""))).strip().lower()
            eval_seed_mode = str(opt.get("eval_seed_mode", "")).strip().lower()
            openai_es_n_seeds = opt.get("openai_es_n_seeds", None)
            fixed_eval_seed_base = opt.get("fixed_eval_seed_base", None)
            log_clip = bool(_get(flat, "log_clip_evolution", True))
            substrate_sigma = substrate.get("sigma", None)
            flow_sigma = _get(flat, "flow_sigma", None)
            effective_flow_sigma = _get(flat, "sigma", None) if legacy_sigma_collision else flow_sigma
            row.update(
                {
                    "optimizer_algorithm": algorithm,
                    "eval_seed_mode": eval_seed_mode,
                    "openai_es_n_seeds": None if openai_es_n_seeds is None else int(openai_es_n_seeds),
                    "fixed_eval_seed_base": None if fixed_eval_seed_base is None else int(fixed_eval_seed_base),
                    "expected_fixed_eval_seed_base": int(expected_fixed_eval_seed_base),
                    "log_clip_evolution": log_clip,
                    "substrate_sigma": None if substrate_sigma is None else float(substrate_sigma),
                    "flat_flow_sigma": None if flow_sigma is None else float(flow_sigma),
                    "effective_flow_sigma": None if effective_flow_sigma is None else float(effective_flow_sigma),
                    "legacy_sigma_collision": bool(legacy_sigma_collision),
                }
            )
            if algorithm not in {
                "mirrored_openai_es",
                "mirrored_batch_openai_es",
                "openai_es",
                "batch_openai_es",
                "mirrored_es",
                "antithetic_openai_es",
            }:
                errors.append(f"{run_dir.name}: optimizer_algorithm={algorithm!r}, expected OpenAI-ES")
            if eval_seed_mode != "fixed":
                errors.append(f"{run_dir.name}: eval_seed_mode={eval_seed_mode!r}, expected 'fixed'")
            if openai_es_n_seeds is None or int(openai_es_n_seeds) != requested_seeds:
                errors.append(
                    f"{run_dir.name}: openai_es_n_seeds={openai_es_n_seeds}, "
                    f"C1 rollout seeds={requested_seeds}"
                )
            if fixed_eval_seed_base is None or int(fixed_eval_seed_base) != expected_fixed_eval_seed_base:
                errors.append(
                    f"{run_dir.name}: fixed_eval_seed_base={fixed_eval_seed_base}, "
                    f"expected={expected_fixed_eval_seed_base} "
                    f"(run_seed_base={expected_seed_base}, run_seed_mode={run_seed_mode})"
                )
            if expected_log_clip is None:
                errors.append(f"{run_dir.name}: generated config has no rollout_log_clip_evolution")
            elif bool(log_clip) != bool(expected_log_clip):
                errors.append(
                    f"{run_dir.name}: optimization log_clip_evolution={log_clip}, "
                    f"generated rollout_log_clip_evolution={expected_log_clip}"
                )
            if has_legacy_mapping:
                row["rollout_flow_sigma_global_not_used_for_optimizer_native_mixed_protocol"] = (
                    None if expected_flow_sigma is None else float(expected_flow_sigma)
                )
            elif expected_flow_sigma is None:
                errors.append(f"{run_dir.name}: generated config has no rollout_flow_sigma")
            elif effective_flow_sigma is None or float(effective_flow_sigma) != float(expected_flow_sigma):
                errors.append(
                    f"{run_dir.name}: optimization effective_flow_sigma={effective_flow_sigma}, "
                    f"generated rollout_flow_sigma={expected_flow_sigma}"
                )
        except Exception as exc:
            errors.append(f"{run_dir}: optimization protocol audit failed: {exc!r}")
    if not rows:
        errors.append("no selected run directories found in generated config")
    return {"status": "ok" if not errors else "failed", "rows": rows, "errors": errors}


def _audit_random_checkpoints(config_path: Path, run_dirs: list[Path]) -> dict[str, Any]:
    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    mode = str(_get(section, "random_checkpoint_selection", "per_source_group")).strip().lower()
    n_random = int(_get(section, "num_random_baselines", 0))
    random_root_raw = _get(section, "random_checkpoint_root", "")
    random_root = _as_repo_path(random_root_raw)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if n_random <= 0:
        return {"status": "ok", "mode": mode, "n_random": n_random, "rows": rows, "errors": []}
    per_source_modes = {
        "per_source_group",
        "per_source_group_optimizer_context",
        "per_source_group_optimizer_init",
        "random_params_optimizer_context",
        "random_params_optimizer_init",
    }
    if mode not in (per_source_modes | {"all_groups_flat", "global_flat", "flat", "optimization_iter0"}):
        errors.append(f"unknown random_checkpoint_selection={mode!r}")

    if mode == "optimization_iter0":
        raw_indices = _get(section, "random_optimizer_native_pop_indices", None)
        if raw_indices not in (None, ""):
            values = OmegaConf.to_container(raw_indices, resolve=True) if OmegaConf.is_config(raw_indices) else raw_indices
            pop_indices = [int(x) for x in values]
        else:
            pop_indices = list(range(n_random))
        if len(pop_indices) < n_random:
            errors.append(f"need {n_random} optimizer iter0 pop indices, got {pop_indices}")
        opt_iter = int(_get(section, "random_optimizer_native_iter", 0))
        for run_dir in sorted(run_dirs):
            selected_path = run_dir / "selected_candidate.json"
            row = {
                "run": run_dir.name,
                "selected_candidate": str(selected_path),
                "optimizer_iter": int(opt_iter),
                "pop_indices": [int(x) for x in pop_indices[:n_random]],
                "ok": False,
            }
            rows.append(row)
            if not selected_path.exists():
                errors.append(f"missing selected_candidate.json for optimization_iter0 random controls: {selected_path}")
                continue
            try:
                selected = json.loads(selected_path.read_text())
                pop_path = Path(str(selected.get("source_pop_traj", "")))
                if not pop_path.is_absolute():
                    pop_path = _REPO_ROOT / pop_path
                row["source_pop_traj"] = str(pop_path)
                if not pop_path.exists():
                    errors.append(f"{run_dir.name}: source_pop_traj not found: {pop_path}")
                    continue
                with pop_path.open("rb") as f:
                    pop = pickle.load(f)
                params = np.asarray(pop.get("params"), dtype=np.float32)
                tau_steps = np.asarray(pop.get("tau_steps")) if "tau_steps" in pop else None
                row["params_shape"] = list(params.shape)
                if params.ndim != 3:
                    errors.append(f"{run_dir.name}: invalid pop_traj params shape {params.shape}")
                    continue
                if opt_iter < 0 or opt_iter >= params.shape[0]:
                    errors.append(f"{run_dir.name}: random_optimizer_native_iter={opt_iter} out of range {params.shape}")
                bad = [int(i) for i in pop_indices[:n_random] if int(i) < 0 or int(i) >= params.shape[1]]
                if bad:
                    errors.append(f"{run_dir.name}: random pop indices out of range for pop_size={params.shape[1]}: {bad}")
                if tau_steps is None:
                    errors.append(f"{run_dir.name}: pop_traj has no tau_steps for optimization_iter0 random controls")
                else:
                    row["tau_steps"] = [int(tau_steps[opt_iter, int(i)]) for i in pop_indices[:n_random] if 0 <= int(i) < params.shape[1]]
                row["ok"] = not bad and tau_steps is not None
            except Exception as exc:
                errors.append(f"{run_dir.name}: optimization_iter0 random audit failed: {exc!r}")
    else:
        if not random_root.exists():
            errors.append(f"random checkpoint root does not exist: {random_root}")

    if mode in per_source_modes:
        for run_dir in sorted(run_dirs):
            try:
                run_idx = int(run_dir.name.split("_", 1)[1])
            except Exception:
                errors.append(f"cannot parse run index from {run_dir.name}")
                continue
            meta_path = run_dir / "selected_candidate.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    source_run_dir = Path(str(meta.get("source_run_dir", "")))
                    if source_run_dir.name.startswith("run_"):
                        run_idx = int(source_run_dir.name.split("_", 1)[1])
                except Exception:
                    pass
            for random_idx in range(n_random):
                best_path = random_root / f"group_{run_idx:03d}" / f"random_{random_idx:03d}" / "best.pkl"
                row = {
                    "run": run_dir.name,
                    "source_run_idx": int(run_idx),
                    "random_idx": int(random_idx),
                    "best_path": str(best_path),
                    "exists": bool(best_path.exists()),
                }
                rows.append(row)
                if not best_path.exists():
                    errors.append(f"missing random checkpoint: {best_path}")
    elif mode in {"all_groups_flat", "global_flat", "flat"}:
        found = sorted(random_root.glob("group_*/random_*/best.pkl"))
        rows.append({"mode": mode, "required": int(n_random), "found": int(len(found)), "random_root": str(random_root)})
        if len(found) < n_random:
            errors.append(f"need {n_random} random checkpoints under {random_root}, found {len(found)}")
    return {
        "status": "ok" if not errors else "failed",
        "mode": mode,
        "n_random": n_random,
        "random_root": str(random_root),
        "rows": rows,
        "errors": errors,
    }


def _find_run(selected_root: Path, requested: str | None) -> tuple[int, Path]:
    if requested is not None:
        text = str(requested)
        run_idx = int(text[4:] if text.startswith("run_") else text)
        run_dir = selected_root / f"run_{run_idx:03d}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Selected run not found: {run_dir}")
        return run_idx, run_dir
    runs = sorted(p for p in selected_root.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No selected run directories found under {selected_root}")
    run_dir = runs[0]
    return int(run_dir.name.split("_", 1)[1]), run_dir


def _run_replay_smoke(
    *,
    config_path: Path,
    selected_root: Path,
    run: str | None,
    seed_idx: int,
    rollout_steps: int,
    output_root: Path,
    atol: float,
) -> dict[str, Any]:
    import flowlenia_minibang_simulate as minibang_sim
    from flowlenia_minibang_common import load_config as load_rollout_config
    from flowlenia_minibang_simulate import _load_lagrangian_series, simulate_batch

    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    c1_cfg = _get(cfg, ("datasets", "flow_lenia", "c1"), {})
    legacy_sigma_collision = bool(_get(c1_cfg, "optimized_replay_legacy_sigma_collision", False))
    optimized_replay_source = str(_get(c1_cfg, "optimized_replay_source", "apf")).strip().lower()
    pair_seed_base = int(_get(section, "run_seed_base", 400003))
    run_idx, run_dir = _find_run(selected_root, run)
    run_seed = pair_seed_base + 2 * int(run_idx) + int(seed_idx)

    best = _load_pickle(run_dir / "best.pkl")
    if isinstance(best, tuple) and len(best) == 2:
        params, loss = best
    else:
        params, loss = best, float("nan")
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    selected_audit = _selected_checkpoint_audit(run_dir, params)
    opt_cfg_path = run_dir / "optimization_config.yaml"
    if not opt_cfg_path.exists():
        raise FileNotFoundError(f"Missing copied optimization_config.yaml: {opt_cfg_path}")
    opt_flat = _flat_optimization_config(opt_cfg_path, legacy_sigma_collision=legacy_sigma_collision)

    rollout_config = resolve_path(_get(section, "rollout_config", None))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"rollout_config not found: {rollout_config}")
    rollout_cfg, rollout_flat = load_rollout_config(rollout_config, [])
    rollout_cfg, rollout_flat = _apply_section_rollout_overrides_for_preflight(rollout_cfg, rollout_flat, section)
    sample_every = int(_get(rollout_flat, "sample_every_steps", _get(rollout_flat, "snapshot_interval", 50)))
    snapshot_interval = int(_get(rollout_flat, "snapshot_interval", sample_every))
    if sample_every != snapshot_interval:
        raise ValueError(f"sample_every_steps={sample_every}, snapshot_interval={snapshot_interval}; expected equality.")
    if int(rollout_steps) % sample_every != 0:
        raise ValueError(f"rollout_steps={rollout_steps} must be divisible by sample_every={sample_every}.")

    flat_dict = OmegaConf.to_container(rollout_flat, resolve=True)
    flat_dict.update(
        {
            "rollout_steps": int(rollout_steps),
            "max_steps": int(rollout_steps),
            "snapshot_interval": sample_every,
            "snapshots_per_file": max(1, int(rollout_steps) // sample_every + 1),
            "batch_size": 1,
            "n_trajectories": 1,
            "run_seed_protocol": "optimization_metric",
            "img_size": 64,
            "video_img_size": 64,
            "save_rgb": False,
            "compute_metrics": False,
            "compute_delta_h": False,
            "compute_clusters": False,
        }
    )
    flat_dict["log_clip_evolution"] = bool(_get(opt_flat, "log_clip_evolution", True))
    if int(flat_dict.get("jit_microbatch", sample_every)) < sample_every:
        flat_dict["jit_microbatch"] = sample_every
    rollout_cfg_i = OmegaConf.create(OmegaConf.to_container(rollout_cfg, resolve=True))
    if rollout_cfg_i.get("minibang", None) is None:
        rollout_cfg_i.minibang = OmegaConf.create()
    rollout_cfg_i.minibang.run_seed_protocol = "optimization_metric"
    rollout_cfg_i.rollout.substrate.rollout_steps = int(rollout_steps)
    rollout_cfg_i.rollout.simulation.rollout_steps = int(rollout_steps)
    rollout_cfg_i.rollout.simulation.max_steps = int(rollout_steps)

    smoke_root = output_root / f"run_{run_idx:03d}_seed_{seed_idx:03d}_{rollout_steps}"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    selected_batch = [
        {
            "traj_id": f"run_{run_idx:03d}_seed_{seed_idx:03d}",
            "selection_idx": 0,
            "source_run_idx": int(run_idx),
            "run_seed": int(run_seed),
            "params": params,
            "loss": float(np.asarray(loss).reshape(-1)[0]) if np.asarray(loss).size else float("nan"),
        }
    ]
    simulate_batch(
        selected_batch=selected_batch,
        cfg=rollout_cfg_i,
        flat_args=flat_dict,
        output_root=smoke_root,
        overwrite=True,
    )
    apf_initial_snapshot = _load_first_apf_snapshot(smoke_root / selected_batch[0]["traj_id"] / "apf_logs")
    apf_steps, apf_xy_all = _load_lagrangian_series(smoke_root / selected_batch[0]["traj_id"] / "apf_logs")
    apf_mask = np.asarray(apf_steps) > 0
    apf_xy = np.asarray(apf_xy_all[apf_mask], dtype=np.float32)
    apf_xy_with_initial = np.asarray(apf_xy_all[np.asarray(apf_steps) >= 0], dtype=np.float32)
    scalar_opt_xy = _optimization_lagrangian_xy(
        opt_flat=opt_flat,
        params=params,
        run_seed=run_seed,
        rollout_steps=int(rollout_steps),
        sample_every_steps=sample_every,
    )
    scalar_opt_xy_with_initial = _optimization_lagrangian_xy(
        opt_flat=opt_flat,
        params=params,
        run_seed=run_seed,
        rollout_steps=int(rollout_steps),
        sample_every_steps=sample_every,
        include_initial=True,
    )
    scalar_initial_snapshot = _optimization_initial_snapshot(
        opt_flat=opt_flat,
        params=params,
        run_seed=run_seed,
    )
    rollout_initial_snapshot = _rollout_flat_initial_snapshot(
        flat_args=flat_dict,
        params=params,
        run_seed=run_seed,
    )
    one_step = _one_step_diagnostic(
        opt_flat=opt_flat,
        rollout_flat=flat_dict,
        params=params,
        run_seed=run_seed,
        rollout_steps=int(rollout_steps),
        sample_every_steps=sample_every,
    )
    apf_style_variants = _apf_style_first_chunk_variants(
        rollout_flat=flat_dict,
        params=params,
        run_seed=run_seed,
        rollout_steps=int(rollout_steps),
        sample_every_steps=sample_every,
    )
    reference_mode = "scalar_selected_candidate"
    reference_details: dict[str, Any] = {}
    opt_xy = scalar_opt_xy
    opt_xy_with_initial = scalar_opt_xy_with_initial
    reference_initial_snapshot = scalar_initial_snapshot
    batch_inputs = _optimizer_batch_reference_inputs(run_dir, int(seed_idx))
    flat_pair_vmap_xy: np.ndarray | None = None
    flat_pair_jit_xy: np.ndarray | None = None
    nested_eager_xy: np.ndarray | None = None
    nested_jit_xy: np.ndarray | None = None
    if batch_inputs is not None:
        reference_mode = "optimizer_original_pop_batch_nested_jit"
        reference_details = {
            "iter": int(batch_inputs["iter"]),
            "pop_idx": int(batch_inputs["pop_idx"]),
            "seed_idx": int(batch_inputs["seed_idx"]),
            "params_batch_shape": list(np.asarray(batch_inputs["params_batch"]).shape),
            "seed_keys_shape": list(np.asarray(batch_inputs["seed_keys"]).shape),
            "selected_seed_key": batch_inputs["selected_seed_key"],
        }
        nested_eager_xy = _optimization_lagrangian_xy_from_optimizer_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=sample_every,
        )
        nested_jit_xy = _optimization_lagrangian_xy_from_optimizer_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=sample_every,
            jit_compile=True,
        )
        opt_xy = nested_jit_xy
        opt_xy_with_initial = _optimization_lagrangian_xy_from_optimizer_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=sample_every,
            include_initial=True,
            jit_compile=True,
        )
        flat_pair_vmap_xy = _optimization_lagrangian_xy_from_flat_pair_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=sample_every,
        )
        flat_pair_jit_xy = _optimization_lagrangian_xy_from_flat_pair_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=sample_every,
            jit_compile=True,
        )
        reference_initial_snapshot = _optimization_initial_snapshot_from_optimizer_batch(
            opt_flat=opt_flat,
            params_batch=np.asarray(batch_inputs["params_batch"], dtype=np.float32),
            seed_keys=np.asarray(batch_inputs["seed_keys"], dtype=np.uint32),
            pop_idx=int(batch_inputs["pop_idx"]),
            seed_idx=int(batch_inputs["seed_idx"]),
        )

    optimizer_context_apf: dict[str, Any] = {"status": "skipped", "reason": "no optimizer batch inputs"}
    if batch_inputs is not None:
        params_batch = np.asarray(batch_inputs["params_batch"], dtype=np.float32)
        seed_keys = np.asarray(batch_inputs["seed_keys"], dtype=np.uint32)
        pop_idx = int(batch_inputs["pop_idx"])
        selected_seed_idx = int(batch_inputs["seed_idx"])
        selected_seed_int = _seed_int_from_prng_key(seed_keys[selected_seed_idx])
        seed_ints = [_seed_int_from_prng_key(seed_keys[i]) for i in range(seed_keys.shape[0])]

        def run_apf_context(
            *,
            context_name: str,
            rows: list[dict[str, Any]],
            selected_traj_id: str,
        ) -> dict[str, Any]:
            context_root = output_root / f"run_{run_idx:03d}_seed_{seed_idx:03d}_{rollout_steps}_{context_name}"
            if context_root.exists():
                shutil.rmtree(context_root)
            context_flat = dict(flat_dict)
            context_flat["batch_size"] = len(rows)
            context_flat["n_trajectories"] = len(rows)
            if int(context_flat.get("jit_microbatch", sample_every)) < sample_every:
                context_flat["jit_microbatch"] = sample_every
            simulate_batch(
                selected_batch=rows,
                cfg=rollout_cfg_i,
                flat_args=context_flat,
                output_root=context_root,
                overwrite=True,
            )
            context_apf_dir = context_root / selected_traj_id / "apf_logs"
            context_initial = _load_first_apf_snapshot(context_apf_dir)
            context_steps, context_xy_all = _load_lagrangian_series(context_apf_dir)
            context_xy = np.asarray(context_xy_all[np.asarray(context_steps) > 0], dtype=np.float32)
            return {
                "status": "ok",
                "context_name": context_name,
                "output_root": str(context_root),
                "selected_traj_id": selected_traj_id,
                "batch_size": int(len(rows)),
                "apf_steps": [int(x) for x in np.asarray(context_steps).reshape(-1)],
                "initial_snapshot_diff_vs_optimizer_reference": _snapshot_diff_summary(
                    context_initial,
                    reference_initial_snapshot,
                ),
                "diff_vs_optimizer_reference": _xy_diff_summary(context_xy, opt_xy, atol=atol),
                "diff_vs_single_selected_apf": _xy_diff_summary(context_xy, apf_xy, atol=atol),
                "diff_vs_scalar_reference": _xy_diff_summary(context_xy, scalar_opt_xy, atol=atol),
            }

        optimizer_context_apf = {
            "status": "skipped",
            "reason": "selected optimizer PRNGKey is not representable as PRNGKey(int)",
            "selected_seed_key": [int(x) for x in np.asarray(seed_keys[selected_seed_idx]).reshape(-1)],
        }
        if selected_seed_int is not None:
            same_seed_rows = []
            for j in range(params_batch.shape[0]):
                same_seed_rows.append(
                    {
                        "traj_id": f"pop_same_seed_pop_{j:03d}_seed_{selected_seed_idx:03d}",
                        "selection_idx": int(j),
                        "source_run_idx": int(run_idx),
                        "run_seed": int(selected_seed_int),
                        "params": np.asarray(params_batch[j], dtype=np.float32),
                        "loss": float("nan"),
                    }
                )
            optimizer_context_apf = {
                "status": "ok",
                "selected_seed_int": int(selected_seed_int),
                "same_seed_pop_batch": run_apf_context(
                    context_name="pop_same_seed_apf",
                    rows=same_seed_rows,
                    selected_traj_id=f"pop_same_seed_pop_{pop_idx:03d}_seed_{selected_seed_idx:03d}",
                ),
            }
        if all(seed_int is not None for seed_int in seed_ints):
            grid_rows = []
            for j in range(params_batch.shape[0]):
                for k_seed, seed_int in enumerate(seed_ints):
                    grid_rows.append(
                        {
                            "traj_id": f"pop_seed_grid_pop_{j:03d}_seed_{k_seed:03d}",
                            "selection_idx": int(j * seed_keys.shape[0] + k_seed),
                            "source_run_idx": int(run_idx),
                            "run_seed": int(seed_int),
                            "params": np.asarray(params_batch[j], dtype=np.float32),
                            "loss": float("nan"),
                        }
                    )
            if optimizer_context_apf.get("status") != "ok":
                optimizer_context_apf = {"status": "ok"}
            optimizer_context_apf["all_seeds_pop_grid"] = run_apf_context(
                context_name="pop_all_seeds_flat_apf",
                rows=grid_rows,
                selected_traj_id=f"pop_seed_grid_pop_{pop_idx:03d}_seed_{selected_seed_idx:03d}",
            )
        elif seed_keys.size:
            optimizer_context_apf["all_seeds_pop_grid"] = {
                "status": "skipped",
                "reason": "at least one optimizer PRNGKey is not representable as PRNGKey(int)",
                "seed_keys": [[int(x) for x in np.asarray(key).reshape(-1)] for key in seed_keys],
            }
    if apf_xy.shape != opt_xy.shape:
        raise ValueError(f"APF/optimization xy shape mismatch: apf={apf_xy.shape}, opt={opt_xy.shape}")
    diff = np.asarray(apf_xy, dtype=np.float32) - np.asarray(opt_xy, dtype=np.float32)
    scalar_diff = np.asarray(apf_xy, dtype=np.float32) - np.asarray(scalar_opt_xy, dtype=np.float32)
    scalar_initial_diff = (
        np.asarray(apf_xy_with_initial[0], dtype=np.float32)
        - np.asarray(scalar_opt_xy_with_initial[0], dtype=np.float32)
    )
    max_abs = float(np.nanmax(np.abs(diff)))
    mean_abs = float(np.nanmean(np.abs(diff)))
    initial_diff = np.asarray(apf_xy_with_initial[0], dtype=np.float32) - np.asarray(opt_xy_with_initial[0], dtype=np.float32)
    per_sample_max = np.nanmax(np.abs(diff).reshape((diff.shape[0], -1)), axis=1)
    per_sample_mean = np.nanmean(np.abs(diff).reshape((diff.shape[0], -1)), axis=1)
    first_failed = np.flatnonzero(per_sample_max > float(atol))
    ok = bool(max_abs <= float(atol))
    first_failed_step = (
        int(np.asarray(apf_steps)[np.asarray(apf_steps) > 0][int(first_failed[0])])
        if first_failed.size
        else None
    )
    initial_max_abs = float(np.nanmax(np.abs(initial_diff)))
    initial_mean_abs = float(np.nanmean(np.abs(initial_diff)))
    apf_chunk_stepper_mode = None
    if "resume_stepper_mode" in apf_initial_snapshot:
        raw_mode = np.asarray(apf_initial_snapshot["resume_stepper_mode"])
        apf_chunk_stepper_mode = str(raw_mode.item() if raw_mode.shape == () else raw_mode.reshape(-1)[0])
    flat_pair_vmap_reference = None
    if flat_pair_vmap_xy is not None:
        flat_pair_vmap_reference = {
            "diff_vs_optimizer_nested_reference": _xy_diff_summary(flat_pair_vmap_xy, opt_xy, atol=atol),
            "diff_vs_single_selected_apf": _xy_diff_summary(flat_pair_vmap_xy, apf_xy, atol=atol),
            "diff_vs_scalar_reference": _xy_diff_summary(flat_pair_vmap_xy, scalar_opt_xy, atol=atol),
        }
    jit_reference = None
    if nested_jit_xy is not None and flat_pair_jit_xy is not None:
        jit_reference = {
            "nested_jit_vs_nested_eager": _xy_diff_summary(nested_jit_xy, nested_eager_xy, atol=atol),
            "nested_jit_vs_single_selected_apf": _xy_diff_summary(nested_jit_xy, apf_xy, atol=atol),
            "flat_jit_vs_flat_eager": _xy_diff_summary(flat_pair_jit_xy, flat_pair_vmap_xy, atol=atol),
            "flat_jit_vs_single_selected_apf": _xy_diff_summary(flat_pair_jit_xy, apf_xy, atol=atol),
            "flat_jit_vs_nested_jit": _xy_diff_summary(flat_pair_jit_xy, nested_jit_xy, atol=atol),
        }
    apf_style_chunk_reference = {
        "no_capture_vs_single_selected_apf": _xy_diff_summary(apf_style_variants["no_capture_xy"], apf_xy, atol=atol),
        "with_capture_vs_single_selected_apf": _xy_diff_summary(apf_style_variants["with_capture_xy"], apf_xy, atol=atol),
        "no_capture_vs_with_capture": _xy_diff_summary(
            apf_style_variants["no_capture_xy"],
            apf_style_variants["with_capture_xy"],
            atol=atol,
        ),
        "no_capture_vs_eager_reference": _xy_diff_summary(apf_style_variants["no_capture_xy"], opt_xy, atol=atol),
        "with_capture_vs_eager_reference": _xy_diff_summary(apf_style_variants["with_capture_xy"], opt_xy, atol=atol),
    }
    if flat_pair_jit_xy is not None:
        apf_style_chunk_reference["no_capture_vs_flat_jit_reference"] = _xy_diff_summary(
            apf_style_variants["no_capture_xy"],
            flat_pair_jit_xy,
            atol=atol,
        )
        apf_style_chunk_reference["with_capture_vs_flat_jit_reference"] = _xy_diff_summary(
            apf_style_variants["with_capture_xy"],
            flat_pair_jit_xy,
            atol=atol,
        )
    first_chunk_trace: dict[str, Any]
    try:
        first_chunk_trace = _first_chunk_trace_diagnostic(
            rollout_flat=flat_dict,
            params=params,
            run_seed=run_seed,
            sample_every_steps=sample_every,
            atol=atol,
        )
        first_chunk_trace["status"] = "ok"
    except Exception as exc:
        first_chunk_trace = {"status": "error", "error": repr(exc)}

    full_rollout_arrays: dict[str, np.ndarray | None] = {
        "single_selected_apf": apf_xy,
        "scalar_eager_reference": scalar_opt_xy,
        "active_reference": opt_xy,
        "optimizer_nested_eager": nested_eager_xy,
        "optimizer_nested_jit": nested_jit_xy,
        "optimizer_flat_eager": flat_pair_vmap_xy,
        "optimizer_flat_jit": flat_pair_jit_xy,
    }
    first_chunk_arrays: dict[str, np.ndarray | None] = {
        "single_selected_apf_first_chunk": apf_xy[:1],
        "scalar_eager_reference_first_chunk": scalar_opt_xy[:1],
        "active_reference_first_chunk": opt_xy[:1],
        "optimizer_nested_eager_first_chunk": None if nested_eager_xy is None else nested_eager_xy[:1],
        "optimizer_nested_jit_first_chunk": None if nested_jit_xy is None else nested_jit_xy[:1],
        "optimizer_flat_eager_first_chunk": None if flat_pair_vmap_xy is None else flat_pair_vmap_xy[:1],
        "optimizer_flat_jit_first_chunk": None if flat_pair_jit_xy is None else flat_pair_jit_xy[:1],
        "apf_style_no_capture_first_chunk": apf_style_variants["no_capture_xy"],
        "apf_style_with_capture_first_chunk": apf_style_variants["with_capture_xy"],
    }
    diagnostic_pack = {
        "runtime_source_audit": _runtime_source_audit(),
        "variant_shapes": {
            "full_rollout": _array_shapes(full_rollout_arrays),
            "first_chunk": _array_shapes(first_chunk_arrays),
        },
        "full_rollout_pairwise_xy": _xy_pairwise_matrix(full_rollout_arrays, atol=atol),
        "first_chunk_pairwise_xy": _xy_pairwise_matrix(first_chunk_arrays, atol=atol),
        "first_chunk_trace": first_chunk_trace,
        "interpretation_hints": [
            "If first_chunk_trace variants disagree before internal_step=50, the mismatch is inside JIT/scan/vmap execution, not MSPD.",
            "If apf_style_* matches single_selected_apf but optimizer_flat_jit does not, APF replay and optimizer replay use different compiled rollout structure.",
            "If optimizer_flat_jit matches single_selected_apf but scalar_eager/nested_eager do not, eager references are not valid for this archived run.",
            "If one_step_diagnostic is zero but first_chunk_trace diverges later, keys/config/init are aligned and the divergence is caused by multi-step numerical execution.",
        ],
    }
    raw_config_diffs = _config_value_diff(
        OmegaConf.create(flat_dict),
        opt_flat,
        [
            "substrate",
            "grid_size",
            "C",
            "k",
            "kernel_components",
            "M",
            "dd",
            "dt",
            "sigma",
            "flow_sigma",
            "border",
            "mix_rule",
            "sobel_impl",
            "base_seed",
            "seed_patch_size",
            "seed_n_patches",
            "seed_mode",
            "p_constant_per_patch",
            "render_mode",
            "clip1",
            "clip2",
            "mutations",
            "mutation_sz",
            "mutation_p",
            "mutation_scale",
            "optimize_mutation_scale",
            "volcano",
            "volcano_sz",
            "volcano_p",
            "volcano_delta",
            "food",
            "food_interval",
            "food_n",
            "food_sz",
            "food_amount",
            "food_consume_rate",
            "food_bonus",
            "mass_decay",
            "food_channel",
            "food_auto_size",
            "food_auto_scale",
            "food_conv_mode",
            "food_vis_scale",
            "food_vis_color",
            "food_diffusion_alpha",
            "mass_clip_eps",
            "mass_renorm",
            "log_clip_evolution",
        ],
    )
    protocol_config_diffs, ignored_config_diffs = _split_protocol_config_diffs(
        raw_config_diffs,
        rollout_flat=OmegaConf.create(flat_dict),
        opt_flat=opt_flat,
        legacy_sigma_collision=legacy_sigma_collision,
    )
    def _one_step_component_ok(diff_name: str, component: str) -> bool:
        diff_block = one_step.get(diff_name, {}) if isinstance(one_step, dict) else {}
        value = diff_block.get(f"{component}_max_abs_diff", float("inf"))
        try:
            return bool(float(value) <= float(atol))
        except Exception:
            return False

    one_step_ok = all(
        _one_step_component_ok(diff_name, component)
        for diff_name in ("initial_diff", "after_one_step_diff")
        for component in ("A", "P", "F", "lagrangian_xy")
    )
    optimizer_native_apf_diagnostic_only = bool(
        (not ok)
        and optimized_replay_source == "optimizer_native"
        and not protocol_config_diffs
        and one_step_ok
    )
    known_execution_divergence = bool(
        (
            (not ok)
            and initial_max_abs <= float(atol)
            and not protocol_config_diffs
            and first_failed_step is not None
        )
        or optimizer_native_apf_diagnostic_only
    )
    smoke_status = "ok" if ok else ("known_execution_divergence" if known_execution_divergence else "failed")
    return {
        "status": smoke_status,
        "strict_status": "ok" if ok else "failed",
        "known_execution_divergence": bool(known_execution_divergence),
        "selected_checkpoint_audit": selected_audit,
        "apf_module_stepper_mode": str(getattr(minibang_sim, "OPTIMIZATION_METRIC_STEPPER_MODE", "missing")),
        "apf_chunk_stepper_mode": apf_chunk_stepper_mode,
        "reference_mode": reference_mode,
        "reference_details": reference_details,
        "legacy_sigma_collision": bool(legacy_sigma_collision),
        "optimized_replay_source": optimized_replay_source,
        "optimizer_native_apf_diagnostic_only": bool(optimizer_native_apf_diagnostic_only),
        "one_step_ok": bool(one_step_ok),
        "flat_pair_vmap_reference": flat_pair_vmap_reference,
        "jit_reference": jit_reference,
        "apf_style_chunk_reference": apf_style_chunk_reference,
        "diagnostic_pack": diagnostic_pack,
        "optimizer_context_apf": optimizer_context_apf,
        "one_step_diagnostic": one_step,
        "initial_snapshot_diff": _snapshot_diff_summary(apf_initial_snapshot, reference_initial_snapshot),
        "scalar_initial_snapshot_diff": _snapshot_diff_summary(apf_initial_snapshot, scalar_initial_snapshot),
        "apf_vs_rollout_config_initial_snapshot_diff": _snapshot_diff_summary(apf_initial_snapshot, rollout_initial_snapshot),
        "rollout_config_vs_optimization_initial_snapshot_diff": _snapshot_diff_summary(
            rollout_initial_snapshot,
            reference_initial_snapshot,
        ),
        "rollout_vs_optimization_config_diffs": protocol_config_diffs,
        "rollout_vs_optimization_raw_config_diffs": raw_config_diffs,
        "rollout_vs_optimization_ignored_config_diffs": ignored_config_diffs,
        "run_idx": int(run_idx),
        "seed_idx": int(seed_idx),
        "run_seed": int(run_seed),
        "rollout_steps": int(rollout_steps),
        "sample_every_steps": int(sample_every),
        "n_samples_compared": int(opt_xy.shape[0]),
        "xy_shape": list(opt_xy.shape),
        "apf_steps": [int(x) for x in np.asarray(apf_steps).reshape(-1)],
        "initial_max_abs_xy_diff": initial_max_abs,
        "initial_mean_abs_xy_diff": initial_mean_abs,
        "scalar_reference_initial_max_abs_xy_diff": float(np.nanmax(np.abs(scalar_initial_diff))),
        "scalar_reference_initial_mean_abs_xy_diff": float(np.nanmean(np.abs(scalar_initial_diff))),
        "scalar_reference_max_abs_xy_diff": float(np.nanmax(np.abs(scalar_diff))),
        "scalar_reference_mean_abs_xy_diff": float(np.nanmean(np.abs(scalar_diff))),
        "per_sample_max_abs_xy_diff": [float(x) for x in per_sample_max.reshape(-1)],
        "per_sample_mean_abs_xy_diff": [float(x) for x in per_sample_mean.reshape(-1)],
        "first_failed_sample_idx": int(first_failed[0]) if first_failed.size else None,
        "first_failed_step": first_failed_step,
        "max_abs_xy_diff": max_abs,
        "mean_abs_xy_diff": mean_abs,
        "atol": float(atol),
        "smoke_output_root": str(smoke_root),
    }


def _audit_existing_results(scores_csv: Path, config_path: Path) -> dict[str, Any]:
    if not scores_csv.exists():
        return {"scores_csv": str(scores_csv), "status": "missing", "errors": [f"missing {scores_csv}"]}
    import pandas as pd

    df = pd.read_csv(scores_csv)
    errors: list[str] = []
    if "metric_seed_protocol" not in df.columns:
        errors.append("checkpoint_scores.csv has no metric_seed_protocol column; metrics are stale.")
    else:
        vals = sorted(set(str(x) for x in df["metric_seed_protocol"].dropna().unique()))
        if vals != ["optimization_metric"]:
            errors.append(f"metric_seed_protocol values are {vals}, expected ['optimization_metric']")
    opt = df[df.get("candidate_kind", "") == "optimized"].copy() if "candidate_kind" in df.columns else df.iloc[0:0]
    if opt.empty:
        errors.append("no optimized rows found in checkpoint_scores.csv")
    for col in ("train_tau_steps", "full_score_train_tau_mspd", "eval_score_mspd"):
        if col not in df.columns:
            errors.append(f"checkpoint_scores.csv has no {col} column")
    if "train_tau_steps" in opt.columns:
        missing_tau = int(opt["train_tau_steps"].isna().sum())
        if missing_tau:
            errors.append(f"{missing_tau} optimized rows have missing train_tau_steps")
    if "c1_replay_source" not in df.columns:
        errors.append("checkpoint_scores.csv has no c1_replay_source column; metrics are stale.")
    else:
        opt_sources = sorted(set(str(x) for x in opt["c1_replay_source"].dropna().unique()))
        expected_opt_sources = [
            "apf_lagrangian",
            "optimizer_native_nested_jit",
            "optimizer_native_nested_jit_legacy_sigma_collision",
        ]
        if not opt_sources or any(src not in expected_opt_sources for src in opt_sources):
            errors.append(
                "optimized c1_replay_source values are "
                f"{opt_sources}, expected one of {expected_opt_sources}"
            )
        if "candidate_kind" in df.columns:
            rand = df[df["candidate_kind"] == "random"]
            if not rand.empty:
                rand_sources = sorted(set(str(x) for x in rand["c1_replay_source"].dropna().unique()))
                expected_rand_sources = [
                    "apf_lagrangian",
                    "optimizer_native_nested_jit_row_params",
                    "optimizer_native_nested_jit_row_params_legacy_sigma_collision",
                ]
                if any(src not in expected_rand_sources for src in rand_sources):
                    errors.append(
                        "random c1_replay_source values are "
                        f"{rand_sources}, expected subset of {expected_rand_sources}"
                    )
    cfg = OmegaConf.load(config_path)
    c1_cfg = cfg.datasets.flow_lenia.c1
    cross_hardware_runs = {
        int(value)
        for value in as_list(
            _get(c1_cfg, "optimizer_reference_cross_hardware_source_run_indices", [])
        )
    }
    cross_hardware_max_ulps = int(
        _get(c1_cfg, "optimizer_reference_cross_hardware_max_ulps", 0)
    )
    exact_col = "optimizer_reference_train_mspd_exact_match"
    validation_col = "optimizer_reference_train_mspd_validation_passed"
    exception_col = "optimizer_reference_cross_hardware_exception_used"
    ulp_col = "optimizer_reference_train_mspd_ulp_distance"
    exact_count = 0
    exception_count = 0
    if validation_col not in opt.columns:
        errors.append(f"checkpoint_scores.csv has no {validation_col} column; metrics are stale.")
    else:
        validation_values = opt[validation_col].astype(str).str.lower().isin({"true", "1"})
        if not bool(validation_values.all()):
            errors.append(
                f"{int((~validation_values).sum())} optimized rows fail train-MSPD validation"
            )
    if exact_col in opt.columns:
        exact_values = opt[exact_col].astype(str).str.lower().isin({"true", "1"})
        exact_count = int(exact_values.sum())
    else:
        errors.append(f"checkpoint_scores.csv has no {exact_col} column; metrics are stale.")
        exact_values = pd.Series(False, index=opt.index)
    if exception_col in opt.columns:
        exception_values = opt[exception_col].astype(str).str.lower().isin({"true", "1"})
        exception_count = int(exception_values.sum())
        if bool(exception_values.any()):
            if "source_optimized_run_idx" not in opt.columns:
                errors.append("cross-hardware exceptions exist but source_optimized_run_idx is missing")
            else:
                exception_runs = set(
                    pd.to_numeric(
                        opt.loc[exception_values, "source_optimized_run_idx"], errors="coerce"
                    ).dropna().astype(int)
                )
                unexpected_runs = sorted(exception_runs - cross_hardware_runs)
                if unexpected_runs:
                    errors.append(
                        f"cross-hardware MSPD exceptions used for unconfigured source runs {unexpected_runs}"
                    )
            if ulp_col not in opt.columns:
                errors.append(f"cross-hardware exceptions exist but {ulp_col} is missing")
            else:
                exception_ulps = pd.to_numeric(opt.loc[exception_values, ulp_col], errors="coerce")
                invalid_ulps = exception_ulps.isna() | (exception_ulps > cross_hardware_max_ulps)
                if bool(invalid_ulps.any()):
                    errors.append(
                        f"{int(invalid_ulps.sum())} cross-hardware MSPD exceptions exceed "
                        f"configured max {cross_hardware_max_ulps} ULP"
                    )
        unvalidated_nonexact = (~exact_values) & (~exception_values)
        if bool(unvalidated_nonexact.any()):
            errors.append(
                f"{int(unvalidated_nonexact.sum())} optimized rows are neither exact nor a validated cross-hardware exception"
            )
    else:
        errors.append(f"checkpoint_scores.csv has no {exception_col} column; metrics are stale.")
    return {
        "scores_csv": str(scores_csv),
        "status": "ok" if not errors else "failed",
        "n_rows": int(len(df)),
        "n_optimized_rows": int(len(opt)),
        "n_optimizer_reference_bit_exact": exact_count,
        "n_optimizer_reference_cross_hardware_ulp": exception_count,
        "optimizer_reference_cross_hardware_source_runs": sorted(cross_hardware_runs),
        "optimizer_reference_cross_hardware_max_ulps": cross_hardware_max_ulps,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast preflight for Flow-Lenia fixed-init C1 replay protocol.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--selected-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--source-optimization-root", default=None)
    parser.add_argument("--scores-csv", default=DEFAULT_SCORES_CSV)
    parser.add_argument("--run", default=None, help="Run index or run_XXX. Defaults to first selected run.")
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--output-root", default="/tmp/asal_flowlenia_c1_replay_preflight")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-existing-results", action="store_true")
    parser.add_argument("--skip-selected-candidate-audit", action="store_true")
    parser.add_argument("--skip-random-checkpoint-audit", action="store_true")
    parser.add_argument("--skip-optimization-protocol-audit", action="store_true")
    parser.add_argument(
        "--allow-known-execution-divergence",
        action="store_true",
        help=(
            "Do not fail when replay smoke matches initial state/config but diverges after rollout steps, "
            "which is the known APF-vs-optimizer JAX execution-path mismatch."
        ),
    )
    parser.add_argument(
        "--require-apf-root-contains",
        default=None,
        help="Fail config audit unless generated APF root contains this substring.",
    )
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    selected_root = Path(args.selected_root)
    source_optimization_root = None if args.source_optimization_root is None else _as_repo_path(args.source_optimization_root)
    scores_csv = Path(args.scores_csv)
    summary: dict[str, Any] = {
        "config_audit": _audit_generated_config(config_path),
    }
    selected_run_dirs = _configured_selected_run_dirs(config_path, selected_root)
    summary["selected_run_dirs"] = [str(path) for path in selected_run_dirs]
    if args.require_apf_root_contains:
        apf_root = str(summary["config_audit"].get("apf_root", ""))
        if str(args.require_apf_root_contains) not in apf_root:
            summary["config_audit"].setdefault("errors", []).append(
                f"apf_root={apf_root!r} does not contain required substring {args.require_apf_root_contains!r}"
            )
    summary["seed_count_audit"] = _audit_seed_count(summary["config_audit"], selected_run_dirs)
    if not args.skip_selected_candidate_audit:
        summary["selected_candidate_audit"] = _audit_selected_candidates(
            selected_run_dirs,
            source_optimization_root=source_optimization_root,
        )
    if not args.skip_optimization_protocol_audit:
        summary["optimization_protocol_audit"] = _audit_optimization_protocol(
            summary["config_audit"],
            selected_run_dirs,
        )
    if not args.skip_random_checkpoint_audit:
        summary["random_checkpoint_audit"] = _audit_random_checkpoints(config_path, selected_run_dirs)
    if not args.skip_existing_results:
        summary["existing_results_audit"] = _audit_existing_results(scores_csv, config_path)
    if not args.skip_smoke:
        summary["replay_smoke"] = _run_replay_smoke(
            config_path=config_path,
            selected_root=selected_root,
            run=args.run,
            seed_idx=int(args.seed_idx),
            rollout_steps=int(args.rollout_steps),
            output_root=Path(args.output_root),
            atol=float(args.atol),
        )

    errors: list[str] = []
    errors.extend(summary["config_audit"].get("errors", []))
    errors.extend(summary["seed_count_audit"].get("errors", []))
    if "selected_candidate_audit" in summary:
        errors.extend(summary["selected_candidate_audit"].get("errors", []))
    if "optimization_protocol_audit" in summary:
        errors.extend(summary["optimization_protocol_audit"].get("errors", []))
    if "random_checkpoint_audit" in summary:
        errors.extend(summary["random_checkpoint_audit"].get("errors", []))
    if "existing_results_audit" in summary:
        errors.extend(summary["existing_results_audit"].get("errors", []))
    if "replay_smoke" in summary and summary["replay_smoke"].get("status") != "ok":
        if (
            args.allow_known_execution_divergence
            and summary["replay_smoke"].get("status") == "known_execution_divergence"
        ):
            summary.setdefault("warnings", []).append(
                "replay_smoke classified as known_execution_divergence and allowed by CLI flag"
            )
        else:
            errors.append(
                "replay_smoke failed: max_abs_xy_diff="
                f"{summary['replay_smoke'].get('max_abs_xy_diff')}"
            )
    summary["status"] = "ok" if not errors else "failed"
    summary["errors"] = errors

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
