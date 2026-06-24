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


def _flat_for_run(run_dir: Path, fallback_config: Path) -> tuple[SimpleNamespace, Path, bool]:
    run_config = run_dir / "optimization_config.yaml"
    if run_config.exists():
        return _flat_optimization_config(run_config), run_config, True
    return _flat_optimization_config(fallback_config), fallback_config, False


def _checkpoint_root_from_suite(cfg: Any) -> Path:
    section = _get(_get(cfg.get("simulation", {}), "flow_lenia_arun_lagrangian_apf", {}), "optimized_checkpoint_roots", None)
    roots = list(section or [])
    if not roots:
        raise ValueError("No simulation.flow_lenia_arun_lagrangian_apf.optimized_checkpoint_roots configured.")
    root = resolve_path(roots[0])
    if root is None or not root.exists():
        raise FileNotFoundError(f"Optimization checkpoint root not found: {root}")
    return root


def _run_dirs(root: Path, max_runs: int | None, run_name: str | None = None) -> list[Path]:
    if run_name is not None:
        name = str(run_name).strip()
        run_dir = root / name
        if not run_dir.is_dir() or not (run_dir / "best.pkl").exists():
            raise FileNotFoundError(f"Requested run {name!r} not found under {root} or missing best.pkl.")
        return [run_dir]

    dirs = [p for p in sorted(root.glob("run_*")) if p.is_dir() and (p / "best.pkl").exists()]
    if max_runs is not None:
        dirs = dirs[: int(max_runs)]
    if not dirs:
        raise FileNotFoundError(f"No run_*/best.pkl checkpoints under {root}")
    return dirs


def _optimizer_eval_keys(
    *,
    seed: int,
    params_init: str,
    best_iter: int,
    pop_idx: int,
    pop_size: int,
    pop_batch: int,
    bs: int,
    init_policy: str = "config",
    ask_split: bool = True,
    eval_split_mode: str = "main",
    extra_splits_after_init: int = 0,
    extra_splits_each_iter_before_eval: int = 0,
) -> jax.Array:
    rng = jax.random.PRNGKey(int(seed))
    policy = str(init_policy).strip().lower()
    mode = str(params_init or "strategy_default").strip().lower().replace("-", "_")
    if policy == "config":
        if mode in {"strategy_default", "optimizer_default", "default"}:
            rng, _rng_init = jax.random.split(rng)
        elif mode in {"substrate_default", "default_params", "smart"}:
            rng, _rng_mean, _rng_init = jax.random.split(rng, 3)
        else:
            raise ValueError(f"Unknown params_init={params_init!r}.")
    elif policy == "none":
        pass
    elif policy == "one":
        rng, _rng_init = jax.random.split(rng)
    elif policy == "three":
        rng, _rng_mean, _rng_init = jax.random.split(rng, 3)
    else:
        raise ValueError(f"Unknown init_policy={init_policy!r}.")
    for _ in range(int(extra_splits_after_init)):
        rng, _unused = jax.random.split(rng)

    for i_iter in range(int(best_iter) + 1):
        if ask_split:
            rng, _rng_ask = jax.random.split(rng)
        for _ in range(int(extra_splits_each_iter_before_eval)):
            rng, _unused = jax.random.split(rng)
        rng_eval = rng
        for start in range(0, int(pop_size), int(pop_batch)):
            end = min(int(pop_size), start + int(pop_batch))
            if eval_split_mode == "main":
                rng_next, rng_metric_parent = jax.random.split(rng_eval)
            elif eval_split_mode == "parent_first":
                rng_metric_parent, rng_next = jax.random.split(rng_eval)
            elif eval_split_mode == "no_presplit":
                rng_next = rng_eval
                rng_metric_parent = rng_eval
            else:
                raise ValueError(f"Unknown eval_split_mode={eval_split_mode!r}.")
            keys = jax.random.split(rng_metric_parent, int(bs))
            if i_iter == int(best_iter) and start <= int(pop_idx) < end:
                return keys
            rng_eval = rng_next
        rng = rng_eval
    raise RuntimeError("Failed to reconstruct optimizer evaluation RNG keys.")


def _optimizer_rng_debug_variants() -> list[dict[str, Any]]:
    return [
        {"variant": "main_config", "init_policy": "config", "ask_split": True, "eval_split_mode": "main"},
        {"variant": "no_init_split", "init_policy": "none", "ask_split": True, "eval_split_mode": "main"},
        {"variant": "force_one_init_split", "init_policy": "one", "ask_split": True, "eval_split_mode": "main"},
        {"variant": "force_three_init_split", "init_policy": "three", "ask_split": True, "eval_split_mode": "main"},
        {"variant": "no_ask_split", "init_policy": "config", "ask_split": False, "eval_split_mode": "main"},
        {"variant": "eval_parent_first", "init_policy": "config", "ask_split": True, "eval_split_mode": "parent_first"},
        {"variant": "eval_no_presplit", "init_policy": "config", "ask_split": True, "eval_split_mode": "no_presplit"},
        {
            "variant": "extra_split_after_init",
            "init_policy": "config",
            "ask_split": True,
            "eval_split_mode": "main",
            "extra_splits_after_init": 1,
        },
        {
            "variant": "extra_split_each_iter_before_eval",
            "init_policy": "config",
            "ask_split": True,
            "eval_split_mode": "main",
            "extra_splits_each_iter_before_eval": 1,
        },
    ]


def _best_pop_candidate(run_dir: Path, saved_best_loss: float) -> dict[str, Any]:
    pop_path = run_dir / "pop_traj.pkl"
    if not pop_path.exists():
        raise FileNotFoundError(f"Cannot replay optimizer evaluation without {pop_path}")
    pop = _load_pickle(pop_path)
    losses = np.asarray(pop["loss"], dtype=np.float64)
    params = np.asarray(pop["params"], dtype=np.float32)
    if losses.ndim != 2:
        raise ValueError(f"{pop_path} loss must have shape (n_iters, pop_size), got {losses.shape}")
    if params.shape[:2] != losses.shape:
        raise ValueError(f"{pop_path} params shape {params.shape} does not match loss shape {losses.shape}")
    best_flat = int(np.nanargmin(losses))
    best_iter, pop_idx = np.unravel_index(best_flat, losses.shape)
    pop_best_loss = float(losses[best_iter, pop_idx])
    tau_raw = 0.0
    if "tau_selector_raw" in pop:
        tau_raw_arr = np.asarray(pop["tau_selector_raw"], dtype=np.float32)
        if tau_raw_arr.shape[:2] != losses.shape:
            raise ValueError(f"{pop_path} tau_selector_raw shape {tau_raw_arr.shape} does not match loss shape {losses.shape}")
        tau_raw = float(tau_raw_arr[best_iter, pop_idx])
    return {
        "best_iter": int(best_iter),
        "pop_idx": int(pop_idx),
        "pop_best_loss": pop_best_loss,
        "saved_loss_minus_pop_best_loss": float(saved_best_loss - pop_best_loss) if np.isfinite(saved_best_loss) else np.nan,
        "params": params[best_iter, pop_idx],
        "tau_selector_raw": tau_raw,
    }


def _build_rescorer(args: SimpleNamespace, *, include_maps: bool = True):
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
    metric_loss_fn = make_metric_loss_fn(metric_cfg, include_maps=include_maps)
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
        out = {
            "loss": loss,
            "score": info["score"],
            "tau_selected_idx": info["tau_selected_idx"],
            "tau_selected_steps": info["tau_best_steps"],
        }
        if include_maps:
            score_by_tau = info["score_by_tau"]
            best_idx = jnp.argmax(score_by_tau)
            out.update(
                {
                    "score_by_tau": score_by_tau,
                    "max_tau_idx": best_idx,
                    "max_tau_steps": info["tau_steps"][best_idx],
                    "max_tau_score": score_by_tau[best_idx],
                }
            )
        else:
            out.update(
                {
                    "max_tau_idx": info["tau_selected_idx"],
                    "max_tau_steps": info["tau_best_steps"],
                    "max_tau_score": info["score_tau_max"],
                }
            )
        return out

    return eval_one, metric_cfg


def _run_replay(
    config_path: str | Path,
    *,
    optimization_config: str | Path,
    output_dir: str | Path | None,
    force: bool,
    max_runs: int | None,
    run_name: str | None,
    debug_rng_variants: bool,
    debug_rng_variant: str | None,
) -> dict[str, Any]:
    suite_cfg, _ = load_config(config_path)
    opt_root = _checkpoint_root_from_suite(suite_cfg)
    opt_cfg_path = resolve_path(optimization_config)
    if opt_cfg_path is None or not opt_cfg_path.exists():
        raise FileNotFoundError(f"Optimization config not found: {opt_cfg_path}")

    if output_dir is None:
        root = resolve_path(_get(suite_cfg.get("meta", {}), "output_root", "analysis/results/paper_suite"))
        out_dir = ensure_dir(Path(root) / "flow_lenia")
    else:
        out_dir = ensure_dir(resolve_path(output_dir) or Path(output_dir))
    replay_path = out_dir / "train_objective_replay.csv"
    variants_path = out_dir / "train_objective_replay_rng_variants.csv"
    summary_path = out_dir / "train_objective_replay_summary.json"
    if not force and replay_path.exists() and (not debug_rng_variants or variants_path.exists()):
        return {"status": "exists", "replay": str(replay_path)}

    run_dirs = _run_dirs(opt_root, max_runs, run_name=run_name)
    first_flat, first_config_path, _first_has_run_config = _flat_for_run(run_dirs[0], opt_cfg_path)
    if not _first_has_run_config:
        raise FileNotFoundError(
            f"Exact optimizer replay requires per-run optimization_config.yaml, missing for {run_dirs[0]}."
        )
    eval_one, metric_cfg = _build_rescorer(first_flat, include_maps=False)
    rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    log_event(
        f"Flow-Lenia train objective replay start n_runs={len(run_dirs)} opt_root={opt_root} evaluator_config={first_config_path}",
        component="train-replay",
    )
    for run_dir in tqdm(run_dirs, desc="train-objective-replay"):
        run_flat, run_config_path, has_run_config = _flat_for_run(run_dir, opt_cfg_path)
        if not has_run_config:
            raise FileNotFoundError(
                f"Exact optimizer replay requires per-run optimization_config.yaml, missing for {run_dir}."
            )
        pop_size = int(getattr(run_flat, "pop_size"))
        pop_batch = int(getattr(run_flat, "pop_batch", pop_size))
        bs = int(getattr(run_flat, "bs", 1))
        params_init = str(getattr(run_flat, "params_init", "strategy_default"))
        seed = int(getattr(run_flat, "seed"))
        _best_params, saved_loss = _load_best_checkpoint(run_dir)
        candidate = _best_pop_candidate(run_dir, saved_loss)
        keys = _optimizer_eval_keys(
            seed=seed,
            params_init=params_init,
            best_iter=int(candidate["best_iter"]),
            pop_idx=int(candidate["pop_idx"]),
            pop_size=pop_size,
            pop_batch=pop_batch,
            bs=bs,
        )
        params = jnp.asarray(np.asarray(candidate["params"], dtype=np.float32), dtype=jnp.float32)
        tau_raw = jnp.asarray(float(candidate["tau_selector_raw"]), dtype=jnp.float32)
        losses: list[float] = []
        scores: list[float] = []
        max_scores: list[float] = []
        selected_tau_steps: list[int] = []
        for key in keys:
            out = eval_one(key, params, tau_raw)
            out_np = {name: np.asarray(jax.device_get(value)) for name, value in out.items()}
            losses.append(float(np.asarray(out_np["loss"]).reshape(-1)[0]))
            scores.append(float(np.asarray(out_np["score"]).reshape(-1)[0]))
            max_scores.append(float(np.asarray(out_np["max_tau_score"]).reshape(-1)[0]))
            selected_tau_steps.append(int(np.asarray(out_np["tau_selected_steps"]).reshape(-1)[0]))
        replay_loss = float(np.mean(np.asarray(losses, dtype=np.float64)))
        replay_mspd = float(np.mean(np.asarray(scores, dtype=np.float64)))
        rows.append(
            {
                "run": run_dir.name,
                "run_config_path": str(run_config_path),
                "run_config_exact": bool(has_run_config),
                "optimizer_seed": seed,
                "bs": bs,
                "pop_size": pop_size,
                "pop_batch": pop_batch,
                "params_init": params_init,
                "best_iter": int(candidate["best_iter"]),
                "pop_idx": int(candidate["pop_idx"]),
                "saved_best_loss": saved_loss,
                "saved_best_mspd": -saved_loss if np.isfinite(saved_loss) else np.nan,
                "pop_best_loss": float(candidate["pop_best_loss"]),
                "pop_best_mspd": -float(candidate["pop_best_loss"]),
                "saved_loss_minus_pop_best_loss": float(candidate["saved_loss_minus_pop_best_loss"]),
                "replay_loss_mean": replay_loss,
                "replay_mspd_mean": replay_mspd,
                "replay_max_tau_mspd_mean": float(np.mean(np.asarray(max_scores, dtype=np.float64))),
                "replay_loss_minus_pop_best_loss": float(replay_loss - float(candidate["pop_best_loss"])),
                "replay_mspd_minus_pop_best_mspd": float(replay_mspd - (-float(candidate["pop_best_loss"]))),
                "replay_mspd_rep_values": ";".join(f"{x:.9g}" for x in scores),
                "replay_loss_rep_values": ";".join(f"{x:.9g}" for x in losses),
                "replay_selected_tau_steps": ";".join(str(x) for x in selected_tau_steps),
                "tau_selector_raw": float(candidate["tau_selector_raw"]),
            }
        )
        _write_csv(replay_path, rows)
        if debug_rng_variants:
            variants = _optimizer_rng_debug_variants()
            if debug_rng_variant is not None:
                requested_variant = str(debug_rng_variant).strip()
                variants = [v for v in variants if str(v["variant"]) == requested_variant]
                if not variants:
                    valid = ", ".join(str(v["variant"]) for v in _optimizer_rng_debug_variants())
                    raise ValueError(f"Unknown --debug-rng-variant={requested_variant!r}. Valid variants: {valid}")
            for variant in variants:
                variant_name = str(variant["variant"])
                keys_variant = _optimizer_eval_keys(
                    seed=seed,
                    params_init=params_init,
                    best_iter=int(candidate["best_iter"]),
                    pop_idx=int(candidate["pop_idx"]),
                    pop_size=pop_size,
                    pop_batch=pop_batch,
                    bs=bs,
                    init_policy=str(variant.get("init_policy", "config")),
                    ask_split=bool(variant.get("ask_split", True)),
                    eval_split_mode=str(variant.get("eval_split_mode", "main")),
                    extra_splits_after_init=int(variant.get("extra_splits_after_init", 0)),
                    extra_splits_each_iter_before_eval=int(variant.get("extra_splits_each_iter_before_eval", 0)),
                )
                variant_scores: list[float] = []
                variant_losses: list[float] = []
                for key in keys_variant:
                    out = eval_one(key, params, tau_raw)
                    out_np = {name: np.asarray(jax.device_get(value)) for name, value in out.items()}
                    variant_losses.append(float(np.asarray(out_np["loss"]).reshape(-1)[0]))
                    variant_scores.append(float(np.asarray(out_np["score"]).reshape(-1)[0]))
                variant_mspd = float(np.mean(np.asarray(variant_scores, dtype=np.float64)))
                variant_loss = float(np.mean(np.asarray(variant_losses, dtype=np.float64)))
                variant_rows.append(
                    {
                        "run": run_dir.name,
                        "variant": variant_name,
                        "optimizer_seed": seed,
                        "best_iter": int(candidate["best_iter"]),
                        "pop_idx": int(candidate["pop_idx"]),
                        "pop_best_mspd": -float(candidate["pop_best_loss"]),
                        "variant_mspd_mean": variant_mspd,
                        "variant_mspd_minus_pop_best_mspd": float(
                            variant_mspd - (-float(candidate["pop_best_loss"]))
                        ),
                        "variant_loss_mean": variant_loss,
                        "variant_loss_minus_pop_best_loss": float(
                            variant_loss - float(candidate["pop_best_loss"])
                        ),
                        "variant_mspd_rep_values": ";".join(f"{x:.9g}" for x in variant_scores),
                    }
                )
                _write_csv(variants_path, variant_rows)

    _write_csv(replay_path, rows)
    if debug_rng_variants:
        _write_csv(variants_path, variant_rows)
    summary = {
        "status": "ok",
        "optimization_root": str(opt_root),
        "fallback_optimization_config": str(opt_cfg_path),
        "evaluator_config": str(first_config_path),
        "n_runs": len(run_dirs),
        "metric_summary": metric_summary(metric_cfg),
        "replay": str(replay_path),
        "rng_variants": str(variants_path) if debug_rng_variants else None,
    }
    write_json(summary_path, to_plain(summary))
    log_event(f"Flow-Lenia train objective replay done replay={replay_path}", component="train-replay")
    return summary


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
    run_name: str | None,
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
    run_dirs = _run_dirs(opt_root, max_runs, run_name=run_name)
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
    parser.add_argument("--run-name", default=None, help="single run directory name, for example run_008")
    parser.add_argument("--n-reps", type=int, default=None, help="fresh stochastic reps per checkpoint; default=optimization.bs")
    parser.add_argument("--seed-base", type=int, default=12345000)
    parser.add_argument(
        "--replay-optimizer-best",
        action="store_true",
        help="re-evaluate the best pop_traj candidate with the exact RNG keys used during optimization",
    )
    parser.add_argument(
        "--debug-rng-variants",
        action="store_true",
        help="with --replay-optimizer-best, also evaluate several RNG reconstruction variants for diagnosis",
    )
    parser.add_argument(
        "--debug-rng-variant",
        default=None,
        help="with --debug-rng-variants, evaluate only one named RNG variant",
    )
    args = parser.parse_args(argv)
    if args.replay_optimizer_best:
        result = _run_replay(
            args.config,
            optimization_config=args.optimization_config,
            output_dir=args.output_dir,
            force=args.force,
            max_runs=args.max_runs,
            run_name=args.run_name,
            debug_rng_variants=args.debug_rng_variants,
            debug_rng_variant=args.debug_rng_variant,
        )
    else:
        result = run(
            args.config,
            optimization_config=args.optimization_config,
            output_dir=args.output_dir,
            force=args.force,
            max_runs=args.max_runs,
            run_name=args.run_name,
            n_reps=args.n_reps,
            seed_base=args.seed_base,
        )
    print(to_plain(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
