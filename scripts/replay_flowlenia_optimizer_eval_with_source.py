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


COMPONENT_KEYS = (
    "score",
    "msc",
    "amp",
    "delta_h_mean",
    "delta_h_std",
    "delta_h_min",
    "delta_h_max",
    "h_real_mean",
    "h_null_mean",
    "h_real_minus_null_mean",
    "h_real_over_null_mean",
    "h_delta_over_real_mean",
    "score_tau_max",
    "score_tau_mean",
    "score_tau_min",
    "msc_tau_max",
    "msc_tau_mean",
    "msc_tau_min",
    "tau_selected_idx",
    "tau_best_steps",
    "dx_norm_mean",
    "speed_norm_mean",
    "position_std_mean",
)


def _activate_source_root(source_root: Path) -> None:
    source_root = source_root.resolve()
    source_scripts = source_root / "scripts"
    this_root = Path(__file__).resolve().parent.parent
    this_scripts = this_root / "scripts"
    remove = {str(this_root), str(this_scripts)}
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) not in remove]
    for path in (str(source_scripts), str(source_root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


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


def _flat_config(path: Path) -> SimpleNamespace:
    from omegaconf import OmegaConf

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


def _get_saved_components(run_dir: Path, best_iter: int, pop_idx: int) -> dict[str, float]:
    data = _load_pickle(run_dir / "data.pkl")
    loss_dict = data.get("loss_dict", {}) if isinstance(data, dict) else {}
    out: dict[str, float] = {}
    for key in COMPONENT_KEYS:
        if key not in loss_dict:
            continue
        arr = __import__("numpy").asarray(loss_dict[key])
        if arr.ndim >= 2 and best_iter < arr.shape[0] and pop_idx < arr.shape[1]:
            out[key] = float(arr[best_iter, pop_idx])
    return out


def _init_lagrangian_points_jax(A0, *, n_particles: int, init_mode: str, border: str, sigma: float, key):
    import jax
    import jax.numpy as jnp

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
        raise ValueError(f"Unknown metric_lagrangian_init_mode={init_mode!r}.")

    if border == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
        pts = jnp.stack((y, x), axis=-1)
    else:
        y = jnp.clip(pts[:, 0], float(sigma), float(sx - sigma))
        x = jnp.clip(pts[:, 1], float(sigma), float(sy - sigma))
        pts = jnp.stack((y, x), axis=-1)
    return pts.astype(jnp.float32)


def _build_source_context(flat: SimpleNamespace):
    import substrates
    import util
    from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config

    base_substrate = substrates.create_substrate(flat.substrate, **util.substrate_kwargs_from_args(flat))
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    if getattr(flat, "rollout_steps", None) is None:
        flat.rollout_steps = substrate.rollout_steps

    defaults = util.metric_periodic_space_defaults(base_substrate)
    if getattr(flat, "metric_periodic", None) is None:
        flat.metric_periodic = bool(defaults["periodic"])
    if getattr(flat, "metric_domain_y", None) is None:
        flat.metric_domain_y = float(defaults["domain_y"])
    if getattr(flat, "metric_domain_x", None) is None:
        flat.metric_domain_x = float(defaults["domain_x"])

    metric_cfg = resolve_metric_config(flat)
    metric_loss_fn = make_metric_loss_fn(metric_cfg, include_maps=False)
    return substrate, metric_cfg, metric_loss_fn


def _best_candidate_and_keys(run_dir: Path, flat: SimpleNamespace, substrate_param_dims: int, tau_extra_dims: int):
    import evosax
    import jax
    import jax.numpy as jnp
    import numpy as np

    pop = _load_pickle(run_dir / "pop_traj.pkl")
    pop_params = np.asarray(pop["params"], dtype=np.float32)
    pop_loss = np.asarray(pop["loss"], dtype=np.float32)
    pop_tau_raw = np.asarray(pop.get("tau_selector_raw", np.zeros(pop_loss.shape, dtype=np.float32)), dtype=np.float32)
    best_flat = int(np.nanargmin(pop_loss))
    best_iter, pop_idx = np.unravel_index(best_flat, pop_loss.shape)

    candidate_dims = int(substrate_param_dims + tau_extra_dims)
    strategy = evosax.Sep_CMA_ES(popsize=int(flat.pop_size), num_dims=candidate_dims, sigma_init=float(flat.sigma))
    es_params = strategy.default_params
    rng = jax.random.PRNGKey(int(flat.seed))
    params_init = str(getattr(flat, "params_init", "strategy_default")).strip().lower().replace("-", "_")
    if params_init not in {"strategy_default", "optimizer_default", "default"}:
        raise ValueError(f"Unsupported params_init for exact replay: {params_init!r}")
    rng, rng_init = jax.random.split(rng)
    es_state = strategy.initialize(rng_init, es_params)

    max_param_diff = 0.0
    max_tau_diff = 0.0
    keys = None
    pop_batch = int(getattr(flat, "pop_batch", int(flat.pop_size)))
    bs = int(getattr(flat, "bs", 1))
    for i_iter in range(pop_loss.shape[0]):
        rng, rng_ask = jax.random.split(rng)
        params_full, es_state = strategy.ask(rng_ask, es_state, es_params)
        params_full_np = np.asarray(jax.device_get(params_full), dtype=np.float32)
        max_param_diff = max(max_param_diff, float(np.max(np.abs(params_full_np[:, :substrate_param_dims] - pop_params[i_iter]))))
        if tau_extra_dims:
            max_tau_diff = max(max_tau_diff, float(np.max(np.abs(params_full_np[:, substrate_param_dims] - pop_tau_raw[i_iter]))))

        rng_eval = rng
        for start in range(0, int(flat.pop_size), pop_batch):
            end = min(int(flat.pop_size), start + pop_batch)
            rng_next, rng_metric_parent = jax.random.split(rng_eval)
            if i_iter == best_iter and start <= pop_idx < end:
                keys = jax.random.split(rng_metric_parent, bs)
            rng_eval = rng_next
        rng = rng_eval
        es_state = strategy.tell(params_full, jnp.asarray(pop_loss[i_iter]), es_state, es_params)

    if keys is None:
        raise RuntimeError("Could not reconstruct best evaluation keys.")
    return {
        "best_iter": int(best_iter),
        "pop_idx": int(pop_idx),
        "pop_best_loss": float(pop_loss[best_iter, pop_idx]),
        "params": pop_params[best_iter, pop_idx],
        "tau_raw": float(pop_tau_raw[best_iter, pop_idx]) if tau_extra_dims else 0.0,
        "keys": keys,
        "max_param_diff": max_param_diff,
        "max_tau_diff": max_tau_diff,
    }


def run(run_dir: Path, source_root: Path, output_csv: Path) -> dict[str, Any]:
    _activate_source_root(source_root)

    import jax
    import jax.numpy as jnp
    import numpy as np

    flat = _flat_config(run_dir / "optimization_config.yaml")
    substrate, metric_cfg, metric_loss_fn = _build_source_context(flat)
    tau_extra_dims = 1 if str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid" else 0
    substrate_param_dims = int(substrate.n_params)
    candidate = _best_candidate_and_keys(run_dir, flat, substrate_param_dims, tau_extra_dims)
    saved_components = _get_saved_components(run_dir, candidate["best_iter"], candidate["pop_idx"])

    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(flat, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(flat, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(flat, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(flat, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(flat, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(flat, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(flat, "metric_lagrangian_diffusion_scale", 1.0))
    log_clip_evolution = bool(getattr(flat, "log_clip_evolution", True))

    def rollout_metric_xy(rng, params):
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng, 4)
        s0 = substrate.init_state(k_state, params)
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

        (_, _, _), xy_seq = jax.lax.scan(chunk_fn, (s0, pts0, ch0), jax.random.split(k_scan, time_sampling))
        return xy_seq

    @jax.jit
    def eval_one(rng, params, tau_raw):
        if log_clip_evolution:
            rng_roll, rng_metric, _rng_clip = jax.random.split(rng, 3)
        else:
            rng_roll, rng_metric = jax.random.split(rng)
        xy = rollout_metric_xy(rng_roll, params)
        loss, info = metric_loss_fn(rng_metric, xy, tau_selector=tau_raw if tau_extra_dims else None)
        out = {"loss": loss}
        for key in COMPONENT_KEYS:
            if key in info:
                out[key] = info[key]
        return out

    params = jnp.asarray(np.asarray(candidate["params"], dtype=np.float32), dtype=jnp.float32)
    tau_raw = jnp.asarray(float(candidate["tau_raw"]), dtype=jnp.float32)
    values: dict[str, list[float]] = {key: [] for key in ("loss",) + COMPONENT_KEYS}
    for key in candidate["keys"]:
        out = eval_one(key, params, tau_raw)
        out_np = {k: np.asarray(jax.device_get(v)) for k, v in out.items()}
        for k, v in out_np.items():
            values.setdefault(k, []).append(float(np.asarray(v).reshape(-1)[0]))

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source_root": str(source_root),
        "best_iter": candidate["best_iter"],
        "pop_idx": candidate["pop_idx"],
        "pop_best_loss": candidate["pop_best_loss"],
        "pop_best_mspd": -candidate["pop_best_loss"],
        "pop_reconstruction_max_param_abs_diff": candidate["max_param_diff"],
        "pop_reconstruction_max_tau_abs_diff": candidate["max_tau_diff"],
        "metric_objective": str(metric_cfg.get("objective", "custom")),
        "metric_msc_term": str(metric_cfg.get("msc_term", "")),
        "metric_scale_normalization": str(metric_cfg.get("scale_normalization", "")),
        "metric_msc_floor": float(metric_cfg.get("msc_floor", 0.0)),
    }
    for key, vals in values.items():
        if not vals:
            continue
        replay = float(np.mean(np.asarray(vals, dtype=np.float64)))
        row[f"replay_{key}"] = replay
        row[f"replay_{key}_rep_values"] = ";".join(f"{x:.9g}" for x in vals)
        saved_key = key
        if key == "loss":
            saved = candidate["pop_best_loss"]
        elif saved_key in saved_components:
            saved = saved_components[saved_key]
        else:
            continue
        row[f"saved_{key}"] = float(saved)
        row[f"diff_{key}"] = float(replay - float(saved))

    _write_csv(output_csv, [row])
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one Flow-Lenia optimizer eval against an arbitrary source snapshot.")
    parser.add_argument("run_dir")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args(argv)

    row = run(Path(args.run_dir), Path(args.source_root), Path(args.output_csv))
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
