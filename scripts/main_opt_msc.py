import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

import evosax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.random import split
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import asal_metrics
import foundation_models
import substrates
import util
from clip_deltah_msc_metric import (
    make_metric_loss_fn,
    metric_summary,
    resolve_metric_config,
    tau_selection_from_latent,
)
from rollout import rollout_simulation


print(jax.devices())
print(jax.default_backend())


def _patch_wandb_pandas_check() -> None:
    """
    Work around environments where pandas cannot be imported due to numpy ABI mismatch.
    wandb checks every logged value via util.is_pandas_data_frame, which may crash.
    """
    try:
        import wandb.util as wandb_util
    except Exception:
        return
    orig = getattr(wandb_util, "is_pandas_data_frame", None)
    if orig is None:
        return

    def _safe_is_pandas_data_frame(val):
        try:
            return orig(val)
        except Exception:
            return False

    wandb_util.is_pandas_data_frame = _safe_is_pandas_data_frame


_patch_wandb_pandas_check()


def _to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.array(x), tree)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _canonicalize_params_init(name):
    if name is None:
        return "strategy_default"
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "strategy_default": "strategy_default",
        "optimizer_default": "strategy_default",
        "default": "strategy_default",
        "substrate_default": "substrate_default",
        "default_params": "substrate_default",
        "smart": "substrate_default",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown params_init {name!r}. Use 'strategy_default' or 'substrate_default'."
        )
    return aliases[normalized]


def _replace_state_fields(state, **updates):
    if hasattr(state, "replace"):
        return state.replace(**updates)
    if hasattr(state, "_replace"):
        return state._replace(**updates)
    for key, value in updates.items():
        setattr(state, key, value)
    return state


def _initialize_strategy_with_mean(strategy, rng_init, es_params, init_mean):
    try:
        return strategy.initialize(rng_init, es_params, init_mean=init_mean)
    except TypeError:
        state = strategy.initialize(rng_init, es_params)
        updates = {}
        if hasattr(state, "mean"):
            updates["mean"] = jnp.asarray(init_mean)
        if hasattr(state, "best_member"):
            updates["best_member"] = jnp.asarray(init_mean)
        if not updates:
            raise RuntimeError(
                "Could not set a custom optimizer initialization mean on this evosax strategy/state."
            )
        return _replace_state_fields(state, **updates)


def _build_candidate_init_mean(
    *,
    substrate,
    rng_mean,
    optimize_tau: bool,
) -> jax.Array:
    init_mean = jnp.asarray(substrate.default_params(rng_mean), dtype=jnp.float32)
    if not optimize_tau:
        return init_mean
    # raw tau latent = 0 -> sigmoid(0)=0.5, i.e. the middle of the tau grid
    tau0 = jnp.zeros((1,), dtype=init_mean.dtype)
    return jnp.concatenate((init_mean, tau0), axis=0)


def _load_resume_state(save_dir):
    if save_dir is None:
        return None
    path = os.path.join(save_dir, "resume_state.pkl")
    if not os.path.exists(path):
        return None
    return util.load_pkl(save_dir, "resume_state")


def _save_resume_state(
    save_dir,
    *,
    next_iter,
    rng,
    es_state,
    pop_size,
    candidate_dims,
    substrate_param_dims,
    optimize_tau,
    params_init,
    data,
    best_params_traj,
    best_tau_traj,
    best_loss_traj,
    pop_params_traj,
    pop_tau_traj,
    pop_loss_traj,
    palette_traj,
):
    if save_dir is None:
        return
    payload = dict(
        version=1,
        next_iter=int(next_iter),
        pop_size=int(pop_size),
        candidate_dims=int(candidate_dims),
        substrate_param_dims=int(substrate_param_dims),
        optimize_tau=bool(optimize_tau),
        params_init=str(params_init),
        rng=np.array(rng),
        es_state=_to_numpy_tree(es_state),
        data=[] if len(data) == 0 else _to_numpy_tree(data),
        best_params_traj=[np.array(x) for x in best_params_traj],
        best_tau_traj=list(best_tau_traj),
        best_loss_traj=np.asarray(best_loss_traj, dtype=np.float32),
        pop_params_traj=[np.array(x) for x in pop_params_traj],
        pop_tau_traj=list(pop_tau_traj),
        pop_loss_traj=[np.array(x) for x in pop_loss_traj],
        palette_traj=list(palette_traj),
    )
    util.save_pkl(save_dir, "resume_state", payload)


def _restore_resume_state(
    checkpoint,
    *,
    pop_size,
    candidate_dims,
    substrate_param_dims,
    optimize_tau,
    params_init,
):
    if checkpoint is None:
        return None
    ckpt_pop_size = int(checkpoint.get("pop_size"))
    if ckpt_pop_size != int(pop_size):
        raise ValueError(
            f"Resume checkpoint pop_size mismatch: checkpoint={ckpt_pop_size}, current={int(pop_size)}."
        )
    ckpt_candidate_dims = int(checkpoint.get("candidate_dims"))
    if ckpt_candidate_dims != int(candidate_dims):
        raise ValueError(
            f"Resume checkpoint candidate_dims mismatch: checkpoint={ckpt_candidate_dims}, current={int(candidate_dims)}."
        )
    ckpt_substrate_dims = int(checkpoint.get("substrate_param_dims"))
    if ckpt_substrate_dims != int(substrate_param_dims):
        raise ValueError(
            f"Resume checkpoint substrate_param_dims mismatch: checkpoint={ckpt_substrate_dims}, "
            f"current={int(substrate_param_dims)}."
        )
    ckpt_optimize_tau = bool(checkpoint.get("optimize_tau", False))
    if ckpt_optimize_tau != bool(optimize_tau):
        raise ValueError(
            f"Resume checkpoint optimize_tau mismatch: checkpoint={ckpt_optimize_tau}, current={bool(optimize_tau)}."
        )
    ckpt_params_init = str(checkpoint.get("params_init", "strategy_default"))
    if ckpt_params_init != str(params_init):
        raise ValueError(
            f"Resume checkpoint params_init mismatch: checkpoint={ckpt_params_init!r}, current={params_init!r}."
        )
    return dict(
        next_iter=int(checkpoint.get("next_iter", 0)),
        rng=jnp.asarray(checkpoint["rng"]),
        es_state=_to_jax_tree(checkpoint["es_state"]),
        data=list(checkpoint.get("data", [])),
        best_params_traj=[np.array(x) for x in checkpoint.get("best_params_traj", [])],
        best_tau_traj=list(checkpoint.get("best_tau_traj", [])),
        best_loss_traj=[float(x) for x in np.asarray(checkpoint.get("best_loss_traj", []))],
        pop_params_traj=[np.array(x) for x in checkpoint.get("pop_params_traj", [])],
        pop_tau_traj=list(checkpoint.get("pop_tau_traj", [])),
        pop_loss_traj=[np.array(x) for x in checkpoint.get("pop_loss_traj", [])],
        palette_traj=list(checkpoint.get("palette_traj", [])),
    )


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


def _normalize_metric_trajectory_source(source_raw) -> str:
    normalized = str(source_raw).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "lagrangian": "lagrangian",
        "state_x": "state_x",
        "direct": "state_x",
        "positions": "state_x",
        "position": "state_x",
        "direct_x": "state_x",
    }
    if normalized not in aliases:
        raise ValueError(
            "metric_trajectory_source must be one of "
            "['auto', 'lagrangian', 'state_x'], "
            f"got {source_raw!r}."
        )
    return aliases[normalized]


def _infer_metric_trajectory_source(args, substrate) -> tuple[str, dict | None]:
    requested = _normalize_metric_trajectory_source(getattr(args, "metric_trajectory_source", "auto"))
    sample_info = None
    if requested == "lagrangian":
        return requested, sample_info
    if requested == "state_x":
        params0 = substrate.default_params(jax.random.PRNGKey(0))
        state0 = substrate.init_state(jax.random.PRNGKey(1), params0)
        if not isinstance(state0, dict) or "x" not in state0:
            raise ValueError(
                "metric_trajectory_source='state_x' requires substrate.init_state(...) "
                "to return a dict containing key 'x'."
            )
        sample_info = dict(tracked_entities=int(state0["x"].shape[0]))
        return requested, sample_info

    if hasattr(substrate, "RT"):
        return "lagrangian", sample_info

    params0 = substrate.default_params(jax.random.PRNGKey(0))
    state0 = substrate.init_state(jax.random.PRNGKey(1), params0)
    if isinstance(state0, dict) and "x" in state0:
        sample_info = dict(tracked_entities=int(state0["x"].shape[0]))
        return "state_x", sample_info

    raise ValueError(
        "Could not infer metric trajectory source automatically. "
        "Set metric_trajectory_source explicitly to 'lagrangian' or 'state_x'."
    )


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/main_opt_msc.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    return cfg, flat


def main(cfg, args):
    wandb_project = str(getattr(args, "wandb_project", "asal"))
    run = wandb.init(project=wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
    try:
        base_substrate = substrates.create_substrate(
            args.substrate,
            **util.substrate_kwargs_from_args(args),
        )
        if hasattr(base_substrate, "debug_return_F"):
            base_substrate.debug_return_F = True
        substrate = substrates.FlattenSubstrateParameters(base_substrate)

        if args.rollout_steps is None:
            args.rollout_steps = substrate.rollout_steps

        # Auto-fill periodic settings for trajectory displacement unwrapping.
        metric_space_defaults = util.metric_periodic_space_defaults(base_substrate)
        if (not hasattr(args, "metric_periodic")) or (getattr(args, "metric_periodic", None) is None):
            args.metric_periodic = bool(metric_space_defaults["periodic"])
        if (not hasattr(args, "metric_domain_y")) or (getattr(args, "metric_domain_y", None) is None):
            args.metric_domain_y = float(metric_space_defaults["domain_y"])
        if (not hasattr(args, "metric_domain_x")) or (getattr(args, "metric_domain_x", None) is None):
            args.metric_domain_x = float(metric_space_defaults["domain_x"])

        metric_cfg = resolve_metric_config(args)
        metric_info = metric_summary(metric_cfg)
        print("Resolved metric config:", metric_info)
        for k, v in metric_info.items():
            if isinstance(v, (list, tuple, dict)):
                run.summary[f"metric_cfg/{k}"] = str(v)
            else:
                run.summary[f"metric_cfg/{k}"] = v
        metric_loss_fn = make_metric_loss_fn(metric_cfg)
        optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
        tau_extra_dims = 1 if optimize_tau else 0
        run.summary["metric_cfg/trainable_tau"] = bool(optimize_tau)
        if optimize_tau:
            run.summary["metric_cfg/tau_grid_steps"] = str(metric_cfg.get("tau_steps_list", []))

        chunk_steps = int(metric_cfg["sample_every_steps"])
        time_sampling = int(metric_cfg["time_sampling"])
        substrate_param_dims = int(substrate.n_params)
        params_init = _canonicalize_params_init(getattr(args, "params_init", "strategy_default"))
        run.summary["optimizer/params_init"] = params_init
        trajectory_source, trajectory_sample_info = _infer_metric_trajectory_source(args, substrate)
        run.summary["metric_cfg/trajectory_source"] = str(trajectory_source)
        if trajectory_sample_info is not None:
            for k, v in trajectory_sample_info.items():
                run.summary[f"metric_cfg/{k}"] = v

        def split_candidate_params(params_full):
            params_sub = params_full[:substrate_param_dims]
            tau_selector = params_full[substrate_param_dims] if optimize_tau else None
            return params_sub, tau_selector

        def split_population_params_np(params_full_np: np.ndarray):
            params_sub_np = params_full_np[:, :substrate_param_dims]
            tau_latent_np = params_full_np[:, substrate_param_dims] if optimize_tau else None
            return params_sub_np, tau_latent_np

        def tau_info_from_latent(raw_tau):
            return tau_selection_from_latent(metric_cfg, raw_tau)

        # Optional CLIP-based evolution loss logging (does NOT affect optimization target).
        log_clip_evolution = bool(getattr(args, "log_clip_evolution", True))
        clip_loss_from_z = None
        rollout_clip = None
        if log_clip_evolution:
            prompts_raw = getattr(args, "prompts", "a biological cell;two biological cells")
            if isinstance(prompts_raw, str):
                prompts = [p for p in prompts_raw.split(";") if p]
            else:
                prompts = OmegaConf.to_container(prompts_raw, resolve=True)
            if len(prompts) == 0:
                raise ValueError("prompts must not be empty when log_clip_evolution=true.")

            clip_time_sampling = int(getattr(args, "clip_time_sampling", min(16, time_sampling)))
            if clip_time_sampling < len(prompts):
                clip_time_sampling = len(prompts)
            if int(args.rollout_steps) % clip_time_sampling != 0:
                raise ValueError(
                    "rollout_steps must be divisible by clip_time_sampling. "
                    f"Got rollout_steps={int(args.rollout_steps)}, clip_time_sampling={clip_time_sampling}."
                )

            fm = foundation_models.create_foundation_model(getattr(args, "foundation_model", "clip"))
            z_txt = fm.embed_txt(prompts)

            coef_prompt = float(getattr(args, "coef_prompt", 0.0))
            coef_softmax = float(getattr(args, "coef_softmax", 0.0))
            coef_oe = float(getattr(args, "coef_oe", 1.0))
            coef_smooth = float(getattr(args, "coef_smooth", 0.2))

            def clip_loss_from_z(z):
                loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
                loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
                loss_oe = asal_metrics.calc_open_endedness_score(z)
                loss_smooth = asal_metrics.calc_gradient_score(z)
                loss_total = (
                    loss_prompt * coef_prompt
                    + loss_softmax * coef_softmax
                    + loss_oe * coef_oe
                    + loss_smooth * coef_smooth
                )
                return dict(
                    clip_evolution_loss=loss_total,
                    clip_loss_prompt=loss_prompt,
                    clip_loss_softmax=loss_softmax,
                    clip_loss_oe=loss_oe,
                    clip_loss_smooth=loss_smooth,
                )

            rollout_clip = jax.jit(
                lambda rng, params: rollout_simulation(
                    rng=rng,
                    params=params,
                    s0=None,
                    substrate=substrate,
                    fm=fm,
                    rollout_steps=int(args.rollout_steps),
                    time_sampling=(clip_time_sampling, True),
                    img_size=224,
                    return_state=False,
                )
            )
            run.summary["clip_logging/enabled"] = True
            run.summary["clip_logging/foundation_model"] = str(getattr(args, "foundation_model", "clip"))
            run.summary["clip_logging/time_sampling"] = int(clip_time_sampling)
            run.summary["clip_logging/prompts"] = str(prompts)
        else:
            run.summary["clip_logging/enabled"] = False

        if trajectory_source == "lagrangian":
            lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
            lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
            lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
            lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
            lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
            lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
            lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))
            run.summary["metric_cfg/lagrangian_n_particles"] = int(lag_n_particles)

            def rollout_metric_xy(rng, params):
                k_state, k_pts, k_ch, k_scan = jax.random.split(rng, 4)
                s0 = substrate.init_state(k_state, params)
                if "F" not in s0:
                    raise ValueError(
                        "State does not contain flow field F. "
                        "For FlowLenia set debug_return_F=true before optimization."
                    )
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
                return xy_seq  # (time_sampling, N_particles, 2)
        elif trajectory_source == "state_x":
            def rollout_metric_xy(rng, params):
                k_state, k_scan = jax.random.split(rng, 2)
                s0 = substrate.init_state(k_state, params)
                if "x" not in s0:
                    raise ValueError(
                        "State does not contain key 'x'. "
                        "Set metric_trajectory_source='lagrangian' or use a substrate with explicit positions."
                    )

                def step_fn(state, key_step):
                    state_next = substrate.step_state(key_step, state, params)
                    return state_next, None

                def chunk_fn(state, key_chunk):
                    state_next, _ = jax.lax.scan(step_fn, state, split(key_chunk, chunk_steps))
                    return state_next, state_next["x"]

                _, xy_seq = jax.lax.scan(
                    chunk_fn,
                    s0,
                    split(k_scan, time_sampling),
                )
                return xy_seq
        else:
            raise ValueError(f"Unhandled metric trajectory source {trajectory_source!r}.")

        def calc_loss(rng, params_full):
            params, tau_selector = split_candidate_params(params_full)
            if log_clip_evolution:
                rng_roll, rng_metric, rng_clip = split(rng, 3)
            else:
                rng_roll, rng_metric = split(rng)
            xy_seq = rollout_metric_xy(rng_roll, params)
            if optimize_tau:
                msc_loss, msc_dict = metric_loss_fn(rng_metric, xy_seq, tau_selector=tau_selector)
            else:
                msc_loss, msc_dict = metric_loss_fn(rng_metric, xy_seq)
            if not log_clip_evolution:
                return msc_loss, msc_dict
            clip_out = rollout_clip(rng_clip, params)
            z_clip = clip_out["z"]
            clip_dict = clip_loss_from_z(z_clip)
            merged = dict(msc_dict, **clip_dict)
            return msc_loss, merged

        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))

        @jax.jit
        def eval_chunk(rng, params_chunk):
            rng, _rng = split(rng)
            loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params_chunk)
            loss = loss.mean(axis=1)
            loss_dict = jax.tree.map(lambda x: x.mean(axis=1), loss_dict)
            return rng, loss, loss_dict

        strategy = evosax.Sep_CMA_ES(
            popsize=args.pop_size,
            num_dims=substrate_param_dims + tau_extra_dims,
            sigma_init=args.sigma,
        )
        es_params = strategy.default_params

        pop_batch = int(getattr(args, "pop_batch", args.pop_size))
        if pop_batch < 1:
            raise ValueError(f"pop_batch must be >= 1, got {pop_batch}.")

        pca_every = int(getattr(args, "pca_every", 25))
        pca_history = int(getattr(args, "pca_history", 100))
        save_every = getattr(args, "save_every", None)
        if save_every is None:
            save_interval = max(1, int(args.n_iters) // 10)
        else:
            save_interval = int(save_every)
            if save_interval < 1:
                raise ValueError(f"save_every must be >= 1, got {save_interval}.")
        resume_enabled = bool(getattr(args, "resume", False))
        run.summary["logging/save_interval"] = int(save_interval)

        data = []
        best_params_traj = []
        best_tau_traj = []
        best_loss_traj = []
        pop_params_traj = []
        pop_tau_traj = []
        pop_loss_traj = []
        palette_traj = []
        start_iter = 0
        resumed = False
        rng = jax.random.PRNGKey(args.seed)
        candidate_dims = int(substrate_param_dims + tau_extra_dims)
        if resume_enabled:
            resume_state = _load_resume_state(args.save_dir)
            if resume_state is not None:
                restored = _restore_resume_state(
                    resume_state,
                    pop_size=args.pop_size,
                    candidate_dims=candidate_dims,
                    substrate_param_dims=substrate_param_dims,
                    optimize_tau=optimize_tau,
                    params_init=params_init,
                )
                start_iter = restored["next_iter"]
                rng = restored["rng"]
                es_state = restored["es_state"]
                data = restored["data"]
                best_params_traj = restored["best_params_traj"]
                best_tau_traj = restored["best_tau_traj"]
                best_loss_traj = restored["best_loss_traj"]
                pop_params_traj = restored["pop_params_traj"]
                pop_tau_traj = restored["pop_tau_traj"]
                pop_loss_traj = restored["pop_loss_traj"]
                palette_traj = restored["palette_traj"]
                resumed = True
                print(f"Resuming optimization from iter {start_iter} using {args.save_dir}/resume_state.pkl")
        if not resumed:
            if params_init == "strategy_default":
                rng, _rng = split(rng)
                es_state = strategy.initialize(_rng, es_params)
            elif params_init == "substrate_default":
                rng, rng_mean, rng_init = jax.random.split(rng, 3)
                init_mean = _build_candidate_init_mean(
                    substrate=substrate,
                    rng_mean=rng_mean,
                    optimize_tau=optimize_tau,
                )
                es_state = _initialize_strategy_with_mean(strategy, rng_init, es_params, init_mean)
            else:
                raise ValueError(f"Unhandled params_init {params_init!r}.")

        run.summary["resume/enabled"] = bool(resume_enabled)
        run.summary["resume/loaded"] = bool(resumed)
        run.summary["resume/start_iter"] = int(start_iter)
        if args.save_dir is not None:
            run.summary["resume/checkpoint_path"] = os.path.join(args.save_dir, "resume_state.pkl")
        if start_iter >= int(args.n_iters):
            print(
                f"Run already completed for n_iters={int(args.n_iters)} "
                f"(resume checkpoint next_iter={int(start_iter)}). Nothing to do."
            )
            return

        pbar = tqdm(range(start_iter, args.n_iters), initial=start_iter, total=args.n_iters)

        for i_iter in pbar:
            rng, _rng = split(rng)
            params_full, es_state = strategy.ask(_rng, es_state, es_params)

            loss_chunks = []
            loss_dict_chunks = []
            rng_eval = rng
            for start in range(0, args.pop_size, pop_batch):
                end = min(args.pop_size, start + pop_batch)
                rng_eval, loss_chunk, loss_dict_chunk = eval_chunk(rng_eval, params_full[start:end])
                loss_chunks.append(loss_chunk)
                loss_dict_chunks.append(loss_dict_chunk)
            rng = rng_eval

            loss_all = jnp.concatenate(loss_chunks, axis=0)
            loss_dict_all = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *loss_dict_chunks)

            es_state = strategy.tell(params_full, loss_all, es_state, es_params)

            best_member_full_np = np.array(es_state.best_member)
            best_member_np = np.array(best_member_full_np[:substrate_param_dims])
            best_params_traj.append(best_member_np)
            if optimize_tau:
                best_tau_traj.append(tau_info_from_latent(best_member_full_np[substrate_param_dims]))
            best_loss_traj.append(float(es_state.best_fitness))
            params_full_np = np.array(params_full)
            pop_params_np, pop_tau_latent_np = split_population_params_np(params_full_np)
            pop_params_traj.append(pop_params_np)
            if optimize_tau:
                tau_idx = []
                tau_steps = []
                tau_frames = []
                for raw_tau in np.asarray(pop_tau_latent_np):
                    info = tau_info_from_latent(raw_tau)
                    tau_idx.append(info["tau_idx"])
                    tau_steps.append(info["tau_steps"])
                    tau_frames.append(info["tau_frames"])
                pop_tau_traj.append(
                    dict(
                        latent=np.asarray(pop_tau_latent_np, dtype=np.float32),
                        idx=np.asarray(tau_idx, dtype=np.int32),
                        steps=np.asarray(tau_steps, dtype=np.int32),
                        frames=np.asarray(tau_frames, dtype=np.int32),
                    )
                )
            pop_loss_traj.append(np.array(loss_all))

            loss_np = np.array(loss_all)
            loss_mean = float(loss_np.mean())
            loss_var = float(loss_np.var())
            best_idx = int(np.argmin(loss_np))

            pca_img = None
            if pca_every > 0 and (i_iter % pca_every == 0) and len(pop_params_traj) > 1:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                    hist = pop_params_traj[-pca_history:]
                    pop_hist = np.stack(hist, axis=0)
                    T_hist, P_hist, D_hist = pop_hist.shape
                    X = pop_hist.reshape(T_hist * P_hist, D_hist)
                    times = np.repeat(np.arange(T_hist), P_hist)

                    X_centered = X - X.mean(axis=0, keepdims=True)
                    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    pcs = X_centered @ Vt[:2].T

                    fig = plt.figure(figsize=(6, 5))
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(pcs[:, 0], pcs[:, 1], times, c=times, cmap="viridis", s=3)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("iter")
                    ax.set_title(f"Population PCA trajectory up to iter {i_iter}")
                    pca_img = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"PCA population logging failed at iter {i_iter}: {e}")

            log_dict = {
                "iter": i_iter,
                "loss_pop_mean": loss_mean,
                "loss_pop_var": loss_var,
                "best_loss": float(es_state.best_fitness),
                "best_loss_raw": float(loss_np[best_idx]),
            }
            for k, v in loss_dict_all.items():
                v_np = np.array(v)
                log_dict[f"metric/{k}_pop_mean"] = float(v_np.mean())
                log_dict[f"metric/{k}_pop_var"] = float(v_np.var())
            if "clip_evolution_loss" in loss_dict_all:
                clip_np = np.array(loss_dict_all["clip_evolution_loss"])
                if clip_np.size > 0:
                    log_dict["metric/clip_evolution_loss_pop_best"] = float(clip_np.min())
                if clip_np.shape == loss_np.shape and clip_np.size >= 2:
                    a = loss_np.astype(np.float64)
                    b = clip_np.astype(np.float64)
                    da = a - a.mean()
                    db = b - b.mean()
                    denom = np.sqrt((da * da).sum() * (db * db).sum())
                    corr = np.nan if denom <= 0 else float((da * db).sum() / denom)
                    log_dict["metric/corr_msc_loss_vs_clip_evolution"] = corr

            if optimize_tau:
                tau_best_pop = tau_info_from_latent(np.asarray(pop_tau_latent_np)[best_idx])
                log_dict["metric/tau_trainable_idx_pop_best"] = float(tau_best_pop["tau_idx"])
                log_dict["metric/tau_trainable_steps_pop_best"] = float(tau_best_pop["tau_steps"])
                log_dict["metric/tau_trainable_frames_pop_best"] = float(tau_best_pop["tau_frames"])
                log_dict["metric/tau_trainable_raw_pop_best"] = float(tau_best_pop["tau_selector_raw"])

            palette_stats = util.flow_lenia_palette_stats(pop_params_np[best_idx], substrate)
            if palette_stats is not None:
                try:
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(6, 2))
                    ax = fig.add_subplot(111)
                    im = ax.imshow(palette_stats["w_soft"], aspect="auto", cmap="viridis")
                    ax.set_xlabel("kernel")
                    ax.set_ylabel("RGB")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    log_dict["pcolor_palette"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"Palette logging failed: {e}")
                ent = palette_stats["entropy"]
                log_dict["pcolor_entropy_mean"] = float(ent.mean())
                log_dict["pcolor_entropy_r"] = float(ent[0])
                log_dict["pcolor_entropy_g"] = float(ent[1])
                log_dict["pcolor_entropy_b"] = float(ent[2])
                base = substrate.substrate if hasattr(substrate, "substrate") else substrate
                n_dyn = int(base.base_dyn_raw.size)
                k = int(base.k)
                params_np = pop_params_np
                w_raw_pop = params_np[:, n_dyn:n_dyn + 3 * k].reshape(params_np.shape[0], 3, k)
                w_raw_pop = w_raw_pop - w_raw_pop.max(axis=2, keepdims=True)
                w_soft_pop = np.exp(w_raw_pop)
                w_soft_pop = w_soft_pop / w_soft_pop.sum(axis=2, keepdims=True)
                eps = 1e-8
                ent_pop = -np.sum(w_soft_pop * np.log(w_soft_pop + eps), axis=2)
                log_dict["pcolor_entropy_pop_mean"] = float(ent_pop.mean())
                log_dict["pcolor_wraw_pop_std"] = float(w_raw_pop.std())
                log_dict["pcolor_wsoft_pop_std"] = float(w_soft_pop.std())

            if pca_img is not None:
                log_dict["pop_pca_traj_3d"] = pca_img
            run.log(log_dict)

            data.append(dict(best_loss=es_state.best_fitness, loss=loss_all, loss_dict=loss_dict_all))
            if palette_stats is not None:
                palette_traj.append(dict(iter=i_iter, **palette_stats))
            pbar.set_postfix(best_loss=es_state.best_fitness.item())

            if args.save_dir is not None and (i_iter % save_interval == 0 or i_iter == args.n_iters - 1):
                data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
                util.save_pkl(args.save_dir, "data", data_save)
                best = (best_member_np, np.array(es_state.best_fitness))
                util.save_pkl(args.save_dir, "best", best)
                if optimize_tau:
                    util.save_json(args.save_dir, "best_tau", best_tau_traj[-1])
                if len(best_params_traj) > 0:
                    traj = dict(
                        params=np.stack(best_params_traj, axis=0),
                        loss=np.array(best_loss_traj),
                    )
                    if optimize_tau and len(best_tau_traj) > 0:
                        traj["tau_idx"] = np.asarray([x["tau_idx"] for x in best_tau_traj], dtype=np.int32)
                        traj["tau_steps"] = np.asarray([x["tau_steps"] for x in best_tau_traj], dtype=np.int32)
                        traj["tau_frames"] = np.asarray([x["tau_frames"] for x in best_tau_traj], dtype=np.int32)
                        traj["tau_selector_raw"] = np.asarray(
                            [x["tau_selector_raw"] for x in best_tau_traj], dtype=np.float32
                        )
                    util.save_pkl(args.save_dir, "best_traj", traj)
                if len(pop_params_traj) > 0:
                    pop_traj = dict(
                        params=np.stack(pop_params_traj, axis=0),
                        loss=np.stack(pop_loss_traj, axis=0),
                    )
                    if optimize_tau and len(pop_tau_traj) > 0:
                        pop_traj["tau_selector_raw"] = np.stack([x["latent"] for x in pop_tau_traj], axis=0)
                        pop_traj["tau_idx"] = np.stack([x["idx"] for x in pop_tau_traj], axis=0)
                        pop_traj["tau_steps"] = np.stack([x["steps"] for x in pop_tau_traj], axis=0)
                        pop_traj["tau_frames"] = np.stack([x["frames"] for x in pop_tau_traj], axis=0)
                    util.save_pkl(args.save_dir, "pop_traj", pop_traj)
                if len(palette_traj) > 0:
                    util.save_pkl(args.save_dir, "palette_traj", palette_traj)
                _save_resume_state(
                    args.save_dir,
                    next_iter=i_iter + 1,
                    rng=rng,
                    es_state=es_state,
                    pop_size=args.pop_size,
                    candidate_dims=candidate_dims,
                    substrate_param_dims=substrate_param_dims,
                    optimize_tau=optimize_tau,
                    params_init=params_init,
                    data=data,
                    best_params_traj=best_params_traj,
                    best_tau_traj=best_tau_traj,
                    best_loss_traj=best_loss_traj,
                    pop_params_traj=pop_params_traj,
                    pop_tau_traj=pop_tau_traj,
                    pop_loss_traj=pop_loss_traj,
                    palette_traj=palette_traj,
                )

    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
