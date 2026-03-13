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


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/main_opt_msc.py <config.yaml>")
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
        if args.substrate == "lenia_flow":
            base_substrate = substrates.create_substrate(
                args.substrate,
                **util.flow_lenia_kwargs_from_args(args),
            )
        else:
            base_substrate = substrates.create_substrate(args.substrate)
        if hasattr(base_substrate, "debug_return_F"):
            base_substrate.debug_return_F = True
        substrate = substrates.FlattenSubstrateParameters(base_substrate)

        if args.rollout_steps is None:
            args.rollout_steps = substrate.rollout_steps

        # Auto-fill periodic settings for lagrangian displacement unwrapping.
        if (not hasattr(args, "metric_periodic")) or (getattr(args, "metric_periodic", None) is None):
            args.metric_periodic = str(getattr(substrate, "border", "wall")) == "torus"
        if (not hasattr(args, "metric_domain_y")) or (getattr(args, "metric_domain_y", None) is None):
            args.metric_domain_y = float(getattr(getattr(substrate, "cfg", None), "X", getattr(substrate, "grid_size", 0)))
        if (not hasattr(args, "metric_domain_x")) or (getattr(args, "metric_domain_x", None) is None):
            args.metric_domain_x = float(getattr(getattr(substrate, "cfg", None), "Y", getattr(substrate, "grid_size", 0)))

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

        lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
        lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
        lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
        lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
        lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
        lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
        lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))

        def rollout_lagrangian_xy(rng, params):
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

        def calc_loss(rng, params_full):
            params, tau_selector = split_candidate_params(params_full)
            if log_clip_evolution:
                rng_roll, rng_metric, rng_clip = split(rng, 3)
            else:
                rng_roll, rng_metric = split(rng)
            xy_seq = rollout_lagrangian_xy(rng_roll, params)
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

        rng = jax.random.PRNGKey(args.seed)
        strategy = evosax.Sep_CMA_ES(
            popsize=args.pop_size,
            num_dims=substrate_param_dims + tau_extra_dims,
            sigma_init=args.sigma,
        )
        es_params = strategy.default_params
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)

        pop_batch = int(getattr(args, "pop_batch", args.pop_size))
        if pop_batch < 1:
            raise ValueError(f"pop_batch must be >= 1, got {pop_batch}.")

        pca_every = int(getattr(args, "pca_every", 25))
        pca_history = int(getattr(args, "pca_history", 100))

        data = []
        best_params_traj = []
        best_tau_traj = []
        best_loss_traj = []
        pop_params_traj = []
        pop_tau_traj = []
        pop_loss_traj = []
        palette_traj = []
        pbar = tqdm(range(args.n_iters))

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

            if args.save_dir is not None and (i_iter % max(1, args.n_iters // 10) == 0 or i_iter == args.n_iters - 1):
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

    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
