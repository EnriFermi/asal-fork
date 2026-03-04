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

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config


print(jax.devices())
print(jax.default_backend())


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

        if int(args.rollout_steps) % int(args.time_sampling) != 0:
            raise ValueError(
                "rollout_steps must be divisible by time_sampling for lagrangian sampling."
            )
        chunk_steps = int(args.rollout_steps) // int(args.time_sampling)

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
                split(k_scan, int(args.time_sampling)),
            )
            return xy_seq  # (time_sampling, N_particles, 2)

        def calc_loss(rng, params):
            rng_roll, rng_metric = split(rng)
            xy_seq = rollout_lagrangian_xy(rng_roll, params)
            return metric_loss_fn(rng_metric, xy_seq)

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
            num_dims=substrate.n_params,
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
        best_loss_traj = []
        pop_params_traj = []
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

            best_params_traj.append(np.array(es_state.best_member))
            best_loss_traj.append(float(es_state.best_fitness))
            pop_params_traj.append(np.array(params_full))
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

            palette_stats = util.flow_lenia_palette_stats(np.array(params_full[best_idx]), substrate)
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
                params_np = np.array(params_full)
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
                best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
                util.save_pkl(args.save_dir, "best", best)
                if len(best_params_traj) > 0:
                    traj = dict(
                        params=np.stack(best_params_traj, axis=0),
                        loss=np.array(best_loss_traj),
                    )
                    util.save_pkl(args.save_dir, "best_traj", traj)
                if len(pop_params_traj) > 0:
                    pop_traj = dict(
                        params=np.stack(pop_params_traj, axis=0),
                        loss=np.stack(pop_loss_traj, axis=0),
                    )
                    util.save_pkl(args.save_dir, "pop_traj", pop_traj)
                if len(palette_traj) > 0:
                    util.save_pkl(args.save_dir, "palette_traj", palette_traj)

    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
