import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from functools import partial

import evosax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.random import split
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import foundation_models
import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config
from rollout import rollout_simulation


print(jax.devices())
print(jax.default_backend())


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
        fm = foundation_models.create_foundation_model(args.foundation_model)
        if args.substrate == "lenia_flow":
            substrate = substrates.create_substrate(
                args.substrate,
                **util.flow_lenia_kwargs_from_args(args),
            )
        else:
            substrate = substrates.create_substrate(args.substrate)
        substrate = substrates.FlattenSubstrateParameters(substrate)

        if args.rollout_steps is None:
            args.rollout_steps = substrate.rollout_steps

        metric_cfg = resolve_metric_config(args)
        metric_info = metric_summary(metric_cfg)
        print("Resolved metric config:", metric_info)
        for k, v in metric_info.items():
            if isinstance(v, (list, tuple, dict)):
                run.summary[f"metric_cfg/{k}"] = str(v)
            else:
                run.summary[f"metric_cfg/{k}"] = v
        metric_loss_fn = make_metric_loss_fn(metric_cfg)

        rollout_fn = partial(
            rollout_simulation,
            substrate=substrate,
            fm=fm,
            rollout_steps=args.rollout_steps,
            time_sampling=(args.time_sampling, True),
            img_size=224,
            return_state=False,
        )

        def calc_loss(rng, params):
            rng_roll, rng_metric = split(rng)
            out = rollout_fn(rng=rng_roll, params=params)
            z = out["z"]  # (time_sampling, D)
            return metric_loss_fn(rng_metric, z)

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
