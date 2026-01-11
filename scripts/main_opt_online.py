import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import wandb
import util
from omegaconf import OmegaConf


print(jax.devices())
print(jax.default_backend())

def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/main_opt_online.py <config.yaml>")
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def main(cfg, args):
    run = wandb.init(project="asal", config=OmegaConf.to_container(cfg, resolve=True))
    try:
        prompts = args.prompts
        if isinstance(prompts, str):
            prompts = [p for p in prompts.split(";") if p]
        else:
            prompts = OmegaConf.to_container(prompts, resolve=True)
        if args.time_sampling < len(prompts):
            args.time_sampling = len(prompts)

        if args.chunk_steps % args.time_sampling != 0:
            raise ValueError("chunk_steps must be divisible by time_sampling for memory-efficient sampling.")

        fm = foundation_models.create_foundation_model(args.foundation_model)
        if args.substrate == "lenia_flow":
            substrate = substrates.create_substrate(
                args.substrate,
                **util.flow_lenia_kwargs_from_args(args),
            )
        else:
            substrate = substrates.create_substrate(args.substrate)
        substrate = substrates.FlattenSubstrateParameters(substrate)

        rollout_fn = partial(
            rollout_simulation,
            substrate=substrate,
            fm=fm,
            rollout_steps=args.chunk_steps,
            time_sampling=(args.time_sampling, True),
            img_size=224,
            return_state=True,
        )

        z_txt = fm.embed_txt(prompts)

        def calc_loss_from_z(z):
            loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
            loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
            loss_oe = asal_metrics.calc_open_endedness_score(z)
            loss_smoothness = asal_metrics.calc_gradient_score(z)
            loss = loss_prompt * args.coef_prompt + \
                loss_softmax * args.coef_softmax + \
                loss_oe * args.coef_oe + \
                loss_smoothness * args.coef_smooth
            return loss

        def eval_chunk(rng_pop, params_pop, state_pop):
            def run_one(rng_bs, params, state_bs):
                def run_bs(rng, s0):
                    out = rollout_fn(rng=rng, params=params, s0=s0)
                    return out["z"], out["state_final"]
                z_bs, state_final = jax.vmap(run_bs)(rng_bs, state_bs)
                loss_bs = jax.vmap(calc_loss_from_z)(z_bs)
                return loss_bs.mean(), state_final
            loss_pop, state_pop_next = jax.vmap(run_one)(rng_pop, params_pop, state_pop)
            return loss_pop, state_pop_next

        eval_chunk_jit = jax.jit(eval_chunk)

        rng = jax.random.PRNGKey(args.seed)
        strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
        es_params = strategy.default_params
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)

        rng, _rng = split(rng)
        params_pop, es_state = strategy.ask(_rng, es_state, es_params)

        rng, _rng = split(rng)
        init_keys = split(_rng, args.bs)

        def init_states(params_pop):
            def init_one(params):
                return jax.vmap(substrate.init_state, in_axes=(0, None))(init_keys, params)
            return jax.vmap(init_one)(params_pop)

        state_pop = init_states(params_pop)

        data = []
        best_params_traj = []
        best_loss_traj = []
        pop_params_traj = []
        pop_loss_traj = []
        pbar = tqdm(range(args.n_iters))

        for i_iter in pbar:
            rng, _rng = split(rng)
            rng_pop = split(_rng, args.pop_size * args.bs).reshape(args.pop_size, args.bs, 2)

            loss_pop, state_pop = eval_chunk_jit(rng_pop, params_pop, state_pop)

            es_state = strategy.tell(params_pop, loss_pop, es_state, es_params)

            best_params_traj.append(np.array(es_state.best_member))
            best_loss_traj.append(float(es_state.best_fitness))
            pop_params_traj.append(np.array(params_pop))
            pop_loss_traj.append(np.array(loss_pop))

            loss_np = np.array(loss_pop)
            loss_mean = float(loss_np.mean())
            loss_var = float(loss_np.var())

            pca_img = None
            if args.pca_every > 0 and (i_iter % args.pca_every == 0) and len(pop_params_traj) > 1:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                    hist = pop_params_traj[-args.pca_history:]
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
                "loss_pop_mean": loss_mean,
                "loss_pop_var": loss_var,
                "best_loss": float(es_state.best_fitness),
                "iter": i_iter,
            }
            if pca_img is not None:
                log_dict["pop_pca_traj_3d"] = pca_img
            run.log(log_dict)

            data.append(dict(best_loss=es_state.best_fitness, loss=loss_pop))
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

            rng, _rng = split(rng)
            params_pop, es_state = strategy.ask(_rng, es_state, es_params)

    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
