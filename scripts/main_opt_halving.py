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
        raise SystemExit("Usage: python scripts/main_opt_halving.py <config.yaml>")
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

        stage_steps = list(args.stage_steps)
        if any(x <= 0 for x in stage_steps):
            raise ValueError("stage_steps must be positive and increasing.")
        if stage_steps != sorted(stage_steps):
            raise ValueError("stage_steps must be strictly increasing.")

        keep_fracs = list(args.keep_fracs)
        if len(keep_fracs) < len(stage_steps):
            keep_fracs = keep_fracs + [keep_fracs[-1]] * (len(stage_steps) - len(keep_fracs))

        for i in range(len(stage_steps)):
            delta = stage_steps[i] - (stage_steps[i - 1] if i > 0 else 0)
            if delta % args.time_sampling != 0:
                raise ValueError("Each stage increment must be divisible by time_sampling.")

        fm = foundation_models.create_foundation_model(args.foundation_model)
        if args.substrate == "lenia_flow":
            substrate = substrates.create_substrate(
                args.substrate,
                **util.flow_lenia_kwargs_from_args(args),
            )
        else:
            substrate = substrates.create_substrate(args.substrate)
        substrate = substrates.FlattenSubstrateParameters(substrate)

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

        def make_eval_chunk(steps):
            rollout_fn = partial(
                rollout_simulation,
                substrate=substrate,
                fm=fm,
                rollout_steps=steps,
                time_sampling=(args.time_sampling, True),
                img_size=224,
                return_state=True,
            )
            def eval_chunk(rng_pop, params_pop, state_pop):
                def run_one(rng_bs, params, state_bs):
                    def run_bs(rng, s0):
                        out = rollout_fn(rng=rng, params=params, s0=s0)
                        state_final = out.get("state_final", None)
                        if state_final is None:
                            state_final = out["state"][-1]
                        return out["z"], state_final
                    z_bs, state_final = jax.vmap(run_bs)(rng_bs, state_bs)
                    return z_bs, state_final
                z_pop, state_pop_next = jax.vmap(run_one)(rng_pop, params_pop, state_pop)
                return z_pop, state_pop_next
            return jax.jit(eval_chunk)

        eval_fns = [make_eval_chunk(stage_steps[0])] + [
            make_eval_chunk(stage_steps[i] - stage_steps[i - 1]) for i in range(1, len(stage_steps))
        ]

        rng = jax.random.PRNGKey(args.seed)
        strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
        es_params = strategy.default_params
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)

        data = []
        best_params_traj = []
        best_loss_traj = []
        pop_params_traj = []
        pop_loss_traj = []
        pbar = tqdm(range(args.n_iters))

        rng, _rng = split(rng)
        init_keys = split(_rng, args.bs)

        def init_states_chunk(params_chunk):
            def init_one(params):
                return jax.vmap(substrate.init_state, in_axes=(0, None))(init_keys, params)
            return jax.vmap(init_one)(params_chunk)

        init_states_chunk_jit = jax.jit(init_states_chunk)
        pop_batch = int(getattr(args, "pop_batch", args.pop_size))

        for i_iter in pbar:
            rng, _rng = split(rng)
            params_full, es_state = strategy.ask(_rng, es_state, es_params)

            state_chunks = []
            for start in range(0, args.pop_size, pop_batch):
                end = min(args.pop_size, start + pop_batch)
                state_chunks.append(init_states_chunk_jit(params_full[start:end]))
            state_alive = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *state_chunks)
            alive_idx = np.arange(args.pop_size, dtype=int)
            z_prefix = [None] * args.pop_size
            final_loss = np.zeros(args.pop_size, dtype=np.float32)
            final_stage = np.zeros(args.pop_size, dtype=np.int32)

            prev_step = 0
            last_loss_alive = None
            for stage_i, (stage_step, keep_frac, eval_fn) in enumerate(zip(stage_steps, keep_fracs, eval_fns)):
                steps_to_run = stage_step - prev_step
                prev_step = stage_step

                rng, _rng = split(rng)
                rng_all = split(_rng, len(alive_idx) * args.bs).reshape(len(alive_idx), args.bs, 2)
                params_alive = params_full[alive_idx]
                z_chunks = []
                state_chunks = []
                for start in range(0, len(alive_idx), pop_batch):
                    end = min(len(alive_idx), start + pop_batch)
                    z_chunk, state_chunk = eval_fn(
                        rng_all[start:end],
                        params_alive[start:end],
                        jax.tree.map(lambda x: x[start:end], state_alive),
                    )
                    z_chunks.append(z_chunk)
                    state_chunks.append(state_chunk)
                z_chunk = jnp.concatenate(z_chunks, axis=0)
                state_alive = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *state_chunks)

                for j, idx in enumerate(alive_idx):
                    if z_prefix[idx] is None:
                        z_prefix[idx] = z_chunk[j]
                    else:
                        z_prefix[idx] = jnp.concatenate([z_prefix[idx], z_chunk[j]], axis=1)

                z_stack = jnp.stack([z_prefix[idx] for idx in alive_idx], axis=0)

                def loss_from_prefix(z_pref):
                    loss_bs = jax.vmap(calc_loss_from_z)(z_pref)
                    return loss_bs.mean()

                last_loss_alive = jax.vmap(loss_from_prefix)(z_stack)
                loss_alive_np = np.array(last_loss_alive)

                keep_n = max(1, int(np.ceil(len(alive_idx) * keep_frac)))
                order = np.argsort(loss_alive_np)
                keep_sel = order[:keep_n]
                drop_sel = order[keep_n:]

                for j in drop_sel:
                    idx = alive_idx[j]
                    final_loss[idx] = loss_alive_np[j]
                    final_stage[idx] = stage_i

                alive_idx = alive_idx[keep_sel]
                state_alive = jax.tree.map(lambda x: x[keep_sel], state_alive)

                if len(alive_idx) == 0:
                    break

            if len(alive_idx) > 0 and last_loss_alive is not None:
                loss_alive_np = np.array(last_loss_alive)
                for j, idx in enumerate(alive_idx):
                    final_loss[idx] = loss_alive_np[j]
                    final_stage[idx] = len(stage_steps)

            order = sorted(range(args.pop_size), key=lambda i: (-final_stage[i], final_loss[i]))
            loss_rank = np.empty(args.pop_size, dtype=np.float32)
            for rank, idx in enumerate(order):
                loss_rank[idx] = float(rank)

            es_state = strategy.tell(params_full, jnp.asarray(loss_rank), es_state, es_params)

            best_params_traj.append(np.array(es_state.best_member))
            best_loss_traj.append(float(es_state.best_fitness))
            pop_params_traj.append(np.array(params_full))
            pop_loss_traj.append(np.array(loss_rank))

            loss_mean = float(loss_rank.mean())
            loss_var = float(loss_rank.var())

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

            data.append(dict(best_loss=es_state.best_fitness, loss=loss_rank))
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

    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
