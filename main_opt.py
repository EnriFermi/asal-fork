import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
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

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=1, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, leave None for the default of the substrate")
group.add_argument("--seed_n_patches", type=int, default=1, help="for lenia_flow: number of random non-overlapping seed patches")
group.add_argument("--mutations", action='store_true', help="for lenia_flow: enable parameter patch mutations during rollout")
group.add_argument("--mutation_sz", type=int, default=20, help="for lenia_flow: size of mutation patch")
group.add_argument("--mutation_p", type=float, default=0.1, help="for lenia_flow: probability of mutation each step")
group.add_argument("--seed_mode", type=str, default='notebook_centers', choices=['center','random_patches','notebook_centers'], help="for lenia_flow: seeding mode")
group.add_argument("--p_constant_per_patch", type=int, default=1, help="for lenia_flow: 1 for per-patch constant P, 0 for per-pixel random P")
group.add_argument("--render_mode", type=str, default='Pcolor', choices=['A','Pcolor'], help="for lenia_flow: rendering mode")
group.add_argument("--food", action='store_true', help="for lenia_flow: enable food mechanics (decay + spawn + consumption)")
group.add_argument("--food_interval", type=int, default=128, help="for lenia_flow: steps between food spawns")
group.add_argument("--food_n", type=int, default=3, help="for lenia_flow: number of food patches per spawn")
group.add_argument("--food_sz", type=int, default=16, help="for lenia_flow: food patch size")
group.add_argument("--food_amount", type=float, default=1.0, help="for lenia_flow: amount of food per cell in patch")
group.add_argument("--food_consume_rate", type=float, default=0.05, help="for lenia_flow: rate of consumption per step per pixel relative to green mass")
group.add_argument("--food_bonus", type=float, default=1.0, help="for lenia_flow: multiplier converting food to mass")
group.add_argument("--mass_decay", type=float, default=0.0, help="for lenia_flow: uniform mass decay per step")
group.add_argument("--food_channel", type=int, default=1, help="for lenia_flow: which channel consumes food (0=R,1=G,2=B)")
group.add_argument("--food_auto_size", action='store_true', help="for lenia_flow: auto-set food patch size to compensate decay per spawn")
group.add_argument("--food_conv_mode", type=str, default='scalar', choices=['scalar','conv'], help="for lenia_flow: consumption mode")
group.add_argument("--food_diffusion_alpha", type=float, default=0.0, help="for lenia_flow: blend factor for food diffusion (0=off)")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use (don't touch this)")
group.add_argument("--time_sampling", type=int, default=32, help="number of images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells", help="prompts to optimize for seperated by ';'")
group.add_argument("--coef_prompt", type=float, default=0., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss (only for multiple temporal prompts)")
group.add_argument("--coef_oe", type=float, default=1., help="coefficient for ASAL open-endedness loss (only for single prompt)")
group.add_argument("--coef_smooth", type=float, default=0.2, help="coefficient for latent embedding smoothness")


group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=8, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--eval_splits", type=int, default=1, help="number of splits of CMA-ES population for loss evaluation (1 = no split)")


# #wandb logging
# group = parser.add_argument_group("logging")
# group = pa


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args



import imageio.v3 as iio
import numpy as np
from IPython.display import Image, display

def show_video(x, fps=25, path="tmp.gif"):
    x = (x*255).astype(np.uint8) if x.dtype != np.uint8 else x
    iio.imwrite(path, x, duration=1/fps)
    # display(Image(path))


def main(args):
    run = wandb.init( project="asal", config={**vars(args)})
    try:
        prompts = args.prompts.split(";")
        if args.time_sampling < len(prompts): # doing multiple prompts
            args.time_sampling = len(prompts)
        print(args)
        
        fm = foundation_models.create_foundation_model(args.foundation_model)
        substrate = substrates.create_substrate(args.substrate)
        # Optional: control initial seeding for FlowLenia
        if hasattr(substrate, 'seed_n_patches'):
            try:
                substrate.seed_n_patches = int(args.seed_n_patches)
            except Exception:
                pass
        if hasattr(substrate, 'seed_mode'):
            try:
                substrate.seed_mode = str(args.seed_mode)
            except Exception:
                pass
        if hasattr(substrate, 'p_constant_per_patch'):
            try:
                substrate.p_constant_per_patch = bool(int(args.p_constant_per_patch))
            except Exception:
                pass
        if hasattr(substrate, 'render_mode'):
            try:
                substrate.render_mode = str(args.render_mode)
            except Exception:
                pass
        # Optional: food mechanics
        if hasattr(substrate, 'food_enabled'):
            try:
                substrate.food_enabled = bool(args.food)
                substrate.food_spawn_interval = int(args.food_interval)
                substrate.food_n_patches = int(args.food_n)
                substrate.food_patch_size = int(args.food_sz)
                substrate.food_amount = float(args.food_amount)
                substrate.food_consume_rate = float(args.food_consume_rate)
                substrate.food_bonus = float(args.food_bonus)
                substrate.mass_decay = float(args.mass_decay)
                substrate.food_green_channel = int(args.food_channel)
                if hasattr(substrate, 'food_auto_size'):
                    substrate.food_auto_size = bool(args.food_auto_size)
                if hasattr(substrate, 'food_conv_mode'):
                    substrate.food_conv_mode = str(args.food_conv_mode)
                if hasattr(substrate, 'food_diffusion_alpha'):
                    substrate.food_diffusion_alpha = float(args.food_diffusion_alpha)
                # Make food visible as white overlay in training videos
                if hasattr(substrate, 'food_vis_color'):
                    substrate.food_vis_color = (1.0, 1.0, 1.0)
            except Exception:
                pass
        # Optional: control mutation behavior for FlowLenia
        if hasattr(substrate, 'mutation_enabled'):
            try:
                substrate.mutation_enabled = bool(args.mutations)
                substrate.mutation_sz = int(args.mutation_sz)
                substrate.mutation_p = float(args.mutation_p)
            except Exception:
                pass
        substrate = substrates.FlattenSubstrateParameters(substrate)
        if args.rollout_steps is None:
            args.rollout_steps = substrate.rollout_steps
        rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=args.rollout_steps, time_sampling=(args.time_sampling, True), img_size=224, return_state=False)

        z_txt = fm.embed_txt(prompts) # P D

        rng = jax.random.PRNGKey(args.seed)
        print(substrate.n_params)
        strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
        es_params = strategy.default_params
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)

        def calc_loss(rng, params): # calculate the loss given the simulation parameters
            rollout_data = rollout_fn(rng, params)
            z = rollout_data['z']

            loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
            loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
            loss_oe = asal_metrics.calc_open_endedness_score(z)
            loss_smoothness = asal_metrics.calc_gradient_score(z)

            loss = loss_prompt * args.coef_prompt + \
                loss_softmax * args.coef_softmax + \
                loss_oe * args.coef_oe + \
                loss_smoothness * args.coef_smooth
            
            loss_dict = dict(loss=loss, loss_prompt=loss_prompt, loss_softmax=loss_softmax, loss_oe=loss_oe)
            return loss, loss_dict, rollout_data['rgb']

        @jax.jit
        def eval_chunk(rng, params_chunk):
            """
            Evaluate loss for a chunk of the CMA-ES population.
            params_chunk: (chunk_size, n_params)
            Returns:
                rng_next, loss_chunk (chunk_size,), loss_dict_chunk, best_loss_chunk, best_rgb_chunk
            """
            rng, _rng = split(rng)
            calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))
            rng, _rng2 = split(rng)
            loss, loss_dict, rgb = calc_loss_vv(split(_rng2, args.bs), params_chunk)
            # mean over init state rng axis (bs)
            loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))
            # best within this chunk
            best_idx = jnp.argmin(loss)
            best_loss_chunk = loss[best_idx]
            best_rgb_chunk = rgb[best_idx, 0]
            return rng, loss, loss_dict, best_loss_chunk, best_rgb_chunk

        def do_iter(es_state, rng): # do one iteration of the optimization with optional population splitting
            rng, _rng = split(rng)
            params_full, next_es_state = strategy.ask(_rng, es_state, es_params)
            pop_size = params_full.shape[0]
            splits = max(1, int(args.eval_splits))
            if splits > pop_size:
                splits = pop_size
            if pop_size % splits != 0:
                raise ValueError(f"pop_size={pop_size} not divisible by eval_splits={splits}; "
                                 f"choose eval_splits that divides pop_size.")
            chunk_size = pop_size // splits

            loss_chunks = []
            loss_dict_chunks = []
            best_rgb = None
            best_loss_scalar = None

            for i in range(splits):
                start = i * chunk_size
                end = start + chunk_size
                params_chunk = params_full[start:end]
                rng, loss_chunk, loss_dict_chunk, best_loss_chunk, best_rgb_chunk = eval_chunk(rng, params_chunk)
                loss_chunks.append(loss_chunk)
                loss_dict_chunks.append(loss_dict_chunk)
                # track best rgb over all chunks
                loss_scalar = float(best_loss_chunk)
                if best_loss_scalar is None or loss_scalar < best_loss_scalar:
                    best_loss_scalar = loss_scalar
                    best_rgb = best_rgb_chunk

            # concatenate losses over population axis
            loss_all = jnp.concatenate(loss_chunks, axis=0)

            # concatenate loss_dict over population axis
            def concat_tree(chunks):
                return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)

            loss_dict_all = concat_tree(loss_dict_chunks)

            # update CMA-ES state with full population loss
            next_es_state = strategy.tell(params_full, loss_all, next_es_state, es_params)
            data = dict(best_loss=next_es_state.best_fitness, loss_dict=loss_dict_all)
            return next_es_state, data, best_rgb, params_full, loss_all


        data = []
        best_params_traj = []
        best_loss_traj = []
        pop_params_traj = []
        pop_loss_traj = []
        pbar = tqdm(range(args.n_iters))
        for i_iter in pbar:
            rng, _rng = split(rng)
            es_state, di, rgb, params_iter, loss_iter = do_iter(es_state, _rng)

            # Track best-so-far parameter trajectory
            best_params_traj.append(np.array(es_state.best_member))
            best_loss_traj.append(float(es_state.best_fitness))
            # Track full CMA-ES population for this iteration
            pop_params_traj.append(np.array(params_iter))
            pop_loss_traj.append(np.array(loss_iter))

            # Population loss statistics (mean/variance over CMA-ES samples)
            loss_np = np.array(loss_iter)
            loss_mean = float(loss_np.mean())
            loss_var = float(loss_np.var())

            # 3D PCA over all population samples seen so far (x,y=PCs, z=time)
            pca_img = None
            if len(pop_params_traj) > 1:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                    pop_hist = np.stack(pop_params_traj, axis=0)  # (T, P, D)
                    T_hist, P_hist, D_hist = pop_hist.shape
                    X = pop_hist.reshape(T_hist * P_hist, D_hist)
                    times = np.repeat(np.arange(T_hist), P_hist)

                    X_centered = X - X.mean(axis=0, keepdims=True)
                    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    pcs = X_centered @ Vt[:2].T  # (N, 2)

                    fig = plt.figure(figsize=(6, 5))
                    ax = fig.add_subplot(111, projection="3d")
                    sc = ax.scatter(pcs[:, 0], pcs[:, 1], times, c=times, cmap="viridis", s=3)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("iter")
                    ax.set_title(f"Population PCA trajectory up to iter {i_iter}")
                    pca_img = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f\"PCA population logging failed at iter {i_iter}: {e}\")

            # Log scalar stats and PCA image for this iteration
            log_dict = {
                "loss_pop_mean": loss_mean,
                "loss_pop_var": loss_var,
                "best_loss": float(es_state.best_fitness),
                "iter": i_iter,
            }
            if pca_img is not None:
                log_dict["pop_pca_traj_3d"] = pca_img
            run.log(log_dict)

            show_video(rgb)
            run.log({'train_sample': wandb.Video((np.asarray(rgb) * 255).astype(np.uint8).transpose(0, 3, 1, 2), fps=4, format="gif")})

            # After step: run a full rollout (all frames) for W&B logging using best-so-far params
            try:
                rng, _rng_vid = split(rng)
                best_params = es_state.best_member
                vid_data = rollout_simulation(_rng_vid, best_params, s0=None, substrate=substrate, fm=None,
                                              rollout_steps=args.rollout_steps, time_sampling='video', img_size=224,
                                              return_state=False, return_mass=True)
                vid = (np.asarray(vid_data['rgb']) * 255).astype(np.uint8).transpose(0, 3, 1, 2)
                log_payload = {'train_video': wandb.Video(vid, fps=8, format='gif')}

                # Log mass trajectory over the rollout to check stability (sum over grid and channels)
                mass_traj = vid_data.get('mass', None)
                food_traj = vid_data.get('food_mass', None)
                if mass_traj is not None:
                    mass_traj = np.asarray(mass_traj)
                    ys = [mass_traj.tolist()]
                    keys = ["mass_total"]
                    if food_traj is not None:
                        food_traj = np.asarray(food_traj)
                        ys.append(food_traj.tolist())
                        keys.append("food_total")
                    line = wandb.plot.line_series(
                        xs=list(range(mass_traj.shape[0])),
                        ys=ys,
                        keys=keys,
                        title="Mass trajectory (best member rollout)",
                        xname="step",
                    )
                    log_payload['train_mass_total_traj'] = line

                run.log(log_payload)
            except Exception as e:
                print(f"Full video logging failed: {e}")

            data.append(di)
            pbar.set_postfix(best_loss=es_state.best_fitness.item())
            if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1): # save data every 10% of the run
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
                        params=np.stack(pop_params_traj, axis=0),  # (T, pop_size, n_params)
                        loss=np.stack(pop_loss_traj, axis=0),      # (T, pop_size)
                    )
                    util.save_pkl(args.save_dir, "pop_traj", pop_traj)

        # (Optional) Final PCA summary is now covered by per-iteration logging above
    finally:
        run.finish()
    

if __name__ == '__main__':
    main(parse_args())
