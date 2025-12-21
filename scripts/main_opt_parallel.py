import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse

import evosax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.random import split
from tqdm.auto import tqdm

import asal_metrics
import foundation_models
import substrates
import util
from rollout import rollout_simulation


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
group.add_argument("--volcano", action='store_true', help="for lenia_flow: enable volcano mutation (mass removal + strong genome change)")
group.add_argument("--volcano_sz", type=int, default=30, help="for lenia_flow: size of volcano patch")
group.add_argument("--volcano_p", type=float, default=0.01, help="for lenia_flow: probability of volcano each step")
group.add_argument("--volcano_delta", type=float, default=5.0, help="for lenia_flow: scale of genome perturbation in volcano")
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
group.add_argument("--food_auto_scale", type=float, default=1.0, help="for lenia_flow: scale factor when auto-sizing food")
group.add_argument("--food_conv_mode", type=str, default='scalar', choices=['scalar','conv'], help="for lenia_flow: consumption mode")
group.add_argument("--food_diffusion_alpha", type=float, default=0.0, help="for lenia_flow: blend factor for food diffusion (0=off)")
group.add_argument("--mass_clip_eps", type=float, default=0.0, help="for lenia_flow: zero-out per-pixel mass below this sum")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use (don't touch this)")
group.add_argument("--time_sampling", type=int, default=32, help="number of images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells", help="prompts to optimize for separated by ';'")
group.add_argument("--coef_prompt", type=float, default=0., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss (only for multiple temporal prompts)")
group.add_argument("--coef_oe", type=float, default=1., help="coefficient for ASAL open-endedness loss (only for single prompt)")
group.add_argument("--coef_smooth", type=float, default=0.2, help="coefficient for latent embedding smoothness")

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=8, help="population size for Sep-CMA-ES (must be divisible by #devices)")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


def main(args):
    devices = jax.local_devices()
    n_dev = len(devices)
    if n_dev < 2:
        raise RuntimeError("main_opt_parallel.py expects at least 2 devices; use main_opt.py for single GPU.")
    if args.pop_size % n_dev != 0:
        raise ValueError(f"pop_size ({args.pop_size}) must be divisible by number of devices ({n_dev}).")
    shard = args.pop_size // n_dev

    run = wandb.init(project="asal", config={**vars(args), "n_devices": n_dev})
    try:
        prompts = args.prompts.split(";")
        if args.time_sampling < len(prompts):
            args.time_sampling = len(prompts)

        fm = foundation_models.create_foundation_model(args.foundation_model)
        substrate = substrates.create_substrate(args.substrate)
        # FlowLenia controls
        if hasattr(substrate, 'seed_n_patches'):
            substrate.seed_n_patches = int(args.seed_n_patches)
        if hasattr(substrate, 'seed_mode'):
            substrate.seed_mode = str(args.seed_mode)
        if hasattr(substrate, 'p_constant_per_patch'):
            substrate.p_constant_per_patch = bool(int(args.p_constant_per_patch))
        if hasattr(substrate, 'render_mode'):
            substrate.render_mode = str(args.render_mode)
        if hasattr(substrate, 'volcano_enabled'):
            substrate.volcano_enabled = bool(args.volcano)
            substrate.volcano_sz = int(args.volcano_sz)
            substrate.volcano_p = float(args.volcano_p)
            substrate.volcano_delta_scale = float(args.volcano_delta)
        if hasattr(substrate, 'mutation_enabled'):
            substrate.mutation_enabled = bool(args.mutations)
            substrate.mutation_sz = int(args.mutation_sz)
            substrate.mutation_p = float(args.mutation_p)
        if hasattr(substrate, 'food_enabled'):
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
            if hasattr(substrate, 'food_auto_scale'):
                substrate.food_auto_scale = float(args.food_auto_scale)
            if hasattr(substrate, 'food_conv_mode'):
                substrate.food_conv_mode = str(args.food_conv_mode)
            if hasattr(substrate, 'food_diffusion_alpha'):
                substrate.food_diffusion_alpha = float(args.food_diffusion_alpha)
            if hasattr(substrate, 'mass_clip_eps'):
                substrate.mass_clip_eps = float(args.mass_clip_eps)
        substrate = substrates.FlattenSubstrateParameters(substrate)
        if args.rollout_steps is None:
            args.rollout_steps = substrate.rollout_steps
        rollout_fn = jax.jit(
            lambda rng, params: rollout_simulation(
                rng,
                params,
                s0=None,
                substrate=substrate,
                fm=fm,
                rollout_steps=args.rollout_steps,
                time_sampling=(args.time_sampling, True),
                img_size=224,
                return_state=False,
            )
        )

        z_txt = fm.embed_txt(prompts)

        rng = jax.random.PRNGKey(args.seed)
        strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
        es_params = strategy.default_params
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)

        def calc_loss(rng_in, params):
            data = rollout_fn(rng_in, params)
            z = data['z']
            loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
            loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
            loss_oe = asal_metrics.calc_open_endedness_score(z)
            loss_smooth = asal_metrics.calc_gradient_score(z)
            loss = (
                loss_prompt * args.coef_prompt +
                loss_softmax * args.coef_softmax +
                loss_oe * args.coef_oe +
                loss_smooth * args.coef_smooth
            )
            return loss, dict(loss_prompt=loss_prompt, loss_softmax=loss_softmax, loss_oe=loss_oe, loss_smooth=loss_smooth)

        def eval_shard(rng_shard, params_shard):
            # rng_shard: (shard, bs, 2)
            # params_shard: (shard, n_params)
            def eval_param(rng_bs, param):
                rng_bs = rng_bs.reshape(args.bs, 2)
                losses, loss_dicts = jax.vmap(calc_loss, in_axes=(0, None))(rng_bs, param)
                loss_mean = losses.mean()
                loss_dict_mean = jax.tree_map(lambda x: x.mean(axis=0), loss_dicts)
                return loss_mean, loss_dict_mean
            return jax.vmap(eval_param)(rng_shard, params_shard)

        p_eval = jax.pmap(eval_shard, in_axes=(0, 0), devices=devices)

        pbar = tqdm(range(args.n_iters))
        best_so_far = None
        best_loss = np.inf

        for i_iter in pbar:
            rng, _rng = split(rng)
            params_full, es_state = strategy.ask(_rng, es_state, es_params)

            rng_eval = jax.random.split(rng, args.pop_size * args.bs)
            rng_eval = rng_eval.reshape(n_dev, shard, args.bs, 2)
            params_sharded = params_full.reshape(n_dev, shard, -1)

            loss_shards, loss_dict_shards = p_eval(rng_eval, params_sharded)
            loss_all = np.array(loss_shards).reshape(-1)
            loss_dict_all = jax.tree_map(lambda x: np.array(x).reshape(-1, *x.shape[2:]), loss_dict_shards)

            es_state = strategy.tell(params_full, jnp.asarray(loss_all), es_state, es_params)

            loss_mean = float(loss_all.mean())
            loss_var = float(loss_all.var())
            best_idx = int(np.argmin(loss_all))
            best_curr = float(loss_all[best_idx])
            if best_curr < best_loss:
                best_loss = best_curr
                best_so_far = np.array(params_full[best_idx])

            log_dict = {
                "iter": i_iter,
                "loss_pop_mean": loss_mean,
                "loss_pop_var": loss_var,
                "best_loss": float(es_state.best_fitness),
                "best_loss_local": best_loss,
            }
            run.log(log_dict)
            pbar.set_postfix(best=best_loss)

            if args.save_dir is not None and (i_iter % max(1, args.n_iters // 10) == 0 or i_iter == args.n_iters - 1):
                util.save_pkl(args.save_dir, "data_parallel_best", (np.array(es_state.best_member), np.array(es_state.best_fitness)))
                if best_so_far is not None:
                    util.save_pkl(args.save_dir, "best_running_parallel", (best_so_far, np.array(best_loss)))

            # Optional: log video of best-so-far on host
            if best_so_far is not None and (i_iter % max(1, args.n_iters // 5) == 0 or i_iter == args.n_iters - 1):
                try:
                    rng, _rng_vid = split(rng)
                    vid_data = rollout_simulation(
                        _rng_vid,
                        best_so_far,
                        s0=None,
                        substrate=substrate,
                        fm=None,
                        rollout_steps=args.rollout_steps,
                        time_sampling='video',
                        img_size=140,
                        return_state=False,
                        return_mass=True,
                    )
                    vid = (np.asarray(vid_data['rgb']) * 255).astype(np.uint8).transpose(0, 3, 1, 2)
                    log_payload = {'train_video': wandb.Video(vid, fps=24, format='gif')}
                    run.log(log_payload)
                except Exception as e:
                    print(f"Video logging failed: {e}")

    finally:
        run.finish()


if __name__ == "__main__":
    main(parse_args())
