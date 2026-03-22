import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

from torch.profiler import profile, record_function, ProfilerActivity
import jax.profiler
# jax.profiler.start_server(6969)

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import wandb
import util

print(jax.devices())
print(jax.default_backend())

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=1, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")
group.add_argument("--resume", action="store_true", help="resume from save_dir/resume_state.pkl if it exists")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, leave None for the default of the substrate")
group.add_argument("--grid_size", type=int, default=128, help="for lenia_flow: grid size")
group.add_argument("--C", type=int, default=1, help="for lenia_flow: number of channels")
group.add_argument("--k", type=int, default=10, help="for lenia_flow: number of kernels")
group.add_argument("--kernel_components", type=int, default=3, help="for lenia_flow: number of kernel components")
group.add_argument("--M", type=str, default="2,1,0;0,2,1;1,0,2", help="for lenia_flow: connectivity matrix as 'a,b,c;d,e,f;g,h,i'")
group.add_argument("--dd", type=int, default=5, help="for lenia_flow: dd parameter")
group.add_argument("--dt", type=float, default=0.2, help="for lenia_flow: dt parameter")
group.add_argument("--flow_sigma", type=float, default=0.65, help="for lenia_flow: sigma parameter")
group.add_argument("--border", type=str, default="wall", help="for lenia_flow: border mode")
group.add_argument("--mix_rule", type=str, default="stoch", help="for lenia_flow: mix rule")
group.add_argument("--base_seed", type=int, default=0, help="for lenia_flow: base random seed for default params")
group.add_argument("--seed_patch_size", type=int, default=20, help="for lenia_flow: size of seed patch")
group.add_argument("--seed_n_patches", type=int, default=1, help="for lenia_flow: number of random non-overlapping seed patches")
group.add_argument("--mutations", action='store_true', help="for lenia_flow: enable parameter patch mutations during rollout")
group.add_argument("--mutation_sz", type=int, default=20, help="for lenia_flow: size of mutation patch")
group.add_argument("--mutation_p", type=float, default=0.1, help="for lenia_flow: probability of mutation each step")
group.add_argument("--mutation_scale", type=float, default=1.0, help="for lenia_flow: scale for mutation noise")
group.add_argument("--optimize_mutation_scale", action='store_true', help="for lenia_flow: make mutation_scale optimizable")
group.add_argument("--volcano", action='store_true', help="for lenia_flow: enable volcano mutation (mass removal + strong genome change)")
group.add_argument("--volcano_sz", type=int, default=30, help="for lenia_flow: size of volcano patch")
group.add_argument("--volcano_p", type=float, default=0.01, help="for lenia_flow: probability of volcano each step")
group.add_argument("--volcano_delta", type=float, default=5.0, help="for lenia_flow: scale of genome perturbation in volcano")
group.add_argument("--seed_mode", type=str, default='notebook_centers', choices=['center','random_patches','notebook_centers'], help="for lenia_flow: seeding mode")
group.add_argument("--p_constant_per_patch", type=int, default=1, help="for lenia_flow: 1 for per-patch constant P, 0 for per-pixel random P")
group.add_argument("--render_mode", type=str, default='Pcolor', choices=['A','Pcolor','PcolorMix'], help="for lenia_flow: rendering mode")
group.add_argument("--clip1", type=float, default=float("inf"), help="for lenia_flow: clip1 for parameter deltas")
group.add_argument("--clip2", type=float, default=float("inf"), help="for lenia_flow: clip2 for parameter deltas")
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
group.add_argument("--food_auto_scale", type=float, default=1.0, help="for lenia_flow: scale factor for auto-sized food compensation")
group.add_argument("--food_conv_mode", type=str, default='scalar', choices=['scalar','conv'], help="for lenia_flow: consumption mode")
group.add_argument("--food_vis_scale", type=float, default=1.0, help="for lenia_flow: food visualization scale")
group.add_argument("--food_vis_color", type=str, default="0.6,0.3,0.0", help="for lenia_flow: food visualization color as 'r,g,b'")
group.add_argument("--food_diffusion_alpha", type=float, default=0.0, help="for lenia_flow: blend factor for food diffusion (0=off)")
group.add_argument("--mass_clip_eps", type=float, default=0.0, help="for lenia_flow: zero-out per-pixel mass below this sum")

group = parser.add_argument_group("evaluation")
group.add_argument(
    "--foundation_model",
    type=str,
    default="clip",
    help="image encoder to use. Supports 'clip', 'siglip2', or a google/siglip2* model id",
)
group.add_argument("--time_sampling", type=int, default=32, help="number of images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a biological cell;two biological cells", help="prompts to optimize for seperated by ';'")
group.add_argument("--coef_prompt", type=float, default=0., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss (only for multiple temporal prompts)")
group.add_argument("--coef_oe", type=float, default=1., help="coefficient for ASAL open-endedness loss (only for single prompt)")
group.add_argument("--coef_smooth", type=float, default=0.2, help="coefficient for latent embedding smoothness")



group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--optimizer", type=str, default="Sep-CMA-ES", help="optimizer to use: Sep-CMA-ES or LM_MA_ES")
group.add_argument(
    "--params_init",
    type=str,
    default="strategy_default",
    help="parameter initialization: strategy_default or substrate_default",
)
group.add_argument("--pop_size", type=int, default=8, help="population size for the selected ES strategy")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")
group.add_argument("--eval_splits", type=int, default=1, help="number of splits of CMA-ES population for loss evaluation (1 = no split)")

group = parser.add_argument_group("logging")
group.add_argument("--wandb_project", type=str, default="asal", help="Weights & Biases project name")
group.add_argument("--pca_every", type=int, default=1, help="Log population PCA every N iters; <=0 disables")
group.add_argument("--pca_history", type=int, default=100, help="History length for PCA trajectory logging")
group.add_argument("--full_video_interval", type=int, default=1, help="Log best-member full video every N iters; <=0 disables")
group.add_argument("--full_video_rollout_steps", type=int, default=None, help="Optional cap on rollout steps for full video logging")
group.add_argument("--full_video_img_size", type=int, default=140, help="Image size for full video logging")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def _canonicalize_optimizer_name(name):
    if name is None:
        return "sep_cma_es"
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "sep_cma_es": "sep_cma_es",
        "sepcmaes": "sep_cma_es",
        "sep_cma": "sep_cma_es",
        "lm_ma_es": "lm_ma_es",
        "lmmaes": "lm_ma_es",
        "lm_ma": "lm_ma_es",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown optimizer {name!r}. Use 'Sep-CMA-ES' or 'LM_MA_ES'."
        )
    return aliases[normalized]


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


def _build_strategy(optimizer_name, *, pop_size, num_dims, sigma_init):
    if optimizer_name == "sep_cma_es":
        strategy_cls = evosax.Sep_CMA_ES
    elif optimizer_name == "lm_ma_es":
        strategy_cls = getattr(evosax, "LM_MA_ES", None)
        if strategy_cls is None:
            raise ValueError(
                "Requested optimizer 'LM_MA_ES', but evosax does not expose LM_MA_ES in this environment."
            )
    else:
        raise ValueError(f"Unhandled optimizer {optimizer_name!r}.")
    return strategy_cls(popsize=pop_size, num_dims=num_dims, sigma_init=sigma_init)


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


def _to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.array(x), tree)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _load_resume_state(save_dir):
    if save_dir is None:
        return None
    path = os.path.join(save_dir, "resume_state.pkl")
    if not os.path.exists(path):
        return None
    return util.load_pkl(save_dir, "resume_state")


def _load_optional_pkl(save_dir, name):
    if save_dir is None:
        return None
    path = os.path.join(save_dir, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return util.load_pkl(save_dir, name)


def _has_legacy_saved_outputs(save_dir):
    if save_dir is None:
        return False
    legacy_names = (
        "data.pkl",
        "best.pkl",
        "best_traj.pkl",
        "pop_traj.pkl",
        "palette_traj.pkl",
    )
    return any(os.path.exists(os.path.join(save_dir, name)) for name in legacy_names)


def _initialize_optimizer_state(args, *, params_init, strategy, es_params, substrate):
    rng = jax.random.PRNGKey(args.seed)
    if params_init == "strategy_default":
        rng, _rng = split(rng)
        es_state = strategy.initialize(_rng, es_params)
    elif params_init == "substrate_default":
        rng, rng_mean, rng_init = jax.random.split(rng, 3)
        init_mean = substrate.default_params(rng_mean)
        es_state = _initialize_strategy_with_mean(strategy, rng_init, es_params, init_mean)
    else:
        raise ValueError(f"Unhandled params_init {params_init!r}.")
    return rng, es_state


def _unstack_tree_axis0(tree):
    if tree is None:
        return []
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return []
    length = int(np.asarray(leaves[0]).shape[0])
    return [jax.tree.map(lambda x: np.array(x[i]), tree) for i in range(length)]


def _save_resume_state(
    save_dir,
    *,
    next_iter,
    rng,
    es_state,
    optimizer_name,
    params_init,
    args,
    substrate_n_params,
    data,
    best_params_traj,
    best_loss_traj,
    pop_params_traj,
    pop_loss_traj,
    palette_traj,
):
    if save_dir is None:
        return
    payload = dict(
        version=1,
        next_iter=int(next_iter),
        optimizer_name=str(optimizer_name),
        params_init=str(params_init),
        pop_size=int(args.pop_size),
        substrate_n_params=int(substrate_n_params),
        n_iters_target=int(args.n_iters),
        rng=np.array(rng),
        es_state=_to_numpy_tree(es_state),
        data=[] if len(data) == 0 else _to_numpy_tree(data),
        best_params_traj=[np.array(x) for x in best_params_traj],
        best_loss_traj=np.asarray(best_loss_traj),
        pop_params_traj=[np.array(x) for x in pop_params_traj],
        pop_loss_traj=[np.array(x) for x in pop_loss_traj],
        palette_traj=palette_traj,
    )
    util.save_pkl(save_dir, "resume_state", payload)


def _restore_resume_state(
    checkpoint,
    *,
    optimizer_name,
    params_init,
    pop_size,
    substrate_n_params,
):
    if checkpoint is None:
        return None
    ckpt_optimizer = str(checkpoint.get("optimizer_name"))
    if ckpt_optimizer != str(optimizer_name):
        raise ValueError(
            f"Resume checkpoint optimizer mismatch: checkpoint={ckpt_optimizer!r}, current={optimizer_name!r}."
        )
    ckpt_params_init = str(checkpoint.get("params_init"))
    if ckpt_params_init != str(params_init):
        raise ValueError(
            f"Resume checkpoint params_init mismatch: checkpoint={ckpt_params_init!r}, current={params_init!r}."
        )
    ckpt_pop_size = int(checkpoint.get("pop_size"))
    if ckpt_pop_size != int(pop_size):
        raise ValueError(
            f"Resume checkpoint pop_size mismatch: checkpoint={ckpt_pop_size}, current={int(pop_size)}."
        )
    ckpt_n_params = int(checkpoint.get("substrate_n_params"))
    if ckpt_n_params != int(substrate_n_params):
        raise ValueError(
            f"Resume checkpoint substrate_n_params mismatch: checkpoint={ckpt_n_params}, current={int(substrate_n_params)}."
        )
    return dict(
        next_iter=int(checkpoint.get("next_iter", 0)),
        rng=jnp.asarray(checkpoint["rng"]),
        es_state=_to_jax_tree(checkpoint["es_state"]),
        data=list(checkpoint.get("data", [])),
        best_params_traj=[np.array(x) for x in checkpoint.get("best_params_traj", [])],
        best_loss_traj=[float(x) for x in np.asarray(checkpoint.get("best_loss_traj", []))],
        pop_params_traj=[np.array(x) for x in checkpoint.get("pop_params_traj", [])],
        pop_loss_traj=[np.array(x) for x in checkpoint.get("pop_loss_traj", [])],
        palette_traj=list(checkpoint.get("palette_traj", [])),
    )


def _restore_legacy_saved_outputs(
    save_dir,
    *,
    args,
    strategy,
    es_params,
    substrate,
    optimizer_name,
    params_init,
):
    pop_traj = _load_optional_pkl(save_dir, "pop_traj")
    if pop_traj is None:
        raise FileNotFoundError(
            f"resume=true, but legacy save_dir {save_dir!r} has no pop_traj.pkl. "
            "Cannot reconstruct optimizer state."
        )
    if not isinstance(pop_traj, dict) or "params" not in pop_traj or "loss" not in pop_traj:
        raise ValueError("Legacy pop_traj.pkl must contain 'params' and 'loss'.")

    pop_params_arr = np.asarray(pop_traj["params"])
    pop_loss_arr = np.asarray(pop_traj["loss"])
    if pop_params_arr.ndim != 3:
        raise ValueError(
            f"Expected pop_traj['params'] to have shape (T, pop_size, n_params), got {pop_params_arr.shape}."
        )
    if pop_loss_arr.shape != pop_params_arr.shape[:2]:
        raise ValueError(
            f"Expected pop_traj['loss'] to have shape {pop_params_arr.shape[:2]}, got {pop_loss_arr.shape}."
        )
    if int(pop_params_arr.shape[1]) != int(args.pop_size):
        raise ValueError(
            f"Legacy pop_traj population size mismatch: checkpoint={int(pop_params_arr.shape[1])}, "
            f"current={int(args.pop_size)}."
        )
    if int(pop_params_arr.shape[2]) != int(substrate.n_params):
        raise ValueError(
            f"Legacy pop_traj parameter size mismatch: checkpoint={int(pop_params_arr.shape[2])}, "
            f"current={int(substrate.n_params)}."
        )

    start_iter = int(pop_params_arr.shape[0])
    rng, es_state = _initialize_optimizer_state(
        args,
        params_init=params_init,
        strategy=strategy,
        es_params=es_params,
        substrate=substrate,
    )

    replay_best_params = []
    replay_best_loss = []
    for i_iter in range(start_iter):
        rng, rng_iter = split(rng)
        params_iter_saved = jnp.asarray(pop_params_arr[i_iter])
        loss_iter_saved = jnp.asarray(pop_loss_arr[i_iter])
        params_iter_asked, ask_state = strategy.ask(rng_iter, es_state, es_params)
        if not np.allclose(
            np.asarray(params_iter_asked),
            np.asarray(params_iter_saved),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "Legacy resume replay mismatch: reconstructed population does not match saved pop_traj. "
                "This usually means the code/config/evosax version differs from the original run."
            )
        es_state = strategy.tell(params_iter_saved, loss_iter_saved, ask_state, es_params)
        replay_best_params.append(np.array(es_state.best_member))
        replay_best_loss.append(float(es_state.best_fitness))
        if args.full_video_interval > 0 and (i_iter % args.full_video_interval == 0):
            rng, _ = split(rng)

    best_traj = _load_optional_pkl(save_dir, "best_traj")
    if best_traj is not None:
        if "params" not in best_traj or "loss" not in best_traj:
            raise ValueError("Legacy best_traj.pkl must contain 'params' and 'loss'.")
        best_params_arr = np.asarray(best_traj["params"])
        best_loss_arr = np.asarray(best_traj["loss"])
        if int(best_params_arr.shape[0]) != start_iter or int(best_loss_arr.shape[0]) != start_iter:
            raise ValueError(
                "Legacy best_traj.pkl length does not match pop_traj.pkl length; cannot resume safely."
            )
        if start_iter > 0:
            if not np.allclose(best_params_arr[-1], replay_best_params[-1], rtol=1e-5, atol=1e-6):
                raise ValueError("Legacy best_traj.pkl final params do not match replayed optimizer state.")
            if not np.allclose(best_loss_arr[-1], replay_best_loss[-1], rtol=1e-6, atol=1e-8):
                raise ValueError("Legacy best_traj.pkl final loss does not match replayed optimizer state.")
        best_params_traj = [np.array(x) for x in best_params_arr]
        best_loss_traj = [float(x) for x in best_loss_arr]
    else:
        best_params_traj = replay_best_params
        best_loss_traj = replay_best_loss

    best_obj = _load_optional_pkl(save_dir, "best")
    if best_obj is not None and start_iter > 0:
        best_member_saved, best_fitness_saved = best_obj
        if not np.allclose(np.asarray(best_member_saved), replay_best_params[-1], rtol=1e-5, atol=1e-6):
            raise ValueError("Legacy best.pkl params do not match replayed optimizer state.")
        if not np.allclose(float(best_fitness_saved), replay_best_loss[-1], rtol=1e-6, atol=1e-8):
            raise ValueError("Legacy best.pkl loss does not match replayed optimizer state.")

    data_stacked = _load_optional_pkl(save_dir, "data")
    data = _unstack_tree_axis0(data_stacked)
    if len(data) not in (0, start_iter):
        raise ValueError("Legacy data.pkl length does not match pop_traj.pkl length.")

    palette_traj = _load_optional_pkl(save_dir, "palette_traj")
    if palette_traj is None:
        palette_traj = []

    return dict(
        next_iter=start_iter,
        rng=rng,
        es_state=es_state,
        data=data,
        best_params_traj=best_params_traj,
        best_loss_traj=best_loss_traj,
        pop_params_traj=[np.array(x) for x in pop_params_arr],
        pop_loss_traj=[np.array(x) for x in pop_loss_arr],
        palette_traj=list(palette_traj),
        resumed_from_legacy=True,
        optimizer_name=str(optimizer_name),
    )



import imageio.v3 as iio
import numpy as np
from IPython.display import Image, display

def show_video(x, fps=25, path="tmp.gif"):
    x = (x*255).astype(np.uint8) if x.dtype != np.uint8 else x
    iio.imwrite(path, x, duration=1/fps)
    # display(Image(path))

def main(args):
    run = wandb.init(project=args.wandb_project, config={**vars(args)})
    try:
        prompts = args.prompts.split(";")
        if args.time_sampling < len(prompts): # doing multiple prompts
            args.time_sampling = len(prompts)
        print(args)

        optimizer_name = _canonicalize_optimizer_name(getattr(args, "optimizer", "Sep-CMA-ES"))
        params_init = _canonicalize_params_init(getattr(args, "params_init", "strategy_default"))
        fm = foundation_models.create_foundation_model(args.foundation_model)
        rollout_img_size = int(getattr(fm, "image_size", 224))
        run.summary["optimizer/name"] = optimizer_name
        run.summary["optimizer/params_init"] = params_init
        run.summary["encoder/name"] = str(args.foundation_model)
        run.summary["encoder/img_size"] = int(rollout_img_size)
        if args.substrate == "lenia_flow":
            substrate = substrates.create_substrate(
                args.substrate,
                **util.flow_lenia_kwargs_from_args(args),
            )
        else:
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
        # Volcano mutation controls
        if hasattr(substrate, 'volcano_enabled'):
            try:
                substrate.volcano_enabled = bool(args.volcano)
                substrate.volcano_sz = int(args.volcano_sz)
                substrate.volcano_p = float(args.volcano_p)
                substrate.volcano_delta_scale = float(args.volcano_delta)
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
                if hasattr(substrate, 'mass_clip_eps'):
                    substrate.mass_clip_eps = float(args.mass_clip_eps)
                # Make food visible as white overlay in training videos
                if hasattr(substrate, 'food_vis_color'):
                    substrate.food_vis_color = (1.0, 1.0, 1.0)
                    print(substrate.food_vis_color)
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
        rollout_fn = partial(
            rollout_simulation,
            s0=None,
            substrate=substrate,
            fm=fm,
            rollout_steps=args.rollout_steps,
            time_sampling=(args.time_sampling, True),
            img_size=rollout_img_size,
            return_state=False,
        )

        z_txt = fm.embed_txt(prompts) # P D

        rng = jax.random.PRNGKey(args.seed)
        print(substrate.n_params)
        strategy = _build_strategy(
            optimizer_name,
            pop_size=args.pop_size,
            num_dims=substrate.n_params,
            sigma_init=args.sigma,
        )
        es_params = strategy.default_params

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
        palette_traj = []
        start_iter = 0
        resumed = False
        resumed_from_legacy = False
        if bool(getattr(args, "resume", False)):
            resume_state = _load_resume_state(args.save_dir)
            if resume_state is not None:
                restored = _restore_resume_state(
                    resume_state,
                    optimizer_name=optimizer_name,
                    params_init=params_init,
                    pop_size=args.pop_size,
                    substrate_n_params=substrate.n_params,
                )
                start_iter = restored["next_iter"]
                rng = restored["rng"]
                es_state = restored["es_state"]
                data = restored["data"]
                best_params_traj = restored["best_params_traj"]
                best_loss_traj = restored["best_loss_traj"]
                pop_params_traj = restored["pop_params_traj"]
                pop_loss_traj = restored["pop_loss_traj"]
                palette_traj = restored["palette_traj"]
                resumed = True
                print(f"Resuming optimization from iter {start_iter} using {args.save_dir}/resume_state.pkl")
            elif _has_legacy_saved_outputs(args.save_dir):
                restored = _restore_legacy_saved_outputs(
                    args.save_dir,
                    args=args,
                    strategy=strategy,
                    es_params=es_params,
                    substrate=substrate,
                    optimizer_name=optimizer_name,
                    params_init=params_init,
                )
                start_iter = restored["next_iter"]
                rng = restored["rng"]
                es_state = restored["es_state"]
                data = restored["data"]
                best_params_traj = restored["best_params_traj"]
                best_loss_traj = restored["best_loss_traj"]
                pop_params_traj = restored["pop_params_traj"]
                pop_loss_traj = restored["pop_loss_traj"]
                palette_traj = restored["palette_traj"]
                resumed = True
                resumed_from_legacy = True
                print(
                    f"Recovered legacy optimization state from iter {start_iter} "
                    f"using saved outputs in {args.save_dir}"
                )
                _save_resume_state(
                    args.save_dir,
                    next_iter=start_iter,
                    rng=rng,
                    es_state=es_state,
                    optimizer_name=optimizer_name,
                    params_init=params_init,
                    args=args,
                    substrate_n_params=substrate.n_params,
                    data=data,
                    best_params_traj=best_params_traj,
                    best_loss_traj=best_loss_traj,
                    pop_params_traj=pop_params_traj,
                    pop_loss_traj=pop_loss_traj,
                    palette_traj=palette_traj,
                )
        if not resumed:
            rng, es_state = _initialize_optimizer_state(
                args,
                params_init=params_init,
                strategy=strategy,
                es_params=es_params,
                substrate=substrate,
            )

        run.summary["resume/enabled"] = bool(getattr(args, "resume", False))
        run.summary["resume/loaded"] = bool(resumed)
        run.summary["resume/legacy_recovered"] = bool(resumed_from_legacy)
        run.summary["resume/start_iter"] = int(start_iter)
        if args.save_dir is not None:
            run.summary["resume/checkpoint_path"] = os.path.join(args.save_dir, "resume_state.pkl")
        if start_iter >= int(args.n_iters):
            print(
                f"Run already completed for n_iters={int(args.n_iters)} "
                f"(resume checkpoint next_iter={int(start_iter)}). Nothing to do."
            )
            return

        save_interval = max(1, args.n_iters // 10)
        pbar = tqdm(range(start_iter, args.n_iters), initial=start_iter, total=args.n_iters)
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        # with jax.profiler.trace("prof_dir", create_perfetto_link=True):
        for i_iter in pbar:
            # with record_function('opt_iter'):
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
            if args.pca_every > 0 and (i_iter % args.pca_every == 0) and len(pop_params_traj) > 1:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                    hist = pop_params_traj[-args.pca_history:]
                    pop_hist = np.stack(hist, axis=0)  # (T, P, D)
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
                    print(f"PCA population logging failed at iter {i_iter}: {e}")

            # Log scalar stats and PCA image for this iteration
            log_dict = {
                "loss_pop_mean": loss_mean,
                "loss_pop_var": loss_var,
                "best_loss": float(es_state.best_fitness),
                "iter": i_iter,
            }
            palette_stats = util.flow_lenia_palette_stats(np.array(es_state.best_member), substrate)
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
            if pca_img is not None:
                log_dict["pop_pca_traj_3d"] = pca_img
            run.log(log_dict)

            # show_video(rgb)
            # run.log({'train_sample': wandb.Video((np.asarray(rgb) * 255).astype(np.uint8).transpose(0, 3, 1, 2), fps=4, format="gif")})

            # After step: run a full rollout (all frames) for W&B logging using best-so-far params
            if args.full_video_interval > 0 and (i_iter % args.full_video_interval == 0):
                try:
                    rng, _rng_vid = split(rng)
                    best_params = es_state.best_member
                    video_rollout_steps = args.rollout_steps
                    if args.full_video_rollout_steps is not None:
                        video_rollout_steps = min(int(video_rollout_steps), int(args.full_video_rollout_steps))
                    vid_data = rollout_simulation(
                        _rng_vid,
                        best_params,
                        s0=None,
                        substrate=substrate,
                        fm=None,
                        rollout_steps=video_rollout_steps,
                        time_sampling='video',
                        img_size=int(args.full_video_img_size),
                        return_state=False,
                        return_mass=True,
                    )
                    vid = (np.asarray(vid_data['rgb']) * 255).astype(np.uint8).transpose(0, 3, 1, 2)
                    log_payload = {'train_video': wandb.Video(vid, fps=24, format='gif')}

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
            if palette_stats is not None:
                palette_traj.append(dict(iter=i_iter, **palette_stats))
            pbar.set_postfix(best_loss=es_state.best_fitness.item())
            if args.save_dir is not None and (i_iter % save_interval == 0 or i_iter == args.n_iters - 1): # save data every 10% of the run
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
                if len(palette_traj) > 0:
                    util.save_pkl(args.save_dir, "palette_traj", palette_traj)
                _save_resume_state(
                    args.save_dir,
                    next_iter=i_iter + 1,
                    rng=rng,
                    es_state=es_state,
                    optimizer_name=optimizer_name,
                    params_init=params_init,
                    args=args,
                    substrate_n_params=substrate.n_params,
                    data=data,
                    best_params_traj=best_params_traj,
                    best_loss_traj=best_loss_traj,
                    pop_params_traj=pop_params_traj,
                    pop_loss_traj=pop_loss_traj,
                    palette_traj=palette_traj,
                )

        # (Optional) Final PCA summary is now covered by per-iteration logging above
    finally:
        run.finish()

    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))


if __name__ == '__main__':
    main(parse_args())
