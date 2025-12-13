import argparse
import copy
import os
from collections import defaultdict
from functools import partial

import evosax
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from PIL import Image
from tqdm.auto import tqdm

from create_sim import rollout_and_embed_simulation

import util
from clip_jax import MyFlaxCLIP
from models.models_gol import GameOfLife

# this script only measures the open-endedness/novelty score of the Game of Life substrate simulations.
parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None) # the directory to save results to

group = parser.add_argument_group("model")
group.add_argument("--grid_size", type=int, default=64) # grid size of the simulation
group.add_argument("--rollout_steps", type=int, default=4096) # number of steps to rollout simulation for

group = parser.add_argument_group("data")
group.add_argument("--n_rollout_imgs", type=int, default=32) # number of images to render during the rollout
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14 (don't change this)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=16) # number of init states to run for each simulation
group.add_argument("--start", type=int, default=0) # start range for params search
group.add_argument("--end", type=int, default=262144) # end range for params search, 0-262144 is the entire substrate.


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    sim = GameOfLife(grid_size=args.grid_size)
    clip_model = MyFlaxCLIP(args.clip_model)
    rng = jax.random.PRNGKey(args.seed)

    def calc_loss(rng, params):
        rollout_data = rollout_and_embed_simulation(rng, params, sim=sim, clip_model=clip_model, rollout_steps=args.rollout_steps, n_rollout_imgs=args.n_rollout_imgs)

        # --------- CLIP OE score --------- 
        z = rollout_data['z'] # T D
        scores_novelty = (z @ z.T) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty = scores_novelty.max(axis=-1) # T

        # --------- Manual OE score (based on pixel metric) ---------
        state_vid = rollout_data['state_vid'] # T H W
        scores_novelty = 1.-jnp.abs(state_vid[None, :] - state_vid[:, None]).mean(axis=(-1, -2)) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty_manual = scores_novelty.max(axis=-1) # T
        return dict(loss_novelty=loss_novelty, loss_novelty_manual=loss_novelty_manual, z_final=z[-1])

    @jax.jit
    def do_iter(params):
        calc_loss_v = jax.vmap(calc_loss, in_axes=(0, None))
        data = calc_loss_v(split(rng, args.bs), params)
        data = dict(loss_novelty=data['loss_novelty'].mean(axis=0),
                    loss_novelty_manual=data['loss_novelty_manual'].mean(axis=0),
                    z_final=data['z_final'][args.bs//2])
        return data

    args.n_iters = args.end - args.start
    all_params = np.arange(args.start, args.end)
    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        di = do_iter(all_params[i_iter])
        data.append(di)

        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            util.save_pkl(args.save_dir, "all_params", all_params)

if __name__ == '__main__':
    main(parse_args())
