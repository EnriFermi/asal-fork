import os
import argparse

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio.v3 as iio

import substrates
from rollout import rollout_simulation
import util


def parse_time_sampling(arg):
    if arg == 'final' or arg == 'video':
        return arg
    try:
        return int(arg)
    except Exception:
        raise ValueError("time_sampling must be 'final', 'video', or an integer")


def main():
    parser = argparse.ArgumentParser(description="Run a simulation using best params from main_opt training and save a GIF.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory containing best.pkl from main_opt.py')
    parser.add_argument('--substrate', type=str, default='lenia_flow', help='Substrate name used during training')
    parser.add_argument('--rollout_steps', type=int, default=None, help='Number of simulation steps (defaults to substrate default)')
    parser.add_argument('--time_sampling', type=str, default='video', help="'final', 'video', or integer for K samples")
    parser.add_argument('--img_size', type=int, default=224, help='Render size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for rollout')
    parser.add_argument('--n_seeds', type=int, default=1, help='For FlowLenia: number of random non-overlapping seed patches')
    parser.add_argument('--mutations', action='store_true', help='For FlowLenia: enable parameter patch mutations during rollout')
    parser.add_argument('--mutation_sz', type=int, default=20, help='For FlowLenia: size of mutation patch')
    parser.add_argument('--mutation_p', type=float, default=0.1, help='For FlowLenia: probability of mutation each step')
    parser.add_argument('--output', type=str, default='tmp.gif', help='Output GIF path')
    args = parser.parse_args()

    best_path = os.path.join(args.save_dir, 'best.pkl')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best.pkl not found in {args.save_dir}. Ensure main_opt.py saved results with --save_dir.")

    best_member, best_fitness = util.load_pkl(args.save_dir, 'best')

    substrate = substrates.create_substrate(args.substrate)
    # If FlowLenia, allow overriding number of seeding patches
    if hasattr(substrate, 'seed_n_patches') and args.n_seeds is not None:
        try:
            substrate.seed_n_patches = int(args.n_seeds)
        except Exception:
            pass
    if hasattr(substrate, 'mutation_enabled'):
        try:
            substrate.mutation_enabled = bool(args.mutations)
            substrate.mutation_sz = int(args.mutation_sz)
            substrate.mutation_p = float(args.mutation_p)
        except Exception:
            pass
    substrate = substrates.FlattenSubstrateParameters(substrate)
    rollout_steps = substrate.rollout_steps if args.rollout_steps is None else args.rollout_steps

    ts = parse_time_sampling(args.time_sampling)

    rng = jax.random.PRNGKey(args.seed)
    rollout_fn = rollout_simulation

    data = rollout_fn(rng, best_member, s0=None, substrate=substrate, fm=None,
                      rollout_steps=rollout_steps, time_sampling=ts, img_size=args.img_size, return_state=False)

    rgb = data['rgb']
    if isinstance(ts, int) or ts == 'video':
        vid = np.asarray(rgb)
    else:  # final image -> make a 1-frame GIF
        vid = np.asarray(rgb)[None]

    vid_u8 = (np.clip(vid, 0.0, 1.0) * 255).astype(np.uint8)
    iio.imwrite(args.output, vid_u8, duration=1/8)  # default 8 fps
    print(f"Saved simulation to {args.output} (best fitness: {np.array(best_fitness).item():.4f})")


if __name__ == '__main__':
    main()
