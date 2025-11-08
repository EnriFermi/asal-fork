import os
import argparse

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio.v3 as iio
import imageio  # for streaming writer

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
    parser.add_argument('--seed_mode', type=str, default='notebook_centers', choices=['center','random_patches','notebook_centers'], help='For FlowLenia: seeding mode')
    parser.add_argument('--p_constant_per_patch', type=int, default=1, help='For FlowLenia: 1 per-patch constant P, 0 per-pixel random P')
    parser.add_argument('--render_mode', type=str, default='Pcolor', choices=['A','Pcolor'], help='For FlowLenia: rendering mode')
    parser.add_argument('--mutations', action='store_true', help='For FlowLenia: enable parameter patch mutations during rollout')
    parser.add_argument('--mutation_sz', type=int, default=20, help='For FlowLenia: size of mutation patch')
    parser.add_argument('--mutation_p', type=float, default=0.1, help='For FlowLenia: probability of mutation each step')
    parser.add_argument('--output', type=str, default='out.mp4', help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=250, help='Output video FPS')
    parser.add_argument('--codec', type=str, default='libx264', help='Video codec (e.g., libx264)')
    parser.add_argument('--macro_block_size', type=int, default=None, help='Macro block size for encoder (set None to disable)')
    parser.add_argument('--batch_steps', type=int, default=256, help='Number of steps per batch when streaming')
    parser.add_argument('--max_steps', type=int, default=None, help='Total number of steps to run; None for until interrupted')
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
    if hasattr(substrate, 'mutation_enabled'):
        try:
            substrate.mutation_enabled = bool(args.mutations)
            substrate.mutation_sz = int(args.mutation_sz)
            substrate.mutation_p = float(args.mutation_p)
        except Exception:
            pass
    substrate = substrates.FlattenSubstrateParameters(substrate)
    rollout_steps = substrate.rollout_steps if args.rollout_steps is None else args.rollout_steps

    rng = jax.random.PRNGKey(args.seed)

    if args.time_sampling != 'video':
        # Non-video modes can be done in one shot safely
        ts = parse_time_sampling(args.time_sampling)
        data = rollout_simulation(rng, best_member, s0=None, substrate=substrate, fm=None,
                                  rollout_steps=rollout_steps, time_sampling=ts, img_size=args.img_size, return_state=False)
        rgb = np.asarray(data['rgb'])
        vid = rgb if isinstance(ts, int) else rgb[None]
        vid_u8 = (np.clip(vid, 0.0, 1.0) * 255).astype(np.uint8)
        iio.imwrite(args.output, vid_u8, fps=args.fps, codec=args.codec, macro_block_size=args.macro_block_size)
        print(f"Saved simulation to {args.output} (best fitness: {np.array(best_fitness).item():.4f})")
        return

    # Streaming writer for 'video': step and render without stacking in memory
    writer = imageio.get_writer(args.output, fps=args.fps, codec=args.codec, macro_block_size=args.macro_block_size)
    try:
        s = substrate.init_state(rng, best_member)
        steps_done = 0
        while args.max_steps is None or steps_done < args.max_steps:
            b = args.batch_steps if args.max_steps is None else min(args.batch_steps, args.max_steps - steps_done)
            for _ in range(b):
                rng, _rng = split(rng)
                s = substrate.step_state(_rng, s, best_member)
                frame = substrate.render_state(s, best_member, img_size=args.img_size)
                frame_u8 = (np.clip(np.asarray(frame), 0.0, 1.0) * 255).astype(np.uint8)
                writer.append_data(frame_u8)
            steps_done += b
    except KeyboardInterrupt:
        print("Interrupted by user; finalizing video...")
    finally:
        writer.close()
        print(f"Saved simulation to {args.output} (best fitness: {np.array(best_fitness).item():.4f})")


if __name__ == '__main__':
    main()
