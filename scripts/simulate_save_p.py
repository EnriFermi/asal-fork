import os
import argparse
from typing import List

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np

import substrates
from rollout import rollout_simulation
import util
import foundation_models
import asal_metrics
from tqdm import tqdm

def parse_time_sampling(arg):
    if arg == 'final' or arg == 'video':
        return arg
    try:
        return int(arg)
    except Exception:
        raise ValueError("time_sampling must be 'final', 'video', or an integer")


def save_chunk(out_dir: str, fps: float, steps: List[int], snaps_P: List[np.ndarray], file_idx: int, snaps_A: List[np.ndarray]=None, use_fp16: bool=True, snaps_rgb: List[np.ndarray]=None):
    if not steps:
        return file_idx
    start_step = int(steps[0])
    end_step = int(steps[-1])
    start_sec = start_step / fps
    end_sec = end_step / fps
    arrP = np.stack(snaps_P, axis=0).astype(np.float16 if use_fp16 else np.float32)
    meta = {
        "steps": np.array(steps, dtype=np.int64),
        "fps": np.array(fps, dtype=np.float32),
    }
    fname = f"P_steps_{start_step}_{end_step}__secs_{start_sec:.3f}_{end_sec:.3f}__idx_{file_idx:04d}.npz"
    path = os.path.join(out_dir, fname)
    if snaps_A is not None:
        arrA = np.stack(snaps_A, axis=0).astype(np.float16 if use_fp16 else np.float32)
        if snaps_rgb is not None:
            arrRGB = np.stack(snaps_rgb, axis=0).astype(np.uint8)
            np.savez_compressed(path, P=arrP, A=arrA, rgb=arrRGB, **meta)
            print(f"Saved {len(steps)} snapshots (P,A,rgb) to {path}")
        else:
            np.savez_compressed(path, P=arrP, A=arrA, **meta)
            print(f"Saved {len(steps)} snapshots (P,A) to {path}")
    else:
        if snaps_rgb is not None:
            arrRGB = np.stack(snaps_rgb, axis=0).astype(np.uint8)
            np.savez_compressed(path, P=arrP, rgb=arrRGB, **meta)
            print(f"Saved {len(steps)} snapshots (P,rgb) to {path}")
        else:
            np.savez_compressed(path, P=arrP, **meta)
            print(f"Saved {len(steps)} snapshots to {path}")
    return file_idx + 1


def main():
    parser = argparse.ArgumentParser(description="Run a simulation and save P snapshots every k steps.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory containing best.pkl from main_opt.py')
    parser.add_argument('--substrate', type=str, default='lenia_flow', help='Substrate name used during training')
    parser.add_argument('--rollout_steps', type=int, default=None, help='Number of simulation steps (defaults to substrate default)')
    parser.add_argument('--time_sampling', type=str, default='video', help="'final', 'video', or integer for K samples (kept for API parity; unused here)")
    parser.add_argument('--img_size', type=int, default=224, help='Render size (unused, kept for parity)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for rollout')
    parser.add_argument('--n_seeds', type=int, default=1, help='For FlowLenia: number of random non-overlapping seed patches')
    parser.add_argument('--seed_mode', type=str, default='notebook_centers', choices=['center','random_patches','notebook_centers'], help='For FlowLenia: seeding mode')
    parser.add_argument('--p_constant_per_patch', type=int, default=1, help='For FlowLenia: 1 per-patch constant P, 0 per-pixel random P')
    parser.add_argument('--render_mode', type=str, default='Pcolor', choices=['A','Pcolor'], help='For FlowLenia: rendering mode')
    parser.add_argument('--mutations', action='store_true', help='For FlowLenia: enable parameter patch mutations during rollout')
    parser.add_argument('--mutation_sz', type=int, default=20, help='For FlowLenia: size of mutation patch')
    parser.add_argument('--mutation_p', type=float, default=0.1, help='For FlowLenia: probability of mutation each step')
    parser.add_argument('--volcano', action='store_true', help='For FlowLenia: enable volcano mutation (mass reshuffle + strong genome change)')
    parser.add_argument('--volcano_sz', type=int, default=30, help='For FlowLenia: size of volcano patch')
    parser.add_argument('--volcano_p', type=float, default=0.01, help='For FlowLenia: probability of volcano each step')
    parser.add_argument('--volcano_delta', type=float, default=5.0, help='For FlowLenia: scale of genome perturbation in volcano')
    # food mechanics
    parser.add_argument('--food', action='store_true', help='For FlowLenia: enable food mechanics (decay + spawn + consumption)')
    parser.add_argument('--food_interval', type=int, default=128, help='For FlowLenia: steps between food spawns')
    parser.add_argument('--food_n', type=int, default=3, help='For FlowLenia: number of food patches per spawn')
    parser.add_argument('--food_sz', type=int, default=16, help='For FlowLenia: food patch size')
    parser.add_argument('--food_amount', type=float, default=1.0, help='For FlowLenia: amount of food per cell in patch')
    parser.add_argument('--food_consume_rate', type=float, default=0.05, help='For FlowLenia: rate of consumption per step per pixel relative to green mass')
    parser.add_argument('--food_bonus', type=float, default=1.0, help='For FlowLenia: multiplier converting food to mass')
    parser.add_argument('--mass_decay', type=float, default=0.0, help='For FlowLenia: uniform mass decay per step')
    parser.add_argument('--food_channel', type=int, default=1, help='For FlowLenia: which channel consumes food (0=R,1=G,2=B)')
    parser.add_argument('--food_auto_size', action='store_true', help='For FlowLenia: auto-set food patch size to compensate decay per spawn')
    parser.add_argument('--food_conv_mode', type=str, default='scalar', choices=['scalar','conv'], help='For FlowLenia: consumption mode')
    parser.add_argument('--food_diffusion_alpha', type=float, default=0.0, help='For FlowLenia: blend factor for food diffusion (0=off)')
    parser.add_argument('--mass_clip_eps', type=float, default=0.0, help='For FlowLenia: zero-out per-pixel mass below this sum')
    parser.add_argument('--output_dir', type=str, default='snapshots_P', help='Directory (inside save_dir) to write P snapshots')
    parser.add_argument('--snapshot_interval', type=int, default=100, help='Steps between P snapshots')
    parser.add_argument('--snapshots_per_file', type=int, default=50, help='Number of snapshots per chunk file')
    parser.add_argument('--save_A', action='store_true', help='Also save A snapshots alongside P')
    parser.add_argument('--save_rgb', action='store_true', help='Also save rendered RGB (uint8) using A and first 3 P channels')
    parser.add_argument('--save_fp16', action='store_true', help='Save snapshots in float16 (default). Disable to save float32.')
    parser.add_argument('--fps', type=int, default=250, help='Virtual FPS to map steps to seconds in filenames')
    parser.add_argument('--batch_steps', type=int, default=256, help='Number of steps per batch when iterating (no rendering)')
    parser.add_argument('--jit_microbatch', type=int, default=64, help='Frames computed per JIT call inside each batch')
    parser.add_argument('--max_steps', type=int, default=None, help='Total number of steps to run; None for rollout_steps')
    parser.add_argument('--traj_iter', type=int, default=None, help='If set, load parameters from best_traj at this 0-based iteration index instead of final best.pkl')
    args = parser.parse_args()

    best_path = os.path.join(args.save_dir, 'best.pkl')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best.pkl not found in {args.save_dir}. Ensure main_opt.py saved results with --save_dir.")

    best_member, best_fitness = util.load_pkl(args.save_dir, 'best')

    # Optionally override params with a specific iteration from best_traj.pkl
    if args.traj_iter is not None:
        traj_path = os.path.join(args.save_dir, 'best_traj.pkl')
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"traj_iter={args.traj_iter} requested but best_traj.pkl not found in {args.save_dir}. "
                f"Re-run main_opt.py with the updated code that saves best_traj.pkl."
            )
        traj = util.load_pkl(args.save_dir, 'best_traj')
        params_arr = traj.get('params', None)
        if params_arr is None:
            raise ValueError(f"best_traj.pkl in {args.save_dir} does not contain 'params'.")
        n_iters_available = params_arr.shape[0]
        if args.traj_iter < 0 or args.traj_iter >= n_iters_available:
            raise ValueError(f"traj_iter {args.traj_iter} out of range [0, {n_iters_available-1}]")
        best_member = params_arr[args.traj_iter]

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
    # Volcano mutation controls
    if hasattr(substrate, 'volcano_enabled'):
        try:
            substrate.volcano_enabled = bool(args.volcano)
            substrate.volcano_sz = int(args.volcano_sz)
            substrate.volcano_p = float(args.volcano_p)
            substrate.volcano_delta_scale = float(args.volcano_delta)
        except Exception:
            pass
    # Food mechanics
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
        except Exception:
            pass
    if hasattr(substrate, 'mass_clip_eps'):
        try:
            substrate.mass_clip_eps = float(args.mass_clip_eps)
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
    total_steps = substrate.rollout_steps if args.rollout_steps is None else args.rollout_steps
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))

    rng = jax.random.PRNGKey(args.seed)
    s = substrate.init_state(rng, best_member)

    # Jitted microbatch stepper to speed up simulation
    def build_batch_stepper(mb: int):
        def run_batch(state, rng):
            rngs = jax.random.split(rng, mb)
            P0 = jnp.zeros((mb, *state["P"].shape), dtype=state["P"].dtype)
            A0 = jnp.zeros((mb, *state["A"].shape), dtype=state["A"].dtype)

            def body(i, carry):
                st, Pbuf, Abuf = carry
                st = substrate.step_state(rngs[i], st, best_member)
                Pbuf = Pbuf.at[i].set(st["P"])
                Abuf = Abuf.at[i].set(st["A"])
                return (st, Pbuf, Abuf)

            state_next, Pbuf, Abuf = jax.lax.fori_loop(0, mb, body, (state, P0, A0))
            return state_next, Pbuf, Abuf

        return jax.jit(run_batch)

    step_micro = build_batch_stepper(int(args.jit_microbatch))

    # Prepare snapshot directory
    out_dir = os.path.join(args.save_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    snapshot_interval = max(1, int(args.snapshot_interval))
    chunk_size = max(1, int(args.snapshots_per_file))
    fps = float(args.fps)

    steps_buf: List[int] = []
    snaps_buf: List[np.ndarray] = []
    snaps_buf_A: List[np.ndarray] = []
    snaps_buf_rgb: List[np.ndarray] = []
    file_idx = 0

    steps_done = 0
    pbar = tqdm(total=total_steps, desc="Simulating")
    while steps_done < total_steps:
        outer_b = min(args.batch_steps, total_steps - steps_done)
        remaining = outer_b
        while remaining > 0:
            mb = int(args.jit_microbatch)
            if remaining < mb:
                mb = remaining
            rng, _rng = split(rng)
            s, batch_P, batch_A = step_micro(s, _rng)
            # Identify which frames in this microbatch need saving
            base_step = steps_done
            idxs = [i for i in range(mb) if (base_step + i) % snapshot_interval == 0]
            if idxs:
                # Only pull required frames to host
                selP = jnp.take(batch_P, jnp.array(idxs), axis=0)
                selP_np = np.asarray(selP)
                selA_np = None
                if args.save_A:
                    selA = jnp.take(batch_A, jnp.array(idxs), axis=0)
                    selA_np = np.asarray(selA)
                selRGB = None
                if args.save_rgb:
                    if selA_np is None:
                        selA = jnp.take(batch_A, jnp.array(idxs), axis=0)
                        selA_np = np.asarray(selA)
                    # Compute RGB = clip(sum(A) * P[:3], 0, 1), then to uint8
                    a_sum = np.sum(selA_np, axis=-1, keepdims=True)
                    p3 = selP_np[..., :3] if selP_np.shape[-1] >= 3 else np.tile(selP_np, (1, 1, 1, int(np.ceil(3 / selP_np.shape[-1]))))[..., :3]
                    rgb = np.clip(a_sum * p3, 0.0, 1.0)
                    selRGB = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                for i_local, i_global in zip(range(selP_np.shape[0]), idxs):
                    global_step = base_step + i_global
                    steps_buf.append(global_step)
                    snaps_buf.append(selP_np[i_local].astype(np.float32 if not args.save_fp16 else np.float16))
                    if args.save_A:
                        snaps_buf_A.append(selA_np[i_local].astype(np.float32 if not args.save_fp16 else np.float16))
                    if args.save_rgb:
                        snaps_buf_rgb.append(selRGB[i_local])
                    if len(snaps_buf) >= chunk_size:
                        # save chunk with optional A
                        file_idx = save_chunk(
                            out_dir,
                            fps,
                            steps_buf,
                            snaps_buf,
                            file_idx,
                            snaps_buf_A if args.save_A else None,
                            use_fp16=args.save_fp16,
                            snaps_rgb=snaps_buf_rgb if args.save_rgb else None,
                        )
                        if args.save_A:
                            snaps_buf_A = []
                        if args.save_rgb:
                            snaps_buf_rgb = []
                        steps_buf, snaps_buf = [], []
            remaining -= mb
            steps_done += mb
            pbar.update(mb)
    pbar.close()

    if snaps_buf:
        file_idx = save_chunk(
            out_dir,
            fps,
            steps_buf,
            snaps_buf,
            file_idx,
            snaps_buf_A if args.save_A else None,
            use_fp16=args.save_fp16,
            snaps_rgb=snaps_buf_rgb if args.save_rgb else None,
        )

    print(f"Finished simulation. Saved {file_idx} chunk files to {out_dir}")


if __name__ == '__main__':
    main()
