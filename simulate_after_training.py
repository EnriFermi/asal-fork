import os
import argparse

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio.v3 as iio
import imageio  # for streaming writer
import matplotlib.pyplot as plt
import wandb

import substrates
from rollout import rollout_simulation
import util
import foundation_models
import asal_metrics


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
    parser.add_argument('--output', type=str, default='out.mp4', help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=250, help='Output video FPS')
    parser.add_argument('--codec', type=str, default='libx264', help='Video codec (e.g., libx264)')
    parser.add_argument('--macro_block_size', type=int, default=None, help='Macro block size for encoder (set None to disable)')
    parser.add_argument('--batch_steps', type=int, default=256, help='Number of steps per batch when streaming (frames written per outer loop)')
    parser.add_argument('--jit_microbatch', type=int, default=64, help='Frames computed per JIT call inside each batch (smaller avoids OOM)')
    parser.add_argument('--max_steps', type=int, default=None, help='Total number of steps to run; None for until interrupted')
    parser.add_argument('--mass_plot', type=str, default='mass.png', help='Path to save mass traces plot (total and per-channel)')
    parser.add_argument('--log_mass_every', type=int, default=1000, help='Print total mass every N frames')
    parser.add_argument('--traj_iter', type=int, default=None, help='If set, load parameters from best_traj at this 0-based iteration index instead of final best.pkl')
    parser.add_argument('--compute_oe', action='store_true', help='If set, compute open-endedness loss over time using CLIP.')
    parser.add_argument('--oe_every', type=int, default=100, help='Steps between open-endedness evaluations')
    parser.add_argument('--oe_plot', type=str, default='oe_loss.png', help='Path to save open-endedness loss plot')
    parser.add_argument('--wandb_project', type=str, default='asal', help='W&B project name for logging simulation dynamics')
    args = parser.parse_args()

    run = wandb.init(project=args.wandb_project, config={**vars(args)})

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
        loss_arr = traj.get('loss', None)
        if loss_arr is not None and loss_arr.shape[0] == n_iters_available:
            best_fitness = loss_arr[args.traj_iter]

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
        except Exception:
            pass
    substrate = substrates.FlattenSubstrateParameters(substrate)
    rollout_steps = substrate.rollout_steps if args.rollout_steps is None else args.rollout_steps

    rng = jax.random.PRNGKey(args.seed)

    # Optional: set up foundation model for open-endedness
    fm = None
    oe_steps = []
    oe_values = []
    oe_embeds = []
    if args.compute_oe:
        fm = foundation_models.create_foundation_model('clip')

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
        run.finish()
        return

    # Build JIT-compiled microbatch stepper that returns (state_next, frames[mb, H, W, 3])
    def build_batch_stepper(mb: int):
        def run_batch(state, rng):
            rngs = jax.random.split(rng, mb)
            frames0 = jnp.zeros((mb, args.img_size, args.img_size, 3), dtype=jnp.float32)
            # infer channel count from current state at trace time
            # we create a dummy mass buffer matching channels in A
            A0 = state["A"]
            C = A0.shape[-1]
            masses0 = jnp.zeros((mb, C), dtype=jnp.float32)

            def body(i, carry):
                s, frames, masses = carry
                s = substrate.step_state(rngs[i], s, best_member)
                frame = substrate.render_state(s, best_member, img_size=args.img_size)
                frames = frames.at[i].set(frame)
                mch = jnp.sum(s["A"], axis=(0, 1))  # per-channel mass
                masses = masses.at[i].set(mch)
                return (s, frames, masses)

            state_next, frames, masses = jax.lax.fori_loop(0, mb, body, (state, frames0, masses0))
            return state_next, frames, masses

        return jax.jit(run_batch)

    step_micro = build_batch_stepper(int(args.jit_microbatch))

    # Streaming writer for 'video': compute frames in jitted microbatches and append
    writer = imageio.get_writer(args.output, fps=args.fps, codec=args.codec, macro_block_size=args.macro_block_size)
    try:
        s = substrate.init_state(rng, best_member)
        # setup mass traces
        C = int(np.asarray(s["A"]).shape[-1])
        mass_total = []
        mass_channels = [ [] for _ in range(C) ]
        steps_done = 0
        while args.max_steps is None or steps_done < args.max_steps:
            outer_b = args.batch_steps if args.max_steps is None else min(args.batch_steps, args.max_steps - steps_done)
            remaining = outer_b
            while remaining > 0:
                mb = int(args.jit_microbatch)
                mb = remaining if remaining < mb else mb
                rng, _rng = split(rng)
                s, batch_frames, batch_masses = step_micro(s, _rng)
                batch_frames = np.asarray(batch_frames[:mb])  # (mb, H, W, 3)
                batch_masses = np.asarray(batch_masses[:mb])  # (mb, C)
                batch_u8 = (np.clip(batch_frames, 0.0, 1.0) * 255).astype(np.uint8)

                for i_frame in range(batch_u8.shape[0]):
                    frame_u8 = batch_u8[i_frame]
                    writer.append_data(frame_u8)

                    # record masses
                    mchs = batch_masses[i_frame]
                    for c in range(C):
                        mass_channels[c].append(float(mchs[c]))
                    m_tot = float(np.sum(mchs))
                    mass_total.append(m_tot)

                    global_step = steps_done + i_frame
                    # log total mass per frame to W&B for trajectory tracking
                    wandb.log({"mass_total": m_tot, "step": global_step})
                    # optional: open-endedness evaluation
                    if args.compute_oe and (global_step % args.oe_every == 0):
                        img = batch_frames[i_frame]  # float32 in [0,1], shape (H, W, 3)
                        z_img = fm.embed_img(jnp.array(img))
                        oe_embeds.append(np.asarray(z_img))
                        oe_steps.append(global_step)
                        if len(oe_embeds) < 2:
                            oe_val = 0.0
                        else:
                            z_all = jnp.asarray(oe_embeds)
                            oe_val = float(asal_metrics.calc_open_endedness_score(z_all))
                        oe_values.append(oe_val)
                        # log to W&B for online visualization
                        wandb.log({"oe_loss": oe_val, "step": global_step})

                # periodic log
                if args.log_mass_every > 0 and (steps_done // args.log_mass_every) != ((steps_done + mb) // args.log_mass_every):
                    print(f"Step {steps_done+mb}: total mass {mass_total[-1]:.6f}")
                remaining -= mb
                steps_done += mb
    except KeyboardInterrupt:
        print("Interrupted by user; finalizing video...")
    finally:
        writer.close()
        print(f"Saved simulation to {args.output} (best fitness: {np.array(best_fitness).item():.4f})")
        # save mass plot
        try:
            plt.figure(figsize=(8,4))
            for c in range(C):
                plt.plot(mass_channels[c], label=f'ch{c}')
            plt.plot(mass_total, label='total', linewidth=2, color='k', alpha=0.7)
            plt.xlabel('frame')
            plt.ylabel('mass (sum over grid)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.mass_plot, dpi=150)
            print(f"Saved mass traces to {args.mass_plot}")
        except Exception as e:
            print(f"Failed to save mass plot: {e}")

        # save open-endedness loss plot if requested
        if args.compute_oe and len(oe_values) > 0:
            try:
                plt.figure(figsize=(8,4))
                plt.plot(oe_steps, oe_values, label='open-endedness loss')
                plt.xlabel('step')
                plt.ylabel('OE loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(args.oe_plot, dpi=150)
                print(f"Saved open-endedness loss traces to {args.oe_plot}")
                # also log plot to W&B
                wandb.log({"oe_loss_plot": wandb.Image(plt.gcf())})
            except Exception as e:
                print(f"Failed to save open-endedness plot: {e}")

        run.finish()


if __name__ == '__main__':
    main()
