import os
import sys

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio.v3 as iio
import imageio  # for streaming writer
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb
import substrates
from rollout import rollout_simulation
import util
import foundation_models
import asal_metrics
from omegaconf import OmegaConf


def parse_time_sampling(arg):
    if isinstance(arg, int):
        return arg
    if arg == 'final' or arg == 'video':
        return arg
    try:
        return int(arg)
    except Exception:
        raise ValueError("time_sampling must be 'final', 'video', or an integer")


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/simulate_after_training.py <config.yaml>")
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def main(cfg, args):
    run = wandb.init(project=args.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))

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

    base_substrate = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    # If FlowLenia, allow overriding number of seeding patches
    if hasattr(base_substrate, 'seed_n_patches') and args.n_seeds is not None:
        try:
            base_substrate.seed_n_patches = int(args.n_seeds)
        except Exception:
            pass
    if hasattr(base_substrate, 'seed_mode'):
        try:
            base_substrate.seed_mode = str(args.seed_mode)
        except Exception:
            pass
    if hasattr(base_substrate, 'p_constant_per_patch'):
        try:
            base_substrate.p_constant_per_patch = bool(int(args.p_constant_per_patch))
        except Exception:
            pass
    if hasattr(base_substrate, 'render_mode'):
        try:
            base_substrate.render_mode = str(args.render_mode)
        except Exception:
            pass
    # Volcano mutation controls
    if hasattr(base_substrate, 'volcano_enabled'):
        try:
            base_substrate.volcano_enabled = bool(args.volcano)
            base_substrate.volcano_sz = int(args.volcano_sz)
            base_substrate.volcano_p = float(args.volcano_p)
            base_substrate.volcano_delta_scale = float(args.volcano_delta)
        except Exception:
            pass
    if hasattr(base_substrate, 'mutation_enabled'):
        try:
            base_substrate.mutation_enabled = bool(args.mutations)
            base_substrate.mutation_sz = int(args.mutation_sz)
            base_substrate.mutation_p = float(args.mutation_p)
            base_substrate.mutation_scale = float(args.mutation_scale)
        except Exception:
            pass
    # Food mechanics
    if hasattr(base_substrate, 'food_enabled'):
        try:
            base_substrate.food_enabled = bool(args.food)
            base_substrate.food_spawn_interval = int(args.food_interval)
            base_substrate.food_n_patches = int(args.food_n)
            base_substrate.food_patch_size = int(args.food_sz)
            base_substrate.food_amount = float(args.food_amount)
            base_substrate.food_consume_rate = float(args.food_consume_rate)
            base_substrate.food_bonus = float(args.food_bonus)
            base_substrate.mass_decay = float(args.mass_decay)
            base_substrate.food_green_channel = int(args.food_channel)
            if hasattr(base_substrate, 'food_auto_size'):
                base_substrate.food_auto_size = bool(args.food_auto_size)
            if hasattr(base_substrate, 'food_auto_scale'):
                base_substrate.food_auto_scale = float(args.food_auto_scale)
            if hasattr(base_substrate, 'food_conv_mode'):
                base_substrate.food_conv_mode = str(args.food_conv_mode)
            if hasattr(base_substrate, 'food_diffusion_alpha'):
                base_substrate.food_diffusion_alpha = float(args.food_diffusion_alpha)
            if hasattr(base_substrate, 'mass_clip_eps'):
                base_substrate.mass_clip_eps = float(args.mass_clip_eps)
        except Exception:
            pass
    # Ensure parameter length matches current substrate expectation to catch mismatches early
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    param_len = int(np.asarray(best_member).size)
    expected_len = int(substrate.n_params)
    if param_len != expected_len:
        raise ValueError(f"Loaded parameter length {param_len} does not match substrate expectation {expected_len}. "
                         f"Check that training and simulation use the same substrate configuration.")
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

    rng, rng_init = split(rng)
    state0 = substrate.init_state(rng_init, best_member)
    track_mass = isinstance(state0, dict) and "A" in state0
    mass_channels_count = int(np.asarray(state0["A"]).shape[-1]) if track_mass else 0

    # Build JIT-compiled microbatch stepper that returns (state_next, frames[mb, H, W, 3], masses[mb, C]).
    def build_batch_stepper(mb: int, *, track_mass: bool, mass_channels_count: int):
        if track_mass:
            def run_batch(state, rng):
                rngs = jax.random.split(rng, mb)
                frames0 = jnp.zeros((mb, args.img_size, args.img_size, 3), dtype=jnp.float32)
                masses0 = jnp.zeros((mb, mass_channels_count), dtype=jnp.float32)

                def body(i, carry):
                    s, frames, masses = carry
                    s = substrate.step_state(rngs[i], s, best_member)
                    frame = substrate.render_state(s, best_member, img_size=args.img_size)
                    frames = frames.at[i].set(frame)
                    mch = jnp.sum(s["A"], axis=(0, 1))
                    masses = masses.at[i].set(mch)
                    return (s, frames, masses)

                state_next, frames, masses = jax.lax.fori_loop(0, mb, body, (state, frames0, masses0))
                return state_next, frames, masses

            return jax.jit(run_batch)

        def run_batch(state, rng):
            rngs = jax.random.split(rng, mb)
            frames0 = jnp.zeros((mb, args.img_size, args.img_size, 3), dtype=jnp.float32)

            def body(i, carry):
                s, frames = carry
                s = substrate.step_state(rngs[i], s, best_member)
                frame = substrate.render_state(s, best_member, img_size=args.img_size)
                frames = frames.at[i].set(frame)
                return (s, frames)

            state_next, frames = jax.lax.fori_loop(0, mb, body, (state, frames0))
            masses = jnp.zeros((mb, 0), dtype=jnp.float32)
            return state_next, frames, masses

        return jax.jit(run_batch)

    step_micro = build_batch_stepper(
        int(args.jit_microbatch),
        track_mass=track_mass,
        mass_channels_count=mass_channels_count,
    )

    # Streaming writer for 'video': compute frames in jitted microbatches and append
    writer = imageio.get_writer(args.output, fps=args.fps, codec=args.codec, macro_block_size=args.macro_block_size)
    try:
        s = state0
        mass_total = []
        mass_channels = [[] for _ in range(mass_channels_count)]
        steps_done = 0
        with tqdm() as pbar:
            print(args.batch_steps, args.max_steps)
            while args.max_steps is None or steps_done < args.max_steps:
                outer_b = args.batch_steps if args.max_steps is None else min(args.batch_steps, args.max_steps - steps_done)
                remaining = outer_b
                while remaining > 0:
                    mb = int(args.jit_microbatch)
                    mb = remaining if remaining < mb else mb
                    rng, _rng = split(rng)
                    s, batch_frames, batch_masses = step_micro(s, _rng)
                    batch_frames = np.asarray(batch_frames[:mb])  # (mb, H, W, 3)
                    batch_masses = np.asarray(batch_masses[:mb]) if track_mass else None
                    batch_u8 = (np.clip(batch_frames, 0.0, 1.0) * 255).astype(np.uint8)

                    for i_frame in range(batch_u8.shape[0]):
                        frame_u8 = batch_u8[i_frame]
                        writer.append_data(frame_u8)

                        global_step = steps_done + i_frame
                        if track_mass:
                            mchs = batch_masses[i_frame]
                            for c in range(mass_channels_count):
                                mass_channels[c].append(float(mchs[c]))
                            m_tot = float(np.sum(mchs))
                            mass_total.append(m_tot)
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
                    if (
                        track_mass
                        and args.log_mass_every > 0
                        and (steps_done // args.log_mass_every) != ((steps_done + mb) // args.log_mass_every)
                        and len(mass_total) > 0
                    ):
                        print(f"Step {steps_done+mb}: total mass {mass_total[-1]:.6f}")
                    remaining -= mb
                    steps_done += mb
                    pbar.update(mb)

    except KeyboardInterrupt:
        print("Interrupted by user; finalizing video...")
    finally:
        writer.close()
        print(f"Saved simulation to {args.output} (best fitness: {np.array(best_fitness).item():.4f})")
        # save mass plot
        if track_mass and len(mass_total) > 0 and getattr(args, "mass_plot", None):
            try:
                plt.figure(figsize=(8,4))
                for c in range(mass_channels_count):
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
    cfg, flat = load_config()
    main(cfg, flat)
