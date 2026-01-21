import os
import sys

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt

import substrates
from rollout import rollout_simulation
import util
import evolutionary_metrics as evo
from omegaconf import OmegaConf
from tqdm import tqdm


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/simulate_frustration.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def to_np(x):
    return np.asarray(x) if x is not None else None


def assemble_blocks(blocks, split_n, block_size, C):
    H = block_size * split_n
    A_full = np.zeros((H, H, C), dtype=np.float32)
    if isinstance(blocks, dict):
        P_full = np.zeros((H, H, blocks["P"].shape[-1]), dtype=np.float32)
        A_blocks = to_np(blocks["A"])
        P_blocks = to_np(blocks["P"])
        Food_blocks = to_np(blocks.get("Food", 0.0))
        B = A_blocks.shape[0]
        for bi in range(B):
            i = bi // split_n
            j = bi % split_n
            i0 = i * block_size
            j0 = j * block_size
            A_full[i0:i0 + block_size, j0:j0 + block_size] = A_blocks[bi]
            P_full[i0:i0 + block_size, j0:j0 + block_size] = P_blocks[bi]
            Food_full = Food_full if "Food_full" in locals() else np.zeros((H, H), dtype=np.float32)
            Food_full[i0:i0 + block_size, j0:j0 + block_size] = Food_blocks[bi]
        return A_full, P_full, Food_full
    P_full = np.zeros((H, H, blocks[0]["P"].shape[-1]), dtype=np.float32)
    Food_full = np.zeros((H, H), dtype=np.float32)
    for bi, blk in enumerate(blocks):
        i = bi // split_n
        j = bi % split_n
        i0 = i * block_size
        j0 = j * block_size
        A_full[i0:i0 + block_size, j0:j0 + block_size] = to_np(blk["A"])
        P_full[i0:i0 + block_size, j0:j0 + block_size] = to_np(blk["P"])
        Food_full[i0:i0 + block_size, j0:j0 + block_size] = to_np(blk.get("Food", 0.0))
    return A_full, P_full, Food_full


def main(cfg, args):
    # load params
    best_path = os.path.join(args.save_dir, "best.pkl")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best.pkl not found in {args.save_dir}")
    params_obj = util.load_pkl(args.save_dir, "best")
    params = params_obj[0] if isinstance(params_obj, tuple) else params_obj
    params = jnp.asarray(params)

    grid_size = int(args.grid_size)
    split_n = int(args.grid_split)
    if grid_size % split_n != 0:
        raise ValueError(f"grid_size {grid_size} not divisible by grid_split {split_n}")
    block_size = grid_size // split_n

    # global substrate
    if args.substrate == "lenia_flow":
        kw = util.flow_lenia_kwargs_from_args(args)
        global_substrate = substrates.create_substrate(args.substrate, **kw)
    else:
        global_substrate = substrates.create_substrate(args.substrate)
    global_substrate = substrates.FlattenSubstrateParameters(global_substrate)

    rng = jax.random.PRNGKey(args.seed)
    state_global = global_substrate.init_state(rng, params)

    # prepare block substrate and batched block state
    kw_block = util.flow_lenia_kwargs_from_args(args)
    kw_block["grid_size"] = block_size
    block_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate("lenia_flow", **kw_block)
    )

    A0 = np.asarray(state_global["A"])
    P0 = np.asarray(state_global["P"])
    Food0 = np.asarray(state_global.get("Food", np.zeros(A0.shape[:2])))

    B = split_n * split_n
    A_blocks = np.zeros((B, block_size, block_size, int(args.C)), dtype=np.float32)
    P_blocks = np.zeros((B, block_size, block_size, int(args.k)), dtype=np.float32)
    Food_blocks = np.zeros((B, block_size, block_size), dtype=np.float32)
    for bi in range(B):
        i = bi // split_n
        j = bi % split_n
        i0 = i * block_size
        j0 = j * block_size
        A_blocks[bi] = A0[i0:i0 + block_size, j0:j0 + block_size]
        P_blocks[bi] = P0[i0:i0 + block_size, j0:j0 + block_size]
        Food_blocks[bi] = Food0[i0:i0 + block_size, j0:j0 + block_size]

    base_block = block_substrate.init_state(rng, params)
    blocks_state = {}
    for k, v in base_block.items():
        if k == "A":
            blocks_state[k] = jnp.asarray(A_blocks)
        elif k == "P":
            blocks_state[k] = jnp.asarray(P_blocks)
        elif k == "Food":
            blocks_state[k] = jnp.asarray(Food_blocks)
        elif k == "mass_cycle_start":
            blocks_state[k] = jnp.sum(jnp.asarray(A_blocks), axis=(1, 2, 3))
        elif k == "t":
            t0 = state_global.get("t", jnp.array(0, dtype=jnp.int32))
            blocks_state[k] = jnp.broadcast_to(t0, (B,))
        else:
            blocks_state[k] = jnp.broadcast_to(v, (B,) + v.shape)

    # step blocks independently for warmup_steps and render to video
    warmup_steps = int(args.warmup_steps)
    def step_one(state, rng_in):
        return block_substrate.step_state(rng_in, state, params)

    step_blocks_vmap = jax.vmap(step_one, in_axes=(0, 0))
    devices = jax.devices()
    n_dev = len(devices)
    use_pmap = bool(getattr(args, "multi_device", False)) and n_dev > 1 and (B % n_dev == 0)
    if use_pmap:
        per_dev = B // n_dev

        def step_blocks_pmap(state_shard, rng_shard):
            return jax.vmap(step_one, in_axes=(0, 0))(state_shard, rng_shard)

        step_blocks_pmap = jax.pmap(step_blocks_pmap)

    output_dir = getattr(args, "output_dir", None) or args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = getattr(args, "output", None)
    if output_path is None:
        output_path = os.path.join(output_dir, "frustration.mp4")
    # setup video writer
    writer = imageio.get_writer(output_path, fps=int(args.fps), codec=args.codec, macro_block_size=args.macro_block_size)

    def assemble_blocks_jax(blocks_list):
        A_blocks = blocks_list["A"]
        P_blocks = blocks_list["P"]
        Food_blocks = blocks_list.get("Food", None)
        C = int(args.C)
        K = int(args.k)
        H = block_size * split_n
        A_grid = A_blocks.reshape((split_n, split_n, block_size, block_size, C))
        A_grid = jnp.transpose(A_grid, (0, 2, 1, 3, 4))
        A_full = A_grid.reshape((H, H, C))
        P_grid = P_blocks.reshape((split_n, split_n, block_size, block_size, K))
        P_grid = jnp.transpose(P_grid, (0, 2, 1, 3, 4))
        P_full = P_grid.reshape((H, H, K))
        if Food_blocks is None:
            Food_full = jnp.zeros((H, H), dtype=A_full.dtype)
        else:
            F_grid = Food_blocks.reshape((split_n, split_n, block_size, block_size))
            F_grid = jnp.transpose(F_grid, (0, 2, 1, 3))
            Food_full = F_grid.reshape((H, H))
        return A_full, P_full, Food_full

    def render_from_blocks_jax(blocks_list):
        A_pre, P_pre, Food_pre = assemble_blocks_jax(blocks_list)
        state_vis = {"A": A_pre, "P": P_pre, "Food": Food_pre}
        return global_substrate.render_state(state_vis, params, img_size=int(args.img_size))

    def render_from_blocks_jax_sharded(blocks_list):
        blocks_flat = jax.tree.map(
            lambda x: x.reshape((B,) + x.shape[2:]),
            blocks_list,
        )
        A_pre, P_pre, Food_pre = assemble_blocks_jax(blocks_flat)
        state_vis = {"A": A_pre, "P": P_pre, "Food": Food_pre}
        return global_substrate.render_state(state_vis, params, img_size=int(args.img_size))

    metrics_cfg = cfg.get("evolutionary_metrics", {})
    metrics_enabled = bool(metrics_cfg.get("enabled", False))
    if metrics_enabled:
        frames_u8 = []
        masses = []

    def mass_from_A(A):
        m = jnp.sum(A, axis=-1)
        if args.img_size is not None:
            m = jax.image.resize(m[..., None], (int(args.img_size), int(args.img_size), 1), method="nearest")[..., 0]
        return m

    def render_and_mass_from_blocks_jax(blocks_list):
        A_pre, P_pre, Food_pre = assemble_blocks_jax(blocks_list)
        state_vis = {"A": A_pre, "P": P_pre, "Food": Food_pre}
        img = global_substrate.render_state(state_vis, params, img_size=int(args.img_size))
        mass = mass_from_A(A_pre)
        return img, mass

    def render_and_mass_from_blocks_jax_sharded(blocks_list):
        blocks_flat = jax.tree.map(
            lambda x: x.reshape((B,) + x.shape[2:]),
            blocks_list,
        )
        A_pre, P_pre, Food_pre = assemble_blocks_jax(blocks_flat)
        state_vis = {"A": A_pre, "P": P_pre, "Food": Food_pre}
        img = global_substrate.render_state(state_vis, params, img_size=int(args.img_size))
        mass = mass_from_A(A_pre)
        return img, mass

    # render initial state before any warmup
    first_frame, first_mass = render_and_mass_from_blocks_jax(blocks_state)
    first_frame = np.asarray(first_frame)
    writer.append_data((np.clip(first_frame, 0.0, 1.0) * 255).astype(np.uint8))
    if metrics_enabled:
        frames_u8.append((np.clip(first_frame, 0.0, 1.0) * 255).astype(np.uint8))
        masses.append(np.asarray(first_mass))

    if warmup_steps > 0:
        warmup_batch_steps = int(args.batch_steps)
        warmup_jit_micro = int(args.jit_microbatch)
        warmup_stepper_cache = {}

        def build_warmup_stepper(mb: int):
            def run_batch(state, rng_in):
                rngs = split(rng_in, mb)
                frames0 = jnp.zeros((mb, int(args.img_size), int(args.img_size), 3), dtype=jnp.float32)
                masses0 = jnp.zeros((mb, int(args.img_size), int(args.img_size)), dtype=jnp.float32)

                def body(i, carry):
                    s, frames, masses = carry
                    rkeys = split(rngs[i], B)
                    s = step_blocks_vmap(s, rkeys)
                    frame, mass = render_and_mass_from_blocks_jax(s)
                    frames = frames.at[i].set(frame)
                    masses = masses.at[i].set(mass)
                    return (s, frames, masses)

                state_next, frames, masses = jax.lax.fori_loop(0, mb, body, (state, frames0, masses0))
                return state_next, frames, masses

            return jax.jit(run_batch)

        def build_warmup_stepper_pmap(mb: int):
            def run_batch(state_shard, rng_in):
                rngs = split(rng_in, mb)
                frames0 = jnp.zeros((mb, int(args.img_size), int(args.img_size), 3), dtype=jnp.float32)
                masses0 = jnp.zeros((mb, int(args.img_size), int(args.img_size)), dtype=jnp.float32)

                def body(i, carry):
                    s, frames, masses = carry
                    rkeys = split(rngs[i], B).reshape((n_dev, per_dev, 2))
                    s = step_blocks_pmap(s, rkeys)
                    frame, mass = render_and_mass_from_blocks_jax_sharded(s)
                    frames = frames.at[i].set(frame)
                    masses = masses.at[i].set(mass)
                    return (s, frames, masses)

                state_next, frames, masses = jax.lax.fori_loop(0, mb, body, (state_shard, frames0, masses0))
                return state_next, frames, masses

            return jax.jit(run_batch)

        def get_warmup_stepper(mb: int):
            if mb not in warmup_stepper_cache:
                if use_pmap:
                    warmup_stepper_cache[mb] = build_warmup_stepper_pmap(mb)
                else:
                    warmup_stepper_cache[mb] = build_warmup_stepper(mb)
            return warmup_stepper_cache[mb]

        steps_done = 0
        pbar = tqdm(total=warmup_steps, desc="warmup", leave=False)
        if use_pmap:
            blocks_state_warm = jax.tree.map(
                lambda x: x.reshape((n_dev, per_dev) + x.shape[1:]),
                blocks_state,
            )
        else:
            blocks_state_warm = blocks_state
        while steps_done < warmup_steps:
            cur = min(warmup_batch_steps, warmup_steps - steps_done)
            inner = 0
            while inner < cur:
                m = min(warmup_jit_micro, cur - inner)
                rng, _rng = split(rng)
                stepper = get_warmup_stepper(m)
                blocks_state_warm, frames, mass_batch = stepper(blocks_state_warm, _rng)
                frames = np.asarray(frames)
                frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
                mass_batch = np.asarray(mass_batch)
                for i in range(frames.shape[0]):
                    f = frames[i]
                    writer.append_data(f)
                    if metrics_enabled:
                        frames_u8.append(f)
                        masses.append(mass_batch[i])
                inner += m
            steps_done += cur
            pbar.update(cur)
        if use_pmap:
            blocks_state = jax.tree.map(
                lambda x: x.reshape((B,) + x.shape[2:]),
                blocks_state_warm,
            )
        else:
            blocks_state = blocks_state_warm

    # save state before walls removed (after warmup)
    A_pre, P_pre, Food_pre = assemble_blocks(blocks_state, split_n, block_size, int(args.C))
    pre_state = dict(A=A_pre, P=P_pre, Food=Food_pre)
    util.save_pkl(output_dir, "state_before_walls", pre_state)

    # merge into global state
    base_global = global_substrate.init_state(rng, params)
    state_merged = dict(base_global)
    state_merged["A"] = jnp.asarray(A_pre)
    state_merged["P"] = jnp.asarray(P_pre)
    state_merged["Food"] = jnp.asarray(Food_pre)
    state_merged["t"] = jnp.array(warmup_steps, dtype=jnp.int32)
    state_merged["mass_cycle_start"] = jnp.sum(state_merged["A"])

    # write one frame right before walls removed (merged state)
    def render_state(state):
        return np.asarray(global_substrate.render_state(state, params, img_size=int(args.img_size)))
    merged_frame = render_state(state_merged)
    writer.append_data((merged_frame * 255).astype(np.uint8))
    if metrics_enabled:
        frames_u8.append((np.clip(merged_frame, 0.0, 1.0) * 255).astype(np.uint8))
        merged_mass = mass_from_A(state_merged["A"])
        masses.append(np.asarray(merged_mass))

    # simulate after walls removed
    max_steps = int(args.max_steps)
    remaining = max(0, max_steps - warmup_steps)
    batch_steps = int(args.batch_steps)
    jit_micro = int(args.jit_microbatch)

    def build_batch_stepper(mb: int):
        def run_batch(state, rng_in):
            rngs = split(rng_in, mb)
            frames0 = jnp.zeros((mb, int(args.img_size), int(args.img_size), 3), dtype=jnp.float32)
            masses0 = jnp.zeros((mb, int(args.img_size), int(args.img_size)), dtype=jnp.float32)

            def body(i, carry):
                s, frames, masses = carry
                s = global_substrate.step_state(rngs[i], s, params)
                frame = global_substrate.render_state(s, params, img_size=int(args.img_size))
                mass = mass_from_A(s["A"])
                frames = frames.at[i].set(frame)
                masses = masses.at[i].set(mass)
                return (s, frames, masses)

            state_next, frames, masses = jax.lax.fori_loop(0, mb, body, (state, frames0, masses0))
            return state_next, frames, masses

        return jax.jit(run_batch)

    stepper_cache = {}

    def get_stepper(mb: int):
        if mb not in stepper_cache:
            stepper_cache[mb] = build_batch_stepper(mb)
        return stepper_cache[mb]

    state = state_merged
    steps_done = 0
    pbar = tqdm(total=remaining, desc="after_walls", leave=True)
    while steps_done < remaining:
        cur = min(batch_steps, remaining - steps_done)
        # inner micro-batches
        inner = 0
        while inner < cur:
            m = min(jit_micro, cur - inner)
            rng, _rng = split(rng)
            stepper = get_stepper(m)
            state, frames, mass_batch = stepper(state, _rng)
            frames = np.asarray(frames)
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
            mass_batch = np.asarray(mass_batch)
            for i in range(frames.shape[0]):
                f = frames[i]
                writer.append_data(f)
                if metrics_enabled:
                    frames_u8.append(f)
                    masses.append(mass_batch[i])
            inner += m
        steps_done += cur
        pbar.update(cur)

    writer.close()

    # save final state
    final_state = dict(
        A=to_np(state["A"]),
        P=to_np(state["P"]),
        Food=to_np(state.get("Food", None)),
    )
    util.save_pkl(output_dir, "state_final", final_state)

    if metrics_enabled:
        frames_rgb_u8 = np.stack(frames_u8, axis=0)
        frames_mass = np.stack(masses, axis=0)
        masses_T, centers, palette = evo.compute_mass_trajectories(
            frames_rgb_u8,
            frames_mass,
            q=int(metrics_cfg.get("q", 4)),
            min_bin_mass=float(metrics_cfg.get("min_bin_mass", 0.0)),
            lam=float(metrics_cfg.get("lam", 18.0)),
            dp_iters=int(metrics_cfg.get("dp_iters", 8)),
            mass_eps_rel=float(metrics_cfg.get("mass_eps_rel", 1e-6)),
            rgb_void=int(metrics_cfg.get("rgb_void", 3)),
            max_rgb_dist=float(metrics_cfg.get("max_rgb_dist", 0.0)),
            backend=metrics_cfg.get("backend", None),
            return_labels=False,
        )
        metrics_dir = metrics_cfg.get("output_dir", None) or output_dir
        os.makedirs(metrics_dir, exist_ok=True)
        prefix = metrics_cfg.get("save_prefix", "evo_metrics")
        np.save(os.path.join(metrics_dir, f"{prefix}_masses.npy"), masses_T)
        np.save(os.path.join(metrics_dir, f"{prefix}_centers.npy"), centers)
        np.save(os.path.join(metrics_dir, f"{prefix}_palette.npy"), palette)

        # plot trajectories
        include_void = bool(metrics_cfg.get("include_void", False))
        top_k = int(metrics_cfg.get("top_k", 12))
        mass_floor = float(metrics_cfg.get("mass_floor", 0.0))
        logy = bool(metrics_cfg.get("logy", False))
        figsize = metrics_cfg.get("figsize", (12, 6))
        title = metrics_cfg.get("title", None)
        M = np.asarray(masses_T, dtype=np.float64)
        T, L = M.shape
        totals = M.sum(axis=0)
        start = 0 if include_void else 1
        idx = np.arange(start, L, dtype=int)
        idx = idx[totals[idx] >= mass_floor]
        idx = idx[np.argsort(-totals[idx])][:top_k]
        x = np.arange(T)
        fig, ax = plt.subplots(figsize=figsize)
        for lab in idx:
            y = M[:, lab]
            c = np.asarray(palette[lab], dtype=np.float64) / 255.0
            ax.plot(x, y, label=str(lab), color=c)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("total mass")
        ax.set_title(title or "Mass trajectories per label")
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(metrics_dir, f"{prefix}_masses.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
