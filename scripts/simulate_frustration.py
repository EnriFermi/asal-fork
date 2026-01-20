import os
import sys

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import imageio
import imageio.v3 as iio

import substrates
from rollout import rollout_simulation
import util
from omegaconf import OmegaConf


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/simulate_frustration.py <config.yaml>")
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

    # prepare block substrates
    blocks = []
    kw_block = util.flow_lenia_kwargs_from_args(args)
    kw_block["grid_size"] = block_size
    block_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate("lenia_flow", **kw_block)
    )

    A0 = np.asarray(state_global["A"])
    P0 = np.asarray(state_global["P"])
    Food0 = np.asarray(state_global.get("Food", np.zeros(A0.shape[:2])))

    for bi in range(split_n * split_n):
        i = bi // split_n
        j = bi % split_n
        i0 = i * block_size
        j0 = j * block_size
        blk_state = block_substrate.init_state(rng, params)
        blk_state = dict(blk_state)
        blk_state["A"] = jnp.asarray(A0[i0:i0 + block_size, j0:j0 + block_size])
        blk_state["P"] = jnp.asarray(P0[i0:i0 + block_size, j0:j0 + block_size])
        blk_state["Food"] = jnp.asarray(Food0[i0:i0 + block_size, j0:j0 + block_size])
        blk_state["t"] = state_global.get("t", jnp.array(0, dtype=jnp.int32))
        blk_state["mass_cycle_start"] = jnp.sum(blk_state["A"])
        blocks.append(blk_state)

    # step blocks independently for warmup_steps and render to video
    warmup_steps = int(args.warmup_steps)
    def step_one(state, rng_in):
        return block_substrate.step_state(rng_in, state, params)

    step_one_jit = jax.jit(step_one)

    output_dir = getattr(args, "output_dir", None) or args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = getattr(args, "output", None)
    if output_path is None:
        output_path = os.path.join(output_dir, "frustration.mp4")
    # setup video writer
    writer = imageio.get_writer(output_path, fps=int(args.fps), codec=args.codec, macro_block_size=args.macro_block_size)

    def render_from_blocks(blocks_list):
        A_pre, P_pre, Food_pre = assemble_blocks(blocks_list, split_n, block_size, int(args.C))
        state_vis = {"A": jnp.asarray(A_pre), "P": jnp.asarray(P_pre), "Food": jnp.asarray(Food_pre)}
        img = global_substrate.render_state(state_vis, params, img_size=int(args.img_size))
        return np.asarray(img)

    # render initial state before any warmup
    writer.append_data((render_from_blocks(blocks) * 255).astype(np.uint8))

    if warmup_steps > 0:
        for _ in range(warmup_steps):
            rng, _rng = split(rng)
            rkeys = split(_rng, len(blocks))
            blocks = [step_one_jit(s, r) for s, r in zip(blocks, rkeys)]
            writer.append_data((render_from_blocks(blocks) * 255).astype(np.uint8))

    # save state before walls removed (after warmup)
    A_pre, P_pre, Food_pre = assemble_blocks(blocks, split_n, block_size, int(args.C))
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
    writer.append_data((render_state(state_merged) * 255).astype(np.uint8))

    # simulate after walls removed
    max_steps = int(args.max_steps)
    remaining = max(0, max_steps - warmup_steps)
    batch_steps = int(args.batch_steps)
    jit_micro = int(args.jit_microbatch)

    def step_batch(state, rng_in, steps):
        def step_fn(s, r):
            ns = global_substrate.step_state(r, s, params)
            return ns, ns
        rngs = split(rng_in, steps)
        s_final, s_hist = jax.lax.scan(step_fn, state, rngs)
        return s_final, s_hist

    step_batch_jit = jax.jit(step_batch)

    state = state_merged
    steps_done = 0
    while steps_done < remaining:
        cur = min(batch_steps, remaining - steps_done)
        # inner micro-batches
        inner = 0
        while inner < cur:
            m = min(jit_micro, cur - inner)
            rng, _rng = split(rng)
            state, s_hist = step_batch_jit(state, _rng, m)
            frames = np.asarray(jax.vmap(lambda s: global_substrate.render_state(s, params, img_size=int(args.img_size)))(s_hist))
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
            for f in frames:
                writer.append_data(f)
            inner += m
        steps_done += cur

    writer.close()

    # save final state
    final_state = dict(
        A=to_np(state["A"]),
        P=to_np(state["P"]),
        Food=to_np(state.get("Food", None)),
    )
    util.save_pkl(output_dir, "state_final", final_state)


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
