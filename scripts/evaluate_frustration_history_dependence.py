import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

import asal_metrics
import foundation_models
import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config


def _patch_wandb_pandas_check() -> None:
    try:
        import wandb.util as wandb_util
    except Exception:
        return
    orig = getattr(wandb_util, "is_pandas_data_frame", None)
    if orig is None:
        return

    def _safe_is_pandas_data_frame(val):
        try:
            return orig(val)
        except Exception:
            return False

    wandb_util.is_pandas_data_frame = _safe_is_pandas_data_frame


_patch_wandb_pandas_check()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_like: str | None, root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return root / path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "run"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/evaluate_frustration_history_dependence.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("source", {}),
        cfg.get("substrate", {}),
        cfg.get("protocol", {}),
        cfg.get("evaluation", {}),
        cfg.get("metric", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def _load_params(args, project_root: Path) -> jax.Array:
    params_path = _resolve_path(getattr(args, "params_path", None), project_root)
    if params_path is not None:
        if not params_path.exists():
            raise FileNotFoundError(f"params_path does not exist: {params_path}")
        if params_path.suffix == ".npy":
            params_obj = np.load(params_path)
        elif params_path.suffix == ".npz":
            with np.load(params_path, allow_pickle=False) as data:
                if "params" not in data.files:
                    raise ValueError(f"{params_path} does not contain array 'params'.")
                params_obj = data["params"]
        elif params_path.suffix == ".pkl":
            params_obj = util.load_pkl(str(params_path.parent), params_path.stem)
        else:
            raise ValueError(f"Unsupported params_path suffix for {params_path}. Use .pkl, .npy, or .npz.")
    else:
        checkpoint_dir = _resolve_path(
            getattr(args, "checkpoint_dir", getattr(args, "source_save_dir", None)),
            project_root,
        )
        if checkpoint_dir is None:
            raise ValueError("Set source.checkpoint_dir or source.params_path.")
        params_name = str(getattr(args, "params_name", "best"))
        params_obj = util.load_pkl(str(checkpoint_dir), params_name)
        if params_obj is None:
            raise FileNotFoundError(f"{params_name}.pkl not found in {checkpoint_dir}")
    params = params_obj[0] if isinstance(params_obj, tuple) else params_obj
    return jnp.asarray(np.asarray(params, dtype=np.float32))


def _create_substrate(args, *, enable_msc: bool = False):
    if args.substrate == "lenia_flow":
        kw = util.flow_lenia_kwargs_from_args(args)
        if enable_msc:
            kw["debug_return_F"] = True
        base = substrates.create_substrate(
            args.substrate,
            **kw,
        )
    else:
        base = substrates.create_substrate(
            args.substrate,
            **util.substrate_kwargs_from_args(args),
        )
    return substrates.FlattenSubstrateParameters(base)


def _resolve_window(args) -> tuple[int, int]:
    total_steps = int(getattr(args, "total_steps"))
    warmup_steps = int(getattr(args, "warmup_steps"))
    end = getattr(args, "late_window_end_steps", None)
    end = total_steps if end is None else int(end)
    start = getattr(args, "late_window_start_steps", None)
    if start is None:
        size = getattr(args, "late_window_size_steps", None)
        if size is None:
            raise ValueError("Specify evaluation.late_window_start_steps or evaluation.late_window_size_steps.")
        start = end - int(size)
    else:
        start = int(start)
    if total_steps < 1:
        raise ValueError("protocol.total_steps must be >= 1.")
    if warmup_steps < 0:
        raise ValueError("protocol.warmup_steps must be >= 0.")
    if not (0 <= start < end <= total_steps):
        raise ValueError(
            f"Late window must satisfy 0 <= start < end <= total_steps. "
            f"Got start={start}, end={end}, total_steps={total_steps}."
        )
    if start < warmup_steps:
        raise ValueError(
            f"Late window must start after walls are removed. "
            f"Got late_window_start_steps={start}, warmup_steps={warmup_steps}."
        )
    return start, end


def _init_lagrangian_points_jax(
    A0: jax.Array,
    *,
    n_particles: int,
    init_mode: str,
    border: str,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    sx = int(A0.shape[0])
    sy = int(A0.shape[1])
    init_mode = str(init_mode).strip().lower()
    if init_mode == "uniform":
        k0, k1 = jax.random.split(key)
        y = jax.random.uniform(k0, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(k1, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif init_mode == "mass":
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(total > 0.0, flat / jnp.maximum(total, 1e-12), jnp.ones_like(flat) / flat.size)
        k_idx, k_jit = jax.random.split(key)
        idx = jax.random.choice(k_idx, flat.size, shape=(n_particles,), replace=True, p=probs)
        iy = idx // sy
        ix = idx % sy
        jitter = jax.random.uniform(k_jit, (n_particles, 2), minval=-0.49, maxval=0.49)
        pts = jnp.stack((iy.astype(jnp.float32) + 0.5, ix.astype(jnp.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unknown metric_lagrangian_init_mode={init_mode!r}. Use 'mass' or 'uniform'.")

    if border == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
        pts = jnp.stack((y, x), axis=-1)
    else:
        lo = float(sigma)
        hi_y = float(sx - sigma)
        hi_x = float(sy - sigma)
        y = jnp.clip(pts[:, 0], lo, hi_y)
        x = jnp.clip(pts[:, 1], lo, hi_x)
        pts = jnp.stack((y, x), axis=-1)
    return pts.astype(jnp.float32)


def _prepare_block_template_state(
    *,
    initial_state,
    block_template,
    split_n: int,
    block_size: int,
    pad: int,
    C: int,
    k: int,
):
    block_sim_size = block_size + 2 * pad
    A0 = np.asarray(initial_state["A"], dtype=np.float32)
    P0 = np.asarray(initial_state["P"], dtype=np.float32)
    Food0 = np.asarray(initial_state.get("Food", np.zeros(A0.shape[:2], dtype=np.float32)), dtype=np.float32)

    B = split_n * split_n
    A_blocks = np.zeros((B, block_sim_size, block_sim_size, C), dtype=np.float32)
    P_blocks = np.zeros((B, block_sim_size, block_sim_size, k), dtype=np.float32)
    Food_blocks = np.zeros((B, block_sim_size, block_sim_size), dtype=np.float32)

    for bi in range(B):
        i = bi // split_n
        j = bi % split_n
        i0 = i * block_size
        j0 = j * block_size
        A_blocks[bi, pad:pad + block_size, pad:pad + block_size] = A0[i0:i0 + block_size, j0:j0 + block_size]
        P_blocks[bi, pad:pad + block_size, pad:pad + block_size] = P0[i0:i0 + block_size, j0:j0 + block_size]
        Food_blocks[bi, pad:pad + block_size, pad:pad + block_size] = Food0[i0:i0 + block_size, j0:j0 + block_size]

    blocks_state = {}
    for key, value in block_template.items():
        if key == "A":
            blocks_state[key] = jnp.asarray(A_blocks)
        elif key == "P":
            blocks_state[key] = jnp.asarray(P_blocks)
        elif key == "Food":
            blocks_state[key] = jnp.asarray(Food_blocks)
        elif key == "mass_cycle_start":
            blocks_state[key] = jnp.sum(jnp.asarray(A_blocks), axis=(1, 2, 3))
        elif key == "t":
            t0 = initial_state.get("t", jnp.array(0, dtype=jnp.int32))
            blocks_state[key] = jnp.broadcast_to(t0, (B,))
        else:
            blocks_state[key] = jnp.broadcast_to(value, (B,) + value.shape)
    return blocks_state


def _assemble_blocks_jax(blocks_state, *, split_n: int, block_size: int, pad: int):
    A_blocks = blocks_state["A"]
    P_blocks = blocks_state["P"]
    Food_blocks = blocks_state.get("Food", None)

    if pad > 0:
        A_blocks = A_blocks[:, pad:pad + block_size, pad:pad + block_size, :]
        P_blocks = P_blocks[:, pad:pad + block_size, pad:pad + block_size, :]
        if Food_blocks is not None:
            Food_blocks = Food_blocks[:, pad:pad + block_size, pad:pad + block_size]

    C = int(A_blocks.shape[-1])
    K = int(P_blocks.shape[-1])
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


def _merge_blocks_into_global_state(initial_state, blocks_state, *, split_n: int, block_size: int, pad: int):
    A_full, P_full, Food_full = _assemble_blocks_jax(
        blocks_state,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
    )
    merged = dict(initial_state)
    merged["A"] = A_full
    merged["P"] = P_full
    merged["Food"] = Food_full
    if "t" in blocks_state:
        merged["t"] = jnp.asarray(blocks_state["t"][0])
    if "mass_cycle_start" in initial_state or "mass_cycle_start" in blocks_state:
        merged["mass_cycle_start"] = jnp.sum(A_full)
    return merged


def _build_state_advancer(substrate, steps: int):
    if steps <= 0:
        return lambda rng_key, state0, params_in: state0

    def advance_fn(rng_key, state0, params_in):
        def body_fn(state, step_key):
            return substrate.step_state(step_key, state, params_in), None

        state_final, _ = jax.lax.scan(body_fn, state0, jax.random.split(rng_key, steps))
        return state_final

    return jax.jit(advance_fn)


def _build_state_chunk_stepper(substrate):
    cache = {}

    def get_stepper(steps: int):
        steps = int(steps)
        if steps <= 0:
            return lambda step_keys, state0, params_in: state0
        if steps not in cache:
            def advance_fn(step_keys, state0, params_in):
                def body_fn(state, step_key):
                    return substrate.step_state(step_key, state, params_in), None

                state_final, _ = jax.lax.scan(body_fn, state0, step_keys)
                return state_final

            cache[steps] = jax.jit(advance_fn)
        return cache[steps]

    return get_stepper


def _build_block_warmupper(block_substrate, n_blocks: int, steps: int):
    if steps <= 0:
        return lambda rng_key, state0, params_in: state0

    def block_step(state, key, params_in):
        return block_substrate.step_state(key, state, params_in)

    vmapped_step = jax.vmap(block_step, in_axes=(0, 0, None))

    def warmup_fn(rng_key, state0, params_in):
        def body_fn(state, step_key):
            block_keys = jax.random.split(step_key, n_blocks)
            return vmapped_step(state, block_keys, params_in), None

        state_final, _ = jax.lax.scan(body_fn, state0, jax.random.split(rng_key, steps))
        return state_final

    return jax.jit(warmup_fn)


def _build_embedding_rollout(substrate, fm, *, window_steps: int, time_sampling: int, img_size: int):
    if window_steps <= 0:
        raise ValueError("Late window must have at least one step.")
    if time_sampling < 1:
        raise ValueError("evaluation.time_sampling must be >= 1.")
    if window_steps % time_sampling != 0:
        raise ValueError(
            f"Late window length ({window_steps}) must be divisible by time_sampling ({time_sampling})."
        )
    chunk_steps = window_steps // time_sampling

    def rollout_fn(rng_key, state0, params_in):
        def step_fn(state, step_key):
            return substrate.step_state(step_key, state, params_in), None

        def chunk_fn(state, chunk_key):
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(chunk_key, chunk_steps))
            img = substrate.render_state(state_next, params=params_in, img_size=img_size)
            z = fm.embed_img(img)
            return state_next, z

        _, z_seq = jax.lax.scan(chunk_fn, state0, jax.random.split(rng_key, time_sampling))
        return z_seq

    return jax.jit(rollout_fn)


def _build_embedding_chunk_stepper(substrate, fm, *, img_size: int):
    cache = {}

    def get_stepper(steps: int):
        steps = int(steps)
        if steps <= 0:
            raise ValueError("Embedding chunk size must be >= 1.")
        if steps not in cache:
            def rollout_chunk(rng_key, state0, params_in):
                def step_fn(state, step_key):
                    return substrate.step_state(step_key, state, params_in), None

                state_next, _ = jax.lax.scan(step_fn, state0, jax.random.split(rng_key, steps))
                img = substrate.render_state(state_next, params=params_in, img_size=img_size)
                z = fm.embed_img(img)
                return state_next, z

            cache[steps] = jax.jit(rollout_chunk)
        return cache[steps]

    return get_stepper


def _build_lagrangian_rollout(
    substrate,
    *,
    rollout_steps: int,
    metric_cfg: dict,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_flow_channel: int,
    lag_flow_reduce: str,
    lag_channel_mode: str,
    lag_noise_model: str,
    lag_diffusion_scale: float,
):
    if rollout_steps <= 0:
        raise ValueError("MSC rollout_steps must be >= 1.")
    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    if rollout_steps != chunk_steps * time_sampling:
        raise ValueError(
            f"MSC rollout mismatch: rollout_steps={rollout_steps}, "
            f"sample_every_steps={chunk_steps}, time_sampling={time_sampling}."
        )

    def rollout_fn(rng_key, state0, params_in):
        if "F" not in state0:
            raise ValueError(
                "State does not contain flow field F. "
                "For FlowLenia set debug_return_F=true before MSC evaluation."
            )
        if not hasattr(substrate, "RT"):
            raise ValueError("Substrate does not provide RT for lagrangian advection.")

        rt = substrate.RT
        k_pts, k_ch, k_scan = jax.random.split(rng_key, 3)
        pts0 = _init_lagrangian_points_jax(
            state0["A"],
            n_particles=lag_n_particles,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, state0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)

        def step_fn(state, key_step):
            st, pts, ch = state
            st = substrate.step_state(key_step, st, params_in)
            lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
            pts, ch = rt.advect_particles(
                points=pts,
                F=st["F"],
                A=st["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=ch,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return (st, pts, ch), None

        def chunk_fn(state, key_chunk):
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, chunk_steps))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (state0, pts0, ch0),
            jax.random.split(k_scan, time_sampling),
        )
        return xy_seq

    return jax.jit(rollout_fn)


def _build_lagrangian_chunk_stepper(
    substrate,
    *,
    chunk_steps: int,
    lag_flow_channel: int,
    lag_flow_reduce: str,
    lag_channel_mode: str,
    lag_noise_model: str,
    lag_diffusion_scale: float,
):
    chunk_steps = int(chunk_steps)
    if chunk_steps <= 0:
        raise ValueError("Lagrangian chunk size must be >= 1.")

    def rollout_chunk(rng_key, carry0, params_in):
        rt = substrate.RT

        def step_fn(state, key_step):
            st, pts, ch = state
            st = substrate.step_state(key_step, st, params_in)
            lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
            pts, ch = rt.advect_particles(
                points=pts,
                F=st["F"],
                A=st["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=ch,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return (st, pts, ch), None

        carry_next, _ = jax.lax.scan(step_fn, carry0, jax.random.split(rng_key, chunk_steps))
        return carry_next, carry_next[1]

    return jax.jit(rollout_chunk)


def _advance_state_with_progress(
    *,
    get_stepper,
    rng_key,
    state0,
    params_in,
    steps: int,
    chunk_steps: int,
    desc: str,
    show_progress: bool,
):
    if steps <= 0:
        return state0
    chunk_steps = max(1, int(chunk_steps))
    all_step_keys = jax.random.split(rng_key, int(steps))
    state = state0
    steps_done = 0
    pbar = tqdm(total=int(steps), desc=desc, leave=False, dynamic_ncols=True, disable=not show_progress)
    try:
        while steps_done < steps:
            cur = min(chunk_steps, steps - steps_done)
            step_keys = all_step_keys[steps_done:steps_done + cur]
            state = get_stepper(cur)(step_keys, state, params_in)
            steps_done += cur
            pbar.update(cur)
    finally:
        pbar.close()
    return state


def _rollout_embeddings_with_progress(
    *,
    get_stepper,
    rng_key,
    state0,
    params_in,
    window_steps: int,
    time_sampling: int,
    desc: str,
    show_progress: bool,
):
    if window_steps <= 0:
        raise ValueError("Embedding rollout window_steps must be >= 1.")
    if time_sampling < 1 or window_steps % time_sampling != 0:
        raise ValueError(
            f"Embedding rollout requires window_steps % time_sampling == 0, got {window_steps} and {time_sampling}."
        )
    chunk_steps = window_steps // time_sampling
    state = state0
    zs = []
    chunk_keys = jax.random.split(rng_key, int(time_sampling))
    pbar = tqdm(total=int(window_steps), desc=desc, leave=False, dynamic_ncols=True, disable=not show_progress)
    try:
        for key_chunk in chunk_keys:
            state, z = get_stepper(chunk_steps)(key_chunk, state, params_in)
            zs.append(np.asarray(jax.device_get(z), dtype=np.float32))
            pbar.update(chunk_steps)
    finally:
        pbar.close()
    return np.stack(zs, axis=0)


def _init_lagrangian_carry(
    *,
    substrate,
    state0,
    key_pts,
    key_ch,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_channel_mode: str,
):
    if "F" not in state0:
        raise ValueError(
            "State does not contain flow field F. "
            "For FlowLenia set debug_return_F=true before MSC evaluation."
        )
    if not hasattr(substrate, "RT"):
        raise ValueError("Substrate does not provide RT for lagrangian advection.")

    rt = substrate.RT
    pts0 = _init_lagrangian_points_jax(
        state0["A"],
        n_particles=lag_n_particles,
        init_mode=lag_init_mode,
        border=str(getattr(rt, "border", "wall")),
        sigma=float(getattr(rt, "sigma", 0.0)),
        key=key_pts,
    )
    if lag_channel_mode in ("fixed", "resample"):
        ch0 = rt.sample_point_channels(pts0, state0["A"], key_ch)
    else:
        ch0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)
    return (state0, pts0, ch0)


def _rollout_lagrangian_with_progress(
    *,
    chunk_stepper,
    substrate,
    rng_key,
    state0,
    params_in,
    time_sampling: int,
    chunk_steps: int,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_channel_mode: str,
    desc: str,
    show_progress: bool,
):
    if time_sampling < 1:
        raise ValueError("Lagrangian rollout time_sampling must be >= 1.")
    k_pts, k_ch, k_scan = jax.random.split(rng_key, 3)
    carry = _init_lagrangian_carry(
        substrate=substrate,
        state0=state0,
        key_pts=k_pts,
        key_ch=k_ch,
        lag_n_particles=lag_n_particles,
        lag_init_mode=lag_init_mode,
        lag_channel_mode=lag_channel_mode,
    )
    xy_seq = []
    chunk_keys = jax.random.split(k_scan, int(time_sampling))
    pbar = tqdm(
        total=int(time_sampling * chunk_steps),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for key_chunk in chunk_keys:
            carry, xy = chunk_stepper(key_chunk, carry, params_in)
            xy_seq.append(np.asarray(jax.device_get(xy), dtype=np.float32))
            pbar.update(chunk_steps)
    finally:
        pbar.close()
    return np.stack(xy_seq, axis=0)


def _sequence_distance(z_a: np.ndarray, z_b: np.ndarray, metric: str) -> tuple[float, np.ndarray]:
    a = np.asarray(z_a, dtype=np.float32)
    b = np.asarray(z_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Embedding sequence shapes must match, got {a.shape} vs {b.shape}.")

    metric = str(metric).strip().lower()
    if metric == "cosine_mean":
        per_t = 1.0 - np.sum(a * b, axis=-1)
    elif metric == "euclidean_mean":
        per_t = np.linalg.norm(a - b, axis=-1)
    elif metric == "sqeuclidean_mean":
        diff = a - b
        per_t = np.sum(diff * diff, axis=-1)
    elif metric == "cosine_last":
        per_t = np.array([1.0 - float(np.sum(a[-1] * b[-1]))], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported evaluation.distance_metric={metric!r}. "
            "Use one of ['cosine_mean', 'euclidean_mean', 'sqeuclidean_mean', 'cosine_last']."
        )
    return float(np.mean(per_t)), np.asarray(per_t, dtype=np.float32)


def _summarize_numeric_pairs(baseline: np.ndarray, effect: np.ndarray) -> dict:
    diff = effect - baseline
    ratio = effect / np.maximum(baseline, 1e-12)
    n_trials = int(baseline.shape[0])

    wilcoxon_greater = None
    wilcoxon_two_sided = None
    if n_trials > 0 and np.any(np.abs(diff) > 0):
        try:
            wilcoxon_greater = float(scipy_stats.wilcoxon(effect, baseline, alternative="greater").pvalue)
        except Exception:
            wilcoxon_greater = None
        try:
            wilcoxon_two_sided = float(scipy_stats.wilcoxon(effect, baseline, alternative="two-sided").pvalue)
        except Exception:
            wilcoxon_two_sided = None

    gt_count = int(np.sum(diff > 0))
    ge_count = int(np.sum(diff >= 0))
    sign_test_greater = None
    if n_trials > 0:
        try:
            sign_test_greater = float(scipy_stats.binomtest(gt_count, n_trials, 0.5, alternative="greater").pvalue)
        except Exception:
            sign_test_greater = None

    return {
        "n_trials": n_trials,
        "mean_baseline": float(np.mean(baseline)),
        "std_baseline": float(np.std(baseline, ddof=1) if n_trials > 1 else 0.0),
        "mean_effect": float(np.mean(effect)),
        "std_effect": float(np.std(effect, ddof=1) if n_trials > 1 else 0.0),
        "mean_effect_minus_baseline": float(np.mean(diff)),
        "median_effect_minus_baseline": float(np.median(diff)),
        "std_effect_minus_baseline": float(np.std(diff, ddof=1) if n_trials > 1 else 0.0),
        "mean_effect_over_baseline_ratio": float(np.mean(ratio)),
        "median_effect_over_baseline_ratio": float(np.median(ratio)),
        "fraction_effect_gt_baseline": float(np.mean(diff > 0)),
        "fraction_effect_ge_baseline": float(np.mean(diff >= 0)),
        "gt_count": gt_count,
        "ge_count": ge_count,
        "wilcoxon_greater_pvalue": wilcoxon_greater,
        "wilcoxon_two_sided_pvalue": wilcoxon_two_sided,
        "sign_test_greater_pvalue": sign_test_greater,
    }


def _summarize_trials(rows: list[dict]) -> dict:
    if not rows:
        return {}
    baseline = np.asarray([float(r["baseline_distance"]) for r in rows], dtype=np.float64)
    effect = np.asarray([float(r["walls_effect_distance"]) for r in rows], dtype=np.float64)
    base = _summarize_numeric_pairs(baseline, effect)
    return {
        "n_trials": int(base["n_trials"]),
        "mean_baseline_distance": float(base["mean_baseline"]),
        "std_baseline_distance": float(base["std_baseline"]),
        "mean_walls_effect_distance": float(base["mean_effect"]),
        "std_walls_effect_distance": float(base["std_effect"]),
        "mean_effect_minus_baseline": float(base["mean_effect_minus_baseline"]),
        "median_effect_minus_baseline": float(base["median_effect_minus_baseline"]),
        "std_effect_minus_baseline": float(base["std_effect_minus_baseline"]),
        "mean_effect_over_baseline_ratio": float(base["mean_effect_over_baseline_ratio"]),
        "median_effect_over_baseline_ratio": float(base["median_effect_over_baseline_ratio"]),
        "fraction_effect_gt_baseline": float(base["fraction_effect_gt_baseline"]),
        "fraction_effect_ge_baseline": float(base["fraction_effect_ge_baseline"]),
        "gt_count": int(base["gt_count"]),
        "ge_count": int(base["ge_count"]),
        "wilcoxon_greater_pvalue": base["wilcoxon_greater_pvalue"],
        "wilcoxon_two_sided_pvalue": base["wilcoxon_two_sided_pvalue"],
        "sign_test_greater_pvalue": base["sign_test_greater_pvalue"],
    }


def _summarize_prefixed_rows(rows: list[dict], *, baseline_key: str, effect_key: str, prefix: str) -> dict:
    filtered = [r for r in rows if baseline_key in r and effect_key in r]
    if not filtered:
        return {}
    baseline = np.asarray([float(r[baseline_key]) for r in filtered], dtype=np.float64)
    effect = np.asarray([float(r[effect_key]) for r in filtered], dtype=np.float64)
    base = _summarize_numeric_pairs(baseline, effect)
    return {f"{prefix}_{k}": v for k, v in base.items()}


def _scalarize_dict(payload: dict) -> dict:
    out = {}
    for key, value in payload.items():
        arr = np.asarray(jax.device_get(value))
        out[key] = arr.item() if arr.shape == () else arr.tolist()
    return out


def _calc_oe_loss_np(z_seq: np.ndarray) -> float:
    val = asal_metrics.calc_open_endedness_score(jnp.asarray(z_seq, dtype=jnp.float32))
    return float(np.asarray(jax.device_get(val)))


def main(cfg, args):
    project_root = _repo_root()
    save_dir = _resolve_path(getattr(args, "save_dir", None), project_root)
    if save_dir is None:
        raise ValueError("meta.save_dir must be set.")
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_text(save_dir / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
    resolved_checkpoint_dir = _resolve_path(
        getattr(args, "checkpoint_dir", getattr(args, "source_save_dir", None)),
        project_root,
    )
    resolved_params_path = _resolve_path(getattr(args, "params_path", None), project_root)

    run = wandb.init(
        project=str(getattr(args, "wandb_project", "asal")),
        mode=str(getattr(args, "wandb_mode", "online")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        enable_clip = bool(getattr(args, "enable_clip", True))
        enable_msc = bool(getattr(args, "enable_msc", cfg.get("metric", None) is not None))
        if not enable_clip and not enable_msc:
            raise ValueError("At least one of evaluation.enable_clip / evaluation.enable_msc must be true.")

        params = _load_params(args, project_root)
        substrate = _create_substrate(args, enable_msc=enable_msc)
        expected_len = int(np.asarray(substrate.default_params(jax.random.PRNGKey(0))).size)
        if int(params.shape[-1]) != expected_len:
            raise ValueError(
                f"Loaded parameter length {int(params.shape[-1])} does not match substrate expectation {expected_len}."
            )

        if str(args.substrate) != "lenia_flow":
            raise ValueError("This evaluator currently supports substrate='lenia_flow' only.")

        total_steps = int(getattr(args, "total_steps"))
        warmup_steps = int(getattr(args, "warmup_steps"))
        late_start, late_end = _resolve_window(args)
        late_window_steps = int(late_end - late_start)
        time_sampling = int(getattr(args, "time_sampling"))
        n_initial_states = int(getattr(args, "n_initial_states"))
        if n_initial_states < 1:
            raise ValueError("evaluation.n_initial_states must be >= 1.")

        grid_size = int(getattr(args, "grid_size"))
        split_n = int(getattr(args, "grid_split"))
        if split_n < 1:
            raise ValueError("protocol.grid_split must be >= 1.")
        if grid_size % split_n != 0:
            raise ValueError(f"grid_size {grid_size} must be divisible by grid_split {split_n}.")
        block_size = grid_size // split_n
        pad = int(getattr(args, "wall_pad", int(args.dd)))
        block_sim_size = block_size + 2 * pad
        n_blocks = split_n * split_n

        block_kwargs = util.flow_lenia_kwargs_from_args(args)
        block_kwargs["grid_size"] = block_sim_size
        block_substrate = substrates.FlattenSubstrateParameters(
            substrates.create_substrate("lenia_flow", **block_kwargs)
        )

        resume = bool(getattr(args, "resume", True))
        save_embeddings = bool(getattr(args, "save_embeddings", True))
        save_lagrangian_tracks = bool(getattr(args, "save_lagrangian_tracks", enable_msc))
        show_inner_progress = bool(getattr(args, "show_inner_progress", True))
        inner_progress_chunk_steps = int(getattr(args, "inner_progress_chunk_steps", 10_000))
        if inner_progress_chunk_steps < 1:
            raise ValueError("evaluation.inner_progress_chunk_steps must be >= 1.")

        fm_name = None
        fm = None
        clip_img_size = None
        distance_metric = None
        embed_rollout = None
        embed_chunk_stepper = None
        if enable_clip:
            fm_name = str(getattr(args, "foundation_model", "clip"))
            fm = foundation_models.create_foundation_model(fm_name)
            clip_img_size = int(getattr(args, "clip_img_size", 224))
            distance_metric = str(getattr(args, "distance_metric", "cosine_mean"))
            embed_chunk_stepper = _build_embedding_chunk_stepper(
                substrate,
                fm,
                img_size=clip_img_size,
            )

        metric_cfg = None
        metric_info = None
        metric_eval = None
        lagrangian_rollout = None
        lagrangian_chunk_stepper = None
        lag_n_particles = None
        lag_init_mode = None
        lag_channel_mode = None
        if enable_msc:
            metric_node = OmegaConf.merge(cfg.get("substrate", {}), cfg.get("metric", {}))
            metric_dict = OmegaConf.to_container(metric_node, resolve=True)
            metric_args = SimpleNamespace(**metric_dict)
            metric_args.rollout_steps = int(late_window_steps)
            if getattr(metric_args, "metric_periodic", None) is None:
                metric_args.metric_periodic = str(getattr(substrate, "border", "wall")) == "torus"
            if getattr(metric_args, "metric_domain_y", None) is None:
                metric_args.metric_domain_y = float(
                    getattr(getattr(substrate, "cfg", None), "X", getattr(substrate, "grid_size", 0))
                )
            if getattr(metric_args, "metric_domain_x", None) is None:
                metric_args.metric_domain_x = float(
                    getattr(getattr(substrate, "cfg", None), "Y", getattr(substrate, "grid_size", 0))
                )
            metric_cfg = resolve_metric_config(metric_args)
            metric_info = metric_summary(metric_cfg)
            metric_info = dict(
                metric_info,
                trajectory_start_steps=int(late_start),
                trajectory_end_steps=int(late_end),
                trajectory_window_steps=int(late_window_steps),
            )
            _write_json(save_dir / "msc_metric_summary.json", metric_info)
            metric_loss_fn = make_metric_loss_fn(metric_cfg)
            metric_eval = jax.jit(metric_loss_fn)
            lag_n_particles = int(getattr(metric_args, "metric_lagrangian_n_particles", 256))
            lag_init_mode = str(getattr(metric_args, "metric_lagrangian_init_mode", "mass"))
            lag_flow_channel = int(getattr(metric_args, "metric_lagrangian_flow_channel", -1))
            lag_flow_reduce = str(getattr(metric_args, "metric_lagrangian_flow_reduce", "mass_weighted"))
            lag_channel_mode = str(getattr(metric_args, "metric_lagrangian_channel_mode", "mix"))
            lag_noise_model = str(getattr(metric_args, "metric_lagrangian_noise_model", "none"))
            lag_diffusion_scale = float(getattr(metric_args, "metric_lagrangian_diffusion_scale", 1.0))
            lagrangian_rollout = _build_lagrangian_rollout(
                substrate,
                rollout_steps=late_window_steps,
                metric_cfg=metric_cfg,
                lag_n_particles=lag_n_particles,
                lag_init_mode=lag_init_mode,
                lag_flow_channel=lag_flow_channel,
                lag_flow_reduce=lag_flow_reduce,
                lag_channel_mode=lag_channel_mode,
                lag_noise_model=lag_noise_model,
                lag_diffusion_scale=lag_diffusion_scale,
            )
            lagrangian_chunk_stepper = _build_lagrangian_chunk_stepper(
                substrate,
                chunk_steps=int(metric_cfg["sample_every_steps"]),
                lag_flow_channel=lag_flow_channel,
                lag_flow_reduce=lag_flow_reduce,
                lag_channel_mode=lag_channel_mode,
                lag_noise_model=lag_noise_model,
                lag_diffusion_scale=lag_diffusion_scale,
            )

        block_template = block_substrate.init_state(jax.random.PRNGKey(0), params)

        control_prefix_advancer = _build_state_advancer(substrate, late_start)
        control_prefix_chunk_stepper = _build_state_chunk_stepper(substrate)
        walls_warmupper = _build_block_warmupper(block_substrate, n_blocks, warmup_steps)
        walls_post_advancer = _build_state_advancer(substrate, late_start - warmup_steps)
        if enable_clip:
            embed_rollout = _build_embedding_rollout(
                substrate,
                fm,
                window_steps=late_window_steps,
                time_sampling=time_sampling,
                img_size=clip_img_size,
            )

        trial_dir = save_dir / "trial_data"
        trial_dir.mkdir(parents=True, exist_ok=True)

        master_key = jax.random.PRNGKey(int(getattr(args, "seed", 0)))
        rows: list[dict] = []

        pbar = tqdm(range(n_initial_states), desc="history_dependence", leave=True)
        for trial_idx in pbar:
            trial_key = jax.random.fold_in(master_key, int(trial_idx))
            (
                k_init,
                k_ctrl_a_prefix,
                k_ctrl_a_window,
                k_ctrl_b_prefix,
                k_ctrl_b_window,
                k_walls_warm,
                k_walls_post,
                k_walls_window,
                k_ctrl_a_msc_roll,
                k_ctrl_a_msc_metric,
                k_ctrl_b_msc_roll,
                k_ctrl_b_msc_metric,
                k_walls_msc_roll,
                k_walls_msc_metric,
            ) = jax.random.split(trial_key, 14)

            trial_json = trial_dir / f"trial_{trial_idx:05d}.json"
            trial_npz = trial_dir / f"trial_{trial_idx:05d}_embeddings.npz"
            trial_lag_npz = trial_dir / f"trial_{trial_idx:05d}_lagrangian.npz"
            trial_complete = (
                trial_json.exists()
                and (not enable_clip or not save_embeddings or trial_npz.exists())
                and (not save_lagrangian_tracks or not enable_msc or trial_lag_npz.exists())
            )
            if resume and trial_complete:
                with trial_json.open("r") as f:
                    row = json.load(f)
                rows.append(row)
                continue

            initial_state = substrate.init_state(k_init, params)

            control_a_start = _advance_state_with_progress(
                get_stepper=control_prefix_chunk_stepper,
                rng_key=k_ctrl_a_prefix,
                state0=initial_state,
                params_in=params,
                steps=late_start,
                chunk_steps=inner_progress_chunk_steps,
                desc=f"trial {trial_idx:05d} control_a_prefix",
                show_progress=show_inner_progress,
            )
            control_b_start = _advance_state_with_progress(
                get_stepper=control_prefix_chunk_stepper,
                rng_key=k_ctrl_b_prefix,
                state0=initial_state,
                params_in=params,
                steps=late_start,
                chunk_steps=inner_progress_chunk_steps,
                desc=f"trial {trial_idx:05d} control_b_prefix",
                show_progress=show_inner_progress,
            )

            block_state0 = _prepare_block_template_state(
                initial_state=initial_state,
                block_template=block_template,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                C=int(getattr(args, "C")),
                k=int(getattr(args, "k")),
            )
            block_state_warm = walls_warmupper(k_walls_warm, block_state0, params)
            merged_state = _merge_blocks_into_global_state(
                initial_state,
                block_state_warm,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
            )
            walls_start = walls_post_advancer(k_walls_post, merged_state, params)

            row = {
                "trial_idx": int(trial_idx),
                "embeddings_path": None if not (enable_clip and save_embeddings) else str(trial_npz),
                "lagrangian_path": None if not (enable_msc and save_lagrangian_tracks) else str(trial_lag_npz),
                "late_window_start_steps": int(late_start),
                "late_window_end_steps": int(late_end),
                "late_window_steps": int(late_window_steps),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(total_steps),
                "clip_time_sampling": None if not enable_clip else int(time_sampling),
                "distance_metric": distance_metric,
                "foundation_model": fm_name,
            }
            pbar_stats = {}

            if enable_clip:
                z_control_a = _rollout_embeddings_with_progress(
                    get_stepper=embed_chunk_stepper,
                    rng_key=k_ctrl_a_window,
                    state0=control_a_start,
                    params_in=params,
                    window_steps=late_window_steps,
                    time_sampling=time_sampling,
                    desc=f"trial {trial_idx:05d} control_a_clip",
                    show_progress=show_inner_progress,
                )
                z_control_b = _rollout_embeddings_with_progress(
                    get_stepper=embed_chunk_stepper,
                    rng_key=k_ctrl_b_window,
                    state0=control_b_start,
                    params_in=params,
                    window_steps=late_window_steps,
                    time_sampling=time_sampling,
                    desc=f"trial {trial_idx:05d} control_b_clip",
                    show_progress=show_inner_progress,
                )
                z_walls = np.asarray(
                    jax.device_get(embed_rollout(k_walls_window, walls_start, params)),
                    dtype=np.float32,
                )

                baseline_distance, baseline_per_t = _sequence_distance(z_control_a, z_control_b, distance_metric)
                walls_a_distance, walls_a_per_t = _sequence_distance(z_control_a, z_walls, distance_metric)
                walls_b_distance, walls_b_per_t = _sequence_distance(z_control_b, z_walls, distance_metric)
                walls_effect_distance = float(0.5 * (walls_a_distance + walls_b_distance))
                clip_oe_loss_control_a = _calc_oe_loss_np(z_control_a)
                clip_oe_loss_control_b = _calc_oe_loss_np(z_control_b)
                clip_oe_loss_walls = _calc_oe_loss_np(z_walls)
                clip_oe_loss_control_mean = 0.5 * (clip_oe_loss_control_a + clip_oe_loss_control_b)

                row.update(
                    {
                        "baseline_distance": float(baseline_distance),
                        "walls_effect_distance": float(walls_effect_distance),
                        "walls_effect_distance_ctrl_a": float(walls_a_distance),
                        "walls_effect_distance_ctrl_b": float(walls_b_distance),
                        "effect_minus_baseline": float(walls_effect_distance - baseline_distance),
                        "effect_over_baseline_ratio": float(walls_effect_distance / max(baseline_distance, 1e-12)),
                        "clip_oe_loss_control_a": float(clip_oe_loss_control_a),
                        "clip_oe_loss_control_b": float(clip_oe_loss_control_b),
                        "clip_oe_loss_control_mean": float(clip_oe_loss_control_mean),
                        "clip_oe_loss_walls": float(clip_oe_loss_walls),
                        "clip_oe_loss_walls_minus_control_mean": float(clip_oe_loss_walls - clip_oe_loss_control_mean),
                    }
                )
                if save_embeddings:
                    _save_npz_atomic(
                        trial_npz,
                        z_control_a=z_control_a,
                        z_control_b=z_control_b,
                        z_walls=z_walls,
                        baseline_per_t=baseline_per_t,
                        walls_ctrl_a_per_t=walls_a_per_t,
                        walls_ctrl_b_per_t=walls_b_per_t,
                    )
                pbar_stats.update(
                    {
                        "clip_base": f"{baseline_distance:.4f}",
                        "clip_delta": f"{(walls_effect_distance - baseline_distance):.4f}",
                    }
                )

            if enable_msc:
                xy_control_a = _rollout_lagrangian_with_progress(
                    chunk_stepper=lagrangian_chunk_stepper,
                    substrate=substrate,
                    rng_key=k_ctrl_a_msc_roll,
                    state0=control_a_start,
                    params_in=params,
                    time_sampling=int(metric_cfg["time_sampling"]),
                    chunk_steps=int(metric_cfg["sample_every_steps"]),
                    lag_n_particles=lag_n_particles,
                    lag_init_mode=lag_init_mode,
                    lag_channel_mode=lag_channel_mode,
                    desc=f"trial {trial_idx:05d} control_a_msc",
                    show_progress=show_inner_progress,
                )
                xy_control_b = _rollout_lagrangian_with_progress(
                    chunk_stepper=lagrangian_chunk_stepper,
                    substrate=substrate,
                    rng_key=k_ctrl_b_msc_roll,
                    state0=control_b_start,
                    params_in=params,
                    time_sampling=int(metric_cfg["time_sampling"]),
                    chunk_steps=int(metric_cfg["sample_every_steps"]),
                    lag_n_particles=lag_n_particles,
                    lag_init_mode=lag_init_mode,
                    lag_channel_mode=lag_channel_mode,
                    desc=f"trial {trial_idx:05d} control_b_msc",
                    show_progress=show_inner_progress,
                )
                xy_walls = np.asarray(
                    jax.device_get(lagrangian_rollout(k_walls_msc_roll, walls_start, params)),
                    dtype=np.float32,
                )

                msc_loss_control_a, msc_metrics_control_a = metric_eval(
                    k_ctrl_a_msc_metric,
                    jnp.asarray(xy_control_a),
                )
                msc_loss_control_b, msc_metrics_control_b = metric_eval(
                    k_ctrl_b_msc_metric,
                    jnp.asarray(xy_control_b),
                )
                msc_loss_walls, msc_metrics_walls = metric_eval(
                    k_walls_msc_metric,
                    jnp.asarray(xy_walls),
                )
                msc_loss_control_a = float(np.asarray(jax.device_get(msc_loss_control_a)))
                msc_loss_control_b = float(np.asarray(jax.device_get(msc_loss_control_b)))
                msc_loss_walls = float(np.asarray(jax.device_get(msc_loss_walls)))
                msc_metrics_control_a = _scalarize_dict(msc_metrics_control_a)
                msc_metrics_control_b = _scalarize_dict(msc_metrics_control_b)
                msc_metrics_walls = _scalarize_dict(msc_metrics_walls)
                msc_loss_control_mean = 0.5 * (msc_loss_control_a + msc_loss_control_b)
                msc_score_control_mean = 0.5 * (
                    float(msc_metrics_control_a["score"]) + float(msc_metrics_control_b["score"])
                )

                row.update(
                    {
                        "msc_loss_control_a": float(msc_loss_control_a),
                        "msc_loss_control_b": float(msc_loss_control_b),
                        "msc_loss_control_mean": float(msc_loss_control_mean),
                        "msc_loss_walls": float(msc_loss_walls),
                        "msc_loss_walls_minus_control_mean": float(msc_loss_walls - msc_loss_control_mean),
                        "msc_score_control_a": float(msc_metrics_control_a["score"]),
                        "msc_score_control_b": float(msc_metrics_control_b["score"]),
                        "msc_score_control_mean": float(msc_score_control_mean),
                        "msc_score_walls": float(msc_metrics_walls["score"]),
                        "msc_score_walls_minus_control_mean": float(
                            float(msc_metrics_walls["score"]) - msc_score_control_mean
                        ),
                        "msc_amp_control_a": float(msc_metrics_control_a["amp"]),
                        "msc_amp_control_b": float(msc_metrics_control_b["amp"]),
                        "msc_amp_walls": float(msc_metrics_walls["amp"]),
                        "msc_component_control_a": float(msc_metrics_control_a["msc"]),
                        "msc_component_control_b": float(msc_metrics_control_b["msc"]),
                        "msc_component_walls": float(msc_metrics_walls["msc"]),
                        "msc_tau_best_steps_control_a": int(msc_metrics_control_a["tau_best_steps"]),
                        "msc_tau_best_steps_control_b": int(msc_metrics_control_b["tau_best_steps"]),
                        "msc_tau_best_steps_walls": int(msc_metrics_walls["tau_best_steps"]),
                        "msc_sample_every_steps": int(metric_cfg["sample_every_steps"]),
                        "msc_time_sampling": int(metric_cfg["time_sampling"]),
                    }
                )
                if save_lagrangian_tracks:
                    _save_npz_atomic(
                        trial_lag_npz,
                        xy_control_a=xy_control_a,
                        xy_control_b=xy_control_b,
                        xy_walls=xy_walls,
                        sample_offsets_steps=(
                            np.arange(1, int(metric_cfg["time_sampling"]) + 1, dtype=np.int32)
                            * int(metric_cfg["sample_every_steps"])
                        ),
                        sample_every_steps=np.asarray(int(metric_cfg["sample_every_steps"]), dtype=np.int32),
                        trajectory_start_steps=np.asarray(int(late_start), dtype=np.int32),
                        trajectory_end_steps=np.asarray(int(late_end), dtype=np.int32),
                        trajectory_window_steps=np.asarray(int(late_window_steps), dtype=np.int32),
                        metric_window_size_steps=np.asarray(int(metric_cfg["window_size_frames"] * metric_cfg["sample_every_steps"]), dtype=np.int32),
                        metric_window_step_steps=np.asarray(int(metric_cfg["window_step_frames"] * metric_cfg["sample_every_steps"]), dtype=np.int32),
                        metric_tau_steps=np.asarray(int(metric_cfg["tau_steps"]), dtype=np.int32),
                    )
                pbar_stats.update(
                    {
                        "msc_ctrl": f"{msc_loss_control_mean:.4f}",
                        "msc_walls": f"{msc_loss_walls:.4f}",
                    }
                )

            _write_json(trial_json, row)
            rows.append(row)
            if pbar_stats:
                pbar.set_postfix(**pbar_stats)

        rows = sorted(rows, key=lambda r: int(r["trial_idx"]))
        summary = {}
        if enable_clip:
            summary.update(_summarize_trials(rows))
            summary.update(
                _summarize_prefixed_rows(
                    rows,
                    baseline_key="clip_oe_loss_control_mean",
                    effect_key="clip_oe_loss_walls",
                    prefix="clip_oe_loss",
                )
            )
        if enable_msc:
            summary.update(
                _summarize_prefixed_rows(
                    rows,
                    baseline_key="msc_loss_control_mean",
                    effect_key="msc_loss_walls",
                    prefix="msc_loss",
                )
            )
            summary.update(
                _summarize_prefixed_rows(
                    rows,
                    baseline_key="msc_score_control_mean",
                    effect_key="msc_score_walls",
                    prefix="msc_score",
                )
            )
        summary.update(
            {
                "n_trials": int(len(rows)),
                "save_dir": str(save_dir),
                "checkpoint_dir": None if resolved_checkpoint_dir is None else str(resolved_checkpoint_dir),
                "params_path": None if resolved_params_path is None else str(resolved_params_path),
                "params_name": str(getattr(args, "params_name", "best")),
                "enable_clip": bool(enable_clip),
                "enable_msc": bool(enable_msc),
                "foundation_model": fm_name,
                "distance_metric": distance_metric,
                "grid_split": int(split_n),
                "wall_pad": int(pad),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(total_steps),
                "late_window_start_steps": int(late_start),
                "late_window_end_steps": int(late_end),
                "late_window_steps": int(late_window_steps),
                "clip_time_sampling": None if not enable_clip else int(time_sampling),
                "resume_enabled": bool(resume),
                "save_embeddings": bool(save_embeddings),
                "save_lagrangian_tracks": bool(save_lagrangian_tracks),
            }
        )
        if metric_info is not None:
            summary["msc_metric_summary"] = metric_info

        _save_csv(save_dir / "trial_results.csv", rows)
        _write_json(save_dir / "summary.json", summary)

        if rows:
            table = wandb.Table(
                columns=list(rows[0].keys()),
                data=[[row[k] for k in rows[0].keys()] for row in rows],
            )
            run.log({"history_dependence/trials": table})
        for key, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                run.summary[f"history_dependence/{key}"] = value
        if metric_info is not None:
            run.summary["history_dependence/msc_metric_summary"] = str(metric_info)

        print(f"Completed trials: {summary.get('n_trials', 0)}")
        if enable_clip:
            print(f"Mean baseline distance: {summary.get('mean_baseline_distance', float('nan')):.6f}")
            print(f"Mean walls-effect distance: {summary.get('mean_walls_effect_distance', float('nan')):.6f}")
            print(f"Mean effect-baseline delta: {summary.get('mean_effect_minus_baseline', float('nan')):.6f}")
            print(f"Fraction effect > baseline: {summary.get('fraction_effect_gt_baseline', float('nan')):.6f}")
            pval = summary.get("wilcoxon_greater_pvalue", None)
            if pval is not None:
                print(f"Wilcoxon p(effect > baseline): {pval:.6g}")
            print(f"Mean CLIP OE walls-control delta: {summary.get('clip_oe_loss_mean_effect_minus_baseline', float('nan')):.6f}")
        if enable_msc:
            print(f"Mean MSC loss walls-control delta: {summary.get('msc_loss_mean_effect_minus_baseline', float('nan')):.6f}")
            print(f"Mean MSC score walls-control delta: {summary.get('msc_score_mean_effect_minus_baseline', float('nan')):.6f}")
    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
