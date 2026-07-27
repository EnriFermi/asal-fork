from __future__ import annotations

import json
import hashlib
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf

import foundation_models
import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config
from evaluate_frustration_history_dependence import (
    _build_block_warmupper,
    _build_lagrangian_chunk_stepper,
    _build_state_chunk_stepper,
    _calc_oe_loss_np,
    _create_substrate,
    _init_lagrangian_carry,
    _load_params,
    _merge_blocks_into_global_state,
    _patch_wandb_pandas_check,
    _prepare_block_template_state,
    _resolve_window,
    _save_npz_atomic,
    _scalarize_dict,
    _sequence_distance,
    _write_json,
)
from paper_check_frustration_eval import (
    _build_image_embedder,
    _extract_render_state,
    _init_run_checkpoint,
    _lag_init_keys,
    _load_run_checkpoint,
    _make_trial_paths,
    _save_csv,
    _save_run_checkpoint,
    _split_global_render_blocks,
    _trim_block_render_state,
    _validate_divisibility,
    _write_text,
)


_patch_wandb_pandas_check()

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for non-interactive environments.
    tqdm = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _flat_cfg(cfg):
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("source", {}),
        cfg.get("substrate", {}),
        cfg.get("protocol", {}),
        cfg.get("evaluation", {}),
        cfg.get("metric", {}),
        cfg.get("logging", {}),
        cfg.get("job", {}),
    )


def _stack_trees(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


def _unstack_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return []
    batch_size = int(leaves[0].shape[0])
    return [jax.tree_util.tree_map(lambda x, i=i: x[i], tree) for i in range(batch_size)]


class _NoProgress:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def _make_progress_bar(*, total: int, desc: str):
    if tqdm is None or total <= 0 or os.environ.get("PAPER_CHECK_DISABLE_TQDM") == "1":
        return _NoProgress()
    return tqdm(
        total=int(total),
        desc=desc,
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
        ascii=True,
    )


def _remaining_lane_steps(lanes: list[dict], *, total_steps: int) -> int:
    out = 0
    for lane in lanes:
        state = lane.get("state")
        if state is None:
            continue
        out += max(0, int(total_steps) - int(state.get("current_step", 0)))
    return int(out)


def _progress_delta(*, current_step: int, next_step: int, total_steps: int, n_lanes: int) -> int:
    advanced = max(0, min(int(next_step), int(total_steps)) - int(current_step))
    return int(advanced * int(n_lanes))


def _take_tree(tree, indices: list[int]):
    return jax.tree_util.tree_map(lambda x: x[np.asarray(indices, dtype=np.int32)], tree)


def _split_rng_batch(rng_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    split = jax.vmap(lambda key: jax.random.split(key, 2))(rng_batch)
    return split[:, 0], split[:, 1]


def _make_step_keys_batch(chunk_keys: jax.Array, steps: int) -> jax.Array:
    return jax.vmap(lambda key: jax.random.split(key, steps))(chunk_keys)


def _optimizer_metric_roll_key(run_seed: int, *, log_clip_evolution: bool) -> jax.Array:
    eval_key = jax.random.PRNGKey(int(run_seed))
    return jax.random.split(eval_key, 3 if log_clip_evolution else 2)[0]


def _optimizer_metric_parts(run_seed: int, *, log_clip_evolution: bool) -> tuple[jax.Array, ...]:
    return tuple(jax.random.split(_optimizer_metric_roll_key(run_seed, log_clip_evolution=log_clip_evolution), 4))


def _optimizer_metric_key_schedule(
    *,
    run_seed: int,
    total_steps: int,
    training_horizon_steps: int,
    chunk_steps: int,
    log_clip_evolution: bool,
) -> jax.Array:
    if training_horizon_steps <= 0 or training_horizon_steps > total_steps:
        raise ValueError(
            "optimization_metric requires 0 < training_horizon_steps <= total_steps; "
            f"got training_horizon_steps={training_horizon_steps}, total_steps={total_steps}."
        )
    if training_horizon_steps % chunk_steps != 0 or total_steps % chunk_steps != 0:
        raise ValueError(
            "optimization_metric key schedule requires training and total horizons divisible by "
            f"chunk_steps={chunk_steps}; got {training_horizon_steps} and {total_steps}."
        )
    scan_key = _optimizer_metric_parts(
        run_seed,
        log_clip_evolution=log_clip_evolution,
    )[3]
    n_train = int(training_horizon_steps // chunk_steps)
    n_total = int(total_steps // chunk_steps)
    training_keys = jax.random.split(scan_key, n_train)
    if n_total == n_train:
        return training_keys
    extension_key = jax.random.fold_in(scan_key, jnp.uint32(training_horizon_steps))
    extension_keys = jax.random.split(extension_key, n_total - n_train)
    return jnp.concatenate((training_keys, extension_keys), axis=0)


def _next_lane_chunk_keys(
    lanes: list[dict],
    group_indices: list[int],
    *,
    current_step: int,
    chunk_steps: int,
) -> tuple[jax.Array, jax.Array]:
    protocols = {str(lanes[idx].get("run_seed_protocol", "legacy")) for idx in group_indices}
    if len(protocols) != 1:
        raise ValueError(f"Mixed run_seed_protocol values in one lane group: {protocols}.")
    protocol = next(iter(protocols))
    if protocol == "optimization_metric":
        chunk_idx = int(current_step // chunk_steps)
        chunk_keys = jnp.stack([lanes[idx]["rng_schedule"][chunk_idx] for idx in group_indices], axis=0)
        return chunk_keys, chunk_keys
    rng_batch = jnp.stack([lanes[idx]["state"]["rng"] for idx in group_indices], axis=0)
    return _split_rng_batch(rng_batch)


def _build_batched_state_chunk_stepper(substrate, chunk_steps: int):
    single = _build_state_chunk_stepper(substrate)(chunk_steps)

    @jax.jit
    def step(step_keys_batch, state_batch, params_batch):
        return jax.vmap(single, in_axes=(0, 0, 0))(step_keys_batch, state_batch, params_batch)

    return step


def _mask_block_spatial_state(state, valid_mask: jax.Array):
    if not isinstance(state, dict):
        raise TypeError("Flow-Lenia block state must be a dict.")

    def mask_leaf(key: str, value):
        if key not in {"A", "P", "F", "Food"}:
            return value
        if value.ndim >= 3 and tuple(value.shape[:3]) == tuple(valid_mask.shape):
            mask = valid_mask.reshape(valid_mask.shape + (1,) * (value.ndim - 3))
            return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))
        return value

    return {key: mask_leaf(str(key), value) for key, value in state.items()}


def _build_batched_block_chunk_stepper(
    block_substrate,
    *,
    n_blocks: int,
    chunk_steps: int,
    valid_mask: jax.Array | None = None,
):
    if valid_mask is None:
        single = _build_block_warmupper(block_substrate, n_blocks, chunk_steps)
    else:
        valid_mask = jnp.asarray(valid_mask, dtype=bool)

        def single(rng_key, state0, params_in):
            def block_step(state, keys):
                next_state = jax.vmap(
                    lambda st, key: block_substrate.step_state(key, st, params_in)
                )(state, keys)
                return _mask_block_spatial_state(next_state, valid_mask), None

            step_keys = jax.random.split(rng_key, chunk_steps)
            block_keys = jax.vmap(lambda key: jax.random.split(key, n_blocks))(step_keys)
            state_final, _ = jax.lax.scan(block_step, state0, block_keys)
            return state_final

    @jax.jit
    def step(chunk_keys_batch, block_state_batch, params_batch):
        return jax.vmap(single, in_axes=(0, 0, 0))(chunk_keys_batch, block_state_batch, params_batch)

    return step


def _build_batched_lagrangian_chunk_stepper(
    substrate,
    *,
    chunk_steps: int,
    lag_flow_channel: int,
    lag_flow_reduce: str,
    lag_channel_mode: str,
    lag_noise_model: str,
    lag_diffusion_scale: float,
):
    single = _build_lagrangian_chunk_stepper(
        substrate,
        chunk_steps=chunk_steps,
        lag_flow_channel=lag_flow_channel,
        lag_flow_reduce=lag_flow_reduce,
        lag_channel_mode=lag_channel_mode,
        lag_noise_model=lag_noise_model,
        lag_diffusion_scale=lag_diffusion_scale,
    )

    @jax.jit
    def step(chunk_keys_batch, lag_carry_batch, params_batch):
        return jax.vmap(single, in_axes=(0, 0, 0))(chunk_keys_batch, lag_carry_batch, params_batch)

    return step


def _extract_positions_from_state(state):
    if isinstance(state, dict):
        if "x" not in state:
            raise ValueError("Explicit-position paper check expects state dict key 'x' or an array state.")
        return state["x"]
    arr = jnp.asarray(state)
    if arr.ndim < 2 or int(arr.shape[-1]) != 2:
        raise ValueError("Explicit-position paper check expects state array shape (..., 2).")
    return arr


def _replace_positions_in_state(state, xy):
    if isinstance(state, dict):
        out = dict(state)
        out["x"] = xy
        return out
    return xy


def _build_generic_batch_embedder(*, substrate, fm, clip_img_size: int):
    embed_batch = _build_image_embedder(fm)

    @jax.jit
    def embed_global_batch(state_batch, params_batch):
        imgs = jax.vmap(
            lambda st, pr: substrate.render_state(st, pr, img_size=clip_img_size),
            in_axes=(0, 0),
        )(state_batch, params_batch)
        return embed_batch(imgs)

    return embed_global_batch


def _metric_space_from_args(args, substrate) -> dict[str, float | bool]:
    defaults = util.metric_periodic_space_defaults(substrate)
    periodic = getattr(args, "metric_periodic", None)
    domain_y = getattr(args, "metric_domain_y", None)
    domain_x = getattr(args, "metric_domain_x", None)
    return dict(
        periodic=bool(defaults["periodic"] if periodic is None else periodic),
        domain_y=float(defaults["domain_y"] if domain_y is None else domain_y),
        domain_x=float(defaults["domain_x"] if domain_x is None else domain_x),
    )


def _position_domain_for_perturbation(args, substrate, xy):
    space = _metric_space_from_args(args, substrate)
    y_min_cfg = getattr(args, "perturbation_domain_min_y", None)
    x_min_cfg = getattr(args, "perturbation_domain_min_x", None)
    y_size_cfg = getattr(args, "perturbation_domain_y", None)
    x_size_cfg = getattr(args, "perturbation_domain_x", None)

    if y_min_cfg is not None and x_min_cfg is not None and y_size_cfg is not None and x_size_cfg is not None:
        lo = jnp.asarray([float(y_min_cfg), float(x_min_cfg)], dtype=xy.dtype)
        span = jnp.asarray([float(y_size_cfg), float(x_size_cfg)], dtype=xy.dtype)
        return lo, jnp.maximum(span, jnp.asarray(1e-6, dtype=xy.dtype)), bool(space["periodic"])

    if float(space["domain_y"]) > 0.0 and float(space["domain_x"]) > 0.0:
        lo = jnp.zeros((2,), dtype=xy.dtype)
        span = jnp.asarray([float(space["domain_y"]), float(space["domain_x"])], dtype=xy.dtype)
        return lo, span, bool(space["periodic"])

    padding = float(getattr(args, "perturbation_domain_padding", 0.05))
    xy_min = jnp.min(xy, axis=0)
    xy_max = jnp.max(xy, axis=0)
    raw_span = jnp.maximum(xy_max - xy_min, jnp.asarray(1e-6, dtype=xy.dtype))
    pad = raw_span * padding
    return xy_min - pad, raw_span + 2.0 * pad, False


def _normalize_state_perturbation(kind: str) -> str:
    normalized = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "cell_shuffle": "cell_shuffle",
        "grid_shuffle": "cell_shuffle",
        "spatial_shuffle": "cell_shuffle",
        "position_permute": "position_permute",
        "permute_positions": "position_permute",
        "permute": "position_permute",
    }
    if normalized not in aliases:
        raise ValueError(
            "protocol.perturbation_kind must be one of "
            "['cell_shuffle', 'position_permute', 'none'], "
            f"got {kind!r}."
        )
    return aliases[normalized]


def _apply_position_perturbation(state, key, *, args, substrate):
    kind = _normalize_state_perturbation(getattr(args, "perturbation_kind", "cell_shuffle"))
    if kind == "none":
        return state

    xy = _extract_positions_from_state(state)
    n = int(xy.shape[0])
    if n < 1:
        return state

    key_perm, key_jitter = jax.random.split(key)
    strength = float(getattr(args, "perturbation_strength", 1.0))
    strength = max(0.0, min(1.0, strength))

    if kind == "position_permute":
        perm = jax.random.permutation(key_perm, n)
        target = xy[perm]
    else:
        grid_split = int(getattr(args, "perturbation_grid_split", getattr(args, "grid_split", 2)))
        if grid_split < 1:
            raise ValueError(f"perturbation_grid_split must be >= 1, got {grid_split}.")
        n_cells = int(grid_split * grid_split)
        lo, span, periodic = _position_domain_for_perturbation(args, substrate, xy)
        one = jnp.asarray(1.0, dtype=xy.dtype)
        norm = (xy - lo) / span
        norm = jnp.clip(norm, 0.0, jnp.nextafter(one, jnp.asarray(0.0, dtype=xy.dtype)))
        scaled = norm * float(grid_split)
        cell = jnp.floor(scaled).astype(jnp.int32)
        frac = scaled - cell.astype(xy.dtype)
        flat = cell[:, 0] * grid_split + cell[:, 1]
        cell_perm = jax.random.permutation(key_perm, n_cells)
        target_flat = cell_perm[flat]
        target_cell = jnp.stack((target_flat // grid_split, target_flat % grid_split), axis=-1)
        target_norm = (target_cell.astype(xy.dtype) + frac) / float(grid_split)
        target = lo + target_norm * span
        jitter_scale = float(getattr(args, "perturbation_jitter", 0.0))
        if jitter_scale > 0.0:
            jitter = jax.random.normal(key_jitter, xy.shape, dtype=xy.dtype)
            target = target + jitter * (span / float(grid_split)) * jitter_scale
        if periodic:
            target = lo + jnp.mod(target - lo, span)
        else:
            target = jnp.clip(target, lo, lo + span)

    xy_new = xy + strength * (target - xy)
    out = _replace_positions_in_state(state, xy_new)

    if isinstance(out, dict) and "v" in out:
        velocity_mode = str(getattr(args, "perturbation_velocity_mode", "keep")).strip().lower()
        if velocity_mode in {"random", "randomize", "randomized"}:
            v = jax.random.normal(key_jitter, out["v"].shape, dtype=out["v"].dtype)
            v = v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            out["v"] = v
        elif velocity_mode in {"keep", "none"}:
            pass
        else:
            raise ValueError(
                "protocol.perturbation_velocity_mode must be one of ['keep', 'randomize'], "
                f"got {velocity_mode!r}."
            )

    return out


def _unwrap_sampled_xy_np(xy_seq: np.ndarray, *, domain_y: float, domain_x: float) -> np.ndarray:
    xy = np.asarray(xy_seq, dtype=np.float32)
    if xy.shape[0] <= 1:
        return xy
    dxy = xy[1:] - xy[:-1]
    if domain_y > 0:
        dxy[..., 0] = (dxy[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
    if domain_x > 0:
        dxy[..., 1] = (dxy[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
    increments = np.cumsum(dxy, axis=0)
    return np.concatenate((xy[:1], xy[:1] + increments), axis=0)


def _build_batch_embedders(
    *,
    substrate,
    fm,
    clip_img_size: int,
    split_n: int,
    block_size: int,
    pad: int,
    global_size: int,
    global_crop_start: int,
):
    embed_batch = _build_image_embedder(fm)

    @jax.jit
    def embed_global_batch(state_batch, params_batch):
        imgs = jax.vmap(
            lambda st, pr: substrate.render_state(_extract_render_state(st), pr, img_size=clip_img_size),
            in_axes=(0, 0),
        )(state_batch, params_batch)
        return embed_batch(imgs)

    @jax.jit
    def embed_blocks_from_block_state_batch(block_state_batch, params_batch):
        def render_trial(block_state, pr):
            blocks = _trim_block_render_state(block_state, pad=pad, block_size=block_size)
            return jax.vmap(lambda st: substrate.render_state(st, pr, img_size=clip_img_size))(blocks)

        imgs = jax.vmap(render_trial, in_axes=(0, 0))(block_state_batch, params_batch)
        batch_size = int(imgs.shape[0])
        n_blocks = int(imgs.shape[1])
        z = embed_batch(imgs.reshape((batch_size * n_blocks,) + tuple(imgs.shape[2:])))
        return z.reshape((batch_size, n_blocks, -1))

    @jax.jit
    def embed_blocks_from_global_state_batch(state_batch, params_batch):
        def render_trial(state, pr):
            padded_size = int(split_n * block_size)
            pad_before = int(global_crop_start)
            pad_after_y = int(padded_size - state["A"].shape[0] - pad_before)
            pad_after_x = int(padded_size - state["A"].shape[1] - pad_before)
            padded = dict(state)
            padded["A"] = jnp.pad(
                state["A"],
                ((pad_before, pad_after_y), (pad_before, pad_after_x), (0, 0)),
            )
            padded["P"] = jnp.pad(
                state["P"],
                ((pad_before, pad_after_y), (pad_before, pad_after_x), (0, 0)),
            )
            if "Food" in state:
                padded["Food"] = jnp.pad(
                    state["Food"],
                    ((pad_before, pad_after_y), (pad_before, pad_after_x)),
                )
            blocks = _split_global_render_blocks(padded, split_n=split_n, block_size=block_size)
            return jax.vmap(lambda st: substrate.render_state(st, pr, img_size=clip_img_size))(blocks)

        imgs = jax.vmap(render_trial, in_axes=(0, 0))(state_batch, params_batch)
        batch_size = int(imgs.shape[0])
        n_blocks = int(imgs.shape[1])
        z = embed_batch(imgs.reshape((batch_size * n_blocks,) + tuple(imgs.shape[2:])))
        return z.reshape((batch_size, n_blocks, -1))

    @jax.jit
    def embed_concat_from_block_state_batch(block_state_batch, params_batch):
        def render_trial(block_state, pr):
            A_blocks = block_state["A"]
            P_blocks = block_state["P"]
            Food_blocks = block_state.get("Food", None)
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
            A_full = A_grid.reshape((H, H, C))[
                global_crop_start:global_crop_start + global_size,
                global_crop_start:global_crop_start + global_size,
            ]

            P_grid = P_blocks.reshape((split_n, split_n, block_size, block_size, K))
            P_grid = jnp.transpose(P_grid, (0, 2, 1, 3, 4))
            P_full = P_grid.reshape((H, H, K))[
                global_crop_start:global_crop_start + global_size,
                global_crop_start:global_crop_start + global_size,
            ]

            if Food_blocks is None:
                Food_full = jnp.zeros((global_size, global_size), dtype=A_full.dtype)
            else:
                F_grid = Food_blocks.reshape((split_n, split_n, block_size, block_size))
                F_grid = jnp.transpose(F_grid, (0, 2, 1, 3))
                Food_full = F_grid.reshape((H, H))[
                    global_crop_start:global_crop_start + global_size,
                    global_crop_start:global_crop_start + global_size,
                ]
            return substrate.render_state({"A": A_full, "P": P_full, "Food": Food_full}, pr, img_size=clip_img_size)

        imgs = jax.vmap(render_trial, in_axes=(0, 0))(block_state_batch, params_batch)
        return embed_batch(imgs)

    return (
        embed_global_batch,
        embed_blocks_from_block_state_batch,
        embed_blocks_from_global_state_batch,
        embed_concat_from_block_state_batch,
    )


def _build_wall_video_renderers(
    *,
    substrate,
    img_size: int,
    split_n: int,
    block_size: int,
    pad: int,
    global_size: int,
    global_crop_start: int,
):
    @jax.jit
    def render_global_batch(state_batch, params_batch):
        return jax.vmap(
            lambda st, pr: substrate.render_state(_extract_render_state(st), pr, img_size=img_size),
            in_axes=(0, 0),
        )(state_batch, params_batch)

    @jax.jit
    def render_blocks_batch(block_state_batch, params_batch):
        def render_trial(block_state, pr):
            A = block_state["A"][:, pad:pad + block_size, pad:pad + block_size]
            P = block_state["P"][:, pad:pad + block_size, pad:pad + block_size]
            C = int(A.shape[-1])
            K = int(P.shape[-1])
            H = int(split_n * block_size)
            A = jnp.transpose(
                A.reshape((split_n, split_n, block_size, block_size, C)),
                (0, 2, 1, 3, 4),
            ).reshape((H, H, C))[
                global_crop_start:global_crop_start + global_size,
                global_crop_start:global_crop_start + global_size,
            ]
            P = jnp.transpose(
                P.reshape((split_n, split_n, block_size, block_size, K)),
                (0, 2, 1, 3, 4),
            ).reshape((H, H, K))[
                global_crop_start:global_crop_start + global_size,
                global_crop_start:global_crop_start + global_size,
            ]
            return substrate.render_state({"A": A, "P": P}, pr, img_size=img_size)

        return jax.vmap(render_trial, in_axes=(0, 0))(block_state_batch, params_batch)

    return render_global_batch, render_blocks_batch


def _block_valid_mask(
    *,
    grid_size: int,
    split_n: int,
    block_size: int,
    pad: int,
    global_crop_start: int,
) -> np.ndarray:
    block_sim_size = int(block_size + 2 * pad)
    mask = np.ones((split_n * split_n, block_sim_size, block_sim_size), dtype=bool)
    for block_idx in range(split_n * split_n):
        row = block_idx // split_n
        col = block_idx % split_n
        block_y0 = row * block_size
        block_x0 = col * block_size
        valid_global_y0 = int(global_crop_start)
        valid_global_x0 = int(global_crop_start)
        valid_global_y1 = int(global_crop_start + grid_size)
        valid_global_x1 = int(global_crop_start + grid_size)
        overlap_y0 = max(block_y0, valid_global_y0)
        overlap_x0 = max(block_x0, valid_global_x0)
        overlap_y1 = min(block_y0 + block_size, valid_global_y1)
        overlap_x1 = min(block_x0 + block_size, valid_global_x1)
        mask[
            block_idx,
            pad:pad + block_size,
            pad:pad + block_size,
        ] = False
        if overlap_y1 > overlap_y0 and overlap_x1 > overlap_x0:
            local_y0 = int(overlap_y0 - block_y0)
            local_x0 = int(overlap_x0 - block_x0)
            local_y1 = int(overlap_y1 - block_y0)
            local_x1 = int(overlap_x1 - block_x0)
            mask[
                block_idx,
                pad + local_y0:pad + local_y1,
                pad + local_x0:pad + local_x1,
            ] = True
    return mask


def _save_wall_video_frame(
    lane: dict,
    frame: np.ndarray,
    *,
    step: int,
    warmup_steps: int,
    split_n: int,
) -> None:
    import imageio.v3 as iio

    frame_u8 = np.clip(np.asarray(frame, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    if int(step) <= int(warmup_steps):
        height, width = frame_u8.shape[:2]
        thickness = max(1, int(round(min(height, width) / 128)))
        for split_idx in range(1, int(split_n)):
            y = int(round(split_idx * height / split_n))
            x = int(round(split_idx * width / split_n))
            frame_u8[max(0, y - thickness):min(height, y + thickness + 1), :, :] = 0
            frame_u8[:, max(0, x - thickness):min(width, x + thickness + 1), :] = 0
    frame_dir = lane["video_frame_dir"]
    frame_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(frame_dir / f"frame_{int(step):07d}.jpg", frame_u8, quality=90)


def _encode_wall_video(lane: dict, *, fps: float, codec: str, keep_frames: bool) -> dict[str, object]:
    if not lane.get("wall_video_enabled", False):
        return {}
    import imageio
    import imageio.v3 as iio

    frame_dir = lane["video_frame_dir"]
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No wall-video frames found for {lane['trial']['trial_idx']}: {frame_dir}.")
    output = lane["video_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=float(fps), codec=str(codec), macro_block_size=1)
    try:
        for frame_path in frame_paths:
            writer.append_data(iio.imread(frame_path))
    finally:
        writer.close()
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return {
        "walls_video_path": str(output),
        "walls_video_frames": int(len(frame_paths)),
        "walls_video_fps": float(fps),
    }


def _load_trials(job_config_paths: list[str | Path]):
    repo = _repo_root()
    trials = []
    for path_like in job_config_paths:
        cfg_path = Path(path_like)
        cfg = OmegaConf.load(cfg_path)
        flat = _flat_cfg(cfg)
        args = SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))
        root_save_dir = Path(repo / str(getattr(args, "save_dir")))
        trial_idx = int(getattr(args, "trial_idx"))
        trial_paths = _make_trial_paths(root_save_dir, trial_idx)
        if trial_paths["trial_row_json"].exists():
            print(f"[paper_check/frustration/batch] skipping completed trial_idx={trial_idx}")
            continue
        trials.append(
            {
                "cfg": cfg,
                "cfg_path": cfg_path,
                "args": args,
                "trial_idx": trial_idx,
                "trial_paths": trial_paths,
                "root_save_dir": root_save_dir,
            }
        )
    return trials


def _assert_common_trial_config(trials: list[dict]) -> None:
    if not trials:
        return
    ref = trials[0]
    ref_sections = {
        "meta_save_dir": str(getattr(ref["args"], "save_dir")),
        "substrate": OmegaConf.to_container(ref["cfg"].get("substrate", {}), resolve=True),
        "protocol": OmegaConf.to_container(ref["cfg"].get("protocol", {}), resolve=True),
        "evaluation": OmegaConf.to_container(ref["cfg"].get("evaluation", {}), resolve=True),
        "metric": OmegaConf.to_container(ref["cfg"].get("metric", {}), resolve=True),
        "logging": OmegaConf.to_container(ref["cfg"].get("logging", {}), resolve=True),
    }
    for trial in trials[1:]:
        cur_sections = {
            "meta_save_dir": str(getattr(trial["args"], "save_dir")),
            "substrate": OmegaConf.to_container(trial["cfg"].get("substrate", {}), resolve=True),
            "protocol": OmegaConf.to_container(trial["cfg"].get("protocol", {}), resolve=True),
            "evaluation": OmegaConf.to_container(trial["cfg"].get("evaluation", {}), resolve=True),
            "metric": OmegaConf.to_container(trial["cfg"].get("metric", {}), resolve=True),
            "logging": OmegaConf.to_container(trial["cfg"].get("logging", {}), resolve=True),
        }
        for key, ref_val in ref_sections.items():
            if cur_sections[key] != ref_val:
                raise ValueError(
                    f"Batch job configs must share identical {key}. "
                    f"Mismatch between {ref['cfg_path']} and {trial['cfg_path']}."
                )


def _save_lane_checkpoint(lane: dict) -> None:
    state = lane["state"]
    mode = state["mode"]
    kwargs = {}
    if mode == "block":
        kwargs["block_state"] = state["block_state"]
    elif mode == "global":
        kwargs["global_state"] = state["global_state"]
    elif mode == "lag":
        kwargs["lag_carry"] = state["lag_carry"]
    else:
        raise ValueError(f"Unknown mode={mode!r}.")
    _save_run_checkpoint(
        lane["checkpoint_path"],
        mode=mode,
        current_step=state["current_step"],
        rng=state["rng"],
        full_steps=state["full_steps"],
        late_steps=state["late_steps"],
        z_late_steps=state.get("z_late_steps", state["late_steps"]),
        xy_late_steps=state.get("xy_late_steps", state["late_steps"]),
        z_full=state["z_full"],
        z_full_blocks=state["z_full_blocks"],
        z_late=state["z_late"],
        xy_late=state["xy_late"],
        **kwargs,
    )


def _lane_output(lane: dict) -> dict[str, np.ndarray]:
    state = lane["state"]
    return {
        "full_steps": np.asarray(state["full_steps"], dtype=np.int32),
        "late_steps": np.asarray(state["late_steps"], dtype=np.int32),
        "z_late_steps": np.asarray(state.get("z_late_steps", state["late_steps"]), dtype=np.int32),
        "xy_late_steps": np.asarray(state.get("xy_late_steps", state["late_steps"]), dtype=np.int32),
        "z_full": _stack_or_empty(state["z_full"], dtype=np.float32),
        "z_full_blocks": _stack_or_empty(state["z_full_blocks"], dtype=np.float32),
        "z_late": _stack_or_empty(state["z_late"], dtype=np.float32),
        "xy_late": _stack_or_empty(state["xy_late"], dtype=np.float32),
    }


def _stack_or_empty(seq: list[np.ndarray], *, dtype=np.float32, trailing_shape: tuple[int, ...] = ()) -> np.ndarray:
    if not seq:
        return np.zeros((0,) + tuple(trailing_shape), dtype=dtype)
    return np.stack([np.asarray(x, dtype=dtype) for x in seq], axis=0)


def _load_init_params(args: Any, repo: Path) -> jax.Array:
    raw = getattr(args, "init_params_path", None)
    if raw is None or str(raw).strip() == "":
        raise ValueError("Flow-Lenia fixed-init frustration requires source.init_params_path.")
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise FileNotFoundError(f"Initialization params not found: {path}.")
    return jnp.asarray(np.asarray(np.load(path), dtype=np.float32))


def _flow_initial_global_state(
    *,
    substrate,
    params: jax.Array,
    init_params: jax.Array,
    run_seed: int,
    run_seed_protocol: str,
    log_clip_evolution: bool,
):
    if run_seed_protocol != "optimization_metric":
        rng = jax.random.PRNGKey(int(run_seed))
        _, init_key = jax.random.split(rng)
        return substrate.init_state(init_key, params)

    state_key = _optimizer_metric_parts(
        run_seed,
        log_clip_evolution=log_clip_evolution,
    )[0]
    state = substrate.init_state(state_key, params)
    init_state = substrate.init_state(state_key, init_params)
    state = dict(state)
    for key in ("A", "P", "Food", "t", "F"):
        if key in init_state:
            state[key] = init_state[key]
    return state


def _pad_flow_spatial_state(
    state: dict[str, jax.Array],
    *,
    pad_before: int,
    pad_after: int,
) -> dict[str, jax.Array]:
    if pad_before == 0 and pad_after == 0:
        return dict(state)
    height = int(state["A"].shape[0])
    width = int(state["A"].shape[1])
    out = dict(state)
    for key, value in state.items():
        if key not in {"A", "P", "F", "Food"}:
            continue
        arr = jnp.asarray(value)
        if arr.ndim < 2 or tuple(arr.shape[:2]) != (height, width):
            continue
        padding = (
            (int(pad_before), int(pad_after)),
            (int(pad_before), int(pad_after)),
        ) + ((0, 0),) * (arr.ndim - 2)
        out[key] = jnp.pad(arr, padding)
    return out


def _init_flow_lane_state(
    *,
    wall_mode: bool,
    run_seed: int,
    substrate,
    block_template,
    params: jax.Array,
    init_params: jax.Array,
    split_n: int,
    block_size: int,
    pad: int,
    wall_global_pad_before: int,
    wall_global_pad_after: int,
    run_seed_protocol: str,
    log_clip_evolution: bool,
    initial_state_override: dict[str, jax.Array] | None = None,
):
    if run_seed_protocol != "optimization_metric":
        return _init_run_checkpoint(
            wall_mode=wall_mode,
            run_seed=run_seed,
            substrate=substrate,
            block_template=block_template,
            params=params,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
        )

    initial_state = (
        initial_state_override
        if initial_state_override is not None
        else _flow_initial_global_state(
            substrate=substrate,
            params=params,
            init_params=init_params,
            run_seed=run_seed,
            run_seed_protocol=run_seed_protocol,
            log_clip_evolution=log_clip_evolution,
        )
    )
    common = dict(
        current_step=0,
        rng=jax.random.PRNGKey(int(run_seed)),
        full_steps=[],
        late_steps=[],
        z_late_steps=[],
        xy_late_steps=[],
        z_full=[],
        z_full_blocks=[],
        z_late=[],
        xy_late=[],
    )
    if not wall_mode:
        return {"mode": "global", "global_state": initial_state, **common}

    initial_state = _pad_flow_spatial_state(
        initial_state,
        pad_before=wall_global_pad_before,
        pad_after=wall_global_pad_after,
    )
    blocks = _prepare_block_template_state(
        initial_state=initial_state,
        block_template=block_template,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
        C=int(initial_state["A"].shape[-1]),
        k=int(initial_state["P"].shape[-1]),
    )
    return {"mode": "block", "block_state": blocks, **common}


def _reference_snapshot(apf_dir: Path, step: int) -> tuple[Path, dict[str, np.ndarray]]:
    for path in sorted(apf_dir.glob("P_steps_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            matches = np.flatnonzero(steps == int(step))
            if matches.size != 1:
                continue
            idx = int(matches[0])
            payload = {
                key: np.asarray(data[key][idx])
                for key in ("A", "P", "F", "state_t", "state_mass_cycle_start")
                if key in data.files
            }
            return path, payload
    raise FileNotFoundError(f"No exact step={step} snapshot found under {apf_dir}.")


def _assert_training_reference(lane: dict, state: dict[str, jax.Array], *, step: int) -> None:
    reference_step = lane.get("training_reference_step")
    if reference_step is None or int(step) != int(reference_step):
        return
    apf_dir = lane.get("reference_apf_dir")
    if apf_dir is None:
        if lane.get("require_training_reference_match", False):
            raise ValueError(f"{lane['variant']} is missing a required training reference APF directory.")
        return

    source_path, reference = _reference_snapshot(Path(apf_dir), int(step))
    state_np = jax.device_get(state)
    checks: dict[str, dict[str, float | bool | str]] = {}
    all_exact = True
    for key in ("A", "P", "F"):
        if key not in reference or key not in state_np:
            checks[key] = {"exact_after_reference_cast": False, "reason": "missing"}
            all_exact = False
            continue
        expected = np.asarray(reference[key])
        actual = np.asarray(state_np[key])
        actual_cast = actual.astype(expected.dtype)
        exact = bool(np.array_equal(actual_cast, expected))
        max_abs = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        checks[key] = {
            "exact_after_reference_cast": exact,
            "reference_dtype": str(expected.dtype),
            "max_abs_vs_reference_values": max_abs,
        }
        all_exact = all_exact and exact

    if "state_t" in reference:
        actual_t = int(np.asarray(state_np.get("t", -1)).item())
        expected_t = int(np.asarray(reference["state_t"]).item())
        exact_t = actual_t == expected_t
        checks["state_t"] = {"exact": exact_t, "actual": actual_t, "expected": expected_t}
        all_exact = all_exact and exact_t
    if "state_mass_cycle_start" in reference:
        actual_mass = float(np.asarray(state_np.get("mass_cycle_start", np.nan)).item())
        expected_mass = float(np.asarray(reference["state_mass_cycle_start"]).item())
        exact_mass = bool(np.asarray(actual_mass, dtype=np.float32) == np.asarray(expected_mass, dtype=np.float32))
        checks["state_mass_cycle_start"] = {
            "exact_float32": exact_mass,
            "actual": actual_mass,
            "expected": expected_mass,
        }
        all_exact = all_exact and exact_mass

    proof = {
        "status": "exact" if all_exact else "mismatch",
        "variant": str(lane["variant"]),
        "run_seed": int(lane["run_seed"]),
        "step": int(step),
        "reference_apf_file": str(source_path),
        "checks": checks,
    }
    proof_path = lane["trial"]["trial_paths"]["trial_artifact_dir"] / f"{lane['variant']}_training_reference_check.json"
    _write_json(proof_path, proof)
    lane["training_reference_proof"] = proof
    if not all_exact:
        raise RuntimeError(
            f"{lane['variant']} diverged from optimizer-native training trajectory at step={step}; "
            f"details={proof_path}."
        )


def _sha256_array(value: jax.Array | np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(jax.device_get(value), dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _bootstrap_cache_metadata(
    *,
    group_idx: int,
    group_trials: list[dict],
    pop_path: Path,
    optimizer_iter: int,
    training_horizon_steps: int,
    base_chunk_steps: int,
    full_embedding_sample_every_steps: int,
    log_clip_evolution: bool,
) -> dict[str, Any]:
    stat = pop_path.stat()
    return {
        "protocol": "optimizer_native_nested_vmap_state_v1",
        "group_idx": int(group_idx),
        "source_pop_traj": str(pop_path.resolve()),
        "source_pop_traj_size": int(stat.st_size),
        "source_pop_traj_mtime_ns": int(stat.st_mtime_ns),
        "optimizer_iter": int(optimizer_iter),
        "training_horizon_steps": int(training_horizon_steps),
        "base_chunk_steps": int(base_chunk_steps),
        "full_embedding_sample_every_steps": int(full_embedding_sample_every_steps),
        "log_clip_evolution": bool(log_clip_evolution),
        "trials": [
            {
                "trial_idx": int(trial["trial_idx"]),
                "candidate_kind": str(getattr(trial["args"], "candidate_kind")),
                "candidate_idx": int(getattr(trial["args"], "candidate_idx", 0)),
                "source_pop_idx": int(
                    getattr(trial["args"], "optimizer_native_source_pop_idx")
                ),
                "execution_pop_idx": int(
                    getattr(trial["args"], "optimizer_native_execution_pop_idx")
                ),
                "control_a_seed_idx": int(
                    getattr(trial["args"], "control_a_optimizer_native_seed_idx")
                ),
                "control_b_seed_idx": int(
                    getattr(trial["args"], "control_b_optimizer_native_seed_idx")
                ),
                "use_row_params": bool(
                    getattr(trial["args"], "optimizer_native_use_row_params")
                ),
                "params_sha256": _sha256_array(trial["params"]),
            }
            for trial in sorted(group_trials, key=lambda item: int(item["trial_idx"]))
        ],
    }


def _put_state_payload(
    payload: dict[str, np.ndarray],
    *,
    prefix: str,
    state: dict[str, jax.Array],
) -> list[str]:
    keys = []
    for key, value in sorted(state.items()):
        payload[f"{prefix}__{key}"] = np.asarray(jax.device_get(value))
        keys.append(str(key))
    return keys


def _state_from_payload(
    data: np.lib.npyio.NpzFile,
    *,
    prefix: str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    return {key: np.asarray(data[f"{prefix}__{key}"]) for key in keys}


def _load_optimizer_native_bootstrap_cache(
    cache_path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> dict[int, dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        metadata_common = {
            key: value for key, value in metadata.items() if key != "trials"
        }
        expected_common = {
            key: value for key, value in expected_metadata.items() if key != "trials"
        }
        cached_trials = {
            int(row["trial_idx"]): row for row in metadata.get("trials", [])
        }
        expected_trials = {
            int(row["trial_idx"]): row
            for row in expected_metadata.get("trials", [])
        }
        mismatched_trials = [
            trial_idx
            for trial_idx, expected_row in expected_trials.items()
            if cached_trials.get(trial_idx) != expected_row
        ]
        if metadata_common != expected_common or mismatched_trials:
            raise RuntimeError(
                "Refusing stale optimizer-native bootstrap cache: "
                f"{cache_path}; mismatched_trials={mismatched_trials}."
            )
        outputs: dict[int, dict[str, Any]] = {}
        for trial_meta in expected_metadata["trials"]:
            trial_idx = int(trial_meta["trial_idx"])
            prefix = f"trial_{trial_idx:05d}"
            state_keys = json.loads(str(np.asarray(data[f"{prefix}__state_keys_json"]).item()))
            outputs[trial_idx] = {
                "initial_state": _state_from_payload(
                    data,
                    prefix=f"{prefix}__initial",
                    keys=state_keys,
                ),
                "control_a_state": _state_from_payload(
                    data,
                    prefix=f"{prefix}__control_a",
                    keys=state_keys,
                ),
                "control_b_state": _state_from_payload(
                    data,
                    prefix=f"{prefix}__control_b",
                    keys=state_keys,
                ),
                "full_steps": np.asarray(data[f"{prefix}__full_steps"], dtype=np.int32),
                "z_full": np.asarray(data[f"{prefix}__z_full"], dtype=np.float32),
            }
    return outputs


def _save_optimizer_native_bootstrap_cache(
    cache_path: Path,
    *,
    metadata: dict[str, Any],
    outputs: dict[int, dict[str, Any]],
) -> None:
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for trial_idx, output in sorted(outputs.items()):
        prefix = f"trial_{int(trial_idx):05d}"
        state_keys = _put_state_payload(
            payload,
            prefix=f"{prefix}__initial",
            state=output["initial_state"],
        )
        _put_state_payload(
            payload,
            prefix=f"{prefix}__control_a",
            state=output["control_a_state"],
        )
        _put_state_payload(
            payload,
            prefix=f"{prefix}__control_b",
            state=output["control_b_state"],
        )
        payload[f"{prefix}__state_keys_json"] = np.asarray(json.dumps(state_keys))
        payload[f"{prefix}__full_steps"] = np.asarray(output["full_steps"], dtype=np.int32)
        payload[f"{prefix}__z_full"] = np.asarray(output["z_full"], dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz_atomic(cache_path, **payload)


def _optimizer_native_bootstrap(
    *,
    trials: list[dict],
    substrate,
    root_save_dir: Path,
    training_horizon_steps: int,
    base_chunk_steps: int,
    full_embedding_sample_every_steps: int,
    log_clip_evolution: bool,
    enable_clip: bool,
    embed_global_batch,
) -> dict[int, dict[str, Any]]:
    if training_horizon_steps % base_chunk_steps != 0:
        raise ValueError(
            "Optimizer-native bootstrap horizon must be divisible by base_chunk_steps."
        )
    if enable_clip and full_embedding_sample_every_steps % base_chunk_steps != 0:
        raise ValueError(
            "Optimizer-native bootstrap full embedding cadence must be divisible by base_chunk_steps."
        )

    grouped: dict[int, list[dict]] = {}
    for trial in trials:
        grouped.setdefault(
            int(getattr(trial["args"], "optimized_run_idx")),
            [],
        ).append(trial)

    def init_one(eval_key, params, init_params):
        rng_roll = jax.random.split(
            eval_key,
            3 if log_clip_evolution else 2,
        )[0]
        k_state, _k_pts, _k_ch, k_scan = jax.random.split(rng_roll, 4)
        state = substrate.init_state(k_state, params)
        init_state = substrate.init_state(k_state, init_params)
        state = dict(state)
        for key in ("A", "P", "Food", "t", "F"):
            if key in init_state:
                state[key] = init_state[key]
        return state, k_scan

    @jax.jit
    def init_all(params_grid, init_params_grid, seed_keys):
        def init_for_params(params, init_params):
            return jax.vmap(
                lambda eval_key: init_one(eval_key, params, init_params)
            )(seed_keys)

        states, scan_keys = jax.vmap(init_for_params)(
            params_grid,
            init_params_grid,
        )
        return states, scan_keys[0]

    @jax.jit
    def advance_chunk(states, params_grid, chunk_keys):
        def advance_for_params(seed_states, params):
            def advance_one_seed(chunk_key, state):
                step_keys = jax.random.split(chunk_key, base_chunk_steps)

                def body(carry, step_key):
                    return substrate.step_state(step_key, carry, params), None

                return jax.lax.scan(body, state, step_keys)[0]

            return jax.vmap(advance_one_seed)(chunk_keys, seed_states)

        return jax.vmap(advance_for_params)(states, params_grid)

    all_outputs: dict[int, dict[str, Any]] = {}
    n_chunks = int(training_horizon_steps // base_chunk_steps)
    for group_idx, group_trials in sorted(grouped.items()):
        first_args = group_trials[0]["args"]
        pop_path = Path(str(getattr(first_args, "optimizer_native_source_pop_traj"))).resolve()
        optimizer_iter = int(getattr(first_args, "optimizer_native_iter"))
        metadata = _bootstrap_cache_metadata(
            group_idx=group_idx,
            group_trials=group_trials,
            pop_path=pop_path,
            optimizer_iter=optimizer_iter,
            training_horizon_steps=training_horizon_steps,
            base_chunk_steps=base_chunk_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
            log_clip_evolution=log_clip_evolution,
        )
        cache_path = (
            root_save_dir
            / "optimizer_native_bootstrap"
            / f"group_{group_idx:03d}_step_{training_horizon_steps:07d}.npz"
        )
        cached = _load_optimizer_native_bootstrap_cache(
            cache_path,
            expected_metadata=metadata,
        )
        if cached is not None:
            print(
                f"[paper_check/frustration/bootstrap] reused group={group_idx} "
                f"cache={cache_path}"
            )
            all_outputs.update(cached)
            continue

        with pop_path.open("rb") as f:
            pop = pickle.load(f)
        pop_params = np.asarray(pop["params"], dtype=np.float32)
        pop_seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)
        if optimizer_iter < 0 or optimizer_iter >= pop_params.shape[0]:
            raise IndexError(
                f"optimizer_iter={optimizer_iter} outside params shape={pop_params.shape}."
            )
        params_grid_np = np.asarray(pop_params[optimizer_iter], dtype=np.float32).copy()
        init_params_grid_np = np.asarray(params_grid_np, dtype=np.float32).copy()
        seed_keys_np = np.asarray(pop_seed_keys[optimizer_iter], dtype=np.uint32)
        expected_population = int(getattr(first_args, "optimizer_native_population_size"))
        expected_seed_count = int(getattr(first_args, "optimizer_native_seed_count"))
        if params_grid_np.shape[0] != expected_population or seed_keys_np.shape[0] != expected_seed_count:
            raise ValueError(
                f"Optimizer-native context shape mismatch for group={group_idx}: "
                f"params={params_grid_np.shape}, seed_keys={seed_keys_np.shape}, "
                f"expected={expected_population}x{expected_seed_count}."
            )

        occupied: dict[int, str] = {}
        for trial in group_trials:
            args = trial["args"]
            exec_idx = int(getattr(args, "optimizer_native_execution_pop_idx"))
            source_idx = int(getattr(args, "optimizer_native_source_pop_idx"))
            candidate_params = np.asarray(jax.device_get(trial["params"]), dtype=np.float32)
            label = str(getattr(args, "candidate_label"))
            previous = occupied.get(exec_idx)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Execution population lane collision for group={group_idx}, lane={exec_idx}: "
                    f"{previous} vs {label}."
                )
            occupied[exec_idx] = label
            if bool(getattr(args, "optimizer_native_use_row_params")):
                params_grid_np[exec_idx] = candidate_params
                init_params_grid_np[exec_idx] = pop_params[optimizer_iter, source_idx]
            else:
                source_params = np.asarray(
                    pop_params[optimizer_iter, source_idx],
                    dtype=np.float32,
                )
                if exec_idx != source_idx or not np.array_equal(candidate_params, source_params):
                    raise RuntimeError(
                        f"Direct optimizer-native candidate mismatch for group={group_idx}, "
                        f"trial={trial['trial_idx']}."
                    )

        params_grid = jnp.asarray(params_grid_np, dtype=jnp.float32)
        init_params_grid = jnp.asarray(init_params_grid_np, dtype=jnp.float32)
        seed_keys = jnp.asarray(seed_keys_np, dtype=jnp.uint32)
        states, scan_keys = init_all(params_grid, init_params_grid, seed_keys)
        scan_chunk_keys = jax.jit(
            lambda keys: jax.vmap(lambda key: jax.random.split(key, n_chunks))(keys)
        )(scan_keys)

        group_outputs: dict[int, dict[str, Any]] = {}
        for trial in group_trials:
            args = trial["args"]
            exec_idx = int(getattr(args, "optimizer_native_execution_pop_idx"))
            seed_a_idx = int(getattr(args, "control_a_optimizer_native_seed_idx"))
            initial_state = jax.tree_util.tree_map(
                lambda value, e=exec_idx, s=seed_a_idx: value[e, s],
                states,
            )
            group_outputs[int(trial["trial_idx"])] = {
                "initial_state": initial_state,
                "full_steps": [],
                "z_full": [],
            }

        pbar = _make_progress_bar(
            total=training_horizon_steps,
            desc=f"optimizer-native bootstrap group={group_idx:03d}",
        )
        for chunk_idx in range(n_chunks):
            states = advance_chunk(
                states,
                params_grid,
                scan_chunk_keys[:, chunk_idx],
            )
            step = int((chunk_idx + 1) * base_chunk_steps)
            if (
                enable_clip
                and step % int(full_embedding_sample_every_steps) == 0
            ):
                selected_states = []
                selected_params = []
                selected_trial_ids = []
                for trial in group_trials:
                    args = trial["args"]
                    exec_idx = int(getattr(args, "optimizer_native_execution_pop_idx"))
                    seed_a_idx = int(getattr(args, "control_a_optimizer_native_seed_idx"))
                    selected_states.append(
                        jax.tree_util.tree_map(
                            lambda value, e=exec_idx, s=seed_a_idx: value[e, s],
                            states,
                        )
                    )
                    selected_params.append(trial["params"])
                    selected_trial_ids.append(int(trial["trial_idx"]))
                z_host = np.asarray(
                    jax.device_get(
                        embed_global_batch(
                            _stack_trees(selected_states),
                            jnp.stack(selected_params, axis=0),
                        )
                    ),
                    dtype=np.float32,
                )
                for local_idx, trial_idx in enumerate(selected_trial_ids):
                    group_outputs[trial_idx]["full_steps"].append(step)
                    group_outputs[trial_idx]["z_full"].append(z_host[local_idx])
            pbar.update(base_chunk_steps)
        pbar.close()

        for trial in group_trials:
            args = trial["args"]
            trial_idx = int(trial["trial_idx"])
            exec_idx = int(getattr(args, "optimizer_native_execution_pop_idx"))
            seed_a_idx = int(getattr(args, "control_a_optimizer_native_seed_idx"))
            seed_b_idx = int(getattr(args, "control_b_optimizer_native_seed_idx"))
            group_outputs[trial_idx]["control_a_state"] = jax.tree_util.tree_map(
                lambda value, e=exec_idx, s=seed_a_idx: value[e, s],
                states,
            )
            group_outputs[trial_idx]["control_b_state"] = jax.tree_util.tree_map(
                lambda value, e=exec_idx, s=seed_b_idx: value[e, s],
                states,
            )
            group_outputs[trial_idx]["full_steps"] = np.asarray(
                group_outputs[trial_idx]["full_steps"],
                dtype=np.int32,
            )
            group_outputs[trial_idx]["z_full"] = _stack_or_empty(
                group_outputs[trial_idx]["z_full"],
                dtype=np.float32,
            )

        _save_optimizer_native_bootstrap_cache(
            cache_path,
            metadata=metadata,
            outputs=group_outputs,
        )
        for output in group_outputs.values():
            for state_key in ("initial_state", "control_a_state", "control_b_state"):
                output[state_key] = jax.tree_util.tree_map(
                    lambda value: np.asarray(jax.device_get(value)),
                    output[state_key],
                )
        print(
            f"[paper_check/frustration/bootstrap] completed group={group_idx} "
            f"cache={cache_path}"
        )
        all_outputs.update(group_outputs)
    return all_outputs


def _load_or_init_lane(
    lane: dict,
    *,
    resume: bool,
    substrate,
    block_template,
    split_n: int,
    block_size: int,
    pad: int,
    wall_global_pad_before: int,
    wall_global_pad_after: int,
):
    state = _load_run_checkpoint(lane["checkpoint_path"]) if resume else None
    if state is None:
        bootstrap_state = lane.get("bootstrap_terminal_state")
        if bootstrap_state is not None and not bool(lane["wall_mode"]):
            state = {
                "mode": "global",
                "global_state": bootstrap_state,
                "current_step": int(lane["training_horizon_steps"]),
                "rng": jax.random.PRNGKey(int(lane["run_seed"])),
                "full_steps": [
                    int(value) for value in np.asarray(
                        lane.get("bootstrap_full_steps", []),
                        dtype=np.int32,
                    )
                ],
                "late_steps": [],
                "z_late_steps": [],
                "xy_late_steps": [],
                "z_full": [
                    np.asarray(value, dtype=np.float32)
                    for value in np.asarray(
                        lane.get("bootstrap_z_full", []),
                        dtype=np.float32,
                    )
                ],
                "z_full_blocks": [],
                "z_late": [],
                "xy_late": [],
            }
        else:
            state = _init_flow_lane_state(
                wall_mode=bool(lane["wall_mode"]),
                run_seed=int(lane["run_seed"]),
                substrate=substrate,
                block_template=block_template,
                params=lane["params"],
                init_params=lane["init_params"],
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                wall_global_pad_before=wall_global_pad_before,
                wall_global_pad_after=wall_global_pad_after,
                run_seed_protocol=str(lane.get("run_seed_protocol", "legacy")),
                log_clip_evolution=bool(lane.get("log_clip_evolution", False)),
                initial_state_override=lane.get("bootstrap_initial_state"),
            )
    if str(lane.get("run_seed_protocol", "legacy")) == "optimization_metric":
        lane["rng_schedule"] = _optimizer_metric_key_schedule(
            run_seed=int(lane["run_seed"]),
            total_steps=int(lane["total_steps"]),
            training_horizon_steps=int(lane["training_horizon_steps"]),
            chunk_steps=int(lane["base_chunk_steps"]),
            log_clip_evolution=bool(lane.get("log_clip_evolution", False)),
        )
    reference_step = lane.get("training_reference_step")
    proof_path = lane["trial"]["trial_paths"]["trial_artifact_dir"] / f"{lane['variant']}_training_reference_check.json"
    if (
        lane.get("require_training_reference_match", False)
        and reference_step is not None
        and int(state["current_step"]) > int(reference_step)
        and not proof_path.exists()
    ):
        raise RuntimeError(
            f"Cannot resume {lane['variant']} beyond training reference step={reference_step} "
            f"without proof file {proof_path}."
        )
    lane["state"] = state


def _group_active_lanes(lanes: list[dict], *, total_steps: int) -> list[list[int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for idx, lane in enumerate(lanes):
        state = lane["state"]
        if int(state["current_step"]) >= int(total_steps):
            continue
        key = (str(state["mode"]), int(state["current_step"]))
        groups.setdefault(key, []).append(idx)
    order = {"block": 0, "global": 1, "lag": 2}
    return [groups[key] for key in sorted(groups.keys(), key=lambda item: (item[1], order.get(item[0], 99)))]


def _append_full_subset(lanes: list[dict], group_indices: list[int], z_host: np.ndarray, *, z_blocks_host: np.ndarray | None = None):
    for local_idx, lane_idx in enumerate(group_indices):
        lane = lanes[lane_idx]
        if not lane["full_embeddings_enabled"]:
            continue
        state = lane["state"]
        state["full_steps"].append(int(state["current_step"]))
        state["z_full"].append(np.asarray(z_host[local_idx], dtype=np.float32))
        if lane["block_embeddings_enabled"]:
            if z_blocks_host is None:
                raise ValueError("z_blocks_host must be provided for block embedding lanes.")
            state["z_full_blocks"].append(np.asarray(z_blocks_host[local_idx], dtype=np.float32))


def _append_late_all(lanes: list[dict], group_indices: list[int], *, z_host: np.ndarray | None, xy_host: np.ndarray | None = None):
    for local_idx, lane_idx in enumerate(group_indices):
        state = lanes[lane_idx]["state"]
        state["late_steps"].append(int(state["current_step"]))
        if z_host is not None:
            state.setdefault("z_late_steps", []).append(int(state["current_step"]))
            state["z_late"].append(np.asarray(z_host[local_idx], dtype=np.float32))
        if xy_host is not None:
            state.setdefault("xy_late_steps", []).append(int(state["current_step"]))
            state["xy_late"].append(np.asarray(xy_host[local_idx], dtype=np.float32))


def _maybe_save_group(lanes: list[dict], group_indices: list[int], *, checkpoint_every_steps: int, total_steps: int):
    for lane_idx in group_indices:
        step = int(lanes[lane_idx]["state"]["current_step"])
        if step >= int(total_steps) or step % int(checkpoint_every_steps) == 0:
            _save_lane_checkpoint(lanes[lane_idx])


def _load_or_init_generic_lane(
    lane: dict,
    *,
    resume: bool,
    substrate,
):
    state = _load_run_checkpoint(lane["checkpoint_path"]) if resume else None
    if state is None:
        rng = jax.random.PRNGKey(int(lane["run_seed"]))
        rng, init_key = jax.random.split(rng)
        state = dict(
            mode="global",
            current_step=0,
            rng=rng,
            global_state=substrate.init_state(init_key, lane["params"]),
            full_steps=[],
            late_steps=[],
            z_late_steps=[],
            xy_late_steps=[],
            z_full=[],
            z_full_blocks=[],
            z_late=[],
            xy_late=[],
        )
        if bool(lane.get("apply_perturbation", False)) and int(lane.get("warmup_steps", 0)) == 0:
            state["global_state"] = _apply_position_perturbation(
                state["global_state"],
                jax.random.fold_in(rng, jnp.uint32(0x50545242)),
                args=lane["args"],
                substrate=substrate,
            )
    lane["state"] = state


def _run_generic_global_lanes(
    *,
    lanes: list[dict],
    total_steps: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
    clip_sample_every_steps: int,
    checkpoint_every_steps: int,
    full_embedding_sample_every_steps: int,
    enable_clip: bool,
    enable_msc: bool,
    batched_state_chunk_stepper,
    embed_global_batch,
    substrate,
):
    if not lanes:
        return

    pbar = _make_progress_bar(
        total=_remaining_lane_steps(lanes, total_steps=total_steps),
        desc=f"frustration generic lanes={len(lanes)}",
    )
    while True:
        active_groups = _group_active_lanes(lanes, total_steps=total_steps)
        if not active_groups:
            break

        for group_indices in active_groups:
            mode = str(lanes[group_indices[0]]["state"]["mode"])
            if mode != "global":
                raise ValueError(f"Generic explicit-position lanes only support mode='global', got {mode!r}.")
            current_step = int(lanes[group_indices[0]]["state"]["current_step"])
            params_batch = jnp.stack([lanes[idx]["params"] for idx in group_indices], axis=0)
            rng_batch = jnp.stack([lanes[idx]["state"]["rng"] for idx in group_indices], axis=0)
            rng_next, chunk_keys = _split_rng_batch(rng_batch)
            next_step = int(current_step + base_chunk_steps)

            global_batch = _stack_trees([lanes[idx]["state"]["global_state"] for idx in group_indices])
            step_keys_batch = _make_step_keys_batch(chunk_keys, base_chunk_steps)
            global_batch = batched_state_chunk_stepper(step_keys_batch, global_batch, params_batch)
            global_states = _unstack_tree(global_batch)

            for local_idx, lane_idx in enumerate(group_indices):
                lane = lanes[lane_idx]
                if (
                    bool(lane.get("apply_perturbation", False))
                    and current_step < int(warmup_steps) <= next_step
                ):
                    global_states[local_idx] = _apply_position_perturbation(
                        global_states[local_idx],
                        jax.random.fold_in(chunk_keys[local_idx], jnp.uint32(0x50545242)),
                        args=lane["args"],
                        substrate=substrate,
                    )

            global_batch = _stack_trees(global_states)
            xy_host = None
            in_late_window = bool(int(late_start) < next_step <= int(late_end))
            need_late_xy = bool(enable_msc and in_late_window)
            need_late_z = bool(
                enable_clip
                and in_late_window
                and ((next_step - int(late_start)) % int(clip_sample_every_steps) == 0)
            )
            need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
            if need_late_xy:
                xy_host = np.asarray(jax.device_get(_extract_positions_from_state(global_batch)), dtype=np.float32)

            if enable_clip and (need_late_z or need_full):
                z_all_host = np.asarray(jax.device_get(embed_global_batch(global_batch, params_batch)), dtype=np.float32)
            else:
                z_all_host = None

            for local_idx, lane_idx in enumerate(group_indices):
                lane = lanes[lane_idx]
                state = lane["state"]
                state["rng"] = rng_next[local_idx]
                state["current_step"] = next_step
                state["global_state"] = global_states[local_idx]
                state["mode"] = "global"

                if need_late_z or need_late_xy:
                    state["late_steps"].append(next_step)
                    if need_late_z:
                        state.setdefault("z_late_steps", []).append(next_step)
                        state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                    if need_late_xy:
                        state.setdefault("xy_late_steps", []).append(next_step)
                        state["xy_late"].append(np.asarray(xy_host[local_idx], dtype=np.float32))

                if need_full and lane["full_embeddings_enabled"]:
                    state["full_steps"].append(next_step)
                    state["z_full"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))

            _maybe_save_group(
                lanes,
                group_indices,
                checkpoint_every_steps=checkpoint_every_steps,
                total_steps=total_steps,
            )
            pbar.update(
                _progress_delta(
                    current_step=current_step,
                    next_step=next_step,
                    total_steps=total_steps,
                    n_lanes=len(group_indices),
                )
            )
    pbar.close()


def _run_control_lanes(
    *,
    lanes: list[dict],
    total_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
    clip_sample_every_steps: int,
    checkpoint_every_steps: int,
    full_embedding_sample_every_steps: int,
    enable_clip: bool,
    enable_msc: bool,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_channel_mode: str,
    batched_state_chunk_stepper,
    batched_lagrangian_chunk_stepper,
    embed_global_batch,
):
    if not lanes:
        return

    pbar = _make_progress_bar(
        total=_remaining_lane_steps(lanes, total_steps=total_steps),
        desc=f"frustration control lanes={len(lanes)}",
    )
    while True:
        active_groups = _group_active_lanes(lanes, total_steps=total_steps)
        if not active_groups:
            break

        for group_indices in active_groups:
            mode = str(lanes[group_indices[0]]["state"]["mode"])
            current_step = int(lanes[group_indices[0]]["state"]["current_step"])
            params_batch = jnp.stack([lanes[idx]["params"] for idx in group_indices], axis=0)
            rng_next, chunk_keys = _next_lane_chunk_keys(
                lanes,
                group_indices,
                current_step=current_step,
                chunk_steps=base_chunk_steps,
            )
            next_step = int(current_step + base_chunk_steps)

            if mode == "global":
                global_batch = _stack_trees([lanes[idx]["state"]["global_state"] for idx in group_indices])
                step_keys_batch = _make_step_keys_batch(chunk_keys, base_chunk_steps)
                global_batch = batched_state_chunk_stepper(step_keys_batch, global_batch, params_batch)
                global_states = _unstack_tree(global_batch)

                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                need_late = bool(
                    (not enable_msc)
                    and int(late_start) < next_step <= int(late_end)
                    and (next_step - int(late_start)) % int(clip_sample_every_steps) == 0
                )
                z_all_host = None
                if enable_clip and need_late:
                    z_all_host = np.asarray(jax.device_get(embed_global_batch(global_batch, params_batch)), dtype=np.float32)
                elif enable_clip and need_full:
                    full_locals = [local for local, lane_idx in enumerate(group_indices) if lanes[lane_idx]["full_embeddings_enabled"]]
                    if full_locals:
                        global_full_batch = _take_tree(global_batch, full_locals)
                        params_full_batch = params_batch[np.asarray(full_locals, dtype=np.int32)]
                        z_full_host = np.asarray(
                            jax.device_get(embed_global_batch(global_full_batch, params_full_batch)),
                            dtype=np.float32,
                        )
                        for arr_local, local_idx in enumerate(full_locals):
                            lane = lanes[group_indices[local_idx]]
                            lane["state"]["full_steps"].append(next_step)
                            lane["state"]["z_full"].append(np.asarray(z_full_host[arr_local], dtype=np.float32))

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    _assert_training_reference(lane, global_states[local_idx], step=next_step)
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if enable_msc and next_step == int(late_start):
                        key_pts, key_ch = _lag_init_keys(int(lane["run_seed"]))
                        state["lag_carry"] = _init_lagrangian_carry(
                            substrate=lane["substrate"],
                            state0=global_states[local_idx],
                            key_pts=key_pts,
                            key_ch=key_ch,
                            lag_n_particles=lag_n_particles,
                            lag_init_mode=lag_init_mode,
                            lag_channel_mode=lag_channel_mode,
                        )
                        state.pop("global_state", None)
                        state["mode"] = "lag"
                    else:
                        state["global_state"] = global_states[local_idx]
                        state["mode"] = "global"
                        if z_all_host is not None:
                            state["late_steps"].append(next_step)
                            state.setdefault("z_late_steps", []).append(next_step)
                            state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                        if need_full and lane["full_embeddings_enabled"] and z_all_host is not None:
                            state["full_steps"].append(next_step)
                            state["z_full"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))

            elif mode == "lag":
                lag_batch = _stack_trees([lanes[idx]["state"]["lag_carry"] for idx in group_indices])
                lag_batch, xy_batch = batched_lagrangian_chunk_stepper(chunk_keys, lag_batch, params_batch)
                global_batch = lag_batch[0]
                global_states = _unstack_tree(global_batch)
                lag_carries = _unstack_tree(lag_batch)

                need_late_xy = bool(next_step <= int(late_end))
                need_late_z = bool(
                    enable_clip
                    and next_step <= int(late_end)
                    and (next_step - int(late_start)) % int(clip_sample_every_steps) == 0
                )
                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                z_all_host = None
                if need_late_z:
                    z_all_host = np.asarray(jax.device_get(embed_global_batch(global_batch, params_batch)), dtype=np.float32)
                elif enable_clip and need_full:
                    full_locals = [local for local, lane_idx in enumerate(group_indices) if lanes[lane_idx]["full_embeddings_enabled"]]
                    if full_locals:
                        global_full_batch = _take_tree(global_batch, full_locals)
                        params_full_batch = params_batch[np.asarray(full_locals, dtype=np.int32)]
                        z_full_host = np.asarray(
                            jax.device_get(embed_global_batch(global_full_batch, params_full_batch)),
                            dtype=np.float32,
                        )
                        for arr_local, local_idx in enumerate(full_locals):
                            lane = lanes[group_indices[local_idx]]
                            lane["state"]["full_steps"].append(next_step)
                            lane["state"]["z_full"].append(np.asarray(z_full_host[arr_local], dtype=np.float32))

                xy_host = np.asarray(jax.device_get(xy_batch), dtype=np.float32) if need_late_xy else None

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if z_all_host is not None:
                        state["late_steps"].append(next_step)
                        state.setdefault("z_late_steps", []).append(next_step)
                        state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                    if xy_host is not None:
                        state.setdefault("xy_late_steps", []).append(next_step)
                        state["xy_late"].append(np.asarray(xy_host[local_idx], dtype=np.float32))
                    if need_full and lane["full_embeddings_enabled"] and z_all_host is not None:
                        state["full_steps"].append(next_step)
                        state["z_full"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                    if next_step == int(late_end) and next_step < int(total_steps):
                        state["global_state"] = global_states[local_idx]
                        state.pop("lag_carry", None)
                        state["mode"] = "global"
                    else:
                        state["lag_carry"] = lag_carries[local_idx]
                        state["mode"] = "lag"

            else:
                raise ValueError(f"Control lanes do not support mode={mode!r}.")

            _maybe_save_group(
                lanes,
                group_indices,
                checkpoint_every_steps=checkpoint_every_steps,
                total_steps=total_steps,
            )
            pbar.update(
                _progress_delta(
                    current_step=current_step,
                    next_step=next_step,
                    total_steps=total_steps,
                    n_lanes=len(group_indices),
                )
            )
    pbar.close()


def _run_walls_lanes(
    *,
    lanes: list[dict],
    total_steps: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
    clip_sample_every_steps: int,
    checkpoint_every_steps: int,
    full_embedding_sample_every_steps: int,
    enable_clip: bool,
    enable_msc: bool,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_channel_mode: str,
    split_n: int,
    block_size: int,
    pad: int,
    batched_state_chunk_stepper,
    batched_block_chunk_stepper,
    batched_lagrangian_chunk_stepper,
    embed_global_batch,
    embed_blocks_from_block_state_batch,
    embed_blocks_from_global_state_batch,
    embed_concat_from_block_state_batch,
    render_global_video_batch,
    render_blocks_video_batch,
    wall_video_sample_every_steps: int,
):
    if not lanes:
        return

    pbar = _make_progress_bar(
        total=_remaining_lane_steps(lanes, total_steps=total_steps),
        desc=f"frustration walls lanes={len(lanes)}",
    )
    while True:
        active_groups = _group_active_lanes(lanes, total_steps=total_steps)
        if not active_groups:
            break

        for group_indices in active_groups:
            mode = str(lanes[group_indices[0]]["state"]["mode"])
            current_step = int(lanes[group_indices[0]]["state"]["current_step"])
            params_batch = jnp.stack([lanes[idx]["params"] for idx in group_indices], axis=0)
            rng_next, chunk_keys = _next_lane_chunk_keys(
                lanes,
                group_indices,
                current_step=current_step,
                chunk_steps=base_chunk_steps,
            )
            next_step = int(current_step + base_chunk_steps)

            if mode == "block":
                block_batch = _stack_trees([lanes[idx]["state"]["block_state"] for idx in group_indices])
                block_batch = batched_block_chunk_stepper(chunk_keys, block_batch, params_batch)
                block_states = _unstack_tree(block_batch)

                if enable_clip and next_step % int(full_embedding_sample_every_steps) == 0:
                    z_full_host = np.asarray(
                        jax.device_get(embed_concat_from_block_state_batch(block_batch, params_batch)),
                        dtype=np.float32,
                    )
                    z_blocks_host = np.asarray(
                        jax.device_get(embed_blocks_from_block_state_batch(block_batch, params_batch)),
                        dtype=np.float32,
                    )
                else:
                    z_full_host = None
                    z_blocks_host = None
                need_video = bool(
                    render_blocks_video_batch is not None
                    and next_step % int(wall_video_sample_every_steps) == 0
                )
                video_host = (
                    np.asarray(
                        jax.device_get(render_blocks_video_batch(block_batch, params_batch)),
                        dtype=np.float32,
                    )
                    if need_video
                    else None
                )

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    if video_host is not None and lane.get("wall_video_enabled", False):
                        _save_wall_video_frame(
                            lane,
                            video_host[local_idx],
                            step=next_step,
                            warmup_steps=warmup_steps,
                            split_n=split_n,
                        )
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if z_full_host is not None:
                        state["full_steps"].append(next_step)
                        state["z_full"].append(np.asarray(z_full_host[local_idx], dtype=np.float32))
                        state["z_full_blocks"].append(np.asarray(z_blocks_host[local_idx], dtype=np.float32))

                    if next_step == int(warmup_steps):
                        merged_state = lane["merge_blocks"](
                            lane["initial_global_state"],
                            block_states[local_idx],
                        )
                        state.pop("block_state", None)
                        if enable_msc and next_step == int(late_start):
                            key_pts, key_ch = _lag_init_keys(int(lane["run_seed"]))
                            state["lag_carry"] = _init_lagrangian_carry(
                                substrate=lane["substrate"],
                                state0=merged_state,
                                key_pts=key_pts,
                                key_ch=key_ch,
                                lag_n_particles=lag_n_particles,
                                lag_init_mode=lag_init_mode,
                                lag_channel_mode=lag_channel_mode,
                            )
                            state["mode"] = "lag"
                        else:
                            state["global_state"] = merged_state
                            state["mode"] = "global"
                    else:
                        state["block_state"] = block_states[local_idx]
                        state["mode"] = "block"

            elif mode == "global":
                global_batch = _stack_trees([lanes[idx]["state"]["global_state"] for idx in group_indices])
                step_keys_batch = _make_step_keys_batch(chunk_keys, base_chunk_steps)
                global_batch = batched_state_chunk_stepper(step_keys_batch, global_batch, params_batch)
                global_states = _unstack_tree(global_batch)

                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                need_late = bool(
                    (not enable_msc)
                    and int(late_start) < next_step <= int(late_end)
                    and (next_step - int(late_start)) % int(clip_sample_every_steps) == 0
                )

                if enable_clip and (need_full or need_late):
                    z_all_host = np.asarray(jax.device_get(embed_global_batch(global_batch, params_batch)), dtype=np.float32)
                else:
                    z_all_host = None

                if enable_clip and need_full:
                    z_blocks_host = np.asarray(
                        jax.device_get(embed_blocks_from_global_state_batch(global_batch, params_batch)),
                        dtype=np.float32,
                    )
                else:
                    z_blocks_host = None
                need_video = bool(
                    render_global_video_batch is not None
                    and next_step % int(wall_video_sample_every_steps) == 0
                )
                video_host = (
                    np.asarray(
                        jax.device_get(render_global_video_batch(global_batch, params_batch)),
                        dtype=np.float32,
                    )
                    if need_video
                    else None
                )

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    if video_host is not None and lane.get("wall_video_enabled", False):
                        _save_wall_video_frame(
                            lane,
                            video_host[local_idx],
                            step=next_step,
                            warmup_steps=warmup_steps,
                            split_n=split_n,
                        )
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if need_full:
                        state["full_steps"].append(next_step)
                        state["z_full"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                        state["z_full_blocks"].append(np.asarray(z_blocks_host[local_idx], dtype=np.float32))

                    if enable_msc and next_step == int(late_start):
                        key_pts, key_ch = _lag_init_keys(int(lane["run_seed"]))
                        state["lag_carry"] = _init_lagrangian_carry(
                            substrate=lane["substrate"],
                            state0=global_states[local_idx],
                            key_pts=key_pts,
                            key_ch=key_ch,
                            lag_n_particles=lag_n_particles,
                            lag_init_mode=lag_init_mode,
                            lag_channel_mode=lag_channel_mode,
                        )
                        state.pop("global_state", None)
                        state["mode"] = "lag"
                    else:
                        state["global_state"] = global_states[local_idx]
                        state["mode"] = "global"
                        if need_late:
                            state["late_steps"].append(next_step)
                            state.setdefault("z_late_steps", []).append(next_step)
                            state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))

            elif mode == "lag":
                lag_batch = _stack_trees([lanes[idx]["state"]["lag_carry"] for idx in group_indices])
                lag_batch, xy_batch = batched_lagrangian_chunk_stepper(chunk_keys, lag_batch, params_batch)
                global_batch = lag_batch[0]
                global_states = _unstack_tree(global_batch)
                lag_carries = _unstack_tree(lag_batch)

                need_late_xy = bool(next_step <= int(late_end))
                need_late_z = bool(
                    enable_clip
                    and next_step <= int(late_end)
                    and (next_step - int(late_start)) % int(clip_sample_every_steps) == 0
                )
                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                if need_late_z or need_full:
                    z_all_host = np.asarray(jax.device_get(embed_global_batch(global_batch, params_batch)), dtype=np.float32)
                else:
                    z_all_host = None
                if enable_clip and need_full:
                    z_blocks_host = np.asarray(
                        jax.device_get(embed_blocks_from_global_state_batch(global_batch, params_batch)),
                        dtype=np.float32,
                    )
                else:
                    z_blocks_host = None
                need_video = bool(
                    render_global_video_batch is not None
                    and next_step % int(wall_video_sample_every_steps) == 0
                )
                video_host = (
                    np.asarray(
                        jax.device_get(render_global_video_batch(global_batch, params_batch)),
                        dtype=np.float32,
                    )
                    if need_video
                    else None
                )

                xy_host = np.asarray(jax.device_get(xy_batch), dtype=np.float32) if need_late_xy else None

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    if video_host is not None and lane.get("wall_video_enabled", False):
                        _save_wall_video_frame(
                            lane,
                            video_host[local_idx],
                            step=next_step,
                            warmup_steps=warmup_steps,
                            split_n=split_n,
                        )
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if need_late_xy or need_late_z:
                        state["late_steps"].append(next_step)
                        if need_late_z:
                            state.setdefault("z_late_steps", []).append(next_step)
                            state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                        if need_late_xy:
                            state.setdefault("xy_late_steps", []).append(next_step)
                            state["xy_late"].append(np.asarray(xy_host[local_idx], dtype=np.float32))
                    if need_full:
                        state["full_steps"].append(next_step)
                        state["z_full"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                        state["z_full_blocks"].append(np.asarray(z_blocks_host[local_idx], dtype=np.float32))
                    if next_step == int(late_end) and next_step < int(total_steps):
                        state["global_state"] = global_states[local_idx]
                        state.pop("lag_carry", None)
                        state["mode"] = "global"
                    else:
                        state["lag_carry"] = lag_carries[local_idx]
                        state["mode"] = "lag"

            else:
                raise ValueError(f"Walls lanes do not support mode={mode!r}.")

            _maybe_save_group(
                lanes,
                group_indices,
                checkpoint_every_steps=checkpoint_every_steps,
                total_steps=total_steps,
            )
            pbar.update(
                _progress_delta(
                    current_step=current_step,
                    next_step=next_step,
                    total_steps=total_steps,
                    n_lanes=len(group_indices),
                )
            )
    pbar.close()


def _finalize_trial(
    *,
    trial: dict,
    run_outputs: dict[str, dict[str, np.ndarray]],
    root_save_dir: Path,
    enable_clip: bool,
    enable_msc: bool,
    distance_metric: str,
    metric_cfg,
    metric_info,
    metric_eval,
):
    args = trial["args"]
    trial_paths = trial["trial_paths"]
    late_start = int(trial["late_start"])
    late_end = int(trial["late_end"])
    seed_x = int(getattr(args, "seed_x"))
    seed_x1 = int(getattr(args, "seed_x1"))

    row = {
        "trial_idx": int(trial["trial_idx"]),
        "optimized_run_idx": int(getattr(args, "optimized_run_idx")),
        "candidate_kind": str(getattr(args, "candidate_kind")),
        "candidate_idx": int(getattr(args, "candidate_idx", 0)),
        "candidate_label": str(getattr(args, "candidate_label")),
        "seed_x": int(seed_x),
        "seed_x1": int(seed_x1),
        "embeddings_path": str(trial_paths["trial_embeddings_npz"].relative_to(root_save_dir)) if enable_clip else None,
        "lagrangian_path": str(trial_paths["trial_lagrangian_npz"].relative_to(root_save_dir)) if enable_msc else None,
        "late_window_start_steps": int(late_start),
        "late_window_end_steps": int(late_end),
        "late_window_steps": int(late_end - late_start),
        "warmup_steps": int(getattr(args, "warmup_steps")),
        "total_steps": int(getattr(args, "total_steps")),
        "clip_time_sampling": int(run_outputs["control_a"]["z_late"].shape[0]) if enable_clip else None,
        "clip_sample_every_steps": int(getattr(args, "clip_sample_every_steps", getattr(args, "sample_every_steps", 0))) if enable_clip else None,
        "distance_metric": distance_metric if enable_clip else None,
        "foundation_model": None if not enable_clip else str(getattr(args, "foundation_model", "clip")),
    }

    if enable_clip:
        z_control_a = run_outputs["control_a"]["z_late"]
        z_control_b = run_outputs["control_b"]["z_late"]
        z_walls = run_outputs["walls"]["z_late"]
        if min(z_control_a.shape[0], z_control_b.shape[0], z_walls.shape[0]) < 1:
            raise ValueError(
                "No late CLIP samples were collected. Decrease evaluation.clip_sample_every_steps "
                "or enlarge the late window."
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

        embed_payload = dict(
            z_control_a=z_control_a,
            z_control_b=z_control_b,
            z_walls=z_walls,
            baseline_per_t=baseline_per_t,
            walls_ctrl_a_per_t=walls_a_per_t,
            walls_ctrl_b_per_t=walls_b_per_t,
            late_sample_steps=np.asarray(run_outputs["control_a"]["z_late_steps"], dtype=np.int32),
            z_late_sample_steps=np.asarray(run_outputs["control_a"]["z_late_steps"], dtype=np.int32),
            z_control_a_full=run_outputs["control_a"]["z_full"],
            z_control_a_full_steps=np.asarray(run_outputs["control_a"]["full_steps"], dtype=np.int32),
            z_walls_full=run_outputs["walls"]["z_full"],
            z_walls_full_steps=np.asarray(run_outputs["walls"]["full_steps"], dtype=np.int32),
            z_walls_blocks_full=run_outputs["walls"]["z_full_blocks"],
            z_walls_blocks_full_steps=np.asarray(run_outputs["walls"]["full_steps"], dtype=np.int32),
        )
        if run_outputs["control_b"]["z_full"].shape[0] > 0:
            embed_payload["z_control_b_full"] = run_outputs["control_b"]["z_full"]
            embed_payload["z_control_b_full_steps"] = np.asarray(run_outputs["control_b"]["full_steps"], dtype=np.int32)
        _save_npz_atomic(trial_paths["trial_embeddings_npz"], **embed_payload)

    if enable_msc:
        xy_control_a = run_outputs["control_a"]["xy_late"]
        xy_control_b = run_outputs["control_b"]["xy_late"]
        xy_walls = run_outputs["walls"]["xy_late"]
        metric_seed_base = int(getattr(args, "metric_seed", seed_x + 10_000_000))
        k_a = jax.random.fold_in(jax.random.PRNGKey(metric_seed_base), 0)
        k_b = jax.random.fold_in(jax.random.PRNGKey(metric_seed_base), 1)
        k_w = jax.random.fold_in(jax.random.PRNGKey(metric_seed_base), 2)
        msc_loss_control_a, msc_metrics_control_a = metric_eval(k_a, jnp.asarray(xy_control_a))
        msc_loss_control_b, msc_metrics_control_b = metric_eval(k_b, jnp.asarray(xy_control_b))
        msc_loss_walls, msc_metrics_walls = metric_eval(k_w, jnp.asarray(xy_walls))
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
                "msc_score_walls_minus_control_mean": float(float(msc_metrics_walls["score"]) - msc_score_control_mean),
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
                "msc_time_sampling": int(xy_control_a.shape[0]),
            }
        )
        _save_npz_atomic(
            trial_paths["trial_lagrangian_npz"],
            xy_control_a=xy_control_a,
            xy_control_b=xy_control_b,
            xy_walls=xy_walls,
            sample_offsets_steps=np.asarray(run_outputs["control_a"]["xy_late_steps"], dtype=np.int32) - int(late_start),
            xy_late_sample_steps=np.asarray(run_outputs["control_a"]["xy_late_steps"], dtype=np.int32),
            sample_every_steps=np.asarray(int(metric_cfg["sample_every_steps"]), dtype=np.int32),
            trajectory_start_steps=np.asarray(int(late_start), dtype=np.int32),
            trajectory_end_steps=np.asarray(int(late_end), dtype=np.int32),
            trajectory_window_steps=np.asarray(int(late_end - late_start), dtype=np.int32),
            metric_window_size_steps=np.asarray(
                int(metric_cfg["window_size_frames"] * metric_cfg["sample_every_steps"]),
                dtype=np.int32,
            ),
            metric_window_step_steps=np.asarray(
                int(metric_cfg["window_step_frames"] * metric_cfg["sample_every_steps"]),
                dtype=np.int32,
            ),
            metric_tau_steps=np.asarray(int(metric_cfg["tau_steps"]), dtype=np.int32),
        )

    _write_json(trial_paths["trial_row_json"], row)
    return row, metric_info


def _run_generic_state_perturbation_trials(
    *,
    trials: list[dict],
    common_args,
    root_save_dir: Path,
    repo: Path,
    run,
    substrate,
    enable_clip: bool,
    enable_msc: bool,
    total_steps: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
    clip_sample_every_steps: int,
    checkpoint_every_steps: int,
    full_embedding_sample_every_steps: int,
):
    params_list = []
    for trial in trials:
        params = _load_params(trial["args"], repo)
        trial["params"] = params
        trial["late_start"] = late_start
        trial["late_end"] = late_end
        _write_text(
            trial["trial_paths"]["trial_artifact_dir"] / "resolved_config.yaml",
            OmegaConf.to_yaml(trial["cfg"], resolve=True),
        )
        params_list.append(params)

    batched_state_chunk_stepper = _build_batched_state_chunk_stepper(substrate, base_chunk_steps)

    embed_global_batch = None
    if enable_clip:
        foundation_model = str(getattr(common_args, "foundation_model", "clip"))
        fm = foundation_models.create_foundation_model(foundation_model)
        embed_global_batch = _build_generic_batch_embedder(
            substrate=substrate,
            fm=fm,
            clip_img_size=int(getattr(common_args, "clip_img_size", 224)),
        )

    metric_cfg = None
    metric_info = None
    metric_eval = None
    positions_unwrapped = False
    metric_space = _metric_space_from_args(common_args, substrate)
    if enable_msc:
        metric_node = OmegaConf.merge(trials[0]["cfg"].get("substrate", {}), trials[0]["cfg"].get("metric", {}))
        metric_dict = OmegaConf.to_container(metric_node, resolve=True)
        metric_args = SimpleNamespace(**metric_dict)
        metric_args.rollout_steps = int(late_end - late_start)
        if getattr(metric_args, "metric_periodic", None) is None:
            metric_args.metric_periodic = bool(metric_space["periodic"])
        if getattr(metric_args, "metric_domain_y", None) is None:
            metric_args.metric_domain_y = float(metric_space["domain_y"])
        if getattr(metric_args, "metric_domain_x", None) is None:
            metric_args.metric_domain_x = float(metric_space["domain_x"])
        metric_cfg = resolve_metric_config(metric_args)
        if int(metric_cfg["sample_every_steps"]) != int(base_chunk_steps):
            raise ValueError(
                "paper_check generic frustration evaluation expects metric.sample_every_steps to define the base chunk size. "
                f"Got metric.sample_every_steps={int(metric_cfg['sample_every_steps'])}, "
                f"sample_every_steps={int(base_chunk_steps)}."
            )
        unwrap_state_x = bool(getattr(metric_args, "metric_unwrap_state_x", True))
        positions_unwrapped = bool(metric_cfg["periodic"] and unwrap_state_x)
        metric_cfg["positions_unwrapped"] = positions_unwrapped
        metric_info = dict(metric_summary(metric_cfg))
        metric_eval = jax.jit(make_metric_loss_fn(metric_cfg))

        params0 = params_list[0]
        state0 = substrate.init_state(jax.random.PRNGKey(0), params0)
        xy0 = _extract_positions_from_state(state0)
        run.summary["metric_cfg/trajectory_source"] = "state_x"
        run.summary["metric_cfg/tracked_entities"] = int(xy0.shape[0])
        run.summary["paper_check/frustration_protocol"] = "state_perturbation"

    resume = bool(getattr(common_args, "resume", True))
    lanes = []
    for trial in trials:
        params = trial["params"]
        seed_x = int(getattr(trial["args"], "seed_x"))
        seed_x1 = int(getattr(trial["args"], "seed_x1"))
        control_a = {
            "trial": trial,
            "variant": "control_a",
            "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "control_a_checkpoint.npz",
            "run_seed": seed_x,
            "params": params,
            "wall_mode": False,
            "full_embeddings_enabled": bool(enable_clip),
            "block_embeddings_enabled": False,
            "apply_perturbation": False,
            "warmup_steps": int(warmup_steps),
            "args": trial["args"],
        }
        control_b = {
            "trial": trial,
            "variant": "control_b",
            "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "control_b_checkpoint.npz",
            "run_seed": seed_x1,
            "params": params,
            "wall_mode": False,
            "full_embeddings_enabled": bool(getattr(common_args, "log_full_embeddings_for_b", False) and enable_clip),
            "block_embeddings_enabled": False,
            "apply_perturbation": False,
            "warmup_steps": int(warmup_steps),
            "args": trial["args"],
        }
        walls = {
            "trial": trial,
            "variant": "walls",
            "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "walls_checkpoint.npz",
            "run_seed": seed_x,
            "params": params,
            "wall_mode": False,
            "full_embeddings_enabled": bool(enable_clip),
            "block_embeddings_enabled": False,
            "apply_perturbation": True,
            "warmup_steps": int(warmup_steps),
            "args": trial["args"],
        }
        trial["control_a_lane"] = control_a
        trial["control_b_lane"] = control_b
        trial["walls_lane"] = walls
        lanes.extend([control_a, control_b, walls])

    for lane in lanes:
        _load_or_init_generic_lane(lane, resume=resume, substrate=substrate)

    _run_generic_global_lanes(
        lanes=lanes,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        late_start=late_start,
        late_end=late_end,
        base_chunk_steps=base_chunk_steps,
        clip_sample_every_steps=clip_sample_every_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        full_embedding_sample_every_steps=full_embedding_sample_every_steps,
        enable_clip=enable_clip,
        enable_msc=enable_msc,
        batched_state_chunk_stepper=batched_state_chunk_stepper,
        embed_global_batch=embed_global_batch,
        substrate=substrate,
    )

    completed_rows = []
    for trial in trials:
        run_outputs = {
            "control_a": _lane_output(trial["control_a_lane"]),
            "control_b": _lane_output(trial["control_b_lane"]),
            "walls": _lane_output(trial["walls_lane"]),
        }
        if enable_msc and positions_unwrapped:
            for output in run_outputs.values():
                output["xy_late"] = _unwrap_sampled_xy_np(
                    output["xy_late"],
                    domain_y=float(metric_cfg["domain_y"]),
                    domain_x=float(metric_cfg["domain_x"]),
                )
        row, _ = _finalize_trial(
            trial=trial,
            run_outputs=run_outputs,
            root_save_dir=root_save_dir,
            enable_clip=enable_clip,
            enable_msc=enable_msc,
            distance_metric=str(getattr(common_args, "distance_metric", "cosine_mean")),
            metric_cfg=metric_cfg,
            metric_info=metric_info,
            metric_eval=metric_eval,
        )
        row["frustration_protocol"] = "state_perturbation"
        _write_json(trial["trial_paths"]["trial_row_json"], row)
        completed_rows.append(row)
        print(f"Completed trial {trial['trial_idx']:05d}")

    for key, value in {
        "paper_check/n_trials_completed_in_batch": int(len(completed_rows)),
        "paper_check/batch_size": int(len(trials)),
    }.items():
        run.summary[key] = value
    if metric_info is not None:
        run.summary["paper_check/msc_metric_summary"] = str(metric_info)
    return completed_rows


def run_batch(job_config_paths: list[str | Path]) -> int:
    trials = _load_trials(job_config_paths)
    if not trials:
        return 0
    _assert_common_trial_config(trials)

    repo = _repo_root()
    common_args = trials[0]["args"]
    root_save_dir = trials[0]["root_save_dir"]
    root_save_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=str(getattr(common_args, "wandb_project", "asal")),
        mode=str(getattr(common_args, "wandb_mode", "online")),
        config={
            "batch_size": int(len(trials)),
            "trial_indices": [int(trial["trial_idx"]) for trial in trials],
            "save_dir": str(root_save_dir),
        },
    )

    try:
        enable_clip = bool(getattr(common_args, "enable_clip", True))
        enable_msc = bool(getattr(common_args, "enable_msc", True))
        if not enable_clip and not enable_msc:
            raise ValueError("At least one of enable_clip or enable_msc must be true.")

        total_steps = int(getattr(common_args, "total_steps"))
        warmup_steps = int(getattr(common_args, "warmup_steps"))
        late_start, late_end = _resolve_window(common_args)
        base_chunk_steps = int(getattr(common_args, "sample_every_steps"))
        checkpoint_every_steps = int(getattr(common_args, "checkpoint_every_steps", base_chunk_steps))
        full_embedding_sample_every_steps = int(getattr(common_args, "full_embedding_sample_every_steps", base_chunk_steps))
        continuation_full_embedding_sample_every_steps = int(
            getattr(
                common_args,
                "continuation_full_embedding_sample_every_steps",
                full_embedding_sample_every_steps,
            )
        )
        clip_sample_every_steps = int(getattr(common_args, "clip_sample_every_steps", base_chunk_steps))
        run_seed_protocol = str(getattr(common_args, "run_seed_protocol", "legacy"))
        training_horizon_steps = int(getattr(common_args, "training_horizon_steps", total_steps))
        training_reference_step = int(getattr(common_args, "training_reference_step", training_horizon_steps))
        require_training_reference_match = bool(
            getattr(common_args, "require_training_reference_match", False)
        )
        training_reference_only = bool(
            getattr(common_args, "training_reference_only", False)
        )
        log_clip_evolution = bool(getattr(common_args, "log_clip_evolution", False))
        wall_video_enabled = bool(getattr(common_args, "wall_video_enabled", False))
        wall_video_sample_every_steps = int(
            getattr(common_args, "wall_video_sample_every_steps", full_embedding_sample_every_steps)
        )
        wall_video_img_size = int(getattr(common_args, "wall_video_img_size", 256))
        wall_video_fps = float(getattr(common_args, "wall_video_fps", 24.0))
        wall_video_codec = str(getattr(common_args, "wall_video_codec", "libx264"))
        wall_video_keep_frames = bool(getattr(common_args, "wall_video_keep_frames", False))
        _validate_divisibility(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
        )
        if continuation_full_embedding_sample_every_steps < base_chunk_steps:
            raise ValueError(
                "continuation_full_embedding_sample_every_steps must be at least "
                f"metric.sample_every_steps={base_chunk_steps}."
            )
        if continuation_full_embedding_sample_every_steps % base_chunk_steps != 0:
            raise ValueError(
                "continuation_full_embedding_sample_every_steps must be divisible by "
                f"metric.sample_every_steps={base_chunk_steps}."
            )
        if clip_sample_every_steps < 1:
            raise ValueError("evaluation.clip_sample_every_steps must be >= 1.")
        if clip_sample_every_steps % base_chunk_steps != 0:
            raise ValueError(
                "evaluation.clip_sample_every_steps must be divisible by "
                f"metric.sample_every_steps={base_chunk_steps}, got {clip_sample_every_steps}."
            )
        if enable_clip and clip_sample_every_steps > int(late_end - late_start):
            raise ValueError(
                "evaluation.clip_sample_every_steps is larger than the late window; "
                f"got clip_sample_every_steps={clip_sample_every_steps}, "
                f"late_window_steps={int(late_end - late_start)}."
            )
        if run_seed_protocol not in {"legacy", "optimization_metric"}:
            raise ValueError(
                "evaluation.run_seed_protocol must be one of ['legacy', 'optimization_metric'], "
                f"got {run_seed_protocol!r}."
            )
        if run_seed_protocol == "optimization_metric":
            if training_reference_step != training_horizon_steps:
                raise ValueError(
                    "The exact optimizer-native preflight currently requires "
                    "training_reference_step == training_horizon_steps."
                )
            if training_horizon_steps > total_steps:
                raise ValueError(
                    f"training_horizon_steps={training_horizon_steps} exceeds total_steps={total_steps}."
                )
        if wall_video_enabled:
            if wall_video_sample_every_steps < base_chunk_steps:
                raise ValueError(
                    "wall_video_sample_every_steps must be at least metric.sample_every_steps."
                )
            if wall_video_sample_every_steps % base_chunk_steps != 0:
                raise ValueError(
                    "wall_video_sample_every_steps must be divisible by "
                    f"metric.sample_every_steps={base_chunk_steps}."
                )
            if wall_video_img_size < 16 or wall_video_fps <= 0:
                raise ValueError("Wall video requires img_size >= 16 and fps > 0.")

        substrate = _create_substrate(
            common_args,
            enable_msc=bool(enable_msc or require_training_reference_match),
        )
        if str(common_args.substrate) != "lenia_flow":
            _run_generic_state_perturbation_trials(
                trials=trials,
                common_args=common_args,
                root_save_dir=root_save_dir,
                repo=repo,
                run=run,
                substrate=substrate,
                enable_clip=enable_clip,
                enable_msc=enable_msc,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                late_start=late_start,
                late_end=late_end,
                base_chunk_steps=base_chunk_steps,
                clip_sample_every_steps=clip_sample_every_steps,
                checkpoint_every_steps=checkpoint_every_steps,
                full_embedding_sample_every_steps=full_embedding_sample_every_steps,
            )
            return 0

        split_n = int(getattr(common_args, "grid_split"))
        grid_size = int(getattr(common_args, "grid_size"))
        if split_n < 1:
            raise ValueError(f"grid_split must be >= 1, got {split_n}.")
        pad = int(getattr(common_args, "wall_pad", int(common_args.dd)))
        block_size = (grid_size + split_n - 1) // split_n
        if (block_size + 2 * pad) % 2 != 0:
            block_size += 1
        padded_grid_size = int(block_size * split_n)
        partition_padding = int(padded_grid_size - grid_size)
        wall_global_pad_before = int(partition_padding // 2)
        wall_global_pad_after = int(partition_padding - wall_global_pad_before)
        block_sim_size = block_size + 2 * pad
        valid_mask = _block_valid_mask(
            grid_size=grid_size,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
            global_crop_start=wall_global_pad_before,
        )

        params_list = []
        for trial in trials:
            params = _load_params(trial["args"], repo)
            init_params = _load_init_params(trial["args"], repo)
            trial["params"] = params
            trial["init_params"] = init_params
            trial["late_start"] = late_start
            trial["late_end"] = late_end
            _write_text(trial["trial_paths"]["trial_artifact_dir"] / "resolved_config.yaml", OmegaConf.to_yaml(trial["cfg"], resolve=True))
            params_list.append(params)

            for variant in ("control_a", "control_b"):
                raw_reference_params = getattr(
                    trial["args"],
                    f"{variant}_reference_params_path",
                    None,
                )
                if raw_reference_params is None:
                    if require_training_reference_match:
                        raise ValueError(
                            f"Missing required source.{variant}_reference_params_path for "
                            f"trial_idx={trial['trial_idx']}."
                        )
                    continue
                reference_params_path = Path(str(raw_reference_params))
                if not reference_params_path.is_absolute():
                    reference_params_path = repo / reference_params_path
                reference_params = np.asarray(np.load(reference_params_path), dtype=np.float32)
                candidate_params = np.asarray(jax.device_get(params), dtype=np.float32)
                if not np.array_equal(candidate_params, reference_params):
                    raise RuntimeError(
                        f"Candidate params do not exactly match {variant} C1 reference params: "
                        f"{reference_params_path}."
                    )

        # Cache-only bootstrap reuse skips init_state, whose seed_state side effect creates RT.
        _ = substrate.seed_state(jax.random.PRNGKey(0), params_list[0])

        block_kwargs = util.flow_lenia_kwargs_from_args(common_args)
        block_kwargs["grid_size"] = block_sim_size
        block_substrate = substrates.FlattenSubstrateParameters(
            substrates.create_substrate("lenia_flow", **block_kwargs)
        )

        batched_state_chunk_stepper = _build_batched_state_chunk_stepper(substrate, base_chunk_steps)
        batched_block_chunk_stepper = _build_batched_block_chunk_stepper(
            block_substrate,
            n_blocks=split_n * split_n,
            chunk_steps=base_chunk_steps,
            valid_mask=jnp.asarray(valid_mask),
        )

        clip_img_size = int(getattr(common_args, "clip_img_size", 224))
        distance_metric = str(getattr(common_args, "distance_metric", "cosine_mean"))
        log_full_embeddings_for_b = bool(getattr(common_args, "log_full_embeddings_for_b", False))

        embed_global_batch = None
        embed_blocks_from_block_state_batch = None
        embed_blocks_from_global_state_batch = None
        embed_concat_from_block_state_batch = None
        if enable_clip:
            foundation_model = str(getattr(common_args, "foundation_model", "clip"))
            fm = foundation_models.create_foundation_model(foundation_model)
            (
                embed_global_batch,
                embed_blocks_from_block_state_batch,
                embed_blocks_from_global_state_batch,
                embed_concat_from_block_state_batch,
            ) = _build_batch_embedders(
                substrate=substrate,
                fm=fm,
                clip_img_size=clip_img_size,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                global_size=grid_size,
                global_crop_start=wall_global_pad_before,
            )

        render_global_video_batch = None
        render_blocks_video_batch = None
        if wall_video_enabled:
            render_global_video_batch, render_blocks_video_batch = _build_wall_video_renderers(
                substrate=substrate,
                img_size=wall_video_img_size,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                global_size=grid_size,
                global_crop_start=wall_global_pad_before,
            )

        metric_cfg = None
        metric_info = None
        metric_eval = None
        batched_lagrangian_chunk_stepper = None
        lag_n_particles = 0
        lag_init_mode = "mass"
        lag_channel_mode = "resample"
        if enable_msc:
            metric_node = OmegaConf.merge(trials[0]["cfg"].get("substrate", {}), trials[0]["cfg"].get("metric", {}))
            metric_dict = OmegaConf.to_container(metric_node, resolve=True)
            metric_args = SimpleNamespace(**metric_dict)
            metric_args.rollout_steps = int(late_end - late_start)
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
            metric_info = dict(metric_summary(metric_cfg))
            metric_eval = jax.jit(make_metric_loss_fn(metric_cfg))
            if int(metric_cfg["sample_every_steps"]) != int(base_chunk_steps):
                raise ValueError(
                    "paper_check frustration batch evaluation expects metric.sample_every_steps to define the base chunk size. "
                    f"Got metric.sample_every_steps={int(metric_cfg['sample_every_steps'])}, "
                    f"sample_every_steps={int(base_chunk_steps)}."
                )
            lag_n_particles = int(getattr(metric_args, "metric_lagrangian_n_particles", 256))
            lag_init_mode = str(getattr(metric_args, "metric_lagrangian_init_mode", "mass"))
            lag_flow_channel = int(getattr(metric_args, "metric_lagrangian_flow_channel", -1))
            lag_flow_reduce = str(getattr(metric_args, "metric_lagrangian_flow_reduce", "mass_weighted"))
            lag_channel_mode = str(getattr(metric_args, "metric_lagrangian_channel_mode", "mix"))
            lag_noise_model = str(getattr(metric_args, "metric_lagrangian_noise_model", "none"))
            lag_diffusion_scale = float(getattr(metric_args, "metric_lagrangian_diffusion_scale", 1.0))
            batched_lagrangian_chunk_stepper = _build_batched_lagrangian_chunk_stepper(
                substrate,
                chunk_steps=base_chunk_steps,
                lag_flow_channel=lag_flow_channel,
                lag_flow_reduce=lag_flow_reduce,
                lag_channel_mode=lag_channel_mode,
                lag_noise_model=lag_noise_model,
                lag_diffusion_scale=lag_diffusion_scale,
            )

        bootstrap_outputs: dict[int, dict[str, Any]] = {}
        if run_seed_protocol == "optimization_metric":
            bootstrap_root_raw = getattr(common_args, "bootstrap_cache_root", None)
            bootstrap_root = (
                root_save_dir
                if bootstrap_root_raw in (None, "")
                else (
                    Path(str(bootstrap_root_raw))
                    if Path(str(bootstrap_root_raw)).is_absolute()
                    else repo / Path(str(bootstrap_root_raw))
                )
            )
            bootstrap_outputs = _optimizer_native_bootstrap(
                trials=trials,
                substrate=substrate,
                root_save_dir=bootstrap_root,
                training_horizon_steps=training_horizon_steps,
                base_chunk_steps=base_chunk_steps,
                full_embedding_sample_every_steps=full_embedding_sample_every_steps,
                log_clip_evolution=log_clip_evolution,
                enable_clip=enable_clip,
                embed_global_batch=embed_global_batch,
            )

        block_init_key = jax.random.PRNGKey(0)
        block_templates = jax.jit(
            jax.vmap(
                lambda params: block_substrate.init_state(block_init_key, params)
            )
        )(jnp.stack(params_list, axis=0))
        for trial_idx, trial in enumerate(trials):
            trial["block_template"] = jax.tree_util.tree_map(
                lambda value, idx=trial_idx: value[idx],
                block_templates,
            )

        resume = bool(getattr(common_args, "resume", True))
        control_lanes = []
        walls_lanes = []
        for trial in trials:
            params = trial["params"]
            init_params = trial["init_params"]
            bootstrap = bootstrap_outputs.get(int(trial["trial_idx"]))
            if run_seed_protocol == "optimization_metric" and bootstrap is None:
                raise RuntimeError(
                    f"Missing optimizer-native bootstrap for trial_idx={trial['trial_idx']}."
                )
            seed_x = int(getattr(trial["args"], "seed_x"))
            seed_x1 = int(getattr(trial["args"], "seed_x1"))
            lane_common = {
                "init_params": init_params,
                "run_seed_protocol": run_seed_protocol,
                "log_clip_evolution": log_clip_evolution,
                "total_steps": total_steps,
                "training_horizon_steps": training_horizon_steps,
                "base_chunk_steps": base_chunk_steps,
                "training_reference_step": training_reference_step,
                "require_training_reference_match": require_training_reference_match,
            }

            control_a = {
                "trial": trial,
                "variant": "control_a",
                "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "control_a_checkpoint.npz",
                "run_seed": seed_x,
                "params": params,
                "wall_mode": False,
                "full_embeddings_enabled": bool(enable_clip),
                "block_embeddings_enabled": False,
                "substrate": substrate,
                "reference_apf_dir": getattr(
                    trial["args"], "control_a_reference_apf_dir", None
                ),
                "bootstrap_terminal_state": (
                    None if bootstrap is None else bootstrap["control_a_state"]
                ),
                "bootstrap_full_steps": (
                    [] if bootstrap is None else bootstrap["full_steps"]
                ),
                "bootstrap_z_full": (
                    [] if bootstrap is None else bootstrap["z_full"]
                ),
                **lane_common,
            }
            control_b = {
                "trial": trial,
                "variant": "control_b",
                "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "control_b_checkpoint.npz",
                "run_seed": seed_x1,
                "params": params,
                "wall_mode": False,
                "full_embeddings_enabled": bool(log_full_embeddings_for_b and enable_clip),
                "block_embeddings_enabled": False,
                "substrate": substrate,
                "reference_apf_dir": getattr(
                    trial["args"], "control_b_reference_apf_dir", None
                ),
                "bootstrap_terminal_state": (
                    None if bootstrap is None else bootstrap["control_b_state"]
                ),
                "bootstrap_full_steps": [],
                "bootstrap_z_full": [],
                **lane_common,
            }
            walls = {
                "trial": trial,
                "variant": "walls",
                "checkpoint_path": trial["trial_paths"]["trial_artifact_dir"] / "walls_checkpoint.npz",
                "run_seed": seed_x,
                "params": params,
                "wall_mode": True,
                "full_embeddings_enabled": bool(enable_clip),
                "block_embeddings_enabled": bool(enable_clip),
                "substrate": substrate,
                "initial_global_state": (
                    bootstrap["initial_state"]
                    if bootstrap is not None
                    else _flow_initial_global_state(
                        substrate=substrate,
                        params=params,
                        init_params=init_params,
                        run_seed=seed_x,
                        run_seed_protocol=run_seed_protocol,
                        log_clip_evolution=log_clip_evolution,
                    )
                ),
                "bootstrap_initial_state": (
                    None if bootstrap is None else bootstrap["initial_state"]
                ),
                "training_reference_step": None,
                "require_training_reference_match": False,
                "wall_video_enabled": wall_video_enabled,
                "video_frame_dir": (
                    trial["trial_paths"]["trial_artifact_dir"] / "walls_video_frames"
                ),
                "video_path": trial["trial_paths"]["trial_artifact_dir"] / "walls.mp4",
                **lane_common,
            }
            walls["training_reference_step"] = None
            walls["require_training_reference_match"] = False
            trial["control_a_lane"] = control_a
            trial["control_b_lane"] = control_b
            trial["walls_lane"] = walls
            control_lanes.extend([control_a, control_b])
            walls_lanes.append(walls)

        for lane in walls_lanes:
            def _merge_fn(
                initial_state,
                blocks_state,
                *,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                pad_before=wall_global_pad_before,
                pad_after=wall_global_pad_after,
                grid_size=grid_size,
            ):
                padded_initial = _pad_flow_spatial_state(
                    initial_state,
                    pad_before=pad_before,
                    pad_after=pad_after,
                )
                merged_padded = _merge_blocks_into_global_state(
                    padded_initial,
                    blocks_state,
                    split_n=split_n,
                    block_size=block_size,
                    pad=pad,
                )
                merged = dict(initial_state)
                for key, value in merged_padded.items():
                    arr = jnp.asarray(value)
                    if arr.ndim >= 2 and tuple(arr.shape[:2]) == (
                        grid_size + pad_before + pad_after,
                        grid_size + pad_before + pad_after,
                    ):
                        merged[key] = arr[
                            pad_before:pad_before + grid_size,
                            pad_before:pad_before + grid_size,
                        ]
                    elif key in {"t", "mass_cycle_start"}:
                        merged[key] = value
                merged["mass_cycle_start"] = jnp.sum(merged["A"])
                return merged

            lane["merge_blocks"] = _merge_fn

        for lane in control_lanes + walls_lanes:
            _load_or_init_lane(
                lane,
                resume=resume,
                substrate=substrate,
                block_template=lane["trial"]["block_template"],
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                wall_global_pad_before=wall_global_pad_before,
                wall_global_pad_after=wall_global_pad_after,
            )

        for lane in control_lanes:
            if (
                lane.get("training_reference_step") is not None
                and int(lane["state"]["current_step"]) == int(lane["training_reference_step"])
            ):
                _assert_training_reference(
                    lane,
                    lane["state"]["global_state"],
                    step=int(lane["training_reference_step"]),
                )
        for lane in walls_lanes:
            initial_proof_lane = {
                **lane,
                "variant": "walls_initial",
                "training_reference_step": 0,
                "reference_apf_dir": getattr(
                    lane["trial"]["args"],
                    "control_a_reference_apf_dir",
                    None,
                ),
                "require_training_reference_match": require_training_reference_match,
            }
            _assert_training_reference(
                initial_proof_lane,
                lane["initial_global_state"],
                step=0,
            )

        _run_control_lanes(
            lanes=control_lanes,
            total_steps=total_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            clip_sample_every_steps=clip_sample_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=continuation_full_embedding_sample_every_steps,
            enable_clip=enable_clip,
            enable_msc=enable_msc,
            lag_n_particles=lag_n_particles,
            lag_init_mode=lag_init_mode,
            lag_channel_mode=lag_channel_mode,
            batched_state_chunk_stepper=batched_state_chunk_stepper,
            batched_lagrangian_chunk_stepper=batched_lagrangian_chunk_stepper,
            embed_global_batch=embed_global_batch,
        )
        if training_reference_only:
            proof_rows = []
            for lane in control_lanes:
                proof_path = (
                    lane["trial"]["trial_paths"]["trial_artifact_dir"]
                    / f"{lane['variant']}_training_reference_check.json"
                )
                if not proof_path.exists():
                    raise RuntimeError(
                        f"Training-reference preflight produced no proof for {lane['variant']}: "
                        f"{proof_path}."
                    )
                proof = json.loads(proof_path.read_text())
                if str(proof.get("status")) != "exact":
                    raise RuntimeError(
                        f"Training-reference preflight is not exact: {proof_path}."
                    )
                proof_rows.append(proof)
            _write_json(
                root_save_dir / "training_reference_preflight_summary.json",
                {
                    "status": "exact",
                    "n_control_lanes": len(proof_rows),
                    "training_reference_step": training_reference_step,
                    "proofs": proof_rows,
                },
            )
            print(
                "[paper_check/frustration/batch] training-reference preflight exact "
                f"for {len(proof_rows)} control lanes"
            )
            return 0
        _run_walls_lanes(
            lanes=walls_lanes,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            clip_sample_every_steps=clip_sample_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=continuation_full_embedding_sample_every_steps,
            enable_clip=enable_clip,
            enable_msc=enable_msc,
            lag_n_particles=lag_n_particles,
            lag_init_mode=lag_init_mode,
            lag_channel_mode=lag_channel_mode,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
            batched_state_chunk_stepper=batched_state_chunk_stepper,
            batched_block_chunk_stepper=batched_block_chunk_stepper,
            batched_lagrangian_chunk_stepper=batched_lagrangian_chunk_stepper,
            embed_global_batch=embed_global_batch,
            embed_blocks_from_block_state_batch=embed_blocks_from_block_state_batch,
            embed_blocks_from_global_state_batch=embed_blocks_from_global_state_batch,
            embed_concat_from_block_state_batch=embed_concat_from_block_state_batch,
            render_global_video_batch=render_global_video_batch,
            render_blocks_video_batch=render_blocks_video_batch,
            wall_video_sample_every_steps=wall_video_sample_every_steps,
        )

        completed_rows = []
        for trial in trials:
            run_outputs = {
                "control_a": _lane_output(trial["control_a_lane"]),
                "control_b": _lane_output(trial["control_b_lane"]),
                "walls": _lane_output(trial["walls_lane"]),
            }
            row, _ = _finalize_trial(
                trial=trial,
                run_outputs=run_outputs,
                root_save_dir=root_save_dir,
                enable_clip=enable_clip,
                enable_msc=enable_msc,
                distance_metric=distance_metric,
                metric_cfg=metric_cfg,
                metric_info=metric_info,
                metric_eval=metric_eval,
            )
            row.update(
                {
                    "run_seed_protocol": run_seed_protocol,
                    "training_horizon_steps": training_horizon_steps,
                    "training_reference_step": training_reference_step,
                    "grid_split": split_n,
                    "wall_partition_padded_grid_size": padded_grid_size,
                    "wall_partition_padding_cells": partition_padding,
                    "wall_partition_padding_before": wall_global_pad_before,
                    "wall_partition_padding_after": wall_global_pad_after,
                }
            )
            row.update(
                _encode_wall_video(
                    trial["walls_lane"],
                    fps=wall_video_fps,
                    codec=wall_video_codec,
                    keep_frames=wall_video_keep_frames,
                )
            )
            _write_json(trial["trial_paths"]["trial_row_json"], row)
            completed_rows.append(row)
            print(f"Completed trial {trial['trial_idx']:05d}")

        for key, value in {
            "paper_check/n_trials_completed_in_batch": int(len(completed_rows)),
            "paper_check/batch_size": int(len(trials)),
        }.items():
            run.summary[key] = value
        if metric_info is not None:
            run.summary["paper_check/msc_metric_summary"] = str(metric_info)
    finally:
        run.finish()

    return 0


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/paper_check_frustration_batch_eval.py <resolved_job_config_1.yaml> [<resolved_job_config_2.yaml> ...]")
    return run_batch(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
