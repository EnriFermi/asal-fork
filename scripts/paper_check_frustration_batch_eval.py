from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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


def _take_tree(tree, indices: list[int]):
    return jax.tree_util.tree_map(lambda x: x[np.asarray(indices, dtype=np.int32)], tree)


def _split_rng_batch(rng_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    split = jax.vmap(lambda key: jax.random.split(key, 2))(rng_batch)
    return split[:, 0], split[:, 1]


def _make_step_keys_batch(chunk_keys: jax.Array, steps: int) -> jax.Array:
    return jax.vmap(lambda key: jax.random.split(key, steps))(chunk_keys)


def _build_batched_state_chunk_stepper(substrate, chunk_steps: int):
    single = _build_state_chunk_stepper(substrate)(chunk_steps)

    @jax.jit
    def step(step_keys_batch, state_batch, params_batch):
        return jax.vmap(single, in_axes=(0, 0, 0))(step_keys_batch, state_batch, params_batch)

    return step


def _build_batched_block_chunk_stepper(block_substrate, *, n_blocks: int, chunk_steps: int):
    single = _build_block_warmupper(block_substrate, n_blocks, chunk_steps)

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
            blocks = _split_global_render_blocks(state, split_n=split_n, block_size=block_size)
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
            return substrate.render_state({"A": A_full, "P": P_full, "Food": Food_full}, pr, img_size=clip_img_size)

        imgs = jax.vmap(render_trial, in_axes=(0, 0))(block_state_batch, params_batch)
        return embed_batch(imgs)

    return (
        embed_global_batch,
        embed_blocks_from_block_state_batch,
        embed_blocks_from_global_state_batch,
        embed_concat_from_block_state_batch,
    )


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
        "z_full": _stack_or_empty(state["z_full"], dtype=np.float32),
        "z_full_blocks": _stack_or_empty(state["z_full_blocks"], dtype=np.float32),
        "z_late": _stack_or_empty(state["z_late"], dtype=np.float32),
        "xy_late": _stack_or_empty(state["xy_late"], dtype=np.float32),
    }


def _stack_or_empty(seq: list[np.ndarray], *, dtype=np.float32, trailing_shape: tuple[int, ...] = ()) -> np.ndarray:
    if not seq:
        return np.zeros((0,) + tuple(trailing_shape), dtype=dtype)
    return np.stack([np.asarray(x, dtype=dtype) for x in seq], axis=0)


def _load_or_init_lane(
    lane: dict,
    *,
    resume: bool,
    substrate,
    block_template,
    split_n: int,
    block_size: int,
    pad: int,
):
    state = _load_run_checkpoint(lane["checkpoint_path"]) if resume else None
    if state is None:
        state = _init_run_checkpoint(
            wall_mode=bool(lane["wall_mode"]),
            run_seed=int(lane["run_seed"]),
            substrate=substrate,
            block_template=block_template,
            params=lane["params"],
            split_n=split_n,
            block_size=block_size,
            pad=pad,
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
            state["z_late"].append(np.asarray(z_host[local_idx], dtype=np.float32))
        if xy_host is not None:
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
            need_late = bool(int(late_start) < next_step <= int(late_end))
            need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
            if enable_msc and need_late:
                xy_host = np.asarray(jax.device_get(_extract_positions_from_state(global_batch)), dtype=np.float32)

            if enable_clip and (need_late or need_full):
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

                if need_late:
                    state["late_steps"].append(next_step)
                    if enable_clip:
                        state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                    if xy_host is not None:
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


def _run_control_lanes(
    *,
    lanes: list[dict],
    total_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
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

    while True:
        active_groups = _group_active_lanes(lanes, total_steps=total_steps)
        if not active_groups:
            break

        for group_indices in active_groups:
            mode = str(lanes[group_indices[0]]["state"]["mode"])
            current_step = int(lanes[group_indices[0]]["state"]["current_step"])
            params_batch = jnp.stack([lanes[idx]["params"] for idx in group_indices], axis=0)
            rng_batch = jnp.stack([lanes[idx]["state"]["rng"] for idx in group_indices], axis=0)
            rng_next, chunk_keys = _split_rng_batch(rng_batch)
            next_step = int(current_step + base_chunk_steps)

            if mode == "global":
                global_batch = _stack_trees([lanes[idx]["state"]["global_state"] for idx in group_indices])
                step_keys_batch = _make_step_keys_batch(chunk_keys, base_chunk_steps)
                global_batch = batched_state_chunk_stepper(step_keys_batch, global_batch, params_batch)
                global_states = _unstack_tree(global_batch)

                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                need_late = bool((not enable_msc) and int(late_start) < next_step <= int(late_end))
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

                need_late = bool(next_step <= int(late_end))
                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
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

                xy_host = np.asarray(jax.device_get(xy_batch), dtype=np.float32) if next_step <= int(late_end) else None

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if z_all_host is not None:
                        state["late_steps"].append(next_step)
                        state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
                    if xy_host is not None:
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


def _run_walls_lanes(
    *,
    lanes: list[dict],
    total_steps: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    base_chunk_steps: int,
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
):
    if not lanes:
        return

    while True:
        active_groups = _group_active_lanes(lanes, total_steps=total_steps)
        if not active_groups:
            break

        for group_indices in active_groups:
            mode = str(lanes[group_indices[0]]["state"]["mode"])
            current_step = int(lanes[group_indices[0]]["state"]["current_step"])
            params_batch = jnp.stack([lanes[idx]["params"] for idx in group_indices], axis=0)
            rng_batch = jnp.stack([lanes[idx]["state"]["rng"] for idx in group_indices], axis=0)
            rng_next, chunk_keys = _split_rng_batch(rng_batch)
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

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
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
                need_late = bool((not enable_msc) and int(late_start) < next_step <= int(late_end))

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

                for local_idx, lane_idx in enumerate(group_indices):
                    lane = lanes[lane_idx]
                    state = lane["state"]
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
                            state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))

            elif mode == "lag":
                lag_batch = _stack_trees([lanes[idx]["state"]["lag_carry"] for idx in group_indices])
                lag_batch, xy_batch = batched_lagrangian_chunk_stepper(chunk_keys, lag_batch, params_batch)
                global_batch = lag_batch[0]
                global_states = _unstack_tree(global_batch)
                lag_carries = _unstack_tree(lag_batch)

                need_late = bool(next_step <= int(late_end))
                need_full = bool(enable_clip and next_step % int(full_embedding_sample_every_steps) == 0)
                if enable_clip and (need_late or need_full):
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

                xy_host = np.asarray(jax.device_get(xy_batch), dtype=np.float32) if need_late else None

                for local_idx, lane_idx in enumerate(group_indices):
                    state = lanes[lane_idx]["state"]
                    state["rng"] = rng_next[local_idx]
                    state["current_step"] = next_step
                    if need_late:
                        state["late_steps"].append(next_step)
                        state["z_late"].append(np.asarray(z_all_host[local_idx], dtype=np.float32))
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
        "distance_metric": distance_metric if enable_clip else None,
        "foundation_model": None if not enable_clip else str(getattr(args, "foundation_model", "clip")),
    }

    if enable_clip:
        z_control_a = run_outputs["control_a"]["z_late"]
        z_control_b = run_outputs["control_b"]["z_late"]
        z_walls = run_outputs["walls"]["z_late"]
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
            late_sample_steps=np.asarray(run_outputs["control_a"]["late_steps"], dtype=np.int32),
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
            sample_offsets_steps=np.asarray(run_outputs["control_a"]["late_steps"], dtype=np.int32) - int(late_start),
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
        _validate_divisibility(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
        )

        substrate = _create_substrate(common_args, enable_msc=enable_msc)
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
                checkpoint_every_steps=checkpoint_every_steps,
                full_embedding_sample_every_steps=full_embedding_sample_every_steps,
            )
            return 0

        split_n = int(getattr(common_args, "grid_split"))
        grid_size = int(getattr(common_args, "grid_size"))
        if grid_size % split_n != 0:
            raise ValueError(f"grid_size {grid_size} must be divisible by grid_split {split_n}.")
        block_size = grid_size // split_n
        pad = int(getattr(common_args, "wall_pad", int(common_args.dd)))
        block_sim_size = block_size + 2 * pad

        params_list = []
        for trial in trials:
            params = _load_params(trial["args"], repo)
            trial["params"] = params
            trial["late_start"] = late_start
            trial["late_end"] = late_end
            _write_text(trial["trial_paths"]["trial_artifact_dir"] / "resolved_config.yaml", OmegaConf.to_yaml(trial["cfg"], resolve=True))
            params_list.append(params)

        block_kwargs = util.flow_lenia_kwargs_from_args(common_args)
        block_kwargs["grid_size"] = block_sim_size
        block_substrate = substrates.FlattenSubstrateParameters(
            substrates.create_substrate("lenia_flow", **block_kwargs)
        )
        block_template = block_substrate.init_state(jax.random.PRNGKey(0), params_list[0])

        batched_state_chunk_stepper = _build_batched_state_chunk_stepper(substrate, base_chunk_steps)
        batched_block_chunk_stepper = _build_batched_block_chunk_stepper(
            block_substrate,
            n_blocks=split_n * split_n,
            chunk_steps=base_chunk_steps,
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

        resume = bool(getattr(common_args, "resume", True))
        control_lanes = []
        walls_lanes = []
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
                "substrate": substrate,
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
                "initial_global_state": substrate.init_state(jax.random.split(jax.random.PRNGKey(seed_x), 2)[1], params),
            }
            trial["control_a_lane"] = control_a
            trial["control_b_lane"] = control_b
            trial["walls_lane"] = walls
            control_lanes.extend([control_a, control_b])
            walls_lanes.append(walls)

        for lane in walls_lanes:
            def _merge_fn(initial_state, blocks_state, *, split_n=split_n, block_size=block_size, pad=pad):
                return _merge_blocks_into_global_state(
                    initial_state,
                    blocks_state,
                    split_n=split_n,
                    block_size=block_size,
                    pad=pad,
                )

            lane["merge_blocks"] = _merge_fn

        for lane in control_lanes + walls_lanes:
            _load_or_init_lane(
                lane,
                resume=resume,
                substrate=substrate,
                block_template=block_template,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
            )

        _run_control_lanes(
            lanes=control_lanes,
            total_steps=total_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
            enable_clip=enable_clip,
            enable_msc=enable_msc,
            lag_n_particles=lag_n_particles,
            lag_init_mode=lag_init_mode,
            lag_channel_mode=lag_channel_mode,
            batched_state_chunk_stepper=batched_state_chunk_stepper,
            batched_lagrangian_chunk_stepper=batched_lagrangian_chunk_stepper,
            embed_global_batch=embed_global_batch,
        )
        _run_walls_lanes(
            lanes=walls_lanes,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
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
