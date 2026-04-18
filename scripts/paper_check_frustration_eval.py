from __future__ import annotations

import csv
import json
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
    _assemble_blocks_jax,
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


_patch_wandb_pandas_check()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/paper_check_frustration_eval.py <resolved_job_config.yaml>")
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
        cfg.get("job", {}),
    )
    return cfg, flat


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_trial_paths(root_dir: Path, trial_idx: int) -> dict[str, Path]:
    trial_name = f"trial_{int(trial_idx):05d}"
    trial_data_dir = root_dir / "trial_data"
    trial_artifact_dir = root_dir / "trial_artifacts" / trial_name
    trial_data_dir.mkdir(parents=True, exist_ok=True)
    trial_artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "trial_name": Path(trial_name),
        "trial_data_dir": trial_data_dir,
        "trial_artifact_dir": trial_artifact_dir,
        "trial_row_json": trial_data_dir / f"{trial_name}.json",
        "trial_embeddings_npz": trial_data_dir / f"{trial_name}_embeddings.npz",
        "trial_lagrangian_npz": trial_data_dir / f"{trial_name}_lagrangian.npz",
    }


def _state_to_np_payload(prefix: str, state: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    if not isinstance(state, dict):
        return {f"{prefix}__array": np.asarray(jax.device_get(state))}
    payload = {}
    for key, value in state.items():
        payload[f"{prefix}{key}"] = np.asarray(jax.device_get(value))
    return payload


def _state_from_npz(prefix: str, data) -> dict[str, jax.Array]:
    array_key = f"{prefix}__array"
    if array_key in data.files:
        return jnp.asarray(np.asarray(data[array_key]))
    out = {}
    for key in data.files:
        if key.startswith(prefix):
            out[key[len(prefix):]] = jnp.asarray(np.asarray(data[key]))
    return out


def _stack_or_empty(seq: list[np.ndarray], *, dtype=np.float32, trailing_shape: tuple[int, ...] = ()) -> np.ndarray:
    if not seq:
        return np.zeros((0,) + tuple(trailing_shape), dtype=dtype)
    return np.stack([np.asarray(x, dtype=dtype) for x in seq], axis=0)


def _load_run_checkpoint(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        ckpt = {
            "mode": str(np.asarray(data["mode"]).item()),
            "current_step": int(np.asarray(data["current_step"]).item()),
            "rng": jnp.asarray(np.asarray(data["rng"], dtype=np.uint32)),
            "full_steps": np.asarray(data["full_steps"], dtype=np.int32).tolist() if "full_steps" in data.files else [],
            "late_steps": np.asarray(data["late_steps"], dtype=np.int32).tolist() if "late_steps" in data.files else [],
            "z_late_steps": np.asarray(data["z_late_steps"], dtype=np.int32).tolist() if "z_late_steps" in data.files else [],
            "xy_late_steps": np.asarray(data["xy_late_steps"], dtype=np.int32).tolist() if "xy_late_steps" in data.files else [],
            "z_full": [arr for arr in np.asarray(data["z_full"], dtype=np.float32)] if "z_full" in data.files else [],
            "z_full_blocks": [arr for arr in np.asarray(data["z_full_blocks"], dtype=np.float32)] if "z_full_blocks" in data.files else [],
            "z_late": [arr for arr in np.asarray(data["z_late"], dtype=np.float32)] if "z_late" in data.files else [],
            "xy_late": [arr for arr in np.asarray(data["xy_late"], dtype=np.float32)] if "xy_late" in data.files else [],
        }
        if not ckpt["z_late_steps"] and ckpt["z_late"]:
            ckpt["z_late_steps"] = list(ckpt["late_steps"])
        if not ckpt["xy_late_steps"] and ckpt["xy_late"]:
            ckpt["xy_late_steps"] = list(ckpt["late_steps"])
        if ckpt["mode"] == "block":
            ckpt["block_state"] = _state_from_npz("block__", data)
        elif ckpt["mode"] == "global":
            ckpt["global_state"] = _state_from_npz("global__", data)
        elif ckpt["mode"] == "lag":
            lag_state = _state_from_npz("lag_state__", data)
            lag_pts = jnp.asarray(np.asarray(data["lag_pts"], dtype=np.float32))
            lag_ch = jnp.asarray(np.asarray(data["lag_ch"], dtype=np.int32))
            ckpt["lag_carry"] = (lag_state, lag_pts, lag_ch)
        else:
            raise ValueError(f"Unknown checkpoint mode={ckpt['mode']!r}.")
        return ckpt


def _save_run_checkpoint(
    path: Path,
    *,
    mode: str,
    current_step: int,
    rng: jax.Array,
    full_steps: list[int],
    late_steps: list[int],
    z_full: list[np.ndarray],
    z_full_blocks: list[np.ndarray],
    z_late: list[np.ndarray],
    xy_late: list[np.ndarray],
    z_late_steps: list[int] | None = None,
    xy_late_steps: list[int] | None = None,
    block_state: dict[str, jax.Array] | None = None,
    global_state: dict[str, jax.Array] | None = None,
    lag_carry=None,
) -> None:
    if z_late_steps is None:
        z_late_steps = late_steps
    if xy_late_steps is None:
        xy_late_steps = late_steps
    payload = {
        "mode": np.asarray(str(mode)),
        "current_step": np.asarray(int(current_step), dtype=np.int32),
        "rng": np.asarray(jax.device_get(rng), dtype=np.uint32),
        "full_steps": np.asarray(full_steps, dtype=np.int32),
        "late_steps": np.asarray(late_steps, dtype=np.int32),
        "z_late_steps": np.asarray(z_late_steps, dtype=np.int32),
        "xy_late_steps": np.asarray(xy_late_steps, dtype=np.int32),
        "z_full": _stack_or_empty(z_full, dtype=np.float32),
        "z_full_blocks": _stack_or_empty(z_full_blocks, dtype=np.float32),
        "z_late": _stack_or_empty(z_late, dtype=np.float32),
        "xy_late": _stack_or_empty(xy_late, dtype=np.float32),
    }
    if mode == "block":
        if block_state is None:
            raise ValueError("block_state must be provided when mode='block'.")
        payload.update(_state_to_np_payload("block__", block_state))
    elif mode == "global":
        if global_state is None:
            raise ValueError("global_state must be provided when mode='global'.")
        payload.update(_state_to_np_payload("global__", global_state))
    elif mode == "lag":
        if lag_carry is None:
            raise ValueError("lag_carry must be provided when mode='lag'.")
        lag_state, lag_pts, lag_ch = lag_carry
        payload.update(_state_to_np_payload("lag_state__", lag_state))
        payload["lag_pts"] = np.asarray(jax.device_get(lag_pts), dtype=np.float32)
        payload["lag_ch"] = np.asarray(jax.device_get(lag_ch), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported checkpoint mode={mode!r}.")
    _save_npz_atomic(path, **payload)


def _extract_render_state(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    out = {"A": state["A"], "P": state["P"]}
    if "Food" in state:
        out["Food"] = state["Food"]
    return out


def _trim_block_render_state(block_state: dict[str, jax.Array], *, pad: int, block_size: int):
    out = {
        "A": block_state["A"][:, pad:pad + block_size, pad:pad + block_size, :],
        "P": block_state["P"][:, pad:pad + block_size, pad:pad + block_size, :],
    }
    if "Food" in block_state:
        out["Food"] = block_state["Food"][:, pad:pad + block_size, pad:pad + block_size]
    return out


def _split_global_render_blocks(state: dict[str, jax.Array], *, split_n: int, block_size: int):
    A = state["A"]
    P = state["P"]
    C = int(A.shape[-1])
    K = int(P.shape[-1])
    A_blocks = A.reshape((split_n, block_size, split_n, block_size, C))
    A_blocks = jnp.transpose(A_blocks, (0, 2, 1, 3, 4)).reshape((split_n * split_n, block_size, block_size, C))
    P_blocks = P.reshape((split_n, block_size, split_n, block_size, K))
    P_blocks = jnp.transpose(P_blocks, (0, 2, 1, 3, 4)).reshape((split_n * split_n, block_size, block_size, K))
    out = {"A": A_blocks, "P": P_blocks}
    if "Food" in state:
        Food = state["Food"]
        Food_blocks = Food.reshape((split_n, block_size, split_n, block_size))
        Food_blocks = jnp.transpose(Food_blocks, (0, 2, 1, 3)).reshape((split_n * split_n, block_size, block_size))
        out["Food"] = Food_blocks
    return out


def _build_image_embedder(fm):
    image_size = int(getattr(fm, "image_size", 224))
    img_mean = jnp.asarray(getattr(fm, "img_mean", np.array([0.48145466, 0.4578275, 0.40821073])), dtype=jnp.float32)
    img_std = jnp.asarray(getattr(fm, "img_std", np.array([0.26862954, 0.26130258, 0.27577711])), dtype=jnp.float32)
    clip_model = getattr(fm, "clip_model", None)
    siglip_model = getattr(fm, "siglip_model", None)

    def _resize(imgs):
        imgs = jnp.asarray(imgs, dtype=jnp.float32)
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        h = int(imgs.shape[1])
        w = int(imgs.shape[2])
        c = int(imgs.shape[3])
        if h != image_size or w != image_size:
            imgs = jax.image.resize(imgs, (int(imgs.shape[0]), image_size, image_size, c), method="bilinear")
        return imgs

    def embed_batch(imgs):
        imgs = _resize(imgs)
        pixels = jnp.transpose((imgs - img_mean) / img_std, (0, 3, 1, 2))
        if clip_model is not None:
            z = clip_model.get_image_features(pixels)
        elif siglip_model is not None:
            z = siglip_model.get_image_features(pixel_values=pixels)
        else:
            z = jax.vmap(fm.embed_img)(imgs)
            return z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    return jax.jit(embed_batch)


def _build_render_embedders(
    *,
    substrate,
    params,
    fm,
    clip_img_size: int,
    split_n: int,
    block_size: int,
    pad: int,
):
    embed_batch = _build_image_embedder(fm)

    @jax.jit
    def embed_global(state):
        img = substrate.render_state(_extract_render_state(state), params, img_size=clip_img_size)
        return embed_batch(img)[0]

    @jax.jit
    def embed_blocks_from_block_state(block_state):
        blocks = _trim_block_render_state(block_state, pad=pad, block_size=block_size)
        imgs = jax.vmap(lambda st: substrate.render_state(st, params, img_size=clip_img_size))(blocks)
        return embed_batch(imgs)

    @jax.jit
    def embed_blocks_from_global_state(state):
        blocks = _split_global_render_blocks(state, split_n=split_n, block_size=block_size)
        imgs = jax.vmap(lambda st: substrate.render_state(st, params, img_size=clip_img_size))(blocks)
        return embed_batch(imgs)

    @jax.jit
    def embed_concat_from_block_state(block_state):
        A_full, P_full, Food_full = _assemble_blocks_jax(
            block_state,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
        )
        img = substrate.render_state({"A": A_full, "P": P_full, "Food": Food_full}, params, img_size=clip_img_size)
        return embed_batch(img)[0]

    return embed_global, embed_blocks_from_block_state, embed_blocks_from_global_state, embed_concat_from_block_state


def _init_run_checkpoint(
    *,
    wall_mode: bool,
    run_seed: int,
    substrate,
    block_template,
    params,
    split_n: int,
    block_size: int,
    pad: int,
):
    rng = jax.random.PRNGKey(int(run_seed))
    rng, init_key = jax.random.split(rng)
    initial_state = substrate.init_state(init_key, params)
    if wall_mode:
        block_state = _prepare_block_template_state(
            initial_state=initial_state,
            block_template=block_template,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
            C=int(initial_state["A"].shape[-1]),
            k=int(initial_state["P"].shape[-1]),
        )
        return dict(
            mode="block",
            current_step=0,
            rng=rng,
            block_state=block_state,
            full_steps=[],
            late_steps=[],
            z_full=[],
            z_full_blocks=[],
            z_late=[],
            xy_late=[],
        )
    return dict(
        mode="global",
        current_step=0,
        rng=rng,
        global_state=initial_state,
        full_steps=[],
        late_steps=[],
        z_full=[],
        z_full_blocks=[],
        z_late=[],
        xy_late=[],
    )


def _lag_init_keys(run_seed: int):
    key = jax.random.fold_in(jax.random.PRNGKey(int(run_seed)), jnp.uint32(0x4D5343))
    return jax.random.split(key, 2)


def _initial_global_state(substrate, params, run_seed: int):
    rng = jax.random.PRNGKey(int(run_seed))
    _, init_key = jax.random.split(rng)
    return substrate.init_state(init_key, params)


def _run_single_variant(
    *,
    variant: str,
    checkpoint_path: Path,
    resume: bool,
    wall_mode: bool,
    full_embeddings_enabled: bool,
    block_embeddings_enabled: bool,
    run_seed: int,
    substrate,
    block_substrate,
    params,
    total_steps: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    split_n: int,
    block_size: int,
    pad: int,
    checkpoint_every_steps: int,
    base_chunk_steps: int,
    full_embedding_sample_every_steps: int,
    enable_clip: bool,
    enable_msc: bool,
    lagrangian_chunk_stepper,
    lag_n_particles: int,
    lag_init_mode: str,
    lag_channel_mode: str,
    block_template,
    embed_global,
    embed_blocks_from_block_state,
    embed_blocks_from_global_state,
    embed_concat_from_block_state,
):
    if resume:
        state = _load_run_checkpoint(checkpoint_path)
    else:
        state = None
    if state is None:
        state = _init_run_checkpoint(
            wall_mode=wall_mode,
            run_seed=run_seed,
            substrate=substrate,
            block_template=block_template,
            params=params,
            split_n=split_n,
            block_size=block_size,
            pad=pad,
        )

    stepper_get = _build_state_chunk_stepper(substrate)
    block_chunk_stepper = _build_block_warmupper(block_substrate, split_n * split_n, base_chunk_steps)
    checkpoint_stride = int(checkpoint_every_steps // base_chunk_steps)
    chunks_since_save = 0

    def maybe_checkpoint(force: bool = False):
        nonlocal chunks_since_save
        if not force and chunks_since_save < checkpoint_stride:
            return
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
            checkpoint_path,
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
        chunks_since_save = 0

    def append_full_from_global(global_state):
        if not (enable_clip and full_embeddings_enabled):
            return
        z = np.asarray(jax.device_get(embed_global(global_state)), dtype=np.float32)
        state["full_steps"].append(int(state["current_step"]))
        state["z_full"].append(z)
        if block_embeddings_enabled:
            z_blocks = np.asarray(jax.device_get(embed_blocks_from_global_state(global_state)), dtype=np.float32)
            state["z_full_blocks"].append(z_blocks)

    def append_full_from_blocks(block_state):
        if not (enable_clip and full_embeddings_enabled):
            return
        z_concat = np.asarray(jax.device_get(embed_concat_from_block_state(block_state)), dtype=np.float32)
        state["full_steps"].append(int(state["current_step"]))
        state["z_full"].append(z_concat)
        if block_embeddings_enabled:
            z_blocks = np.asarray(jax.device_get(embed_blocks_from_block_state(block_state)), dtype=np.float32)
            state["z_full_blocks"].append(z_blocks)

    def append_late_from_global(global_state):
        state["late_steps"].append(int(state["current_step"]))
        if enable_clip:
            z = np.asarray(jax.device_get(embed_global(global_state)), dtype=np.float32)
            state["z_late"].append(z)

    while int(state["current_step"]) < int(total_steps):
        mode = state["mode"]

        if mode == "block":
            state["rng"], chunk_key = jax.random.split(state["rng"])
            state["block_state"] = block_chunk_stepper(chunk_key, state["block_state"], params)
            state["current_step"] += int(base_chunk_steps)
            if enable_clip and full_embeddings_enabled and state["current_step"] % full_embedding_sample_every_steps == 0:
                append_full_from_blocks(state["block_state"])
            if int(state["current_step"]) == int(warmup_steps):
                merged_state = _merge_blocks_into_global_state(
                    _initial_global_state(substrate, params, run_seed),
                    state["block_state"],
                    split_n=split_n,
                    block_size=block_size,
                    pad=pad,
                )
                state.pop("block_state", None)
                if enable_msc and int(state["current_step"]) == int(late_start):
                    key_pts, key_ch = _lag_init_keys(run_seed)
                    state["lag_carry"] = _init_lagrangian_carry(
                        substrate=substrate,
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

        elif mode == "global":
            state["rng"], chunk_key = jax.random.split(state["rng"])
            step_keys = jax.random.split(chunk_key, base_chunk_steps)
            state["global_state"] = stepper_get(base_chunk_steps)(step_keys, state["global_state"], params)
            state["current_step"] += int(base_chunk_steps)

            if enable_clip and full_embeddings_enabled and state["current_step"] % full_embedding_sample_every_steps == 0:
                append_full_from_global(state["global_state"])

            if enable_msc and int(state["current_step"]) == int(late_start):
                key_pts, key_ch = _lag_init_keys(run_seed)
                state["lag_carry"] = _init_lagrangian_carry(
                    substrate=substrate,
                    state0=state["global_state"],
                    key_pts=key_pts,
                    key_ch=key_ch,
                    lag_n_particles=lag_n_particles,
                    lag_init_mode=lag_init_mode,
                    lag_channel_mode=lag_channel_mode,
                )
                state.pop("global_state", None)
                state["mode"] = "lag"
            elif (not enable_msc) and int(late_start) < int(state["current_step"]) <= int(late_end):
                append_late_from_global(state["global_state"])

        elif mode == "lag":
            state["rng"], chunk_key = jax.random.split(state["rng"])
            state["lag_carry"], xy = lagrangian_chunk_stepper(chunk_key, state["lag_carry"], params)
            state["current_step"] += int(base_chunk_steps)
            global_state = state["lag_carry"][0]

            if enable_clip and full_embeddings_enabled and state["current_step"] % full_embedding_sample_every_steps == 0:
                append_full_from_global(global_state)

            if int(state["current_step"]) <= int(late_end):
                append_late_from_global(global_state)
                state["xy_late"].append(np.asarray(jax.device_get(xy), dtype=np.float32))

            if int(state["current_step"]) == int(late_end) and int(state["current_step"]) < int(total_steps):
                state["global_state"] = global_state
                state.pop("lag_carry", None)
                state["mode"] = "global"

        else:
            raise ValueError(f"Unknown mode={mode!r}.")

        chunks_since_save += 1
        maybe_checkpoint(force=False)

    maybe_checkpoint(force=True)

    result = {
        "full_steps": np.asarray(state["full_steps"], dtype=np.int32),
        "late_steps": np.asarray(state["late_steps"], dtype=np.int32),
        "z_full": _stack_or_empty(state["z_full"], dtype=np.float32),
        "z_full_blocks": _stack_or_empty(state["z_full_blocks"], dtype=np.float32),
        "z_late": _stack_or_empty(state["z_late"], dtype=np.float32),
        "xy_late": _stack_or_empty(state["xy_late"], dtype=np.float32),
    }
    return result


def _validate_divisibility(*, total_steps: int, warmup_steps: int, late_start: int, late_end: int, base_chunk_steps: int, checkpoint_every_steps: int, full_embedding_sample_every_steps: int) -> None:
    values = {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "late_window_start_steps": late_start,
        "late_window_end_steps": late_end,
        "checkpoint_every_steps": checkpoint_every_steps,
        "full_embedding_sample_every_steps": full_embedding_sample_every_steps,
    }
    for name, value in values.items():
        if value % base_chunk_steps != 0:
            raise ValueError(
                f"{name} must be divisible by metric.sample_every_steps={base_chunk_steps}, got {value}."
            )


def main(cfg, args):
    if str(getattr(args, "substrate")) != "lenia_flow":
        if len(sys.argv) < 2:
            raise SystemExit("Usage: python scripts/paper_check_frustration_eval.py <resolved_job_config.yaml>")
        from paper_check_frustration_batch_eval import run_batch

        return run_batch([sys.argv[1]])

    project_root = _repo_root()
    root_save_dir = Path(project_root / str(getattr(args, "save_dir")))
    root_save_dir.mkdir(parents=True, exist_ok=True)
    trial_idx = int(getattr(args, "trial_idx"))
    trial_paths = _make_trial_paths(root_save_dir, trial_idx)
    if trial_paths["trial_row_json"].exists():
        print(f"Trial {trial_idx:05d} already completed: {trial_paths['trial_row_json']}")
        return

    _write_text(trial_paths["trial_artifact_dir"] / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        project=str(getattr(args, "wandb_project", "asal")),
        mode=str(getattr(args, "wandb_mode", "online")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        enable_clip = bool(getattr(args, "enable_clip", True))
        enable_msc = bool(getattr(args, "enable_msc", True))
        if not enable_clip and not enable_msc:
            raise ValueError("At least one of enable_clip or enable_msc must be true.")

        params = _load_params(args, project_root)
        substrate = _create_substrate(args, enable_msc=enable_msc)
        if str(args.substrate) != "lenia_flow":
            raise ValueError("paper_check frustration evaluation currently supports substrate='lenia_flow' only.")

        total_steps = int(getattr(args, "total_steps"))
        warmup_steps = int(getattr(args, "warmup_steps"))
        late_start, late_end = _resolve_window(args)
        base_chunk_steps = int(getattr(args, "sample_every_steps"))
        checkpoint_every_steps = int(getattr(args, "checkpoint_every_steps", base_chunk_steps))
        full_embedding_sample_every_steps = int(getattr(args, "full_embedding_sample_every_steps", base_chunk_steps))
        if checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be >= 1.")
        if full_embedding_sample_every_steps < 1:
            raise ValueError("full_embedding_sample_every_steps must be >= 1.")
        _validate_divisibility(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            late_start=late_start,
            late_end=late_end,
            base_chunk_steps=base_chunk_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            full_embedding_sample_every_steps=full_embedding_sample_every_steps,
        )

        split_n = int(getattr(args, "grid_split"))
        grid_size = int(getattr(args, "grid_size"))
        if grid_size % split_n != 0:
            raise ValueError(f"grid_size {grid_size} must be divisible by grid_split {split_n}.")
        block_size = grid_size // split_n
        pad = int(getattr(args, "wall_pad", int(args.dd)))
        block_sim_size = block_size + 2 * pad

        block_kwargs = util.flow_lenia_kwargs_from_args(args)
        block_kwargs["grid_size"] = block_sim_size
        block_substrate = substrates.FlattenSubstrateParameters(
            substrates.create_substrate("lenia_flow", **block_kwargs)
        )
        block_template = block_substrate.init_state(jax.random.PRNGKey(0), params)

        clip_img_size = int(getattr(args, "clip_img_size", 224))
        distance_metric = str(getattr(args, "distance_metric", "cosine_mean"))
        log_full_embeddings_for_b = bool(getattr(args, "log_full_embeddings_for_b", False))

        fm = None
        embed_global = None
        embed_blocks_from_block_state = None
        embed_blocks_from_global_state = None
        embed_concat_from_block_state = None
        if enable_clip:
            foundation_model = str(getattr(args, "foundation_model", "clip"))
            fm = foundation_models.create_foundation_model(foundation_model)
            (
                embed_global,
                embed_blocks_from_block_state,
                embed_blocks_from_global_state,
                embed_concat_from_block_state,
            ) = _build_render_embedders(
                substrate=substrate,
                params=params,
                fm=fm,
                clip_img_size=clip_img_size,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
            )

        metric_cfg = None
        metric_info = None
        metric_eval = None
        lagrangian_chunk_stepper = None
        lag_n_particles = 0
        lag_init_mode = "mass"
        lag_channel_mode = "resample"
        if enable_msc:
            metric_node = OmegaConf.merge(cfg.get("substrate", {}), cfg.get("metric", {}))
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
                    "paper_check frustration evaluation expects metric.sample_every_steps to define the base chunk size. "
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
            lagrangian_chunk_stepper = _build_lagrangian_chunk_stepper(
                substrate,
                chunk_steps=base_chunk_steps,
                lag_flow_channel=lag_flow_channel,
                lag_flow_reduce=lag_flow_reduce,
                lag_channel_mode=lag_channel_mode,
                lag_noise_model=lag_noise_model,
                lag_diffusion_scale=lag_diffusion_scale,
            )

        resume = bool(getattr(args, "resume", True))
        seed_x = int(getattr(args, "seed_x"))
        seed_x1 = int(getattr(args, "seed_x1"))

        run_outputs = {}
        for variant, run_seed, wall_mode, full_enabled, block_enabled in (
            ("control_a", seed_x, False, enable_clip, False),
            ("control_b", seed_x1, False, log_full_embeddings_for_b and enable_clip, False),
            ("walls", seed_x, True, enable_clip, True),
        ):
            ckpt_path = trial_paths["trial_artifact_dir"] / f"{variant}_checkpoint.npz"
            run_outputs[variant] = _run_single_variant(
                variant=variant,
                checkpoint_path=ckpt_path,
                resume=resume,
                wall_mode=wall_mode,
                full_embeddings_enabled=bool(full_enabled),
                block_embeddings_enabled=bool(block_enabled),
                run_seed=run_seed,
                substrate=substrate,
                block_substrate=block_substrate,
                params=params,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                late_start=late_start,
                late_end=late_end,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                checkpoint_every_steps=checkpoint_every_steps,
                base_chunk_steps=base_chunk_steps,
                full_embedding_sample_every_steps=full_embedding_sample_every_steps,
                enable_clip=enable_clip,
                enable_msc=enable_msc,
                lagrangian_chunk_stepper=lagrangian_chunk_stepper,
                lag_n_particles=lag_n_particles,
                lag_init_mode=lag_init_mode,
                lag_channel_mode=lag_channel_mode,
                block_template=block_template,
                embed_global=embed_global,
                embed_blocks_from_block_state=embed_blocks_from_block_state,
                embed_blocks_from_global_state=embed_blocks_from_global_state,
                embed_concat_from_block_state=embed_concat_from_block_state,
            )

        row = {
            "trial_idx": int(trial_idx),
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
            "warmup_steps": int(warmup_steps),
            "total_steps": int(total_steps),
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

        local_rows = []
        for path in sorted(trial_paths["trial_data_dir"].glob("trial_*.json")):
            with path.open("r") as f:
                local_rows.append(json.load(f))
        local_rows = sorted(local_rows, key=lambda item: int(item["trial_idx"]))
        _save_csv(root_save_dir / "trial_results.csv", local_rows)

        summary = {
            "n_trials_local": int(len(local_rows)),
            "save_dir": str(root_save_dir),
            "resume_enabled": bool(resume),
            "candidate_kind": str(getattr(args, "candidate_kind")),
            "candidate_label": str(getattr(args, "candidate_label")),
        }
        if metric_info is not None:
            summary["msc_metric_summary"] = metric_info
        _write_json(root_save_dir / "summary.json", summary)

        for key, value in row.items():
            if isinstance(value, (int, float)) and value is not None:
                run.summary[f"paper_check/{key}"] = value
        if metric_info is not None:
            run.summary["paper_check/msc_metric_summary"] = str(metric_info)
        print(f"Completed trial {trial_idx:05d}")
    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
