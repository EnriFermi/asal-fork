#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf


PROTOCOL_VERSION = "flowlenia-c5-full-length-deltah-opt-one-v2-shadow-tracker"
DEFAULT_TRIAL_DIR = (
    _REPO_ROOT
    / "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "frustration_simulation/trial_artifacts/trial_00024"
)
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "flow_lenia/c5_full_length_delta_h_opt006_v2_shadow_tracker"
)
_APF_RE = re.compile(r"^P_steps_(\d+)_(\d+)__.*\.npz$")
_BRANCHES = (
    ("control_a", 0, "Control A, no walls"),
    ("control_b", 1, "Control B, no walls"),
    ("walls", 2, "Walls"),
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _REPO_ROOT / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(path)


def _merge_json(path: Path, update: dict[str, Any]) -> None:
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update(update)
    _write_json(path, payload)


def _save_npz(path: Path, *, compressed: bool = False, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    saver = np.savez_compressed if compressed else np.savez
    with temporary.open("wb") as handle:
        saver(handle, **payload)
    temporary.replace(path)


def _state_to_payload(prefix: str, state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{key}": np.asarray(value)
        for key, value in sorted(state.items())
    }


def _state_from_npz(data: Any, prefix: str) -> dict[str, np.ndarray]:
    marker = f"{prefix}__"
    return {
        key[len(marker):]: np.asarray(data[key])
        for key in data.files
        if key.startswith(marker)
    }


def _compare_states(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_exact = True
    shared = sorted(set(actual) & set(expected))
    for key in shared:
        left = np.asarray(actual[key])
        right = np.asarray(expected[key])
        exact = bool(left.shape == right.shape and np.array_equal(left, right))
        if left.shape == right.shape and np.issubdtype(left.dtype, np.number):
            max_abs = float(
                np.max(
                    np.abs(
                        left.astype(np.complex128)
                        - right.astype(np.complex128)
                    )
                )
            )
        else:
            max_abs = None
        checks[key] = {
            "exact": exact,
            "actual_dtype": str(left.dtype),
            "expected_dtype": str(right.dtype),
            "shape": list(left.shape),
            "max_abs": max_abs,
        }
        all_exact = all_exact and exact
    missing_actual = sorted(set(expected) - set(actual))
    missing_expected = sorted(set(actual) - set(expected))
    return {
        "all_exact": bool(
            all_exact and not missing_actual and not missing_expected
        ),
        "missing_actual": missing_actual,
        "missing_expected": missing_expected,
        "checks": checks,
    }


def _apf_files(apf_dir: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in apf_dir.glob("P_steps_*.npz"):
        match = _APF_RE.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    paths = [path for _, path in sorted(found)]
    if not paths:
        raise FileNotFoundError(f"No APF chunks under {apf_dir}.")
    return paths


def _load_apf_prefix(
    apf_dir: Path,
    *,
    n_particles: int,
    end_step: int,
) -> dict[str, Any]:
    xy_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    points0 = None
    channels0 = None
    channels_end = None
    source_files: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    for path in _apf_files(apf_dir):
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            keep = steps <= int(end_step)
            if not np.any(keep):
                continue
            xy = np.asarray(data["lagrangian_xy"][keep, :n_particles], dtype=np.float32)
            channels = np.asarray(data["lagrangian_c"][keep, :n_particles], dtype=np.int32)
            kept_steps = steps[keep]
            if points0 is None:
                if int(kept_steps[0]) != 0:
                    raise ValueError(f"First APF sample is not step 0 in {path}.")
                points0 = xy[0].copy()
                channels0 = channels[0].copy()
            channels_end = channels[-1].copy()
            xy_parts.append(xy)
            step_parts.append(kept_steps)
            for requested in (0, 50, int(end_step)):
                matches = np.flatnonzero(steps == requested)
                if matches.size == 1:
                    index = int(matches[0])
                    snapshots[requested] = {
                        key: np.asarray(data[key][index])
                        for key in ("A", "P", "F")
                    }
            source_files.append(
                {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )

    steps = np.concatenate(step_parts, axis=0)
    xy = np.concatenate(xy_parts, axis=0)
    unique, indices = np.unique(steps, return_index=True)
    steps = unique
    xy = xy[indices]
    expected = np.arange(0, int(end_step) + 1, 50, dtype=np.int64)
    if not np.array_equal(steps, expected):
        raise ValueError(
            f"APF steps are not contiguous 0..{end_step} by 50: "
            f"shape={steps.shape}, first={steps[:5]}, last={steps[-5:]}."
        )
    if points0 is None or channels0 is None or channels_end is None:
        raise RuntimeError("Could not load APF particle boundary states.")
    if int(end_step) not in snapshots:
        raise RuntimeError(f"APF has no state snapshot at step={end_step}.")
    return {
        "steps": steps,
        "xy": xy,
        "points0": points0,
        "channels0": channels0,
        "channels_end": channels_end,
        "snapshots": snapshots,
        "source_files": source_files,
    }


def _load_bootstrap_state(
    path: Path,
    *,
    trial_idx: int,
    variant: str,
) -> dict[str, np.ndarray]:
    trial_prefix = f"trial_{trial_idx:05d}"
    state_name = "initial" if variant == "initial" else variant
    with np.load(path, allow_pickle=False) as data:
        keys = json.loads(str(data[f"{trial_prefix}__state_keys_json"].item()))
        return {
            key: np.asarray(data[f"{trial_prefix}__{state_name}__{key}"])
            for key in keys
        }


def _load_reference_checkpoint_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return _state_from_npz(data, "lag_state")


def _protocol_id(
    *,
    cfg_path: Path,
    params: np.ndarray,
    initial_state: dict[str, np.ndarray],
    source_files: list[dict[str, Any]],
    n_particles: int,
) -> str:
    payload = {
        "version": PROTOCOL_VERSION,
        "config_sha256": _sha256_file(cfg_path),
        "params_sha256": _sha256_array(params),
        "initial_state_sha256": {
            key: _sha256_array(value)
            for key, value in sorted(initial_state.items())
        },
        "source_apf": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in source_files
        ],
        "n_particles": int(n_particles),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _branch_dir(output_dir: Path, branch: str) -> Path:
    return output_dir / "trajectory_chunks" / branch


def _chunk_path(
    output_dir: Path,
    branch: str,
    first_step: int,
    last_step: int,
) -> Path:
    return (
        _branch_dir(output_dir, branch)
        / f"xy_steps_{first_step:07d}_{last_step:07d}.npz"
    )


def _save_checkpoint(
    path: Path,
    *,
    protocol_id: str,
    branch: str,
    step: int,
    mode: str,
    state: dict[str, Any],
    points: Any,
    channels: Any,
) -> None:
    payload = {
        "protocol_id": np.asarray(protocol_id),
        "branch": np.asarray(branch),
        "step": np.asarray(step, dtype=np.int64),
        "mode": np.asarray(mode),
        "points": np.asarray(points, dtype=np.float32),
        "channels": np.asarray(channels, dtype=np.int32),
        **_state_to_payload("state", state),
    }
    _save_npz(path, **payload)


def _load_checkpoint(
    path: Path,
    *,
    expected_protocol_id: str,
    expected_branch: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        protocol_id = str(np.asarray(data["protocol_id"]).item())
        branch = str(np.asarray(data["branch"]).item())
        if protocol_id != expected_protocol_id or branch != expected_branch:
            raise RuntimeError(
                f"Incompatible checkpoint {path}: protocol={protocol_id}, "
                f"branch={branch}."
            )
        return {
            "step": int(np.asarray(data["step"]).item()),
            "mode": str(np.asarray(data["mode"]).item()),
            "state": _state_from_npz(data, "state"),
            "points": np.asarray(data["points"], dtype=np.float32),
            "channels": np.asarray(data["channels"], dtype=np.int32),
        }


def _existing_chunk_steps(
    output_dir: Path,
    branch: str,
    *,
    protocol_id: str,
) -> np.ndarray:
    steps_parts: list[np.ndarray] = []
    for path in sorted(_branch_dir(output_dir, branch).glob("xy_steps_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            found_protocol = str(np.asarray(data["protocol_id"]).item())
            if found_protocol != protocol_id:
                raise RuntimeError(f"Incompatible trajectory chunk: {path}.")
            steps_parts.append(np.asarray(data["steps"], dtype=np.int64))
    if not steps_parts:
        return np.zeros((0,), dtype=np.int64)
    steps = np.concatenate(steps_parts)
    if np.unique(steps).size != steps.size or not np.all(np.diff(steps) > 0):
        raise RuntimeError(f"Overlapping or unordered chunks for {branch}.")
    return steps


def _write_track_chunk(
    output_dir: Path,
    branch: str,
    *,
    protocol_id: str,
    steps: list[int],
    xy: list[np.ndarray],
) -> Path:
    if not steps:
        raise ValueError("Cannot write an empty trajectory chunk.")
    path = _chunk_path(output_dir, branch, int(steps[0]), int(steps[-1]))
    _save_npz(
        path,
        protocol_id=np.asarray(protocol_id),
        branch=np.asarray(branch),
        steps=np.asarray(steps, dtype=np.int64),
        xy=np.stack(xy, axis=0).astype(np.float32, copy=False),
    )
    return path


def _load_branch_trajectory(
    output_dir: Path,
    branch: str,
    *,
    protocol_id: str,
    apf_prefix: dict[str, Any] | None,
    total_steps: int,
    sample_every_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    steps_parts: list[np.ndarray] = []
    xy_parts: list[np.ndarray] = []
    if apf_prefix is not None:
        steps_parts.append(
            np.asarray(apf_prefix["steps"][1:], dtype=np.int64)
        )
        xy_parts.append(
            np.asarray(apf_prefix["xy"][1:], dtype=np.float32)
        )
    for path in sorted(_branch_dir(output_dir, branch).glob("xy_steps_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["protocol_id"]).item()) != protocol_id:
                raise RuntimeError(f"Incompatible trajectory chunk: {path}.")
            steps_parts.append(np.asarray(data["steps"], dtype=np.int64))
            xy_parts.append(np.asarray(data["xy"], dtype=np.float32))
    if not steps_parts:
        raise FileNotFoundError(f"No trajectory chunks for {branch}.")
    steps = np.concatenate(steps_parts)
    xy = np.concatenate(xy_parts, axis=0)
    order = np.argsort(steps)
    steps = steps[order]
    xy = xy[order]
    expected = np.arange(
        int(sample_every_steps),
        int(total_steps) + 1,
        int(sample_every_steps),
        dtype=np.int64,
    )
    if not np.array_equal(steps, expected):
        raise RuntimeError(
            f"Incomplete trajectory for {branch}: got {steps.shape[0]} samples, "
            f"expected {expected.shape[0]}."
        )
    return steps, xy


def _runtime_from_config(cfg: Any):
    import jax
    import jax.numpy as jnp
    import substrates
    import util
    from paper_check_frustration_batch_eval import _create_substrate, _flat_cfg

    flat = SimpleNamespace(
        **OmegaConf.to_container(_flat_cfg(cfg), resolve=True)
    )
    substrate = _create_substrate(flat, enable_msc=True)
    params = np.asarray(np.load(cfg.job.control_a_reference_params_path), dtype=np.float32)
    params_jax = jnp.asarray(params)
    _ = substrate.seed_state(jax.random.PRNGKey(0), params_jax)

    split_n = int(cfg.protocol.grid_split)
    grid_size = int(cfg.substrate.grid_size)
    pad = int(cfg.protocol.wall_pad)
    block_size = (grid_size + split_n - 1) // split_n
    if (block_size + 2 * pad) % 2 != 0:
        block_size += 1
    padded_grid_size = int(block_size * split_n)
    partition_padding = int(padded_grid_size - grid_size)
    crop_before = int(partition_padding // 2)
    crop_after = int(partition_padding - crop_before)
    block_kwargs = util.flow_lenia_kwargs_from_args(flat)
    block_kwargs["grid_size"] = int(block_size + 2 * pad)
    block_state_kwargs = dict(block_kwargs)
    block_state_kwargs["debug_return_F"] = False
    block_state_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate("lenia_flow", **block_state_kwargs)
    )
    _ = block_state_substrate.seed_state(
        jax.random.PRNGKey(0),
        params_jax,
    )
    block_kwargs["debug_return_F"] = True
    block_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate("lenia_flow", **block_kwargs)
    )
    _ = block_substrate.seed_state(jax.random.PRNGKey(0), params_jax)
    return {
        "flat": flat,
        "substrate": substrate,
        "block_state_substrate": block_state_substrate,
        "block_substrate": block_substrate,
        "params": params,
        "params_jax": params_jax,
        "split_n": split_n,
        "grid_size": grid_size,
        "pad": pad,
        "block_size": block_size,
        "padded_grid_size": padded_grid_size,
        "crop_before": crop_before,
        "crop_after": crop_after,
    }


def _build_block_lagrangian_stepper(
    block_substrate: Any,
    *,
    n_blocks: int,
    chunk_steps: int,
    valid_mask: Any,
    cfg: Any,
):
    import jax
    import jax.numpy as jnp
    from paper_check_frustration_batch_eval import _mask_block_spatial_state

    rt = block_substrate.RT
    flow_channel = int(cfg.metric.metric_lagrangian_flow_channel)
    flow_reduce = str(cfg.metric.metric_lagrangian_flow_reduce)
    channel_mode = str(cfg.metric.metric_lagrangian_channel_mode)
    noise_model = str(cfg.metric.metric_lagrangian_noise_model)
    diffusion_scale = float(cfg.metric.metric_lagrangian_diffusion_scale)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    @jax.jit
    def step(rng_key, carry, params):
        def body(inner_carry, step_key):
            block_state, points, channels = inner_carry
            block_keys = jax.random.split(step_key, n_blocks)
            block_state = jax.vmap(
                lambda state, key: block_substrate.step_state(
                    key, state, params
                )
            )(block_state, block_keys)
            block_state = _mask_block_spatial_state(block_state, valid_mask)
            lag_keys = jax.vmap(
                lambda key: jax.random.fold_in(
                    key, jnp.uint32(0x4C4147)
                )
            )(block_keys)

            def advect(points_one, channels_one, flow, mass, lag_key):
                return rt.advect_particles(
                    points=points_one,
                    F=flow,
                    A=mass,
                    channel=flow_channel,
                    reduce=flow_reduce,
                    point_channels=channels_one,
                    channel_mode=channel_mode,
                    key=lag_key,
                    noise_model=noise_model,
                    diffusion_scale=diffusion_scale,
                )

            points, channels = jax.vmap(advect)(
                points,
                channels,
                block_state["F"],
                block_state["A"],
                lag_keys,
            )
            return (block_state, points, channels), None

        return jax.lax.scan(
            body,
            carry,
            jax.random.split(rng_key, int(chunk_steps)),
        )[0]

    return step


def _build_block_state_only_stepper(
    block_substrate: Any,
    *,
    n_blocks: int,
    chunk_steps: int,
    valid_mask: Any,
):
    import jax
    import jax.numpy as jnp
    from paper_check_frustration_batch_eval import _mask_block_spatial_state

    valid_mask = jnp.asarray(valid_mask, dtype=bool)

    @jax.jit
    def step(rng_key, block_state, params):
        def body(state, step_key):
            block_keys = jax.random.split(step_key, n_blocks)
            next_state = jax.vmap(
                lambda one_state, key: block_substrate.step_state(
                    key, one_state, params
                )
            )(state, block_keys)
            return _mask_block_spatial_state(next_state, valid_mask), None

        return jax.lax.scan(
            body,
            block_state,
            jax.random.split(rng_key, int(chunk_steps)),
        )[0]

    return step


def _build_global_state_only_stepper(substrate: Any, *, chunk_steps: int):
    import jax

    @jax.jit
    def step(rng_key, state, params):
        def body(inner_state, step_key):
            return substrate.step_state(step_key, inner_state, params), None

        return jax.lax.scan(
            body,
            state,
            jax.random.split(rng_key, int(chunk_steps)),
        )[0]

    return step


def _pack_wall_particles(
    points: np.ndarray,
    channels: np.ndarray,
    *,
    split_n: int,
    block_size: int,
    pad: int,
    crop_before: int,
) -> dict[str, np.ndarray]:
    padded = np.asarray(points, dtype=np.float32) + float(crop_before)
    block_rc = np.floor((padded - 0.5) / float(block_size)).astype(np.int32)
    block_rc = np.clip(block_rc, 0, split_n - 1)
    block_idx = block_rc[:, 0] * split_n + block_rc[:, 1]
    counts = np.bincount(block_idx, minlength=split_n * split_n)
    max_count = int(np.max(counts))
    packed_points = np.full(
        (split_n * split_n, max_count, 2),
        float(pad) + 0.5,
        dtype=np.float32,
    )
    packed_channels = np.zeros(
        (split_n * split_n, max_count),
        dtype=np.int32,
    )
    slots = np.zeros((points.shape[0],), dtype=np.int32)
    cursor = np.zeros((split_n * split_n,), dtype=np.int32)
    for particle_idx, block in enumerate(block_idx):
        slot = int(cursor[block])
        slots[particle_idx] = slot
        cursor[block] += 1
        row = int(block // split_n)
        col = int(block % split_n)
        packed_points[block, slot, 0] = (
            padded[particle_idx, 0] - row * block_size + pad
        )
        packed_points[block, slot, 1] = (
            padded[particle_idx, 1] - col * block_size + pad
        )
        packed_channels[block, slot] = channels[particle_idx]
    return {
        "points": packed_points,
        "channels": packed_channels,
        "block_idx": block_idx.astype(np.int32),
        "slots": slots,
        "counts": counts.astype(np.int32),
    }


def _unpack_wall_particles(
    packed_points: Any,
    packed_channels: Any,
    *,
    block_idx: np.ndarray,
    slots: np.ndarray,
    split_n: int,
    block_size: int,
    pad: int,
    crop_before: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_host = np.asarray(packed_points, dtype=np.float32)
    channels_host = np.asarray(packed_channels, dtype=np.int32)
    selected = points_host[block_idx, slots].copy()
    rows = block_idx // split_n
    cols = block_idx % split_n
    selected[:, 0] += rows * block_size - pad - crop_before
    selected[:, 1] += cols * block_size - pad - crop_before
    return selected, channels_host[block_idx, slots].copy()


def _merge_wall_state(
    initial_state: dict[str, Any],
    block_state: dict[str, Any],
    *,
    split_n: int,
    block_size: int,
    pad: int,
    crop_before: int,
    crop_after: int,
    grid_size: int,
):
    import jax.numpy as jnp
    from evaluate_frustration_history_dependence import (
        _merge_blocks_into_global_state,
    )
    from paper_check_frustration_batch_eval import _pad_flow_spatial_state

    padded_initial = _pad_flow_spatial_state(
        initial_state,
        pad_before=crop_before,
        pad_after=crop_after,
    )
    merged_padded = _merge_blocks_into_global_state(
        padded_initial,
        block_state,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
    )
    merged = dict(initial_state)
    padded_shape = (
        grid_size + crop_before + crop_after,
        grid_size + crop_before + crop_after,
    )
    for key, value in merged_padded.items():
        arr = jnp.asarray(value)
        if arr.ndim >= 2 and tuple(arr.shape[:2]) == padded_shape:
            merged[key] = arr[
                crop_before:crop_before + grid_size,
                crop_before:crop_before + grid_size,
            ]
        elif key in {"t", "mass_cycle_start"}:
            merged[key] = value
    merged["mass_cycle_start"] = jnp.sum(merged["A"])
    return merged


def _save_branch_manifest(
    output_dir: Path,
    branch: str,
    *,
    protocol_id: str,
    start_step: int,
    total_steps: int,
    sample_every_steps: int,
    seed: int,
    final_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": protocol_id,
        "branch": branch,
        "start_step": int(start_step),
        "total_steps": int(total_steps),
        "sample_every_steps": int(sample_every_steps),
        "run_seed": int(seed),
        "final_state_sha256": {
            key: _sha256_array(np.asarray(value))
            for key, value in sorted(final_state.items())
        },
    }
    if extra:
        payload.update(extra)
    _write_json(_branch_dir(output_dir, branch) / "manifest.json", payload)


def _run_global_branch(
    *,
    output_dir: Path,
    branch: str,
    protocol_id: str,
    substrate: Any,
    params: Any,
    initial_state: dict[str, Any],
    initial_points: np.ndarray,
    initial_channels: np.ndarray,
    seed: int,
    start_step: int,
    total_steps: int,
    training_horizon_steps: int,
    sample_every_steps: int,
    checkpoint_every_steps: int,
    log_clip_evolution: bool,
    cfg: Any,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from evaluate_frustration_history_dependence import (
        _build_lagrangian_chunk_stepper,
    )
    from paper_check_frustration_batch_eval import (
        _optimizer_metric_key_schedule,
    )

    branch_dir = _branch_dir(output_dir, branch)
    branch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = branch_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") == "complete"
            and manifest.get("protocol_id") == protocol_id
        ):
            print(f"[{branch}] complete cache hit", flush=True)
            final_path = branch_dir / "final_state.npz"
            with np.load(final_path, allow_pickle=False) as data:
                return _state_from_npz(data, "state")

    checkpoint_path = branch_dir / "resume_checkpoint.npz"
    resumed = _load_checkpoint(
        checkpoint_path,
        expected_protocol_id=protocol_id,
        expected_branch=branch,
    )
    if resumed is None:
        current_step = int(start_step)
        state = {
            key: jnp.asarray(value)
            for key, value in initial_state.items()
        }
        points = jnp.asarray(initial_points, dtype=jnp.float32)
        channels = jnp.asarray(initial_channels, dtype=jnp.int32)
    else:
        current_step = int(resumed["step"])
        if resumed["mode"] != "global":
            raise RuntimeError(
                f"Expected global checkpoint for {branch}, got {resumed['mode']}."
            )
        state = {
            key: jnp.asarray(value)
            for key, value in resumed["state"].items()
        }
        points = jnp.asarray(resumed["points"], dtype=jnp.float32)
        channels = jnp.asarray(resumed["channels"], dtype=jnp.int32)
        print(f"[{branch}] resuming at step={current_step}", flush=True)

    existing_steps = _existing_chunk_steps(
        output_dir,
        branch,
        protocol_id=protocol_id,
    )
    expected_existing = np.arange(
        int(start_step) + sample_every_steps,
        current_step + 1,
        sample_every_steps,
        dtype=np.int64,
    )
    if not np.array_equal(existing_steps, expected_existing):
        raise RuntimeError(
            f"Chunk/checkpoint mismatch for {branch}: chunks={existing_steps.shape[0]}, "
            f"checkpoint_step={current_step}."
        )

    schedule = _optimizer_metric_key_schedule(
        run_seed=int(seed),
        total_steps=int(total_steps),
        training_horizon_steps=int(training_horizon_steps),
        chunk_steps=int(sample_every_steps),
        log_clip_evolution=bool(log_clip_evolution),
    )
    stepper = _build_lagrangian_chunk_stepper(
        substrate,
        chunk_steps=sample_every_steps,
        lag_flow_channel=int(cfg.metric.metric_lagrangian_flow_channel),
        lag_flow_reduce=str(cfg.metric.metric_lagrangian_flow_reduce),
        lag_channel_mode=str(cfg.metric.metric_lagrangian_channel_mode),
        lag_noise_model=str(cfg.metric.metric_lagrangian_noise_model),
        lag_diffusion_scale=float(
            cfg.metric.metric_lagrangian_diffusion_scale
        ),
    )
    state_only_stepper = _build_global_state_only_stepper(
        substrate,
        chunk_steps=sample_every_steps,
    )
    coupled_from_step = int(cfg.evaluation.late_window_start_steps)
    carry = (state, points, channels)
    chunk_steps: list[int] = []
    chunk_xy: list[np.ndarray] = []
    wall_start = time.monotonic()
    while current_step < total_steps:
        chunk_idx = int(current_step // sample_every_steps)
        chunk_key = schedule[chunk_idx]
        if current_step < coupled_from_step:
            state_before, points_before, channels_before = carry
            canonical_state = state_only_stepper(
                chunk_key,
                state_before,
                params,
            )
            shadow_carry, xy = stepper(
                chunk_key,
                (state_before, points_before, channels_before),
                params,
            )
            carry = (
                canonical_state,
                shadow_carry[1],
                shadow_carry[2],
            )
        else:
            carry, xy = stepper(chunk_key, carry, params)
        current_step += sample_every_steps
        chunk_steps.append(current_step)
        chunk_xy.append(
            np.asarray(jax.device_get(xy), dtype=np.float32)
        )
        if (
            current_step % checkpoint_every_steps == 0
            or current_step == total_steps
        ):
            chunk_path = _write_track_chunk(
                output_dir,
                branch,
                protocol_id=protocol_id,
                steps=chunk_steps,
                xy=chunk_xy,
            )
            state_host = jax.device_get(carry[0])
            points_host = np.asarray(jax.device_get(carry[1]), dtype=np.float32)
            channels_host = np.asarray(jax.device_get(carry[2]), dtype=np.int32)
            _save_checkpoint(
                checkpoint_path,
                protocol_id=protocol_id,
                branch=branch,
                step=current_step,
                mode="global",
                state=state_host,
                points=points_host,
                channels=channels_host,
            )
            elapsed = time.monotonic() - wall_start
            rate = (current_step - int(start_step)) / max(elapsed, 1e-9)
            remaining = (total_steps - current_step) / max(rate, 1e-9)
            print(
                f"[{branch}] step={current_step}/{total_steps} "
                f"chunk={chunk_path.name} rate={rate:.1f} step/s "
                f"eta={remaining / 60.0:.1f} min",
                flush=True,
            )
            chunk_steps = []
            chunk_xy = []

    final_state = jax.device_get(carry[0])
    _save_npz(
        branch_dir / "final_state.npz",
        protocol_id=np.asarray(protocol_id),
        **_state_to_payload("state", final_state),
    )
    _save_branch_manifest(
        output_dir,
        branch,
        protocol_id=protocol_id,
        start_step=start_step,
        total_steps=total_steps,
        sample_every_steps=sample_every_steps,
        seed=seed,
        final_state=final_state,
        extra={
            "state_execution": (
                "Canonical state-only chunks with a discarded shadow replay for "
                f"particle tracking before step {coupled_from_step}; coupled "
                "state+tracker chunks thereafter, matching the original C5 late window."
            ),
            "coupled_tracker_from_step": coupled_from_step,
        },
    )
    return final_state


def _run_walls_branch(
    *,
    output_dir: Path,
    protocol_id: str,
    runtime: dict[str, Any],
    initial_state: dict[str, Any],
    initial_points: np.ndarray,
    initial_channels: np.ndarray,
    seed: int,
    total_steps: int,
    warmup_steps: int,
    training_horizon_steps: int,
    sample_every_steps: int,
    checkpoint_every_steps: int,
    log_clip_evolution: bool,
    cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from evaluate_frustration_history_dependence import (
        _build_lagrangian_chunk_stepper,
        _prepare_block_template_state,
    )
    from paper_check_frustration_batch_eval import (
        _block_valid_mask,
        _optimizer_metric_key_schedule,
    )

    branch = "walls"
    branch_dir = _branch_dir(output_dir, branch)
    branch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = branch_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") == "complete"
            and manifest.get("protocol_id") == protocol_id
        ):
            print("[walls] complete cache hit", flush=True)
            with np.load(
                branch_dir / "final_state.npz", allow_pickle=False
            ) as data:
                return _state_from_npz(data, "state"), dict(
                    manifest.get("wall_particle_transition", {})
                )

    params = runtime["params_jax"]
    substrate = runtime["substrate"]
    block_state_substrate = runtime["block_state_substrate"]
    block_substrate = runtime["block_substrate"]
    split_n = int(runtime["split_n"])
    grid_size = int(runtime["grid_size"])
    pad = int(runtime["pad"])
    block_size = int(runtime["block_size"])
    crop_before = int(runtime["crop_before"])
    crop_after = int(runtime["crop_after"])
    n_blocks = split_n * split_n
    valid_mask = _block_valid_mask(
        grid_size=grid_size,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
        global_crop_start=crop_before,
    )
    block_init_key = jax.random.PRNGKey(0)
    block_template = block_state_substrate.init_state(
        block_init_key,
        params,
    )
    padded_initial = {
        key: jnp.asarray(value)
        for key, value in initial_state.items()
    }
    from paper_check_frustration_batch_eval import _pad_flow_spatial_state

    padded_initial = _pad_flow_spatial_state(
        padded_initial,
        pad_before=crop_before,
        pad_after=crop_after,
    )
    block_state0 = _prepare_block_template_state(
        initial_state=padded_initial,
        block_template=block_template,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
        C=int(cfg.substrate.C),
        k=int(cfg.substrate.k),
    )
    packed = _pack_wall_particles(
        initial_points,
        initial_channels,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
        crop_before=crop_before,
    )

    checkpoint_path = branch_dir / "resume_checkpoint.npz"
    resumed = _load_checkpoint(
        checkpoint_path,
        expected_protocol_id=protocol_id,
        expected_branch=branch,
    )
    if resumed is None:
        current_step = 0
        mode = "block"
        state = block_state0
        points = jnp.asarray(packed["points"])
        channels = jnp.asarray(packed["channels"])
    else:
        current_step = int(resumed["step"])
        mode = str(resumed["mode"])
        state = {
            key: jnp.asarray(value)
            for key, value in resumed["state"].items()
        }
        points = jnp.asarray(resumed["points"])
        channels = jnp.asarray(resumed["channels"])
        print(f"[walls] resuming at step={current_step} mode={mode}", flush=True)

    existing_steps = _existing_chunk_steps(
        output_dir,
        branch,
        protocol_id=protocol_id,
    )
    expected_existing = np.arange(
        sample_every_steps,
        current_step + 1,
        sample_every_steps,
        dtype=np.int64,
    )
    if not np.array_equal(existing_steps, expected_existing):
        raise RuntimeError(
            f"Chunk/checkpoint mismatch for walls at step={current_step}."
        )
    schedule = _optimizer_metric_key_schedule(
        run_seed=int(seed),
        total_steps=int(total_steps),
        training_horizon_steps=int(training_horizon_steps),
        chunk_steps=int(sample_every_steps),
        log_clip_evolution=bool(log_clip_evolution),
    )
    block_stepper = _build_block_lagrangian_stepper(
        block_substrate,
        n_blocks=n_blocks,
        chunk_steps=sample_every_steps,
        valid_mask=jnp.asarray(valid_mask),
        cfg=cfg,
    )
    block_state_only_stepper = _build_block_state_only_stepper(
        block_state_substrate,
        n_blocks=n_blocks,
        chunk_steps=sample_every_steps,
        valid_mask=jnp.asarray(valid_mask),
    )
    global_stepper = _build_lagrangian_chunk_stepper(
        substrate,
        chunk_steps=sample_every_steps,
        lag_flow_channel=int(cfg.metric.metric_lagrangian_flow_channel),
        lag_flow_reduce=str(cfg.metric.metric_lagrangian_flow_reduce),
        lag_channel_mode=str(cfg.metric.metric_lagrangian_channel_mode),
        lag_noise_model=str(cfg.metric.metric_lagrangian_noise_model),
        lag_diffusion_scale=float(
            cfg.metric.metric_lagrangian_diffusion_scale
        ),
    )
    global_state_only_stepper = _build_global_state_only_stepper(
        substrate,
        chunk_steps=sample_every_steps,
    )
    coupled_from_step = int(cfg.evaluation.late_window_start_steps)
    carry = (state, points, channels)
    chunk_steps: list[int] = []
    chunk_xy: list[np.ndarray] = []
    transition: dict[str, Any] = {}
    wall_start = time.monotonic()
    while current_step < total_steps:
        chunk_idx = int(current_step // sample_every_steps)
        chunk_key = schedule[chunk_idx]
        if mode == "block":
            state_before, points_before, channels_before = carry
            canonical_state = block_state_only_stepper(
                chunk_key,
                state_before,
                params,
            )
            shadow_state_before = dict(state_before)
            if "F" not in shadow_state_before:
                mass = shadow_state_before["A"]
                shadow_state_before["F"] = jnp.zeros(
                    mass.shape[:3] + (2, mass.shape[-1]),
                    dtype=mass.dtype,
                )
            shadow_carry = block_stepper(
                chunk_key,
                (
                    shadow_state_before,
                    points_before,
                    channels_before,
                ),
                params,
            )
            carry = (
                canonical_state,
                shadow_carry[1],
                shadow_carry[2],
            )
            points_global, channels_global = _unpack_wall_particles(
                jax.device_get(carry[1]),
                jax.device_get(carry[2]),
                block_idx=packed["block_idx"],
                slots=packed["slots"],
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                crop_before=crop_before,
            )
            xy = points_global
        else:
            if current_step < coupled_from_step:
                state_before, points_before, channels_before = carry
                canonical_state = global_state_only_stepper(
                    chunk_key,
                    state_before,
                    params,
                )
                shadow_carry, xy_jax = global_stepper(
                    chunk_key,
                    (state_before, points_before, channels_before),
                    params,
                )
                carry = (
                    canonical_state,
                    shadow_carry[1],
                    shadow_carry[2],
                )
            else:
                carry, xy_jax = global_stepper(
                    chunk_key,
                    carry,
                    params,
                )
            xy = np.asarray(jax.device_get(xy_jax), dtype=np.float32)
        current_step += sample_every_steps
        chunk_steps.append(current_step)
        chunk_xy.append(np.asarray(xy, dtype=np.float32))

        if mode == "block" and current_step == warmup_steps:
            block_state = carry[0]
            points_global, channels_global = _unpack_wall_particles(
                jax.device_get(carry[1]),
                jax.device_get(carry[2]),
                block_idx=packed["block_idx"],
                slots=packed["slots"],
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                crop_before=crop_before,
            )
            merged_state = _merge_wall_state(
                {
                    key: jnp.asarray(value)
                    for key, value in initial_state.items()
                },
                block_state,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                crop_before=crop_before,
                crop_after=crop_after,
                grid_size=grid_size,
            )
            outside = np.any(
                (points_global < 0.0)
                | (points_global > float(grid_size)),
                axis=1,
            )
            transition = {
                "step": int(current_step),
                "particle_count": int(points_global.shape[0]),
                "outside_global_domain_count_before_first_global_step": int(
                    np.sum(outside)
                ),
                "max_distance_outside_domain": float(
                    max(
                        0.0,
                        -float(np.min(points_global)),
                        float(np.max(points_global)) - float(grid_size),
                    )
                ),
                "semantics": (
                    "Particles are tracked inside their original padded wall block, "
                    "mapped back to global coordinates at wall removal, and then "
                    "continued in the merged global simulation."
                ),
            }
            carry = (
                merged_state,
                jnp.asarray(points_global, dtype=jnp.float32),
                jnp.asarray(channels_global, dtype=jnp.int32),
            )
            mode = "global"

        if (
            current_step % checkpoint_every_steps == 0
            or current_step == total_steps
        ):
            chunk_path = _write_track_chunk(
                output_dir,
                branch,
                protocol_id=protocol_id,
                steps=chunk_steps,
                xy=chunk_xy,
            )
            state_host = jax.device_get(carry[0])
            points_host = np.asarray(jax.device_get(carry[1]), dtype=np.float32)
            channels_host = np.asarray(jax.device_get(carry[2]), dtype=np.int32)
            _save_checkpoint(
                checkpoint_path,
                protocol_id=protocol_id,
                branch=branch,
                step=current_step,
                mode=mode,
                state=state_host,
                points=points_host,
                channels=channels_host,
            )
            elapsed = time.monotonic() - wall_start
            rate = current_step / max(elapsed, 1e-9)
            remaining = (total_steps - current_step) / max(rate, 1e-9)
            print(
                f"[walls] step={current_step}/{total_steps} mode={mode} "
                f"chunk={chunk_path.name} rate={rate:.1f} step/s "
                f"eta={remaining / 60.0:.1f} min",
                flush=True,
            )
            chunk_steps = []
            chunk_xy = []

    final_state = jax.device_get(carry[0])
    _save_npz(
        branch_dir / "final_state.npz",
        protocol_id=np.asarray(protocol_id),
        **_state_to_payload("state", final_state),
    )
    _save_branch_manifest(
        output_dir,
        branch,
        protocol_id=protocol_id,
        start_step=0,
        total_steps=total_steps,
        sample_every_steps=sample_every_steps,
        seed=seed,
        final_state=final_state,
        extra={
            "wall_particle_transition": transition,
            "wall_particle_block_counts": packed["counts"].tolist(),
            "wall_particle_padded_count_per_block": int(
                packed["points"].shape[1]
            ),
            "state_execution": (
                "Canonical state-only chunks with a discarded shadow replay for "
                f"particle tracking before step {coupled_from_step}; coupled "
                "state+tracker chunks thereafter, matching the original C5 late window."
            ),
            "coupled_tracker_from_step": coupled_from_step,
        },
    )
    return final_state, transition


def _training_reference_check(
    state: dict[str, np.ndarray],
    snapshot: dict[str, np.ndarray],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_exact = True
    for key in ("A", "P", "F"):
        actual = np.asarray(state[key])
        expected = np.asarray(snapshot[key])
        exact = bool(np.array_equal(actual.astype(expected.dtype), expected))
        checks[key] = {
            "exact_after_reference_cast": exact,
            "reference_dtype": str(expected.dtype),
            "max_abs_vs_reference_values": float(
                np.max(
                    np.abs(
                        actual.astype(np.float32)
                        - expected.astype(np.float32)
                    )
                )
            ),
        }
        all_exact = all_exact and exact
    return {"status": "exact" if all_exact else "mismatch", "checks": checks}


def _select_negative_run_evidence(output_dir: Path) -> dict[str, Any]:
    import pandas as pd

    table = (
        _REPO_ROOT
        / "analysis/results/"
        "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
        "flow_lenia/frustration_run_level.csv"
    )
    frame = pd.read_csv(table)
    column = (
        "embedding_cloud_chamfer_cosine__"
        "anchor_effect_minus_baseline__delta_vs_random_median"
    )
    selected = frame.loc[frame[column].astype(float).idxmin()]
    evidence = {
        "source_table": str(table),
        "selection_metric": column,
        "selected_optimized_run_idx": int(selected["optimized_run_idx"]),
        "selected_value": float(selected[column]),
        "is_global_minimum": True,
    }
    _write_json(output_dir / "negative_run_selection.json", evidence)
    return evidence


def simulate(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    trial_dir = _resolve(args.trial_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = trial_dir / "resolved_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    trial_idx = int(cfg.meta.trial_idx)
    run_idx = int(cfg.meta.optimized_run_idx)
    evidence = _select_negative_run_evidence(output_dir)
    if run_idx != int(evidence["selected_optimized_run_idx"]):
        raise RuntimeError(
            f"Trial run opt_{run_idx:03d} is not the strongest negative run "
            f"opt_{int(evidence['selected_optimized_run_idx']):03d}."
        )

    runtime = _runtime_from_config(cfg)
    params = runtime["params"]
    n_particles = int(args.n_particles)
    if n_particles < 2:
        raise ValueError("--n-particles must be >= 2.")
    apf_dir = _resolve(cfg.job.control_a_reference_apf_dir)
    training_horizon = int(cfg.evaluation.training_horizon_steps)
    apf = _load_apf_prefix(
        apf_dir,
        n_particles=n_particles,
        end_step=training_horizon,
    )
    bootstrap_root = _resolve(cfg.evaluation.bootstrap_cache_root)
    bootstrap_path = (
        bootstrap_root
        / "optimizer_native_bootstrap"
        / f"group_{run_idx:03d}_step_{training_horizon:07d}.npz"
    )
    initial_state = _load_bootstrap_state(
        bootstrap_path,
        trial_idx=trial_idx,
        variant="initial",
    )
    control_a_start = _load_bootstrap_state(
        bootstrap_path,
        trial_idx=trial_idx,
        variant="control_a",
    )
    protocol_id = _protocol_id(
        cfg_path=cfg_path,
        params=params,
        initial_state=initial_state,
        source_files=apf["source_files"],
        n_particles=n_particles,
    )

    params_b = np.asarray(
        np.load(cfg.job.control_b_reference_params_path),
        dtype=np.float32,
    )
    if not np.array_equal(params, params_b):
        raise RuntimeError("Control A and B parameter arrays differ.")
    init_equality = {
        branch: {
            key: _sha256_array(value)
            for key, value in sorted(initial_state.items())
        }
        for branch in ("control_a", "control_b", "walls")
    }
    initial_hash_sets = {
        tuple(sorted(value.items())) for value in init_equality.values()
    }
    if len(initial_hash_sets) != 1:
        raise RuntimeError("Branch initial states are not identical.")
    training_check = _training_reference_check(
        control_a_start,
        apf["snapshots"][training_horizon],
    )
    if training_check["status"] != "exact":
        raise RuntimeError("Control A bootstrap does not match optimizer APF.")

    sample_every = int(cfg.metric.sample_every_steps)
    total_steps = int(cfg.protocol.total_steps)
    warmup_steps = int(cfg.protocol.warmup_steps)
    seed_a = int(cfg.job.seed_x)
    seed_b = int(cfg.job.seed_x1)
    checkpoint_every = int(args.checkpoint_every_steps)
    if checkpoint_every % sample_every != 0:
        raise ValueError(
            "--checkpoint-every-steps must be divisible by sample cadence."
        )
    audit_path = output_dir / "protocol_audit.json"
    _write_json(
        audit_path,
        {
            "status": "simulation_running",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_id": protocol_id,
            "trial_dir": str(trial_dir),
            "trial_idx": trial_idx,
            "optimized_run_idx": run_idx,
            "candidate_label": str(cfg.meta.candidate_label),
            "negative_run_selection": evidence,
            "params_sha256": _sha256_array(params),
            "params_control_a_equals_control_b": True,
            "n_particles": n_particles,
            "particle_selection": (
                f"particle indices [0,{n_particles}) from exact optimizer-native "
                "Control A APF at step 0"
            ),
            "branch_initial_state_sha256": init_equality,
            "all_branch_initial_states_exactly_identical": True,
            "control_a_training_reference": training_check,
            "control_a_steps_0_to_300000_source": (
                "Copied directly from optimizer-native APF, no resimulation."
            ),
            "control_a_apf_dir": str(apf_dir),
            "control_a_apf_source_files": apf["source_files"],
            "bootstrap_cache": str(bootstrap_path),
            "seed_a": seed_a,
            "seed_b": seed_b,
            "seed_semantics": (
                "A and walls use seed_a. B starts from A's exact state and "
                "particle initialization, but uses the seed_b simulation schedule."
            ),
            "state_execution": (
                "Before the original C5 late-window tracker start, each canonical "
                "simulation chunk is executed by the state-only kernel. A shadow "
                "replay of that same chunk advances passive particles and its state "
                "output is discarded. From the original late-window start onward, "
                "the original coupled state+tracker kernel is used."
            ),
            "wall_protocol": {
                "grid_split": int(runtime["split_n"]),
                "block_size": int(runtime["block_size"]),
                "wall_pad": int(runtime["pad"]),
                "padded_grid_size": int(runtime["padded_grid_size"]),
                "global_crop_before": int(runtime["crop_before"]),
                "global_crop_after": int(runtime["crop_after"]),
                "walls_removed_step": warmup_steps,
            },
            "metric_seed": int(cfg.job.metric_seed),
            "metric_folds": {"control_a": 0, "control_b": 1, "walls": 2},
            "jax_backend": str(jax.default_backend()),
            "jax_version": str(jax.__version__),
        },
    )

    final_a = _run_global_branch(
        output_dir=output_dir,
        branch="control_a",
        protocol_id=protocol_id,
        substrate=runtime["substrate"],
        params=runtime["params_jax"],
        initial_state=control_a_start,
        initial_points=apf["xy"][-1],
        initial_channels=apf["channels_end"],
        seed=seed_a,
        start_step=training_horizon,
        total_steps=total_steps,
        training_horizon_steps=training_horizon,
        sample_every_steps=sample_every,
        checkpoint_every_steps=checkpoint_every,
        log_clip_evolution=bool(cfg.logging.log_clip_evolution),
        cfg=cfg,
    )
    final_b = _run_global_branch(
        output_dir=output_dir,
        branch="control_b",
        protocol_id=protocol_id,
        substrate=runtime["substrate"],
        params=runtime["params_jax"],
        initial_state=initial_state,
        initial_points=apf["points0"],
        initial_channels=apf["channels0"],
        seed=seed_b,
        start_step=0,
        total_steps=total_steps,
        training_horizon_steps=training_horizon,
        sample_every_steps=sample_every,
        checkpoint_every_steps=checkpoint_every,
        log_clip_evolution=bool(cfg.logging.log_clip_evolution),
        cfg=cfg,
    )
    final_walls, transition = _run_walls_branch(
        output_dir=output_dir,
        protocol_id=protocol_id,
        runtime=runtime,
        initial_state=initial_state,
        initial_points=apf["points0"],
        initial_channels=apf["channels0"],
        seed=seed_a,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        training_horizon_steps=training_horizon,
        sample_every_steps=sample_every,
        checkpoint_every_steps=checkpoint_every,
        log_clip_evolution=bool(cfg.logging.log_clip_evolution),
        cfg=cfg,
    )

    reference_a = _load_reference_checkpoint_state(
        trial_dir / "control_a_checkpoint.npz"
    )
    reference_walls = _load_reference_checkpoint_state(
        trial_dir / "walls_checkpoint.npz"
    )
    final_a_check = _compare_states(final_a, reference_a)
    final_walls_check = _compare_states(final_walls, reference_walls)
    for branch, prefix in (
        ("control_a", apf),
        ("control_b", None),
        ("walls", None),
    ):
        _load_branch_trajectory(
            output_dir,
            branch,
            protocol_id=protocol_id,
            apf_prefix=prefix,
            total_steps=total_steps,
            sample_every_steps=sample_every,
        )
    _merge_json(
        audit_path,
        {
            "status": "simulation_complete",
            "wall_particle_transition": transition,
            "control_a_final_state_vs_existing_c5": final_a_check,
            "walls_final_state_vs_existing_c5": final_walls_check,
            "control_b_final_state_sha256": {
                key: _sha256_array(np.asarray(value))
                for key, value in sorted(final_b.items())
            },
            "all_trajectories_complete": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "simulation_complete",
                "protocol_id": protocol_id,
                "control_a_final_exact_vs_existing_c5": final_a_check[
                    "all_exact"
                ],
                "walls_final_exact_vs_existing_c5": final_walls_check[
                    "all_exact"
                ],
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


def _metric_config(cfg: Any) -> dict[str, Any]:
    from clip_deltah_msc_metric import resolve_metric_config

    merged = OmegaConf.merge(cfg.get("substrate", {}), cfg.get("metric", {}))
    values = OmegaConf.to_container(merged, resolve=True)
    metric_args = SimpleNamespace(**values)
    metric_args.rollout_steps = int(cfg.protocol.total_steps)
    metric_args.time_sampling = int(
        int(cfg.protocol.total_steps) // int(cfg.metric.sample_every_steps)
    )
    if getattr(metric_args, "metric_periodic", None) is None:
        metric_args.metric_periodic = False
    if getattr(metric_args, "metric_domain_y", None) is None:
        metric_args.metric_domain_y = float(cfg.substrate.grid_size)
    if getattr(metric_args, "metric_domain_x", None) is None:
        metric_args.metric_domain_x = float(cfg.substrate.grid_size)
    return resolve_metric_config(metric_args)


def _save_metric_map(
    path: Path,
    *,
    summary: dict[str, Any],
    branch: str,
    metric_seed: int,
    metric_fold: int,
    metric_cfg: dict[str, Any],
    protocol_id: str,
) -> None:
    payload: dict[str, Any] = {
        "branch": np.asarray(branch),
        "protocol_id": np.asarray(protocol_id),
        "metric_seed": np.asarray(metric_seed, dtype=np.int64),
        "metric_fold": np.asarray(metric_fold, dtype=np.int32),
        "metric_cfg_json": np.asarray(
            json.dumps(metric_cfg, sort_keys=True, default=_json_default)
        ),
    }
    for key, value in summary.items():
        if isinstance(value, (np.ndarray, np.generic, int, float)):
            payload[key] = np.asarray(value)
    _save_npz(path, compressed=True, **payload)


def score(args: argparse.Namespace) -> None:
    from analysis.history_dependence.trajectory_metrics import (
        compute_delta_h_summary,
    )
    import jax

    trial_dir = _resolve(args.trial_dir)
    output_dir = _resolve(args.output_dir)
    cfg = OmegaConf.load(trial_dir / "resolved_config.yaml")
    audit_path = output_dir / "protocol_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Simulation audit is missing: {audit_path}. Run --phase simulate."
        )
    audit = json.loads(audit_path.read_text())
    if audit.get("status") not in {"simulation_complete", "complete"}:
        raise RuntimeError(
            f"Simulation is not complete: status={audit.get('status')}."
        )
    protocol_id = str(audit["protocol_id"])
    n_particles = int(audit["n_particles"])
    apf = _load_apf_prefix(
        _resolve(cfg.job.control_a_reference_apf_dir),
        n_particles=n_particles,
        end_step=int(cfg.evaluation.training_horizon_steps),
    )
    metric_cfg = _metric_config(cfg)
    metric_seed = int(cfg.job.metric_seed)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for branch, fold, _title in _BRANCHES:
        map_path = maps_dir / f"{branch}_full_length_delta_h.npz"
        if map_path.exists() and not args.force:
            with np.load(map_path, allow_pickle=False) as data:
                if (
                    str(np.asarray(data["protocol_id"]).item())
                    == protocol_id
                ):
                    print(f"[score] {branch}: cache hit", flush=True)
                    summaries[branch] = {
                        "score_scalar": float(
                            np.asarray(data["score_scalar"]).item()
                        ),
                        "tau_best_steps": int(
                            np.asarray(data["tau_best_steps"]).item()
                        ),
                        "map_shape": list(
                            np.asarray(data["delta_h_map"]).shape
                        ),
                    }
                    continue
        steps, xy = _load_branch_trajectory(
            output_dir,
            branch,
            protocol_id=protocol_id,
            apf_prefix=apf if branch == "control_a" else None,
            total_steps=int(cfg.protocol.total_steps),
            sample_every_steps=int(cfg.metric.sample_every_steps),
        )
        print(
            f"[score] {branch}: xy={xy.shape}, backend={jax.default_backend()}",
            flush=True,
        )
        summary = compute_delta_h_summary(
            xy,
            metric_cfg,
            metric_rng_seed=metric_seed,
            metric_rng_fold_in=fold,
            progress_desc=f"full-length {branch}",
            progress_enabled=True,
        )
        _save_metric_map(
            map_path,
            summary=summary,
            branch=branch,
            metric_seed=metric_seed,
            metric_fold=fold,
            metric_cfg=metric_cfg,
            protocol_id=protocol_id,
        )
        summaries[branch] = {
            "score_scalar": float(summary["score_scalar"]),
            "tau_best_steps": int(summary["tau_best_steps"]),
            "map_shape": list(np.asarray(summary["delta_h_map"]).shape),
        }
        del xy
    _write_json(output_dir / "metric_config.json", metric_cfg)
    _write_json(
        output_dir / "metric_summary.json",
        {
            "status": "complete",
            "protocol_id": protocol_id,
            "metric_seed": metric_seed,
            "backend": str(jax.default_backend()),
            "branches": summaries,
        },
    )
    _merge_json(
        audit_path,
        {
            "status": "scoring_complete",
            "metric_backend": str(jax.default_backend()),
            "full_length_metric_summary": summaries,
        },
    )


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (middle[0] - values[0])],
            middle,
            [values[-1] + (values[-1] - middle[-1])],
        )
    )


def _load_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _draw_branch(
    axes: tuple[Any, Any],
    data: dict[str, np.ndarray],
    *,
    title: str,
    vmax: float,
    training_horizon: int,
    walls_removed: int,
) -> Any:
    ax_map, ax_curve = axes
    delta_h = np.asarray(data["delta_h_map"], dtype=np.float64)
    tau = np.asarray(data["tau_steps"], dtype=np.float64)
    starts = np.asarray(data["window_start_steps"], dtype=np.float64)
    image = ax_map.pcolormesh(
        _coordinate_edges(starts / 1000.0),
        _coordinate_edges(tau / 1000.0),
        delta_h,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )
    best_tau = int(np.asarray(data["tau_best_steps"]).item())
    ax_map.axhline(best_tau / 1000.0, color="white", linewidth=1.1)
    for boundary, color in (
        (training_horizon, "#f97316"),
        (walls_removed, "#ef4444"),
    ):
        ax_map.axvline(
            boundary / 1000.0,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
        ax_curve.axvline(
            boundary / 1000.0,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
    ax_map.set_title(
        f"{title} | score={float(data['score_scalar']):.5g}, "
        f"argmax tau={best_tau:,}"
    )
    ax_map.set_ylabel("tau (thousand steps)")
    best = np.asarray(data["delta_h_best"], dtype=np.float64)
    ax_curve.plot(starts / 1000.0, best, color="#0f766e", linewidth=1.0)
    ax_curve.fill_between(
        starts / 1000.0,
        0.0,
        best,
        color="#5eead4",
        alpha=0.28,
    )
    ax_curve.set_xlabel("simulation step (thousands)")
    ax_curve.set_ylabel("Delta-H at selected tau")
    ax_curve.grid(alpha=0.2, linewidth=0.6)
    return image


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial_dir = _resolve(args.trial_dir)
    output_dir = _resolve(args.output_dir)
    cfg = OmegaConf.load(trial_dir / "resolved_config.yaml")
    maps = {
        branch: _load_map(
            output_dir / "maps" / f"{branch}_full_length_delta_h.npz"
        )
        for branch, _fold, _title in _BRANCHES
    }
    finite = np.concatenate(
        [
            np.asarray(data["delta_h_map"], dtype=np.float64).reshape(-1)
            for data in maps.values()
        ]
    )
    finite = finite[np.isfinite(finite)]
    vmax = max(float(np.percentile(finite, 99.5)), 1e-12)
    training_horizon = int(cfg.evaluation.training_horizon_steps)
    walls_removed = int(cfg.protocol.warmup_steps)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    individual_paths: list[str] = []
    for branch, _fold, title in _BRANCHES:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13.0, 6.6),
            gridspec_kw={"height_ratios": [3.0, 1.15]},
            constrained_layout=True,
        )
        image = _draw_branch(
            (axes[0], axes[1]),
            maps[branch],
            title=title,
            vmax=vmax,
            training_horizon=training_horizon,
            walls_removed=walls_removed,
        )
        fig.colorbar(image, ax=axes[0], pad=0.01, label="Delta-H")
        fig.suptitle(
            "Flow-Lenia opt_006: full-length Delta-H\n"
            "orange: optimizer horizon; red: wall removal",
            fontsize=13,
        )
        output = figures_dir / f"{branch}_full_length_delta_h.png"
        fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        individual_paths.append(str(output))

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14.0, 13.0),
        gridspec_kw={"width_ratios": [3.4, 1.3]},
        constrained_layout=True,
    )
    image = None
    for row, (branch, _fold, title) in enumerate(_BRANCHES):
        image = _draw_branch(
            (axes[row, 0], axes[row, 1]),
            maps[branch],
            title=title,
            vmax=vmax,
            training_horizon=training_horizon,
            walls_removed=walls_removed,
        )
    if image is not None:
        fig.colorbar(image, ax=axes[:, 0], pad=0.01, label="Delta-H")
    fig.suptitle(
        "Flow-Lenia opt_006 full-length Delta-H | shared color scale\n"
        "Control A exact optimizer prefix; Control B same init, different "
        "simulation RNG; walls same init as A/B",
        fontsize=14,
    )
    combined_path = figures_dir / "opt_006_full_length_delta_h_all.png"
    fig.savefig(combined_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "status": "complete",
        "shared_vmax_p99_5": vmax,
        "individual_figures": individual_paths,
        "combined_figure": str(combined_path),
    }
    _write_json(output_dir / "plot_summary.json", summary)
    audit_path = output_dir / "protocol_audit.json"
    _merge_json(
        audit_path,
        {
            "status": "complete",
            "figures": summary,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one strongly negative Flow-Lenia C5 optimized run and plot "
            "full-length Delta-H for Control A, same-init Control B, and walls."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("simulate", "score", "plot", "all"),
        default="all",
    )
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-particles", type=int, default=256)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=50_000,
    )
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase in {"simulate", "all"}:
        simulate(args)
    if args.phase in {"score", "all"}:
        score(args)
    if args.phase in {"plot", "all"}:
        plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
