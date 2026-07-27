#!/usr/bin/env python3
"""Targeted mass-preserving wall diagnostic for one selected C5 point.

This script deliberately writes outside the canonical C5 result tree.  It
replays the three branches of run_003/optimized/point_00 with the frozen C5
RNG and batch topology, but projects A after every hard-wall mask so each
isolated block preserves its previous per-channel mass.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import imageio.v2 as imageio
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import flowlenia_c5_branch_frustration as c5
from flowlenia_c5_branch_analysis import (
    _embedding_chamfer_cosine,
    _field_components,
    _font,
    _rgb_u8,
    _wall_lines,
    _with_field_pyramid,
)
from paper_check_frustration_batch_eval import _mask_block_spatial_state


RUN_IDX = 3
CANDIDATE_ID = "run_003_optimized"
POINT_ID = 0
EXPECTED_ROW_IDS = (540, 541, 542)
CAPTURE_RELATIVE_STEPS = np.asarray(
    [0, 2850, 5700, 8550, 10_000, 11_400, 14_250, 17_100, 20_000],
    dtype=np.int64,
)
COMMON_RELATIVE_STEPS = np.asarray(
    [0, 2850, 5700, 8550, 11_400, 14_250, 17_100, 20_000],
    dtype=np.int64,
)
DEFAULT_ROOT = (
    c5.DEFAULT_OUTPUT_ROOT
    / "selected_examples"
    / "mass_preserving_wall_probe_run_003_optimized_point_00"
)
PROBE_VERSION = "flowlenia-c5-mass-preserving-wall-probe-v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selected_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = c5._resolve(c5.DEFAULT_OUTPUT_ROOT)
    plan_path = root / "paired_plan.csv"
    protocol_path = root / "protocol.json"
    rows = c5._read_csv(plan_path)
    protocol = json.loads(protocol_path.read_text())
    if c5._plan_identity_hash(rows) != protocol["plan_sha256"]:
        raise RuntimeError("Canonical paired plan hash failed")
    selected = [
        row
        for row in rows
        if c5._as_int(row["run_idx"]) == RUN_IDX
        and row["candidate_id"] == CANDIDATE_ID
        and c5._as_int(row["point_id"]) == POINT_ID
    ]
    selected.sort(key=lambda row: c5._as_int(row["branch_id"]))
    row_ids = tuple(c5._as_int(row["row_id"]) for row in selected)
    branch_ids = tuple(c5._as_int(row["branch_id"]) for row in selected)
    if row_ids != EXPECTED_ROW_IDS or branch_ids != (0, 1, 2):
        raise RuntimeError(
            f"Selected rows changed: row_ids={row_ids}, branches={branch_ids}"
        )
    return selected, protocol


def _make_block_state_stepper(
    block_substrate: Any,
    *,
    n_blocks: int,
    original_batch_size: int,
    valid_mask: Any,
    geometry: c5.SimulationGeometry,
    mutation_spec: c5.GlobalMutationSpec,
    block_rt_gumbel: Any,
    project_mass: bool,
) -> Callable[[int], Callable[..., Any]]:
    """Clone the canonical block stepper with one optional post-mask operation."""

    cache: dict[int, Any] = {}
    valid_mask = jnp.asarray(valid_mask, dtype=bool)
    expanded_valid_mask = valid_mask[..., None]

    def get(n_steps: int):
        n_steps = int(n_steps)
        if n_steps in cache:
            return cache[n_steps]

        @jax.jit
        def step(
            state_in: Any,
            subkeys: Any,
            params_in: Any,
            original_batch_index: Any,
        ) -> Any:
            selected = c5._selected_step_keys(
                subkeys,
                original_batch_index,
                n_steps=n_steps,
                original_batch_size=original_batch_size,
            )

            def one_lane(
                lane_key: Any,
                lane_state: Any,
                lane_params: Any,
            ) -> Any:
                block_keys = jnp.broadcast_to(lane_key, (n_blocks, 2))
                next_state = jax.vmap(
                    lambda state, key, gumbel: (
                        block_substrate.step_state_with_reintegration_gumbel(
                            key,
                            state,
                            lane_params,
                            gumbel,
                        )
                    )
                )(lane_state, block_keys, block_rt_gumbel)
                mutation_delta = c5._global_mutation_delta(
                    lane_key,
                    spec=mutation_spec,
                    dtype=next_state["P"].dtype,
                )
                mutation_blocks = c5._partition_global_field(
                    mutation_delta,
                    geometry=geometry,
                )
                next_state = {
                    **next_state,
                    "P": next_state["P"] + mutation_blocks,
                }
                next_state = _mask_block_spatial_state(
                    next_state,
                    valid_mask,
                )
                if project_mass:
                    target = jnp.sum(
                        lane_state["A"],
                        axis=(1, 2),
                        keepdims=True,
                    )
                    retained = jnp.sum(
                        next_state["A"],
                        axis=(1, 2),
                        keepdims=True,
                    )
                    scale = jnp.where(
                        target <= 0.0,
                        1.0,
                        target / jnp.maximum(retained, 1.0e-20),
                    )
                    projected_a = next_state["A"] * scale
                    projected_a = jnp.where(
                        expanded_valid_mask,
                        projected_a,
                        jnp.zeros((), dtype=projected_a.dtype),
                    )
                    next_state = {**next_state, "A": projected_a}
                return next_state

            vmapped_lane = jax.vmap(one_lane, in_axes=(0, 0, 0))

            def body(state: Any, keys: Any):
                return vmapped_lane(keys, state, params_in), None

            return jax.lax.scan(body, state_in, selected)[0]

        cache[n_steps] = step
        return step

    return get


def _prepare_audit_batch(
    rows: Sequence[dict[str, str]],
    engine: dict[str, Any],
) -> dict[str, Any]:
    runtime = engine["runtime"]
    unpadded = [
        c5._load_simulation_item(
            row,
            runtime["substrate"],
            snapshot_cache=engine["snapshot_cache"],
            params_cache=engine["params_cache"],
            state_template_cache=engine["state_template_cache"],
        )
        for row in rows
    ]
    items, real_n = c5._pad_items(
        unpadded,
        c5.SIMULATION_BATCH_SIZE,
    )
    for item in items:
        item["args"] = runtime["args"]
    block_state, roundtrip = c5._prepare_block_state_batch(items, runtime)
    if not roundtrip["all_ap_exact"]:
        raise RuntimeError("Initial split/merge is not exact")
    return {
        "items": items,
        "real_n": real_n,
        "state": block_state,
        "rng": jnp.stack(
            [jnp.asarray(item["rng"]) for item in items],
            axis=0,
        ),
        "params": jnp.stack(
            [
                jnp.asarray(item["params"], dtype=jnp.float32)
                for item in items
            ],
            axis=0,
        ),
        "indices": jnp.asarray(
            [item["original_batch_index"] for item in items],
            dtype=jnp.int32,
        ),
    }


def _tree_exact(left: Any, right: Any) -> dict[str, Any]:
    left_paths, _ = jax.tree_util.tree_flatten_with_path(left)
    right_paths, _ = jax.tree_util.tree_flatten_with_path(right)
    if [str(path) for path, _ in left_paths] != [
        str(path) for path, _ in right_paths
    ]:
        raise RuntimeError("Tree paths differ")
    fields = {}
    all_exact = True
    for (path, left_leaf), (_, right_leaf) in zip(
        left_paths,
        right_paths,
        strict=True,
    ):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        exact = bool(np.array_equal(left_array, right_array))
        max_abs = (
            float(
                np.max(
                    np.abs(
                        left_array.astype(np.float64)
                        - right_array.astype(np.float64)
                    )
                )
            )
            if left_array.size
            else 0.0
        )
        fields[str(path)] = {"exact": exact, "max_abs": max_abs}
        all_exact = all_exact and exact
    return {"all_exact": all_exact, "fields": fields}


def _stepper_audit(
    rows: Sequence[dict[str, str]],
    engine: dict[str, Any],
    clone_stepper: Callable[[int], Callable[..., Any]],
    projected_stepper: Callable[[int], Callable[..., Any]],
) -> dict[str, Any]:
    batch = _prepare_audit_batch(rows, engine)
    _next_rng, subkeys = c5._split_rng_batch(batch["rng"])
    canonical = engine["block_state_stepper"](c5.JIT_MICROBATCH)(
        batch["state"],
        subkeys,
        batch["params"],
        batch["indices"],
    )
    clone = clone_stepper(c5.JIT_MICROBATCH)(
        batch["state"],
        subkeys,
        batch["params"],
        batch["indices"],
    )
    exactness = _tree_exact(canonical, clone)
    if not exactness["all_exact"]:
        raise RuntimeError("Unmodified probe stepper differs from canonical stepper")

    projected = projected_stepper(c5.JIT_MICROBATCH)(
        batch["state"],
        subkeys,
        batch["params"],
        batch["indices"],
    )
    initial_mass = np.asarray(
        jax.device_get(jnp.sum(batch["state"]["A"], axis=(2, 3))),
        dtype=np.float64,
    )
    projected_mass = np.asarray(
        jax.device_get(jnp.sum(projected["A"], axis=(2, 3))),
        dtype=np.float64,
    )
    scale = np.maximum(np.abs(initial_mass), 1.0e-12)
    relative_error = np.abs(projected_mass - initial_mass) / scale
    return {
        "status": "passed",
        "steps": c5.JIT_MICROBATCH,
        "outer_batch_size": c5.SIMULATION_BATCH_SIZE,
        "real_rows": len(rows),
        "clone_vs_canonical": exactness,
        "projection_mass_error": {
            "max_abs": float(np.max(np.abs(projected_mass - initial_mass))),
            "max_relative": float(np.max(relative_error)),
            "median_relative": float(np.median(relative_error)),
        },
    }


def _save_corrected_branch(
    root: Path,
    item: dict[str, Any],
    captures: list[dict[str, Any]],
    relative_steps: np.ndarray,
    *,
    roundtrip: dict[str, Any],
    batch_context: dict[str, Any],
) -> Path:
    row = item["row"]
    branch_id = c5._as_int(row["branch_id"])
    path = root / "branches" / f"branch_{branch_id:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    states = {
        key: np.stack(
            [np.asarray(capture["state"][key]) for capture in captures],
            axis=0,
        )
        for key in ("A", "P", "F")
    }
    mass_by_channel = np.sum(
        states["A"].astype(np.float64),
        axis=(1, 2),
    )
    mass_total = np.sum(mass_by_channel, axis=-1)
    absolute_steps = relative_steps + c5._as_int(row["step"])
    np.savez_compressed(
        path,
        steps=absolute_steps,
        relative_steps=relative_steps,
        A=states["A"].astype(np.float16),
        P=states["P"].astype(np.float16),
        F=states["F"].astype(np.float16),
        mass_by_channel_float32=mass_by_channel,
        mass_total_float32=mass_total,
        resume_batch_rng_key=np.stack(
            [
                np.asarray(capture["rng"], dtype=np.uint32)
                for capture in captures
            ],
            axis=0,
        ),
        row_id=np.asarray(c5._as_int(row["row_id"]), dtype=np.int32),
        branch_id=np.asarray(branch_id, dtype=np.int32),
        branch_seed=np.asarray(
            c5._as_int(row["branch_seed"]),
            dtype=np.int64,
        ),
    )
    _write_json(
        path.with_suffix(".metadata.json"),
        {
            "probe_version": PROBE_VERSION,
            "canonical_row_id": c5._as_int(row["row_id"]),
            "canonical_free_branch_dir": row["free_branch_dir"],
            "canonical_absorbing_wall_branch_dir": row["walls_branch_dir"],
            "capture_relative_steps": relative_steps,
            "mass_projection": (
                "after every confined step and hard mask, scale A separately "
                "within every block and channel to its previous-step mass"
            ),
            "P_and_F_projection": "none",
            "split_merge_roundtrip": roundtrip,
            "outer_batch": batch_context,
            "artifact_sha256": _sha256_file(path),
        },
    )
    return path


def _run_corrected(
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    root: Path,
    engine: dict[str, Any],
    projected_stepper: Callable[[int], Callable[..., Any]],
) -> tuple[dict[str, Any], dict[int, Path]]:
    written: dict[int, Path] = {}

    def writer(
        item: dict[str, Any],
        captures: list[dict[str, Any]],
        relative_steps: np.ndarray,
        *,
        protocol: dict[str, Any],
        roundtrip: dict[str, Any],
        geometry: c5.SimulationGeometry,
        batch_context: dict[str, Any],
    ) -> None:
        del protocol, geometry
        branch_id = c5._as_int(item["row"]["branch_id"])
        written[branch_id] = _save_corrected_branch(
            root,
            item,
            captures,
            relative_steps,
            roundtrip=roundtrip,
            batch_context=batch_context,
        )

    old_writer = c5._write_wall_branch
    old_capture = c5._capture_steps_from_free
    c5._write_wall_branch = writer
    c5._capture_steps_from_free = lambda _row: CAPTURE_RELATIVE_STEPS.copy()
    engine["block_state_stepper"] = projected_stepper
    try:
        result = c5._run_simulation_batch(
            rows,
            protocol=protocol,
            batch_size=c5.SIMULATION_BATCH_SIZE,
            mode="walls",
            output_root=root,
            shared_engine=engine,
        )
    finally:
        c5._write_wall_branch = old_writer
        c5._capture_steps_from_free = old_capture
    if tuple(sorted(written)) != (0, 1, 2):
        raise RuntimeError(f"Missing corrected branches: {sorted(written)}")
    return result, written


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _common_indices(relative: np.ndarray) -> np.ndarray:
    lookup = {int(value): idx for idx, value in enumerate(relative)}
    missing = [
        int(value)
        for value in COMMON_RELATIVE_STEPS
        if int(value) not in lookup
    ]
    if missing:
        raise RuntimeError(f"Missing common capture offsets: {missing}")
    return np.asarray(
        [lookup[int(value)] for value in COMMON_RELATIVE_STEPS],
        dtype=np.int64,
    )


def _load_canonical_embeddings(
    rows: Sequence[dict[str, str]],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    manifest_path = (
        c5._resolve(c5.DEFAULT_OUTPUT_ROOT) / "embedding_manifest.csv"
    )
    with manifest_path.open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    lookup = {
        (int(row["row_id"]), row["arm"]): Path(row["embedding_cache"])
        for row in manifest
    }
    free = []
    absorbing = []
    sources = []
    for row in rows:
        row_id = c5._as_int(row["row_id"])
        free_path = lookup[(row_id, "free")]
        absorbing_path = lookup[(row_id, "walls")]
        free.append(_load_npz(free_path)["z"].astype(np.float32))
        absorbing.append(_load_npz(absorbing_path)["z"].astype(np.float32))
        sources.extend(
            [
                {
                    "row_id": row_id,
                    "arm": "free",
                    "path": free_path,
                    "sha256": _sha256_file(free_path),
                },
                {
                    "row_id": row_id,
                    "arm": "absorbing_walls",
                    "path": absorbing_path,
                    "sha256": _sha256_file(absorbing_path),
                },
            ]
        )
    return free, absorbing, {
        "manifest": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "sources": sources,
    }


def _embed_corrected(
    corrected: Sequence[dict[str, np.ndarray]],
    corrected_indices: np.ndarray,
    free_z: Sequence[np.ndarray],
    root: Path,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    import foundation_models

    started = time.monotonic()
    model = foundation_models.create_foundation_model("clip")
    output = []
    calls = 0
    for branch_id, arrays in enumerate(corrected):
        rgb = c5_analysis_render(
            arrays["A"][corrected_indices],
            arrays["P"][corrected_indices],
        )
        branch_z = []
        for frame_idx, frame in enumerate(rgb):
            if frame_idx == 0:
                z = np.asarray(free_z[branch_id][0], dtype=np.float32)
            else:
                z = np.asarray(
                    jax.device_get(
                        model.embed_img(
                            jnp.asarray(
                                np.ascontiguousarray(frame, dtype=np.float32)
                            )
                        )
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                calls += 1
            z = z / max(float(np.linalg.norm(z)), 1.0e-12)
            branch_z.append(z)
        output.append(np.stack(branch_z, axis=0))
        print(
            f"[clip] corrected branch {branch_id}/2 complete",
            flush=True,
        )
    path = root / "corrected_clip_embeddings.npz"
    np.savez_compressed(
        path,
        z=np.stack(output, axis=0),
        relative_steps=COMMON_RELATIVE_STEPS,
        inference_mode=np.asarray(
            "authoritative_c2_unjitted_single_frame"
        ),
        model_id=np.asarray("openai/clip-vit-base-patch32"),
    )
    return output, {
        "model_id": "openai/clip-vit-base-patch32",
        "inference_mode": "authoritative_c2_unjitted_single_frame",
        "new_model_calls": calls,
        "t0_embedding_reused_from_bitwise_identical_free_frame": True,
        "elapsed_seconds": time.monotonic() - started,
        "artifact": path,
        "artifact_sha256": _sha256_file(path),
    }


def c5_analysis_render(a_value: np.ndarray, p_value: np.ndarray) -> np.ndarray:
    from paper_suite_c2_branching import _render_apf_rgb

    return _render_apf_rgb({"A": a_value, "P": p_value})


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _within_metrics(
    z_values: Sequence[np.ndarray],
    fields: Sequence[dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, float]:
    clip = []
    clip_sync = []
    field = []
    for left, right in combinations(range(3), 2):
        clip.append(
            _embedding_chamfer_cosine(
                z_values[left][indices],
                z_values[right][indices],
            )
        )
        clip_sync.append(
            float(
                np.mean(
                    np.clip(
                        1.0
                        - np.sum(
                            z_values[left][indices]
                            * z_values[right][indices],
                            axis=-1,
                        ),
                        0.0,
                        2.0,
                    )
                )
            )
        )
        field.append(
            _field_components(
                fields[left],
                fields[right],
                indices,
            )[0]
        )
    return {
        "clip_chamfer": _median(clip),
        "clip_synchronized": _median(clip_sync),
        "field": _median(field),
    }


def _paired_metrics(
    left_z: Sequence[np.ndarray],
    right_z: Sequence[np.ndarray],
    left_fields: Sequence[dict[str, Any]],
    right_fields: Sequence[dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, float]:
    clip = []
    clip_sync = []
    field = []
    for branch_id in range(3):
        clip.append(
            _embedding_chamfer_cosine(
                left_z[branch_id][indices],
                right_z[branch_id][indices],
            )
        )
        clip_sync.append(
            float(
                np.mean(
                    np.clip(
                        1.0
                        - np.sum(
                            left_z[branch_id][indices]
                            * right_z[branch_id][indices],
                            axis=-1,
                        ),
                        0.0,
                        2.0,
                    )
                )
            )
        )
        field.append(
            _field_components(
                left_fields[branch_id],
                right_fields[branch_id],
                indices,
            )[0]
        )
    return {
        "clip_chamfer": _median(clip),
        "clip_synchronized": _median(clip_sync),
        "field": _median(field),
    }


def _make_video_frame(
    rgb_by_arm: dict[str, Sequence[np.ndarray]],
    *,
    frame_idx: int,
    relative_step: int,
    mass_by_arm: dict[str, Sequence[np.ndarray]],
) -> np.ndarray:
    tile_size = 220
    header = 54
    left = 178
    gap = 4
    arms = (
        ("free", "Free", (0, 110, 80)),
        ("absorbing", "Absorbing walls", (165, 45, 115)),
        ("projected", "Mass-projected walls", (35, 95, 170)),
    )
    canvas = Image.new(
        "RGB",
        (
            left + 3 * tile_size + 2 * gap,
            header + 3 * tile_size + 2 * gap,
        ),
        color=(245, 247, 249),
    )
    draw = ImageDraw.Draw(canvas)
    phase = "walls active" if relative_step <= c5.WALL_STEPS else "walls removed"
    draw.text(
        (10, 7),
        (
            f"{CANDIDATE_ID} | point {POINT_ID:02d} | "
            f"+{relative_step:,} steps | {phase}"
        ),
        fill=(20, 25, 30),
        font=_font(17, bold=True),
    )
    for branch_id in range(3):
        x = left + branch_id * (tile_size + gap)
        draw.text(
            (x + 72, 33),
            f"Branch {branch_id}",
            fill=(45, 50, 56),
            font=_font(13),
        )
    for arm_idx, (key, label, color) in enumerate(arms):
        y = header + arm_idx * (tile_size + gap)
        draw.text(
            (8, y + 80),
            label,
            fill=color,
            font=_font(16, bold=True),
        )
        median_mass = float(
            np.median(
                [
                    mass_by_arm[key][branch_id][frame_idx]
                    for branch_id in range(3)
                ]
            )
        )
        draw.text(
            (8, y + 108),
            f"median mass {median_mass:,.1f}",
            fill=(65, 70, 76),
            font=_font(12),
        )
        for branch_id in range(3):
            x = left + branch_id * (tile_size + gap)
            image = Image.fromarray(
                rgb_by_arm[key][branch_id][frame_idx]
            ).resize(
                (tile_size, tile_size),
                resample=Image.Resampling.NEAREST,
            )
            if key != "free" and relative_step <= c5.WALL_STEPS:
                _wall_lines(image)
            canvas.paste(image, (x, y))
    return np.asarray(canvas)


def _create_visuals(
    root: Path,
    relative: np.ndarray,
    free_arrays: Sequence[dict[str, np.ndarray]],
    absorbing_arrays: Sequence[dict[str, np.ndarray]],
    corrected_arrays: Sequence[dict[str, np.ndarray]],
    corrected_indices: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rgb = {
        "free": [
            _rgb_u8({"A": item["A"], "P": item["P"]})
            for item in free_arrays
        ],
        "absorbing": [
            _rgb_u8({"A": item["A"], "P": item["P"]})
            for item in absorbing_arrays
        ],
        "projected": [
            _rgb_u8(
                {
                    "A": item["A"][corrected_indices],
                    "P": item["P"][corrected_indices],
                }
            )
            for item in corrected_arrays
        ],
    }
    mass = {
        "free": [
            np.sum(item["A"].astype(np.float64), axis=(1, 2, 3))
            for item in free_arrays
        ],
        "absorbing": [
            np.sum(item["A"].astype(np.float64), axis=(1, 2, 3))
            for item in absorbing_arrays
        ],
        "projected": [
            item["mass_total_float32"][corrected_indices]
            for item in corrected_arrays
        ],
    }

    video_path = root / "free_vs_absorbing_vs_mass_projected.mp4"
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio.get_writer(
        video_path,
        fps=24,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        pixelformat="yuv420p",
    )
    try:
        for frame_idx, rel_step in enumerate(relative):
            frame = _make_video_frame(
                rgb,
                frame_idx=frame_idx,
                relative_step=int(rel_step),
                mass_by_arm=mass,
            )
            for _ in range(12):
                writer.append_data(frame)
    finally:
        writer.close()

    reader = imageio.get_reader(video_path)
    decoded = 0
    shape = None
    try:
        for frame in reader:
            decoded += 1
            shape = tuple(int(value) for value in frame.shape)
    finally:
        reader.close()
    if decoded != len(relative) * 12:
        raise RuntimeError(f"Video decode count differs: {decoded}")

    mass_rows = []
    for arm in ("free", "absorbing", "projected"):
        for branch_id in range(3):
            for frame_idx, rel_step in enumerate(relative):
                mass_rows.append(
                    {
                        "arm": arm,
                        "branch_id": branch_id,
                        "relative_step": int(rel_step),
                        "total_mass": float(mass[arm][branch_id][frame_idx]),
                    }
                )
    mass_csv = root / "mass_trajectory_common_captures.csv"
    with mass_csv.open("w", newline="") as stream:
        writer_csv = csv.DictWriter(
            stream,
            fieldnames=tuple(mass_rows[0]),
        )
        writer_csv.writeheader()
        writer_csv.writerows(mass_rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        constrained_layout=True,
    )
    colors = {
        "free": "#009E73",
        "absorbing": "#CC79A7",
        "projected": "#0072B2",
    }
    labels = {
        "free": "Free",
        "absorbing": "Absorbing walls",
        "projected": "Mass-projected walls",
    }
    for arm in ("free", "absorbing", "projected"):
        values = np.stack(mass[arm], axis=0)
        axes[0].plot(
            relative,
            np.median(values, axis=0),
            marker="o",
            linewidth=2,
            label=labels[arm],
            color=colors[arm],
        )
        axes[0].fill_between(
            relative,
            np.min(values, axis=0),
            np.max(values, axis=0),
            color=colors[arm],
            alpha=0.12,
        )
    axes[0].axvline(c5.WALL_STEPS, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Relative simulation step",
        ylabel="Total A mass",
        title="Mass over the selected replay",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    within = metrics["post_release"]["within_branch"]
    names = ("free", "absorbing_walls", "mass_projected_walls")
    axes[1].bar(
        np.arange(3),
        [within[name]["clip_chamfer"] for name in names],
        color=[colors["free"], colors["absorbing"], colors["projected"]],
    )
    axes[1].set_xticks(
        np.arange(3),
        ["Free", "Absorbing\nwalls", "Mass-projected\nwalls"],
    )
    axes[1].set(
        ylabel="Within-arm CLIP Chamfer",
        title="Post-release branch divergence",
    )
    axes[1].grid(axis="y", alpha=0.2)
    figure_path = root / "mass_and_post_release_divergence.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "video": video_path,
        "video_sha256": _sha256_file(video_path),
        "video_frames": decoded,
        "video_shape": shape,
        "video_fps": 24,
        "display_frames_per_real_snapshot": 12,
        "mass_csv": mass_csv,
        "mass_csv_sha256": _sha256_file(mass_csv),
        "figure": figure_path,
        "figure_sha256": _sha256_file(figure_path),
    }


def _analyze(
    rows: list[dict[str, str]],
    corrected_paths: dict[int, Path],
    root: Path,
) -> dict[str, Any]:
    free_arrays = []
    absorbing_arrays = []
    corrected_arrays = []
    for row in rows:
        free_arrays.append(
            c5._branch_arrays(
                c5._resolve(row["free_branch_dir"]),
                keys={
                    "A",
                    "P",
                    "F",
                    "resume_batch_rng_key",
                },
            )
        )
        absorbing_arrays.append(
            c5._branch_arrays(
                c5._resolve(row["walls_branch_dir"]),
                keys={
                    "A",
                    "P",
                    "F",
                    "resume_batch_rng_key",
                },
            )
        )
        corrected_arrays.append(
            _load_npz(corrected_paths[c5._as_int(row["branch_id"])])
        )

    free_relative = (
        np.asarray(free_arrays[0]["steps"], dtype=np.int64)
        - c5._as_int(rows[0]["step"])
    )
    if not np.array_equal(free_relative, COMMON_RELATIVE_STEPS):
        raise RuntimeError(f"Canonical free offsets changed: {free_relative}")
    corrected_relative = np.asarray(
        corrected_arrays[0]["relative_steps"],
        dtype=np.int64,
    )
    corrected_indices = _common_indices(corrected_relative)

    initial_exact = {}
    rng_exact = {}
    for branch_id in range(3):
        initial_exact[branch_id] = {
            key: bool(
                np.array_equal(
                    free_arrays[branch_id][key][0],
                    corrected_arrays[branch_id][key][corrected_indices[0]],
                )
            )
            for key in ("A", "P", "F")
        }
        rng_exact[branch_id] = bool(
            np.array_equal(
                free_arrays[branch_id]["resume_batch_rng_key"],
                corrected_arrays[branch_id]["resume_batch_rng_key"][
                    corrected_indices
                ],
            )
        )
    if not all(all(item.values()) for item in initial_exact.values()):
        raise RuntimeError(f"Corrected/free t0 mismatch: {initial_exact}")
    if not all(rng_exact.values()):
        raise RuntimeError(f"Corrected/free RNG capture mismatch: {rng_exact}")

    free_z, absorbing_z, embedding_sources = _load_canonical_embeddings(rows)
    corrected_z, corrected_embedding = _embed_corrected(
        corrected_arrays,
        corrected_indices,
        free_z,
        root,
    )

    free_fields = [
        _with_field_pyramid({"A": item["A"], "P": item["P"]})
        for item in free_arrays
    ]
    absorbing_fields = [
        _with_field_pyramid({"A": item["A"], "P": item["P"]})
        for item in absorbing_arrays
    ]
    corrected_fields = [
        _with_field_pyramid(
            {
                "A": item["A"][corrected_indices],
                "P": item["P"][corrected_indices],
            }
        )
        for item in corrected_arrays
    ]
    post = np.flatnonzero(COMMON_RELATIVE_STEPS > c5.WALL_STEPS)
    within = {
        "free": _within_metrics(free_z, free_fields, post),
        "absorbing_walls": _within_metrics(
            absorbing_z,
            absorbing_fields,
            post,
        ),
        "mass_projected_walls": _within_metrics(
            corrected_z,
            corrected_fields,
            post,
        ),
    }
    paired = {
        "free_vs_absorbing_walls": _paired_metrics(
            free_z,
            absorbing_z,
            free_fields,
            absorbing_fields,
            post,
        ),
        "free_vs_mass_projected_walls": _paired_metrics(
            free_z,
            corrected_z,
            free_fields,
            corrected_fields,
            post,
        ),
    }
    scores = {
        "free_minus_absorbing_within_clip": (
            within["free"]["clip_chamfer"]
            - within["absorbing_walls"]["clip_chamfer"]
        ),
        "free_minus_mass_projected_within_clip": (
            within["free"]["clip_chamfer"]
            - within["mass_projected_walls"]["clip_chamfer"]
        ),
        "absorbing_primary_excess_clip": (
            paired["free_vs_absorbing_walls"]["clip_chamfer"]
            - within["free"]["clip_chamfer"]
        ),
        "mass_projected_primary_excess_clip": (
            paired["free_vs_mass_projected_walls"]["clip_chamfer"]
            - within["free"]["clip_chamfer"]
        ),
    }

    mass_summary = {}
    release_idx = int(
        np.flatnonzero(corrected_relative == c5.WALL_STEPS)[0]
    )
    for label, arrays, indices in (
        (
            "free",
            free_arrays,
            np.arange(len(COMMON_RELATIVE_STEPS)),
        ),
        (
            "absorbing_walls",
            absorbing_arrays,
            np.arange(len(COMMON_RELATIVE_STEPS)),
        ),
        (
            "mass_projected_walls",
            corrected_arrays,
            corrected_indices,
        ),
    ):
        branch_mass = []
        for item in arrays:
            if label == "mass_projected_walls":
                values = item["mass_total_float32"][indices]
            else:
                values = np.sum(
                    item["A"][indices].astype(np.float64),
                    axis=(1, 2, 3),
                )
            branch_mass.append(values)
        branch_mass_array = np.stack(branch_mass, axis=0)
        loss = (
            1.0
            - branch_mass_array[:, -1] / branch_mass_array[:, 0]
        )
        mass_summary[label] = {
            "median_by_common_step": np.median(
                branch_mass_array,
                axis=0,
            ),
            "median_fraction_lost_at_20000": float(np.median(loss)),
            "branch_fraction_lost_at_20000": loss,
        }
    corrected_release_mass = np.asarray(
        [
            item["mass_total_float32"][release_idx]
            for item in corrected_arrays
        ],
        dtype=np.float64,
    )
    corrected_start_mass = np.asarray(
        [
            item["mass_total_float32"][0]
            for item in corrected_arrays
        ],
        dtype=np.float64,
    )
    mass_summary["mass_projected_walls"][
        "branch_fraction_lost_at_release"
    ] = 1.0 - corrected_release_mass / corrected_start_mass

    metrics = {
        "probe_version": PROBE_VERSION,
        "selected_identity": {
            "run_idx": RUN_IDX,
            "candidate_id": CANDIDATE_ID,
            "point_id": POINT_ID,
            "row_ids": EXPECTED_ROW_IDS,
            "absolute_step": c5._as_int(rows[0]["step"]),
            "condition": rows[0]["condition"],
            "branch_seeds": [
                c5._as_int(row["branch_seed"]) for row in rows
            ],
        },
        "initial_state_exact_float16_vs_free": initial_exact,
        "top_level_rng_exact_vs_free_at_common_captures": rng_exact,
        "post_release": {
            "relative_steps": COMMON_RELATIVE_STEPS[post],
            "within_branch": within,
            "paired_same_seed": paired,
            "scores": scores,
        },
        "mass": mass_summary,
        "canonical_embedding_sources": embedding_sources,
        "corrected_embedding": corrected_embedding,
    }
    visuals = _create_visuals(
        root,
        COMMON_RELATIVE_STEPS,
        free_arrays,
        absorbing_arrays,
        corrected_arrays,
        corrected_indices,
        metrics,
    )
    metrics["visuals"] = visuals
    _write_json(root / "metrics.json", metrics)
    return metrics


def main() -> int:
    root = c5._resolve(DEFAULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    rows, protocol = _load_selected_rows()
    engine = c5._create_wall_engine(rows[0])
    runtime = engine["runtime"]
    clone_stepper = _make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=engine["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
        project_mass=False,
    )
    projected_stepper = _make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=engine["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
        project_mass=True,
    )

    started = time.monotonic()
    print("[probe] auditing unmodified stepper clone", flush=True)
    stepper_audit = _stepper_audit(
        rows,
        engine,
        clone_stepper,
        projected_stepper,
    )
    _write_json(root / "stepper_audit.json", stepper_audit)
    print("[probe] running mass-projected replay", flush=True)
    run_result, paths = _run_corrected(
        rows,
        protocol,
        root,
        engine,
        projected_stepper,
    )
    print("[probe] computing matched metrics and visuals", flush=True)
    metrics = _analyze(rows, paths, root)
    summary = {
        "status": "complete",
        "probe_version": PROBE_VERSION,
        "canonical_c5_untouched": True,
        "output_root": root,
        "stepper_audit": stepper_audit,
        "simulation": run_result,
        "key_results": metrics["post_release"],
        "mass": metrics["mass"],
        "visuals": metrics["visuals"],
        "elapsed_seconds": time.monotonic() - started,
        "script": Path(__file__).resolve(),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    _write_json(root / "summary.json", summary)
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
