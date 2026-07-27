from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

import flowlenia_rng_sensitivity_experiment as base
from flowlenia_minibang_common import load_config
from flowlenia_minibang_simulate import _make_substrate


PROTOCOL_VERSION = "flowlenia-rng-sensitivity-clip-chamfer-v1"
SOURCE_ROOT = _REPO_ROOT / (
    "analysis/results/flowlenia_rng_sensitivity_"
    "trajectory20_shared4_9branch_10k_v1"
)
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / (
    "analysis/results/flowlenia_rng_sensitivity_clip_chamfer_"
    "trajectory20_shared4_9branch_10k_v1"
)

HORIZONS = tuple(range(1_000, 10_001, 1_000))
SNAPSHOT_INTERVAL = 50
FRAMES_PER_HORIZON = 8
CLIP_MICROBATCH = 10
CLIP_MODEL_NAME = "clip"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
PILOT_CANDIDATE = "run_000_optimized"
PARITY_STEP = 1_000
PARITY_FRAMES = 16
PARITY_MAX_COSINE = 5.0e-6
PARITY_MAX_CHAMFER_DELTA = 2.0e-5

REPLAY_METRICS = (
    "a_relative_l1",
    "p_mass_weighted_l1",
    "render_l1",
    "flow_relative_l1",
    "mass_relative",
    "duplicate_a_max_abs",
    "duplicate_p_max_abs",
    "duplicate_f_max_abs",
    "duplicate_rng_max_abs",
)

CODE_FILES = (
    Path(__file__),
    _REPO_ROOT / "scripts/flowlenia_rng_sensitivity_experiment.py",
    _REPO_ROOT / "scripts/flowlenia_minibang_simulate.py",
    _REPO_ROOT / "scripts/paper_suite_c2_branching.py",
    _REPO_ROOT / "foundation_models/clip.py",
    _REPO_ROOT / "substrates/__init__.py",
    _REPO_ROOT / "substrates/lenia_flow/lenia_flow.py",
    _REPO_ROOT / "substrates/lenia_flow/reintegration_tracking.py",
    _REPO_ROOT / "substrates/lenia_flow/utils.py",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return value.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


def _sample_offsets(horizon: int) -> tuple[int, ...]:
    if horizon <= 0 or horizon % SNAPSHOT_INTERVAL != 0:
        raise ValueError(f"Invalid horizon: {horizon}")
    capture_count = horizon // SNAPSHOT_INTERVAL + 1
    indices = np.linspace(0, capture_count - 1, FRAMES_PER_HORIZON).astype(
        np.int64
    )
    offsets = tuple(int(value * SNAPSHOT_INTERVAL) for value in indices)
    if len(offsets) != FRAMES_PER_HORIZON or offsets[0] != 0 or offsets[-1] != horizon:
        raise RuntimeError(f"Invalid frame schedule for horizon {horizon}: {offsets}")
    return offsets


HORIZON_OFFSETS = {horizon: _sample_offsets(horizon) for horizon in HORIZONS}
CAPTURE_STEPS = tuple(
    sorted({step for offsets in HORIZON_OFFSETS.values() for step in offsets})
)


def _code_fingerprint() -> dict[str, Any]:
    files = {
        str(path.resolve().relative_to(_REPO_ROOT)): _sha256_file(path.resolve())
        for path in CODE_FILES
    }
    return {"files": files, "identity_sha256": _identity_sha256(files)}


def _source_file_identity(source_root: Path) -> dict[str, Any]:
    names = (
        "protocol.json",
        "candidates.csv",
        "contexts.csv",
        "branches.csv",
        "completion_audit.json",
    )
    return {
        name: {
            "path": str((source_root / name).resolve()),
            "sha256": _sha256_file(source_root / name),
        }
        for name in names
    }


def prepare(output_root: Path, source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_audit = json.loads((source_root / "completion_audit.json").read_text())
    source_protocol = json.loads((source_root / "protocol.json").read_text())
    if source_audit.get("status") != "complete":
        raise RuntimeError("Source RNG-sensitivity experiment is not complete")
    if int(source_audit.get("simulation_batches", -1)) != 80:
        raise RuntimeError(f"Unexpected source audit: {source_audit}")
    if tuple(source_protocol["design"]["metric_steps"]) != base.METRIC_STEPS:
        raise RuntimeError("Source metric grid changed")
    if int(source_protocol["design"]["n_unique_rng_branches"]) != 9:
        raise RuntimeError("Source branch protocol changed")

    output_root.mkdir(parents=True, exist_ok=True)
    source_files = _source_file_identity(source_root)
    code = _code_fingerprint()
    design = {
        "source_plan_sha256": source_protocol["plan_sha256"],
        "horizon_steps": base.HORIZON_STEPS,
        "horizons": HORIZONS,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "frames_per_horizon": FRAMES_PER_HORIZON,
        "horizon_offsets": {str(k): value for k, value in HORIZON_OFFSETS.items()},
        "capture_steps": CAPTURE_STEPS,
        "n_capture_steps": len(CAPTURE_STEPS),
        "n_unique_rng_branches": base.N_UNIQUE_BRANCHES,
        "n_total_branches": base.N_BRANCHES,
        "duplicate_branch": base.DUPLICATE_BRANCH,
        "duplicate_of_branch": base.DUPLICATE_OF_BRANCH,
        "trajectory_contexts_per_candidate": 20,
        "shared_contexts_per_candidate": 4,
        "external_state_perturbation": 0.0,
        "clip_microbatch": CLIP_MICROBATCH,
    }
    metric = {
        "render": "clip(sum(A,channels)[...,None] * P[...,:3], 0, 1)",
        "foundation_model": CLIP_MODEL_NAME,
        "model_id": CLIP_MODEL_ID,
        "embedding_normalization": "L2",
        "frame_cost": "cosine distance = 1 - dot(z_i,z_j)",
        "trajectory_distance": (
            "symmetric Chamfer: 0.5 * (mean_i min_j d_ij + "
            "mean_j min_i d_ij)"
        ),
        "context_aggregation": "median over all 36 pairs of 9 unique branches",
        "candidate_aggregation": "mean over 20 visited-C1 contexts",
        "primary_endpoint": "candidate mean trajectory-arm CLIP-Chamfer at 10k",
        "secondary_endpoint": "normalized AUC over horizon grid 0,1k,...,10k",
        "primary_test": "one-sided exact Mann-Whitney optimized > random",
        "inference_mode": "fixed-size Flax CLIP batch inference",
        "exact_frame_canonicalization": (
            "known bit-exact controls reuse one embedding: all branches at t=0; "
            "duplicate branch 9 reuses branch 0 at every capture"
        ),
        "parity_reference": "authoritative unjitted single-frame fm.embed_img",
        "parity_max_cosine_tolerance": PARITY_MAX_COSINE,
        "parity_max_chamfer_delta_tolerance": PARITY_MAX_CHAMFER_DELTA,
    }
    plan_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "source_files": source_files,
        "design": design,
        "metric": metric,
        "simulation_code_bundle_sha256": code["identity_sha256"],
    }
    protocol = {
        **plan_identity,
        "plan_sha256": _identity_sha256(plan_identity),
        "simulation_code_files": code["files"],
        "source_root": str(source_root),
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing != _jsonable(protocol):
            raise RuntimeError(
                f"Existing CLIP protocol does not match current code/design: {protocol_path}"
            )
    else:
        _write_json(protocol_path, protocol)

    for name in ("candidates.csv", "contexts.csv", "branches.csv"):
        target = output_root / name
        if target.exists() and _sha256_file(target) != source_files[name]["sha256"]:
            raise RuntimeError(f"Existing copied manifest changed: {target}")
        if not target.exists():
            shutil.copy2(source_root / name, target)

    report = {
        "status": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "source_plan_sha256": source_protocol["plan_sha256"],
        "candidate_count": len(base._read_csv(output_root / "candidates.csv")),
        "context_count": len(base._read_csv(output_root / "contexts.csv")),
        "capture_steps": len(CAPTURE_STEPS),
    }
    _write_json(output_root / "prepare_audit.json", report)
    return report


def load_protocol(output_root: Path) -> dict[str, Any]:
    path = output_root / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Protocol mismatch in {path}")
    current_code = _code_fingerprint()
    if current_code["identity_sha256"] != protocol["simulation_code_bundle_sha256"]:
        raise RuntimeError("Simulation code changed after protocol was frozen")
    source_root = Path(protocol["source_root"])
    for name, identity in protocol["source_files"].items():
        if _sha256_file(source_root / name) != identity["sha256"]:
            raise RuntimeError(f"Source protocol input changed: {name}")
    return protocol


def _model_fingerprint(fm: Any) -> dict[str, Any]:
    import jax

    digest = hashlib.sha256()
    n_bytes = 0
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(fm.clip_model.params)
    for path, leaf in leaves_with_paths:
        value = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        digest.update("/".join(str(item) for item in path).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value).cast("B"))
        n_bytes += int(value.nbytes)
    identity = {
        "foundation_model": CLIP_MODEL_NAME,
        "model_id": CLIP_MODEL_ID,
        "model_revision": getattr(fm.clip_model.config, "_commit_hash", None),
        "weights_sha256": digest.hexdigest(),
        "parameter_leaves": len(leaves_with_paths),
        "parameter_bytes": n_bytes,
        "model_config": fm.clip_model.config.to_dict(),
        "image_processor_config": fm.processor.image_processor.to_dict(),
        "foundation_wrapper_sha256": _sha256_file(
            _REPO_ROOT / "foundation_models/clip.py"
        ),
        "transformers_version": importlib.metadata.version("transformers"),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _write_or_validate_model_identity(output_root: Path, fm: Any) -> dict[str, Any]:
    identity = _model_fingerprint(fm)
    path = output_root / "clip_model_identity.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != _jsonable(identity):
            raise RuntimeError("Loaded CLIP model does not match frozen model identity")
    else:
        _write_json(path, identity)
    return identity


def _make_rgb_capture(n_contexts: int):
    import jax
    import jax.numpy as jnp

    def capture(states):
        prefix = (n_contexts, base.N_BRANCHES)
        a_value = states["A"].reshape((*prefix, *states["A"].shape[1:]))
        p_value = states["P"].reshape((*prefix, *states["P"].shape[1:]))
        mass = jnp.sum(a_value, axis=-1, keepdims=True)
        return jnp.clip(mass * p_value[..., :3], 0.0, 1.0)

    return jax.jit(capture)


def _make_clip_embedder(fm: Any):
    import jax
    import jax.numpy as jnp

    mean = jnp.asarray(fm.img_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.asarray(fm.img_std, dtype=jnp.float32)[None, None, None, :]

    @jax.jit
    def embed(frames):
        if frames.shape[0] != CLIP_MICROBATCH:
            raise ValueError(f"Expected CLIP batch {CLIP_MICROBATCH}, got {frames.shape}")
        images = jax.image.resize(
            jnp.asarray(frames, dtype=jnp.float32),
            (CLIP_MICROBATCH, 224, 224, 3),
            method="bilinear",
        )
        images = jnp.transpose((images - mean) / std, (0, 3, 1, 2))
        embedding = fm.clip_model.get_image_features(images)
        return embedding / jnp.clip(
            jnp.linalg.norm(embedding, axis=-1, keepdims=True), 1.0e-12
        )

    return embed


def _embed_rgb(
    rgb: Any, embedder: Any, *, step: int
) -> tuple[np.ndarray, dict[str, float]]:
    import jax

    flat = rgb.reshape((-1, *rgb.shape[-3:]))
    if flat.shape[0] != base.SIMULATION_BATCH_SIZE:
        raise RuntimeError(f"Unexpected RGB batch: {flat.shape}")
    queued = [
        embedder(flat[start : start + CLIP_MICROBATCH])
        for start in range(0, flat.shape[0], CLIP_MICROBATCH)
    ]
    values = jax.device_get(queued)
    embeddings = np.concatenate(
        [np.asarray(value, dtype=np.float32) for value in values], axis=0
    ).reshape((base.CONTEXTS_PER_BATCH, base.N_BRANCHES, -1))
    raw_duplicate = float(
        np.max(
            np.abs(
                embeddings[:, base.DUPLICATE_OF_BRANCH]
                - embeddings[:, base.DUPLICATE_BRANCH]
            )
        )
    )
    raw_initial = (
        float(np.max(np.abs(embeddings - embeddings[:, :1])))
        if int(step) == 0
        else 0.0
    )

    # Flax batch GEMMs are not row-invariant at the last few bits. Reuse one
    # embedding wherever the simulation audit guarantees bit-exact RGB input.
    embeddings[:, base.DUPLICATE_BRANCH] = embeddings[
        :, base.DUPLICATE_OF_BRANCH
    ]
    if int(step) == 0:
        embeddings[:] = embeddings[:, :1]
    return embeddings, {
        "raw_duplicate_embedding_max_abs": raw_duplicate,
        "raw_initial_embedding_max_abs": raw_initial,
    }


def _chamfer(z_left: np.ndarray, z_right: np.ndarray) -> float:
    left = np.asarray(z_left, dtype=np.float64)
    right = np.asarray(z_right, dtype=np.float64)
    left /= np.clip(np.linalg.norm(left, axis=-1, keepdims=True), 1.0e-12, None)
    right /= np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1.0e-12, None)
    distance = 1.0 - left @ right.T
    return float(
        0.5
        * (
            np.mean(np.min(distance, axis=1))
            + np.mean(np.min(distance, axis=0))
        )
    )


def _clip_parity(fm: Any, rgb: Any, batch_z: np.ndarray) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    flat = rgb.reshape((-1, *rgb.shape[-3:]))[:PARITY_FRAMES]
    queued = [fm.embed_img(jnp.asarray(flat[index], dtype=jnp.float32)) for index in range(PARITY_FRAMES)]
    single = np.asarray(jax.device_get(queued), dtype=np.float32).reshape(
        (PARITY_FRAMES, -1)
    )
    single /= np.clip(np.linalg.norm(single, axis=-1, keepdims=True), 1.0e-12, None)
    batched = np.asarray(batch_z, dtype=np.float32).reshape((-1, batch_z.shape[-1]))[
        :PARITY_FRAMES
    ]
    batched /= np.clip(np.linalg.norm(batched, axis=-1, keepdims=True), 1.0e-12, None)
    cosine = 1.0 - np.sum(single * batched, axis=-1)
    split = PARITY_FRAMES // 2
    single_chamfer = _chamfer(single[:split], single[split:])
    batch_chamfer = _chamfer(batched[:split], batched[split:])
    return {
        "max_embedding_abs": float(np.max(np.abs(single - batched))),
        "mean_embedding_abs": float(np.mean(np.abs(single - batched))),
        "max_cosine_distance": float(np.max(np.abs(cosine))),
        "mean_cosine_distance": float(np.mean(np.abs(cosine))),
        "single_frame_chamfer": single_chamfer,
        "batch_chamfer": batch_chamfer,
        "chamfer_abs_delta": abs(single_chamfer - batch_chamfer),
    }


def _source_batch_path(source_root: Path, candidate_id: str, batch_idx: int) -> Path:
    return source_root / "simulation" / candidate_id / f"batch_{batch_idx:02d}.npz"


def _output_batch_path(output_root: Path, candidate_id: str, batch_idx: int) -> Path:
    return output_root / "simulation" / candidate_id / f"batch_{batch_idx:02d}.npz"


def _load_context_states(
    substrate: Any,
    args: Any,
    params: Any,
    context_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], np.ndarray]:
    import jax

    latent_template = dict(substrate.init_state(jax.random.PRNGKey(0), params))
    states: list[dict[str, Any]] = []
    hashes: list[str] = []
    snapshot_cache: dict[tuple[Path, int], dict[str, np.ndarray]] = {}
    for row in context_rows:
        if row["arm"] == "trajectory":
            key = (_resolve(row["source_chunk_path"]), int(row["source_step"]))
            snapshot = snapshot_cache.get(key)
            if snapshot is None:
                snapshot = base._load_snapshot(*key)
                snapshot_cache[key] = snapshot
            state = base._replace_physical_state(latent_template, snapshot)
            state_hash = base._physical_state_hash(snapshot)
        else:
            state, state_hash = base._shared_state(
                substrate,
                latent_template,
                params,
                int(row["shared_run_seed"]),
                args,
            )
        states.append(state)
        hashes.append(state_hash)
    return base._tree_stack(states), np.asarray(hashes)


def _simulate_batch(
    *,
    output_root: Path,
    source_root: Path,
    protocol: dict[str, Any],
    model_identity: dict[str, Any],
    substrate: Any,
    args: Any,
    params: Any,
    candidate: dict[str, str],
    context_rows: list[dict[str, str]],
    batch_idx: int,
    stepper: Any,
    metric_capture: Any,
    rgb_capture: Any,
    clip_embedder: Any,
    fm: Any,
    parity_check: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    contexts_state, source_hashes = _load_context_states(
        substrate, args, params, context_rows
    )
    batched_state = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value[:, None], base.N_BRANCHES, axis=1).reshape(
            (base.SIMULATION_BATCH_SIZE, *value.shape[1:])
        ),
        contexts_state,
    )
    rng = jnp.asarray(
        np.stack(
            [
                base._context_rng_key(row, branch_idx)
                for row in context_rows
                for branch_idx in range(base.N_BRANCHES)
            ],
            axis=0,
        ),
        dtype=jnp.uint32,
    )
    params_batch = jnp.repeat(params[None], base.SIMULATION_BATCH_SIZE, axis=0)

    metric_buffers: dict[str, list[np.ndarray]] = {}
    embedding_frames: list[np.ndarray] = []
    raw_duplicate_embedding_max = 0.0
    raw_initial_embedding_max = 0.0
    parity: dict[str, float] | None = None
    metric_step_set = set(base.METRIC_STEPS)
    capture_step_set = set(CAPTURE_STEPS)
    started = time.monotonic()
    for step in range(0, base.HORIZON_STEPS + 1, base.STEP_CHUNK):
        if step in metric_step_set:
            captured = jax.device_get(metric_capture(batched_state, rng))
            for key, value in captured.items():
                metric_buffers.setdefault(key, []).append(np.asarray(value))
        if step in capture_step_set:
            rgb = rgb_capture(batched_state)
            embedding, raw_controls = _embed_rgb(
                rgb, clip_embedder, step=step
            )
            raw_duplicate_embedding_max = max(
                raw_duplicate_embedding_max,
                raw_controls["raw_duplicate_embedding_max_abs"],
            )
            raw_initial_embedding_max = max(
                raw_initial_embedding_max,
                raw_controls["raw_initial_embedding_max_abs"],
            )
            embedding_frames.append(embedding)
            if parity_check and step == PARITY_STEP:
                parity = _clip_parity(fm, rgb, embedding)
        if step < base.HORIZON_STEPS:
            batched_state, rng = stepper(batched_state, rng, params_batch)
    elapsed = time.monotonic() - started

    metrics = {
        key: np.stack(values, axis=1) for key, values in metric_buffers.items()
    }
    embeddings = np.stack(embedding_frames, axis=2).astype(np.float32)
    if embeddings.shape != (
        base.CONTEXTS_PER_BATCH,
        base.N_BRANCHES,
        len(CAPTURE_STEPS),
        512,
    ):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    source_path = _source_batch_path(source_root, candidate["candidate_id"], batch_idx)
    replay_exact: dict[str, bool] = {}
    with np.load(source_path, allow_pickle=False) as source:
        if not np.array_equal(
            source_hashes, np.asarray(source["source_state_hashes"])
        ):
            raise RuntimeError(f"Source state hashes changed for {source_path}")
        for key in REPLAY_METRICS:
            replay_exact[key] = np.array_equal(
                np.asarray(metrics[key]), np.asarray(source[key])
            )
    if not all(replay_exact.values()):
        failed = [key for key, exact in replay_exact.items() if not exact]
        raise RuntimeError(f"L1 replay mismatch for {source_path}: {failed}")

    duplicate_max = float(
        np.max(
            np.abs(
                embeddings[:, base.DUPLICATE_OF_BRANCH]
                - embeddings[:, base.DUPLICATE_BRANCH]
            )
        )
    )
    initial_max = float(
        np.max(np.abs(embeddings[:, :, 0] - embeddings[:, :1, 0]))
    )
    if duplicate_max != 0.0 or initial_max != 0.0:
        raise RuntimeError(
            f"CLIP controls failed: duplicate={duplicate_max}, initial={initial_max}"
        )
    if parity_check:
        if parity is None:
            raise RuntimeError("Pilot CLIP parity was not captured")
        if parity["max_cosine_distance"] > PARITY_MAX_COSINE:
            raise RuntimeError(f"CLIP cosine parity failed: {parity}")
        if parity["chamfer_abs_delta"] > PARITY_MAX_CHAMFER_DELTA:
            raise RuntimeError(f"CLIP Chamfer parity failed: {parity}")

    payload = {
        "protocol_version": np.asarray(PROTOCOL_VERSION),
        "plan_sha256": np.asarray(protocol["plan_sha256"]),
        "simulation_code_bundle_sha256": np.asarray(
            protocol["simulation_code_bundle_sha256"]
        ),
        "clip_model_identity_sha256": np.asarray(model_identity["identity_sha256"]),
        "candidate_id": np.asarray(candidate["candidate_id"]),
        "candidate_kind": np.asarray(candidate["candidate_kind"]),
        "params_sha256": np.asarray(candidate["params_sha256"]),
        "batch_idx": np.asarray(batch_idx, dtype=np.int32),
        "context_indices": np.asarray(
            [int(row["context_idx"]) for row in context_rows], dtype=np.int32
        ),
        "source_state_hashes": source_hashes,
        "source_batch_path": np.asarray(str(source_path.resolve())),
        "source_batch_sha256": np.asarray(_sha256_file(source_path)),
        "capture_steps": np.asarray(CAPTURE_STEPS, dtype=np.int32),
        "clip_embeddings": embeddings,
        "replay_exact": np.asarray(all(replay_exact.values()), dtype=np.bool_),
        "replay_metric_names": np.asarray(REPLAY_METRICS),
        "replay_metric_exact": np.asarray(
            [replay_exact[key] for key in REPLAY_METRICS], dtype=np.bool_
        ),
        "duplicate_embedding_max_abs": np.asarray(duplicate_max, dtype=np.float64),
        "initial_embedding_max_abs": np.asarray(initial_max, dtype=np.float64),
        "raw_duplicate_embedding_max_abs": np.asarray(
            raw_duplicate_embedding_max, dtype=np.float64
        ),
        "raw_initial_embedding_max_abs": np.asarray(
            raw_initial_embedding_max, dtype=np.float64
        ),
        "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
        "parity_checked": np.asarray(parity is not None, dtype=np.bool_),
        "parity_max_embedding_abs": np.asarray(
            np.nan if parity is None else parity["max_embedding_abs"], dtype=np.float64
        ),
        "parity_max_cosine_distance": np.asarray(
            np.nan if parity is None else parity["max_cosine_distance"],
            dtype=np.float64,
        ),
        "parity_chamfer_abs_delta": np.asarray(
            np.nan if parity is None else parity["chamfer_abs_delta"],
            dtype=np.float64,
        ),
    }
    return payload


def _validate_batch(
    path: Path,
    *,
    candidate: dict[str, str],
    batch_idx: int,
    expected_context_indices: list[int],
    protocol: dict[str, Any],
    model_identity_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            checks = (
                str(np.asarray(data["protocol_version"]).item()) == PROTOCOL_VERSION,
                str(np.asarray(data["plan_sha256"]).item()) == protocol["plan_sha256"],
                str(np.asarray(data["simulation_code_bundle_sha256"]).item())
                == protocol["simulation_code_bundle_sha256"],
                str(np.asarray(data["clip_model_identity_sha256"]).item())
                == model_identity_sha256,
                str(np.asarray(data["candidate_id"]).item())
                == candidate["candidate_id"],
                str(np.asarray(data["params_sha256"]).item())
                == candidate["params_sha256"],
                int(np.asarray(data["batch_idx"]).item()) == batch_idx,
                np.array_equal(
                    np.asarray(data["context_indices"], dtype=np.int32),
                    np.asarray(expected_context_indices, dtype=np.int32),
                ),
                np.array_equal(
                    np.asarray(data["capture_steps"], dtype=np.int32),
                    np.asarray(CAPTURE_STEPS, dtype=np.int32),
                ),
                np.asarray(data["clip_embeddings"]).shape
                == (
                    base.CONTEXTS_PER_BATCH,
                    base.N_BRANCHES,
                    len(CAPTURE_STEPS),
                    512,
                ),
                bool(np.asarray(data["replay_exact"]).item()),
                float(np.asarray(data["duplicate_embedding_max_abs"]).item()) == 0.0,
                float(np.asarray(data["initial_embedding_max_abs"]).item()) == 0.0,
                np.all(np.isfinite(np.asarray(data["clip_embeddings"]))),
            )
            source_path = Path(str(np.asarray(data["source_batch_path"]).item()))
            source_valid = (
                source_path.exists()
                and _sha256_file(source_path)
                == str(np.asarray(data["source_batch_sha256"]).item())
            )
            return all(checks) and source_valid
    except Exception:
        return False


def simulate(
    output_root: Path,
    *,
    candidate_ids: set[str] | None,
    pilot: bool,
) -> dict[str, Any]:
    protocol = load_protocol(output_root)
    source_root = Path(protocol["source_root"])
    candidates = base._read_csv(output_root / "candidates.csv")
    contexts = base._read_csv(output_root / "contexts.csv")
    if pilot:
        candidate_ids = {PILOT_CANDIDATE}
    if candidate_ids is not None:
        known = {row["candidate_id"] for row in candidates}
        unknown = candidate_ids.difference(known)
        if unknown:
            raise RuntimeError(f"Unknown candidates: {sorted(unknown)}")
        candidates = [row for row in candidates if row["candidate_id"] in candidate_ids]
    if not candidates:
        raise RuntimeError("No candidates selected")

    config_path = _resolve(
        json.loads((source_root / "protocol.json").read_text())["input_files"][
            "simulation_config"
        ]
    )
    _cfg, flat = load_config(config_path)
    args = SimpleNamespace(**dict(OmegaConf.to_container(flat, resolve=True)))

    import foundation_models
    import jax
    import jax.numpy as jnp

    substrate = _make_substrate(args)
    first_params = np.asarray(
        np.load(_resolve(candidates[0]["params_path"])), dtype=np.float32
    )
    _ = substrate.init_state(jax.random.PRNGKey(0), jnp.asarray(first_params))
    stepper = base._make_stepper(substrate, base.SIMULATION_BATCH_SIZE)
    metric_capture = base._make_metric_capture(base.CONTEXTS_PER_BATCH)
    rgb_capture = _make_rgb_capture(base.CONTEXTS_PER_BATCH)

    fm = foundation_models.create_foundation_model(CLIP_MODEL_NAME)
    model_identity = _write_or_validate_model_identity(output_root, fm)
    clip_embedder = _make_clip_embedder(fm)

    completed_before = 0
    durations: list[float] = []
    started = time.monotonic()
    for candidate_pos, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["candidate_id"]
        candidate_contexts = sorted(
            [row for row in contexts if row["candidate_id"] == candidate_id],
            key=lambda row: int(row["context_idx"]),
        )
        if len(candidate_contexts) != base.N_CONTEXTS:
            raise RuntimeError(
                f"{candidate_id} has {len(candidate_contexts)} contexts"
            )
        params_np = np.asarray(
            np.load(_resolve(candidate["params_path"])), dtype=np.float32
        )
        if params_np.shape != (int(substrate.n_params),):
            raise RuntimeError(f"Unexpected params shape for {candidate_id}: {params_np.shape}")
        params = jnp.asarray(params_np)

        for batch_idx in range(base.N_CONTEXTS // base.CONTEXTS_PER_BATCH):
            batch_contexts = candidate_contexts[
                batch_idx
                * base.CONTEXTS_PER_BATCH : (batch_idx + 1)
                * base.CONTEXTS_PER_BATCH
            ]
            expected_indices = [int(row["context_idx"]) for row in batch_contexts]
            output_path = _output_batch_path(output_root, candidate_id, batch_idx)
            if _validate_batch(
                output_path,
                candidate=candidate,
                batch_idx=batch_idx,
                expected_context_indices=expected_indices,
                protocol=protocol,
                model_identity_sha256=model_identity["identity_sha256"],
            ):
                completed_before += 1
                print(f"[skip] {candidate_id} batch={batch_idx} valid", flush=True)
                continue
            if output_path.exists():
                quarantine = output_path.with_suffix(
                    output_path.suffix + f".invalid-{int(time.time())}"
                )
                shutil.move(output_path, quarantine)
                print(f"[quarantine] {output_path} -> {quarantine}", flush=True)

            print(
                f"[simulate+clip] {candidate_id} ({candidate_pos}/{len(candidates)}) "
                f"batch={batch_idx} contexts={expected_indices[0]}..{expected_indices[-1]}",
                flush=True,
            )
            payload = _simulate_batch(
                output_root=output_root,
                source_root=source_root,
                protocol=protocol,
                model_identity=model_identity,
                substrate=substrate,
                args=args,
                params=params,
                candidate=candidate,
                context_rows=batch_contexts,
                batch_idx=batch_idx,
                stepper=stepper,
                metric_capture=metric_capture,
                rgb_capture=rgb_capture,
                clip_embedder=clip_embedder,
                fm=fm,
                parity_check=(
                    candidate_id == PILOT_CANDIDATE and batch_idx == 0
                ),
            )
            _save_npz(output_path, payload)
            if not _validate_batch(
                output_path,
                candidate=candidate,
                batch_idx=batch_idx,
                expected_context_indices=expected_indices,
                protocol=protocol,
                model_identity_sha256=model_identity["identity_sha256"],
            ):
                raise RuntimeError(f"Post-write validation failed: {output_path}")
            duration = float(np.asarray(payload["elapsed_seconds"]).item())
            durations.append(duration)
            finished = completed_before + len(durations)
            requested_total = len(candidates) * 2
            rate = len(durations) / max(time.monotonic() - started, 1.0e-9)
            eta = (requested_total - finished) / max(rate, 1.0e-9)
            progress = {
                "status": "running",
                "last_candidate": candidate_id,
                "last_batch": batch_idx,
                "completed_in_request": len(durations),
                "skipped_valid_in_request": completed_before,
                "requested_batches": requested_total,
                "elapsed_seconds": time.monotonic() - started,
                "eta_seconds": max(0.0, eta),
            }
            _write_json(output_root / "simulation_progress.json", progress)
            print(
                f"[done] {candidate_id} batch={batch_idx} {duration:.1f}s "
                f"request={finished}/{requested_total} eta={max(0.0, eta) / 60:.1f}m",
                flush=True,
            )

    result = {
        "status": "complete_for_request",
        "candidate_count": len(candidates),
        "completed_new_batches": len(durations),
        "reused_valid_batches": completed_before,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_root / "last_simulation_request.json", result)
    return result


def audit(output_root: Path, *, require_complete: bool) -> dict[str, Any]:
    protocol = load_protocol(output_root)
    model_path = output_root / "clip_model_identity.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model_identity = json.loads(model_path.read_text())
    candidates = base._read_csv(output_root / "candidates.csv")
    contexts = base._read_csv(output_root / "contexts.csv")
    context_by_candidate = {
        candidate["candidate_id"]: sorted(
            [row for row in contexts if row["candidate_id"] == candidate["candidate_id"]],
            key=lambda row: int(row["context_idx"]),
        )
        for candidate in candidates
    }
    valid = 0
    missing: list[str] = []
    invalid: list[str] = []
    replay_exact = 0
    duplicate_max = 0.0
    initial_max = 0.0
    raw_duplicate_max = 0.0
    raw_initial_max = 0.0
    parity_rows: list[dict[str, float]] = []
    elapsed = 0.0
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        for batch_idx in range(2):
            expected = [
                int(row["context_idx"])
                for row in context_by_candidate[candidate_id][
                    batch_idx
                    * base.CONTEXTS_PER_BATCH : (batch_idx + 1)
                    * base.CONTEXTS_PER_BATCH
                ]
            ]
            path = _output_batch_path(output_root, candidate_id, batch_idx)
            if not path.exists():
                missing.append(str(path))
                continue
            if not _validate_batch(
                path,
                candidate=candidate,
                batch_idx=batch_idx,
                expected_context_indices=expected,
                protocol=protocol,
                model_identity_sha256=model_identity["identity_sha256"],
            ):
                invalid.append(str(path))
                continue
            valid += 1
            with np.load(path, allow_pickle=False) as data:
                replay_exact += int(bool(np.asarray(data["replay_exact"]).item()))
                duplicate_max = max(
                    duplicate_max,
                    float(np.asarray(data["duplicate_embedding_max_abs"]).item()),
                )
                initial_max = max(
                    initial_max,
                    float(np.asarray(data["initial_embedding_max_abs"]).item()),
                )
                raw_duplicate_max = max(
                    raw_duplicate_max,
                    float(
                        np.asarray(data["raw_duplicate_embedding_max_abs"]).item()
                    ),
                )
                raw_initial_max = max(
                    raw_initial_max,
                    float(np.asarray(data["raw_initial_embedding_max_abs"]).item()),
                )
                elapsed += float(np.asarray(data["elapsed_seconds"]).item())
                if bool(np.asarray(data["parity_checked"]).item()):
                    parity_rows.append(
                        {
                            "max_embedding_abs": float(
                                np.asarray(data["parity_max_embedding_abs"]).item()
                            ),
                            "max_cosine_distance": float(
                                np.asarray(data["parity_max_cosine_distance"]).item()
                            ),
                            "chamfer_abs_delta": float(
                                np.asarray(data["parity_chamfer_abs_delta"]).item()
                            ),
                        }
                    )
    expected_total = len(candidates) * 2
    complete = valid == expected_total and not missing and not invalid
    status = "complete" if complete else "partial"
    report = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol["simulation_code_bundle_sha256"],
        "model_identity_sha256": model_identity["identity_sha256"],
        "candidate_count": len(candidates),
        "expected_batches": expected_total,
        "valid_batches": valid,
        "missing_outputs": missing,
        "invalid_outputs": invalid,
        "replay_exact_batches": replay_exact,
        "max_duplicate_embedding_abs": duplicate_max,
        "max_initial_embedding_abs": initial_max,
        "max_raw_batch_duplicate_embedding_abs": raw_duplicate_max,
        "max_raw_batch_initial_embedding_abs": raw_initial_max,
        "clip_parity_checks": parity_rows,
        "summed_batch_seconds": elapsed,
    }
    _write_json(output_root / "simulation_audit.json", report)
    if require_complete and not complete:
        raise RuntimeError(f"CLIP simulation is incomplete: {report}")
    if parity_rows:
        if max(row["max_cosine_distance"] for row in parity_rows) > PARITY_MAX_COSINE:
            raise RuntimeError(f"CLIP parity audit failed: {parity_rows}")
        if max(row["chamfer_abs_delta"] for row in parity_rows) > PARITY_MAX_CHAMFER_DELTA:
            raise RuntimeError(f"CLIP Chamfer parity audit failed: {parity_rows}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay FlowLenia RNG forks and capture CLIP embeddings."
    )
    parser.add_argument(
        "phase", choices=("prepare", "pilot", "simulate", "audit")
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _resolve(args.output_root)
    source_root = _resolve(args.source_root)
    if args.phase == "prepare":
        result = prepare(output_root, source_root)
    elif args.phase == "pilot":
        prepare(output_root, source_root)
        result = simulate(output_root, candidate_ids=None, pilot=True)
        pilot_audit = audit(output_root, require_complete=False)
        if pilot_audit["valid_batches"] < 2 or not pilot_audit["clip_parity_checks"]:
            raise RuntimeError(f"Pilot audit failed: {pilot_audit}")
        _write_json(
            output_root / "pilot_audit.json",
            {**pilot_audit, "status": "passed"},
        )
    elif args.phase == "simulate":
        result = simulate(
            output_root,
            candidate_ids=set(args.candidate) if args.candidate else None,
            pilot=False,
        )
    else:
        result = audit(output_root, require_complete=bool(args.require_complete))
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
