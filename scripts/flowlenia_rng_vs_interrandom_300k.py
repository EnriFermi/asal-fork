from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

import flowlenia_rng_sensitivity_experiment as old_base
from flowlenia_minibang_common import load_config
from flowlenia_minibang_simulate import _make_substrate


PROTOCOL_VERSION = "flowlenia-rng-vs-interrandom-shared-init-300k-v1"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / (
    "analysis/results/"
    "flowlenia_rng_vs_interrandom_shared_init_8rng_dup_300k_v1"
)
SOURCE_BASE_ROOT = _REPO_ROOT / (
    "analysis/results/"
    "flowlenia_rng_sensitivity_trajectory20_shared4_9branch_10k_v1"
)
SOURCE_CLIP_ROOT = _REPO_ROOT / (
    "analysis/results/"
    "flowlenia_rng_sensitivity_clip_chamfer_"
    "trajectory20_shared4_9branch_10k_v1"
)

HORIZON_STEPS = 300_000
PREFLIGHT_STEPS = 1_000
STEP_CHUNK = 50
CHECKPOINT_INTERVAL = 50_000
PROGRESS_INTERVAL = 5_000
SHARED_CONTEXT_INDICES = (20, 21, 22, 23)
N_CONTEXTS = 4
N_UNIQUE_BRANCHES = 8
N_BRANCHES = 9
DUPLICATE_BRANCH = 8
DUPLICATE_OF_BRANCH = 0
BATCH_SIZE = N_CONTEXTS * N_BRANCHES
CLIP_MICROBATCH = N_BRANCHES
CLIP_MODEL_NAME = "clip"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
AUDIT_CANDIDATE = "run_000_random_000"

POINT_STEPS = tuple(range(0, HORIZON_STEPS + 1, 1_000))
STATE_METRIC_STEPS = POINT_STEPS
CHAMFER_HORIZONS = (
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_000,
    7_000,
    8_000,
    9_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    300_000,
)
FRAMES_PER_CHAMFER = 8


def _sample_offsets(horizon: int) -> tuple[int, ...]:
    positions = np.linspace(
        0, horizon // STEP_CHUNK, FRAMES_PER_CHAMFER
    ).astype(np.int64)
    values = tuple(int(value * STEP_CHUNK) for value in positions)
    if values[0] != 0 or values[-1] != horizon:
        raise RuntimeError(f"Invalid Chamfer offsets for {horizon}: {values}")
    return values


CHAMFER_OFFSETS = {
    horizon: _sample_offsets(horizon) for horizon in CHAMFER_HORIZONS
}
EMBED_STEPS = POINT_STEPS
PAIR_LEFT = np.asarray(
    [
        left
        for left in range(N_UNIQUE_BRANCHES)
        for right in range(left + 1, N_UNIQUE_BRANCHES)
    ],
    dtype=np.int32,
)
PAIR_RIGHT = np.asarray(
    [
        right
        for left in range(N_UNIQUE_BRANCHES)
        for right in range(left + 1, N_UNIQUE_BRANCHES)
    ],
    dtype=np.int32,
)

CODE_FILES = (
    Path(__file__),
    _REPO_ROOT / "scripts/flowlenia_rng_sensitivity_experiment.py",
    _REPO_ROOT / "scripts/flowlenia_minibang_simulate.py",
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


def _sha256_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _identity_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def _save_npz(path: Path, payload: dict[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
    os.replace(tmp, path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args,
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _code_fingerprint() -> dict[str, Any]:
    files = {
        str(path.relative_to(_REPO_ROOT)): _sha256_file(path)
        for path in CODE_FILES
    }
    return {"files": files, "bundle_sha256": _identity_sha256(files)}


def _hardware_identity() -> dict[str, Any]:
    import jax

    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
        "devices": devices,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _source_identities() -> dict[str, Any]:
    paths = {
        "base_protocol": SOURCE_BASE_ROOT / "protocol.json",
        "base_candidates": SOURCE_BASE_ROOT / "candidates.csv",
        "base_contexts": SOURCE_BASE_ROOT / "contexts.csv",
        "base_branches": SOURCE_BASE_ROOT / "branches.csv",
        "clip_protocol": SOURCE_CLIP_ROOT / "protocol.json",
        "clip_model_identity": SOURCE_CLIP_ROOT / "clip_model_identity.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source provenance: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _simulation_identity(flat_args: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "save_dir",
        "output_dir",
        "output",
        "video_fps",
        "codec",
        "macro_block_size",
        "snapshots_per_file",
        "compress",
        "save_A",
        "save_F",
        "save_rgb",
        "save_lagrangian",
        "save_lagrangian_channels",
        "compute_metrics",
    }
    return {key: value for key, value in flat_args.items() if key not in ignored}


def prepare(output_root: Path) -> dict[str, Any]:
    source_files = _source_identities()
    old_protocol = json.loads((SOURCE_BASE_ROOT / "protocol.json").read_text())
    old_candidates = _read_csv(SOURCE_BASE_ROOT / "candidates.csv")
    old_contexts = _read_csv(SOURCE_BASE_ROOT / "contexts.csv")
    candidates = [
        dict(row) for row in old_candidates if row["candidate_kind"] == "random"
    ]
    expected_candidate_ids = [
        f"run_{run_idx:03d}_random_{candidate_idx:03d}"
        for run_idx in range(10)
        for candidate_idx in range(3)
    ]
    if [row["candidate_id"] for row in candidates] != expected_candidate_ids:
        raise RuntimeError("Expected 30 random candidates in canonical order")
    for row in candidates:
        params_path = _resolve(row["params_path"])
        if _sha256_file(params_path) != row["params_sha256"]:
            raise RuntimeError(f"Parameter hash mismatch: {row['candidate_id']}")

    contexts: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = sorted(
            [
                row
                for row in old_contexts
                if row["candidate_id"] == candidate["candidate_id"]
                and int(row["context_idx"]) in SHARED_CONTEXT_INDICES
            ],
            key=lambda row: int(row["context_idx"]),
        )
        if [int(row["context_idx"]) for row in rows] != list(
            SHARED_CONTEXT_INDICES
        ):
            raise RuntimeError(
                f"Missing shared contexts for {candidate['candidate_id']}"
            )
        for local_idx, row in enumerate(rows):
            contexts.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "context_idx": local_idx,
                    "source_context_idx": int(row["context_idx"]),
                    "shared_run_seed": int(row["shared_run_seed"]),
                    "rollout_seed_idx": int(row["rollout_seed_idx"]),
                    "anchor_idx": int(row["anchor_idx"]),
                    "arm": "shared_state",
                    "source_step": 0,
                }
            )

    branches = [
        {
            "branch_idx": branch_idx,
            "effective_branch_idx": (
                DUPLICATE_OF_BRANCH
                if branch_idx == DUPLICATE_BRANCH
                else branch_idx
            ),
            "branch_seed": (
                old_base.BRANCH_SEED_BASE + DUPLICATE_OF_BRANCH
                if branch_idx == DUPLICATE_BRANCH
                else old_base.BRANCH_SEED_BASE + branch_idx
            ),
            "duplicate_of": (
                DUPLICATE_OF_BRANCH if branch_idx == DUPLICATE_BRANCH else ""
            ),
            "scientific_rng_branch": branch_idx < N_UNIQUE_BRANCHES,
        }
        for branch_idx in range(N_BRANCHES)
    ]

    config_path = _resolve(old_protocol["input_files"]["simulation_config"])
    if _sha256_file(config_path) != old_protocol["input_files"][
        "simulation_config_sha256"
    ]:
        raise RuntimeError("Source simulation config changed")
    _cfg, flat = load_config(config_path)
    flat_args = dict(OmegaConf.to_container(flat, resolve=True))
    if str(flat_args.get("substrate")) != "lenia_flow":
        raise RuntimeError("Expected substrate=lenia_flow")
    if float(flat_args.get("sigma", np.nan)) != 0.2:
        raise RuntimeError(f"Expected sigma=0.2, got {flat_args.get('sigma')}")
    if float(flat_args.get("flow_sigma", np.nan)) != 0.2:
        raise RuntimeError(
            f"Expected flow_sigma=0.2, got {flat_args.get('flow_sigma')}"
        )
    if str(flat_args.get("mix_rule")) != "stoch":
        raise RuntimeError("Expected stochastic FlowLenia mixing")

    code = _code_fingerprint()
    plan_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "candidates": candidates,
        "contexts": contexts,
        "branches": branches,
        "source_files": source_files,
        "simulation_config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "identity": _simulation_identity(flat_args),
        },
        "code_bundle_sha256": code["bundle_sha256"],
        "design": {
            "question": (
                "At fixed random parameters and exact initial state, does "
                "continuation-RNG divergence reach the divergence between "
                "independently sampled random parameters?"
            ),
            "n_random_candidates": 30,
            "n_shared_initial_states": N_CONTEXTS,
            "shared_initial_state_source": (
                "bit-exact source contexts 20..23 from the prior fixed-context "
                "RNG-sensitivity experiment"
            ),
            "n_unique_rng_branches": N_UNIQUE_BRANCHES,
            "n_total_lanes_per_state": N_BRANCHES,
            "duplicate_branch": DUPLICATE_BRANCH,
            "duplicate_of_branch": DUPLICATE_OF_BRANCH,
            "lanes_per_candidate_batch": BATCH_SIZE,
            "batches_per_candidate": 1,
            "horizon_steps": HORIZON_STEPS,
            "no_early_stopping": True,
            "step_chunk": STEP_CHUNK,
            "restart_checkpoint_interval": CHECKPOINT_INTERVAL,
            "point_steps": POINT_STEPS,
            "state_metric_steps": STATE_METRIC_STEPS,
            "embedding_steps": EMBED_STEPS,
            "trajectory_window_steps": 20_000,
            "trajectory_frame_interval": 1_000,
            "trajectory_frames_per_window": 20,
            "clip_microbatch": CLIP_MICROBATCH,
            "repeat_audit_candidate": AUDIT_CANDIDATE,
        },
        "metrics": {
            "primary_frame_distance": (
                "CLIP cosine distance at identical absolute simulation step"
            ),
            "within_candidate": (
                "pairwise distance over the 28 pairs of 8 unique RNG branches"
            ),
            "between_random_candidates_primary": (
                "distance over all 435 random-candidate pairs using matched shared "
                "initial state and matched unique RNG branch"
            ),
            "time_matched_ratio": (
                "within-candidate RNG CLIP-Chamfer divided by inter-random "
                "CLIP-Chamfer in the same 20k window"
            ),
            "harness": (
                "distance between branch 0 and branch 8, which receive identical "
                "initial state, parameters, and RNG key in separate batch lanes"
            ),
            "secondary_state_metric": "mass-normalized A-field L1",
            "secondary_trajectory_metric": (
                "symmetric CLIP-Chamfer over 8 frames from 0 through each horizon"
            ),
            "crossing": (
                "computed post hoc; no simulation stopping depends on a crossing"
            ),
            "harness_interpretation": (
                "No subtraction. Horizons where harness is comparable with "
                "within-RNG distance are flagged as attribution-limited."
            ),
        },
        "rng_protocol": {
            "unique_branch_keys": (
                "exact same branch-key construction and branch indices 0..7 as "
                "the prior 10k fixed-context experiment"
            ),
            "duplicate": "branch 8 receives the exact branch-0 uint32 key",
            "common_random_numbers": (
                "shared initial-state and branch keys are matched across all "
                "random parameter candidates"
            ),
        },
    }
    protocol = {
        **plan_identity,
        "plan_sha256": _identity_sha256(plan_identity),
        "code_files": code["files"],
        "git_state_at_prepare": _git_state(),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != _jsonable(protocol):
            raise RuntimeError(
                f"Existing protocol differs; use a new output root: {output_root}"
            )
    else:
        _write_csv(output_root / "candidates.csv", candidates, list(candidates[0]))
        _write_csv(output_root / "contexts.csv", contexts, list(contexts[0]))
        _write_csv(output_root / "branches.csv", branches, list(branches[0]))
        _write_json(protocol_path, protocol)

    report = {
        "status": "ready",
        "plan_sha256": protocol["plan_sha256"],
        "candidate_count": len(candidates),
        "context_count": len(contexts),
        "branch_count": len(branches),
        "batch_size": BATCH_SIZE,
        "horizon_steps": HORIZON_STEPS,
        "point_count": len(POINT_STEPS),
        "embedding_capture_count": len(EMBED_STEPS),
        "state_metric_count": len(STATE_METRIC_STEPS),
    }
    _write_json(output_root / "prepare_audit.json", report)
    return protocol


def load_protocol(output_root: Path) -> dict[str, Any]:
    path = output_root / "protocol.json"
    if not path.exists():
        return prepare(output_root)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Unexpected protocol in {path}")
    current = _code_fingerprint()
    if current["bundle_sha256"] != protocol["code_bundle_sha256"]:
        raise RuntimeError("Simulation code changed after protocol was frozen")
    for identity in protocol["source_files"].values():
        path = Path(identity["path"])
        if _sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"Source provenance changed: {path}")
    return protocol


def _model_fingerprint(fm: Any) -> dict[str, Any]:
    import jax

    digest = hashlib.sha256()
    n_bytes = 0
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(
        fm.clip_model.params
    )
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


def _write_or_validate_identity(
    output_root: Path, name: str, identity: dict[str, Any]
) -> None:
    path = output_root / name
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != _jsonable(identity):
            raise RuntimeError(f"Runtime identity changed: {path}")
    else:
        _write_json(path, identity)


def _branch_rng_key(row: dict[str, str], branch_idx: int) -> np.ndarray:
    effective = (
        DUPLICATE_OF_BRANCH if branch_idx == DUPLICATE_BRANCH else branch_idx
    )
    old_row = {
        "arm": "shared_state",
        "rollout_seed_idx": row["rollout_seed_idx"],
        "anchor_idx": row["anchor_idx"],
    }
    return old_base._context_rng_key(old_row, effective)


def _physical_state_hash(state: dict[str, Any]) -> str:
    import jax

    host = jax.device_get(state)
    return _sha256_arrays(
        np.asarray(host["A"]),
        np.asarray(host["P"]),
        np.asarray(host["F"]),
        np.asarray(host["Food"]),
        np.asarray(host["t"]),
        np.asarray(host["mass_cycle_start"]),
    )


def _build_initial_batch(
    substrate: Any,
    args: Any,
    params: Any,
    contexts: list[dict[str, str]],
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    latent_template = dict(substrate.init_state(jax.random.PRNGKey(0), params))
    states: list[dict[str, Any]] = []
    legacy_hashes: list[str] = []
    full_hashes: list[str] = []
    for row in contexts:
        state, legacy_hash = old_base._shared_state(
            substrate,
            latent_template,
            params,
            int(row["shared_run_seed"]),
            args,
        )
        states.append(state)
        legacy_hashes.append(legacy_hash)
        full_hashes.append(_physical_state_hash(state))
    context_state = old_base._tree_stack(states)
    batched_state = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value[:, None], N_BRANCHES, axis=1).reshape(
            (BATCH_SIZE, *value.shape[1:])
        ),
        context_state,
    )
    rng = jnp.asarray(
        np.stack(
            [
                _branch_rng_key(row, branch_idx)
                for row in contexts
                for branch_idx in range(N_BRANCHES)
            ],
            axis=0,
        ),
        dtype=jnp.uint32,
    )
    return (
        batched_state,
        rng,
        np.asarray(legacy_hashes),
        np.asarray(full_hashes),
    )


def _make_stepper(substrate: Any):
    import jax
    import jax.numpy as jnp

    def advance(states_in, rng_in, params_in):
        split = jax.vmap(lambda key: jax.random.split(key, 2))(rng_in)
        rng_out = split[:, 0]
        chunk_keys = split[:, 1]
        per_lane = jax.vmap(lambda key: jax.random.split(key, STEP_CHUNK))(
            chunk_keys
        )
        per_step = jnp.swapaxes(per_lane, 0, 1)

        def scan_body(states, keys):
            next_states = jax.vmap(substrate.step_state)(keys, states, params_in)
            return next_states, None

        states_out, _ = jax.lax.scan(scan_body, states_in, per_step)
        return states_out, rng_out

    return jax.jit(advance)


def _make_rgb_capture():
    import jax
    import jax.numpy as jnp

    def capture(states):
        a_value = states["A"].reshape(
            (N_CONTEXTS, N_BRANCHES, *states["A"].shape[1:])
        )
        p_value = states["P"].reshape(
            (N_CONTEXTS, N_BRANCHES, *states["P"].shape[1:])
        )
        mass = jnp.sum(a_value, axis=-1, keepdims=True)
        return jnp.clip(mass * p_value[..., :3], 0.0, 1.0)

    return jax.jit(capture)


def _make_state_metric_capture():
    import jax
    import jax.numpy as jnp

    left = jnp.asarray(PAIR_LEFT)
    right = jnp.asarray(PAIR_RIGHT)

    def capture(states, rng):
        a_value = states["A"].reshape(
            (N_CONTEXTS, N_BRANCHES, *states["A"].shape[1:])
        )
        a_left = a_value[:, left]
        a_right = a_value[:, right]
        numerator = jnp.sum(
            jnp.abs(a_left - a_right), axis=(-3, -2, -1)
        )
        denominator = 0.5 * (
            jnp.sum(jnp.abs(a_left), axis=(-3, -2, -1))
            + jnp.sum(jnp.abs(a_right), axis=(-3, -2, -1))
        )
        pair_l1 = numerator / jnp.maximum(denominator, 1.0e-12)

        duplicate_left = a_value[:, DUPLICATE_OF_BRANCH]
        duplicate_right = a_value[:, DUPLICATE_BRANCH]
        duplicate_num = jnp.sum(
            jnp.abs(duplicate_left - duplicate_right), axis=(-3, -2, -1)
        )
        duplicate_den = 0.5 * (
            jnp.sum(jnp.abs(duplicate_left), axis=(-3, -2, -1))
            + jnp.sum(jnp.abs(duplicate_right), axis=(-3, -2, -1))
        )
        duplicate_l1 = duplicate_num / jnp.maximum(duplicate_den, 1.0e-12)
        duplicate_max = jnp.max(
            jnp.abs(duplicate_left - duplicate_right), axis=(-3, -2, -1)
        )
        total_mass = jnp.sum(a_value, axis=(-3, -2, -1))
        rng_view = rng.reshape((N_CONTEXTS, N_BRANCHES, 2))
        duplicate_rng_max = jnp.max(
            jnp.abs(
                rng_view[:, DUPLICATE_OF_BRANCH].astype(jnp.int64)
                - rng_view[:, DUPLICATE_BRANCH].astype(jnp.int64)
            ),
            axis=-1,
        )
        return {
            "a_pair_relative_l1": pair_l1,
            "a_duplicate_relative_l1": duplicate_l1,
            "a_duplicate_max_abs": duplicate_max,
            "total_mass": total_mass,
            "duplicate_rng_max_abs": duplicate_rng_max,
        }

    return jax.jit(capture)


def _make_clip_embedder(fm: Any):
    import jax
    import jax.numpy as jnp

    mean = jnp.asarray(fm.img_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.asarray(fm.img_std, dtype=jnp.float32)[None, None, None, :]

    @jax.jit
    def embed(frames):
        if frames.shape[0] != CLIP_MICROBATCH:
            raise ValueError(
                f"Expected CLIP batch {CLIP_MICROBATCH}, got {frames.shape}"
            )
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


def _capture_embeddings(rgb: Any, embedder: Any) -> np.ndarray:
    import jax

    queued = [embedder(rgb[context_idx]) for context_idx in range(N_CONTEXTS)]
    values = np.asarray(jax.device_get(queued), dtype=np.float32)
    if values.shape != (N_CONTEXTS, N_BRANCHES, 512):
        raise RuntimeError(f"Unexpected CLIP embedding shape: {values.shape}")
    return values


def _candidate_output_path(
    output_root: Path, candidate_id: str, *, audit: bool
) -> Path:
    directory = "audit_repeat" if audit else "simulation"
    return output_root / directory / candidate_id / "trajectory.npz"


def _candidate_checkpoint_path(
    output_root: Path, candidate_id: str, *, audit: bool
) -> Path:
    directory = "audit_repeat" if audit else "simulation"
    return output_root / directory / candidate_id / "restart_state.npz"


def _checkpoint_payload(
    *,
    protocol: dict[str, Any],
    candidate: dict[str, str],
    step: int,
    state: dict[str, Any],
    rng: Any,
    source_state_hashes: np.ndarray,
    source_state_full_hashes: np.ndarray,
    embedding_steps_done: list[int],
    embeddings: list[np.ndarray],
    metric_steps_done: list[int],
    metrics: dict[str, list[np.ndarray]],
    elapsed_seconds: float,
    audit: bool,
) -> dict[str, Any]:
    import jax

    host_state = jax.device_get(state)
    payload: dict[str, Any] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION),
        "plan_sha256": np.asarray(protocol["plan_sha256"]),
        "candidate_id": np.asarray(candidate["candidate_id"]),
        "params_sha256": np.asarray(candidate["params_sha256"]),
        "audit_repeat": np.asarray(audit, dtype=np.bool_),
        "completed_step": np.asarray(step, dtype=np.int32),
        "rng": np.asarray(jax.device_get(rng), dtype=np.uint32),
        "source_state_hashes": source_state_hashes,
        "source_state_full_hashes": source_state_full_hashes,
        "embedding_steps": np.asarray(embedding_steps_done, dtype=np.int32),
        "clip_embeddings": np.stack(embeddings, axis=2).astype(np.float32),
        "state_metric_steps": np.asarray(metric_steps_done, dtype=np.int32),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "state_keys": np.asarray(sorted(host_state)),
    }
    for key, value in host_state.items():
        payload[f"state__{key}"] = np.asarray(value)
    for key, values in metrics.items():
        payload[key] = np.stack(values, axis=1)
    return payload


def _load_restart(
    path: Path,
    *,
    protocol: dict[str, Any],
    candidate: dict[str, str],
    audit: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import jax.numpy as jnp

    with np.load(path, allow_pickle=False) as data:
        checks = (
            str(np.asarray(data["protocol_version"]).item()) == PROTOCOL_VERSION,
            str(np.asarray(data["plan_sha256"]).item())
            == protocol["plan_sha256"],
            str(np.asarray(data["candidate_id"]).item())
            == candidate["candidate_id"],
            str(np.asarray(data["params_sha256"]).item())
            == candidate["params_sha256"],
            bool(np.asarray(data["audit_repeat"]).item()) == audit,
        )
        if not all(checks):
            raise RuntimeError(f"Restart checkpoint identity mismatch: {path}")
        step = int(np.asarray(data["completed_step"]).item())
        if (
            step < 0
            or step > HORIZON_STEPS
            or (
                step % CHECKPOINT_INTERVAL != 0
                and step != PREFLIGHT_STEPS
            )
        ):
            raise RuntimeError(f"Invalid restart step {step}: {path}")
        state = {
            str(key): jnp.asarray(data[f"state__{key}"])
            for key in np.asarray(data["state_keys"]).tolist()
        }
        metric_names = (
            "a_pair_relative_l1",
            "a_duplicate_relative_l1",
            "a_duplicate_max_abs",
            "total_mass",
            "duplicate_rng_max_abs",
        )
        return {
            "step": step,
            "state": state,
            "rng": jnp.asarray(data["rng"], dtype=jnp.uint32),
            "source_state_hashes": np.asarray(data["source_state_hashes"]),
            "source_state_full_hashes": np.asarray(
                data["source_state_full_hashes"]
            ),
            "embedding_steps": np.asarray(
                data["embedding_steps"], dtype=np.int32
            ).tolist(),
            "embeddings": [
                np.asarray(data["clip_embeddings"][:, :, idx], dtype=np.float32)
                for idx in range(np.asarray(data["clip_embeddings"]).shape[2])
            ],
            "metric_steps": np.asarray(
                data["state_metric_steps"], dtype=np.int32
            ).tolist(),
            "metrics": {
                key: [
                    np.asarray(data[key][:, idx])
                    for idx in range(np.asarray(data[key]).shape[1])
                ]
                for key in metric_names
            },
            "elapsed_seconds": float(np.asarray(data["elapsed_seconds"]).item()),
        }


def _validate_output(
    path: Path,
    *,
    protocol: dict[str, Any],
    candidate: dict[str, str],
    horizon: int,
    audit: bool,
) -> bool:
    if not path.exists():
        return False
    expected_embed_steps = [step for step in EMBED_STEPS if step <= horizon]
    expected_metric_steps = [
        step for step in STATE_METRIC_STEPS if step <= horizon
    ]
    try:
        with np.load(path, allow_pickle=False) as data:
            checks = (
                str(np.asarray(data["protocol_version"]).item())
                == PROTOCOL_VERSION,
                str(np.asarray(data["plan_sha256"]).item())
                == protocol["plan_sha256"],
                str(np.asarray(data["candidate_id"]).item())
                == candidate["candidate_id"],
                str(np.asarray(data["params_sha256"]).item())
                == candidate["params_sha256"],
                bool(np.asarray(data["audit_repeat"]).item()) == audit,
                int(np.asarray(data["horizon_steps"]).item()) == horizon,
                np.array_equal(
                    np.asarray(data["embedding_steps"], dtype=np.int32),
                    np.asarray(expected_embed_steps, dtype=np.int32),
                ),
                np.array_equal(
                    np.asarray(data["state_metric_steps"], dtype=np.int32),
                    np.asarray(expected_metric_steps, dtype=np.int32),
                ),
                np.asarray(data["clip_embeddings"]).shape
                == (N_CONTEXTS, N_BRANCHES, len(expected_embed_steps), 512),
                np.asarray(data["a_pair_relative_l1"]).shape
                == (
                    N_CONTEXTS,
                    len(expected_metric_steps),
                    len(PAIR_LEFT),
                ),
                np.all(np.isfinite(np.asarray(data["clip_embeddings"]))),
                np.all(np.isfinite(np.asarray(data["a_pair_relative_l1"]))),
                float(np.max(np.asarray(data["duplicate_rng_max_abs"]))) == 0.0,
            )
            return all(checks)
    except Exception:
        return False


def _write_progress(
    output_root: Path,
    *,
    status: str,
    candidate_id: str,
    candidate_position: int,
    total_candidates: int,
    step: int,
    horizon: int,
    elapsed_seconds: float,
    audit: bool,
) -> None:
    rate = step / elapsed_seconds if elapsed_seconds > 0 else 0.0
    candidate_eta = (horizon - step) / rate if rate > 0 else None
    path = output_root / (
        "audit_progress.json" if audit else "simulation_progress.json"
    )
    _write_json(
        path,
        {
            "status": status,
            "audit_repeat": audit,
            "candidate_id": candidate_id,
            "candidate_position": candidate_position,
            "total_candidates": total_candidates,
            "completed_step": step,
            "horizon_steps": horizon,
            "candidate_fraction": step / horizon if horizon else 1.0,
            "elapsed_seconds_including_resumed_work": elapsed_seconds,
            "steps_per_second": rate,
            "candidate_eta_seconds": candidate_eta,
            "updated_unix": time.time(),
        },
    )


def _simulate_candidate(
    *,
    output_root: Path,
    protocol: dict[str, Any],
    candidate: dict[str, str],
    contexts: list[dict[str, str]],
    substrate: Any,
    args: Any,
    params: Any,
    stepper: Any,
    rgb_capture: Any,
    state_metric_capture: Any,
    clip_embedder: Any,
    horizon: int,
    audit: bool,
    candidate_position: int,
    total_candidates: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    output_path = _candidate_output_path(
        output_root, candidate["candidate_id"], audit=audit
    )
    if _validate_output(
        output_path,
        protocol=protocol,
        candidate=candidate,
        horizon=horizon,
        audit=audit,
    ):
        print(f"[skip] valid {output_path}", flush=True)
        return {"status": "reused", "path": str(output_path)}
    if output_path.exists():
        quarantine = output_path.with_suffix(
            output_path.suffix + f".invalid-{int(time.time())}"
        )
        shutil.move(output_path, quarantine)
        print(f"[quarantine] {output_path} -> {quarantine}", flush=True)

    state, rng, source_hashes, source_full_hashes = _build_initial_batch(
        substrate, args, params, contexts
    )
    params_batch = jnp.repeat(params[None], BATCH_SIZE, axis=0)
    restart_path = _candidate_checkpoint_path(
        output_root, candidate["candidate_id"], audit=audit
    )
    restart = _load_restart(
        restart_path,
        protocol=protocol,
        candidate=candidate,
        audit=audit,
    )
    if restart is None:
        step = 0
        embedding_steps_done: list[int] = []
        embeddings: list[np.ndarray] = []
        metric_steps_done: list[int] = []
        metrics: dict[str, list[np.ndarray]] = {
            "a_pair_relative_l1": [],
            "a_duplicate_relative_l1": [],
            "a_duplicate_max_abs": [],
            "total_mass": [],
            "duplicate_rng_max_abs": [],
        }
        previous_elapsed = 0.0
    else:
        step = restart["step"]
        state = restart["state"]
        rng = restart["rng"]
        if not np.array_equal(
            source_hashes, restart["source_state_hashes"]
        ):
            raise RuntimeError(f"Initial state hashes changed for {candidate['candidate_id']}")
        if not np.array_equal(
            source_full_hashes, restart["source_state_full_hashes"]
        ):
            raise RuntimeError(
                f"Full initial state hashes changed for {candidate['candidate_id']}"
            )
        embedding_steps_done = restart["embedding_steps"]
        embeddings = restart["embeddings"]
        metric_steps_done = restart["metric_steps"]
        metrics = restart["metrics"]
        previous_elapsed = restart["elapsed_seconds"]
        print(
            f"[resume] {candidate['candidate_id']} audit={audit} step={step}",
            flush=True,
        )

    expected_embed_prefix = (
        []
        if restart is None and step == 0
        else [value for value in EMBED_STEPS if value <= step]
    )
    expected_metric_prefix = (
        []
        if restart is None and step == 0
        else [value for value in STATE_METRIC_STEPS if value <= step]
    )
    if step == horizon:
        expected_embed_prefix = [value for value in EMBED_STEPS if value <= horizon]
        expected_metric_prefix = [
            value for value in STATE_METRIC_STEPS if value <= horizon
        ]
    elif step not in (0,) and step % CHECKPOINT_INTERVAL != 0:
        raise RuntimeError(f"Unexpected resume step: {step}")
    if embedding_steps_done != expected_embed_prefix:
        raise RuntimeError(
            f"Embedding checkpoint prefix mismatch at step {step}"
        )
    if metric_steps_done != expected_metric_prefix:
        raise RuntimeError(f"Metric checkpoint prefix mismatch at step {step}")

    embed_set = set(EMBED_STEPS)
    metric_set = set(STATE_METRIC_STEPS)
    started = time.monotonic()
    if step == 0 and not embedding_steps_done:
        rgb = rgb_capture(state)
        embeddings.append(_capture_embeddings(rgb, clip_embedder))
        embedding_steps_done.append(0)
    if step == 0 and not metric_steps_done:
        captured = jax.device_get(state_metric_capture(state, rng))
        for key, value in captured.items():
            metrics[key].append(np.asarray(value))
        metric_steps_done.append(0)

    initial_pair_max = float(np.max(metrics["a_pair_relative_l1"][0]))
    initial_duplicate_max = float(
        np.max(metrics["a_duplicate_max_abs"][0])
    )
    initial_rng_duplicate = float(
        np.max(metrics["duplicate_rng_max_abs"][0])
    )
    if max(initial_pair_max, initial_duplicate_max, initial_rng_duplicate) != 0.0:
        raise RuntimeError(
            "Initial branches or duplicate keys are not bit-exact: "
            f"pair={initial_pair_max}, duplicate={initial_duplicate_max}, "
            f"rng={initial_rng_duplicate}"
        )

    while step < horizon:
        state, rng = stepper(state, rng, params_batch)
        step += STEP_CHUNK
        if step in embed_set and step <= horizon:
            rgb = rgb_capture(state)
            embeddings.append(_capture_embeddings(rgb, clip_embedder))
            embedding_steps_done.append(step)
        if step in metric_set and step <= horizon:
            captured = jax.device_get(state_metric_capture(state, rng))
            for key, value in captured.items():
                metrics[key].append(np.asarray(value))
            metric_steps_done.append(step)

        elapsed = previous_elapsed + time.monotonic() - started
        if step % PROGRESS_INTERVAL == 0 or step == horizon:
            _write_progress(
                output_root,
                status="running",
                candidate_id=candidate["candidate_id"],
                candidate_position=candidate_position,
                total_candidates=total_candidates,
                step=step,
                horizon=horizon,
                elapsed_seconds=elapsed,
                audit=audit,
            )
            print(
                f"[progress] {candidate['candidate_id']} audit={audit} "
                f"step={step}/{horizon} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if step % CHECKPOINT_INTERVAL == 0 or step == horizon:
            payload = _checkpoint_payload(
                protocol=protocol,
                candidate=candidate,
                step=step,
                state=state,
                rng=rng,
                source_state_hashes=source_hashes,
                source_state_full_hashes=source_full_hashes,
                embedding_steps_done=embedding_steps_done,
                embeddings=embeddings,
                metric_steps_done=metric_steps_done,
                metrics=metrics,
                elapsed_seconds=elapsed,
                audit=audit,
            )
            _save_npz(restart_path, payload, compressed=False)
            print(
                f"[checkpoint] {candidate['candidate_id']} audit={audit} "
                f"step={step} path={restart_path}",
                flush=True,
            )

    elapsed = previous_elapsed + time.monotonic() - started
    payload = _checkpoint_payload(
        protocol=protocol,
        candidate=candidate,
        step=horizon,
        state=state,
        rng=rng,
        source_state_hashes=source_hashes,
        source_state_full_hashes=source_full_hashes,
        embedding_steps_done=embedding_steps_done,
        embeddings=embeddings,
        metric_steps_done=metric_steps_done,
        metrics=metrics,
        elapsed_seconds=elapsed,
        audit=audit,
    )
    for key in list(payload):
        if key.startswith("state__") or key in ("state_keys", "rng"):
            del payload[key]
    payload["horizon_steps"] = np.asarray(horizon, dtype=np.int32)
    payload["pair_left"] = PAIR_LEFT
    payload["pair_right"] = PAIR_RIGHT
    payload["branch_rng_keys_initial"] = np.stack(
        [
            _branch_rng_key(row, branch_idx)
            for row in contexts
            for branch_idx in range(N_BRANCHES)
        ],
        axis=0,
    ).reshape((N_CONTEXTS, N_BRANCHES, 2))
    _save_npz(output_path, payload, compressed=True)
    if not _validate_output(
        output_path,
        protocol=protocol,
        candidate=candidate,
        horizon=horizon,
        audit=audit,
    ):
        raise RuntimeError(f"Post-write validation failed: {output_path}")
    _write_progress(
        output_root,
        status="candidate_complete",
        candidate_id=candidate["candidate_id"],
        candidate_position=candidate_position,
        total_candidates=total_candidates,
        step=horizon,
        horizon=horizon,
        elapsed_seconds=elapsed,
        audit=audit,
    )
    return {
        "status": "complete",
        "path": str(output_path),
        "elapsed_seconds": elapsed,
    }


def _load_runtime(output_root: Path):
    protocol = load_protocol(output_root)
    candidates = _read_csv(output_root / "candidates.csv")
    contexts = _read_csv(output_root / "contexts.csv")
    config_path = Path(protocol["simulation_config"]["path"])
    _cfg, flat = load_config(config_path)
    args = SimpleNamespace(
        **dict(OmegaConf.to_container(flat, resolve=True))
    )

    import foundation_models
    import jax
    import jax.numpy as jnp

    substrate = _make_substrate(args)
    first_params = jnp.asarray(
        np.load(_resolve(candidates[0]["params_path"])), dtype=jnp.float32
    )
    _ = substrate.init_state(jax.random.PRNGKey(0), first_params)
    fm = foundation_models.create_foundation_model(CLIP_MODEL_NAME)
    model_identity = _model_fingerprint(fm)
    old_model_identity = json.loads(
        (SOURCE_CLIP_ROOT / "clip_model_identity.json").read_text(
            encoding="utf-8"
        )
    )
    if model_identity["identity_sha256"] != old_model_identity[
        "identity_sha256"
    ]:
        raise RuntimeError("CLIP model differs from the prior 10k experiment")
    _write_or_validate_identity(
        output_root, "clip_model_identity.json", model_identity
    )
    hardware = _hardware_identity()
    _write_or_validate_identity(output_root, "runtime_identity.json", hardware)
    return {
        "protocol": protocol,
        "candidates": candidates,
        "contexts": contexts,
        "args": args,
        "substrate": substrate,
        "stepper": _make_stepper(substrate),
        "rgb_capture": _make_rgb_capture(),
        "state_metric_capture": _make_state_metric_capture(),
        "clip_embedder": _make_clip_embedder(fm),
    }


def _contexts_for(
    contexts: list[dict[str, str]], candidate_id: str
) -> list[dict[str, str]]:
    rows = sorted(
        [row for row in contexts if row["candidate_id"] == candidate_id],
        key=lambda row: int(row["context_idx"]),
    )
    if len(rows) != N_CONTEXTS:
        raise RuntimeError(f"Expected {N_CONTEXTS} contexts for {candidate_id}")
    return rows


def _preflight_comparison(
    output_root: Path,
    protocol: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    new_path = _candidate_output_path(
        output_root / "preflight", candidate["candidate_id"], audit=True
    )
    old_base_path = (
        SOURCE_BASE_ROOT
        / "simulation"
        / candidate["candidate_id"]
        / "batch_01.npz"
    )
    old_clip_path = (
        SOURCE_CLIP_ROOT
        / "simulation"
        / candidate["candidate_id"]
        / "batch_01.npz"
    )
    with (
        np.load(new_path, allow_pickle=False) as new,
        np.load(old_base_path, allow_pickle=False) as old_state,
        np.load(old_clip_path, allow_pickle=False) as old_clip,
    ):
        new_state_steps = np.asarray(new["state_metric_steps"], dtype=np.int32)
        old_state_steps = np.asarray(old_state["metric_steps"], dtype=np.int32)
        shared_old_local = np.arange(8, 12)
        common_state_steps = np.intersect1d(
            new_state_steps, old_state_steps
        )
        old_pair_left = np.asarray(old_state["pair_left"])
        old_pair_right = np.asarray(old_state["pair_right"])
        old_pair_indices = [
            int(
                np.flatnonzero(
                    (old_pair_left == left) & (old_pair_right == right)
                )[0]
            )
            for left, right in zip(PAIR_LEFT, PAIR_RIGHT)
        ]
        new_state_values = np.asarray(new["a_pair_relative_l1"])[
            :,
            np.searchsorted(new_state_steps, common_state_steps),
        ]
        old_state_values = np.asarray(old_state["a_relative_l1"])[
            shared_old_local
        ][:, np.searchsorted(old_state_steps, common_state_steps)][
            :, :, old_pair_indices
        ]

        new_embed_steps = np.asarray(new["embedding_steps"], dtype=np.int32)
        old_embed_steps = np.asarray(old_clip["capture_steps"], dtype=np.int32)
        common_embed_steps = np.intersect1d(
            new_embed_steps, old_embed_steps
        )
        new_z = np.asarray(new["clip_embeddings"])[
            :,
            :N_UNIQUE_BRANCHES,
            np.searchsorted(new_embed_steps, common_embed_steps),
        ]
        old_z = np.asarray(old_clip["clip_embeddings"])[
            shared_old_local,
            :N_UNIQUE_BRANCHES,
        ][:, :, np.searchsorted(old_embed_steps, common_embed_steps)]
        new_z = new_z / np.clip(
            np.linalg.norm(new_z, axis=-1, keepdims=True), 1.0e-12, None
        )
        old_z = old_z / np.clip(
            np.linalg.norm(old_z, axis=-1, keepdims=True), 1.0e-12, None
        )
        replay_cosine = 1.0 - np.sum(new_z * old_z, axis=-1)
        rng_match = np.array_equal(
            np.asarray(new["branch_rng_keys_initial"])[:, :N_UNIQUE_BRANCHES],
            np.asarray(
                [
                    [
                        old_base._context_rng_key(
                            {
                                "arm": "shared_state",
                                "rollout_seed_idx": str(context_idx),
                                "anchor_idx": str(context_idx),
                            },
                            branch_idx,
                        )
                        for branch_idx in range(N_UNIQUE_BRANCHES)
                    ]
                    for context_idx in range(N_CONTEXTS)
                ],
                dtype=np.uint32,
            ),
        )
        report = {
            "status": "passed",
            "plan_sha256": protocol["plan_sha256"],
            "candidate_id": candidate["candidate_id"],
            "horizon_steps": PREFLIGHT_STEPS,
            "source_state_hashes_match": np.array_equal(
                np.asarray(new["source_state_hashes"]),
                np.asarray(old_state["source_state_hashes"])[shared_old_local],
            ),
            "unique_rng_keys_match": rng_match,
            "duplicate_key_matches_branch_0": np.array_equal(
                np.asarray(new["branch_rng_keys_initial"])[:, DUPLICATE_BRANCH],
                np.asarray(new["branch_rng_keys_initial"])[
                    :, DUPLICATE_OF_BRANCH
                ],
            ),
            "common_state_metric_steps": common_state_steps,
            "a_pair_replay_exact": np.array_equal(
                new_state_values, old_state_values
            ),
            "a_pair_replay_max_abs": float(
                np.max(np.abs(new_state_values - old_state_values))
            ),
            "common_embedding_steps": common_embed_steps,
            "same_trajectory_clip_replay_cosine_max": float(
                np.max(np.abs(replay_cosine))
            ),
            "same_trajectory_clip_replay_cosine_median": float(
                np.median(np.abs(replay_cosine))
            ),
            "new_duplicate_a_max": float(
                np.max(np.asarray(new["a_duplicate_max_abs"]))
            ),
            "new_duplicate_clip_cosine_max": float(
                np.max(
                    1.0
                    - np.sum(
                        np.asarray(new["clip_embeddings"])[
                            :, DUPLICATE_OF_BRANCH
                        ]
                        * np.asarray(new["clip_embeddings"])[
                            :, DUPLICATE_BRANCH
                        ],
                        axis=-1,
                    )
                )
            ),
        }
    required = (
        report["source_state_hashes_match"],
        report["unique_rng_keys_match"],
        report["duplicate_key_matches_branch_0"],
        report["new_duplicate_a_max"] == 0.0,
    )
    if not all(required):
        report["status"] = "failed"
    _write_json(output_root / "preflight_audit.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"Preflight failed: {report}")
    return report


def run_preflight(output_root: Path) -> dict[str, Any]:
    runtime = _load_runtime(output_root)
    candidate = runtime["candidates"][0]
    contexts = _contexts_for(runtime["contexts"], candidate["candidate_id"])
    params = np.asarray(
        np.load(_resolve(candidate["params_path"])), dtype=np.float32
    )
    import jax.numpy as jnp

    _simulate_candidate(
        output_root=output_root / "preflight",
        protocol=runtime["protocol"],
        candidate=candidate,
        contexts=contexts,
        substrate=runtime["substrate"],
        args=runtime["args"],
        params=jnp.asarray(params),
        stepper=runtime["stepper"],
        rgb_capture=runtime["rgb_capture"],
        state_metric_capture=runtime["state_metric_capture"],
        clip_embedder=runtime["clip_embedder"],
        horizon=PREFLIGHT_STEPS,
        audit=True,
        candidate_position=1,
        total_candidates=1,
    )
    return _preflight_comparison(
        output_root, runtime["protocol"], candidate
    )


def run_simulation(
    output_root: Path,
    *,
    candidate_ids: set[str] | None,
    audit: bool,
) -> dict[str, Any]:
    preflight_path = output_root / "preflight_audit.json"
    if not preflight_path.exists():
        raise RuntimeError("Run and pass preflight before the 300k simulation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("Preflight did not pass")
    runtime = _load_runtime(output_root)
    candidates = runtime["candidates"]
    if audit:
        candidates = [
            row for row in candidates if row["candidate_id"] == AUDIT_CANDIDATE
        ]
    elif candidate_ids:
        known = {row["candidate_id"] for row in candidates}
        unknown = candidate_ids.difference(known)
        if unknown:
            raise RuntimeError(f"Unknown candidates: {sorted(unknown)}")
        candidates = [
            row for row in candidates if row["candidate_id"] in candidate_ids
        ]
    results = []
    started = time.monotonic()
    for position, candidate in enumerate(candidates, start=1):
        contexts = _contexts_for(
            runtime["contexts"], candidate["candidate_id"]
        )
        params_np = np.asarray(
            np.load(_resolve(candidate["params_path"])), dtype=np.float32
        )
        if params_np.shape != (int(runtime["substrate"].n_params),):
            raise RuntimeError(
                f"Unexpected parameter shape for {candidate['candidate_id']}: "
                f"{params_np.shape}"
            )
        import jax.numpy as jnp

        result = _simulate_candidate(
            output_root=output_root,
            protocol=runtime["protocol"],
            candidate=candidate,
            contexts=contexts,
            substrate=runtime["substrate"],
            args=runtime["args"],
            params=jnp.asarray(params_np),
            stepper=runtime["stepper"],
            rgb_capture=runtime["rgb_capture"],
            state_metric_capture=runtime["state_metric_capture"],
            clip_embedder=runtime["clip_embedder"],
            horizon=HORIZON_STEPS,
            audit=audit,
            candidate_position=position,
            total_candidates=len(candidates),
        )
        results.append({"candidate_id": candidate["candidate_id"], **result})
    report = {
        "status": "complete",
        "audit_repeat": audit,
        "horizon_steps": HORIZON_STEPS,
        "candidate_count": len(candidates),
        "results": results,
        "wall_seconds_this_invocation": time.monotonic() - started,
    }
    name = "audit_simulation_completion.json" if audit else "simulation_completion.json"
    _write_json(output_root / name, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FlowLenia init-only continuation-RNG versus inter-random "
            "divergence experiment."
        )
    )
    parser.add_argument(
        "stage",
        choices=("prepare", "preflight", "simulate", "audit"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--candidate-id", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_root = _resolve(cli.output_root)
    if cli.stage == "prepare":
        protocol = prepare(output_root)
        print(
            f"Prepared {output_root} plan_sha256={protocol['plan_sha256']}",
            flush=True,
        )
        return
    if cli.stage == "preflight":
        print(json.dumps(_jsonable(run_preflight(output_root)), indent=2))
        return
    result = run_simulation(
        output_root,
        candidate_ids=set(cli.candidate_id) if cli.candidate_id else None,
        audit=cli.stage == "audit",
    )
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
