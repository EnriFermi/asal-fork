#!/usr/bin/env python3
"""Production RNG-only, mass-preserving Flow-Lenia C5 horizon grid.

The wall prefix is shared across all horizons. At each half-horizon release
point, the state and continuation RNG are forked into a native global rollout.
This preserves the old C5 half-walls/half-free design while avoiding repeated
simulation of common wall prefixes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ENV_BIN = str(Path(sys.executable).resolve().parent)
if (Path(_ENV_BIN) / "ptxas").exists():
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np

import flowlenia_c5_branch_frustration as old_c5
import flowlenia_c5_mass_preserving_wall_probe as mass_probe
from flowlenia_minibang_common import list_apf_chunks


PROTOCOL_VERSION = "flowlenia-c5-rng-only-mass-projected-horizon-grid-v2"
HORIZONS = (5_000, 10_000, 15_000, 20_000, 30_000)
RELEASE_STEPS = {horizon: horizon // 2 for horizon in HORIZONS}
CAPTURE_STEPS = {
    5_000: (0, 700, 1_400, 2_100, 2_850, 3_550, 4_250, 5_000),
    10_000: (0, 1_400, 2_850, 4_250, 5_700, 7_100, 8_550, 10_000),
    15_000: (0, 2_100, 4_250, 6_400, 8_550, 10_700, 12_850, 15_000),
    20_000: (0, 2_850, 5_700, 8_550, 11_400, 14_250, 17_100, 20_000),
    30_000: (0, 4_250, 8_550, 12_850, 17_100, 21_400, 25_700, 30_000),
}
FREE_CAPTURE_UNION = tuple(
    sorted({step for values in CAPTURE_STEPS.values() for step in values})
)
PREFIX_CAPTURE_STEPS = tuple(
    sorted(
        {
            step
            for horizon, values in CAPTURE_STEPS.items()
            for step in values[:4]
            if step < RELEASE_STEPS[horizon]
        }
    )
)
ALL_PREFIX_EVENTS = tuple(
    sorted(set(PREFIX_CAPTURE_STEPS) | set(RELEASE_STEPS.values()))
)
SIMULATION_BATCH_SIZE = old_c5.SIMULATION_BATCH_SIZE
JIT_MICROBATCH = old_c5.JIT_MICROBATCH
EXPECTED_ROWS = old_c5.EXPECTED_PLAN_ROWS
DEFAULT_OLD_C5_ROOT = old_c5.DEFAULT_OUTPUT_ROOT
DEFAULT_SWEEP_PLAN = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "c2_noise_horizon_sweep/full/sweep_plan.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
)
SIMULATION_CODE_FILES = (
    "scripts/flowlenia_c5_rng_only_horizon_grid.py",
    "scripts/flowlenia_c5_branch_frustration.py",
    "scripts/flowlenia_c5_mass_preserving_wall_probe.py",
    "scripts/flowlenia_minibang_resume.py",
    "scripts/flowlenia_minibang_resume_batch.py",
    "scripts/flowlenia_minibang_simulate.py",
    "scripts/paper_check_frustration_batch_eval.py",
    "substrates/lenia_flow/lenia_flow.py",
    "substrates/lenia_flow/reintegration_tracking.py",
)
PLAN_FIELDS = (
    "row_id",
    "run_idx",
    "trial_idx",
    "candidate_kind",
    "candidate_idx",
    "candidate_id",
    "source_traj_id",
    "source_traj_dir",
    "source_config_path",
    "source_config_sha256",
    "source_simulation_config_sha256",
    "params_path",
    "params_sha256",
    "condition",
    "pair_id",
    "point_id",
    "window_index",
    "step",
    "delta_h",
    "branch_id",
    "branch_seed",
    "perturb_a_std",
    "perturb_p_std",
    "perturb_lagrangian_xy_std",
    "free_branch_dir",
    "free_provenance",
    "wall_grid_path",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return value.resolve()


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
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in PLAN_FIELDS})


def _stable_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _code_fingerprint() -> dict[str, Any]:
    files = {
        relative: _sha256_file(_REPO_ROOT / relative)
        for relative in SIMULATION_CODE_FILES
    }
    return {
        "files": files,
        "bundle_sha256": _sha256_bytes(
            _stable_json(files).encode("utf-8")
        ),
    }


def _plan_hash(rows: Sequence[dict[str, Any]]) -> str:
    identity = [
        {
            field: str(row[field])
            for field in PLAN_FIELDS
            if field not in {"free_branch_dir", "wall_grid_path"}
        }
        for row in rows
    ]
    return _sha256_bytes(_stable_json(identity).encode("utf-8"))


def _wall_grid_path(root: Path, row: dict[str, Any]) -> Path:
    return (
        root
        / "wall_grid"
        / str(row["candidate_id"])
        / (
            f"{row['condition']}_{int(row['point_id']):03d}_"
            f"step_{int(row['step']):06d}"
        )
        / f"branch_{int(row['branch_id']):02d}.npz"
    )


def _rng_only_lookup(
    sweep_plan_path: Path,
) -> dict[tuple[int, int, str, int, int], dict[str, str]]:
    rows = [
        row
        for row in _read_csv(sweep_plan_path)
        if float(row["strength"]) == 0.0
    ]
    if len(rows) != 450:
        raise RuntimeError(f"Expected 450 optimized RNG-only rows, found {len(rows)}")
    lookup = {}
    for row in rows:
        key = (
            int(row["run_idx"]),
            int(row["pair_id"]),
            str(row["condition"]),
            int(row["step"]),
            int(row["branch_id"]),
        )
        if key in lookup:
            raise RuntimeError(f"Duplicate RNG-only row {key}")
        if any(float(row[field]) != 0.0 for field in ("a_std", "p_std", "lagrangian_xy_std")):
            raise RuntimeError(f"Nonzero perturbation in RNG-only row {key}")
        lookup[key] = row
    return lookup


def build_plan(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output_root = _resolve(args.output_root)
    old_root = _resolve(args.old_c5_root)
    old_plan_path = old_root / "paired_plan.csv"
    old_protocol_path = old_root / "protocol.json"
    sweep_plan_path = _resolve(args.sweep_plan)
    old_rows = _read_csv(old_plan_path)
    if len(old_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} old selection rows")
    old_protocol = json.loads(old_protocol_path.read_text())
    if old_c5._plan_identity_hash(old_rows) != old_protocol["plan_sha256"]:
        raise RuntimeError("Old C5 plan identity failed")
    rng_only = _rng_only_lookup(sweep_plan_path)
    rows: list[dict[str, Any]] = []
    for old in old_rows:
        row = {
            field: old[field]
            for field in PLAN_FIELDS
            if field in old
        }
        row.update(
            {
                "perturb_a_std": 0.0,
                "perturb_p_std": 0.0,
                "perturb_lagrangian_xy_std": 0.0,
            }
        )
        if old["candidate_kind"] == "optimized":
            key = (
                int(old["run_idx"]),
                int(old["pair_id"]),
                str(old["condition"]),
                int(old["step"]),
                int(old["branch_id"]),
            )
            source = rng_only.get(key)
            if source is None:
                raise RuntimeError(f"Missing RNG-only optimized branch {key}")
            if int(source["branch_seed"]) != int(old["branch_seed"]):
                raise RuntimeError(f"Branch seed mismatch for {key}")
            row["free_branch_dir"] = str(_resolve(source["branch_dir"]))
            row["free_provenance"] = "reused_rng_only_sweep"
        else:
            row["free_branch_dir"] = str(
                output_root
                / "free_random"
                / str(old["candidate_id"])
                / (
                    f"{old['condition']}_{int(old['point_id']):03d}_"
                    f"step_{int(old['step']):06d}"
                )
                / f"branch_{int(old['branch_id']):02d}"
            )
            row["free_provenance"] = "generated_rng_only_30k"
        row["wall_grid_path"] = str(_wall_grid_path(output_root, row))
        rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Built {len(rows)} rows, expected {EXPECTED_ROWS}")
    if Counter(row["candidate_kind"] for row in rows) != {
        "optimized": 450,
        "random": 1350,
    }:
        raise RuntimeError("Candidate row counts changed")
    if any(
        float(row[field]) != 0.0
        for row in rows
        for field in (
            "perturb_a_std",
            "perturb_p_std",
            "perturb_lagrangian_xy_std",
        )
    ):
        raise RuntimeError("Plan contains external state perturbation")

    plan_hash = _plan_hash(rows)
    code = _code_fingerprint()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": plan_hash,
        "simulation_code_bundle_sha256": code["bundle_sha256"],
        "simulation_code_files": code["files"],
        "source_selection_plan": old_plan_path,
        "source_selection_plan_sha256": _sha256_file(old_plan_path),
        "rng_only_sweep_plan": sweep_plan_path,
        "rng_only_sweep_plan_sha256": _sha256_file(sweep_plan_path),
        "n_rows": len(rows),
        "n_candidates": 40,
        "n_points_per_candidate": 15,
        "n_branches_per_point": 3,
        "horizons": HORIZONS,
        "release_steps": RELEASE_STEPS,
        "capture_steps": CAPTURE_STEPS,
        "free_capture_union": FREE_CAPTURE_UNION,
        "outer_batch_size": SIMULATION_BATCH_SIZE,
        "optimizer_native_batch_size": old_c5.OPTIMIZER_NATIVE_BATCH_SIZE,
        "jit_microbatch": JIT_MICROBATCH,
        "branch_protocol": {
            "A_std": 0.0,
            "P_std": 0.0,
            "lagrangian_xy_std": 0.0,
            "difference": "folded continuation branch_seed only",
        },
        "wall_protocol": {
            "split": "3x3 blocks with five-cell hard-zero padding",
            "release": "walls active for exactly half of each horizon",
            "mass_projection": (
                "after every confined transition and hard mask, scale A "
                "separately within every block/channel to previous-step mass"
            ),
            "P_and_F_projection": "none",
            "shared_prefix": (
                "one wall prefix through step 15000 is forked with its exact "
                "continuation RNG at each release point"
            ),
        },
        "metric_windows": (
            "the last four scaled capture timestamps for every horizon; all "
            "four occur strictly after wall release"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "plan.csv"
    protocol_path = output_root / "protocol.json"
    if plan_path.exists() or protocol_path.exists():
        if not plan_path.exists() or not protocol_path.exists():
            raise RuntimeError("Partial plan/protocol files exist")
        existing_rows = _read_csv(plan_path)
        existing_protocol = json.loads(protocol_path.read_text())
        if (
            _plan_hash(existing_rows) != plan_hash
            or existing_protocol.get("plan_sha256") != plan_hash
            or existing_protocol.get("protocol_version") != PROTOCOL_VERSION
            or existing_protocol.get("simulation_code_bundle_sha256")
            != code["bundle_sha256"]
        ):
            raise RuntimeError(
                "Existing output root has a different plan or simulation code"
            )
    else:
        _write_csv(plan_path, rows)
        _write_json(protocol_path, protocol)
    _write_json(
        output_root / "plan_summary.json",
        {
            "status": "complete",
            "n_rows": len(rows),
            "plan_sha256": plan_hash,
            "optimized_free_reused": 450,
            "random_free_required": 1350,
            "wall_grid_outputs_required": EXPECTED_ROWS,
        },
    )
    return _read_csv(plan_path), protocol


def load_plan(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = _resolve(args.output_root)
    if not (root / "plan.csv").exists() or not (root / "protocol.json").exists():
        return build_plan(args)
    rows = _read_csv(root / "plan.csv")
    protocol = json.loads((root / "protocol.json").read_text())
    if _plan_hash(rows) != protocol["plan_sha256"]:
        raise RuntimeError("Plan hash mismatch")
    code = _code_fingerprint()["bundle_sha256"]
    if protocol.get("simulation_code_bundle_sha256") != code:
        raise RuntimeError(
            "Simulation code changed after protocol creation; use a new root"
        )
    return rows, protocol


def _free_arrays(row: dict[str, str]) -> dict[str, np.ndarray]:
    branch_dir = _resolve(row["free_branch_dir"])
    chunks = list_apf_chunks(branch_dir / "apf_logs")
    if not chunks:
        raise FileNotFoundError(f"No free APF under {branch_dir}")
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("steps", "A", "P")
    }
    for path, _s0, _s1, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            for key in parts:
                if key not in data:
                    raise RuntimeError(f"{path} lacks {key}")
                parts[key].append(np.asarray(data[key]))
    result = {
        key: np.concatenate(values, axis=0)
        for key, values in parts.items()
    }
    order = np.argsort(result["steps"], kind="stable")
    return {key: value[order] for key, value in result.items()}


def _free_audit(
    row: dict[str, str],
    *,
    check_source_start: bool,
    source_start_cache: dict[
        tuple[str, int],
        dict[str, np.ndarray],
    ]
    | None = None,
) -> dict[str, Any]:
    branch_dir = _resolve(row["free_branch_dir"])
    metadata_path = branch_dir / "resume_metadata.json"
    result = {
        "ready": False,
        "row_id": int(row["row_id"]),
        "branch_dir": str(branch_dir),
        "reason": "",
    }
    if not metadata_path.exists():
        result["reason"] = "missing resume_metadata.json"
        return result
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "start_step": int(row["step"]),
        "branch_seed": int(row["branch_seed"]),
        "perturb_a_std": 0.0,
        "perturb_p_std": 0.0,
        "perturb_lagrangian_xy_std": 0.0,
        "original_batch_size": old_c5.OPTIMIZER_NATIVE_BATCH_SIZE,
        "jit_microbatch": JIT_MICROBATCH,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if int(metadata.get("end_step", -1)) < int(row["step"]) + max(HORIZONS):
        mismatches["end_step"] = {
            "expected_at_least": int(row["step"]) + max(HORIZONS),
            "actual": metadata.get("end_step"),
        }
    if mismatches:
        result["reason"] = f"metadata mismatch: {mismatches}"
        return result
    try:
        arrays = _free_arrays(row)
    except Exception as exc:
        result["reason"] = f"APF load failed: {exc}"
        return result
    relative = np.asarray(arrays["steps"], dtype=np.int64) - int(row["step"])
    missing_steps = sorted(set(FREE_CAPTURE_UNION) - set(relative.tolist()))
    if missing_steps:
        result["reason"] = f"missing capture steps: {missing_steps}"
        return result
    if not all(np.all(np.isfinite(arrays[key])) for key in ("A", "P")):
        result["reason"] = "non-finite free A/P"
        return result
    start_exact = None
    if check_source_start:
        source_dir = _resolve(row["source_traj_dir"])
        path, _step, idx = old_c5._find_snapshot(
            source_dir / "apf_logs",
            int(row["step"]),
        )
        cache_key = (str(path), int(idx))
        source = (
            source_start_cache.get(cache_key)
            if source_start_cache is not None
            else None
        )
        if source is None:
            with np.load(path, allow_pickle=False) as data:
                source = {
                    key: np.asarray(data[key][idx])
                    for key in ("A", "P")
                }
            if source_start_cache is not None:
                source_start_cache[cache_key] = source
        first_idx = int(np.flatnonzero(relative == 0)[0])
        start_exact = {
            key: bool(
                np.array_equal(
                    arrays[key][first_idx],
                    np.asarray(source[key]).astype(arrays[key].dtype),
                )
            )
            for key in ("A", "P")
        }
        if not all(start_exact.values()):
            result["reason"] = f"branch origin differs from source: {start_exact}"
            return result
    result.update(
        {
            "ready": True,
            "reason": "",
            "relative_steps": relative.tolist(),
            "start_exact": start_exact,
            "start_hashes": {
                key: _sha256_array(
                    arrays[key][int(np.flatnonzero(relative == 0)[0])]
                )
                for key in ("A", "P")
            },
        }
    )
    return result


def generate_free(args: argparse.Namespace) -> dict[str, Any]:
    rows, protocol = load_plan(args)
    output_root = _resolve(args.output_root)
    random_rows = [row for row in rows if row["candidate_kind"] == "random"]
    optimized_rows = [
        row for row in rows if row["candidate_kind"] == "optimized"
    ]
    source_start_cache: dict[
        tuple[str, int],
        dict[str, np.ndarray],
    ] = {}
    optimized_failed = [
        audit
        for row in optimized_rows
        if not (
            audit := _free_audit(
                row,
                check_source_start=True,
                source_start_cache=source_start_cache,
            )
        )["ready"]
    ]
    if optimized_failed:
        raise RuntimeError(
            f"RNG-only optimized free audit failed: {optimized_failed[:3]}"
        )
    missing = []
    for row in random_rows:
        audit = _free_audit(row, check_source_start=False)
        if audit["ready"]:
            continue
        branch_dir = _resolve(row["free_branch_dir"])
        if branch_dir.exists() and any(branch_dir.iterdir()):
            raise RuntimeError(
                f"Invalid nonempty random free output: {branch_dir}: {audit}"
            )
        missing.append(row)
    jobs = [
        {
            "source_traj_dir": row["source_traj_dir"],
            "step": int(row["step"]),
            "additional_steps": max(HORIZONS),
            "output_dir": row["free_branch_dir"],
            "branch_seed": int(row["branch_seed"]),
            "perturb_a_std": 0.0,
            "perturb_p_std": 0.0,
            "perturb_lagrangian_xy_std": 0.0,
            "capture_relative_steps": FREE_CAPTURE_UNION,
            "output_fields": (
                "steps",
                "P",
                "A",
                "resume_batch_rng_key",
                "resume_batch_size",
                "resume_batch_index",
                "resume_jit_microbatch",
                "resume_snapshot_interval",
                "state_t",
                "state_mass_cycle_start",
            ),
            "output_compress": True,
            "output_max_snapshots_per_chunk": len(FREE_CAPTURE_UNION),
            "ignore_output_paths_in_simulation_signature": True,
        }
        for row in missing
    ]
    jobs_path = output_root / "free_random_jobs.json"
    _write_json(
        jobs_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "plan_sha256": protocol["plan_sha256"],
            "jobs": jobs,
        },
    )
    if jobs:
        print(
            f"[free] generating {len(jobs)} RNG-only random branches",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"),
                "--jobs-json",
                str(jobs_path),
                "--batch-size",
                str(SIMULATION_BATCH_SIZE),
            ],
            cwd=_REPO_ROOT,
            check=True,
        )
    failed = [
        audit
        for row in random_rows
        if not (
            audit := _free_audit(
                row,
                check_source_start=True,
                source_start_cache=source_start_cache,
            )
        )["ready"]
    ]
    if failed:
        raise RuntimeError(f"Random free audit failed: {failed[:3]}")
    summary = {
        "status": "complete",
        "plan_sha256": protocol["plan_sha256"],
        "optimized_reused": len(optimized_rows),
        "random_reused": len(random_rows) - len(jobs),
        "random_generated": len(jobs),
        "all_branch_origins_exact": True,
    }
    _write_json(output_root / "free_summary.json", summary)
    return summary


def _pad_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    real_n = len(items)
    if real_n < 1 or real_n > SIMULATION_BATCH_SIZE:
        raise ValueError(f"Invalid real batch size {real_n}")
    padded = list(items)
    while len(padded) < SIMULATION_BATCH_SIZE:
        padded.append(items[len(padded) % real_n])
    return padded, real_n


def _merge_batch_fn(
    geometry: old_c5.SimulationGeometry,
) -> Callable[[Any, Any], Any]:
    @jax.jit
    def merge(initial_state: Any, block_state: Any) -> Any:
        return jax.vmap(
            lambda initial, blocks: old_c5._merge_block_state(
                initial,
                blocks,
                geometry=geometry,
            )
        )(initial_state, block_state)

    return merge


def _capture(
    state: Any,
    rng: Any,
    *,
    real_n: int,
) -> dict[str, np.ndarray]:
    host = jax.device_get(
        {
            "A": state["A"][:real_n],
            "P": state["P"][:real_n],
            "rng": rng[:real_n],
        }
    )
    a_float = np.asarray(host["A"], dtype=np.float32)
    return {
        "A": a_float.astype(np.float16),
        "P": np.asarray(host["P"], dtype=np.float16),
        "mass_total_float32": np.sum(
            a_float.astype(np.float64),
            axis=(1, 2, 3),
        ),
        "rng": np.asarray(host["rng"], dtype=np.uint32),
    }


def _stack_lane_captures(
    captures: Sequence[dict[str, np.ndarray]],
    lane: int,
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([capture[key][lane] for capture in captures], axis=0)
        for key in ("A", "P", "mass_total_float32", "rng")
    }


def _wall_output_audit(
    row: dict[str, str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    path = _resolve(row["wall_grid_path"])
    metadata_path = path.with_suffix(".metadata.json")
    result = {
        "ready": False,
        "row_id": int(row["row_id"]),
        "path": str(path),
        "reason": "",
    }
    if not path.exists() or not metadata_path.exists():
        result["reason"] = "missing NPZ or metadata"
        return result
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        result["reason"] = f"metadata read failed: {exc}"
        return result
    expected = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "row_id": int(row["row_id"]),
        "branch_seed": int(row["branch_seed"]),
        "A_std": 0.0,
        "P_std": 0.0,
        "lagrangian_xy_std": 0.0,
        "artifact_sha256": _sha256_file(path),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        result["reason"] = f"metadata mismatch: {mismatches}"
        return result
    try:
        with np.load(path, allow_pickle=False) as data:
            horizons = np.asarray(data["horizons"], dtype=np.int64)
            capture_steps = np.asarray(data["capture_steps"], dtype=np.int64)
            a_value = np.asarray(data["A"])
            p_value = np.asarray(data["P"])
            rng = np.asarray(data["resume_batch_rng_key"])
    except Exception as exc:
        result["reason"] = f"NPZ read failed: {exc}"
        return result
    expected_capture = np.asarray(
        [CAPTURE_STEPS[horizon] for horizon in HORIZONS],
        dtype=np.int64,
    )
    if not np.array_equal(horizons, np.asarray(HORIZONS)):
        result["reason"] = "horizons mismatch"
        return result
    if not np.array_equal(capture_steps, expected_capture):
        result["reason"] = "capture grid mismatch"
        return result
    if a_value.shape[:2] != (len(HORIZONS), 8):
        result["reason"] = f"A shape mismatch: {a_value.shape}"
        return result
    if p_value.shape[:2] != (len(HORIZONS), 8):
        result["reason"] = f"P shape mismatch: {p_value.shape}"
        return result
    if rng.shape != (len(HORIZONS), 8, 2):
        result["reason"] = f"RNG shape mismatch: {rng.shape}"
        return result
    if not np.all(a_value[:, 0] == a_value[0, 0]):
        result["reason"] = "horizon starts differ in A"
        return result
    if not np.all(p_value[:, 0] == p_value[0, 0]):
        result["reason"] = "horizon starts differ in P"
        return result
    if not np.all(rng[:, 0] == rng[0, 0]):
        result["reason"] = "horizon starts differ in RNG"
        return result
    if not np.all(np.isfinite(a_value)) or not np.all(np.isfinite(p_value)):
        result["reason"] = "non-finite A/P"
        return result
    result.update(
        {
            "ready": True,
            "reason": "",
            "artifact_sha256": expected["artifact_sha256"],
            "initial_A_sha256": _sha256_array(a_value[0, 0]),
            "initial_P_sha256": _sha256_array(p_value[0, 0]),
        }
    )
    return result


def _save_wall_lane(
    *,
    row: dict[str, str],
    protocol: dict[str, Any],
    captures_by_horizon: dict[int, Sequence[dict[str, np.ndarray]]],
    lane: int,
    real_row_ids: Sequence[int],
    mass_audit: dict[str, Any],
) -> None:
    path = _resolve(row["wall_grid_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".metadata.json").exists():
        raise RuntimeError(f"Refusing to overwrite wall output {path}")
    lane_values = {
        horizon: _stack_lane_captures(captures_by_horizon[horizon], lane)
        for horizon in HORIZONS
    }
    np.savez_compressed(
        path,
        horizons=np.asarray(HORIZONS, dtype=np.int32),
        release_steps=np.asarray(
            [RELEASE_STEPS[horizon] for horizon in HORIZONS],
            dtype=np.int32,
        ),
        capture_steps=np.asarray(
            [CAPTURE_STEPS[horizon] for horizon in HORIZONS],
            dtype=np.int32,
        ),
        A=np.stack(
            [lane_values[horizon]["A"] for horizon in HORIZONS],
            axis=0,
        ),
        P=np.stack(
            [lane_values[horizon]["P"] for horizon in HORIZONS],
            axis=0,
        ),
        mass_total_float32=np.stack(
            [
                lane_values[horizon]["mass_total_float32"]
                for horizon in HORIZONS
            ],
            axis=0,
        ),
        resume_batch_rng_key=np.stack(
            [lane_values[horizon]["rng"] for horizon in HORIZONS],
            axis=0,
        ),
        row_id=np.asarray(int(row["row_id"]), dtype=np.int32),
        branch_seed=np.asarray(int(row["branch_seed"]), dtype=np.int64),
    )
    metadata = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "row_id": int(row["row_id"]),
        "candidate_id": row["candidate_id"],
        "point_id": int(row["point_id"]),
        "branch_id": int(row["branch_id"]),
        "branch_seed": int(row["branch_seed"]),
        "A_std": 0.0,
        "P_std": 0.0,
        "lagrangian_xy_std": 0.0,
        "free_branch_dir": row["free_branch_dir"],
        "free_provenance": row["free_provenance"],
        "horizons": HORIZONS,
        "release_steps": RELEASE_STEPS,
        "capture_steps": CAPTURE_STEPS,
        "outer_batch": {
            "fixed_batch_size": SIMULATION_BATCH_SIZE,
            "real_row_ids": list(real_row_ids),
            "lane_index": lane,
            "padding": "repeat real rows cyclically; padded outputs discarded",
        },
        "mass_projection": (
            "post-mask per-block/per-channel A scaling to previous-step mass"
        ),
        "mass_audit": mass_audit,
        "artifact_sha256": _sha256_file(path),
    }
    _write_json(path.with_suffix(".metadata.json"), metadata)


def _simulate_wall_batch(
    rows: list[dict[str, str]],
    *,
    protocol: dict[str, Any],
    engine: dict[str, Any],
    write_outputs: bool,
    output_path_override: Callable[[dict[str, str]], Path] | None = None,
) -> dict[str, Any]:
    runtime = engine["runtime"]
    snapshot_cache: dict[Any, Any] = {}
    items_unpadded = [
        old_c5._load_simulation_item(
            row,
            runtime["substrate"],
            snapshot_cache=snapshot_cache,
            params_cache=engine["params_cache"],
            state_template_cache=engine["state_template_cache"],
        )
        for row in rows
    ]
    items, real_n = _pad_items(items_unpadded)
    for item in items:
        item["args"] = runtime["args"]
    if any(
        item["original_batch_size"] != old_c5.OPTIMIZER_NATIVE_BATCH_SIZE
        or item["jit_microbatch"] != JIT_MICROBATCH
        for item in items
    ):
        raise RuntimeError("Optimizer-native batch/JIT mismatch")
    initial_states = [item["state"] for item in items]
    initial_state = old_c5._stack_trees(initial_states)
    block_state, roundtrip = old_c5._prepare_block_state_batch(
        items,
        runtime,
    )
    if not roundtrip["all_ap_exact"]:
        raise RuntimeError("Block split/merge is not exact")
    rng = jnp.stack([jnp.asarray(item["rng"]) for item in items], axis=0)
    params = jnp.stack(
        [jnp.asarray(item["params"], dtype=jnp.float32) for item in items],
        axis=0,
    )
    indices = jnp.asarray(
        [item["original_batch_index"] for item in items],
        dtype=jnp.int32,
    )
    merge_batch = engine["merge_batch"]
    block_stepper = engine["mass_projected_block_stepper"]
    global_stepper = engine["global_state_stepper"]
    prefix_captures: dict[int, dict[str, np.ndarray]] = {
        0: _capture(initial_state, rng, real_n=real_n)
    }
    release_states: dict[int, tuple[Any, Any]] = {}
    mass_audits: dict[int, dict[str, Any]] = {}
    initial_block_mass = np.asarray(
        jax.device_get(jnp.sum(block_state["A"], axis=(2, 3))),
        dtype=np.float64,
    )
    rel_step = 0
    started = time.monotonic()
    for target in ALL_PREFIX_EVENTS:
        while rel_step < target:
            n_steps = min(JIT_MICROBATCH, target - rel_step)
            rng, subkeys = old_c5._split_rng_batch(rng)
            block_state = block_stepper(n_steps)(
                block_state,
                subkeys,
                params,
                indices,
            )
            rel_step += n_steps
        if target in PREFIX_CAPTURE_STEPS and target != 0:
            merged = merge_batch(initial_state, block_state)
            prefix_captures[target] = _capture(
                merged,
                rng,
                real_n=real_n,
            )
        if target in RELEASE_STEPS.values():
            merged = merge_batch(initial_state, block_state)
            release_states[target] = (merged, rng)
            current_mass = np.asarray(
                jax.device_get(jnp.sum(block_state["A"], axis=(2, 3))),
                dtype=np.float64,
            )
            scale = np.maximum(np.abs(initial_block_mass), 1.0e-12)
            rel_error = np.abs(current_mass - initial_block_mass) / scale
            mass_audits[target] = {
                "max_abs": float(
                    np.max(np.abs(current_mass - initial_block_mass))
                ),
                "max_relative": float(np.max(rel_error)),
                "median_relative": float(np.median(rel_error)),
            }
        elapsed = time.monotonic() - started
        print(
            f"[wall-prefix] B={real_n}/{SIMULATION_BATCH_SIZE} "
            f"step={target}/{max(RELEASE_STEPS.values())} "
            f"elapsed={elapsed / 60:.1f}m",
            flush=True,
        )
    if set(release_states) != set(RELEASE_STEPS.values()):
        raise RuntimeError("Missing release states")

    captures_by_horizon: dict[int, list[dict[str, np.ndarray]]] = {}
    for horizon in HORIZONS:
        release = RELEASE_STEPS[horizon]
        pre_steps = CAPTURE_STEPS[horizon][:4]
        if any(step >= release for step in pre_steps):
            raise RuntimeError(f"Invalid pre-release frames for {horizon}")
        captures = [prefix_captures[step] for step in pre_steps]
        state_global, rng_global = release_states[release]
        current = release
        for target in CAPTURE_STEPS[horizon][4:]:
            while current < target:
                n_steps = min(JIT_MICROBATCH, target - current)
                rng_global, subkeys = old_c5._split_rng_batch(rng_global)
                state_global = global_stepper(n_steps)(
                    state_global,
                    subkeys,
                    params,
                    indices,
                )
                current += n_steps
            captures.append(
                _capture(state_global, rng_global, real_n=real_n)
            )
        if current != horizon or len(captures) != 8:
            raise RuntimeError(
                f"Horizon {horizon} ended at {current} with {len(captures)} frames"
            )
        captures_by_horizon[horizon] = captures
        print(
            f"[wall-fork] B={real_n}/{SIMULATION_BATCH_SIZE} "
            f"horizon={horizon} complete",
            flush=True,
        )

    initial_exact = {}
    for lane, row in enumerate(rows):
        free = _free_arrays(row)
        relative = free["steps"] - int(row["step"])
        zero = int(np.flatnonzero(relative == 0)[0])
        initial_exact[str(int(row["row_id"]))] = {
            "A": bool(
                np.array_equal(
                    captures_by_horizon[HORIZONS[0]][0]["A"][lane],
                    free["A"][zero],
                )
            ),
            "P": bool(
                np.array_equal(
                    captures_by_horizon[HORIZONS[0]][0]["P"][lane],
                    free["P"][zero],
                )
            ),
        }
    if not all(all(value.values()) for value in initial_exact.values()):
        raise RuntimeError(f"Wall/free initial state differs: {initial_exact}")

    if write_outputs:
        real_row_ids = [int(row["row_id"]) for row in rows]
        for lane, row in enumerate(rows):
            write_row = dict(row)
            if output_path_override is not None:
                write_row["wall_grid_path"] = str(output_path_override(row))
            _save_wall_lane(
                row=write_row,
                protocol=protocol,
                captures_by_horizon=captures_by_horizon,
                lane=lane,
                real_row_ids=real_row_ids,
                mass_audit={
                    str(release): value
                    for release, value in mass_audits.items()
                },
            )
    return {
        "n_real": real_n,
        "row_ids": [int(row["row_id"]) for row in rows],
        "elapsed_seconds": time.monotonic() - started,
        "mass_audits": mass_audits,
        "initial_exact": initial_exact,
        "captures_by_horizon": captures_by_horizon,
    }


def _create_engine(row: dict[str, str]) -> dict[str, Any]:
    engine = old_c5._create_wall_engine(row)
    runtime = engine["runtime"]
    engine["mass_projected_block_stepper"] = (
        mass_probe._make_block_state_stepper(
            runtime["block_substrate"],
            n_blocks=runtime["geometry"].n_blocks,
            original_batch_size=engine["original_batch_size"],
            valid_mask=runtime["valid_mask"],
            geometry=runtime["geometry"],
            mutation_spec=runtime["mutation_spec"],
            block_rt_gumbel=runtime["block_rt_gumbel"],
            project_mass=True,
        )
    )
    engine["clone_block_stepper"] = mass_probe._make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=engine["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
        project_mass=False,
    )
    engine["merge_batch"] = _merge_batch_fn(runtime["geometry"])
    return engine


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    rows, protocol = load_plan(args)
    selected = [
        row
        for row in rows
        if row["candidate_id"] == mass_probe.CANDIDATE_ID
        and int(row["point_id"]) == mass_probe.POINT_ID
    ]
    selected.sort(key=lambda row: int(row["branch_id"]))
    if [int(row["row_id"]) for row in selected] != list(
        mass_probe.EXPECTED_ROW_IDS
    ):
        raise RuntimeError("Selected preflight rows changed")
    for row in selected:
        audit = _free_audit(row, check_source_start=True)
        if not audit["ready"]:
            raise RuntimeError(f"Preflight free audit failed: {audit}")
    engine = _create_engine(selected[0])
    stepper_audit = mass_probe._stepper_audit(
        selected,
        engine,
        engine["clone_block_stepper"],
        engine["mass_projected_block_stepper"],
    )
    preflight_root = _resolve(args.output_root) / "preflight_outputs"
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    result = _simulate_wall_batch(
        selected,
        protocol=protocol,
        engine=engine,
        write_outputs=True,
        output_path_override=lambda row: (
            preflight_root / f"row_{int(row['row_id']):04d}.npz"
        ),
    )
    parity = {}
    targeted_root = _resolve(
        old_c5.DEFAULT_OUTPUT_ROOT
        / "selected_examples"
        / "rng_only_wall_probe_run_003_optimized_point_00"
        / "mass_projected_walls"
    )
    all_exact = True
    for row in selected:
        branch_id = int(row["branch_id"])
        current_path = preflight_root / f"row_{int(row['row_id']):04d}.npz"
        reference_path = targeted_root / f"branch_{branch_id:02d}.npz"
        with np.load(current_path, allow_pickle=False) as current, np.load(
            reference_path,
            allow_pickle=False,
        ) as reference:
            horizon_idx = int(
                np.flatnonzero(current["horizons"] == 20_000)[0]
            )
            ref_indices = mass_probe._common_indices(
                reference["relative_steps"]
            )
            fields = {
                "A": bool(
                    np.array_equal(
                        current["A"][horizon_idx],
                        reference["A"][ref_indices],
                    )
                ),
                "P": bool(
                    np.array_equal(
                        current["P"][horizon_idx],
                        reference["P"][ref_indices],
                    )
                ),
                "rng": bool(
                    np.array_equal(
                        current["resume_batch_rng_key"][horizon_idx],
                        reference["resume_batch_rng_key"][ref_indices],
                    )
                ),
            }
        exact = all(fields.values())
        all_exact = all_exact and exact
        parity[str(int(row["row_id"]))] = fields
    if not all_exact:
        raise RuntimeError(f"20k prefix-fork parity failed: {parity}")
    report = {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "stepper_clone": stepper_audit,
        "rng_only_free_origins_exact": True,
        "prefix_fork_20k_exact_vs_independent_replay": parity,
        "mass_audits": result["mass_audits"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    _write_json(_resolve(args.output_root) / "preflight.json", report)
    return report


def run_walls(args: argparse.Namespace) -> dict[str, Any]:
    rows, protocol = load_plan(args)
    preflight_path = _resolve(args.output_root) / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("Run preflight before production walls")
    preflight_report = json.loads(preflight_path.read_text())
    if (
        preflight_report.get("status") != "passed"
        or preflight_report.get("plan_sha256") != protocol["plan_sha256"]
    ):
        raise RuntimeError("Preflight identity/status mismatch")
    missing_free = [
        audit
        for row in rows
        if not (
            audit := _free_audit(row, check_source_start=False)
        )["ready"]
    ]
    if missing_free:
        raise RuntimeError(
            f"Free branches incomplete before walls: {missing_free[:3]}"
        )
    batches = [
        rows[start : start + SIMULATION_BATCH_SIZE]
        for start in range(0, len(rows), SIMULATION_BATCH_SIZE)
    ]
    if any(len(batch) != SIMULATION_BATCH_SIZE for batch in batches):
        raise RuntimeError("Production rows are not divisible by batch size")
    engine = _create_engine(rows[0])
    max_batches = (
        len(batches)
        if args.max_batches is None
        else min(len(batches), int(args.max_batches))
    )
    started = time.monotonic()
    completed_rows = 0
    simulated_batches = 0
    for batch_idx, batch in enumerate(batches[:max_batches]):
        audits = [
            _wall_output_audit(row, protocol)
            for row in batch
        ]
        invalid_nonempty = [
            audit
            for audit in audits
            if not audit["ready"]
            and (
                Path(audit["path"]).exists()
                or Path(audit["path"]).with_suffix(".metadata.json").exists()
            )
        ]
        if invalid_nonempty:
            raise RuntimeError(
                f"Invalid nonempty wall outputs: {invalid_nonempty[:3]}"
            )
        if all(audit["ready"] for audit in audits):
            completed_rows += len(batch)
            print(
                f"[walls] batch {batch_idx + 1}/{len(batches)} reused",
                flush=True,
            )
            continue
        result = _simulate_wall_batch(
            batch,
            protocol=protocol,
            engine=engine,
            write_outputs=True,
        )
        simulated_batches += 1
        completed_rows += result["n_real"]
        engine["snapshot_cache"].clear()
        elapsed = time.monotonic() - started
        rate = (batch_idx + 1) / max(elapsed, 1.0e-9)
        eta = (max_batches - batch_idx - 1) / max(rate, 1.0e-9)
        progress = {
            "status": "running",
            "plan_sha256": protocol["plan_sha256"],
            "batches_total": len(batches),
            "batches_targeted": max_batches,
            "batches_processed": batch_idx + 1,
            "rows_processed": completed_rows,
            "simulated_batches": simulated_batches,
            "elapsed_seconds": elapsed,
            "eta_seconds_for_target": eta,
        }
        _write_json(_resolve(args.output_root) / "walls_progress.json", progress)
        print(
            f"[walls] batch {batch_idx + 1}/{max_batches} complete; "
            f"eta={eta / 3600:.2f}h",
            flush=True,
        )
    all_audits = [_wall_output_audit(row, protocol) for row in rows]
    ready = sum(audit["ready"] for audit in all_audits)
    summary = {
        "status": "complete" if ready == len(rows) else "partial",
        "plan_sha256": protocol["plan_sha256"],
        "n_ready": ready,
        "n_expected": len(rows),
        "n_failed_or_missing": len(rows) - ready,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "simulated_batches_this_run": simulated_batches,
    }
    _write_json(_resolve(args.output_root) / "walls_summary.json", summary)
    return summary


def completion_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows, protocol = load_plan(args)
    source_start_cache: dict[
        tuple[str, int],
        dict[str, np.ndarray],
    ] = {}
    free_audits = [
        _free_audit(
            row,
            check_source_start=True,
            source_start_cache=source_start_cache,
        )
        for row in rows
    ]
    wall_audits = [_wall_output_audit(row, protocol) for row in rows]
    free_ready = sum(audit["ready"] for audit in free_audits)
    wall_ready = sum(audit["ready"] for audit in wall_audits)
    start_pairs = []
    pair_exact = True
    for row, free_audit, wall_audit in zip(
        rows,
        free_audits,
        wall_audits,
        strict=True,
    ):
        exact = bool(
            free_audit["ready"]
            and wall_audit["ready"]
            and free_audit["start_hashes"]["A"]
            == wall_audit["initial_A_sha256"]
            and free_audit["start_hashes"]["P"]
            == wall_audit["initial_P_sha256"]
        )
        pair_exact = pair_exact and exact
        if not exact:
            start_pairs.append(
                {
                    "row_id": int(row["row_id"]),
                    "free": free_audit,
                    "wall": wall_audit,
                }
            )
    report = {
        "status": (
            "passed"
            if free_ready == EXPECTED_ROWS
            and wall_ready == EXPECTED_ROWS
            and pair_exact
            else "failed"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "free_ready": free_ready,
        "wall_ready": wall_ready,
        "expected": EXPECTED_ROWS,
        "all_external_perturbations_zero": all(
            float(row[field]) == 0.0
            for row in rows
            for field in (
                "perturb_a_std",
                "perturb_p_std",
                "perturb_lagrangian_xy_std",
            )
        ),
        "all_wall_free_initial_A_P_exact": pair_exact,
        "initial_mismatches": start_pairs[:20],
    }
    _write_json(_resolve(args.output_root) / "completion_audit.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"Completion audit failed: {report}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("plan", "preflight", "free", "walls", "audit", "all"),
        default="all",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--old-c5-root", default=str(DEFAULT_OLD_C5_ROOT))
    parser.add_argument("--sweep-plan", default=str(DEFAULT_SWEEP_PLAN))
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase == "plan":
        rows, protocol = build_plan(args)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "n_rows": len(rows),
                    "plan_sha256": protocol["plan_sha256"],
                },
                indent=2,
            )
        )
    elif args.phase == "preflight":
        print(json.dumps(_jsonable(preflight(args)), indent=2))
    elif args.phase == "free":
        print(json.dumps(_jsonable(generate_free(args)), indent=2))
    elif args.phase == "walls":
        print(json.dumps(_jsonable(run_walls(args)), indent=2))
    elif args.phase == "audit":
        print(json.dumps(_jsonable(completion_audit(args)), indent=2))
    else:
        build_plan(args)
        preflight(args)
        generate_free(args)
        run_walls(args)
        completion_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
