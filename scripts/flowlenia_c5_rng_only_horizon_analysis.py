#!/usr/bin/env python3
"""Paper analysis for the RNG-only, mass-preserving Flow-Lenia C5 grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import binomtest, wilcoxon

from flowlenia_c5_branch_analysis import _foundation_model_fingerprint
from paper_suite_c2_branching import (
    _embedding_chamfer_cosine,
    _pool_spatial,
    _render_apf_rgb,
)


ANALYSIS_VERSION = "flowlenia-c5-rng-only-horizon-analysis-v1"
SIMULATION_PROTOCOL = "flowlenia-c5-rng-only-mass-projected-horizon-grid-v2"
FOUNDATION_MODEL = "clip"
HORIZONS = (5_000, 10_000, 15_000, 20_000, 30_000)
PRIMARY_HORIZON = 20_000
VIDEO_HORIZON = 30_000
CAPTURE_COUNT = 8
ROW_COUNT = 1_800
POINT_COUNT = 600
CANDIDATE_COUNT = 40
RUN_COUNT = 10
FIELD_SCALES = (1, 2, 4)
PAIR_TYPES = (
    "paired_same_seed",
    "paired_off_seed",
    "free_within",
    "walls_within",
)
SUMMARY_METRICS = (
    "excess_clip_post_release",
    "excess_clip_sync_post_release",
    "excess_clip_full_future",
    "spread_delta_clip_post_release",
    "paired_same_seed_clip_post_release",
    "free_within_clip_post_release",
    "walls_within_clip_post_release",
    "pair_alignment_clip_post_release",
    "excess_field_post_release",
    "excess_A_post_release",
    "excess_P_post_release",
    "excess_mass_rel_post_release",
    "paired_same_seed_mass_delta_rel_post_release",
)
REQUIRED_FIGURE_STEMS = (
    "c5_primary_by_run",
    "c5_run_contrasts",
    "flow_c5_frustration_clean",
    "flow_c5_frustration_paper",
    "c5_candidate_heatmap",
    "c5_condition_effects",
    "c5_time_resolved",
    "c5_clip_vs_field",
    "c5_delta_h_relation",
    "c5_horizon_sensitivity",
    "c5_mass_diagnostics",
)
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2"
)
PLAN_HASH_FIELDS = (
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
    "free_provenance",
)
OPT_COLOR = "#D55E00"
RANDOM_COLOR = "#0072B2"
WALL_COLOR = "#CC79A7"
FREE_COLOR = "#009E73"
NEUTRAL = "#58616B"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


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
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def _analysis_fingerprint() -> dict[str, Any]:
    paths = {
        "analysis": Path(__file__).resolve(),
        "rendering_and_clip_helpers": (
            _REPO_ROOT / "scripts/paper_suite_c2_branching.py"
        ),
        "foundation_wrapper": _REPO_ROOT / "foundation_models/clip.py",
    }
    files = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }
    identity = {"analysis_version": ANALYSIS_VERSION, "files": files}
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _load_inputs(output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan_path = output_root / "plan.csv"
    protocol_path = output_root / "protocol.json"
    simulation_audit_path = output_root / "completion_audit.json"
    if not all(path.exists() for path in (plan_path, protocol_path, simulation_audit_path)):
        raise FileNotFoundError("C5 plan, protocol, or simulation audit is missing")
    with plan_path.open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    protocol = json.loads(protocol_path.read_text())
    simulation_audit = json.loads(simulation_audit_path.read_text())
    plan_identity = [
        {field: str(row[field]) for field in PLAN_HASH_FIELDS}
        for row in raw_rows
    ]
    found_plan_hash = _identity_sha256(plan_identity)
    if found_plan_hash != protocol.get("plan_sha256"):
        raise RuntimeError(
            f"Plan identity mismatch: {found_plan_hash} != {protocol.get('plan_sha256')}"
        )
    if protocol.get("protocol_version") != SIMULATION_PROTOCOL:
        raise RuntimeError(f"Unexpected simulation protocol: {protocol.get('protocol_version')}")
    if simulation_audit.get("status") != "passed":
        raise RuntimeError("Simulation completion audit did not pass")
    if int(simulation_audit.get("free_ready", -1)) != ROW_COUNT:
        raise RuntimeError("Simulation audit does not contain all free rows")
    if int(simulation_audit.get("wall_ready", -1)) != ROW_COUNT:
        raise RuntimeError("Simulation audit does not contain all wall rows")
    if not simulation_audit.get("all_external_perturbations_zero", False):
        raise RuntimeError("Simulation audit contains external perturbations")
    if not simulation_audit.get("all_wall_free_initial_A_P_exact", False):
        raise RuntimeError("Simulation audit lacks exact free/wall starts")
    if tuple(int(value) for value in protocol.get("horizons", [])) != HORIZONS:
        raise RuntimeError("Horizon grid changed")
    for relative, expected_sha in protocol["simulation_code_files"].items():
        found_sha = _sha256_file(_REPO_ROOT / relative)
        if found_sha != expected_sha:
            raise RuntimeError(f"Simulation source changed after run: {relative}")

    plan = pd.DataFrame(raw_rows)
    integer_columns = (
        "row_id",
        "run_idx",
        "trial_idx",
        "candidate_idx",
        "pair_id",
        "point_id",
        "window_index",
        "step",
        "branch_id",
        "branch_seed",
    )
    float_columns = (
        "delta_h",
        "perturb_a_std",
        "perturb_p_std",
        "perturb_lagrangian_xy_std",
    )
    for column in integer_columns:
        plan[column] = plan[column].astype(int)
    for column in float_columns:
        plan[column] = plan[column].astype(float)
    if len(plan) != ROW_COUNT or plan["row_id"].tolist() != list(range(ROW_COUNT)):
        raise RuntimeError("Plan must contain contiguous row IDs 0..1799")
    if any(not np.all(plan[column].to_numpy() == 0.0) for column in float_columns[1:]):
        raise RuntimeError("Plan contains nonzero state perturbation")
    counts = plan.groupby(["candidate_id", "point_id"]).size()
    if len(counts) != POINT_COUNT or not bool((counts == 3).all()):
        raise RuntimeError("Expected 600 points with three branches each")
    candidates = plan[["run_idx", "candidate_id", "candidate_kind", "candidate_idx"]].drop_duplicates()
    if len(candidates) != CANDIDATE_COUNT:
        raise RuntimeError("Expected 40 candidates")
    for run_idx, rows in candidates.groupby("run_idx"):
        if len(rows[rows["candidate_kind"] == "optimized"]) != 1:
            raise RuntimeError(f"Run {run_idx} lacks one optimized candidate")
        if len(rows[rows["candidate_kind"] == "random"]) != 3:
            raise RuntimeError(f"Run {run_idx} lacks three random candidates")
    return plan, protocol


def _free_npz(branch_dir: Path) -> Path:
    paths = sorted((branch_dir / "apf_logs").glob("*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one free APF file in {branch_dir}, found {len(paths)}")
    return paths[0]


def _load_free(row: Any, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    path = _free_npz(_resolve(row.free_branch_dir))
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in ("steps", "A", "P")}
    order = np.argsort(arrays["steps"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    relative = np.asarray(arrays["steps"], dtype=np.int64) - int(row.step)
    expected = np.asarray(protocol["free_capture_union"], dtype=np.int64)
    if not np.array_equal(relative, expected):
        raise RuntimeError(f"Free capture grid mismatch for row {row.row_id}")
    arrays["relative_steps"] = relative
    arrays["path"] = np.asarray(str(path))
    return arrays


def _load_wall(row: Any, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    path = _resolve(row.wall_grid_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in ("horizons", "release_steps", "capture_steps", "A", "P")
        }
    if not np.array_equal(arrays["horizons"], np.asarray(HORIZONS, dtype=np.int32)):
        raise RuntimeError(f"Wall horizon grid mismatch for row {row.row_id}")
    expected_capture = np.asarray(
        [protocol["capture_steps"][str(horizon)] for horizon in HORIZONS],
        dtype=np.int32,
    )
    if not np.array_equal(arrays["capture_steps"], expected_capture):
        raise RuntimeError(f"Wall capture grid mismatch for row {row.row_id}")
    expected_release = np.asarray(
        [protocol["release_steps"][str(horizon)] for horizon in HORIZONS],
        dtype=np.int32,
    )
    if not np.array_equal(arrays["release_steps"], expected_release):
        raise RuntimeError(f"Wall release grid mismatch for row {row.row_id}")
    arrays["path"] = np.asarray(str(path))
    return arrays


def _source_path(row: Any, arm: str) -> Path:
    if arm == "free":
        return _free_npz(_resolve(row.free_branch_dir))
    return _resolve(row.wall_grid_path)


def _cache_path(cache_dir: Path, row_id: int, arm: str) -> Path:
    return cache_dir / f"row_{row_id:04d}_{arm}.npz"


def _read_embedding_cache(
    path: Path,
    *,
    arm: str,
    source_sha256: str,
    expected_steps: np.ndarray,
    model_identity_sha256: str,
    analysis_identity_sha256: str,
    plan_sha256: str,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            checks = (
                str(np.asarray(data["analysis_version"]).item()) == ANALYSIS_VERSION,
                str(np.asarray(data["arm"]).item()) == arm,
                str(np.asarray(data["source_sha256"]).item()) == source_sha256,
                str(np.asarray(data["model_identity_sha256"]).item()) == model_identity_sha256,
                str(np.asarray(data["analysis_identity_sha256"]).item()) == analysis_identity_sha256,
                str(np.asarray(data["plan_sha256"]).item()) == plan_sha256,
                np.array_equal(np.asarray(data["relative_steps"]), expected_steps),
            )
            z = np.asarray(data["z"], dtype=np.float32)
    except Exception:
        return None
    expected_shape = (23, 512) if arm == "free" else (5, 8, 512)
    if not all(checks) or z.shape != expected_shape or not np.all(np.isfinite(z)):
        return None
    norms = np.linalg.norm(z, axis=-1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-5:
        return None
    return z


def _save_embedding_cache(
    path: Path,
    *,
    z: np.ndarray,
    row_id: int,
    arm: str,
    source_path: Path,
    source_sha256: str,
    relative_steps: np.ndarray,
    model_identity_sha256: str,
    analysis_identity_sha256: str,
    plan_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(z, dtype=np.float32)
    value /= np.clip(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12, None)
    np.savez_compressed(
        path,
        z=value,
        row_id=np.asarray(row_id, dtype=np.int32),
        arm=np.asarray(arm),
        source_path=np.asarray(str(source_path)),
        source_sha256=np.asarray(source_sha256),
        relative_steps=np.asarray(relative_steps, dtype=np.int64),
        model_identity_sha256=np.asarray(model_identity_sha256),
        analysis_identity_sha256=np.asarray(analysis_identity_sha256),
        plan_sha256=np.asarray(plan_sha256),
        analysis_version=np.asarray(ANALYSIS_VERSION),
        inference_mode=np.asarray(
            "authoritative_c2_unjitted_single_frame_grouped_device_sync"
        ),
    )


def _embed_one(fm: Any, frame: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    value = np.asarray(
        jax.device_get(fm.embed_img(jnp.asarray(frame, dtype=jnp.float32))),
        dtype=np.float32,
    ).reshape(-1)
    return value / np.clip(np.linalg.norm(value), 1e-12, None)


def _embed_fields(
    fm: Any,
    *,
    a_value: np.ndarray,
    p_value: np.ndarray,
    exact_frame_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    import jax
    import jax.numpy as jnp

    leading = a_value.shape[:-3]
    flat_a = np.asarray(a_value).reshape((-1, *a_value.shape[-3:]))
    flat_p = np.asarray(p_value).reshape((-1, *p_value.shape[-3:]))
    rgb = np.asarray(_render_apf_rgb({"A": flat_a, "P": flat_p}), dtype=np.float32)
    frame_shas = []
    new_frames: dict[str, np.ndarray] = {}
    for frame in rgb:
        contiguous = np.ascontiguousarray(frame, dtype=np.float32)
        frame_sha = _array_sha256(contiguous)
        frame_shas.append(frame_sha)
        if frame_sha not in exact_frame_cache and frame_sha not in new_frames:
            new_frames[frame_sha] = contiguous
    queued = [
        fm.embed_img(jnp.asarray(frame, dtype=jnp.float32))
        for frame in new_frames.values()
    ]
    if queued:
        values = jax.device_get(queued)
        for frame_sha, value in zip(new_frames, values, strict=True):
            z = np.asarray(value, dtype=np.float32).reshape(-1)
            exact_frame_cache[frame_sha] = z / np.clip(
                np.linalg.norm(z), 1e-12, None
            )
    embeddings = [exact_frame_cache[frame_sha] for frame_sha in frame_shas]
    reused = len(frame_shas) - len(new_frames)
    return np.stack(embeddings).reshape((*leading, -1)), reused


def _embedding_fingerprint() -> dict[str, Any]:
    implementation = {
        name: inspect.getsource(function)
        for name, function in (
            ("array_sha256", _array_sha256),
            ("read_embedding_cache", _read_embedding_cache),
            ("save_embedding_cache", _save_embedding_cache),
            ("embed_one", _embed_one),
            ("embed_fields", _embed_fields),
        )
    }
    identity = {
        "embedding_version": "flowlenia-c5-authoritative-single-frame-cache-v1",
        "implementation_sha256": _identity_sha256(implementation),
        "rendering_helpers_sha256": _sha256_file(
            _REPO_ROOT / "scripts/paper_suite_c2_branching.py"
        ),
        "foundation_wrapper_sha256": _sha256_file(
            _REPO_ROOT / "foundation_models/clip.py"
        ),
        "inference_mode": (
            "authoritative_c2_unjitted_single_frame; grouped device "
            "synchronization only"
        ),
        "inference_batch_frames": 1,
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def preflight(plan: pd.DataFrame, protocol: dict[str, Any], output_root: Path) -> dict[str, Any]:
    import foundation_models

    rows = plan[(plan["candidate_id"] == "run_000_optimized") & (plan["point_id"] == 0)].sort_values("branch_id")
    if rows["branch_id"].tolist() != [0, 1, 2]:
        raise RuntimeError("Preflight point does not contain branches 0,1,2")
    zero_field_errors = []
    for row in rows.itertuples():
        free = _load_free(row, protocol)
        wall = _load_wall(row, protocol)
        for horizon_idx in range(len(HORIZONS)):
            zero_field_errors.extend(
                [
                    float(np.max(np.abs(np.asarray(free["A"][0], dtype=np.float32) - np.asarray(wall["A"][horizon_idx, 0], dtype=np.float32)))),
                    float(np.max(np.abs(np.asarray(free["P"][0], dtype=np.float32) - np.asarray(wall["P"][horizon_idx, 0], dtype=np.float32)))),
                ]
            )
    first = rows.iloc[0]
    row = next(pd.DataFrame([first]).itertuples(index=False))
    free = _load_free(row, protocol)
    wall = _load_wall(row, protocol)
    horizon_idx = HORIZONS.index(PRIMARY_HORIZON)
    offsets = np.asarray(protocol["capture_steps"][str(PRIMARY_HORIZON)], dtype=np.int64)
    free_indices = np.asarray([int(np.flatnonzero(free["relative_steps"] == step)[0]) for step in offsets])
    free_rgb = _render_apf_rgb({"A": free["A"][free_indices], "P": free["P"][free_indices]})
    wall_rgb = _render_apf_rgb({"A": wall["A"][horizon_idx], "P": wall["P"][horizon_idx]})
    render_zero = float(np.max(np.abs(np.asarray(free_rgb[0]) - np.asarray(wall_rgb[0]))))
    fm = foundation_models.create_foundation_model(FOUNDATION_MODEL)
    free_z = _embed_one(fm, np.asarray(free_rgb[0], dtype=np.float32))
    wall_z = _embed_one(fm, np.asarray(wall_rgb[0], dtype=np.float32))
    embedding_zero = float(np.max(np.abs(free_z - wall_z)))
    exact_frame_cache: dict[str, np.ndarray] = {}
    free_union_z, _ = _embed_fields(
        fm,
        a_value=free["A"],
        p_value=free["P"],
        exact_frame_cache=exact_frame_cache,
    )
    wall_grid_z, reused_frames = _embed_fields(
        fm,
        a_value=wall["A"],
        p_value=wall["P"],
        exact_frame_cache=exact_frame_cache,
    )
    grouped_sync_max_abs = float(
        np.max(np.abs(free_union_z[0] - free_z))
    )
    post_release = np.flatnonzero(offsets > PRIMARY_HORIZON // 2)
    preflight_chamfer = _embedding_chamfer_cosine(
        free_union_z[free_indices][post_release],
        wall_grid_z[horizon_idx][post_release],
    )
    windows = {
        "wall_phase": np.flatnonzero(
            (offsets > 0) & (offsets <= PRIMARY_HORIZON // 2)
        ),
        "post_release": post_release,
        "full_future": np.flatnonzero(offsets > 0),
    }
    preflight_pair = _pair_record(
        base={"horizon_steps": PRIMARY_HORIZON},
        pair_type="paired_same_seed",
        left_arm="free",
        right_arm="walls",
        left_branch=0,
        right_branch=0,
        left_z=free_union_z[free_indices],
        right_z=wall_grid_z[horizon_idx],
        left_fields=_with_pyramid(
            free["A"][free_indices], free["P"][free_indices]
        ),
        right_fields=_with_pyramid(
            wall["A"][horizon_idx], wall["P"][horizon_idx]
        ),
        windows=windows,
    )
    finite_pair_metrics = all(
        np.isfinite(value)
        for key, value in preflight_pair.items()
        if key.startswith(
            ("clip_", "field_", "A_", "P_", "mass_rel_", "mass_delta_rel_")
        )
    )
    report = {
        "status": (
            "passed"
            if max(zero_field_errors) == 0.0
            and render_zero == 0.0
            and embedding_zero <= 1e-6
            and free_union_z.shape == (23, 512)
            and wall_grid_z.shape == (5, 8, 512)
            and np.isfinite(preflight_chamfer)
            and finite_pair_metrics
            and grouped_sync_max_abs <= 1e-5
            else "failed"
        ),
        "analysis_version": ANALYSIS_VERSION,
        "simulation_protocol": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "row_ids": rows["row_id"].astype(int).tolist(),
        "horizons": HORIZONS,
        "primary_horizon": PRIMARY_HORIZON,
        "all_capture_windows_have_four_post_release_frames": all(
            sum(int(step) > int(protocol["release_steps"][str(horizon)]) for step in protocol["capture_steps"][str(horizon)]) == 4
            for horizon in HORIZONS
        ),
        "max_abs_free_wall_start_A_P": max(zero_field_errors),
        "max_abs_rendered_start": render_zero,
        "max_abs_independent_clip_start": embedding_zero,
        "clip_start_cosine_distance": float(np.clip(1.0 - np.dot(free_z, wall_z), 0.0, 2.0)),
        "free_embedding_shape": free_union_z.shape,
        "wall_embedding_shape": wall_grid_z.shape,
        "post_release_clip_chamfer_20k": preflight_chamfer,
        "post_release_field_distance_20k": preflight_pair[
            "field_post_release"
        ],
        "all_preflight_pair_metrics_finite": finite_pair_metrics,
        "max_abs_grouped_sync_vs_per_frame_sync": grouped_sync_max_abs,
        "grouped_sync_tolerance": 1e-5,
        "unique_rendered_frames": len(exact_frame_cache),
        "exact_frame_reuse": reused_frames,
        "jax_backend": __import__("jax").default_backend(),
        "analysis_fingerprint": _analysis_fingerprint(),
    }
    _write_json(output_root / "analysis_preflight.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"Analysis preflight failed: {report}")
    return report


def ensure_embeddings(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    output_root: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    import jax
    import foundation_models

    if json.loads((output_root / "analysis_preflight.json").read_text()).get("status") != "passed":
        raise RuntimeError("Analysis preflight must pass before embeddings")
    analysis_fingerprint = _analysis_fingerprint()
    embedding_fingerprint = _embedding_fingerprint()
    embedding_identity = embedding_fingerprint["identity_sha256"]
    fm = foundation_models.create_foundation_model(FOUNDATION_MODEL)
    model_fingerprint = _foundation_model_fingerprint(fm)
    model_identity = model_fingerprint["identity_sha256"]
    cache_dir = output_root / "clip_horizon_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    started = time.monotonic()
    for task_idx, row in enumerate(plan.sort_values("row_id").itertuples(), start=1):
        for arm in ("free", "walls"):
            source = _source_path(row, arm)
            source_sha = _sha256_file(source)
            relative = (
                np.asarray(protocol["free_capture_union"], dtype=np.int64)
                if arm == "free"
                else np.asarray([protocol["capture_steps"][str(h)] for h in HORIZONS], dtype=np.int64)
            )
            target = _cache_path(cache_dir, int(row.row_id), arm)
            z = None if force else _read_embedding_cache(
                target,
                arm=arm,
                source_sha256=source_sha,
                expected_steps=relative,
                model_identity_sha256=model_identity,
                analysis_identity_sha256=embedding_identity,
                plan_sha256=protocol["plan_sha256"],
            )
            status = "reused" if z is not None else "pending"
            if z is None:
                pending.append(
                    {
                        "row": row,
                        "arm": arm,
                        "source": source,
                        "source_sha": source_sha,
                        "relative": relative,
                        "target": target,
                    }
                )
            manifest_rows.append(
                {
                    "row_id": int(row.row_id),
                    "run_idx": int(row.run_idx),
                    "candidate_id": str(row.candidate_id),
                    "candidate_kind": str(row.candidate_kind),
                    "candidate_idx": int(row.candidate_idx),
                    "point_id": int(row.point_id),
                    "branch_id": int(row.branch_id),
                    "arm": arm,
                    "source_path": str(source),
                    "source_sha256": source_sha,
                    "embedding_cache": str(target),
                    "status": status,
                    "model_identity_sha256": model_identity,
                    "embedding_identity_sha256": embedding_identity,
                }
            )
        if task_idx % 100 == 0:
            elapsed = time.monotonic() - started
            print(f"[embeddings] audited {task_idx}/{ROW_COUNT} rows pending={len(pending)} elapsed={elapsed / 60:.1f}m", flush=True)

    exact_frame_cache: dict[str, np.ndarray] = {}
    reused_frames = 0
    compute_started = time.monotonic()
    manifest_index = {(row["row_id"], row["arm"]): idx for idx, row in enumerate(manifest_rows)}
    for completed, item in enumerate(pending, start=1):
        row = item["row"]
        if item["arm"] == "free":
            arrays = _load_free(row, protocol)
        else:
            arrays = _load_wall(row, protocol)
        z, reused = _embed_fields(
            fm,
            a_value=arrays["A"],
            p_value=arrays["P"],
            exact_frame_cache=exact_frame_cache,
        )
        reused_frames += reused
        _save_embedding_cache(
            item["target"],
            z=z,
            row_id=int(row.row_id),
            arm=item["arm"],
            source_path=item["source"],
            source_sha256=item["source_sha"],
            relative_steps=item["relative"],
            model_identity_sha256=model_identity,
            analysis_identity_sha256=embedding_identity,
            plan_sha256=protocol["plan_sha256"],
        )
        manifest_rows[manifest_index[(int(row.row_id), item["arm"])]]["status"] = "computed"
        if completed % 10 == 0 or completed == len(pending):
            elapsed = time.monotonic() - compute_started
            rate = completed / max(elapsed, 1e-9)
            eta = (len(pending) - completed) / max(rate, 1e-9)
            progress = {
                "status": "running",
                "completed_branches": completed,
                "pending_branches": len(pending),
                "total_branches": 2 * ROW_COUNT,
                "exact_frame_reuse": reused_frames,
                "unique_embedded_frames": len(exact_frame_cache),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }
            _write_json(output_root / "embedding_progress.json", progress)
            print(f"[embeddings] computed {completed}/{len(pending)} branches rate={rate:.2f}/s eta={eta / 60:.1f}m frame_reuse={reused_frames}", flush=True)

    manifest = pd.DataFrame(manifest_rows).sort_values(["row_id", "arm"]).reset_index(drop=True)
    manifest["embedding_cache_sha256"] = [_sha256_file(Path(path)) for path in manifest["embedding_cache"]]
    _write_table(output_root / "embedding_manifest.csv", manifest)

    cache_lookup = {(int(row.row_id), str(row.arm)): Path(row.embedding_cache) for row in manifest.itertuples()}
    zero_errors = []
    zero_cosines = []
    for row_id in range(ROW_COUNT):
        with np.load(cache_lookup[(row_id, "free")], allow_pickle=False) as data:
            free_zero = np.asarray(data["z"], dtype=np.float32)[0]
        with np.load(cache_lookup[(row_id, "walls")], allow_pickle=False) as data:
            wall_zero = np.asarray(data["z"], dtype=np.float32)[:, 0]
        zero_errors.extend(np.max(np.abs(wall_zero - free_zero[None, :]), axis=1).tolist())
        zero_cosines.extend(np.clip(1.0 - wall_zero @ free_zero, 0.0, 2.0).tolist())
    zero_audit = {
        "status": "passed" if len(zero_errors) == ROW_COUNT * len(HORIZONS) and max(zero_errors) <= 1e-6 else "failed",
        "n_pairs": len(zero_errors),
        "n_embedding_exact": int(np.sum(np.asarray(zero_errors) == 0.0)),
        "max_embedding_abs": max(zero_errors),
        "max_cosine_distance": max(zero_cosines),
    }
    _write_json(output_root / "embedding_pair_zero_audit.json", zero_audit)
    if zero_audit["status"] != "passed":
        raise RuntimeError(f"Embedding start parity failed: {zero_audit}")
    embedding_protocol = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_fingerprint": analysis_fingerprint,
        "producer_analysis_identity_sha256": analysis_fingerprint[
            "identity_sha256"
        ],
        "embedding_fingerprint": embedding_fingerprint,
        "embedding_identity_sha256": embedding_identity,
        "simulation_protocol": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol["simulation_code_bundle_sha256"],
        "foundation_model": FOUNDATION_MODEL,
        "model_fingerprint": model_fingerprint,
        "model_identity_sha256": model_identity,
        "inference_mode": (
            "authoritative_c2_unjitted_single_frame; batch=1 per forward; "
            "device synchronization grouped per source artifact"
        ),
        "inference_batch_frames": 1,
        "free_frames_per_branch": 23,
        "wall_frames_per_branch": 40,
        "n_embedding_caches": len(manifest),
        "n_computed": int((manifest["status"] == "computed").sum()),
        "n_reused": int((manifest["status"] == "reused").sum()),
        "identical_start_audit": zero_audit,
        "embedding_manifest_sha256": _sha256_file(output_root / "embedding_manifest.csv"),
        "runtime": {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    _write_json(output_root / "embedding_protocol.json", embedding_protocol)
    _write_json(
        output_root / "embedding_progress.json",
        {
            "status": "complete",
            "n_embedding_caches": len(manifest),
            "n_computed": embedding_protocol["n_computed"],
            "n_reused": embedding_protocol["n_reused"],
            "exact_frame_reuse": reused_frames,
            "unique_embedded_frames_this_process": len(exact_frame_cache),
        },
    )
    return manifest


def _load_z(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=np.float32)


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _with_pyramid(a_value: np.ndarray, p_value: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "A": np.asarray(a_value),
        "P": np.asarray(p_value),
        "_pyramid": {},
    }
    for key in ("A", "P"):
        source = np.asarray(result[key], dtype=np.float32)
        result["_pyramid"][key] = {
            scale: _pool_spatial(source, scale)
            for scale in FIELD_SCALES
        }
    return result


def _select_fields(fields: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {
        "A": fields["A"][indices],
        "P": fields["P"][indices],
        "_pyramid": {
            key: {
                scale: fields["_pyramid"][key][scale][indices]
                for scale in FIELD_SCALES
            }
            for key in ("A", "P")
        },
    }


def _pair_record(
    *,
    base: dict[str, Any],
    pair_type: str,
    left_arm: str,
    right_arm: str,
    left_branch: int,
    right_branch: int,
    left_z: np.ndarray,
    right_z: np.ndarray,
    left_fields: dict[str, Any],
    right_fields: dict[str, Any],
    windows: dict[str, np.ndarray],
) -> dict[str, Any]:
    row = {
        **base,
        "pair_type": pair_type,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_branch": int(left_branch),
        "right_branch": int(right_branch),
    }
    sync = np.clip(1.0 - np.sum(left_z * right_z, axis=-1), 0.0, 2.0)
    left_mass = np.sum(np.asarray(left_fields["A"], dtype=np.float64), axis=(1, 2, 3))
    right_mass = np.sum(np.asarray(right_fields["A"], dtype=np.float64), axis=(1, 2, 3))
    mass_scale = np.clip(np.abs(left_mass), 1e-12, None)
    field_mse: dict[str, dict[int, np.ndarray]] = {"A": {}, "P": {}}
    for key in ("A", "P"):
        for scale in FIELD_SCALES:
            difference = (
                np.asarray(left_fields["_pyramid"][key][scale], dtype=np.float32)
                - np.asarray(right_fields["_pyramid"][key][scale], dtype=np.float32)
            )
            field_mse[key][scale] = np.mean(
                difference * difference,
                axis=tuple(range(1, difference.ndim)),
            )
    for window_name, indices in windows.items():
        row[f"clip_{window_name}"] = _embedding_chamfer_cosine(
            left_z[indices], right_z[indices]
        )
        row[f"clip_sync_{window_name}"] = float(np.mean(sync[indices]))
        row[f"mass_rel_{window_name}"] = float(
            np.mean(np.abs(right_mass[indices] - left_mass[indices]) / mass_scale[indices])
        )
        row[f"mass_delta_rel_{window_name}"] = float(
            np.mean((right_mass[indices] - left_mass[indices]) / mass_scale[indices])
        )
        components = {}
        for key in ("A", "P"):
            components[key] = float(
                np.mean(
                    [
                        np.sqrt(np.mean(field_mse[key][scale][indices]))
                        for scale in FIELD_SCALES
                    ]
                )
            )
            row[f"{key}_{window_name}"] = components[key]
        row[f"field_{window_name}"] = float(np.mean([components["A"], components["P"]]))
    return row


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_boot: int = 20_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _run_stat(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = _bootstrap_median_ci(values, seed=seed)
    nonzero = values[values != 0.0]
    sign_p = (
        float(
            binomtest(
                int(np.sum(nonzero > 0)),
                n=int(nonzero.size),
                p=0.5,
                alternative="greater",
            ).pvalue
        )
        if nonzero.size
        else 1.0
    )
    if nonzero.size:
        try:
            wilcoxon_p = float(
                wilcoxon(
                    values,
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
    else:
        wilcoxon_p = 1.0
    return {
        "n_runs": int(values.size),
        "median": _median(values),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "bootstrap_median_ci_low": low,
        "bootstrap_median_ci_high": high,
        "n_positive": int(np.sum(values > 0)),
        "n_negative": int(np.sum(values < 0)),
        "sign_test_p_greater": sign_p,
        "wilcoxon_p_greater": wilcoxon_p,
    }


def compute_metrics(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    embedding_manifest_path = output_root / "embedding_manifest.csv"
    embedding_protocol_path = output_root / "embedding_protocol.json"
    if not embedding_manifest_path.exists() or not embedding_protocol_path.exists():
        raise RuntimeError("Embedding artifacts are missing")
    manifest = pd.read_csv(embedding_manifest_path)
    embedding_protocol = json.loads(embedding_protocol_path.read_text())
    current_analysis = _analysis_fingerprint()
    if (
        embedding_protocol.get("embedding_identity_sha256")
        != _embedding_fingerprint()["identity_sha256"]
    ):
        raise RuntimeError("Embedding cache belongs to different embedding implementation")
    if embedding_protocol.get("inference_batch_frames") != 1:
        raise RuntimeError("Embedding inference was not single-frame")
    if len(manifest) != 2 * ROW_COUNT:
        raise RuntimeError("Embedding manifest must contain 3600 rows")
    model_identities = set(manifest["model_identity_sha256"].astype(str))
    if len(model_identities) != 1:
        raise RuntimeError("Embedding manifest contains multiple model identities")
    model_identity = next(iter(model_identities))
    cache_lookup = {
        (int(row.row_id), str(row.arm)): Path(row.embedding_cache)
        for row in manifest.itertuples()
    }
    metric_inputs = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_identity_sha256": current_analysis["identity_sha256"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_protocol": protocol["protocol_version"],
        "simulation_code_bundle_sha256": protocol["simulation_code_bundle_sha256"],
        "embedding_manifest_sha256": _sha256_file(embedding_manifest_path),
        "embedding_protocol_sha256": _sha256_file(embedding_protocol_path),
        "model_identity_sha256": model_identity,
        "horizons": HORIZONS,
        "primary_horizon": PRIMARY_HORIZON,
    }
    metric_input_sha256 = _identity_sha256(metric_inputs)
    pair_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    groups = list(plan.groupby(["candidate_id", "point_id"], sort=True))
    started = time.monotonic()
    for group_idx, ((candidate_id, point_id), rows) in enumerate(groups, start=1):
        rows = rows.sort_values("branch_id")
        if rows["branch_id"].tolist() != [0, 1, 2]:
            raise RuntimeError(f"Bad branches for {candidate_id}/point_{point_id}")
        stable_columns = (
            "run_idx",
            "trial_idx",
            "candidate_kind",
            "candidate_idx",
            "condition",
            "pair_id",
            "window_index",
            "step",
            "delta_h",
        )
        if any(rows[column].nunique(dropna=False) != 1 for column in stable_columns):
            raise RuntimeError(f"Inconsistent point metadata for {candidate_id}/{point_id}")
        first = rows.iloc[0]
        common = {
            "run_idx": int(first["run_idx"]),
            "trial_idx": int(first["trial_idx"]),
            "candidate_id": str(candidate_id),
            "candidate_kind": str(first["candidate_kind"]),
            "candidate_idx": int(first["candidate_idx"]),
            "condition": str(first["condition"]),
            "point_id": int(point_id),
            "pair_id": int(first["pair_id"]),
            "step": int(first["step"]),
            "delta_h": float(first["delta_h"]),
        }
        free_all: list[dict[str, Any]] = []
        wall_all: list[dict[str, Any]] = []
        free_z_all: list[np.ndarray] = []
        wall_z_all: list[np.ndarray] = []
        free_relative = None
        wall_capture = None
        for row in rows.itertuples():
            free = _load_free(row, protocol)
            wall = _load_wall(row, protocol)
            if free_relative is None:
                free_relative = np.asarray(free["relative_steps"], dtype=np.int64)
                wall_capture = np.asarray(wall["capture_steps"], dtype=np.int64)
            free_all.append(_with_pyramid(free["A"], free["P"]))
            wall_all.append(
                _with_pyramid(
                    wall["A"].reshape((-1, *wall["A"].shape[-3:])),
                    wall["P"].reshape((-1, *wall["P"].shape[-3:])),
                )
            )
            free_z_all.append(_load_z(cache_lookup[(int(row.row_id), "free")]))
            wall_z_all.append(_load_z(cache_lookup[(int(row.row_id), "walls")]))
        assert free_relative is not None and wall_capture is not None

        for horizon_idx, horizon in enumerate(HORIZONS):
            relative = np.asarray(protocol["capture_steps"][str(horizon)], dtype=np.int64)
            release_step = int(protocol["release_steps"][str(horizon)])
            free_indices = np.asarray(
                [int(np.flatnonzero(free_relative == step)[0]) for step in relative],
                dtype=np.int64,
            )
            wall_indices = horizon_idx * CAPTURE_COUNT + np.arange(CAPTURE_COUNT)
            if not np.array_equal(wall_capture[horizon_idx], relative):
                raise RuntimeError(f"Wall offsets changed for horizon {horizon}")
            windows = {
                "wall_phase": np.flatnonzero((relative > 0) & (relative <= release_step)),
                "post_release": np.flatnonzero(relative > release_step),
                "full_future": np.flatnonzero(relative > 0),
            }
            if [indices.size for indices in windows.values()] != [3, 4, 7]:
                raise RuntimeError(f"Unexpected windows for horizon {horizon}")
            base = {
                **common,
                "horizon_steps": int(horizon),
                "release_step": release_step,
            }
            free_fields = [_select_fields(item, free_indices) for item in free_all]
            wall_fields = [_select_fields(item, wall_indices) for item in wall_all]
            free_z = [item[free_indices] for item in free_z_all]
            wall_z = [item[horizon_idx] for item in wall_z_all]
            current_pairs: list[dict[str, Any]] = []
            for branch_id in range(3):
                current_pairs.append(
                    _pair_record(
                        base=base,
                        pair_type="paired_same_seed",
                        left_arm="free",
                        right_arm="walls",
                        left_branch=branch_id,
                        right_branch=branch_id,
                        left_z=free_z[branch_id],
                        right_z=wall_z[branch_id],
                        left_fields=free_fields[branch_id],
                        right_fields=wall_fields[branch_id],
                        windows=windows,
                    )
                )
                frame_distance = np.clip(
                    1.0 - np.sum(free_z[branch_id] * wall_z[branch_id], axis=-1),
                    0.0,
                    2.0,
                )
                for frame_idx, (relative_step, distance) in enumerate(zip(relative, frame_distance, strict=True)):
                    frame_rows.append(
                        {
                            **base,
                            "branch_id": branch_id,
                            "frame_idx": frame_idx,
                            "relative_step": int(relative_step),
                            "relative_fraction": float(relative_step / horizon),
                            "wall_active": bool(0 < relative_step <= release_step),
                            "paired_cosine_distance": float(distance),
                        }
                    )
            for left_branch in range(3):
                for right_branch in range(3):
                    if left_branch == right_branch:
                        continue
                    current_pairs.append(
                        _pair_record(
                            base=base,
                            pair_type="paired_off_seed",
                            left_arm="free",
                            right_arm="walls",
                            left_branch=left_branch,
                            right_branch=right_branch,
                            left_z=free_z[left_branch],
                            right_z=wall_z[right_branch],
                            left_fields=free_fields[left_branch],
                            right_fields=wall_fields[right_branch],
                            windows=windows,
                        )
                    )
            for left_branch, right_branch in combinations(range(3), 2):
                current_pairs.append(
                    _pair_record(
                        base=base,
                        pair_type="free_within",
                        left_arm="free",
                        right_arm="free",
                        left_branch=left_branch,
                        right_branch=right_branch,
                        left_z=free_z[left_branch],
                        right_z=free_z[right_branch],
                        left_fields=free_fields[left_branch],
                        right_fields=free_fields[right_branch],
                        windows=windows,
                    )
                )
                current_pairs.append(
                    _pair_record(
                        base=base,
                        pair_type="walls_within",
                        left_arm="walls",
                        right_arm="walls",
                        left_branch=left_branch,
                        right_branch=right_branch,
                        left_z=wall_z[left_branch],
                        right_z=wall_z[right_branch],
                        left_fields=wall_fields[left_branch],
                        right_fields=wall_fields[right_branch],
                        windows=windows,
                    )
                )
            pair_rows.extend(current_pairs)
            point = dict(base)
            for metric in ("clip", "clip_sync", "field", "A", "P", "mass_rel", "mass_delta_rel"):
                for window_name in windows:
                    column = f"{metric}_{window_name}"
                    by_type = {
                        pair_type: _median(
                            pair[column]
                            for pair in current_pairs
                            if pair["pair_type"] == pair_type
                        )
                        for pair_type in PAIR_TYPES
                    }
                    for pair_type, value in by_type.items():
                        point[f"{pair_type}_{column}"] = value
                    point[f"excess_{column}"] = by_type["paired_same_seed"] - by_type["free_within"]
                    point[f"spread_delta_{column}"] = by_type["walls_within"] - by_type["free_within"]
                    point[f"pair_alignment_{column}"] = by_type["paired_same_seed"] - by_type["paired_off_seed"]
            point_rows.append(point)
        if group_idx % 10 == 0 or group_idx == len(groups):
            elapsed = time.monotonic() - started
            rate = group_idx / max(elapsed, 1e-9)
            eta = (len(groups) - group_idx) / max(rate, 1e-9)
            _write_json(
                output_root / "metrics_progress.json",
                {
                    "status": "running",
                    "points_processed": group_idx,
                    "points_total": len(groups),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
            )
            print(f"[metrics] points {group_idx}/{len(groups)} rate={rate:.2f}/s eta={eta / 60:.1f}m", flush=True)

    pair_frame = pd.DataFrame(pair_rows)
    frame_frame = pd.DataFrame(frame_rows)
    point_frame = pd.DataFrame(point_rows)
    expected_pairs = POINT_COUNT * len(HORIZONS) * 15
    expected_frames = POINT_COUNT * len(HORIZONS) * 3 * CAPTURE_COUNT
    if len(pair_frame) != expected_pairs or len(frame_frame) != expected_frames:
        raise RuntimeError("Pair/frame metric row count mismatch")
    if len(point_frame) != POINT_COUNT * len(HORIZONS):
        raise RuntimeError("Point metric row count mismatch")

    id_columns = ("run_idx", "trial_idx", "candidate_id", "candidate_kind", "candidate_idx", "horizon_steps")
    metric_columns = [
        column
        for column in point_frame.columns
        if column.startswith(("paired_", "free_", "walls_", "excess_", "spread_", "pair_alignment_"))
        and column != "pair_id"
    ]
    candidate_rows = []
    for (candidate_id, horizon), rows in point_frame.groupby(["candidate_id", "horizon_steps"], sort=True):
        first = rows.iloc[0]
        out = {column: first[column] for column in id_columns}
        out["n_points"] = int(len(rows))
        for column in metric_columns:
            out[column] = _median(rows[column])
        for condition in ("high", "mid", "low"):
            condition_rows = rows[rows["condition"] == condition]
            for column in (
                "excess_clip_post_release",
                "excess_field_post_release",
                "paired_same_seed_clip_post_release",
                "free_within_clip_post_release",
            ):
                out[f"{condition}_{column}"] = _median(condition_rows[column])
        candidate_rows.append(out)
    candidate_frame = pd.DataFrame(candidate_rows)
    if len(candidate_frame) != CANDIDATE_COUNT * len(HORIZONS):
        raise RuntimeError("Candidate summary row count mismatch")

    candidate_time = (
        frame_frame.groupby(
            [
                "run_idx",
                "trial_idx",
                "candidate_id",
                "candidate_kind",
                "candidate_idx",
                "horizon_steps",
                "release_step",
                "frame_idx",
                "relative_step",
                "relative_fraction",
                "wall_active",
            ],
            as_index=False,
        )["paired_cosine_distance"]
        .median()
        .rename(columns={"paired_cosine_distance": "candidate_median_paired_cosine"})
    )

    run_rows: list[dict[str, Any]] = []
    for (run_idx, horizon), rows in candidate_frame.groupby(["run_idx", "horizon_steps"], sort=True):
        optimized = rows[rows["candidate_kind"] == "optimized"]
        random = rows[rows["candidate_kind"] == "random"].sort_values("candidate_idx")
        if len(optimized) != 1 or len(random) != 3:
            raise RuntimeError(f"Run {run_idx}/horizon {horizon} candidate mismatch")
        out: dict[str, Any] = {"run_idx": int(run_idx), "horizon_steps": int(horizon)}
        for metric in SUMMARY_METRICS:
            opt_value = float(optimized.iloc[0][metric])
            random_values = random[metric].to_numpy(dtype=np.float64)
            out[f"opt_{metric}"] = opt_value
            for idx, value in enumerate(random_values):
                out[f"random_{idx}_{metric}"] = float(value)
            out[f"random_median_{metric}"] = float(np.median(random_values))
            out[f"contrast_{metric}"] = opt_value - float(np.median(random_values))
        run_rows.append(out)
    run_frame = pd.DataFrame(run_rows)

    condition_rows = []
    for (run_idx, horizon), rows in candidate_frame.groupby(["run_idx", "horizon_steps"], sort=True):
        optimized = rows[rows["candidate_kind"] == "optimized"].iloc[0]
        random = rows[rows["candidate_kind"] == "random"]
        for condition in ("high", "mid", "low"):
            for metric in ("excess_clip_post_release", "excess_field_post_release"):
                column = f"{condition}_{metric}"
                opt_value = float(optimized[column])
                random_values = random[column].to_numpy(dtype=np.float64)
                condition_rows.append(
                    {
                        "run_idx": int(run_idx),
                        "horizon_steps": int(horizon),
                        "condition": condition,
                        "metric": metric,
                        "optimized": opt_value,
                        "random_median": float(np.median(random_values)),
                        "contrast": opt_value - float(np.median(random_values)),
                    }
                )
    condition_frame = pd.DataFrame(condition_rows)

    run_time_rows = []
    for (run_idx, horizon, frame_idx), rows in candidate_time.groupby(
        ["run_idx", "horizon_steps", "frame_idx"], sort=True
    ):
        optimized = rows[rows["candidate_kind"] == "optimized"]
        random = rows[rows["candidate_kind"] == "random"]
        first = rows.iloc[0]
        opt_value = float(optimized["candidate_median_paired_cosine"].iloc[0])
        random_median = float(np.median(random["candidate_median_paired_cosine"]))
        run_time_rows.append(
            {
                "run_idx": int(run_idx),
                "horizon_steps": int(horizon),
                "frame_idx": int(frame_idx),
                "relative_step": int(first["relative_step"]),
                "relative_fraction": float(first["relative_fraction"]),
                "wall_active": bool(first["wall_active"]),
                "optimized": opt_value,
                "random_median": random_median,
                "contrast": opt_value - random_median,
            }
        )
    run_time_frame = pd.DataFrame(run_time_rows)

    statistical_rows = []
    statistical_summary: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "primary_horizon": PRIMARY_HORIZON,
        "primary_metric": "excess_clip_post_release",
        "candidate_aggregation": "median over 15 selected points",
        "random_control_aggregation": "median of three matched random candidates per run",
        "run_effect": "optimized minus matched-random median",
        "horizon_role": "20k confirmatory; 5k, 10k, 15k, and 30k sensitivity analyses",
        "tests": {},
    }
    for horizon in HORIZONS:
        horizon_rows = run_frame[run_frame["horizon_steps"] == horizon]
        statistical_summary["tests"][str(horizon)] = {}
        for metric_idx, metric in enumerate(SUMMARY_METRICS):
            values = horizon_rows[f"contrast_{metric}"].to_numpy(dtype=np.float64)
            result = _run_stat(values, seed=20_260_721 + horizon + metric_idx)
            statistical_summary["tests"][str(horizon)][metric] = result
            statistical_rows.append(
                {
                    "horizon_steps": horizon,
                    "metric": metric,
                    "primary": bool(horizon == PRIMARY_HORIZON and metric == "excess_clip_post_release"),
                    "inference_role": (
                        "confirmatory_primary"
                        if horizon == PRIMARY_HORIZON and metric == "excess_clip_post_release"
                        else "sensitivity_or_secondary_unadjusted"
                    ),
                    **result,
                }
            )
    primary = statistical_summary["tests"][str(PRIMARY_HORIZON)]["excess_clip_post_release"]
    statistical_summary["primary_result"] = primary
    statistical_frame = pd.DataFrame(statistical_rows)

    tables = {
        "branch_pair_metrics.csv": pair_frame,
        "branch_frame_metrics.csv": frame_frame,
        "point_metrics.csv": point_frame,
        "candidate_summary.csv": candidate_frame,
        "candidate_frame_summary.csv": candidate_time,
        "run_summary.csv": run_frame,
        "run_condition_summary.csv": condition_frame,
        "run_frame_summary.csv": run_time_frame,
        "c5_statistical_table.csv": statistical_frame,
    }
    for filename, frame in tables.items():
        _write_table(output_root / filename, frame)
    (output_root / "c5_statistical_table.tex").write_text(
        statistical_frame.to_latex(
            index=False,
            escape=True,
            float_format=lambda value: f"{value:.4g}",
        )
    )
    _write_json(output_root / "statistical_summary.json", statistical_summary)
    metric_protocol = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_fingerprint": current_analysis,
        "metric_inputs": metric_inputs,
        "metric_input_sha256": metric_input_sha256,
        "rendering": "clip(sum(A_channels) * P_rgb, 0, 1)",
        "inference": "authoritative C2 unjitted single-frame CLIP",
        "capture_relative_steps": {str(h): protocol["capture_steps"][str(h)] for h in HORIZONS},
        "release_steps": {str(h): protocol["release_steps"][str(h)] for h in HORIZONS},
        "windows": {
            "wall_phase": "three retained frames with 0 < step <= horizon/2",
            "post_release": "four retained frames with step > horizon/2",
            "full_future": "all seven nonzero retained frames",
        },
        "clip_distance": "symmetric Chamfer over L2-normalized CLIP frame embeddings with cosine cost",
        "field_distance": {
            "fields": ["A", "P"],
            "scales": FIELD_SCALES,
            "definition": "mean per-field multiscale RMS",
        },
        "point_primary": "median same-seed free/walls post-release CLIP Chamfer minus median pairwise free/free post-release CLIP Chamfer",
        "candidate_aggregation": "median over 15 points",
        "run_primary": "optimized candidate minus median of three matched random candidates",
        "primary_horizon": PRIMARY_HORIZON,
        "multiplicity": "20k CLIP primary is confirmatory; all other horizons and metrics are unadjusted sensitivity diagnostics",
    }
    _write_json(output_root / "metric_protocol.json", metric_protocol)
    artifact_paths = [
        *tables.keys(),
        "c5_statistical_table.tex",
        "statistical_summary.json",
        "metric_protocol.json",
    ]
    artifacts = {
        name: {
            "path": str(output_root / name),
            "sha256": _sha256_file(output_root / name),
            "bytes": int((output_root / name).stat().st_size),
            "rows": len(tables[name]) if name in tables else None,
        }
        for name in artifact_paths
    }
    artifact_manifest = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_identity_sha256": current_analysis["identity_sha256"],
        "plan_sha256": protocol["plan_sha256"],
        "model_identity_sha256": model_identity,
        "metric_input_sha256": metric_input_sha256,
        "artifacts": artifacts,
    }
    artifact_manifest["manifest_identity_sha256"] = _identity_sha256(artifact_manifest)
    _write_json(output_root / "metric_artifact_manifest.json", artifact_manifest)
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "metric_input_sha256": metric_input_sha256,
        "metric_artifact_manifest_sha256": _sha256_file(output_root / "metric_artifact_manifest.json"),
        "n_branch_pairs": len(pair_frame),
        "n_branch_frames": len(frame_frame),
        "n_points": len(point_frame),
        "n_candidates": len(candidate_frame),
        "n_run_horizons": len(run_frame),
        "primary_horizon": PRIMARY_HORIZON,
        "primary_metric": "excess_clip_post_release",
        "primary_result": primary,
    }
    _write_json(output_root / "metrics_summary.json", summary)
    _write_json(output_root / "metrics_progress.json", {"status": "complete", **summary})
    return summary


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def build_plots(output_root: Path) -> dict[str, Any]:
    _setup_plot_style()
    manifest_path = output_root / "metric_artifact_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Metric artifact manifest is missing")
    metric_manifest = json.loads(manifest_path.read_text())
    analysis_fingerprint = _analysis_fingerprint()
    if (
        metric_manifest.get("status") != "complete"
        or metric_manifest.get("analysis_identity_sha256")
        != analysis_fingerprint["identity_sha256"]
    ):
        raise RuntimeError("Metric artifacts belong to different analysis code")
    candidates = pd.read_csv(output_root / "candidate_summary.csv")
    runs = pd.read_csv(output_root / "run_summary.csv")
    conditions = pd.read_csv(output_root / "run_condition_summary.csv")
    run_time = pd.read_csv(output_root / "run_frame_summary.csv")
    points = pd.read_csv(output_root / "point_metrics.csv")
    stats = json.loads((output_root / "statistical_summary.json").read_text())
    figure_dir = output_root / "figures"
    generated: list[Path] = []
    primary = "excess_clip_post_release"
    primary_candidates = candidates[candidates["horizon_steps"] == PRIMARY_HORIZON]
    primary_runs = runs[runs["horizon_steps"] == PRIMARY_HORIZON].sort_values("run_idx")

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for run_idx in range(RUN_COUNT):
        rows = primary_candidates[primary_candidates["run_idx"] == run_idx]
        opt = float(rows[rows["candidate_kind"] == "optimized"][primary].iloc[0])
        random_values = (
            rows[rows["candidate_kind"] == "random"]
            .sort_values("candidate_idx")[primary]
            .to_numpy(dtype=float)
        )
        ax.plot([run_idx, run_idx], [np.median(random_values), opt], color="#B5BBC2", lw=1.0, zorder=1)
        ax.scatter(run_idx + np.asarray([-0.13, 0.0, 0.13]), random_values, s=24, color=RANDOM_COLOR, alpha=0.75, zorder=2)
        ax.scatter(run_idx, opt, s=42, color=OPT_COLOR, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(range(RUN_COUNT), [f"{idx:03d}" for idx in range(RUN_COUNT)])
    ax.set_xlabel("Optimization run")
    ax.set_ylabel("Excess post-release CLIP divergence")
    ax.set_title("C5 paired frustration effect at 20k steps")
    ax.scatter([], [], color=OPT_COLOR, label="Optimized")
    ax.scatter([], [], color=RANDOM_COLOR, label="Random controls")
    ax.legend(loc="best", ncols=2)
    generated.extend(_save_figure(fig, figure_dir / "c5_primary_by_run"))

    contrast_column = f"contrast_{primary}"
    values = primary_runs[contrast_column].to_numpy(dtype=float)
    primary_test = stats["tests"][str(PRIMARY_HORIZON)][primary]

    def contrast_bar(stem: str, *, title: bool) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
        colors = np.where(values >= 0, "#3FA447", "#D94C4C")
        ax.bar(np.arange(RUN_COUNT), values, color=colors, width=0.7)
        ax.scatter(np.arange(RUN_COUNT), values, color=colors, edgecolor="white", lw=0.7, s=32, zorder=3)
        ax.axhline(0, color="#222222", lw=0.9)
        ax.set_xticks(range(RUN_COUNT), [f"{idx:03d}" for idx in range(RUN_COUNT)])
        ax.set_xlabel("Matched optimization run")
        ax.set_ylabel("F(optimized) - median F(random)")
        if title:
            ax.set_title(
                "C5 mass-preserving RNG-only contrast at 20k\n"
                f"positive={int(np.sum(values > 0))}/{RUN_COUNT}   "
                f"median={np.median(values):.4g}   "
                f"sign-test p={primary_test['sign_test_p_greater']:.3g}"
            )
        generated.extend(_save_figure(fig, figure_dir / stem))

    contrast_bar("c5_run_contrasts", title=True)
    contrast_bar("flow_c5_frustration_clean", title=True)
    contrast_bar("flow_c5_frustration_paper", title=False)

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15.5, 4.5), constrained_layout=True, sharey=True)
    matrices = []
    for horizon in HORIZONS:
        matrix = np.empty((RUN_COUNT, 4), dtype=float)
        horizon_rows = candidates[candidates["horizon_steps"] == horizon]
        for run_idx in range(RUN_COUNT):
            rows = horizon_rows[horizon_rows["run_idx"] == run_idx]
            matrix[run_idx, 0] = float(rows[rows["candidate_kind"] == "optimized"][primary].iloc[0])
            matrix[run_idx, 1:] = (
                rows[rows["candidate_kind"] == "random"]
                .sort_values("candidate_idx")[primary]
                .to_numpy(dtype=float)
            )
        matrices.append(matrix)
    bound = max(float(np.nanpercentile(np.abs(np.stack(matrices)), 98)), 1e-8)
    image = None
    for axis, horizon, matrix in zip(axes, HORIZONS, matrices, strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound))
        axis.set_xticks(range(4), ["Opt", "R0", "R1", "R2"], rotation=45, ha="right")
        axis.set_title(f"{horizon // 1000}k")
    axes[0].set_yticks(range(RUN_COUNT), [f"Run {idx:03d}" for idx in range(RUN_COUNT)])
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82)
    colorbar.set_label("Excess post-release CLIP divergence")
    fig.suptitle("Candidate-level frustration across continuation horizons")
    generated.extend(_save_figure(fig, figure_dir / "c5_candidate_heatmap"))

    condition_colors = {"high": OPT_COLOR, "mid": "#E69F00", "low": RANDOM_COLOR}
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    x = np.asarray(HORIZONS, dtype=float) / 1000.0
    for condition_idx, condition in enumerate(("high", "mid", "low")):
        medians, lows, highs = [], [], []
        for horizon in HORIZONS:
            values_condition = conditions[
                (conditions["horizon_steps"] == horizon)
                & (conditions["condition"] == condition)
                & (conditions["metric"] == primary)
            ]["contrast"].to_numpy(dtype=float)
            low, high = _bootstrap_median_ci(values_condition, seed=71_000 + horizon + condition_idx)
            medians.append(float(np.median(values_condition)))
            lows.append(low)
            highs.append(high)
        ax.plot(x, medians, marker="o", color=condition_colors[condition], label=f"{condition.capitalize()} Delta-H")
        ax.fill_between(x, lows, highs, color=condition_colors[condition], alpha=0.12)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.axvline(PRIMARY_HORIZON / 1000, color=NEUTRAL, lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("Continuation horizon (thousand steps)")
    ax.set_ylabel("Median optimized-minus-random effect")
    ax.set_title("C5 effect by selected-state Delta-H stratum")
    ax.legend(loc="best")
    generated.extend(_save_figure(fig, figure_dir / "c5_condition_effects"))

    fig, (ax_abs, ax_contrast) = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True, sharex=True)
    horizon_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(HORIZONS)))
    for horizon, color in zip(HORIZONS, horizon_colors, strict=True):
        med_abs, med_contrast = [], []
        for frame_idx in range(CAPTURE_COUNT):
            frame_values = run_time[
                (run_time["horizon_steps"] == horizon)
                & (run_time["frame_idx"] == frame_idx)
            ]
            med_abs.append(float(np.median(frame_values["optimized"])))
            med_contrast.append(float(np.median(frame_values["contrast"])))
        fraction = (
            run_time[run_time["horizon_steps"] == horizon]
            .sort_values("frame_idx")
            .drop_duplicates("frame_idx")["relative_fraction"]
            .to_numpy(dtype=float)
        )
        ax_abs.plot(fraction, med_abs, color=color, marker="o", ms=3, label=f"{horizon // 1000}k")
        ax_contrast.plot(fraction, med_contrast, color=color, marker="o", ms=3)
    for axis in (ax_abs, ax_contrast):
        axis.axvspan(0, 0.5, color=WALL_COLOR, alpha=0.10)
        axis.axvline(0.5, color=WALL_COLOR, lw=1.0, ls="--")
        axis.set_xlabel("Fraction of continuation horizon")
    ax_abs.set_ylabel("Paired frame cosine distance")
    ax_abs.set_title("Absolute wall effect")
    ax_abs.legend(loc="best", ncols=2)
    ax_contrast.axhline(0, color="#222222", lw=0.8)
    ax_contrast.set_ylabel("Optimized minus random median")
    ax_contrast.set_title("Run-blocked contrast")
    generated.extend(_save_figure(fig, figure_dir / "c5_time_resolved"))

    fig, ax = plt.subplots(figsize=(5.8, 4.5), constrained_layout=True)
    for kind, marker, label in (("optimized", "o", "Optimized"), ("random", "x", "Random")):
        subset = candidates[candidates["candidate_kind"] == kind]
        scatter = ax.scatter(
            subset["excess_field_post_release"],
            subset["excess_clip_post_release"],
            c=subset["horizon_steps"] / 1000.0,
            cmap="viridis",
            marker=marker,
            s=34 if kind == "optimized" else 24,
            alpha=0.8,
            label=label,
        )
    ax.axhline(0, color="#222222", lw=0.7)
    ax.axvline(0, color="#222222", lw=0.7)
    ax.set_xlabel("Excess post-release multiscale A/P divergence")
    ax.set_ylabel("Excess post-release CLIP divergence")
    ax.set_title("Pixel-field and perceptual C5 estimands")
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Horizon (thousand steps)")
    generated.extend(_save_figure(fig, figure_dir / "c5_clip_vs_field"))

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15.5, 3.3), constrained_layout=True, sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS, strict=True):
        horizon_points = points[points["horizon_steps"] == horizon]
        for kind, color in (("optimized", OPT_COLOR), ("random", RANDOM_COLOR)):
            subset = horizon_points[horizon_points["candidate_kind"] == kind]
            axis.scatter(subset["delta_h"], subset[primary], color=color, s=8, alpha=0.35)
        axis.axhline(0, color="#222222", lw=0.7)
        axis.set_title(f"{horizon // 1000}k")
        axis.set_xlabel("Delta-H")
    axes[0].set_ylabel("Excess CLIP divergence")
    fig.suptitle("Selected-state Delta-H and frustration effect")
    generated.extend(_save_figure(fig, figure_dir / "c5_delta_h_relation"))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for run_idx in range(RUN_COUNT):
        run_values = runs[runs["run_idx"] == run_idx].sort_values("horizon_steps")
        ax.plot(run_values["horizon_steps"] / 1000.0, run_values[contrast_column], color="#AEB5BD", lw=0.8, alpha=0.8)
    medians, lows, highs = [], [], []
    for horizon in HORIZONS:
        horizon_values = runs[runs["horizon_steps"] == horizon][contrast_column].to_numpy(dtype=float)
        low, high = _bootstrap_median_ci(horizon_values, seed=81_000 + horizon)
        medians.append(float(np.median(horizon_values)))
        lows.append(low)
        highs.append(high)
    ax.plot(x, medians, color=FREE_COLOR, marker="o", lw=2.2, label="Median run effect")
    ax.fill_between(x, lows, highs, color=FREE_COLOR, alpha=0.18, label="95% bootstrap CI")
    ax.axhline(0, color="#222222", lw=0.8)
    ax.axvline(PRIMARY_HORIZON / 1000, color=OPT_COLOR, lw=1.0, ls="--", label="Confirmatory horizon")
    ax.set_xticks(x)
    ax.set_xlabel("Continuation horizon (thousand steps)")
    ax.set_ylabel("Optimized minus matched-random frustration")
    ax.set_title("C5 horizon sensitivity")
    ax.legend(loc="best")
    generated.extend(_save_figure(fig, figure_dir / "c5_horizon_sensitivity"))

    fig, (ax_abs, ax_signed) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for column, axis, title in (
        ("contrast_excess_mass_rel_post_release", ax_abs, "Excess absolute mass divergence"),
        ("contrast_paired_same_seed_mass_delta_rel_post_release", ax_signed, "Signed wall-minus-free mass"),
    ):
        medians = []
        for horizon in HORIZONS:
            horizon_values = runs[runs["horizon_steps"] == horizon][column].to_numpy(dtype=float)
            medians.append(float(np.median(horizon_values)))
            axis.scatter(np.full(horizon_values.shape, horizon / 1000.0), horizon_values, color=NEUTRAL, s=14, alpha=0.55)
        axis.plot(x, medians, color=WALL_COLOR, marker="o", lw=2.0)
        axis.axhline(0, color="#222222", lw=0.8)
        axis.set_xticks(x)
        axis.set_xlabel("Horizon (thousand steps)")
        axis.set_title(title)
    ax_abs.set_ylabel("Optimized minus random median")
    generated.extend(_save_figure(fig, figure_dir / "c5_mass_diagnostics"))

    stems = sorted({path.stem for path in generated})
    if stems != sorted(REQUIRED_FIGURE_STEMS):
        raise RuntimeError(f"Figure stem mismatch: {stems}")
    published_dir = output_root.parents[1] / "figures" / "c5_rng_only_mass_preserving_horizon_grid"
    published_dir.mkdir(parents=True, exist_ok=True)
    for path in generated:
        shutil.copy2(path, published_dir / path.name)
    figure_inputs = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_identity_sha256": analysis_fingerprint["identity_sha256"],
        "metric_artifact_manifest_sha256": _sha256_file(manifest_path),
        "metric_artifact_identity_sha256": metric_manifest["manifest_identity_sha256"],
    }
    files = {
        path.name: {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "published_path": str(published_dir / path.name),
            "published_sha256": _sha256_file(published_dir / path.name),
        }
        for path in generated
    }
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "figure_inputs": figure_inputs,
        "figure_input_sha256": _identity_sha256(figure_inputs),
        "required_stems": REQUIRED_FIGURE_STEMS,
        "n_figure_stems": len(stems),
        "n_figure_files": len(generated),
        "published_dir": str(published_dir),
        "files": files,
        "primary_result": primary_test,
    }
    _write_json(output_root / "figures_summary.json", summary)
    return summary


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if not path.exists():
        filename = "VeraBd.ttf" if bold else "Vera.ttf"
        path = Path("/usr/share/fonts/truetype/ttf-bitstream-vera") / filename
    return ImageFont.truetype(str(path), size=size)


def _rgb_u8(a_value: np.ndarray, p_value: np.ndarray) -> np.ndarray:
    rgb = _render_apf_rgb({"A": a_value, "P": p_value})
    return np.rint(np.asarray(rgb) * 255.0).clip(0, 255).astype(np.uint8)


def _draw_wall_lines(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for fraction in (1 / 3, 2 / 3):
        x_value = int(round(width * fraction))
        y_value = int(round(height * fraction))
        draw.line((x_value, 0, x_value, height), fill=(255, 255, 255), width=3)
        draw.line((0, y_value, width, y_value), fill=(255, 255, 255), width=3)
        draw.line((x_value + 3, 0, x_value + 3, height), fill=(20, 20, 20), width=1)
        draw.line((0, y_value + 3, width, y_value + 3), fill=(20, 20, 20), width=1)


def _video_frame(
    free_rgb: list[np.ndarray],
    wall_rgb: list[np.ndarray],
    *,
    frame_idx: int,
    relative_step: int,
    candidate_id: str,
    point_id: int,
    release_step: int,
) -> np.ndarray:
    tile_size = 256
    header = 52
    left = 102
    gap = 4
    canvas = Image.new(
        "RGB",
        (left + 3 * tile_size + 2 * gap, header + 2 * tile_size + gap),
        color=(245, 247, 249),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 7),
        f"{candidate_id} | point {point_id:02d} | +{relative_step:,} / {VIDEO_HORIZON:,} steps",
        fill=(20, 25, 30),
        font=_font(18, bold=True),
    )
    wall_active = bool(0 < relative_step <= release_step)
    phase = "walls active" if wall_active else ("shared start" if relative_step == 0 else "walls removed")
    draw.text(
        (canvas.width - 146, 30),
        phase,
        fill=(145, 45, 100) if wall_active else (20, 100, 70),
        font=_font(13, bold=True),
    )
    for branch_id in range(3):
        x_value = left + branch_id * (tile_size + gap)
        draw.text((x_value + 84, 32), f"Branch {branch_id}", fill=(45, 50, 56), font=_font(13))
        free_image = Image.fromarray(free_rgb[branch_id][frame_idx]).resize(
            (tile_size, tile_size), resample=Image.Resampling.NEAREST
        )
        wall_image = Image.fromarray(wall_rgb[branch_id][frame_idx]).resize(
            (tile_size, tile_size), resample=Image.Resampling.NEAREST
        )
        if wall_active:
            _draw_wall_lines(wall_image)
        canvas.paste(free_image, (x_value, header))
        canvas.paste(wall_image, (x_value, header + tile_size + gap))
    draw.text((18, header + tile_size // 2 - 10), "Free", fill=(20, 80, 55), font=_font(18, bold=True))
    draw.text((8, header + tile_size + gap + tile_size // 2 - 23), "Walls", fill=(145, 45, 100), font=_font(18, bold=True))
    draw.text((4, header + tile_size + gap + tile_size // 2 + 2), "mass fixed", fill=(145, 45, 100), font=_font(13))
    return np.asarray(canvas)


def _validate_video(path: Path, *, expected_frames: int) -> dict[str, Any]:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    result: dict[str, Any] = {
        "passed": False,
        "frames": 0,
        "expected_frames": int(expected_frames),
        "width": 0,
        "height": 0,
    }
    if not path.exists() or path.stat().st_size <= 10_000:
        return result
    reader = None
    try:
        reader = imageio.get_reader(path)
        expected_shape = None
        for frame in reader:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise RuntimeError(f"Unexpected video frame shape: {array.shape}")
            if expected_shape is None:
                expected_shape = array.shape
            elif expected_shape != array.shape:
                raise RuntimeError("Video frame shapes changed")
            result["frames"] += 1
        if expected_shape is not None:
            result["height"] = int(expected_shape[0])
            result["width"] = int(expected_shape[1])
        result["passed"] = bool(
            result["frames"] == expected_frames
            and result["width"] > 0
            and result["height"] > 0
        )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if reader is not None:
            reader.close()
    return result


def _representative_points(plan: pd.DataFrame) -> dict[int, int]:
    optimized = plan[plan["candidate_kind"] == "optimized"]
    result = {}
    for run_idx, rows in optimized.groupby("run_idx"):
        by_point = rows.groupby("point_id", as_index=False)["delta_h"].first()
        selected = by_point.sort_values(["delta_h", "point_id"], ascending=[False, True]).iloc[0]
        result[int(run_idx)] = int(selected["point_id"])
    if set(result) != set(range(RUN_COUNT)):
        raise RuntimeError("Representative-point selection lacks runs")
    return result


def build_videos(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    output_root: Path,
    *,
    fps: int,
    hold_frames: int,
    force: bool,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    embedding_manifest = pd.read_csv(output_root / "embedding_manifest.csv")
    source_lookup = {
        (int(row.row_id), str(row.arm)): {
            "path": str(row.source_path),
            "sha256": str(row.source_sha256),
        }
        for row in embedding_manifest.itertuples()
    }
    analysis_identity = _analysis_fingerprint()["identity_sha256"]
    videos_dir = output_root / "videos_by_candidate"
    videos_dir.mkdir(parents=True, exist_ok=True)
    representative = _representative_points(plan)
    candidates = (
        plan[["run_idx", "candidate_id", "candidate_kind", "candidate_idx"]]
        .drop_duplicates()
        .sort_values(["run_idx", "candidate_kind", "candidate_idx"])
    )
    horizon_idx = HORIZONS.index(VIDEO_HORIZON)
    offsets = np.asarray(protocol["capture_steps"][str(VIDEO_HORIZON)], dtype=np.int64)
    release_step = int(protocol["release_steps"][str(VIDEO_HORIZON)])
    expected_frames = CAPTURE_COUNT * int(hold_frames)
    manifest_rows = []
    for candidate_number, candidate in enumerate(candidates.itertuples(), start=1):
        point_id = representative[int(candidate.run_idx)]
        rows = plan[
            (plan["candidate_id"] == candidate.candidate_id)
            & (plan["point_id"] == point_id)
        ].sort_values("branch_id")
        if rows["branch_id"].tolist() != [0, 1, 2]:
            raise RuntimeError(f"Video point lacks branches: {candidate.candidate_id}")
        sources = [
            {
                "row_id": int(row.row_id),
                "branch_id": int(row.branch_id),
                "free": source_lookup[(int(row.row_id), "free")],
                "walls": source_lookup[(int(row.row_id), "walls")],
            }
            for row in rows.itertuples()
        ]
        video_input = {
            "analysis_version": ANALYSIS_VERSION,
            "analysis_identity_sha256": analysis_identity,
            "plan_sha256": protocol["plan_sha256"],
            "candidate_id": str(candidate.candidate_id),
            "point_id": point_id,
            "horizon_steps": VIDEO_HORIZON,
            "release_step": release_step,
            "capture_steps": offsets,
            "fps": int(fps),
            "hold_frames": int(hold_frames),
            "sources": sources,
        }
        video_input["input_sha256"] = _identity_sha256(video_input)
        output = videos_dir / f"{candidate.candidate_id}_point_{point_id:02d}_horizon_{VIDEO_HORIZON:05d}.mp4"
        provenance_path = output.with_suffix(".provenance.json")
        existing_provenance = {}
        if provenance_path.exists():
            try:
                existing_provenance = json.loads(provenance_path.read_text())
            except Exception:
                existing_provenance = {}
        validation = _validate_video(output, expected_frames=expected_frames)
        output_sha = _sha256_file(output) if output.exists() else ""
        reusable = bool(
            not force
            and validation["passed"]
            and existing_provenance.get("input_sha256") == video_input["input_sha256"]
            and existing_provenance.get("video_sha256") == output_sha
        )
        if reusable:
            status = "reused"
        else:
            free_rgb: list[np.ndarray] = []
            wall_rgb: list[np.ndarray] = []
            for row in rows.itertuples():
                free = _load_free(row, protocol)
                wall = _load_wall(row, protocol)
                free_indices = np.asarray(
                    [int(np.flatnonzero(free["relative_steps"] == step)[0]) for step in offsets],
                    dtype=np.int64,
                )
                free_rgb.append(_rgb_u8(free["A"][free_indices], free["P"][free_indices]))
                wall_rgb.append(_rgb_u8(wall["A"][horizon_idx], wall["P"][horizon_idx]))
            writer = imageio.get_writer(
                output,
                fps=int(fps),
                codec="libx264",
                quality=8,
                macro_block_size=2,
                pixelformat="yuv420p",
            )
            try:
                for frame_idx, relative_step in enumerate(offsets):
                    frame = _video_frame(
                        free_rgb,
                        wall_rgb,
                        frame_idx=frame_idx,
                        relative_step=int(relative_step),
                        candidate_id=str(candidate.candidate_id),
                        point_id=point_id,
                        release_step=release_step,
                    )
                    for _ in range(int(hold_frames)):
                        writer.append_data(frame)
            finally:
                writer.close()
            validation = _validate_video(output, expected_frames=expected_frames)
            if not validation["passed"]:
                raise RuntimeError(f"Generated video failed validation: {output}")
            output_sha = _sha256_file(output)
            _write_json(
                provenance_path,
                {
                    **video_input,
                    "video": str(output),
                    "video_sha256": output_sha,
                    "bytes": int(output.stat().st_size),
                    "validation": validation,
                },
            )
            status = "generated"
        manifest_rows.append(
            {
                "run_idx": int(candidate.run_idx),
                "candidate_id": str(candidate.candidate_id),
                "candidate_kind": str(candidate.candidate_kind),
                "candidate_idx": int(candidate.candidate_idx),
                "point_id": point_id,
                "selection_rule": "maximum optimized-candidate Delta-H point within run; same point for all four matched candidates",
                "horizon_steps": VIDEO_HORIZON,
                "release_step": release_step,
                "video": str(output),
                "video_sha256": output_sha,
                "video_input_sha256": video_input["input_sha256"],
                "provenance": str(provenance_path),
                "provenance_sha256": _sha256_file(provenance_path),
                "bytes": int(output.stat().st_size),
                "fps": int(fps),
                "hold_frames": int(hold_frames),
                "decoded_frames": int(validation["frames"]),
                "width": int(validation["width"]),
                "height": int(validation["height"]),
                "status": status,
            }
        )
        print(f"[videos] {candidate_number}/{len(candidates)} {candidate.candidate_id}: {status}", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    _write_table(output_root / "video_manifest.csv", manifest)
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_identity_sha256": analysis_identity,
        "n_videos": len(manifest),
        "n_generated": int((manifest["status"] == "generated").sum()),
        "n_reused": int((manifest["status"] == "reused").sum()),
        "horizon_steps": VIDEO_HORIZON,
        "fps": int(fps),
        "hold_frames": int(hold_frames),
        "manifest_sha256": _sha256_file(output_root / "video_manifest.csv"),
    }
    _write_json(output_root / "videos_summary.json", summary)
    return summary


def _write_report(output_root: Path, checks: dict[str, Any]) -> Path:
    table = pd.read_csv(output_root / "c5_statistical_table.csv")
    primary_rows = table[table["metric"] == "excess_clip_post_release"].sort_values("horizon_steps")
    lines = [
        "# Flow-Lenia C5 RNG-only mass-preserving horizon report",
        "",
        "## Protocol",
        "",
        "- Branches differ only through folded continuation RNG; external state noise is zero.",
        "- The wall arm uses mass-preserving 3x3 compartments for the first half of each horizon, then native free dynamics.",
        "- Horizons: 5k, 10k, 15k, 20k, and 30k steps.",
        "- The four retained frames after wall release define the primary window.",
        "- Frustration at a point is median same-seed free/wall CLIP Chamfer minus median free/free CLIP Chamfer.",
        "- Candidate values are medians over 15 matched points; run effects are optimized minus the median of three random candidates.",
        "- The inherited 20k horizon is confirmatory; the other horizons are sensitivity analyses.",
        "",
        "## CLIP run-level result",
        "",
        "| Horizon | Median effect | 95% bootstrap CI | Positive runs | Sign p | Wilcoxon p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows.itertuples():
        lines.append(
            f"| {int(row.horizon_steps):,} | {float(row.median):+.6g} | "
            f"[{float(row.bootstrap_median_ci_low):+.6g}, {float(row.bootstrap_median_ci_high):+.6g}] | "
            f"{int(row.n_positive)}/{int(row.n_runs)} | {float(row.sign_test_p_greater):.4g} | "
            f"{float(row.wilcoxon_p_greater):.4g} |"
        )
    primary = primary_rows[primary_rows["horizon_steps"] == PRIMARY_HORIZON].iloc[0]
    lines.extend(
        [
            "",
            "## Confirmatory interpretation",
            "",
            (
                "The 20k one-sided confirmatory test supports a positive C5 effect under this protocol."
                if float(primary["wilcoxon_p_greater"]) < 0.05
                else "The 20k one-sided confirmatory test does not reject the null under this protocol."
            ),
            "The remaining horizons and field/mass measures are sensitivity diagnostics and are not additional confirmatory tests.",
            "",
            "## Completion",
            "",
            f"- Analysis audit: `{checks.get('status', 'unknown')}`.",
            f"- Free embedding caches: {2 * ROW_COUNT // 2:,}; wall embedding caches: {2 * ROW_COUNT // 2:,}.",
            f"- Point-horizon rows: {POINT_COUNT * len(HORIZONS):,}.",
            f"- Candidate videos: {CANDIDATE_COUNT} at the 30k horizon.",
            f"- Figures: {len(REQUIRED_FIGURE_STEMS)} stems in PNG and PDF.",
            "",
        ]
    )
    path = output_root / "C5_RNG_ONLY_MASS_PRESERVING_HORIZON_REPORT.md"
    path.write_text("\n".join(lines))
    return path


def completion_audit(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    expected_tables = {
        "embedding_manifest.csv": 2 * ROW_COUNT,
        "branch_pair_metrics.csv": POINT_COUNT * len(HORIZONS) * 15,
        "branch_frame_metrics.csv": POINT_COUNT * len(HORIZONS) * 3 * CAPTURE_COUNT,
        "point_metrics.csv": POINT_COUNT * len(HORIZONS),
        "candidate_summary.csv": CANDIDATE_COUNT * len(HORIZONS),
        "candidate_frame_summary.csv": CANDIDATE_COUNT * len(HORIZONS) * CAPTURE_COUNT,
        "run_summary.csv": RUN_COUNT * len(HORIZONS),
        "run_condition_summary.csv": RUN_COUNT * len(HORIZONS) * 3 * 2,
        "run_frame_summary.csv": RUN_COUNT * len(HORIZONS) * CAPTURE_COUNT,
        "c5_statistical_table.csv": len(HORIZONS) * len(SUMMARY_METRICS),
        "video_manifest.csv": CANDIDATE_COUNT,
    }
    table_checks = {}
    for filename, expected in expected_tables.items():
        path = output_root / filename
        found = len(pd.read_csv(path)) if path.exists() else -1
        table_checks[filename] = {"passed": found == expected, "found": found, "expected": expected}
    embedding_protocol = json.loads((output_root / "embedding_protocol.json").read_text())
    zero_audit = json.loads((output_root / "embedding_pair_zero_audit.json").read_text())
    metric_summary = json.loads((output_root / "metrics_summary.json").read_text())
    figure_summary = json.loads((output_root / "figures_summary.json").read_text())
    video_summary = json.loads((output_root / "videos_summary.json").read_text())
    embedding_manifest = pd.read_csv(output_root / "embedding_manifest.csv")
    cache_files_ready = int(sum(Path(path).exists() for path in embedding_manifest["embedding_cache"]))
    figure_files = [output_root / "figures" / f"{stem}.{suffix}" for stem in REQUIRED_FIGURE_STEMS for suffix in ("png", "pdf")]
    valid_figures = int(sum(path.exists() and path.stat().st_size > 1_000 for path in figure_files))
    video_manifest = pd.read_csv(output_root / "video_manifest.csv")
    video_errors = []
    for row in video_manifest.itertuples():
        validation = _validate_video(Path(row.video), expected_frames=CAPTURE_COUNT * int(row.hold_frames))
        if not validation["passed"] or _sha256_file(Path(row.video)) != str(row.video_sha256):
            video_errors.append(str(row.video))
    primary_table = pd.read_csv(output_root / "c5_statistical_table.csv")
    primary_rows = primary_table[
        (primary_table["horizon_steps"] == PRIMARY_HORIZON)
        & (primary_table["metric"] == "excess_clip_post_release")
        & (primary_table["primary"].astype(bool))
    ]
    checks = {
        "simulation": {
            "passed": json.loads((output_root / "completion_audit.json").read_text()).get("status") == "passed",
        },
        "plan": {"passed": len(plan) == ROW_COUNT and protocol.get("plan_sha256") is not None},
        "tables": {"passed": all(item["passed"] for item in table_checks.values()), "details": table_checks},
        "embeddings": {
            "passed": (
                embedding_protocol.get("status") == "complete"
                and embedding_protocol.get("inference_batch_frames") == 1
                and cache_files_ready == 2 * ROW_COUNT
                and zero_audit.get("status") == "passed"
            ),
            "cache_files_ready": cache_files_ready,
            "zero_audit": zero_audit,
        },
        "metrics": {
            "passed": metric_summary.get("status") == "complete" and len(primary_rows) == 1,
            "primary_rows": len(primary_rows),
        },
        "figures": {
            "passed": figure_summary.get("status") == "complete" and valid_figures == 2 * len(REQUIRED_FIGURE_STEMS),
            "valid_files": valid_figures,
            "expected_files": 2 * len(REQUIRED_FIGURE_STEMS),
        },
        "videos": {
            "passed": video_summary.get("status") == "complete" and len(video_errors) == 0,
            "n_valid": CANDIDATE_COUNT - len(video_errors),
            "errors": video_errors,
        },
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    report_stub = {"status": "passed" if not failed else "failed", "checks": checks}
    report_path = _write_report(output_root, report_stub)
    audit = {
        "status": "passed" if not failed else "failed",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_fingerprint": _analysis_fingerprint(),
        "simulation_protocol": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol["simulation_code_bundle_sha256"],
        "checks": checks,
        "report": str(report_path),
        "report_sha256": _sha256_file(report_path),
    }
    audit["audit_identity_sha256"] = _identity_sha256(audit)
    _write_json(output_root / "analysis_completion_audit.json", audit)
    if failed:
        raise RuntimeError(f"Analysis completion audit failed: {failed}")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the RNG-only mass-preserving Flow-Lenia C5 horizon grid."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "embeddings", "metrics", "plots", "videos", "audit", "all"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--video-hold-frames", type=int, default=6)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-videos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = _resolve(args.output_root)
    plan, protocol = _load_inputs(output_root)
    if args.phase in {"preflight", "all"}:
        print(json.dumps(_jsonable(preflight(plan, protocol, output_root)), indent=2))
    if args.phase in {"embeddings", "all"}:
        ensure_embeddings(
            plan,
            protocol,
            output_root,
            force=bool(args.force_embeddings),
        )
    if args.phase in {"metrics", "all"}:
        print(json.dumps(_jsonable(compute_metrics(plan, protocol, output_root)), indent=2))
    if args.phase in {"plots", "all"}:
        print(json.dumps(_jsonable(build_plots(output_root)), indent=2))
    if args.phase in {"videos", "all"}:
        print(
            json.dumps(
                _jsonable(
                    build_videos(
                        plan,
                        protocol,
                        output_root,
                        fps=int(args.video_fps),
                        hold_frames=int(args.video_hold_frames),
                        force=bool(args.force_videos),
                    )
                ),
                indent=2,
            )
        )
    if args.phase in {"audit", "all"}:
        print(json.dumps(_jsonable(completion_audit(plan, protocol, output_root)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
