#!/usr/bin/env python3
"""Paper analysis for paired C2-style Flow-Lenia frustration branches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
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

from flowlenia_c5_branch_frustration import (
    DEFAULT_C2_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_PLAN_ROWS,
    PROTOCOL_VERSION,
    WALL_STEPS,
    _plan_identity_hash,
    _sha256_file,
)
from paper_suite_c2_branching import (
    _clip_embedding_cache_path,
    _embedding_chamfer_cosine,
    _pool_spatial,
    _render_apf_rgb,
)


ANALYSIS_VERSION = "flowlenia-c5-c2-paired-analysis-v3"
FOUNDATION_MODEL = "clip"
CAPTURE_COUNT = 8
POINT_COUNT = 600
CANDIDATE_COUNT = 40
RUN_COUNT = 10
SUMMARY_METRIC_COUNT = 13
BRANCH_PAIR_COUNT = POINT_COUNT * (3 + 6 + 3 + 3)
FRAME_PAIR_COUNT = POINT_COUNT * 3 * CAPTURE_COUNT
FIELD_SCALES = (1, 2, 4)
OPT_COLOR = "#D55E00"
RANDOM_COLOR = "#0072B2"
WALL_COLOR = "#CC79A7"
FREE_COLOR = "#009E73"
NEUTRAL = "#5B6573"
UPSTREAM_CLIP_MAX_ABS_TOL = 1.0e-5
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
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return value.resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_clean(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _analysis_code_fingerprint() -> dict[str, Any]:
    paths = {
        "analysis": Path(__file__).resolve(),
        "rendering_and_clip_helpers": (
            _REPO_ROOT / "scripts/paper_suite_c2_branching.py"
        ).resolve(),
        "foundation_wrapper": (
            _REPO_ROOT / "foundation_models/clip.py"
        ).resolve(),
    }
    files = {
        name: {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for name, path in paths.items()
    }
    identity = {
        "analysis_version": ANALYSIS_VERSION,
        "files": files,
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _foundation_model_fingerprint(fm: Any) -> dict[str, Any]:
    import jax

    digest = hashlib.sha256()
    n_bytes = 0
    leaves_with_paths, _tree = jax.tree_util.tree_flatten_with_path(
        fm.clip_model.params
    )
    for path, leaf in leaves_with_paths:
        arr = np.ascontiguousarray(
            np.asarray(jax.device_get(leaf))
        )
        path_text = "/".join(str(component) for component in path)
        digest.update(path_text.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(arr).cast("B"))
        n_bytes += int(arr.nbytes)
    model_config = _json_clean(fm.clip_model.config.to_dict())
    image_processor_config = _json_clean(
        fm.processor.image_processor.to_dict()
    )
    identity = {
        "foundation_model": FOUNDATION_MODEL,
        "model_id": "openai/clip-vit-base-patch32",
        "model_revision": getattr(
            fm.clip_model.config,
            "_commit_hash",
            None,
        ),
        "weights_sha256": digest.hexdigest(),
        "parameter_leaves": len(leaves_with_paths),
        "parameter_bytes": n_bytes,
        "foundation_wrapper_sha256": _sha256_file(
            _REPO_ROOT / "foundation_models/clip.py"
        ),
        "transformers_version": importlib.metadata.version(
            "transformers"
        ),
        "jax_version": jax.__version__,
        "image_mean": np.asarray(fm.img_mean).tolist(),
        "image_std": np.asarray(fm.img_std).tolist(),
        "model_config": model_config,
        "model_config_sha256": _identity_sha256(model_config),
        "image_processor_config": image_processor_config,
        "image_processor_config_sha256": _identity_sha256(
            image_processor_config
        ),
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _load_inputs(output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan_path = output_root / "paired_plan.csv"
    protocol_path = output_root / "protocol.json"
    if not plan_path.exists() or not protocol_path.exists():
        raise FileNotFoundError(
            f"Missing paired plan/protocol under {output_root}; run the plan phase."
        )
    with plan_path.open(newline="") as stream:
        raw_records = list(csv.DictReader(stream))
    plan = pd.DataFrame(raw_records)
    protocol = json.loads(protocol_path.read_text())
    found_hash = _plan_identity_hash(raw_records)
    if found_hash != protocol.get("plan_sha256"):
        raise RuntimeError(
            f"Plan hash mismatch: found={found_hash}, protocol={protocol.get('plan_sha256')}"
        )
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Simulation protocol version mismatch: "
            f"found={protocol.get('protocol_version')} expected={PROTOCOL_VERSION}"
        )
    if len(str(protocol.get("simulation_code_bundle_sha256", ""))) != 64:
        raise RuntimeError("Protocol lacks a simulation-code bundle fingerprint")
    if len(plan) != EXPECTED_PLAN_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_PLAN_ROWS} plan rows, found {len(plan)}")
    for column in (
        "row_id",
        "run_idx",
        "trial_idx",
        "candidate_idx",
        "point_id",
        "pair_id",
        "window_index",
        "step",
        "branch_id",
        "branch_seed",
        "horizon_steps",
        "wall_steps",
    ):
        plan[column] = plan[column].astype(int)
    for column in (
        "delta_h",
        "perturb_a_std",
        "perturb_p_std",
        "perturb_lagrangian_xy_std",
    ):
        plan[column] = plan[column].astype(float)
    if sorted(plan["row_id"].tolist()) != list(range(EXPECTED_PLAN_ROWS)):
        raise RuntimeError("Plan row_id values are not contiguous 0..1799")
    counts = plan.groupby(["candidate_id", "point_id"]).size()
    if len(counts) != POINT_COUNT or not bool((counts == 3).all()):
        raise RuntimeError(
            "Plan must contain 600 candidate/point groups with three branches each"
        )
    pairing_columns = [
        "condition",
        "pair_id",
        "point_id",
        "window_index",
        "step",
        "branch_id",
        "branch_seed",
        "horizon_steps",
        "wall_steps",
        "perturb_a_std",
        "perturb_p_std",
        "perturb_lagrangian_xy_std",
        "selection_reference_traj_id",
    ]
    sort_columns = ["point_id", "branch_id"]
    for run_idx, run_rows in plan.groupby("run_idx", sort=True):
        candidates = (
            run_rows[
                [
                    "candidate_id",
                    "candidate_kind",
                    "candidate_idx",
                ]
            ]
            .drop_duplicates()
            .sort_values(["candidate_kind", "candidate_idx"])
        )
        if (
            len(candidates) != 4
            or int((candidates["candidate_kind"] == "optimized").sum()) != 1
            or int((candidates["candidate_kind"] == "random").sum()) != 3
        ):
            raise RuntimeError(
                f"Run {run_idx} does not contain one optimized and three random candidates"
            )
        reference = None
        reference_id = None
        for candidate_id, candidate_rows in run_rows.groupby(
            "candidate_id",
            sort=True,
        ):
            if len(candidate_rows) != 45:
                raise RuntimeError(
                    f"{candidate_id} has {len(candidate_rows)} plan rows, expected 45"
                )
            pairing = (
                candidate_rows.sort_values(sort_columns)[pairing_columns]
                .reset_index(drop=True)
            )
            if reference is None:
                reference = pairing
                reference_id = candidate_id
            elif not pairing.equals(reference):
                raise RuntimeError(
                    "Matched candidate plan mismatch in run "
                    f"{run_idx}: {candidate_id} differs from {reference_id}"
                )
    stable_point_columns = [
        "run_idx",
        "trial_idx",
        "candidate_kind",
        "candidate_idx",
        "condition",
        "pair_id",
        "window_index",
        "step",
        "delta_h",
        "horizon_steps",
        "wall_steps",
        "perturb_a_std",
        "perturb_p_std",
        "perturb_lagrangian_xy_std",
    ]
    for (candidate_id, point_id), point_rows in plan.groupby(
        ["candidate_id", "point_id"],
        sort=True,
    ):
        if point_rows["branch_id"].sort_values().tolist() != [0, 1, 2]:
            raise RuntimeError(
                f"{candidate_id}/point_{point_id} lacks branches 0,1,2"
            )
        if any(
            point_rows[column].nunique(dropna=False) != 1
            for column in stable_point_columns
        ):
            raise RuntimeError(
                f"Inconsistent point metadata for {candidate_id}/point_{point_id}"
            )
    return plan, protocol


def _branch_npz(branch_dir: Path) -> Path:
    files = sorted((branch_dir / "apf_logs").glob("*.npz"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one APF chunk under {branch_dir}, found {len(files)}")
    return files[0]


def _load_ap_fields(branch_dir: Path) -> dict[str, np.ndarray]:
    path = _branch_npz(branch_dir)
    with np.load(path, allow_pickle=False) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        arrays = {
            "steps": steps,
            "A": np.asarray(data["A"]),
            "P": np.asarray(data["P"]),
        }
    order = np.argsort(steps, kind="stable")
    return {key: value[order] for key, value in arrays.items()}


def _embedding_cache_path(
    cache_dir: Path,
    *,
    row_id: int,
    arm: str,
    source_sha256: str,
    inference_batch_frames: int,
    model_identity_sha256: str,
    analysis_code_identity_sha256: str,
) -> Path:
    digest = hashlib.sha256(
        (
            f"{ANALYSIS_VERSION}|{FOUNDATION_MODEL}|{row_id}|{arm}|"
            f"{source_sha256}|batch_frames={int(inference_batch_frames)}|"
            f"model={model_identity_sha256}|code={analysis_code_identity_sha256}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    return cache_dir / f"row_{row_id:04d}_{arm}_{digest}.npz"


def _read_embedding_cache(
    path: Path,
    *,
    source_sha256: str,
    relative_steps: np.ndarray,
    inference_batch_frames: int,
    model_identity_sha256: str,
    analysis_code_identity_sha256: str,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["analysis_version"]).item()) != ANALYSIS_VERSION:
                return None
            if str(np.asarray(data["source_apf_sha256"]).item()) != source_sha256:
                return None
            if int(np.asarray(data["inference_batch_frames"]).item()) != int(
                inference_batch_frames
            ):
                return None
            if (
                str(np.asarray(data["model_identity_sha256"]).item())
                != model_identity_sha256
            ):
                return None
            if (
                str(
                    np.asarray(
                        data["analysis_code_identity_sha256"]
                    ).item()
                )
                != analysis_code_identity_sha256
            ):
                return None
            if not np.array_equal(
                np.asarray(data["relative_steps"], dtype=np.int64),
                np.asarray(relative_steps, dtype=np.int64),
            ):
                return None
            z = np.asarray(data["z"], dtype=np.float32)
    except Exception:
        return None
    if z.ndim != 2 or z.shape[0] != CAPTURE_COUNT or not np.all(np.isfinite(z)):
        return None
    return z


def _save_embedding_cache(
    path: Path,
    *,
    z: np.ndarray,
    row_id: int,
    arm: str,
    branch_dir: Path,
    source_sha256: str,
    relative_steps: np.ndarray,
    provenance: str,
    inference_batch_frames: int,
    model_identity_sha256: str,
    analysis_code_identity_sha256: str,
    upstream_cache: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z, dtype=np.float32)
    z = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12, None)
    np.savez_compressed(
        path,
        z=z,
        row_id=np.asarray(row_id, dtype=np.int32),
        arm=np.asarray(arm),
        branch_dir=np.asarray(str(branch_dir)),
        relative_steps=np.asarray(relative_steps, dtype=np.int64),
        source_apf_sha256=np.asarray(source_sha256),
        foundation_model=np.asarray(FOUNDATION_MODEL),
        analysis_version=np.asarray(ANALYSIS_VERSION),
        provenance=np.asarray(provenance),
        inference_batch_frames=np.asarray(
            int(inference_batch_frames),
            dtype=np.int32,
        ),
        model_identity_sha256=np.asarray(model_identity_sha256),
        analysis_code_identity_sha256=np.asarray(
            analysis_code_identity_sha256
        ),
        upstream_cache=np.asarray("" if upstream_cache is None else str(upstream_cache)),
    )


def _load_rgb(branch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_ap_fields(branch_dir)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    relative = steps - int(steps[0])
    if steps.size != CAPTURE_COUNT:
        raise RuntimeError(f"{branch_dir} has {steps.size} frames, expected 8")
    if not np.array_equal(relative, np.asarray([0, 2850, 5700, 8550, 11400, 14250, 17100, 20000])):
        raise RuntimeError(f"Unexpected capture offsets in {branch_dir}: {relative.tolist()}")
    return _render_apf_rgb({"A": arrays["A"], "P": arrays["P"]}), relative


def ensure_embeddings(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    output_root: Path,
    c2_root: Path,
    batch_frames: int,
    force: bool,
) -> pd.DataFrame:
    import jax
    import jax.numpy as jnp
    from jax.experimental.compilation_cache import compilation_cache

    cache_dir = output_root / "clip_embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    compile_dir = output_root / "jax_metric_compilation_cache"
    compile_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    compilation_cache.set_cache_dir(str(compile_dir))
    batch_frames = int(batch_frames)
    if batch_frames != 1:
        raise ValueError(
            "C2 parity requires --batch-frames=1: CLIP must use the "
            "authoritative unjitted single-frame inference path."
        )
    import foundation_models

    fm = foundation_models.create_foundation_model(FOUNDATION_MODEL)
    model_fingerprint = _foundation_model_fingerprint(fm)
    model_identity_sha256 = model_fingerprint["identity_sha256"]
    analysis_code_fingerprint = _analysis_code_fingerprint()
    analysis_code_identity_sha256 = analysis_code_fingerprint[
        "identity_sha256"
    ]

    manifest_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    tasks = [
        (row, arm, Path(row[f"{'free' if arm == 'free' else 'walls'}_branch_dir"]))
        for _, row in plan.sort_values("row_id").iterrows()
        for arm in ("free", "walls")
    ]
    for task_idx, (row, arm, branch_dir) in enumerate(tasks, start=1):
        branch_dir = _resolve(branch_dir)
        apf_path = _branch_npz(branch_dir)
        source_sha = _sha256_file(apf_path)
        with np.load(apf_path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
        relative = steps - int(steps[0])
        target = _embedding_cache_path(
            cache_dir,
            row_id=int(row["row_id"]),
            arm=arm,
            source_sha256=source_sha,
            inference_batch_frames=batch_frames,
            model_identity_sha256=model_identity_sha256,
            analysis_code_identity_sha256=analysis_code_identity_sha256,
        )
        z = None if force else _read_embedding_cache(
            target,
            source_sha256=source_sha,
            relative_steps=relative,
            inference_batch_frames=batch_frames,
            model_identity_sha256=model_identity_sha256,
            analysis_code_identity_sha256=analysis_code_identity_sha256,
        )
        provenance = "analysis_cache"
        upstream = None
        if z is None:
            pending.append(
                {
                    "row": row,
                    "arm": arm,
                    "branch_dir": branch_dir,
                    "apf_path": apf_path,
                    "source_sha": source_sha,
                    "relative": relative,
                    "target": target,
                }
            )
            provenance = "pending"
        manifest_rows.append(
            {
                "row_id": int(row["row_id"]),
                "run_idx": int(row["run_idx"]),
                "candidate_id": str(row["candidate_id"]),
                "candidate_kind": str(row["candidate_kind"]),
                "candidate_idx": int(row["candidate_idx"]),
                "point_id": int(row["point_id"]),
                "branch_id": int(row["branch_id"]),
                "arm": arm,
                "branch_dir": str(branch_dir),
                "apf_path": str(apf_path),
                "source_apf_sha256": source_sha,
                "model_identity_sha256": model_identity_sha256,
                "analysis_code_identity_sha256": (
                    analysis_code_identity_sha256
                ),
                "embedding_cache": str(target),
                "provenance": provenance,
                "upstream_cache": "" if upstream is None else str(upstream),
            }
        )
        if task_idx % 500 == 0:
            print(
                f"[embeddings] audited {task_idx}/{len(tasks)}; pending={len(pending)}",
                flush=True,
            )

    if pending:
        started = time.monotonic()
        completed = 0
        exact_frame_cache: dict[str, np.ndarray] = {}
        reused_exact_frames = 0
        for item in pending:
            rgb, relative = _load_rgb(item["branch_dir"])
            if not np.array_equal(relative, item["relative"]):
                raise RuntimeError("APF offsets changed during embedding")
            zs = []
            for frame in rgb:
                frame = np.ascontiguousarray(frame, dtype=np.float32)
                frame_sha = hashlib.sha256(
                    memoryview(frame).cast("B")
                ).hexdigest()
                z = exact_frame_cache.get(frame_sha)
                if z is None:
                    z = np.asarray(
                        jax.device_get(fm.embed_img(jnp.asarray(frame))),
                        dtype=np.float32,
                    ).reshape(-1)
                    exact_frame_cache[frame_sha] = z
                else:
                    reused_exact_frames += 1
                zs.append(z)
            _save_embedding_cache(
                item["target"],
                z=np.stack(zs, axis=0),
                row_id=int(item["row"]["row_id"]),
                arm=item["arm"],
                branch_dir=item["branch_dir"],
                source_sha256=item["source_sha"],
                relative_steps=item["relative"],
                provenance="computed_c2_single_frame_clip",
                inference_batch_frames=batch_frames,
                model_identity_sha256=model_identity_sha256,
                analysis_code_identity_sha256=(
                    analysis_code_identity_sha256
                ),
            )
            manifest_index = 2 * int(item["row"]["row_id"]) + (
                0 if item["arm"] == "free" else 1
            )
            manifest_rows[manifest_index][
                "provenance"
            ] = "computed_c2_single_frame_clip"
            completed += 1
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 1e-9)
            remaining = (len(pending) - completed) / max(rate, 1e-9)
            if completed == len(pending) or completed % 20 == 0:
                print(
                    f"[embeddings] {completed}/{len(pending)} new branches, "
                    f"rate={rate:.2f}/s eta={remaining / 60:.1f} min "
                    f"exact-frame-reuse={reused_exact_frames}",
                    flush=True,
                )

    manifest = pd.DataFrame(manifest_rows).sort_values(["row_id", "arm"])
    missing = [path for path in manifest["embedding_cache"] if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} embedding caches are missing")
    manifest["embedding_cache_sha256"] = [
        _sha256_file(Path(path))
        for path in manifest["embedding_cache"]
    ]
    upstream_errors = []
    for idx, row in manifest[
        (manifest["candidate_kind"] == "optimized")
        & (manifest["arm"] == "free")
    ].iterrows():
        branch_dir = Path(row["branch_dir"])
        upstream = _clip_embedding_cache_path(
            branch_dir,
            cache_dir=c2_root / "clip_embedding_cache",
            foundation_model=FOUNDATION_MODEL,
            max_chunks=4,
            max_snapshots_per_chunk=8,
            max_frames=32,
        )
        if not upstream.exists():
            continue
        with np.load(row["embedding_cache"], allow_pickle=False) as current_data:
            current = np.asarray(current_data["z"], dtype=np.float32)
        with np.load(upstream, allow_pickle=False) as upstream_data:
            reference = np.asarray(upstream_data["z"], dtype=np.float32)
        max_abs = float(np.max(np.abs(current - reference)))
        upstream_errors.append(max_abs)
        manifest.at[idx, "upstream_cache"] = str(upstream)
        manifest.at[idx, "max_abs_vs_upstream_single_frame_cache"] = max_abs

    zero_errors = []
    zero_cosine = []
    for row_id, rows in manifest.groupby("row_id", sort=True):
        free_path = rows[rows["arm"] == "free"]["embedding_cache"].iloc[0]
        wall_path = rows[rows["arm"] == "walls"]["embedding_cache"].iloc[0]
        free = _load_z(free_path)[0]
        walls = _load_z(wall_path)[0]
        zero_errors.append(float(np.max(np.abs(free - walls))))
        zero_cosine.append(float(np.clip(1.0 - np.sum(free * walls), 0.0, 2.0)))
    zero_audit = {
        "status": (
            "passed"
            if len(zero_errors) == EXPECTED_PLAN_ROWS
            and max(zero_errors, default=float("inf")) <= 1e-6
            else "failed"
        ),
        "n_pairs": len(zero_errors),
        "n_embedding_exact": int(np.sum(np.asarray(zero_errors) == 0.0)),
        "max_embedding_abs": max(zero_errors, default=float("nan")),
        "max_cosine_distance": max(zero_cosine, default=float("nan")),
        "inference_batch_frames": batch_frames,
    }
    _write_json(
        output_root / "embedding_pair_zero_audit.json",
        _json_clean(zero_audit),
    )
    if zero_audit["status"] != "passed":
        raise RuntimeError(f"Identical arm starts failed CLIP parity: {zero_audit}")

    upstream_audit = {
        "status": (
            "passed"
            if len(upstream_errors) == 450
            and max(upstream_errors, default=float("inf"))
            <= UPSTREAM_CLIP_MAX_ABS_TOL
            else "failed"
        ),
        "n_compared": len(upstream_errors),
        "expected": 450,
        "max_abs": max(upstream_errors, default=float("nan")),
        "median_max_abs": _median(upstream_errors),
        "max_abs_tolerance": UPSTREAM_CLIP_MAX_ABS_TOL,
        "used_for_final_metrics": False,
    }
    if upstream_audit["status"] != "passed":
        raise RuntimeError(
            "Uniform CLIP inference does not match authoritative C2 "
            f"embedding references: {upstream_audit}"
        )

    embedding_protocol = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_script_sha256": analysis_code_fingerprint["files"][
            "analysis"
        ]["sha256"],
        "analysis_code_fingerprint": analysis_code_fingerprint,
        "analysis_code_identity_sha256": (
            analysis_code_identity_sha256
        ),
        "simulation_protocol_version": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "foundation_model": FOUNDATION_MODEL,
        "model_id": "openai/clip-vit-base-patch32",
        "model_fingerprint": model_fingerprint,
        "inference_batch_frames": batch_frames,
        "inference_mode": "authoritative_c2_unjitted_single_frame",
        "frames_per_branch": CAPTURE_COUNT,
        "model_calls_per_uncached_branch": CAPTURE_COUNT,
        "exact_frame_cache": (
            "SHA-256 keyed within-process reuse only for bitwise-identical "
            "float32 rendered frames"
        ),
        "padding": "none",
        "ordering": "row_id ascending, then free arm, then walls arm; eight chronological frames per arm",
        "n_uniform_embedding_caches": len(manifest),
        "upstream_c2_reference": upstream_audit,
        "identical_start_audit": zero_audit,
        "runtime": {
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    _write_table(output_root / "embedding_manifest.csv", manifest)
    embedding_protocol["embedding_manifest_sha256"] = _sha256_file(
        output_root / "embedding_manifest.csv"
    )
    _write_json(
        output_root / "embedding_protocol.json",
        _json_clean(embedding_protocol),
    )
    return manifest


def _load_z(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["z"], dtype=np.float32)


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _field_components(
    left: dict[str, Any],
    right: dict[str, Any],
    indices: np.ndarray,
) -> tuple[float, float, float]:
    def distance(key: str) -> float:
        values = []
        for scale in FIELD_SCALES:
            left_value = left["_pyramid"][key][scale][indices]
            right_value = right["_pyramid"][key][scale][indices]
            diff = np.asarray(left_value, dtype=np.float32) - np.asarray(
                right_value,
                dtype=np.float32,
            )
            values.append(float(np.sqrt(np.mean(diff * diff))))
        return float(np.mean(values))

    a = distance("A")
    p = distance("P")
    return float(np.mean([a, p])), float(a), float(p)


def _with_field_pyramid(fields: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "A": fields["A"],
        "P": fields["P"],
        "_pyramid": {},
    }
    for key in ("A", "P"):
        source = np.asarray(fields[key], dtype=np.float32)
        result["_pyramid"][key] = {
            scale: _pool_spatial(source, scale)
            for scale in FIELD_SCALES
        }
    return result


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
    for window_name, indices in windows.items():
        row[f"clip_{window_name}"] = _embedding_chamfer_cosine(
            left_z[indices],
            right_z[indices],
        )
        row[f"clip_sync_{window_name}"] = float(
            np.mean(
                np.clip(
                    1.0
                    - np.sum(
                        left_z[indices] * right_z[indices],
                        axis=-1,
                    ),
                    0.0,
                    2.0,
                )
            )
        )
        left_mass = np.sum(
            np.asarray(left_fields["A"][indices], dtype=np.float64),
            axis=(1, 2, 3),
        )
        right_mass = np.sum(
            np.asarray(right_fields["A"][indices], dtype=np.float64),
            axis=(1, 2, 3),
        )
        mass_scale = np.clip(np.abs(left_mass), 1e-12, None)
        row[f"mass_rel_{window_name}"] = float(
            np.mean(np.abs(right_mass - left_mass) / mass_scale)
        )
        row[f"mass_delta_rel_{window_name}"] = float(
            np.mean((right_mass - left_mass) / mass_scale)
        )
        field, a_value, p_value = _field_components(
            left_fields,
            right_fields,
            indices,
        )
        row[f"field_{window_name}"] = field
        row[f"A_{window_name}"] = a_value
        row[f"P_{window_name}"] = p_value
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
    stats = np.median(samples, axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


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
    manifest: pd.DataFrame,
    *,
    output_root: Path,
) -> dict[str, Any]:
    analysis_code_fingerprint = _analysis_code_fingerprint()
    analysis_code_identity_sha256 = analysis_code_fingerprint[
        "identity_sha256"
    ]
    embedding_manifest_path = output_root / "embedding_manifest.csv"
    embedding_protocol_path = output_root / "embedding_protocol.json"
    if not embedding_manifest_path.exists() or not embedding_protocol_path.exists():
        raise RuntimeError("Embedding provenance files are missing")
    metric_inputs = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": analysis_code_identity_sha256,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_protocol_version": protocol["protocol_version"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "embedding_manifest_sha256": _sha256_file(
            embedding_manifest_path
        ),
        "embedding_protocol_sha256": _sha256_file(
            embedding_protocol_path
        ),
    }
    model_identities = set(
        manifest["model_identity_sha256"].astype(str)
    )
    if len(model_identities) != 1:
        raise RuntimeError(
            "Embedding manifest contains multiple model identities: "
            f"{sorted(model_identities)}"
        )
    model_identity_sha256 = next(iter(model_identities))
    metric_inputs["model_identity_sha256"] = model_identity_sha256
    metric_input_sha256 = _identity_sha256(metric_inputs)
    cache_lookup = {
        (int(row.row_id), str(row.arm)): str(row.embedding_cache)
        for row in manifest.itertuples()
    }
    pair_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    groups = list(plan.groupby(["candidate_id", "point_id"], sort=True))
    started = time.monotonic()
    for group_idx, ((candidate_id, point_id), rows) in enumerate(groups, start=1):
        rows = rows.sort_values("branch_id")
        if rows["branch_id"].tolist() != [0, 1, 2]:
            raise RuntimeError(f"Bad branch IDs for {candidate_id} point {point_id}")
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
            "horizon_steps",
            "wall_steps",
            "perturb_a_std",
            "perturb_p_std",
            "perturb_lagrangian_xy_std",
        )
        inconsistent = [
            column
            for column in stable_columns
            if rows[column].nunique(dropna=False) != 1
        ]
        if inconsistent:
            raise RuntimeError(
                f"Inconsistent metadata for {candidate_id}/point_{point_id}: "
                f"{inconsistent}"
            )
        first = rows.iloc[0]
        base = {
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
        free_fields: list[dict[str, Any]] = []
        wall_fields: list[dict[str, Any]] = []
        free_z: list[np.ndarray] = []
        wall_z: list[np.ndarray] = []
        relative = None
        for row in rows.itertuples():
            free = _load_ap_fields(_resolve(row.free_branch_dir))
            walls = _load_ap_fields(_resolve(row.walls_branch_dir))
            rel_free = np.asarray(free["steps"], dtype=np.int64) - int(row.step)
            rel_walls = np.asarray(walls["steps"], dtype=np.int64) - int(row.step)
            if not np.array_equal(rel_free, rel_walls):
                raise RuntimeError(f"Arm frame mismatch at row {row.row_id}")
            if relative is None:
                relative = rel_free
            elif not np.array_equal(relative, rel_free):
                raise RuntimeError(f"Branch frame mismatch for {candidate_id}/{point_id}")
            free_fields.append(
                _with_field_pyramid({"A": free["A"], "P": free["P"]})
            )
            wall_fields.append(
                _with_field_pyramid({"A": walls["A"], "P": walls["P"]})
            )
            free_z.append(_load_z(cache_lookup[(int(row.row_id), "free")]))
            wall_z.append(_load_z(cache_lookup[(int(row.row_id), "walls")]))
        assert relative is not None
        windows = {
            "wall_phase": np.flatnonzero((relative > 0) & (relative <= WALL_STEPS)),
            "post_release": np.flatnonzero(relative > WALL_STEPS),
            "full_future": np.flatnonzero(relative > 0),
        }
        if any(indices.size < 1 for indices in windows.values()):
            raise RuntimeError(f"Empty metric window for {candidate_id}/{point_id}")

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
                1.0
                - np.sum(
                    free_z[branch_id] * wall_z[branch_id],
                    axis=-1,
                ),
                0.0,
                2.0,
            )
            for frame_idx, (rel_step, value) in enumerate(
                zip(relative, frame_distance, strict=True)
            ):
                frame_rows.append(
                    {
                        **base,
                        "branch_id": branch_id,
                        "frame_idx": frame_idx,
                        "relative_step": int(rel_step),
                        "wall_active": bool(rel_step <= WALL_STEPS),
                        "paired_cosine_distance": float(value),
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
        for metric in (
            "clip",
            "clip_sync",
            "field",
            "A",
            "P",
            "mass_rel",
            "mass_delta_rel",
        ):
            for window in windows:
                column = f"{metric}_{window}"
                by_type = {
                    pair_type: _median(
                        row[column]
                        for row in current_pairs
                        if row["pair_type"] == pair_type
                    )
                    for pair_type in (
                        "paired_same_seed",
                        "paired_off_seed",
                        "free_within",
                        "walls_within",
                    )
                }
                for pair_type, value in by_type.items():
                    point[f"{pair_type}_{column}"] = value
                point[f"excess_{column}"] = (
                    by_type["paired_same_seed"] - by_type["free_within"]
                )
                point[f"spread_delta_{column}"] = (
                    by_type["walls_within"] - by_type["free_within"]
                )
                point[f"pair_alignment_{column}"] = (
                    by_type["paired_same_seed"] - by_type["paired_off_seed"]
                )
        point_rows.append(point)
        if group_idx % 20 == 0 or group_idx == len(groups):
            elapsed = time.monotonic() - started
            rate = group_idx / max(elapsed, 1e-9)
            eta = (len(groups) - group_idx) / max(rate, 1e-9)
            print(
                f"[metrics] points {group_idx}/{len(groups)}, "
                f"rate={rate:.2f}/s eta={eta / 60:.1f} min",
                flush=True,
            )

    pair_frame = pd.DataFrame(pair_rows)
    frame_frame = pd.DataFrame(frame_rows)
    point_frame = pd.DataFrame(point_rows)
    if len(pair_frame) != BRANCH_PAIR_COUNT:
        raise RuntimeError(f"Expected {BRANCH_PAIR_COUNT} pair rows, found {len(pair_frame)}")
    if len(frame_frame) != FRAME_PAIR_COUNT:
        raise RuntimeError(f"Expected {FRAME_PAIR_COUNT} frame rows, found {len(frame_frame)}")
    if len(point_frame) != POINT_COUNT:
        raise RuntimeError(f"Expected {POINT_COUNT} point rows, found {len(point_frame)}")

    id_columns = [
        "run_idx",
        "trial_idx",
        "candidate_id",
        "candidate_kind",
        "candidate_idx",
    ]
    metric_columns = [
        column
        for column in point_frame.columns
        if column.startswith(("paired_", "free_", "walls_", "excess_", "spread_", "pair_"))
        and column != "pair_id"
    ]
    candidate_rows = []
    for candidate_id, rows in point_frame.groupby("candidate_id", sort=True):
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
    if len(candidate_frame) != CANDIDATE_COUNT:
        raise RuntimeError(
            f"Expected {CANDIDATE_COUNT} candidate rows, found {len(candidate_frame)}"
        )

    candidate_time = (
        frame_frame.groupby(
            [
                "run_idx",
                "trial_idx",
                "candidate_id",
                "candidate_kind",
                "candidate_idx",
                "relative_step",
            ],
            as_index=False,
        )["paired_cosine_distance"]
        .median()
        .rename(columns={"paired_cosine_distance": "candidate_median_paired_cosine"})
    )

    summary_metrics = [
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
    ]
    run_rows: list[dict[str, Any]] = []
    for run_idx, rows in candidate_frame.groupby("run_idx", sort=True):
        optimized = rows[rows["candidate_kind"] == "optimized"]
        random = rows[rows["candidate_kind"] == "random"].sort_values("candidate_idx")
        if len(optimized) != 1 or len(random) != 3:
            raise RuntimeError(
                f"Run {run_idx} must have one optimized and three random candidates"
            )
        out: dict[str, Any] = {"run_idx": int(run_idx)}
        for metric in summary_metrics:
            opt_value = float(optimized.iloc[0][metric])
            random_values = random[metric].to_numpy(dtype=np.float64)
            out[f"opt_{metric}"] = opt_value
            for idx, value in enumerate(random_values):
                out[f"random_{idx}_{metric}"] = float(value)
            out[f"random_median_{metric}"] = float(np.median(random_values))
            out[f"contrast_{metric}"] = opt_value - float(np.median(random_values))
        run_rows.append(out)
    run_frame = pd.DataFrame(run_rows)
    if len(run_frame) != RUN_COUNT:
        raise RuntimeError(f"Expected {RUN_COUNT} run rows, found {len(run_frame)}")

    condition_rows: list[dict[str, Any]] = []
    for run_idx, rows in candidate_frame.groupby("run_idx", sort=True):
        optimized = rows[rows["candidate_kind"] == "optimized"].iloc[0]
        random = rows[rows["candidate_kind"] == "random"]
        for condition in ("high", "mid", "low"):
            for metric in (
                "excess_clip_post_release",
                "excess_field_post_release",
            ):
                column = f"{condition}_{metric}"
                opt_value = float(optimized[column])
                random_values = random[column].to_numpy(dtype=np.float64)
                condition_rows.append(
                    {
                        "run_idx": int(run_idx),
                        "condition": condition,
                        "metric": metric,
                        "optimized": opt_value,
                        "random_median": float(np.median(random_values)),
                        "contrast": opt_value - float(np.median(random_values)),
                    }
                )
    condition_frame = pd.DataFrame(condition_rows)

    run_time_rows: list[dict[str, Any]] = []
    for (run_idx, rel_step), rows in candidate_time.groupby(
        ["run_idx", "relative_step"],
        sort=True,
    ):
        optimized = rows[rows["candidate_kind"] == "optimized"]
        random = rows[rows["candidate_kind"] == "random"]
        run_time_rows.append(
            {
                "run_idx": int(run_idx),
                "relative_step": int(rel_step),
                "optimized": float(optimized.iloc[0]["candidate_median_paired_cosine"]),
                "random_median": _median(random["candidate_median_paired_cosine"]),
                "contrast": float(optimized.iloc[0]["candidate_median_paired_cosine"])
                - _median(random["candidate_median_paired_cosine"]),
            }
        )
    run_time_frame = pd.DataFrame(run_time_rows)

    statistics: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": (
            analysis_code_identity_sha256
        ),
        "simulation_protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "metric_input_sha256": metric_input_sha256,
        "model_identity_sha256": model_identity_sha256,
        "statistical_unit": "optimization run (n=10)",
        "candidate_aggregation": "median over 15 selected divergence points",
        "random_control_aggregation": "median of three matched random candidates per run",
        "run_effect": "optimized candidate minus matched-random median",
        "multiplicity_policy": (
            "The single pre-specified primary metric is confirmatory. "
            "The remaining twelve metrics are secondary diagnostics and "
            "their one-sided p-values are exploratory and unadjusted."
        ),
        "tests": {},
    }
    for idx, metric in enumerate(summary_metrics):
        contrast = run_frame[f"contrast_{metric}"].to_numpy(dtype=np.float64)
        statistics["tests"][metric] = _run_stat(contrast, seed=20_260_719 + idx)

    primary = "excess_clip_post_release"
    observed = _median(run_frame[f"contrast_{primary}"])
    statistics["candidate_label_randomization"] = {
        "reported": False,
        "reason": (
            "The optimized and random candidates were not randomly assigned or "
            "exchangeable: optimization selected one candidate and its trajectory "
            "defined the matched divergence points. Candidate-label permutations "
            "are therefore not a valid design-based null."
        ),
    }
    statistics["primary_metric"] = primary
    statistics["primary_observed_median_run_effect"] = observed
    statistical_table = pd.DataFrame(
        [
            {
                "metric": metric,
                "primary": metric == primary,
                "inference_role": (
                    "confirmatory_primary"
                    if metric == primary
                    else "exploratory_secondary_unadjusted"
                ),
                **statistics["tests"][metric],
            }
            for metric in summary_metrics
        ]
    )
    if len(statistical_table) != SUMMARY_METRIC_COUNT:
        raise RuntimeError(
            f"Expected {SUMMARY_METRIC_COUNT} summary metrics, "
            f"found {len(statistical_table)}"
        )

    _write_table(output_root / "branch_pair_metrics.csv", pair_frame)
    _write_table(output_root / "branch_frame_metrics.csv", frame_frame)
    _write_table(output_root / "point_metrics.csv", point_frame)
    _write_table(output_root / "candidate_summary.csv", candidate_frame)
    _write_table(output_root / "candidate_frame_summary.csv", candidate_time)
    _write_table(output_root / "run_summary.csv", run_frame)
    _write_table(output_root / "run_condition_summary.csv", condition_frame)
    _write_table(output_root / "run_frame_summary.csv", run_time_frame)
    _write_table(
        output_root / "c5_statistical_table.csv",
        statistical_table,
    )
    (output_root / "c5_statistical_table.tex").write_text(
        statistical_table.to_latex(
            index=False,
            escape=True,
            float_format=lambda value: f"{value:.4g}",
        )
    )
    _write_json(output_root / "statistical_summary.json", _json_clean(statistics))
    metric_protocol = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_fingerprint": analysis_code_fingerprint,
        "analysis_code_identity_sha256": (
            analysis_code_identity_sha256
        ),
        "simulation_protocol_version": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "metric_inputs": metric_inputs,
        "metric_input_sha256": metric_input_sha256,
        "foundation_model": FOUNDATION_MODEL,
        "model_identity_sha256": model_identity_sha256,
        "rendering": "clip(sum(A_channels) * P_rgb, 0, 1)",
        "capture_relative_steps": [0, 2850, 5700, 8550, 11400, 14250, 17100, 20000],
        "windows": {
            "wall_phase": "0 < relative_step <= 10000 (3 retained frames)",
            "post_release": "relative_step > 10000 (4 retained frames)",
            "full_future": "relative_step > 0 (7 retained frames)",
        },
        "clip_distance": "symmetric Chamfer distance over L2-normalized CLIP frame embeddings using cosine cost",
        "synchronized_clip_distance": "mean cosine distance between frame embeddings at identical retained timestamps",
        "field_distance": {
            "fields": ["A", "P"],
            "scales": list(FIELD_SCALES),
            "definition": "mean of per-field multiscale RMS distances",
        },
        "mass_diagnostic": "mean absolute and signed free-normalized A-mass difference at retained timestamps",
        "point_primary": "median same-seed free/walls post-release CLIP Chamfer minus median pairwise free/free post-release CLIP Chamfer",
        "candidate_aggregation": "median over 15 points",
        "condition_semantics": "high/mid/low labels and absolute steps are selected on the matched optimized C2 trajectory and reused unchanged for its three random controls",
        "run_primary": "optimized candidate value minus median of three random candidate values",
        "candidate_label_randomization": (
            "not reported because optimized/random candidate labels are not "
            "exchangeable by design"
        ),
        "same_seed_control": "paired same-seed free/walls distance is reported beside all six off-seed distances",
        "multiplicity_policy": statistics["multiplicity_policy"],
    }
    _write_json(output_root / "metric_protocol.json", metric_protocol)
    artifact_rows = {
        "branch_pair_metrics.csv": len(pair_frame),
        "branch_frame_metrics.csv": len(frame_frame),
        "point_metrics.csv": len(point_frame),
        "candidate_summary.csv": len(candidate_frame),
        "candidate_frame_summary.csv": len(candidate_time),
        "run_summary.csv": len(run_frame),
        "run_condition_summary.csv": len(condition_frame),
        "run_frame_summary.csv": len(run_time_frame),
        "c5_statistical_table.csv": len(statistical_table),
        "c5_statistical_table.tex": None,
        "statistical_summary.json": None,
        "metric_protocol.json": None,
    }
    artifacts = {}
    for filename, row_count in artifact_rows.items():
        path = output_root / filename
        artifacts[filename] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "rows": row_count,
        }
    artifact_manifest = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": (
            analysis_code_identity_sha256
        ),
        "simulation_protocol_version": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "model_identity_sha256": model_identity_sha256,
        "metric_input_sha256": metric_input_sha256,
        "artifacts": artifacts,
    }
    artifact_manifest["manifest_identity_sha256"] = _identity_sha256(
        artifact_manifest
    )
    _write_json(
        output_root / "metric_artifact_manifest.json",
        artifact_manifest,
    )
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "metric_input_sha256": metric_input_sha256,
        "metric_artifact_manifest_sha256": _sha256_file(
            output_root / "metric_artifact_manifest.json"
        ),
        "metric_artifact_identity_sha256": artifact_manifest[
            "manifest_identity_sha256"
        ],
        "n_branch_pairs": len(pair_frame),
        "n_branch_frames": len(frame_frame),
        "n_points": len(point_frame),
        "n_candidates": len(candidate_frame),
        "n_runs": len(run_frame),
        "primary_metric": primary,
        "primary_median_run_effect": observed,
        "statistical_summary": str(output_root / "statistical_summary.json"),
    }
    _write_json(output_root / "metrics_summary.json", _json_clean(summary))
    return summary


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_plots(output_root: Path) -> dict[str, Any]:
    _setup_plot_style()
    metric_manifest_path = output_root / "metric_artifact_manifest.json"
    if not metric_manifest_path.exists():
        raise RuntimeError("Metric artifact manifest is missing")
    metric_manifest = json.loads(metric_manifest_path.read_text())
    analysis_code_fingerprint = _analysis_code_fingerprint()
    if (
        metric_manifest.get("status") != "complete"
        or metric_manifest.get("analysis_code_identity_sha256")
        != analysis_code_fingerprint["identity_sha256"]
    ):
        raise RuntimeError(
            "Metric artifacts do not belong to the current analysis code"
        )
    figure_inputs = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": analysis_code_fingerprint[
            "identity_sha256"
        ],
        "metric_artifact_manifest_sha256": _sha256_file(
            metric_manifest_path
        ),
        "metric_artifact_identity_sha256": metric_manifest[
            "manifest_identity_sha256"
        ],
    }
    figure_input_sha256 = _identity_sha256(figure_inputs)
    candidates = pd.read_csv(output_root / "candidate_summary.csv")
    runs = pd.read_csv(output_root / "run_summary.csv")
    conditions = pd.read_csv(output_root / "run_condition_summary.csv")
    run_time = pd.read_csv(output_root / "run_frame_summary.csv")
    points = pd.read_csv(output_root / "point_metrics.csv")
    stats = json.loads((output_root / "statistical_summary.json").read_text())
    figure_dir = output_root / "figures"
    generated: list[str] = []
    primary = "excess_clip_post_release"

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for run_idx in range(RUN_COUNT):
        rows = candidates[candidates["run_idx"] == run_idx]
        opt = float(rows[rows["candidate_kind"] == "optimized"][primary].iloc[0])
        random = rows[rows["candidate_kind"] == "random"].sort_values("candidate_idx")
        random_values = random[primary].to_numpy(dtype=float)
        ax.plot(
            [run_idx, run_idx],
            [np.median(random_values), opt],
            color="#B4BAC2",
            lw=1.0,
            zorder=1,
        )
        ax.scatter(
            run_idx + np.asarray([-0.13, 0.0, 0.13]),
            random_values,
            s=24,
            color=RANDOM_COLOR,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(run_idx, opt, s=42, color=OPT_COLOR, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(range(RUN_COUNT), [f"{idx:03d}" for idx in range(RUN_COUNT)])
    ax.set_xlabel("Optimization run")
    ax.set_ylabel("Excess post-release CLIP divergence")
    ax.set_title("C5 paired frustration effect by matched run")
    ax.scatter([], [], color=OPT_COLOR, label="Optimized")
    ax.scatter([], [], color=RANDOM_COLOR, label="Random controls")
    ax.legend(loc="best", ncols=2)
    generated.extend(_save_figure(fig, figure_dir / "c5_primary_by_run"))

    values = runs[f"contrast_{primary}"].to_numpy(dtype=float)
    median_value = float(np.median(values))
    primary_test = stats["tests"][primary]

    def frustration_bar(stem: str, *, clean_title: bool) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
        colors = np.where(values >= 0, "#3FA447", "#D94C4C")
        ax.bar(np.arange(RUN_COUNT), values, color=colors, width=0.7)
        ax.scatter(
            np.arange(RUN_COUNT),
            values,
            color=colors,
            edgecolor="white",
            lw=0.7,
            s=32,
            zorder=3,
        )
        ax.axhline(0, color="#222222", lw=0.9)
        ax.set_xticks(range(RUN_COUNT), [f"{idx:03d}" for idx in range(RUN_COUNT)])
        ax.set_xlabel("Matched optimization run")
        ax.set_ylabel("F(optimized) - median F(random)")
        if clean_title:
            ax.set_title(
                "C5 paired frustration contrast: Flow-Lenia\n"
                f"positive={int(np.sum(values > 0))}/{RUN_COUNT}   "
                f"median={median_value:.4g}   "
                f"sign-test p={primary_test['sign_test_p_greater']:.3g}"
            )
        generated.extend(_save_figure(fig, figure_dir / stem))

    frustration_bar("c5_run_contrasts", clean_title=True)
    frustration_bar("flow_c5_frustration_clean", clean_title=True)
    frustration_bar("flow_c5_frustration_paper", clean_title=False)

    matrix = np.empty((RUN_COUNT, 4), dtype=float)
    for run_idx in range(RUN_COUNT):
        rows = candidates[candidates["run_idx"] == run_idx]
        matrix[run_idx, 0] = float(
            rows[rows["candidate_kind"] == "optimized"][primary].iloc[0]
        )
        matrix[run_idx, 1:] = (
            rows[rows["candidate_kind"] == "random"]
            .sort_values("candidate_idx")[primary]
            .to_numpy(dtype=float)
        )
    bound = float(np.nanpercentile(np.abs(matrix), 98))
    bound = max(bound, 1e-8)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    ax.set_xticks(range(4), ["Optimized", "Random 0", "Random 1", "Random 2"])
    ax.set_yticks(range(RUN_COUNT), [f"Run {idx:03d}" for idx in range(RUN_COUNT)])
    ax.set_title("Candidate-level excess frustration")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("Excess post-release CLIP divergence")
    generated.extend(_save_figure(fig, figure_dir / "c5_candidate_heatmap"))

    condition_order = ["high", "mid", "low"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for run_idx in range(RUN_COUNT):
        rows = conditions[
            (conditions["run_idx"] == run_idx)
            & (conditions["metric"] == "excess_clip_post_release")
        ].set_index("condition")
        y = np.asarray([rows.loc[item, "contrast"] for item in condition_order], dtype=float)
        ax.plot(range(3), y, color="#B6BCC4", lw=0.8, alpha=0.75)
        ax.scatter(range(3), y, color=NEUTRAL, s=12, alpha=0.75)
    medians = []
    lows = []
    highs = []
    for idx, condition in enumerate(condition_order):
        values = conditions[
            (conditions["condition"] == condition)
            & (conditions["metric"] == "excess_clip_post_release")
        ]["contrast"].to_numpy(dtype=float)
        low, high = _bootstrap_median_ci(values, seed=7_000 + idx)
        medians.append(float(np.median(values)))
        lows.append(low)
        highs.append(high)
    ax.errorbar(
        range(3),
        medians,
        yerr=[
            np.asarray(medians) - np.asarray(lows),
            np.asarray(highs) - np.asarray(medians),
        ],
        color=OPT_COLOR,
        marker="o",
        lw=2.0,
        capsize=3,
        label="Median run effect, 95% bootstrap CI",
    )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(range(3), ["High ΔH", "Mid ΔH", "Low ΔH"])
    ax.set_ylabel("Optimized minus random median")
    ax.set_title("C5 effect by C2 divergence stratum")
    ax.legend(loc="best")
    generated.extend(_save_figure(fig, figure_dir / "c5_condition_effects"))

    fig, (ax_abs, ax_contrast) = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.7),
        constrained_layout=True,
        sharex=True,
    )
    relative_steps = sorted(run_time["relative_step"].unique())
    opt_median = []
    random_median = []
    contrast_median = []
    contrast_low = []
    contrast_high = []
    for idx, rel_step in enumerate(relative_steps):
        rows = run_time[run_time["relative_step"] == rel_step]
        opt_median.append(float(np.median(rows["optimized"])))
        random_median.append(float(np.median(rows["random_median"])))
        contrast_values = rows["contrast"].to_numpy(dtype=float)
        contrast_median.append(float(np.median(contrast_values)))
        low, high = _bootstrap_median_ci(contrast_values, seed=8_000 + idx)
        contrast_low.append(low)
        contrast_high.append(high)
    x = np.asarray(relative_steps, dtype=float) / 1000.0
    ax_abs.plot(x, opt_median, color=OPT_COLOR, marker="o", label="Optimized")
    ax_abs.plot(x, random_median, color=RANDOM_COLOR, marker="o", label="Random median")
    ax_abs.set_ylabel("Paired frame cosine distance")
    ax_abs.set_title("Absolute wall effect")
    ax_abs.legend(loc="best")
    ax_contrast.plot(x, contrast_median, color=FREE_COLOR, marker="o")
    ax_contrast.fill_between(x, contrast_low, contrast_high, color=FREE_COLOR, alpha=0.2)
    ax_contrast.axhline(0, color="#222222", lw=0.8)
    ax_contrast.set_ylabel("Optimized minus random median")
    ax_contrast.set_title("Run-blocked contrast")
    for axis in (ax_abs, ax_contrast):
        axis.axvspan(0, WALL_STEPS / 1000.0, color=WALL_COLOR, alpha=0.12)
        axis.axvline(WALL_STEPS / 1000.0, color=WALL_COLOR, ls="--", lw=1.2)
        axis.set_xlabel("Steps after branch (thousands)")
    generated.extend(_save_figure(fig, figure_dir / "c5_time_resolved"))

    fig, ax = plt.subplots(figsize=(5.3, 4.3), constrained_layout=True)
    for kind, color, label in (
        ("optimized", OPT_COLOR, "Optimized"),
        ("random", RANDOM_COLOR, "Random"),
    ):
        subset = candidates[candidates["candidate_kind"] == kind]
        ax.scatter(
            subset["excess_field_post_release"],
            subset["excess_clip_post_release"],
            color=color,
            s=32,
            alpha=0.8,
            label=label,
        )
    ax.axhline(0, color="#222222", lw=0.7)
    ax.axvline(0, color="#222222", lw=0.7)
    ax.set_xlabel("Excess post-release multiscale A/P divergence")
    ax.set_ylabel("Excess post-release CLIP divergence")
    ax.set_title("Pixel-field and perceptual C5 estimands")
    ax.legend(loc="best")
    generated.extend(_save_figure(fig, figure_dir / "c5_clip_vs_field"))

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for kind, color, label in (
        ("optimized", OPT_COLOR, "Optimized"),
        ("random", RANDOM_COLOR, "Random"),
    ):
        subset = points[points["candidate_kind"] == kind]
        ax.scatter(
            subset["delta_h"],
            subset["excess_clip_post_release"],
            color=color,
            s=10,
            alpha=0.35,
            label=label,
        )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xlabel("ΔH at selected branch state")
    ax.set_ylabel("Excess post-release CLIP divergence")
    ax.set_title("Selected-state ΔH and frustration effect")
    ax.legend(loc="best")
    generated.extend(_save_figure(fig, figure_dir / "c5_delta_h_relation"))

    generated_paths = [Path(path) for path in generated]
    stems = sorted({path.stem for path in generated_paths})
    if stems != sorted(REQUIRED_FIGURE_STEMS):
        raise RuntimeError(
            f"Figure stem mismatch: found={stems}, "
            f"expected={sorted(REQUIRED_FIGURE_STEMS)}"
        )
    files = {
        path.name: {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in generated_paths
    }
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "figure_inputs": figure_inputs,
        "figure_input_sha256": figure_input_sha256,
        "n_figure_files": len(generated),
        "n_figure_stems": len(generated) // 2,
        "required_stems": list(REQUIRED_FIGURE_STEMS),
        "files": files,
        "primary_statistic": stats["primary_observed_median_run_effect"],
    }
    _write_json(output_root / "figures_summary.json", _json_clean(summary))
    return summary


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if not path.exists():
        fallback = "VeraBd.ttf" if bold else "Vera.ttf"
        path = Path("/usr/share/fonts/truetype/ttf-bitstream-vera") / fallback
    return ImageFont.truetype(str(path), size=size)


def _rgb_u8(fields: dict[str, np.ndarray]) -> np.ndarray:
    return np.rint(_render_apf_rgb(fields) * 255.0).clip(0, 255).astype(np.uint8)


def _wall_lines(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for fraction in (1 / 3, 2 / 3):
        x = int(round(width * fraction))
        y = int(round(height * fraction))
        draw.line((x, 0, x, height), fill=(255, 255, 255), width=3)
        draw.line((0, y, width, y), fill=(255, 255, 255), width=3)
        draw.line((x + 3, 0, x + 3, height), fill=(20, 20, 20), width=1)
        draw.line((0, y + 3, width, y + 3), fill=(20, 20, 20), width=1)


def _video_frame(
    free_rgb: list[np.ndarray],
    wall_rgb: list[np.ndarray],
    *,
    frame_idx: int,
    relative_step: int,
    candidate_id: str,
    point_id: int,
    wall_active: bool,
) -> np.ndarray:
    tile_size = 256
    header = 48
    left = 86
    gap = 4
    canvas = Image.new(
        "RGB",
        (left + 3 * tile_size + 2 * gap, header + 2 * tile_size + gap),
        color=(245, 247, 249),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 7),
        f"{candidate_id} | point {point_id:02d} | +{relative_step:,} steps",
        fill=(20, 25, 30),
        font=_font(18, bold=True),
    )
    for branch_id in range(3):
        x = left + branch_id * (tile_size + gap)
        draw.text(
            (x + 84, 29),
            f"Branch {branch_id}",
            fill=(45, 50, 56),
            font=_font(13),
        )
        free = Image.fromarray(free_rgb[branch_id][frame_idx]).resize(
            (tile_size, tile_size),
            resample=Image.Resampling.NEAREST,
        )
        walls = Image.fromarray(wall_rgb[branch_id][frame_idx]).resize(
            (tile_size, tile_size),
            resample=Image.Resampling.NEAREST,
        )
        if wall_active:
            _wall_lines(walls)
        canvas.paste(free, (x, header))
        canvas.paste(walls, (x, header + tile_size + gap))
    draw.text((10, header + tile_size // 2 - 10), "Free", fill=(20, 80, 55), font=_font(18, bold=True))
    draw.text(
        (7, header + tile_size + gap + tile_size // 2 - 22),
        "Walls",
        fill=(145, 45, 100),
        font=_font(18, bold=True),
    )
    draw.text(
        (7, header + tile_size + gap + tile_size // 2 + 2),
        "then free",
        fill=(145, 45, 100),
        font=_font(13),
    )
    phase = "walls active" if wall_active else "walls removed"
    draw.text(
        (canvas.width - 132, 8),
        phase,
        fill=(145, 45, 100) if wall_active else (20, 100, 70),
        font=_font(13, bold=True),
    )
    return np.asarray(canvas)


def _representative_video_points(plan: pd.DataFrame) -> dict[int, int]:
    optimized = plan[plan["candidate_kind"] == "optimized"]
    representative: dict[int, int] = {}
    for run_idx, rows in optimized.groupby("run_idx"):
        by_point = rows.groupby("point_id", as_index=False)["delta_h"].first()
        representative[int(run_idx)] = int(
            by_point.sort_values(
                ["delta_h", "point_id"],
                ascending=[False, True],
            ).iloc[0]["point_id"]
        )
    if set(representative) != set(range(RUN_COUNT)):
        raise RuntimeError(
            f"Representative video points lack runs: {sorted(representative)}"
        )
    return representative


def _video_input_identity(
    rows: pd.DataFrame,
    *,
    candidate_id: str,
    point_id: int,
    fps: int,
    hold_frames: int,
) -> dict[str, Any]:
    if rows["branch_id"].sort_values().tolist() != [0, 1, 2]:
        raise RuntimeError(
            f"Video input for {candidate_id}/point_{point_id} lacks three branches"
        )
    sources = []
    for row in rows.sort_values("branch_id").itertuples():
        for arm, branch_dir in (
            ("free", row.free_branch_dir),
            ("walls", row.walls_branch_dir),
        ):
            path = _branch_npz(_resolve(branch_dir))
            sources.append(
                {
                    "row_id": int(row.row_id),
                    "branch_id": int(row.branch_id),
                    "arm": arm,
                    "apf_path": str(path),
                    "apf_sha256": _sha256_file(path),
                }
            )
    identity = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": _analysis_code_fingerprint()[
            "identity_sha256"
        ],
        "candidate_id": candidate_id,
        "point_id": int(point_id),
        "fps": int(fps),
        "hold_frames_per_snapshot": int(hold_frames),
        "expected_frames": CAPTURE_COUNT * int(hold_frames),
        "sources": sources,
    }
    identity["input_sha256"] = _identity_sha256(identity)
    return identity


def _validate_video_file(
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    os.environ.setdefault(
        "IMAGEIO_FFMPEG_EXE",
        imageio_ffmpeg.get_ffmpeg_exe(),
    )

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
        frame_count = 0
        for frame in reader:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise RuntimeError(f"Unexpected video frame shape {array.shape}")
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise RuntimeError(
                    f"Inconsistent video frame shape {array.shape}"
                )
            frame_count += 1
        if expected_shape is not None:
            result["height"] = int(expected_shape[0])
            result["width"] = int(expected_shape[1])
        result["frames"] = frame_count
        result["passed"] = (
            frame_count == int(expected_frames)
            and result["width"] > 0
            and result["height"] > 0
        )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if reader is not None:
            reader.close()
    return result


def build_videos(
    plan: pd.DataFrame,
    *,
    output_root: Path,
    fps: int,
    hold_frames: int,
    force: bool,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    videos_dir = output_root / "videos_by_candidate"
    videos_dir.mkdir(parents=True, exist_ok=True)
    representative = _representative_video_points(plan)
    manifest_rows = []
    candidates = (
        plan[
            [
                "run_idx",
                "candidate_id",
                "candidate_kind",
                "candidate_idx",
            ]
        ]
        .drop_duplicates()
        .sort_values(["run_idx", "candidate_kind", "candidate_idx"])
    )
    for candidate_number, candidate in enumerate(candidates.itertuples(), start=1):
        point_id = representative[int(candidate.run_idx)]
        rows = plan[
            (plan["candidate_id"] == candidate.candidate_id)
            & (plan["point_id"] == point_id)
        ].sort_values("branch_id")
        output = videos_dir / f"{candidate.candidate_id}_point_{point_id:02d}.mp4"
        provenance_path = output.with_suffix(".provenance.json")
        video_input = _video_input_identity(
            rows,
            candidate_id=str(candidate.candidate_id),
            point_id=point_id,
            fps=int(fps),
            hold_frames=int(hold_frames),
        )
        expected_frames = CAPTURE_COUNT * int(hold_frames)
        cached_provenance = {}
        if provenance_path.exists():
            try:
                cached_provenance = json.loads(
                    provenance_path.read_text()
                )
            except Exception:
                cached_provenance = {}
        validation = _validate_video_file(
            output,
            expected_frames=expected_frames,
        )
        output_sha256 = (
            _sha256_file(output)
            if output.exists()
            else ""
        )
        reusable = bool(
            not force
            and validation["passed"]
            and cached_provenance.get("input_sha256")
            == video_input["input_sha256"]
            and cached_provenance.get("video_sha256")
            == output_sha256
            and int(cached_provenance.get("expected_frames", -1))
            == expected_frames
        )
        if reusable:
            status = "reused"
        else:
            free_rgb = []
            wall_rgb = []
            relative = None
            for row in rows.itertuples():
                free = _load_ap_fields(_resolve(row.free_branch_dir))
                walls = _load_ap_fields(_resolve(row.walls_branch_dir))
                free_rgb.append(_rgb_u8({"A": free["A"], "P": free["P"]}))
                wall_rgb.append(_rgb_u8({"A": walls["A"], "P": walls["P"]}))
                current = np.asarray(free["steps"], dtype=np.int64) - int(row.step)
                if relative is None:
                    relative = current
                elif not np.array_equal(relative, current):
                    raise RuntimeError(f"Video frame mismatch for {candidate.candidate_id}")
            assert relative is not None
            writer = imageio.get_writer(
                output,
                fps=int(fps),
                codec="libx264",
                quality=8,
                macro_block_size=2,
                pixelformat="yuv420p",
            )
            try:
                for frame_idx, rel_step in enumerate(relative):
                    frame = _video_frame(
                        free_rgb,
                        wall_rgb,
                        frame_idx=frame_idx,
                        relative_step=int(rel_step),
                        candidate_id=str(candidate.candidate_id),
                        point_id=point_id,
                        wall_active=bool(rel_step <= WALL_STEPS),
                    )
                    for _ in range(int(hold_frames)):
                        writer.append_data(frame)
            finally:
                writer.close()
            validation = _validate_video_file(
                output,
                expected_frames=expected_frames,
            )
            if not validation["passed"]:
                raise RuntimeError(
                    f"Generated video failed decode audit: {output}: {validation}"
                )
            output_sha256 = _sha256_file(output)
            video_provenance = {
                **video_input,
                "video": str(output),
                "video_sha256": output_sha256,
                "bytes": int(output.stat().st_size),
                "validation": validation,
            }
            _write_json(provenance_path, video_provenance)
            status = "generated"
        if not provenance_path.exists():
            raise RuntimeError(f"Missing video provenance: {provenance_path}")
        manifest_rows.append(
            {
                "run_idx": int(candidate.run_idx),
                "candidate_id": str(candidate.candidate_id),
                "candidate_kind": str(candidate.candidate_kind),
                "candidate_idx": int(candidate.candidate_idx),
                "point_id": point_id,
                "selection_rule": "maximum optimized-candidate delta_h within matched run; same point reused for all four candidates",
                "video": str(output),
                "video_sha256": output_sha256,
                "video_input_sha256": video_input["input_sha256"],
                "provenance": str(provenance_path),
                "provenance_sha256": _sha256_file(provenance_path),
                "bytes": int(output.stat().st_size),
                "fps": int(fps),
                "hold_frames_per_snapshot": int(hold_frames),
                "decoded_frames": int(validation["frames"]),
                "width": int(validation["width"]),
                "height": int(validation["height"]),
                "status": status,
            }
        )
        print(
            f"[videos] {candidate_number}/{len(candidates)} "
            f"{candidate.candidate_id}: {status}",
            flush=True,
        )
    manifest = pd.DataFrame(manifest_rows)
    _write_table(output_root / "video_manifest.csv", manifest)
    summary = {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code_identity_sha256": _analysis_code_fingerprint()[
            "identity_sha256"
        ],
        "n_videos": len(manifest),
        "n_generated": int((manifest["status"] == "generated").sum()),
        "n_reused": int((manifest["status"] == "reused").sum()),
        "manifest": str(output_root / "video_manifest.csv"),
        "manifest_sha256": _sha256_file(
            output_root / "video_manifest.csv"
        ),
    }
    _write_json(output_root / "videos_summary.json", summary)
    return summary


def completion_audit(
    plan: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    pairing_columns = [
        "condition",
        "pair_id",
        "point_id",
        "window_index",
        "step",
        "branch_id",
        "branch_seed",
        "horizon_steps",
        "wall_steps",
        "perturb_a_std",
        "perturb_p_std",
        "perturb_lagrangian_xy_std",
        "selection_reference_traj_id",
    ]
    pairing_signatures: dict[int, dict[str, str]] = {}
    for run_idx, run_rows in plan.groupby("run_idx", sort=True):
        pairing_signatures[int(run_idx)] = {
            str(candidate_id): _identity_sha256(
                candidate_rows.sort_values(
                    ["point_id", "branch_id"]
                )[pairing_columns].to_dict(orient="records")
            )
            for candidate_id, candidate_rows in run_rows.groupby(
                "candidate_id",
                sort=True,
            )
        }
    checks["paired_plan_integrity"] = {
        "passed": bool(
            len(pairing_signatures) == RUN_COUNT
            and all(
                len(signatures) == 4
                and len(set(signatures.values())) == 1
                for signatures in pairing_signatures.values()
            )
        ),
        "run_candidate_pairing_signatures": pairing_signatures,
        "description": (
            "all four candidates in every run have identical selected "
            "conditions, absolute steps, branch seeds, perturbations, and horizons"
        ),
    }
    protocol_audit_path = output_root / "protocol_audit.json"
    protocol_audit = (
        json.loads(protocol_audit_path.read_text())
        if protocol_audit_path.exists()
        else {}
    )
    checks["simulation_protocol_audit"] = {
        "passed": (
            protocol_audit.get("status") == "passed"
            and protocol_audit.get("protocol_version")
            == PROTOCOL_VERSION
            and protocol_audit.get("plan_sha256")
            == protocol["plan_sha256"]
            and protocol_audit.get(
                "simulation_code_bundle_sha256"
            )
            == protocol["simulation_code_bundle_sha256"]
            and int(protocol_audit.get("n_selected", -1)) == EXPECTED_PLAN_ROWS
            and int(protocol_audit.get("n_free_ready", -1)) == EXPECTED_PLAN_ROWS
            and int(protocol_audit.get("n_walls_ready", -1)) == EXPECTED_PLAN_ROWS
            and protocol_audit.get("full_protocol_scope") is True
            and protocol_audit.get("all_params_exact") is True
            and protocol_audit.get("all_initial_states_exact") is True
            and protocol_audit.get(
                "all_top_level_rng_streams_exact"
            )
            is True
            and protocol_audit.get(
                "all_global_mutation_streams_exact"
            )
            is True
            and protocol_audit.get(
                "simulation_code_fingerprint_exact"
            )
            is True
            and protocol_audit.get(
                "all_simulation_configs_exact"
            )
            is True
            and protocol_audit.get(
                "all_optimizer_native_batch_indices_exact"
            )
            is True
            and protocol_audit.get(
                "free_cache_equivalence_exact"
            )
            is True
            and protocol_audit.get("preflight_exact") is True
        ),
        "path": str(protocol_audit_path),
    }
    expected_tables = {
        "embedding_manifest.csv": 2 * EXPECTED_PLAN_ROWS,
        "branch_pair_metrics.csv": BRANCH_PAIR_COUNT,
        "branch_frame_metrics.csv": FRAME_PAIR_COUNT,
        "point_metrics.csv": POINT_COUNT,
        "candidate_summary.csv": CANDIDATE_COUNT,
        "candidate_frame_summary.csv": CANDIDATE_COUNT * CAPTURE_COUNT,
        "run_summary.csv": RUN_COUNT,
        "run_condition_summary.csv": RUN_COUNT * 3 * 2,
        "run_frame_summary.csv": RUN_COUNT * CAPTURE_COUNT,
        "c5_statistical_table.csv": SUMMARY_METRIC_COUNT,
        "video_manifest.csv": CANDIDATE_COUNT,
    }
    loaded_tables: dict[str, pd.DataFrame] = {}
    for filename, expected_rows in expected_tables.items():
        path = output_root / filename
        try:
            frame = pd.read_csv(path) if path.exists() else None
        except Exception:
            frame = None
        actual = len(frame) if frame is not None else -1
        if frame is not None:
            loaded_tables[filename] = frame
        checks[f"table_{filename}"] = {
            "passed": actual == expected_rows,
            "expected_rows": expected_rows,
            "actual_rows": actual,
            "path": str(path),
        }
    try:
        embeddings = loaded_tables["embedding_manifest.csv"]
        pairs = loaded_tables["branch_pair_metrics.csv"]
        frames = loaded_tables["branch_frame_metrics.csv"]
        points = loaded_tables["point_metrics.csv"]
        candidates = loaded_tables["candidate_summary.csv"]
        candidate_frames = loaded_tables["candidate_frame_summary.csv"]
        runs = loaded_tables["run_summary.csv"]
        conditions = loaded_tables["run_condition_summary.csv"]
        run_frames = loaded_tables["run_frame_summary.csv"]
        videos = loaded_tables["video_manifest.csv"]
        offsets = {0, 2850, 5700, 8550, 11400, 14250, 17100, 20000}
        expected_embedding_keys = {
            (
                int(row.row_id),
                int(row.run_idx),
                str(row.candidate_id),
                str(row.candidate_kind),
                int(row.candidate_idx),
                int(row.point_id),
                int(row.branch_id),
                arm,
            )
            for row in plan.itertuples()
            for arm in ("free", "walls")
        }
        actual_embedding_keys = {
            (
                int(row.row_id),
                int(row.run_idx),
                str(row.candidate_id),
                str(row.candidate_kind),
                int(row.candidate_idx),
                int(row.point_id),
                int(row.branch_id),
                str(row.arm),
            )
            for row in embeddings.itertuples()
        }
        embedding_groups = embeddings.groupby("row_id")["arm"].agg(
            lambda values: set(values)
        )
        pair_groups = pairs.groupby(["candidate_id", "point_id"])
        pair_type_counts = pair_groups["pair_type"].value_counts().unstack(
            fill_value=0
        )
        frame_groups = frames.groupby(["candidate_id", "point_id"])
        point_condition_counts = (
            points.groupby(["candidate_id", "condition"]).size()
        )
        candidate_composition = (
            candidates.groupby(["run_idx", "candidate_kind"]).size()
        )
        checks["table_key_integrity"] = {
            "passed": bool(
                not embeddings.duplicated(["row_id", "arm"]).any()
                and actual_embedding_keys == expected_embedding_keys
                and set(embeddings["row_id"]) == set(range(EXPECTED_PLAN_ROWS))
                and len(embedding_groups) == EXPECTED_PLAN_ROWS
                and all(value == {"free", "walls"} for value in embedding_groups)
                and not pairs.duplicated(
                    [
                        "candidate_id",
                        "point_id",
                        "pair_type",
                        "left_branch",
                        "right_branch",
                    ]
                ).any()
                and bool((pair_groups.size() == 15).all())
                and bool(
                    (
                        pair_type_counts[
                            [
                                "paired_same_seed",
                                "paired_off_seed",
                                "free_within",
                                "walls_within",
                            ]
                        ]
                        == np.asarray([3, 6, 3, 3])
                    ).all().all()
                )
                and not frames.duplicated(
                    ["candidate_id", "point_id", "branch_id", "frame_idx"]
                ).any()
                and bool((frame_groups.size() == 24).all())
                and set(frames["relative_step"].astype(int)) == offsets
                and not points.duplicated(["candidate_id", "point_id"]).any()
                and bool((points.groupby("candidate_id").size() == 15).all())
                and bool((point_condition_counts == 5).all())
                and not candidates.duplicated(["candidate_id"]).any()
                and all(
                    candidate_composition.get((run_idx, "optimized"), 0) == 1
                    and candidate_composition.get((run_idx, "random"), 0) == 3
                    for run_idx in range(RUN_COUNT)
                )
                and set(
                    candidates.loc[
                        candidates["candidate_kind"] == "random",
                        "candidate_idx",
                    ].astype(int)
                )
                == {0, 1, 2}
                and not candidate_frames.duplicated(
                    ["candidate_id", "relative_step"]
                ).any()
                and bool(
                    (
                        candidate_frames.groupby("candidate_id").size()
                        == CAPTURE_COUNT
                    ).all()
                )
                and set(candidate_frames["relative_step"].astype(int)) == offsets
                and list(sorted(runs["run_idx"].astype(int))) == list(range(RUN_COUNT))
                and not runs.duplicated(["run_idx"]).any()
                and bool((conditions.groupby("run_idx").size() == 6).all())
                and bool((run_frames.groupby("run_idx").size() == 8).all())
                and set(run_frames["relative_step"].astype(int)) == offsets
                and not videos.duplicated(["candidate_id"]).any()
            ),
            "description": (
                "unique primary keys, exact branch/frame cardinalities, "
                "1 optimized + 3 random candidates per run, and fixed offsets"
            ),
        }
        finite_tables = (
            "branch_pair_metrics.csv",
            "branch_frame_metrics.csv",
            "point_metrics.csv",
            "candidate_summary.csv",
            "candidate_frame_summary.csv",
            "run_summary.csv",
            "run_condition_summary.csv",
            "run_frame_summary.csv",
            "c5_statistical_table.csv",
        )
        nonfinite: dict[str, list[str]] = {}
        for filename in finite_tables:
            frame = loaded_tables[filename]
            bad_columns = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if not np.isfinite(
                    frame[column].to_numpy(dtype=np.float64)
                ).all()
            ]
            if bad_columns:
                nonfinite[filename] = bad_columns
        checks["finite_metrics"] = {
            "passed": not nonfinite,
            "nonfinite_columns": nonfinite,
        }
        current_code_identity = _analysis_code_fingerprint()[
            "identity_sha256"
        ]
        embedding_cache_errors = []
        for row in embeddings.itertuples():
            cache_path = Path(row.embedding_cache)
            apf_path = Path(row.apf_path)
            try:
                if _sha256_file(apf_path) != str(
                    row.source_apf_sha256
                ):
                    raise RuntimeError("source APF hash mismatch")
                if _sha256_file(cache_path) != str(
                    row.embedding_cache_sha256
                ):
                    raise RuntimeError("embedding cache hash mismatch")
                with np.load(cache_path, allow_pickle=False) as data:
                    if (
                        int(np.asarray(data["row_id"]).item())
                        != int(row.row_id)
                        or str(np.asarray(data["arm"]).item())
                        != str(row.arm)
                        or str(
                            np.asarray(
                                data[
                                    "analysis_code_identity_sha256"
                                ]
                            ).item()
                        )
                        != current_code_identity
                        or str(
                            np.asarray(
                                data["model_identity_sha256"]
                            ).item()
                        )
                        != str(row.model_identity_sha256)
                        or str(
                            np.asarray(
                                data["source_apf_sha256"]
                            ).item()
                        )
                        != str(row.source_apf_sha256)
                    ):
                        raise RuntimeError(
                            "embedding cache provenance mismatch"
                        )
            except Exception as exc:
                embedding_cache_errors.append(
                    {
                        "row_id": int(row.row_id),
                        "arm": str(row.arm),
                        "error": repr(exc),
                    }
                )
                if len(embedding_cache_errors) >= 20:
                    break
        checks["embedding_cache_integrity"] = {
            "passed": not embedding_cache_errors,
            "n_audited": (
                len(embeddings)
                if not embedding_cache_errors
                else None
            ),
            "errors": embedding_cache_errors,
        }
    except Exception as exc:
        checks["table_key_integrity"] = {
            "passed": False,
            "error": repr(exc),
        }
        checks["finite_metrics"] = {
            "passed": False,
            "error": repr(exc),
        }
        checks["embedding_cache_integrity"] = {
            "passed": False,
            "error": repr(exc),
        }
    metric_artifact_path = output_root / "metric_artifact_manifest.json"
    try:
        metric_artifact = json.loads(metric_artifact_path.read_text())
        stored_identity = metric_artifact.get(
            "manifest_identity_sha256"
        )
        identity_payload = dict(metric_artifact)
        identity_payload.pop("manifest_identity_sha256", None)
        expected_artifacts = {
            "branch_pair_metrics.csv",
            "branch_frame_metrics.csv",
            "point_metrics.csv",
            "candidate_summary.csv",
            "candidate_frame_summary.csv",
            "run_summary.csv",
            "run_condition_summary.csv",
            "run_frame_summary.csv",
            "c5_statistical_table.csv",
            "c5_statistical_table.tex",
            "statistical_summary.json",
            "metric_protocol.json",
        }
        artifact_errors = []
        artifacts = metric_artifact.get("artifacts", {})
        if set(artifacts) != expected_artifacts:
            artifact_errors.append("artifact filename set mismatch")
        for filename, record in artifacts.items():
            path = output_root / filename
            if (
                not path.exists()
                or _sha256_file(path) != record.get("sha256")
                or int(path.stat().st_size)
                != int(record.get("bytes", -1))
            ):
                artifact_errors.append(filename)
                continue
            if (
                filename in expected_tables
                and int(record.get("rows", -1))
                != int(expected_tables[filename])
            ):
                artifact_errors.append(f"{filename}:rows")
        checks["metric_artifact_integrity"] = {
            "passed": bool(
                metric_artifact.get("status") == "complete"
                and metric_artifact.get("analysis_version")
                == ANALYSIS_VERSION
                and metric_artifact.get(
                    "analysis_code_identity_sha256"
                )
                == _analysis_code_fingerprint()["identity_sha256"]
                and metric_artifact.get(
                    "simulation_protocol_version"
                )
                == PROTOCOL_VERSION
                and metric_artifact.get("plan_sha256")
                == protocol["plan_sha256"]
                and metric_artifact.get(
                    "simulation_code_bundle_sha256"
                )
                == protocol["simulation_code_bundle_sha256"]
                and stored_identity == _identity_sha256(identity_payload)
                and not artifact_errors
            ),
            "path": str(metric_artifact_path),
            "errors": artifact_errors,
        }
    except Exception as exc:
        metric_artifact = {}
        checks["metric_artifact_integrity"] = {
            "passed": False,
            "error": repr(exc),
        }
    figure_summary_path = output_root / "figures_summary.json"
    try:
        figure_summary = json.loads(figure_summary_path.read_text())
        figure_files = figure_summary.get("files", {})
        expected_figure_names = {
            f"{stem}{suffix}"
            for stem in REQUIRED_FIGURE_STEMS
            for suffix in (".png", ".pdf")
        }
        figure_errors = []
        if set(figure_files) != expected_figure_names:
            figure_errors.append("figure filename set mismatch")
        for filename, record in figure_files.items():
            path = output_root / "figures" / filename
            if (
                not path.exists()
                or _sha256_file(path) != record.get("sha256")
                or int(path.stat().st_size)
                != int(record.get("bytes", -1))
            ):
                figure_errors.append(filename)
                continue
            try:
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        image.verify()
                elif path.read_bytes()[:4] != b"%PDF":
                    raise RuntimeError("invalid PDF header")
            except Exception:
                figure_errors.append(f"{filename}:decode")
        figure_inputs = figure_summary.get("figure_inputs", {})
        expected_figure_input = {
            "analysis_version": ANALYSIS_VERSION,
            "analysis_code_identity_sha256": (
                _analysis_code_fingerprint()["identity_sha256"]
            ),
            "metric_artifact_manifest_sha256": _sha256_file(
                metric_artifact_path
            ),
            "metric_artifact_identity_sha256": metric_artifact.get(
                "manifest_identity_sha256"
            ),
        }
        checks["paper_figures"] = {
            "passed": bool(
                figure_summary.get("status") == "complete"
                and figure_summary.get("analysis_version")
                == ANALYSIS_VERSION
                and figure_summary.get("required_stems")
                == list(REQUIRED_FIGURE_STEMS)
                and int(figure_summary.get("n_figure_stems", -1))
                == len(REQUIRED_FIGURE_STEMS)
                and int(figure_summary.get("n_figure_files", -1))
                == 2 * len(REQUIRED_FIGURE_STEMS)
                and figure_inputs == expected_figure_input
                and figure_summary.get("figure_input_sha256")
                == _identity_sha256(expected_figure_input)
                and not figure_errors
            ),
            "errors": figure_errors,
            "paths": [
                str(output_root / "figures" / name)
                for name in sorted(figure_files)
            ],
        }
    except Exception as exc:
        checks["paper_figures"] = {
            "passed": False,
            "error": repr(exc),
        }
    video_manifest_path = output_root / "video_manifest.csv"
    video_errors = []
    valid_videos = 0
    try:
        video_manifest = pd.read_csv(video_manifest_path)
        representative = _representative_video_points(plan)
        for row in video_manifest.itertuples():
            path = Path(row.video)
            provenance_path = Path(row.provenance)
            candidate_rows = plan[
                (plan["candidate_id"] == str(row.candidate_id))
                & (plan["point_id"] == int(row.point_id))
            ].sort_values("branch_id")
            expected_point = representative[int(row.run_idx)]
            expected_input = _video_input_identity(
                candidate_rows,
                candidate_id=str(row.candidate_id),
                point_id=int(row.point_id),
                fps=int(row.fps),
                hold_frames=int(row.hold_frames_per_snapshot),
            )
            provenance = json.loads(provenance_path.read_text())
            validation = _validate_video_file(
                path,
                expected_frames=(
                    CAPTURE_COUNT
                    * int(row.hold_frames_per_snapshot)
                ),
            )
            if not (
                int(row.point_id) == expected_point
                and validation["passed"]
                and _sha256_file(path) == str(row.video_sha256)
                and _sha256_file(provenance_path)
                == str(row.provenance_sha256)
                and str(row.video_input_sha256)
                == expected_input["input_sha256"]
                and provenance.get("input_sha256")
                == expected_input["input_sha256"]
                and provenance.get("video_sha256")
                == str(row.video_sha256)
                and int(row.decoded_frames)
                == int(validation["frames"])
                and int(row.width) == int(validation["width"])
                and int(row.height) == int(validation["height"])
            ):
                raise RuntimeError("video provenance/decode mismatch")
            valid_videos += 1
    except Exception as exc:
        video_errors.append(repr(exc))
    checks["videos"] = {
        "passed": (
            valid_videos == CANDIDATE_COUNT
            and not video_errors
        ),
        "n_valid": valid_videos,
        "n_expected": CANDIDATE_COUNT,
        "errors": video_errors,
    }
    metadata_files = (
        output_root / "embedding_protocol.json",
        output_root / "embedding_pair_zero_audit.json",
        output_root / "metric_protocol.json",
        output_root / "statistical_summary.json",
        output_root / "metrics_summary.json",
        output_root / "metric_artifact_manifest.json",
        output_root / "figures_summary.json",
        output_root / "videos_summary.json",
        output_root / "c5_statistical_table.tex",
    )
    checks["metadata"] = {
        "passed": all(path.exists() and path.stat().st_size > 10 for path in metadata_files),
        "paths": [str(path) for path in metadata_files],
    }
    try:
        embedding_protocol = json.loads(
            (output_root / "embedding_protocol.json").read_text()
        )
        zero_audit = json.loads(
            (output_root / "embedding_pair_zero_audit.json").read_text()
        )
        metric_protocol = json.loads(
            (output_root / "metric_protocol.json").read_text()
        )
        statistics = json.loads(
            (output_root / "statistical_summary.json").read_text()
        )
        metric_summary = json.loads(
            (output_root / "metrics_summary.json").read_text()
        )
        figure_summary = json.loads(
            (output_root / "figures_summary.json").read_text()
        )
        video_summary = json.loads(
            (output_root / "videos_summary.json").read_text()
        )
        model_fingerprint = embedding_protocol.get(
            "model_fingerprint", {}
        )
        analysis_code_fingerprint = _analysis_code_fingerprint()
        analysis_code_identity_sha256 = analysis_code_fingerprint[
            "identity_sha256"
        ]
        upstream = embedding_protocol.get("upstream_c2_reference", {})
        tests = statistics.get("tests", {})
        checks["metadata_integrity"] = {
            "passed": bool(
                embedding_protocol.get("analysis_version")
                == ANALYSIS_VERSION
                and embedding_protocol.get("analysis_script_sha256")
                == _sha256_file(Path(__file__).resolve())
                and embedding_protocol.get(
                    "analysis_code_identity_sha256"
                )
                == analysis_code_identity_sha256
                and embedding_protocol.get(
                    "analysis_code_fingerprint"
                )
                == analysis_code_fingerprint
                and embedding_protocol.get(
                    "simulation_protocol_version"
                )
                == PROTOCOL_VERSION
                and embedding_protocol.get("plan_sha256")
                == protocol["plan_sha256"]
                and embedding_protocol.get(
                    "simulation_code_bundle_sha256"
                )
                == protocol["simulation_code_bundle_sha256"]
                and embedding_protocol.get(
                    "embedding_manifest_sha256"
                )
                == _sha256_file(
                    output_root / "embedding_manifest.csv"
                )
                and int(
                    embedding_protocol.get(
                        "n_uniform_embedding_caches",
                        -1,
                    )
                )
                == 2 * EXPECTED_PLAN_ROWS
                and len(model_fingerprint.get("weights_sha256", "")) == 64
                and len(model_fingerprint.get("identity_sha256", "")) == 64
                and len(
                    model_fingerprint.get(
                        "model_config_sha256",
                        "",
                    )
                )
                == 64
                and len(
                    model_fingerprint.get(
                        "image_processor_config_sha256",
                        "",
                    )
                )
                == 64
                and model_fingerprint.get(
                    "model_config_sha256"
                )
                == _identity_sha256(
                    model_fingerprint.get("model_config", {})
                )
                and model_fingerprint.get(
                    "image_processor_config_sha256"
                )
                == _identity_sha256(
                    model_fingerprint.get(
                        "image_processor_config",
                        {},
                    )
                )
                and upstream.get("status") == "passed"
                and int(upstream.get("n_compared", -1)) == 450
                and float(upstream.get("max_abs", np.inf))
                <= float(upstream.get("max_abs_tolerance", -np.inf))
                and zero_audit.get("status") == "passed"
                and int(zero_audit.get("n_pairs", -1))
                == EXPECTED_PLAN_ROWS
                and metric_protocol.get("analysis_version")
                == ANALYSIS_VERSION
                and statistics.get("analysis_version")
                == ANALYSIS_VERSION
                and metric_protocol.get(
                    "analysis_code_identity_sha256"
                )
                == analysis_code_identity_sha256
                and statistics.get(
                    "analysis_code_identity_sha256"
                )
                == analysis_code_identity_sha256
                and metric_protocol.get(
                    "simulation_protocol_version"
                )
                == PROTOCOL_VERSION
                and statistics.get(
                    "simulation_protocol_version"
                )
                == PROTOCOL_VERSION
                and metric_protocol.get("plan_sha256")
                == protocol["plan_sha256"]
                and statistics.get("plan_sha256")
                == protocol["plan_sha256"]
                and metric_protocol.get(
                    "simulation_code_bundle_sha256"
                )
                == protocol["simulation_code_bundle_sha256"]
                and statistics.get(
                    "simulation_code_bundle_sha256"
                )
                == protocol["simulation_code_bundle_sha256"]
                and metric_protocol.get("model_identity_sha256")
                == model_fingerprint.get("identity_sha256")
                and statistics.get("model_identity_sha256")
                == model_fingerprint.get("identity_sha256")
                and metric_protocol.get("metric_input_sha256")
                == statistics.get("metric_input_sha256")
                == metric_artifact.get("metric_input_sha256")
                and metric_protocol.get("metric_input_sha256")
                == _identity_sha256(
                    metric_protocol.get("metric_inputs", {})
                )
                and metric_summary.get("status") == "complete"
                and metric_summary.get("analysis_version")
                == ANALYSIS_VERSION
                and metric_summary.get("plan_sha256")
                == protocol["plan_sha256"]
                and metric_summary.get(
                    "metric_artifact_manifest_sha256"
                )
                == _sha256_file(
                    output_root / "metric_artifact_manifest.json"
                )
                and metric_summary.get(
                    "metric_artifact_identity_sha256"
                )
                == metric_artifact.get(
                    "manifest_identity_sha256"
                )
                and int(metric_summary.get("n_runs", -1)) == RUN_COUNT
                and len(tests) == SUMMARY_METRIC_COUNT
                and set(tests)
                == set(
                    pd.read_csv(
                        output_root / "c5_statistical_table.csv"
                    )["metric"].astype(str)
                )
                and all(
                    int(test.get("n_runs", -1)) == RUN_COUNT
                    for test in tests.values()
                )
                and statistics.get(
                    "candidate_label_randomization", {}
                ).get("reported")
                is False
                and int(figure_summary.get("n_figure_stems", -1))
                == len(REQUIRED_FIGURE_STEMS)
                and figure_summary.get("status") == "complete"
                and int(video_summary.get("n_videos", -1))
                == CANDIDATE_COUNT
                and video_summary.get("status") == "complete"
                and video_summary.get("analysis_version")
                == ANALYSIS_VERSION
                and video_summary.get(
                    "analysis_code_identity_sha256"
                )
                == analysis_code_identity_sha256
                and video_summary.get("manifest_sha256")
                == _sha256_file(output_root / "video_manifest.csv")
            ),
            "model_identity_sha256": model_fingerprint.get(
                "identity_sha256"
            ),
            "upstream_c2_reference": upstream,
        }
    except Exception as exc:
        checks["metadata_integrity"] = {
            "passed": False,
            "error": repr(exc),
        }
    all_passed = all(bool(value["passed"]) for value in checks.values())
    audit = {
        "status": "passed" if all_passed else "incomplete",
        "analysis_version": ANALYSIS_VERSION,
        "simulation_protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "n_plan_rows": len(plan),
        "checks": checks,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    _write_json(output_root / "completion_audit.json", _json_clean(audit))
    if not all_passed:
        failed = [key for key, value in checks.items() if not value["passed"]]
        raise RuntimeError(f"Completion audit is incomplete: {failed}")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired free versus walls-then-free Flow-Lenia C5 branches."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("embeddings", "metrics", "plots", "videos", "audit", "all"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--c2-root", type=Path, default=DEFAULT_C2_ROOT)
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=1,
        help="Must be 1 to reproduce the authoritative C2 CLIP path.",
    )
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-hold-frames", type=int, default=6)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-videos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = _resolve(args.output_root)
    c2_root = _resolve(args.c2_root)
    plan, protocol = _load_inputs(output_root)
    manifest = None
    if args.phase in {"embeddings", "metrics", "all"}:
        manifest = ensure_embeddings(
            plan,
            protocol,
            output_root=output_root,
            c2_root=c2_root,
            batch_frames=int(args.batch_frames),
            force=bool(args.force_embeddings),
        )
    if args.phase in {"metrics", "all"}:
        assert manifest is not None
        compute_metrics(plan, protocol, manifest, output_root=output_root)
    if args.phase in {"plots", "all"}:
        build_plots(output_root)
    if args.phase in {"videos", "all"}:
        build_videos(
            plan,
            output_root=output_root,
            fps=int(args.video_fps),
            hold_frames=int(args.video_hold_frames),
            force=bool(args.force_videos),
        )
    if args.phase in {"audit", "all"}:
        completion_audit(plan, protocol, output_root=output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
