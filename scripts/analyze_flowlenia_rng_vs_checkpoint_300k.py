from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flowlenia_rng_vs_checkpoint_300k as experiment


DEFAULT_OUTPUT_ROOT = experiment.DEFAULT_OUTPUT_ROOT
CHECKPOINT_PAIRS = tuple(
    (left, right)
    for left in range(10)
    for right in range(left + 1, 10)
)
TARGET_HORIZONS = (10_000, 50_000, 100_000, 200_000, 300_000)
TARGET_FRACTIONS = (0.25, 0.50, 0.75, 0.90, 1.00)
CHAMFER_BIN_WIDTH = 20_000
CHAMFER_BIN_FRAME_INTERVAL = 1_000
ATTAINMENT_REFERENCE_START = 200_000
ATTAINMENT_BOOTSTRAP_REPLICATES = 5_000
ATTAINMENT_BOOTSTRAP_SEED = 6_103_003


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
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left64 /= np.clip(
        np.linalg.norm(left64, axis=-1, keepdims=True), 1.0e-12, None
    )
    right64 /= np.clip(
        np.linalg.norm(right64, axis=-1, keepdims=True), 1.0e-12, None
    )
    return np.clip(1.0 - np.sum(left64 * right64, axis=-1), 0.0, 2.0)


def _chamfer(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left64 /= np.clip(
        np.linalg.norm(left64, axis=-1, keepdims=True), 1.0e-12, None
    )
    right64 /= np.clip(
        np.linalg.norm(right64, axis=-1, keepdims=True), 1.0e-12, None
    )
    distance = np.clip(1.0 - left64 @ right64.T, 0.0, 2.0)
    return float(
        0.5
        * (
            np.mean(np.min(distance, axis=1))
            + np.mean(np.min(distance, axis=0))
        )
    )


def _first_reach(
    steps: np.ndarray, values: np.ndarray, target: np.ndarray | float
) -> int | None:
    target_values = (
        np.full(values.shape, float(target), dtype=np.float64)
        if np.ndim(target) == 0
        else np.asarray(target, dtype=np.float64)
    )
    hits = np.flatnonzero((steps > 0) & (values >= target_values))
    return int(steps[int(hits[0])]) if hits.size else None


def _read_candidates(output_root: Path) -> list[dict[str, str]]:
    with (output_root / "candidates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["run_idx"]))
    expected = [f"run_{idx:03d}_optimized" for idx in range(10)]
    if [row["candidate_id"] for row in rows] != expected:
        raise RuntimeError("Candidate manifest is not run_000..run_009")
    return rows


def _load_data(output_root: Path) -> dict[str, Any]:
    protocol = experiment.load_protocol(output_root)
    candidates = _read_candidates(output_root)
    arrays: list[np.ndarray] = []
    state_pair: list[np.ndarray] = []
    state_duplicate: list[np.ndarray] = []
    state_duplicate_max: list[np.ndarray] = []
    total_mass: list[np.ndarray] = []
    source_hashes: list[np.ndarray] = []
    source_full_hashes: list[np.ndarray] = []
    branch_keys: list[np.ndarray] = []
    elapsed: list[float] = []
    embed_steps: np.ndarray | None = None
    metric_steps: np.ndarray | None = None
    paths: list[Path] = []
    for candidate in candidates:
        path = experiment._candidate_output_path(
            output_root, candidate["candidate_id"], audit=False
        )
        if not experiment._validate_output(
            path,
            protocol=protocol,
            candidate=candidate,
            horizon=experiment.HORIZON_STEPS,
            audit=False,
        ):
            raise RuntimeError(f"Missing or invalid result: {path}")
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            local_embed_steps = np.asarray(
                data["embedding_steps"], dtype=np.int32
            )
            local_metric_steps = np.asarray(
                data["state_metric_steps"], dtype=np.int32
            )
            if embed_steps is None:
                embed_steps = local_embed_steps
                metric_steps = local_metric_steps
            if not np.array_equal(local_embed_steps, embed_steps):
                raise RuntimeError(f"Embedding grid mismatch: {path}")
            if not np.array_equal(local_metric_steps, metric_steps):
                raise RuntimeError(f"State metric grid mismatch: {path}")
            arrays.append(
                np.asarray(data["clip_embeddings"], dtype=np.float32)
            )
            state_pair.append(
                np.asarray(data["a_pair_relative_l1"], dtype=np.float64)
            )
            state_duplicate.append(
                np.asarray(data["a_duplicate_relative_l1"], dtype=np.float64)
            )
            state_duplicate_max.append(
                np.asarray(data["a_duplicate_max_abs"], dtype=np.float64)
            )
            total_mass.append(
                np.asarray(data["total_mass"], dtype=np.float64)
            )
            source_hashes.append(np.asarray(data["source_state_hashes"]))
            source_full_hashes.append(
                np.asarray(data["source_state_full_hashes"])
            )
            branch_keys.append(
                np.asarray(data["branch_rng_keys_initial"], dtype=np.uint32)
            )
            elapsed.append(float(np.asarray(data["elapsed_seconds"]).item()))
    if embed_steps is None or metric_steps is None:
        raise RuntimeError("No simulation data")
    return {
        "protocol": protocol,
        "candidates": candidates,
        "paths": paths,
        "z": np.stack(arrays, axis=0),
        "embed_steps": embed_steps,
        "state_pair": np.stack(state_pair, axis=0),
        "state_duplicate": np.stack(state_duplicate, axis=0),
        "state_duplicate_max": np.stack(state_duplicate_max, axis=0),
        "total_mass": np.stack(total_mass, axis=0),
        "metric_steps": metric_steps,
        "source_hashes": np.stack(source_hashes, axis=0),
        "source_full_hashes": np.stack(source_full_hashes, axis=0),
        "branch_keys": np.stack(branch_keys, axis=0),
        "elapsed": np.asarray(elapsed, dtype=np.float64),
    }


def _pointwise_distances(z: np.ndarray) -> dict[str, np.ndarray]:
    within = np.stack(
        [
            _cosine(
                z[:, :, left],
                z[:, :, right],
            )
            for left, right in zip(
                experiment.PAIR_LEFT, experiment.PAIR_RIGHT
            )
        ],
        axis=2,
    )
    harness = _cosine(
        z[:, :, experiment.DUPLICATE_OF_BRANCH],
        z[:, :, experiment.DUPLICATE_BRANCH],
    )
    between_matched = np.stack(
        [
            _cosine(
                z[left, :, : experiment.N_UNIQUE_BRANCHES],
                z[right, :, : experiment.N_UNIQUE_BRANCHES],
            )
            for left, right in CHECKPOINT_PAIRS
        ],
        axis=0,
    )
    between_all = np.stack(
        [
            _cosine(
                z[left, :, : experiment.N_UNIQUE_BRANCHES, None],
                z[right, :, None, : experiment.N_UNIQUE_BRANCHES],
            )
            for left, right in CHECKPOINT_PAIRS
        ],
        axis=0,
    )
    return {
        "within": within,
        "harness": harness,
        "between_matched": between_matched,
        "between_all": between_all,
    }


def _trajectory_distances(
    z: np.ndarray, embed_steps: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in experiment.CHAMFER_HORIZONS:
        offsets = experiment.CHAMFER_OFFSETS[horizon]
        indices = np.searchsorted(embed_steps, np.asarray(offsets))
        if not np.array_equal(embed_steps[indices], np.asarray(offsets)):
            raise RuntimeError(f"Missing Chamfer offsets for horizon {horizon}")

        within_values = []
        harness_values = []
        between_values = []
        for candidate_idx in range(10):
            for context_idx in range(experiment.N_CONTEXTS):
                for left, right in zip(
                    experiment.PAIR_LEFT, experiment.PAIR_RIGHT
                ):
                    within_values.append(
                        _chamfer(
                            z[candidate_idx, context_idx, left, indices],
                            z[candidate_idx, context_idx, right, indices],
                        )
                    )
                harness_values.append(
                    _chamfer(
                        z[
                            candidate_idx,
                            context_idx,
                            experiment.DUPLICATE_OF_BRANCH,
                            indices,
                        ],
                        z[
                            candidate_idx,
                            context_idx,
                            experiment.DUPLICATE_BRANCH,
                            indices,
                        ],
                    )
                )
        for left_checkpoint, right_checkpoint in CHECKPOINT_PAIRS:
            for context_idx in range(experiment.N_CONTEXTS):
                for branch_idx in range(experiment.N_UNIQUE_BRANCHES):
                    between_values.append(
                        _chamfer(
                            z[
                                left_checkpoint,
                                context_idx,
                                branch_idx,
                                indices,
                            ],
                            z[
                                right_checkpoint,
                                context_idx,
                                branch_idx,
                                indices,
                            ],
                        )
                    )
        within_array = np.asarray(within_values)
        between_array = np.asarray(between_values)
        harness_array = np.asarray(harness_values)
        rows.append(
            {
                "horizon_steps": horizon,
                "within_median": float(np.median(within_array)),
                "within_mean": float(np.mean(within_array)),
                "within_q25": float(np.quantile(within_array, 0.25)),
                "within_q75": float(np.quantile(within_array, 0.75)),
                "between_matched_median": float(np.median(between_array)),
                "between_matched_mean": float(np.mean(between_array)),
                "between_matched_q25": float(
                    np.quantile(between_array, 0.25)
                ),
                "between_matched_q75": float(
                    np.quantile(between_array, 0.75)
                ),
                "harness_median": float(np.median(harness_array)),
                "harness_max": float(np.max(harness_array)),
                "median_ratio": float(
                    np.median(within_array)
                    / max(np.median(between_array), 1.0e-15)
                ),
            }
        )
    return rows


def _binned_chamfer_distances(
    z: np.ndarray, embed_steps: np.ndarray
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    normalized = np.asarray(z, dtype=np.float32)
    normalized /= np.clip(
        np.linalg.norm(normalized, axis=-1, keepdims=True), 1.0e-12, None
    )
    within_candidate_rows: list[dict[str, Any]] = []
    between_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    within_by_bin: list[np.ndarray] = []
    between_by_bin: list[np.ndarray] = []

    for bin_start in range(0, experiment.HORIZON_STEPS, CHAMFER_BIN_WIDTH):
        bin_end = bin_start + CHAMFER_BIN_WIDTH
        frame_steps = np.arange(
            bin_start + CHAMFER_BIN_FRAME_INTERVAL,
            bin_end + 1,
            CHAMFER_BIN_FRAME_INTERVAL,
            dtype=np.int32,
        )
        frame_indices = np.searchsorted(embed_steps, frame_steps)
        if not np.array_equal(embed_steps[frame_indices], frame_steps):
            raise RuntimeError(
                f"Missing 1k CLIP frames for bin ({bin_start}, {bin_end}]"
            )

        left = normalized[
            :, :, experiment.PAIR_LEFT
        ][:, :, :, frame_indices]
        right = normalized[
            :, :, experiment.PAIR_RIGHT
        ][:, :, :, frame_indices]
        within_distance = np.clip(
            1.0
            - np.einsum(
                "acpfd,acpgd->acpfg",
                left,
                right,
                optimize=True,
            ),
            0.0,
            2.0,
        )
        within = 0.5 * (
            np.mean(np.min(within_distance, axis=-1), axis=-1)
            + np.mean(np.min(within_distance, axis=-2), axis=-1)
        )

        between_left = np.stack(
            [
                normalized[
                    checkpoint_left,
                    :,
                    : experiment.N_UNIQUE_BRANCHES,
                ][:, :, frame_indices]
                for checkpoint_left, _checkpoint_right in CHECKPOINT_PAIRS
            ],
            axis=0,
        )
        between_right = np.stack(
            [
                normalized[
                    checkpoint_right,
                    :,
                    : experiment.N_UNIQUE_BRANCHES,
                ][:, :, frame_indices]
                for _checkpoint_left, checkpoint_right in CHECKPOINT_PAIRS
            ],
            axis=0,
        )
        between_distance = np.clip(
            1.0
            - np.einsum(
                "pcbfd,pcbgd->pcbfg",
                between_left,
                between_right,
                optimize=True,
            ),
            0.0,
            2.0,
        )
        between = 0.5 * (
            np.mean(np.min(between_distance, axis=-1), axis=-1)
            + np.mean(np.min(between_distance, axis=-2), axis=-1)
        )
        within_by_bin.append(np.asarray(within, dtype=np.float64))
        between_by_bin.append(np.asarray(between, dtype=np.float64))

        harness_left = normalized[
            :, :, experiment.DUPLICATE_OF_BRANCH
        ][:, :, frame_indices]
        harness_right = normalized[
            :, :, experiment.DUPLICATE_BRANCH
        ][:, :, frame_indices]
        harness_distance = np.clip(
            1.0
            - np.einsum(
                "acfd,acgd->acfg",
                harness_left,
                harness_right,
                optimize=True,
            ),
            0.0,
            2.0,
        )
        harness = 0.5 * (
            np.mean(np.min(harness_distance, axis=-1), axis=-1)
            + np.mean(np.min(harness_distance, axis=-2), axis=-1)
        )

        summary_rows.append(
            {
                "bin_start_exclusive": bin_start,
                "bin_end_inclusive": bin_end,
                "bin_center": 0.5 * (bin_start + bin_end),
                "frame_interval": CHAMFER_BIN_FRAME_INTERVAL,
                "frames_per_trajectory_segment": len(frame_steps),
                "within_rng_median": float(np.median(within)),
                "within_rng_mean": float(np.mean(within)),
                "within_rng_q25": float(np.quantile(within, 0.25)),
                "within_rng_q75": float(np.quantile(within, 0.75)),
                "between_checkpoint_matched_median": float(
                    np.median(between)
                ),
                "between_checkpoint_matched_mean": float(np.mean(between)),
                "between_checkpoint_matched_q25": float(
                    np.quantile(between, 0.25)
                ),
                "between_checkpoint_matched_q75": float(
                    np.quantile(between, 0.75)
                ),
                "median_ratio_within_over_between": float(
                    np.median(within) / max(np.median(between), 1.0e-15)
                ),
                "mean_ratio_within_over_between": float(
                    np.mean(within) / max(np.mean(between), 1.0e-15)
                ),
                "duplicate_harness_median": float(np.median(harness)),
                "duplicate_harness_max": float(np.max(harness)),
            }
        )
        for candidate_idx in range(10):
            values = within[candidate_idx]
            within_candidate_rows.append(
                {
                    "candidate_id": f"run_{candidate_idx:03d}_optimized",
                    "run_idx": candidate_idx,
                    "bin_start_exclusive": bin_start,
                    "bin_end_inclusive": bin_end,
                    "chamfer_median": float(np.median(values)),
                    "chamfer_mean": float(np.mean(values)),
                    "chamfer_q25": float(np.quantile(values, 0.25)),
                    "chamfer_q75": float(np.quantile(values, 0.75)),
                }
            )
        for pair_idx, (checkpoint_left, checkpoint_right) in enumerate(
            CHECKPOINT_PAIRS
        ):
            values = between[pair_idx]
            between_pair_rows.append(
                {
                    "left_candidate_id": (
                        f"run_{checkpoint_left:03d}_optimized"
                    ),
                    "right_candidate_id": (
                        f"run_{checkpoint_right:03d}_optimized"
                    ),
                    "left_run_idx": checkpoint_left,
                    "right_run_idx": checkpoint_right,
                    "bin_start_exclusive": bin_start,
                    "bin_end_inclusive": bin_end,
                    "chamfer_median": float(np.median(values)),
                    "chamfer_mean": float(np.mean(values)),
                    "chamfer_q25": float(np.quantile(values, 0.25)),
                    "chamfer_q75": float(np.quantile(values, 0.75)),
                }
            )
    return (
        summary_rows,
        within_candidate_rows,
        between_pair_rows,
        np.stack(within_by_bin, axis=-1),
        np.stack(between_by_bin, axis=-1),
    )


def _repeat_audit(
    output_root: Path,
    protocol: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    main_path = experiment._candidate_output_path(
        output_root, candidate["candidate_id"], audit=False
    )
    repeat_path = experiment._candidate_output_path(
        output_root, candidate["candidate_id"], audit=True
    )
    if not experiment._validate_output(
        repeat_path,
        protocol=protocol,
        candidate=candidate,
        horizon=experiment.HORIZON_STEPS,
        audit=True,
    ):
        raise RuntimeError(f"Missing or invalid repeat audit: {repeat_path}")
    keys = (
        "embedding_steps",
        "clip_embeddings",
        "state_metric_steps",
        "a_pair_relative_l1",
        "a_duplicate_relative_l1",
        "a_duplicate_max_abs",
        "total_mass",
        "duplicate_rng_max_abs",
        "source_state_hashes",
        "source_state_full_hashes",
        "branch_rng_keys_initial",
    )
    comparisons: dict[str, Any] = {}
    with (
        np.load(main_path, allow_pickle=False) as main,
        np.load(repeat_path, allow_pickle=False) as repeat,
    ):
        for key in keys:
            left = np.asarray(main[key])
            right = np.asarray(repeat[key])
            exact = np.array_equal(left, right)
            comparisons[key] = {
                "exact": exact,
                "max_abs": (
                    float(np.max(np.abs(left.astype(np.float64) - right)))
                    if np.issubdtype(left.dtype, np.number)
                    else None
                ),
            }

    main_restart = experiment._candidate_checkpoint_path(
        output_root, candidate["candidate_id"], audit=False
    )
    repeat_restart = experiment._candidate_checkpoint_path(
        output_root, candidate["candidate_id"], audit=True
    )
    with (
        np.load(main_restart, allow_pickle=False) as main,
        np.load(repeat_restart, allow_pickle=False) as repeat,
    ):
        state_keys = [str(key) for key in np.asarray(main["state_keys"])]
        restart_exact = np.array_equal(main["rng"], repeat["rng"])
        state_comparisons = {}
        for key in state_keys:
            left = np.asarray(main[f"state__{key}"])
            right = np.asarray(repeat[f"state__{key}"])
            exact = np.array_equal(left, right)
            restart_exact = restart_exact and exact
            state_comparisons[key] = {
                "exact": exact,
                "max_abs": float(
                    np.max(np.abs(left.astype(np.complex128) - right))
                ),
            }
    report = {
        "status": (
            "passed"
            if all(item["exact"] for item in comparisons.values())
            and restart_exact
            else "failed"
        ),
        "candidate_id": candidate["candidate_id"],
        "output_comparisons": comparisons,
        "final_restart_state_exact": restart_exact,
        "final_state_comparisons": state_comparisons,
        "main_sha256": _sha256_file(main_path),
        "repeat_sha256": _sha256_file(repeat_path),
    }
    return report


def _make_pointwise_rows(
    steps: np.ndarray, distances: dict[str, np.ndarray]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    within = distances["within"]
    harness = distances["harness"]
    between = distances["between_matched"]
    between_all = distances["between_all"]
    summary_rows = []
    candidate_rows = []
    pair_rows = []
    harness_rows = []
    for step_idx, step in enumerate(steps):
        within_values = within[..., step_idx]
        between_values = between[..., step_idx]
        between_all_values = between_all[..., step_idx]
        harness_values = harness[..., step_idx]
        w_median = float(np.median(within_values))
        b_median = float(np.median(between_values))
        h_median = float(np.median(harness_values))
        summary_rows.append(
            {
                "step": int(step),
                "within_rng_median": w_median,
                "within_rng_mean": float(np.mean(within_values)),
                "within_rng_q25": float(np.quantile(within_values, 0.25)),
                "within_rng_q75": float(np.quantile(within_values, 0.75)),
                "between_checkpoint_matched_median": b_median,
                "between_checkpoint_matched_mean": float(
                    np.mean(between_values)
                ),
                "between_checkpoint_matched_q25": float(
                    np.quantile(between_values, 0.25)
                ),
                "between_checkpoint_matched_q75": float(
                    np.quantile(between_values, 0.75)
                ),
                "between_checkpoint_all_rng_median": float(
                    np.median(between_all_values)
                ),
                "between_checkpoint_all_rng_mean": float(
                    np.mean(between_all_values)
                ),
                "harness_median": h_median,
                "harness_mean": float(np.mean(harness_values)),
                "harness_max": float(np.max(harness_values)),
                "median_ratio_within_over_between": (
                    w_median / b_median if b_median > 0 else float("nan")
                ),
                "mean_ratio_within_over_between": (
                    float(np.mean(within_values))
                    / float(np.mean(between_values))
                    if float(np.mean(between_values)) > 0
                    else float("nan")
                ),
                "harness_over_within_median": (
                    h_median / w_median if w_median > 0 else float("nan")
                ),
            }
        )
        for candidate_idx in range(10):
            values = within[candidate_idx, ..., step_idx]
            candidate_rows.append(
                {
                    "candidate_id": f"run_{candidate_idx:03d}_optimized",
                    "run_idx": candidate_idx,
                    "step": int(step),
                    "within_rng_median": float(np.median(values)),
                    "within_rng_mean": float(np.mean(values)),
                    "within_rng_q25": float(np.quantile(values, 0.25)),
                    "within_rng_q75": float(np.quantile(values, 0.75)),
                }
            )
            h_values = harness[candidate_idx, ..., step_idx]
            harness_rows.append(
                {
                    "candidate_id": f"run_{candidate_idx:03d}_optimized",
                    "run_idx": candidate_idx,
                    "step": int(step),
                    "harness_median": float(np.median(h_values)),
                    "harness_mean": float(np.mean(h_values)),
                    "harness_max": float(np.max(h_values)),
                }
            )
        for pair_idx, (left, right) in enumerate(CHECKPOINT_PAIRS):
            values = between[pair_idx, ..., step_idx]
            pair_rows.append(
                {
                    "left_candidate_id": f"run_{left:03d}_optimized",
                    "right_candidate_id": f"run_{right:03d}_optimized",
                    "left_run_idx": left,
                    "right_run_idx": right,
                    "step": int(step),
                    "between_matched_median": float(np.median(values)),
                    "between_matched_mean": float(np.mean(values)),
                    "between_matched_q25": float(
                        np.quantile(values, 0.25)
                    ),
                    "between_matched_q75": float(
                        np.quantile(values, 0.75)
                    ),
                }
            )
    return summary_rows, candidate_rows, pair_rows, harness_rows


def _crossing_rows(
    steps: np.ndarray, summary_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    within_median = np.asarray(
        [row["within_rng_median"] for row in summary_rows], dtype=np.float64
    )
    within_mean = np.asarray(
        [row["within_rng_mean"] for row in summary_rows], dtype=np.float64
    )
    between_median = np.asarray(
        [row["between_checkpoint_matched_median"] for row in summary_rows],
        dtype=np.float64,
    )
    between_mean = np.asarray(
        [row["between_checkpoint_matched_mean"] for row in summary_rows],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for target_horizon in TARGET_HORIZONS:
        target_idx = int(np.flatnonzero(steps == target_horizon)[0])
        for aggregation, within, between in (
            ("median", within_median, between_median),
            ("mean", within_mean, between_mean),
        ):
            target_value = float(between[target_idx])
            for fraction in TARGET_FRACTIONS:
                threshold = target_value * fraction
                crossing = _first_reach(steps, within, threshold)
                rows.append(
                    {
                        "aggregation": aggregation,
                        "target_horizon_steps": target_horizon,
                        "target_between_checkpoint_distance": target_value,
                        "target_fraction": fraction,
                        "threshold_distance": threshold,
                        "first_reach_step": (
                            crossing if crossing is not None else ""
                        ),
                        "reached_by_300k": crossing is not None,
                    }
                )
    dynamic_median = _first_reach(steps, within_median, between_median)
    dynamic_mean = _first_reach(steps, within_mean, between_mean)
    report = {
        "dynamic_same_time_crossing_median_step": dynamic_median,
        "dynamic_same_time_crossing_mean_step": dynamic_mean,
        "median_ratio_at_10k": float(
            within_median[np.flatnonzero(steps == 10_000)[0]]
            / between_median[np.flatnonzero(steps == 10_000)[0]]
        ),
        "median_ratio_at_50k": float(
            within_median[np.flatnonzero(steps == 50_000)[0]]
            / between_median[np.flatnonzero(steps == 50_000)[0]]
        ),
        "median_ratio_at_100k": float(
            within_median[np.flatnonzero(steps == 100_000)[0]]
            / between_median[np.flatnonzero(steps == 100_000)[0]]
        ),
        "median_ratio_at_200k": float(
            within_median[np.flatnonzero(steps == 200_000)[0]]
            / between_median[np.flatnonzero(steps == 200_000)[0]]
        ),
        "median_ratio_at_300k": float(
            within_median[np.flatnonzero(steps == 300_000)[0]]
            / between_median[np.flatnonzero(steps == 300_000)[0]]
        ),
    }
    return rows, report


def _plot_pointwise(
    figure_dir: Path,
    steps: np.ndarray,
    summary_rows: list[dict[str, Any]],
) -> None:
    within = np.asarray([row["within_rng_median"] for row in summary_rows])
    within_q25 = np.asarray([row["within_rng_q25"] for row in summary_rows])
    within_q75 = np.asarray([row["within_rng_q75"] for row in summary_rows])
    between = np.asarray(
        [row["between_checkpoint_matched_median"] for row in summary_rows]
    )
    between_q25 = np.asarray(
        [row["between_checkpoint_matched_q25"] for row in summary_rows]
    )
    between_q75 = np.asarray(
        [row["between_checkpoint_matched_q75"] for row in summary_rows]
    )
    harness = np.asarray([row["harness_median"] for row in summary_rows])
    harness_max = np.asarray([row["harness_max"] for row in summary_rows])
    ratio = within / np.clip(between, 1.0e-15, None)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 7.2), sharex=True, constrained_layout=True
    )
    ax = axes[0]
    ax.fill_between(
        steps,
        within_q25,
        within_q75,
        color="#147D92",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(steps, within, color="#147D92", lw=2.2, label="Within checkpoint: RNG")
    ax.fill_between(
        steps,
        between_q25,
        between_q75,
        color="#D1495B",
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        steps,
        between,
        color="#D1495B",
        lw=2.2,
        label="Between optimized checkpoints",
    )
    ax.set_ylim(
        0,
        1.08
        * max(float(np.max(within_q75)), float(np.max(between_q75))),
    )
    ax.set_ylabel("CLIP cosine distance")
    ax.set_title("Flow-Lenia divergence from exact shared initial states")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    ax.text(
        0.99,
        0.04,
        f"duplicate harness max = {np.max(harness_max):.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    ax = axes[1]
    ax.plot(steps, ratio, color="#147D92", lw=2.2)
    ax.axhline(1.0, color="#D1495B", lw=1.4, ls="--")
    for fraction in (0.25, 0.50, 0.75):
        ax.axhline(fraction, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Simulation steps from the common initial state")
    ax.set_ylabel("Within-RNG / between-checkpoint")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"flowlenia_rng_vs_checkpoint_curve.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)

    early = steps <= 10_000
    fig, axes = plt.subplots(
        2, 1, figsize=(7.8, 6.4), sharex=True, constrained_layout=True
    )
    ax = axes[0]
    ax.fill_between(
        steps[early],
        within_q25[early],
        within_q75[early],
        color="#147D92",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        steps[early],
        within[early],
        color="#147D92",
        lw=2.2,
        label="Within checkpoint: RNG",
    )
    ax.fill_between(
        steps[early],
        between_q25[early],
        between_q75[early],
        color="#D1495B",
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        steps[early],
        between[early],
        color="#D1495B",
        lw=2.2,
        label="Between optimized checkpoints",
    )
    ax.set_ylim(
        0,
        1.08
        * max(
            float(np.max(within_q75[early])),
            float(np.max(between_q75[early])),
        ),
    )
    ax.set_ylabel("CLIP cosine distance")
    ax.set_title("Early-time divergence from exact shared initial states")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(steps[early], ratio[early], color="#147D92", lw=2.2)
    ax.axhline(1.0, color="#D1495B", lw=1.4, ls="--")
    ax.axvline(100, color="#555555", lw=1.0, ls=":")
    ax.scatter(
        [100],
        [ratio[np.flatnonzero(steps == 100)[0]]],
        color="#147D92",
        edgecolor="white",
        linewidth=0.7,
        s=36,
        zorder=4,
    )
    ax.text(
        100,
        1.03,
        "first crossing: 100 steps",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.set_xlabel("Simulation steps from the common initial state")
    ax.set_ylabel("Within-RNG / between-checkpoint")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_vs_checkpoint_curve_early_zoom.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_per_checkpoint(
    figure_dir: Path,
    steps: np.ndarray,
    distances: dict[str, np.ndarray],
) -> None:
    within = np.median(distances["within"], axis=(1, 2))
    between = np.median(distances["between_matched"], axis=(0, 1, 2))
    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx in range(10):
        ax.plot(
            steps,
            within[idx],
            color=colors[idx],
            lw=1.25,
            alpha=0.9,
            label=f"opt_{idx:03d}",
        )
    ax.plot(
        steps,
        between,
        color="black",
        lw=2.3,
        ls="--",
        label="between-checkpoint median",
    )
    ax.set_xlabel("Simulation steps from the common initial state")
    ax.set_ylabel("CLIP cosine distance")
    ax.set_title("RNG divergence per optimized checkpoint")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_vs_checkpoint_per_optimized_run.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_chamfer(
    figure_dir: Path, rows: list[dict[str, Any]]
) -> None:
    horizons = np.asarray([row["horizon_steps"] for row in rows])
    within = np.asarray([row["within_median"] for row in rows])
    between = np.asarray([row["between_matched_median"] for row in rows])
    harness = np.asarray([row["harness_median"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ax.plot(
        horizons,
        within,
        marker="o",
        ms=3,
        color="#147D92",
        label="Within checkpoint: RNG",
    )
    ax.plot(
        horizons,
        between,
        marker="o",
        ms=3,
        color="#D1495B",
        label="Between optimized checkpoints",
    )
    ax.plot(
        horizons,
        harness,
        color="#444444",
        lw=1.3,
        label="Duplicate harness",
    )
    ax.set_xlabel("Trajectory horizon (steps)")
    ax.set_ylabel("8-frame CLIP-Chamfer")
    ax.set_title("Secondary trajectory-level divergence")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"flowlenia_rng_vs_checkpoint_chamfer.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_binned_chamfer(
    figure_dir: Path, rows: list[dict[str, Any]]
) -> None:
    centers = np.asarray([row["bin_center"] for row in rows])
    within = np.asarray([row["within_rng_median"] for row in rows])
    within_q25 = np.asarray([row["within_rng_q25"] for row in rows])
    within_q75 = np.asarray([row["within_rng_q75"] for row in rows])
    between = np.asarray(
        [row["between_checkpoint_matched_median"] for row in rows]
    )
    between_q25 = np.asarray(
        [row["between_checkpoint_matched_q25"] for row in rows]
    )
    between_q75 = np.asarray(
        [row["between_checkpoint_matched_q75"] for row in rows]
    )
    harness_max = max(float(row["duplicate_harness_max"]) for row in rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    ax.fill_between(
        centers,
        within_q25,
        within_q75,
        color="#147D92",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        centers,
        within,
        color="#147D92",
        marker="o",
        ms=4,
        lw=2.2,
        label="Within checkpoint: RNG",
    )
    ax.fill_between(
        centers,
        between_q25,
        between_q75,
        color="#D1495B",
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        centers,
        between,
        color="#D1495B",
        marker="o",
        ms=4,
        lw=2.2,
        label="Between optimized checkpoints",
    )
    ax.set_xlim(0, experiment.HORIZON_STEPS)
    ax.set_ylim(
        0,
        1.08
        * max(float(np.max(within_q75)), float(np.max(between_q75))),
    )
    ax.set_xlabel("Simulation step (20k-window center)")
    ax.set_ylabel("20-frame CLIP-Chamfer")
    ax.set_title("Local trajectory divergence in non-overlapping 20k windows")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.04,
        f"duplicate harness max = {harness_max:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_vs_checkpoint_chamfer_20k_bins.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _anchor_pairwise_rows(
    within_rows: list[dict[str, Any]],
    between_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    within_lookup = {
        (int(row["run_idx"]), int(row["bin_end_inclusive"])): row
        for row in within_rows
    }
    between_lookup = {
        (
            int(row["left_run_idx"]),
            int(row["right_run_idx"]),
            int(row["bin_end_inclusive"]),
        ): row
        for row in between_rows
    }
    pairwise: list[dict[str, Any]] = []
    for anchor_idx in range(10):
        for other_idx in range(10):
            if anchor_idx == other_idx:
                continue
            left_idx, right_idx = sorted((anchor_idx, other_idx))
            for bin_end in range(
                CHAMFER_BIN_WIDTH,
                experiment.HORIZON_STEPS + 1,
                CHAMFER_BIN_WIDTH,
            ):
                within = within_lookup[(anchor_idx, bin_end)]
                between = between_lookup[(left_idx, right_idx, bin_end)]
                within_median = float(within["chamfer_median"])
                between_median = float(between["chamfer_median"])
                within_mean = float(within["chamfer_mean"])
                between_mean = float(between["chamfer_mean"])
                pairwise.append(
                    {
                        "anchor_candidate_id": (
                            f"run_{anchor_idx:03d}_optimized"
                        ),
                        "other_candidate_id": (
                            f"run_{other_idx:03d}_optimized"
                        ),
                        "anchor_run_idx": anchor_idx,
                        "other_run_idx": other_idx,
                        "bin_start_exclusive": bin_end - CHAMFER_BIN_WIDTH,
                        "bin_end_inclusive": bin_end,
                        "bin_center": bin_end - 0.5 * CHAMFER_BIN_WIDTH,
                        "within_anchor_rng_median": within_median,
                        "within_anchor_rng_mean": within_mean,
                        "within_anchor_rng_q25": float(
                            within["chamfer_q25"]
                        ),
                        "within_anchor_rng_q75": float(
                            within["chamfer_q75"]
                        ),
                        "between_anchor_other_median": between_median,
                        "between_anchor_other_mean": between_mean,
                        "between_anchor_other_q25": float(
                            between["chamfer_q25"]
                        ),
                        "between_anchor_other_q75": float(
                            between["chamfer_q75"]
                        ),
                        "median_ratio_within_over_between": (
                            within_median / max(between_median, 1.0e-15)
                        ),
                        "mean_ratio_within_over_between": (
                            within_mean / max(between_mean, 1.0e-15)
                        ),
                        "within_exceeds_between_median": (
                            within_median > between_median
                        ),
                    }
                )

    summary: list[dict[str, Any]] = []
    for anchor_idx in range(10):
        for other_idx in range(10):
            if anchor_idx == other_idx:
                continue
            values = [
                row
                for row in pairwise
                if int(row["anchor_run_idx"]) == anchor_idx
                and int(row["other_run_idx"]) == other_idx
            ]
            median_ratios = np.asarray(
                [
                    float(row["median_ratio_within_over_between"])
                    for row in values
                ],
                dtype=np.float64,
            )
            mean_ratios = np.asarray(
                [
                    float(row["mean_ratio_within_over_between"])
                    for row in values
                ],
                dtype=np.float64,
            )
            summary.append(
                {
                    "anchor_candidate_id": (
                        f"run_{anchor_idx:03d}_optimized"
                    ),
                    "other_candidate_id": (
                        f"run_{other_idx:03d}_optimized"
                    ),
                    "anchor_run_idx": anchor_idx,
                    "other_run_idx": other_idx,
                    "n_20k_bins": len(values),
                    "median_of_binwise_median_ratios": float(
                        np.median(median_ratios)
                    ),
                    "mean_of_binwise_median_ratios": float(
                        np.mean(median_ratios)
                    ),
                    "median_of_binwise_mean_ratios": float(
                        np.median(mean_ratios)
                    ),
                    "fraction_bins_within_exceeds_between": float(
                        np.mean(median_ratios > 1.0)
                    ),
                    "min_binwise_median_ratio": float(
                        np.min(median_ratios)
                    ),
                    "max_binwise_median_ratio": float(
                        np.max(median_ratios)
                    ),
                }
            )
    return pairwise, summary


def _run_matched_attainment(
    within_by_bin: np.ndarray,
    between_by_bin: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    expected_bins = experiment.HORIZON_STEPS // CHAMFER_BIN_WIDTH
    expected_within_shape = (
        10,
        experiment.N_CONTEXTS,
        len(experiment.PAIR_LEFT),
        expected_bins,
    )
    expected_between_shape = (
        len(CHECKPOINT_PAIRS),
        experiment.N_CONTEXTS,
        experiment.N_UNIQUE_BRANCHES,
        expected_bins,
    )
    if within_by_bin.shape != expected_within_shape:
        raise RuntimeError(
            "Unexpected within-run Chamfer shape: "
            f"{within_by_bin.shape} != {expected_within_shape}"
        )
    if between_by_bin.shape != expected_between_shape:
        raise RuntimeError(
            "Unexpected between-run Chamfer shape: "
            f"{between_by_bin.shape} != {expected_between_shape}"
        )

    bin_ends = np.arange(
        CHAMFER_BIN_WIDTH,
        experiment.HORIZON_STEPS + 1,
        CHAMFER_BIN_WIDTH,
        dtype=np.int32,
    )
    reference_bins = np.flatnonzero(
        bin_ends - CHAMFER_BIN_WIDTH >= ATTAINMENT_REFERENCE_START
    )
    if not np.array_equal(
        bin_ends[reference_bins],
        np.arange(
            ATTAINMENT_REFERENCE_START + CHAMFER_BIN_WIDTH,
            experiment.HORIZON_STEPS + 1,
            CHAMFER_BIN_WIDTH,
            dtype=np.int32,
        ),
    ):
        raise RuntimeError("Unexpected attainment reference-bin grid")

    between_anchor = np.full(
        (
            10,
            10,
            experiment.N_CONTEXTS,
            experiment.N_UNIQUE_BRANCHES,
            expected_bins,
        ),
        np.nan,
        dtype=np.float64,
    )
    for pair_idx, (left, right) in enumerate(CHECKPOINT_PAIRS):
        between_anchor[left, right] = between_by_bin[pair_idx]
        between_anchor[right, left] = between_by_bin[pair_idx]

    within_mean = np.mean(within_by_bin, axis=(1, 2))
    reference = np.empty(10, dtype=np.float64)
    for anchor_idx in range(10):
        other_indices = [idx for idx in range(10) if idx != anchor_idx]
        reference[anchor_idx] = float(
            np.mean(
                between_anchor[
                    anchor_idx,
                    other_indices,
                    :,
                    :,
                    :,
                ][..., reference_bins]
            )
        )
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0):
        raise RuntimeError(f"Invalid run-matched reference values: {reference}")

    ratios = within_mean / reference[:, None]
    curve_steps = np.concatenate(([0], bin_ends))
    curves = np.concatenate((np.zeros((10, 1)), ratios), axis=1)

    rng = np.random.default_rng(ATTAINMENT_BOOTSTRAP_SEED)
    bootstrap = np.zeros(
        (ATTAINMENT_BOOTSTRAP_REPLICATES, len(curve_steps)),
        dtype=np.float64,
    )
    context_alpha = np.ones(experiment.N_CONTEXTS, dtype=np.float64)
    branch_alpha = np.ones(
        experiment.N_UNIQUE_BRANCHES, dtype=np.float64
    )
    other_alpha = np.ones(9, dtype=np.float64)
    late_weights = np.full(
        len(reference_bins), 1.0 / len(reference_bins), dtype=np.float64
    )
    for bootstrap_idx in range(ATTAINMENT_BOOTSTRAP_REPLICATES):
        sampled_anchors = rng.integers(0, 10, size=10)
        sampled_curves = []
        for anchor_idx in sampled_anchors:
            context_weights = rng.dirichlet(context_alpha)
            branch_weights = rng.dirichlet(branch_alpha)
            pair_weights = (
                branch_weights[experiment.PAIR_LEFT]
                * branch_weights[experiment.PAIR_RIGHT]
            )
            pair_weights /= np.sum(pair_weights)
            local_within = np.einsum(
                "c,p,cpb->b",
                context_weights,
                pair_weights,
                within_by_bin[anchor_idx],
                optimize=True,
            )

            other_indices = np.asarray(
                [idx for idx in range(10) if idx != anchor_idx],
                dtype=np.int32,
            )
            other_weights = rng.dirichlet(other_alpha)
            local_between = between_anchor[
                anchor_idx,
                other_indices,
                :,
                :,
                :,
            ][..., reference_bins]
            local_reference = float(
                np.einsum(
                    "o,c,r,l,ocrl->",
                    other_weights,
                    context_weights,
                    branch_weights,
                    late_weights,
                    local_between,
                    optimize=True,
                )
            )
            sampled_curves.append(local_within / local_reference)
        bootstrap[bootstrap_idx, 1:] = np.mean(
            np.stack(sampled_curves), axis=0
        )

    aggregate = np.mean(curves, axis=0)
    confidence_low, confidence_high = np.quantile(
        bootstrap, (0.025, 0.975), axis=0
    )
    run_rows: list[dict[str, Any]] = []
    for run_idx in range(10):
        for step_idx, step in enumerate(curve_steps):
            source_bin_idx = step_idx - 1
            run_rows.append(
                {
                    "candidate_id": f"run_{run_idx:03d}_optimized",
                    "run_idx": run_idx,
                    "step": int(step),
                    "window_start_exclusive": (
                        int(step - CHAMFER_BIN_WIDTH) if step > 0 else ""
                    ),
                    "window_end_inclusive": int(step) if step > 0 else "",
                    "within_rng_chamfer_mean": (
                        float(within_mean[run_idx, source_bin_idx])
                        if step > 0
                        else 0.0
                    ),
                    "run_matched_interopt_reference": float(
                        reference[run_idx]
                    ),
                    "attainment_ratio": float(curves[run_idx, step_idx]),
                }
            )

    summary_rows = [
        {
            "step": int(step),
            "run_balanced_mean_attainment": float(aggregate[step_idx]),
            "run_q25": float(np.quantile(curves[:, step_idx], 0.25)),
            "run_q75": float(np.quantile(curves[:, step_idx], 0.75)),
            "hierarchical_bootstrap_ci_low": float(
                confidence_low[step_idx]
            ),
            "hierarchical_bootstrap_ci_high": float(
                confidence_high[step_idx]
            ),
        }
        for step_idx, step in enumerate(curve_steps)
    ]

    crossing_rows: list[dict[str, Any]] = []
    for run_idx in range(10):
        sustained = np.flatnonzero(
            (ratios[run_idx, :-1] >= 1.0)
            & (ratios[run_idx, 1:] >= 1.0)
        )
        first_reach = (
            int(bin_ends[int(sustained[0])]) if sustained.size else None
        )
        crossing_rows.append(
            {
                "candidate_id": f"run_{run_idx:03d}_optimized",
                "run_idx": run_idx,
                "run_matched_interopt_reference": float(reference[run_idx]),
                "first_two_consecutive_windows_at_or_above_reference": (
                    first_reach if first_reach is not None else ""
                ),
                "reached_sustainably_by_300k": first_reach is not None,
                "maximum_attainment_ratio": float(
                    np.max(ratios[run_idx])
                ),
                "attainment_ratio_at_300k": float(ratios[run_idx, -1]),
            }
        )

    aggregate_sustained = np.flatnonzero(
        (aggregate[1:-1] >= 1.0) & (aggregate[2:] >= 1.0)
    )
    aggregate_first_reach = (
        int(bin_ends[int(aggregate_sustained[0])])
        if aggregate_sustained.size
        else None
    )
    report = {
        "metric": "20-frame symmetric CLIP-Chamfer in 20k windows",
        "reference_definition": (
            "For each anchor opt, arithmetic mean over the nine other "
            "optimized checkpoints, four matched initial states, eight "
            "matched RNG branches, and the five windows in (200k, 300k]."
        ),
        "aggregation": (
            "Arithmetic means within each anchor, then an equal-weight mean "
            "over the ten anchor runs."
        ),
        "bootstrap": {
            "method": (
                "Cluster resampling of anchor runs with hierarchical "
                "Bayesian-bootstrap weights for contexts, RNG branches, "
                "and other optimized checkpoints."
            ),
            "replicates": ATTAINMENT_BOOTSTRAP_REPLICATES,
            "seed": ATTAINMENT_BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "runs_with_two_consecutive_windows_at_or_above_reference": int(
            sum(
                bool(row["reached_sustainably_by_300k"])
                for row in crossing_rows
            )
        ),
        "aggregate_first_two_window_reach_step": aggregate_first_reach,
        "aggregate_attainment_at_300k": float(aggregate[-1]),
        "aggregate_maximum_attainment": float(np.max(aggregate)),
        "per_run_reference": {
            f"run_{idx:03d}_optimized": float(reference[idx])
            for idx in range(10)
        },
    }
    return run_rows, summary_rows, crossing_rows, report


def _plot_run_matched_attainment(
    figure_dir: Path,
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    crossing_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    steps = np.asarray([row["step"] for row in summary_rows], dtype=np.int32)
    aggregate = np.asarray(
        [row["run_balanced_mean_attainment"] for row in summary_rows],
        dtype=np.float64,
    )
    confidence_low = np.asarray(
        [row["hierarchical_bootstrap_ci_low"] for row in summary_rows],
        dtype=np.float64,
    )
    confidence_high = np.asarray(
        [row["hierarchical_bootstrap_ci_high"] for row in summary_rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for run_idx in range(10):
        rows = sorted(
            (
                row
                for row in run_rows
                if int(row["run_idx"]) == run_idx
            ),
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["attainment_ratio"]) for row in rows],
            color="#78838C",
            lw=1.1,
            alpha=0.5,
            label=(
                "Individual optimized runs (n=10)"
                if run_idx == 0
                else "_nolegend_"
            ),
        )

    ax.fill_between(
        steps,
        confidence_low,
        confidence_high,
        color="#147D92",
        alpha=0.16,
        linewidth=0,
        label="95% hierarchical bootstrap CI",
    )
    ax.plot(
        steps,
        aggregate,
        color="#111111",
        lw=3.0,
        label="Equal-weight mean over runs",
        zorder=5,
    )
    ax.axhline(1.0, color="#B23A48", lw=1.8, ls="--", zorder=2)
    ax.text(
        experiment.HORIZON_STEPS,
        1.025,
        "run-matched inter-opt reference",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#9E2F3D",
    )
    ax.scatter(
        [0],
        [0],
        s=44,
        color="#111111",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    ax.annotate(
        "exact shared state",
        xy=(0, 0),
        xytext=(20_000, 0.12),
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.9},
        fontsize=8,
        color="#444444",
    )

    for row in crossing_rows:
        crossing = row[
            "first_two_consecutive_windows_at_or_above_reference"
        ]
        if crossing == "":
            continue
        ax.scatter(
            [int(crossing)],
            [1.0],
            marker="v",
            s=48,
            color="#59636B",
            edgecolor="white",
            linewidth=0.7,
            zorder=7,
        )

    aggregate_crossing = report["aggregate_first_two_window_reach_step"]
    if aggregate_crossing is not None:
        ax.annotate(
            f"mean first reaches for two windows: {aggregate_crossing // 1000}k",
            xy=(aggregate_crossing, 1.0),
            xytext=(aggregate_crossing + 15_000, 1.20),
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.9},
            fontsize=8,
            color="#333333",
        )

    run_reach_count = int(
        report["runs_with_two_consecutive_windows_at_or_above_reference"]
    )
    ax.text(
        0.99,
        0.045,
        (
            f"{run_reach_count}/10 runs reach the reference for "
            "two consecutive windows"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.set_xlim(0, experiment.HORIZON_STEPS)
    ax.set_ylim(
        0,
        max(
            1.35,
            1.08 * float(np.max(confidence_high)),
            1.08
            * max(float(row["attainment_ratio"]) for row in run_rows),
        ),
    )
    ax.set_xlabel("Simulation steps from the exact shared initial state")
    ax.set_ylabel(
        "Within-RNG CLIP-Chamfer / run-matched inter-opt reference"
    )
    ax.grid(axis="both", alpha=0.18)
    fig.suptitle(
        "RNG divergence approaches the inter-checkpoint scale",
        fontsize=16,
        y=0.985,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.012,
        (
            "20k windows; each run is normalized by its own anchor-vs-other-"
            "opt mean over 200k-300k. Triangles mark the first of two "
            "consecutive windows at or above 1."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.105, right=0.985)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_run_matched_interopt_attainment.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_anchor_pairwise_chamfer(
    figure_dir: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    global_max = max(
        max(float(row["within_anchor_rng_q75"]) for row in rows),
        max(float(row["between_anchor_other_q75"]) for row in rows),
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(12.0, 16.5),
        sharex=True,
        sharey=True,
    )
    for anchor_idx, ax in enumerate(axes.flat):
        anchor_rows = [
            row for row in rows if int(row["anchor_run_idx"]) == anchor_idx
        ]
        centers = np.asarray(
            sorted({float(row["bin_center"]) for row in anchor_rows})
        )
        within_lookup = {
            float(row["bin_center"]): float(row["within_anchor_rng_median"])
            for row in anchor_rows
        }
        within = np.asarray([within_lookup[center] for center in centers])
        between_curves = []
        for other_idx in range(10):
            if other_idx == anchor_idx:
                continue
            comparison = sorted(
                [
                    row
                    for row in anchor_rows
                    if int(row["other_run_idx"]) == other_idx
                ],
                key=lambda row: float(row["bin_center"]),
            )
            curve = np.asarray(
                [
                    float(row["between_anchor_other_median"])
                    for row in comparison
                ]
            )
            between_curves.append(curve)
            ax.plot(
                centers,
                curve,
                color="#D1495B",
                lw=0.85,
                alpha=0.28,
            )
        ax.plot(
            centers,
            np.median(np.stack(between_curves), axis=0),
            color="#D1495B",
            lw=2.0,
            ls="--",
        )
        ax.plot(centers, within, color="#147D92", lw=2.2)
        ax.set_title(f"opt_{anchor_idx:03d}", fontsize=10)
        ax.set_xlim(0, experiment.HORIZON_STEPS)
        ax.set_ylim(0, 1.06 * global_max)
        ax.grid(alpha=0.16)
    for ax in axes[-1]:
        ax.set_xlabel("20k-window center")
    for ax in axes[:, 0]:
        ax.set_ylabel("CLIP-Chamfer")
    handles = [
        plt.Line2D(
            [0], [0], color="#147D92", lw=2.2, label="Within anchor: RNG"
        ),
        plt.Line2D(
            [0],
            [0],
            color="#D1495B",
            lw=0.9,
            alpha=0.4,
            label="Anchor vs each other opt",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#D1495B",
            lw=2.0,
            ls="--",
            label="Median over the 9 other opts",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Within-RNG versus anchor-to-each-checkpoint divergence",
        fontsize=15,
        y=0.988,
    )
    fig.subplots_adjust(
        top=0.925,
        bottom=0.045,
        left=0.075,
        right=0.985,
        hspace=0.28,
        wspace=0.08,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_vs_each_opt_chamfer_20k_bins.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)

    for anchor_idx in range(10):
        anchor_rows = [
            row for row in rows if int(row["anchor_run_idx"]) == anchor_idx
        ]
        centers = np.asarray(
            sorted({float(row["bin_center"]) for row in anchor_rows})
        )
        within_lookup = {
            float(row["bin_center"]): row for row in anchor_rows
        }
        within = np.asarray(
            [
                float(within_lookup[center]["within_anchor_rng_median"])
                for center in centers
            ]
        )
        within_q25 = np.asarray(
            [
                float(within_lookup[center]["within_anchor_rng_q25"])
                for center in centers
            ]
        )
        within_q75 = np.asarray(
            [
                float(within_lookup[center]["within_anchor_rng_q75"])
                for center in centers
            ]
        )
        fig, ax = plt.subplots(figsize=(8.3, 5.0), constrained_layout=True)
        ax.fill_between(
            centers,
            within_q25,
            within_q75,
            color="#147D92",
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            centers,
            within,
            color="#147D92",
            lw=2.6,
            label=f"Within opt_{anchor_idx:03d}: RNG",
        )
        for other_idx in range(10):
            if other_idx == anchor_idx:
                continue
            comparison = sorted(
                [
                    row
                    for row in anchor_rows
                    if int(row["other_run_idx"]) == other_idx
                ],
                key=lambda row: float(row["bin_center"]),
            )
            ax.plot(
                centers,
                [
                    float(row["between_anchor_other_median"])
                    for row in comparison
                ],
                color=colors[other_idx],
                lw=1.25,
                alpha=0.9,
                label=f"vs opt_{other_idx:03d}",
            )
        ax.set_xlim(0, experiment.HORIZON_STEPS)
        ax.set_ylim(0, 1.06 * global_max)
        ax.set_xlabel("Simulation step (20k-window center)")
        ax.set_ylabel("20-frame CLIP-Chamfer")
        ax.set_title(
            f"opt_{anchor_idx:03d}: RNG divergence versus every other opt"
        )
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        for suffix in ("png", "pdf"):
            fig.savefig(
                figure_dir
                / (
                    f"flowlenia_opt_{anchor_idx:03d}_rng_vs_other_opts_"
                    f"chamfer_20k_bins.{suffix}"
                ),
                dpi=240,
                bbox_inches="tight",
            )
        plt.close(fig)

    ratio_matrix = np.full((10, 10), np.nan, dtype=np.float64)
    for row in summary_rows:
        ratio_matrix[
            int(row["anchor_run_idx"]), int(row["other_run_idx"])
        ] = float(row["median_of_binwise_median_ratios"])
    finite = ratio_matrix[np.isfinite(ratio_matrix)]
    span = max(
        float(np.max(np.abs(finite - 1.0))),
        0.05,
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.5), constrained_layout=True)
    image = ax.imshow(
        ratio_matrix,
        cmap="RdBu_r",
        vmin=1.0 - span,
        vmax=1.0 + span,
        interpolation="nearest",
    )
    for anchor_idx in range(10):
        for other_idx in range(10):
            if anchor_idx == other_idx:
                continue
            value = ratio_matrix[anchor_idx, other_idx]
            ax.text(
                other_idx,
                anchor_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if abs(value - 1.0) > 0.55 * span
                    else "black"
                ),
            )
    ax.set_xticks(range(10), [f"{idx:03d}" for idx in range(10)])
    ax.set_yticks(range(10), [f"{idx:03d}" for idx in range(10)])
    ax.set_xlabel("Other optimized checkpoint")
    ax.set_ylabel("Anchor optimized checkpoint")
    ax.set_title(
        "Median across 20k bins: within-RNG(anchor) / between(anchor, other)"
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("Chamfer ratio (1 = equal)")
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_anchor_pairwise_chamfer_ratio_heatmap.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_fixed_target_crossings(
    figure_dir: Path, rows: list[dict[str, Any]]
) -> None:
    horizons = np.asarray(TARGET_HORIZONS, dtype=np.int32)

    def values(aggregation: str, fraction: float) -> np.ndarray:
        lookup = {
            (
                row["aggregation"],
                int(row["target_horizon_steps"]),
                float(row["target_fraction"]),
            ): row["first_reach_step"]
            for row in rows
        }
        return np.asarray(
            [
                (
                    float(lookup[(aggregation, int(horizon), fraction)])
                    if lookup[(aggregation, int(horizon), fraction)] != ""
                    else np.nan
                )
                for horizon in horizons
            ],
            dtype=np.float64,
        )

    series = (
        ("median, 90%", values("median", 0.90), "#147D92", "o", "-"),
        ("median, 100%", values("median", 1.00), "#147D92", "s", "--"),
        ("mean, 90%", values("mean", 0.90), "#D1495B", "o", "-"),
        ("mean, 100%", values("mean", 1.00), "#D1495B", "s", "--"),
    )
    fig, ax = plt.subplots(figsize=(7.7, 4.8), constrained_layout=True)
    for label, crossing, color, marker, line_style in series:
        ax.plot(
            horizons,
            crossing,
            color=color,
            marker=marker,
            ls=line_style,
            lw=1.8,
            ms=5,
            label=label,
        )
        missing = np.flatnonzero(~np.isfinite(crossing))
        for index in missing:
            ax.scatter(
                horizons[index],
                experiment.HORIZON_STEPS,
                marker="^",
                s=48,
                facecolors="none",
                edgecolors=color,
                zorder=4,
            )
            ax.annotate(
                ">300k",
                (horizons[index], experiment.HORIZON_STEPS),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color=color,
            )
    ax.plot(
        horizons,
        horizons,
        color="#777777",
        lw=1.1,
        ls=":",
        label="target horizon",
    )
    ax.set_xlabel("Between-checkpoint target horizon (steps)")
    ax.set_ylabel("First within-RNG reach step")
    ax.set_title("Time to reach a fixed between-checkpoint divergence level")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8_000, 340_000)
    ax.set_ylim(40, 400_000)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir
            / f"flowlenia_rng_vs_checkpoint_fixed_target_crossings.{suffix}",
            dpi=240,
            bbox_inches="tight",
        )
    plt.close(fig)


def analyze(output_root: Path) -> dict[str, Any]:
    data = _load_data(output_root)
    z = data["z"]
    distances = _pointwise_distances(z)
    point_indices = np.searchsorted(
        data["embed_steps"], np.asarray(experiment.POINT_STEPS)
    )
    if not np.array_equal(
        data["embed_steps"][point_indices],
        np.asarray(experiment.POINT_STEPS),
    ):
        raise RuntimeError("Pointwise capture grid is incomplete")
    point_distances = {
        key: value[..., point_indices] for key, value in distances.items()
    }
    steps = np.asarray(experiment.POINT_STEPS, dtype=np.int32)
    (
        summary_rows,
        candidate_rows,
        pair_rows,
        harness_rows,
    ) = _make_pointwise_rows(steps, point_distances)
    crossing_rows, crossing_report = _crossing_rows(steps, summary_rows)
    chamfer_rows = _trajectory_distances(z, data["embed_steps"])
    (
        binned_chamfer_rows,
        binned_within_candidate_rows,
        binned_between_pair_rows,
        binned_within_raw,
        binned_between_raw,
    ) = _binned_chamfer_distances(z, data["embed_steps"])
    (
        anchor_pairwise_rows,
        anchor_pairwise_summary_rows,
    ) = _anchor_pairwise_rows(
        binned_within_candidate_rows,
        binned_between_pair_rows,
    )
    (
        attainment_rows,
        attainment_summary_rows,
        attainment_crossing_rows,
        attainment_report,
    ) = _run_matched_attainment(
        binned_within_raw,
        binned_between_raw,
    )
    repeat = _repeat_audit(
        output_root, data["protocol"], data["candidates"][0]
    )

    tables = output_root / "analysis" / "tables"
    figures = output_root / "analysis" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(
        tables / "pointwise_curve_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        tables / "within_rng_per_optimized_run.csv",
        candidate_rows,
        list(candidate_rows[0]),
    )
    _write_csv(
        tables / "between_checkpoint_pair_curves.csv",
        pair_rows,
        list(pair_rows[0]),
    )
    _write_csv(
        tables / "duplicate_harness_per_optimized_run.csv",
        harness_rows,
        list(harness_rows[0]),
    )
    _write_csv(
        tables / "fixed_target_crossings.csv",
        crossing_rows,
        list(crossing_rows[0]),
    )
    _write_csv(
        tables / "trajectory_chamfer_summary.csv",
        chamfer_rows,
        list(chamfer_rows[0]),
    )
    _write_csv(
        tables / "chamfer_20k_bin_summary.csv",
        binned_chamfer_rows,
        list(binned_chamfer_rows[0]),
    )
    _write_csv(
        tables / "chamfer_20k_bin_within_per_optimized_run.csv",
        binned_within_candidate_rows,
        list(binned_within_candidate_rows[0]),
    )
    _write_csv(
        tables / "chamfer_20k_bin_between_checkpoint_pairs.csv",
        binned_between_pair_rows,
        list(binned_between_pair_rows[0]),
    )
    _write_csv(
        tables / "chamfer_20k_bin_anchor_pairwise.csv",
        anchor_pairwise_rows,
        list(anchor_pairwise_rows[0]),
    )
    _write_csv(
        tables / "chamfer_20k_bin_anchor_pairwise_summary.csv",
        anchor_pairwise_summary_rows,
        list(anchor_pairwise_summary_rows[0]),
    )
    _write_csv(
        tables / "rng_run_matched_interopt_attainment.csv",
        attainment_rows,
        list(attainment_rows[0]),
    )
    _write_csv(
        tables / "rng_run_matched_interopt_attainment_summary.csv",
        attainment_summary_rows,
        list(attainment_summary_rows[0]),
    )
    _write_csv(
        tables / "rng_run_matched_interopt_attainment_crossings.csv",
        attainment_crossing_rows,
        list(attainment_crossing_rows[0]),
    )
    _write_json(output_root / "analysis" / "repeat_audit.json", repeat)
    _write_json(
        output_root / "analysis" / "rng_run_matched_interopt_attainment.json",
        attainment_report,
    )

    _plot_pointwise(figures, steps, summary_rows)
    _plot_per_checkpoint(figures, steps, point_distances)
    _plot_chamfer(figures, chamfer_rows)
    _plot_binned_chamfer(figures, binned_chamfer_rows)
    _plot_anchor_pairwise_chamfer(
        figures,
        anchor_pairwise_rows,
        anchor_pairwise_summary_rows,
    )
    _plot_fixed_target_crossings(figures, crossing_rows)
    _plot_run_matched_attainment(
        figures,
        attainment_rows,
        attainment_summary_rows,
        attainment_crossing_rows,
        attainment_report,
    )

    initial_state_match = bool(
        np.all(
            data["source_full_hashes"]
            == data["source_full_hashes"][:1]
        )
    )
    rng_match = bool(
        np.all(data["branch_keys"] == data["branch_keys"][:1])
    )
    duplicate_a_max = float(np.max(data["state_duplicate_max"]))
    duplicate_a_relative_max = float(np.max(data["state_duplicate"]))
    initial_mass = data["total_mass"][:, :, 0, : experiment.N_UNIQUE_BRANCHES]
    final_mass = data["total_mass"][:, :, -1, : experiment.N_UNIQUE_BRANCHES]
    final_mass_ratio = final_mass / np.clip(initial_mass, 1.0e-12, None)
    harness = point_distances["harness"]
    within = point_distances["within"]
    positive_steps = steps > 0
    harness_median = np.median(harness[..., positive_steps], axis=(0, 1))
    within_median = np.median(within[..., positive_steps], axis=(0, 1, 2))
    attribution_limited = harness_median >= 0.1 * within_median
    report = {
        "status": (
            "complete"
            if initial_state_match
            and rng_match
            and duplicate_a_max == 0.0
            and repeat["status"] == "passed"
            else "failed_audit"
        ),
        "protocol_version": experiment.PROTOCOL_VERSION,
        "plan_sha256": data["protocol"]["plan_sha256"],
        "candidate_count": 10,
        "shared_initial_states": experiment.N_CONTEXTS,
        "unique_rng_branches": experiment.N_UNIQUE_BRANCHES,
        "horizon_steps": experiment.HORIZON_STEPS,
        "initial_physical_state_hashes_match_across_checkpoints": initial_state_match,
        "initial_rng_keys_match_across_checkpoints": rng_match,
        "duplicate_a_max_abs_over_all_runs_and_steps": duplicate_a_max,
        "duplicate_a_relative_l1_max": duplicate_a_relative_max,
        "duplicate_clip_cosine_distance_max": float(np.max(harness)),
        "repeat_audit_status": repeat["status"],
        "mass_audit": {
            "initial_mass_min": float(np.min(initial_mass)),
            "initial_mass_max": float(np.max(initial_mass)),
            "final_mass_min": float(np.min(final_mass)),
            "final_mass_max": float(np.max(final_mass)),
            "final_over_initial_min": float(np.min(final_mass_ratio)),
            "final_over_initial_median": float(np.median(final_mass_ratio)),
            "final_over_initial_max": float(np.max(final_mass_ratio)),
            "nonfinite_count": int(
                np.count_nonzero(~np.isfinite(data["total_mass"]))
            ),
            "near_zero_count": int(
                np.count_nonzero(data["total_mass"] <= 1.0e-8)
            ),
        },
        "harness_attribution_limited_positive_step_count": int(
            np.count_nonzero(attribution_limited)
        ),
        "harness_attribution_limited_first_step": (
            int(steps[positive_steps][np.flatnonzero(attribution_limited)[0]])
            if np.any(attribution_limited)
            else None
        ),
        "crossing": crossing_report,
        "run_matched_interopt_attainment": attainment_report,
        "total_simulation_elapsed_seconds_sum": float(
            np.sum(data["elapsed"])
        ),
        "input_trajectory_sha256": {
            path.parent.name: _sha256_file(path)
            for path in data["paths"]
        },
        "tables": sorted(str(path) for path in tables.glob("*.csv")),
        "figures": sorted(str(path) for path in figures.glob("*")),
        "interpretation_guardrail": (
            "This quantifies stochastic/RNG-induced Flow-Lenia trajectory "
            "sensitivity under a fixed batching harness. It is not a "
            "deterministic Lyapunov-exponent claim."
        ),
    }
    _write_json(output_root / "analysis" / "analysis_report.json", report)
    if report["status"] != "complete":
        raise RuntimeError(f"Analysis audit failed: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FlowLenia continuation-RNG versus optimized-checkpoint "
            "divergence through 300k steps."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    report = analyze(_resolve(cli.output_root))
    print(json.dumps(_jsonable(report), indent=2))


if __name__ == "__main__":
    main()
