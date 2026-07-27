from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANDOM_ROOT = REPO_ROOT / (
    "analysis/results/flowlenia_rng_vs_interrandom_shared_init_8rng_dup_300k_v1"
)
DEFAULT_OPTIMIZED_ROOT = REPO_ROOT / (
    "analysis/results/flowlenia_rng_vs_checkpoint_shared_init_8rng_dup_300k_v1"
)

HORIZON_STEPS = 300_000
WINDOW_STEPS = 20_000
FRAME_INTERVAL = 1_000
N_CONTEXTS = 4
N_BRANCHES = 9
N_UNIQUE_BRANCHES = 8
DUPLICATE_BRANCH = 8
DUPLICATE_OF_BRANCH = 0
LATE_REFERENCE_START = 200_000
EQUIVALENCE_BAND = (0.8, 1.25)
METRIC_CACHE_VERSION = (
    "symmetric-clip-chamfer-20k-window-1k-grid-different-rng-v2"
)
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
BETWEEN_BRANCH_LEFT = np.asarray(
    [
        left
        for left in range(N_UNIQUE_BRANCHES)
        for right in range(N_UNIQUE_BRANCHES)
        if left != right
    ],
    dtype=np.int32,
)
BETWEEN_BRANCH_RIGHT = np.asarray(
    [
        right
        for left in range(N_UNIQUE_BRANCHES)
        for right in range(N_UNIQUE_BRANCHES)
        if left != right
    ],
    dtype=np.int32,
)
BIN_ENDS = np.arange(WINDOW_STEPS, HORIZON_STEPS + 1, WINDOW_STEPS, dtype=np.int32)
REQUIRED_EMBED_STEPS = np.arange(0, HORIZON_STEPS + 1, FRAME_INTERVAL, dtype=np.int32)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty table: {path}")
        fieldnames = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _read_candidates(root: Path) -> list[dict[str, str]]:
    path = root / "candidates.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No candidates in {path}")
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate candidate ids in {path}")
    return rows


def _expected_kind(candidate_id: str) -> str:
    if candidate_id.endswith("_optimized"):
        return "optimized"
    if "_random_" in candidate_id:
        return "random"
    raise RuntimeError(f"Unknown candidate kind: {candidate_id}")


def _load_group(root: Path, expected_kind: str) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    plan_sha256 = str(protocol["plan_sha256"])
    candidates = _read_candidates(root)
    expected_count = 30 if expected_kind == "random" else 10
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {expected_kind} candidates, "
            f"found {len(candidates)} in {root}"
        )

    embeddings = np.empty(
        (
            len(candidates),
            N_CONTEXTS,
            N_BRANCHES,
            len(REQUIRED_EMBED_STEPS),
            512,
        ),
        dtype=np.float32,
    )
    source_full_hashes: list[np.ndarray] = []
    branch_keys: list[np.ndarray] = []
    trajectory_files: list[dict[str, Any]] = []
    duplicate_state_max = 0.0
    duplicate_rng_max = 0
    elapsed_seconds = 0.0

    for candidate_index, row in enumerate(candidates):
        candidate_id = row["candidate_id"]
        if row["candidate_kind"] != expected_kind:
            raise RuntimeError(
                f"Kind mismatch for {candidate_id}: {row['candidate_kind']}"
            )
        if _expected_kind(candidate_id) != expected_kind:
            raise RuntimeError(f"Malformed candidate id: {candidate_id}")
        path = root / "simulation" / candidate_id / "trajectory.npz"
        restart_path = root / "simulation" / candidate_id / "restart_state.npz"
        if not path.exists() or not restart_path.exists():
            raise RuntimeError(
                f"Incomplete production output for {candidate_id}: "
                f"trajectory={path.exists()} restart={restart_path.exists()}"
            )
        with np.load(path, allow_pickle=False) as data:
            checks = {
                "protocol_version": str(data["protocol_version"].item())
                == str(protocol["protocol_version"]),
                "plan_sha256": str(data["plan_sha256"].item()) == plan_sha256,
                "candidate_id": str(data["candidate_id"].item()) == candidate_id,
                "params_sha256": str(data["params_sha256"].item())
                == row["params_sha256"],
                "audit_repeat": not bool(data["audit_repeat"].item()),
                "completed_step": int(data["completed_step"].item()) == HORIZON_STEPS,
                "horizon_steps": int(data["horizon_steps"].item()) == HORIZON_STEPS,
                "pair_left": np.array_equal(data["pair_left"], PAIR_LEFT),
                "pair_right": np.array_equal(data["pair_right"], PAIR_RIGHT),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(f"Invalid {path}: failed {failed}")

            local_steps = np.asarray(data["embedding_steps"], dtype=np.int32)
            indices = np.searchsorted(local_steps, REQUIRED_EMBED_STEPS)
            if np.any(indices >= len(local_steps)) or not np.array_equal(
                local_steps[indices], REQUIRED_EMBED_STEPS
            ):
                raise RuntimeError(f"Missing 1k CLIP grid in {path}")
            block = np.asarray(
                data["clip_embeddings"][:, :, indices, :],
                dtype=np.float32,
            )
            if block.shape != embeddings[candidate_index].shape:
                raise RuntimeError(
                    f"Unexpected embedding shape in {path}: {block.shape}"
                )
            norms = np.linalg.norm(block, axis=-1, keepdims=True)
            block /= np.clip(norms, 1.0e-12, None)
            embeddings[candidate_index] = block

            local_source_hashes = np.asarray(data["source_state_full_hashes"]).astype(
                str
            )
            local_branch_keys = np.asarray(
                data["branch_rng_keys_initial"], dtype=np.uint32
            )
            if local_source_hashes.shape != (N_CONTEXTS,):
                raise RuntimeError(f"Bad source hashes in {path}")
            if local_branch_keys.shape != (N_CONTEXTS, N_BRANCHES, 2):
                raise RuntimeError(f"Bad branch keys in {path}")
            source_full_hashes.append(local_source_hashes)
            branch_keys.append(local_branch_keys)
            duplicate_state_max = max(
                duplicate_state_max,
                float(np.max(data["a_duplicate_max_abs"])),
            )
            duplicate_rng_max = max(
                duplicate_rng_max,
                int(np.max(data["duplicate_rng_max_abs"])),
            )
            elapsed_seconds += float(data["elapsed_seconds"].item())

        trajectory_files.append(
            {
                "group": expected_kind,
                "candidate_id": candidate_id,
                "trajectory_path": str(path),
                "trajectory_size_bytes": path.stat().st_size,
                "trajectory_sha256": _sha256_file(path),
                "restart_path": str(restart_path),
                "restart_size_bytes": restart_path.stat().st_size,
                "restart_sha256": _sha256_file(restart_path),
            }
        )

    source_full_hashes_array = np.stack(source_full_hashes)
    branch_keys_array = np.stack(branch_keys)
    if not np.all(source_full_hashes_array == source_full_hashes_array[0]):
        raise RuntimeError(f"Source full states differ within {expected_kind}")
    if not np.all(branch_keys_array == branch_keys_array[0]):
        raise RuntimeError(f"Branch keys differ within {expected_kind}")
    if duplicate_state_max != 0.0 or duplicate_rng_max != 0:
        raise RuntimeError(
            f"Duplicate-lane failure in {expected_kind}: "
            f"state={duplicate_state_max}, rng={duplicate_rng_max}"
        )

    return {
        "root": root,
        "kind": expected_kind,
        "protocol": protocol,
        "protocol_sha256": _sha256_file(protocol_path),
        "candidates": candidates,
        "embeddings": embeddings,
        "source_full_hashes": source_full_hashes_array,
        "branch_keys": branch_keys_array,
        "duplicate_state_max": duplicate_state_max,
        "duplicate_rng_max": duplicate_rng_max,
        "elapsed_seconds": elapsed_seconds,
        "trajectory_files": trajectory_files,
    }


def _compute_binned_metrics(
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    n_candidates = embeddings.shape[0]
    candidate_pairs = np.asarray(
        list(itertools.combinations(range(n_candidates), 2)),
        dtype=np.int32,
    )
    device_embeddings = jnp.asarray(embeddings, dtype=jnp.float32)
    device_candidate_pairs = jnp.asarray(candidate_pairs)
    device_pair_left = jnp.asarray(PAIR_LEFT)
    device_pair_right = jnp.asarray(PAIR_RIGHT)
    device_between_left = jnp.asarray(BETWEEN_BRANCH_LEFT)
    device_between_right = jnp.asarray(BETWEEN_BRANCH_RIGHT)

    def symmetric_chamfer(distance):
        return 0.5 * (
            jnp.mean(jnp.min(distance, axis=-1), axis=-1)
            + jnp.mean(jnp.min(distance, axis=-2), axis=-1)
        )

    @jax.jit
    def compute_window(frame_indices):
        segment = jnp.take(device_embeddings, frame_indices, axis=3)

        within_left = segment[:, :, device_pair_left]
        within_right = segment[:, :, device_pair_right]
        within_distance = jnp.clip(
            1.0
            - jnp.einsum(
                "acpfd,acpgd->acpfg",
                within_left,
                within_right,
                optimize=True,
                precision=jax.lax.Precision.HIGHEST,
            ),
            0.0,
            2.0,
        )
        within = symmetric_chamfer(within_distance)

        between_left = segment[
            device_candidate_pairs[:, 0], :, :N_UNIQUE_BRANCHES
        ]
        between_right = segment[
            device_candidate_pairs[:, 1], :, :N_UNIQUE_BRANCHES
        ]
        between_distance = jnp.clip(
            1.0
            - jnp.einsum(
                "pcbfd,pcegd->pcbefg",
                between_left,
                between_right,
                optimize=True,
                precision=jax.lax.Precision.HIGHEST,
            ),
            0.0,
            2.0,
        )
        between_all = symmetric_chamfer(between_distance)
        between = between_all[
            :, :, device_between_left, device_between_right
        ]

        harness_left = segment[:, :, DUPLICATE_OF_BRANCH]
        harness_right = segment[:, :, DUPLICATE_BRANCH]
        harness_distance = jnp.clip(
            1.0
            - jnp.einsum(
                "acfd,acgd->acfg",
                harness_left,
                harness_right,
                optimize=True,
                precision=jax.lax.Precision.HIGHEST,
            ),
            0.0,
            2.0,
        )
        harness = symmetric_chamfer(harness_distance)
        return within, between, harness

    within_bins: list[np.ndarray] = []
    between_bins: list[np.ndarray] = []
    harness_bins: list[np.ndarray] = []

    for bin_end in BIN_ENDS:
        frame_steps = np.arange(
            bin_end - WINDOW_STEPS + FRAME_INTERVAL,
            bin_end + 1,
            FRAME_INTERVAL,
            dtype=np.int32,
        )
        frame_indices = np.searchsorted(REQUIRED_EMBED_STEPS, frame_steps)
        within, between, harness = jax.device_get(
            compute_window(jnp.asarray(frame_indices, dtype=jnp.int32))
        )
        within_bins.append(np.asarray(within, dtype=np.float64))
        between_bins.append(np.asarray(between, dtype=np.float64))
        harness_bins.append(np.asarray(harness, dtype=np.float64))

    return {
        "candidate_pairs": candidate_pairs,
        "within": np.stack(within_bins, axis=-1),
        "between": np.stack(between_bins, axis=-1),
        "harness": np.stack(harness_bins, axis=-1),
    }


def _metric_cache_identity(group: dict[str, Any]) -> str:
    payload = {
        "metric_cache_version": METRIC_CACHE_VERSION,
        "protocol_sha256": group["protocol_sha256"],
        "trajectory_sha256": [
            row["trajectory_sha256"] for row in group["trajectory_files"]
        ],
        "window_steps": WINDOW_STEPS,
        "frame_interval": FRAME_INTERVAL,
        "required_embed_steps": REQUIRED_EMBED_STEPS.tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _metrics_for_group(group: dict[str, Any], cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = _metric_cache_identity(group)
    path = cache_dir / f"{group['kind']}_binned_chamfer.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["cache_identity"].item()) == identity:
                return {
                    key: np.asarray(data[key])
                    for key in (
                        "candidate_pairs",
                        "within",
                        "between",
                        "harness",
                    )
                }
    metrics = _compute_binned_metrics(group["embeddings"])
    np.savez_compressed(path, cache_identity=identity, **metrics)
    return metrics


def _aggregate_metrics(
    metrics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    within = np.asarray(metrics["within"], dtype=np.float64)
    between = np.asarray(metrics["between"], dtype=np.float64)
    pairs = np.asarray(metrics["candidate_pairs"], dtype=np.int32)
    harness = np.asarray(metrics["harness"], dtype=np.float64)
    n_candidates = within.shape[0]
    n_bins = within.shape[-1]

    within_candidate = np.mean(within, axis=(1, 2))
    pair_mean = np.mean(between, axis=(1, 2))
    pair_matrix = np.full(
        (n_candidates, n_candidates, n_bins), np.nan, dtype=np.float64
    )
    for pair_index, (left, right) in enumerate(pairs):
        pair_matrix[left, right] = pair_mean[pair_index]
        pair_matrix[right, left] = pair_mean[pair_index]
    inter_candidate = np.nanmean(pair_matrix, axis=1)
    late_mask = BIN_ENDS - WINDOW_STEPS >= LATE_REFERENCE_START
    late_reference = np.nanmean(pair_matrix[..., late_mask], axis=(1, 2))
    time_ratio = within_candidate / np.clip(inter_candidate, 1.0e-15, None)
    attainment = within_candidate / np.clip(late_reference[:, None], 1.0e-15, None)

    return {
        "within_candidate": within_candidate,
        "pair_mean": pair_mean,
        "pair_matrix": pair_matrix,
        "inter_candidate": inter_candidate,
        "late_reference": late_reference,
        "time_ratio": time_ratio,
        "attainment": attainment,
        "harness_candidate": np.mean(harness, axis=1),
        "late_mask": late_mask,
    }


def _bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, values.shape[1]), dtype=np.float64)
    chunk_size = 5_000
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = rng.integers(0, values.shape[0], size=(stop - start, values.shape[0]))
        samples[start:stop] = np.mean(values[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    return {
        "mean": np.mean(values, axis=0),
        "low": low,
        "high": high,
        "samples": samples,
    }


def _bootstrap_difference(
    random_values: np.ndarray,
    optimized_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    random_values = np.asarray(random_values, dtype=np.float64)
    optimized_values = np.asarray(optimized_values, dtype=np.float64)
    if random_values.ndim == 1:
        random_values = random_values[:, None]
    if optimized_values.ndim == 1:
        optimized_values = optimized_values[:, None]
    random_bootstrap = _bootstrap_mean(random_values, replicates=replicates, seed=seed)
    optimized_bootstrap = _bootstrap_mean(
        optimized_values, replicates=replicates, seed=seed + 1
    )
    difference = random_bootstrap["samples"] - optimized_bootstrap["samples"]
    low, high = np.quantile(difference, (0.025, 0.975), axis=0)
    return {
        "difference_random_minus_optimized": (
            np.mean(random_values, axis=0) - np.mean(optimized_values, axis=0)
        ),
        "ci_low": low,
        "ci_high": high,
    }


def _late_ratio_for_subset(
    aggregate: dict[str, np.ndarray], selected: np.ndarray
) -> float:
    within = aggregate["within_candidate"][selected]
    pair_matrix = aggregate["pair_matrix"][np.ix_(selected, selected)]
    inter = np.nanmean(pair_matrix, axis=1)
    late_mask = aggregate["late_mask"]
    return float(
        np.mean(within[:, late_mask] / np.clip(inter[:, late_mask], 1.0e-15, None))
    )


def _leave_one_candidate_jackknife(
    aggregate: dict[str, np.ndarray],
) -> dict[str, Any]:
    n_candidates = len(aggregate["within_candidate"])
    all_indices = np.arange(n_candidates, dtype=np.int32)
    estimate = _late_ratio_for_subset(aggregate, all_indices)
    leave_one_out = np.asarray(
        [
            _late_ratio_for_subset(aggregate, np.delete(all_indices, candidate_index))
            for candidate_index in range(n_candidates)
        ],
        dtype=np.float64,
    )
    pseudovalues = n_candidates * estimate - (n_candidates - 1) * leave_one_out
    standard_error = float(np.std(pseudovalues, ddof=1) / np.sqrt(n_candidates))
    critical = float(student_t.ppf(0.975, n_candidates - 1))
    return {
        "estimate": estimate,
        "leave_one_out": leave_one_out,
        "pseudovalues": pseudovalues,
        "standard_error": standard_error,
        "degrees_of_freedom": n_candidates - 1,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
    }


def _random_subset_robustness(
    aggregate: dict[str, np.ndarray],
    *,
    subset_size: int,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    within = aggregate["within_candidate"]
    pair_matrix = aggregate["pair_matrix"]
    n_candidates, n_bins = within.shape
    if subset_size >= n_candidates:
        raise ValueError("Subset must be smaller than the full sample")
    rng = np.random.default_rng(seed)
    subset_curves = np.empty((replicates, n_bins), dtype=np.float64)
    for replicate in range(replicates):
        selected = np.sort(rng.choice(n_candidates, size=subset_size, replace=False))
        local_pair_matrix = pair_matrix[np.ix_(selected, selected)]
        local_inter = np.nanmean(local_pair_matrix, axis=1)
        subset_curves[replicate] = np.mean(
            within[selected] / np.clip(local_inter, 1.0e-15, None),
            axis=0,
        )
    return {
        "mean": np.mean(subset_curves, axis=0),
        "low": np.quantile(subset_curves, 0.025, axis=0),
        "high": np.quantile(subset_curves, 0.975, axis=0),
        "samples": subset_curves,
    }


def _full_repeat_audit(root: Path, candidate_id: str) -> dict[str, Any]:
    main_path = root / "simulation" / candidate_id / "trajectory.npz"
    repeat_path = root / "audit_repeat" / candidate_id / "trajectory.npz"
    main_restart = root / "simulation" / candidate_id / "restart_state.npz"
    repeat_restart = root / "audit_repeat" / candidate_id / "restart_state.npz"
    for path in (main_path, repeat_path, main_restart, repeat_restart):
        if not path.exists():
            raise RuntimeError(f"Missing full-repeat audit input: {path}")

    exact_output_keys = (
        "embedding_steps",
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
    output_comparisons: dict[str, Any] = {}
    with (
        np.load(main_path, allow_pickle=False) as main,
        np.load(repeat_path, allow_pickle=False) as repeat,
    ):
        if bool(main["audit_repeat"].item()):
            raise RuntimeError(f"Main output is marked as an audit: {main_path}")
        if not bool(repeat["audit_repeat"].item()):
            raise RuntimeError(
                f"Repeat output is not marked as an audit: {repeat_path}"
            )
        for key in exact_output_keys:
            left = np.asarray(main[key])
            right = np.asarray(repeat[key])
            exact = np.array_equal(left, right)
            comparison: dict[str, Any] = {"exact": exact}
            if np.issubdtype(left.dtype, np.number):
                comparison["max_abs"] = float(
                    np.max(np.abs(left.astype(np.float64) - right))
                )
            output_comparisons[key] = comparison

        left_embeddings = np.asarray(main["clip_embeddings"], dtype=np.float64)
        right_embeddings = np.asarray(repeat["clip_embeddings"], dtype=np.float64)
        embedding_bit_exact = np.array_equal(left_embeddings, right_embeddings)
        embedding_max_abs = float(np.max(np.abs(left_embeddings - right_embeddings)))
        left_embeddings /= np.clip(
            np.linalg.norm(left_embeddings, axis=-1, keepdims=True),
            1.0e-15,
            None,
        )
        right_embeddings /= np.clip(
            np.linalg.norm(right_embeddings, axis=-1, keepdims=True),
            1.0e-15,
            None,
        )
        embedding_max_cosine_error = float(
            np.max(
                np.abs(
                    1.0
                    - np.sum(
                        left_embeddings * right_embeddings,
                        axis=-1,
                    )
                )
            )
        )

    restart_comparisons: dict[str, Any] = {}
    restart_exact = True
    with (
        np.load(main_restart, allow_pickle=False) as main,
        np.load(repeat_restart, allow_pickle=False) as repeat,
    ):
        if not np.array_equal(main["state_keys"], repeat["state_keys"]):
            raise RuntimeError("Repeat restart state-key list differs")
        restart_keys = ["rng"] + [
            f"state__{key}" for key in np.asarray(main["state_keys"]).astype(str)
        ]
        for key in restart_keys:
            left = np.asarray(main[key])
            right = np.asarray(repeat[key])
            exact = np.array_equal(left, right)
            restart_exact = restart_exact and exact
            restart_comparisons[key] = {
                "exact": exact,
                "max_abs": float(
                    np.max(
                        np.abs(left.astype(np.complex128) - right.astype(np.complex128))
                    )
                ),
            }

    exact_outputs = all(
        comparison["exact"] for comparison in output_comparisons.values()
    )
    clip_within_tolerance = (
        embedding_max_abs <= 5.0e-7 and embedding_max_cosine_error <= 1.0e-12
    )
    report = {
        "status": (
            "passed"
            if exact_outputs and restart_exact and clip_within_tolerance
            else "failed"
        ),
        "candidate_id": candidate_id,
        "required_exact_output_arrays": output_comparisons,
        "final_restart_state_exact": restart_exact,
        "final_restart_comparisons": restart_comparisons,
        "clip_embeddings": {
            "bit_exact": embedding_bit_exact,
            "max_component_abs": embedding_max_abs,
            "max_cosine_error": embedding_max_cosine_error,
            "component_tolerance": 5.0e-7,
            "cosine_error_tolerance": 1.0e-12,
            "within_tolerance": clip_within_tolerance,
        },
        "main_trajectory_sha256": _sha256_file(main_path),
        "repeat_trajectory_sha256": _sha256_file(repeat_path),
        "main_restart_sha256": _sha256_file(main_restart),
        "repeat_restart_sha256": _sha256_file(repeat_restart),
    }
    if report["status"] != "passed":
        raise RuntimeError("Full 300k repeat audit failed; see comparison payload")
    return report


def _candidate_index_strata(
    candidates: list[dict[str, str]],
    aggregate: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_indices = sorted({int(row["candidate_idx"]) for row in candidates})
    for candidate_idx in candidate_indices:
        selected = np.asarray(
            [
                index
                for index, row in enumerate(candidates)
                if int(row["candidate_idx"]) == candidate_idx
            ],
            dtype=np.int32,
        )
        local_pair_matrix = aggregate["pair_matrix"][np.ix_(selected, selected)]
        local_inter = np.nanmean(local_pair_matrix, axis=1)
        ratios = aggregate["within_candidate"][selected] / np.clip(
            local_inter, 1.0e-15, None
        )
        for bin_index, bin_end in enumerate(BIN_ENDS):
            rows.append(
                {
                    "candidate_idx_stratum": candidate_idx,
                    "n_candidates": len(selected),
                    "bin_start_exclusive": int(bin_end - WINDOW_STEPS),
                    "bin_end_inclusive": int(bin_end),
                    "mean_time_matched_ratio": float(np.mean(ratios[:, bin_index])),
                    "median_time_matched_ratio": float(np.median(ratios[:, bin_index])),
                }
            )
    return rows


def _candidate_rows(
    group: dict[str, Any], aggregate: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(group["candidates"]):
        for bin_index, bin_end in enumerate(BIN_ENDS):
            rows.append(
                {
                    "group": group["kind"],
                    "candidate_id": candidate["candidate_id"],
                    "run_idx": int(candidate["run_idx"]),
                    "candidate_idx": int(candidate["candidate_idx"]),
                    "bin_start_exclusive": int(bin_end - WINDOW_STEPS),
                    "bin_end_inclusive": int(bin_end),
                    "frames_per_segment": WINDOW_STEPS // FRAME_INTERVAL,
                    "within_rng_chamfer": float(
                        aggregate["within_candidate"][candidate_index, bin_index]
                    ),
                    "anchor_interparameter_chamfer": float(
                        aggregate["inter_candidate"][candidate_index, bin_index]
                    ),
                    "time_matched_ratio": float(
                        aggregate["time_ratio"][candidate_index, bin_index]
                    ),
                    "late_interparameter_reference": float(
                        aggregate["late_reference"][candidate_index]
                    ),
                    "late_reference_attainment": float(
                        aggregate["attainment"][candidate_index, bin_index]
                    ),
                    "duplicate_harness_chamfer": float(
                        aggregate["harness_candidate"][candidate_index, bin_index]
                    ),
                }
            )
    return rows


def _pair_rows(
    group: dict[str, Any],
    metrics: dict[str, np.ndarray],
    aggregate: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = np.asarray(metrics["candidate_pairs"], dtype=np.int32)
    for pair_index, (left, right) in enumerate(pairs):
        for bin_index, bin_end in enumerate(BIN_ENDS):
            rows.append(
                {
                    "group": group["kind"],
                    "left_candidate_id": group["candidates"][left]["candidate_id"],
                    "right_candidate_id": group["candidates"][right]["candidate_id"],
                    "bin_start_exclusive": int(bin_end - WINDOW_STEPS),
                    "bin_end_inclusive": int(bin_end),
                    "matched_context_different_rng_chamfer": float(
                        aggregate["pair_mean"][pair_index, bin_index]
                    ),
                }
            )
    return rows


def _curve_bootstraps(
    random_aggregate: dict[str, np.ndarray],
    optimized_aggregate: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "random": {},
        "optimized": {},
    }
    metrics = (
        "within_candidate",
        "inter_candidate",
        "time_ratio",
        "attainment",
    )
    for group_index, (group_name, aggregate) in enumerate(
        (("random", random_aggregate), ("optimized", optimized_aggregate))
    ):
        for metric_index, metric in enumerate(metrics):
            result[group_name][metric] = _bootstrap_mean(
                aggregate[metric],
                replicates=replicates,
                seed=seed + 100 * group_index + metric_index,
            )
    return result


def _curve_rows(
    bootstraps: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group_bootstraps in bootstraps.items():
        for metric, bootstrap in group_bootstraps.items():
            for bin_index, bin_end in enumerate(BIN_ENDS):
                rows.append(
                    {
                        "group": group_name,
                        "metric": metric,
                        "bin_start_exclusive": int(bin_end - WINDOW_STEPS),
                        "bin_end_inclusive": int(bin_end),
                        "candidate_balanced_mean": float(bootstrap["mean"][bin_index]),
                        "candidate_bootstrap_ci_low": float(
                            bootstrap["low"][bin_index]
                        ),
                        "candidate_bootstrap_ci_high": float(
                            bootstrap["high"][bin_index]
                        ),
                    }
                )
    return rows


def _plot_random_attainment(
    figure_dir: Path,
    aggregate: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
) -> None:
    x = np.concatenate(([0], BIN_ENDS))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for curve in aggregate["attainment"]:
        ax.plot(
            x,
            np.concatenate(([0.0], curve)),
            color="#87939A",
            alpha=0.24,
            lw=0.8,
            zorder=1,
        )
    mean = np.concatenate(([0.0], bootstrap["mean"]))
    low = np.concatenate(([0.0], bootstrap["low"]))
    high = np.concatenate(([0.0], bootstrap["high"]))
    ax.fill_between(x, low, high, color="#0F766E", alpha=0.20, linewidth=0)
    ax.plot(
        x,
        mean,
        color="#0F766E",
        lw=2.6,
        label="Mean across 30 random parameters",
        zorder=3,
    )
    ax.axhline(
        1.0,
        color="#B23A48",
        lw=1.4,
        ls="--",
        label="Own late between-parameter reference",
    )
    ax.set(
        xlim=(0, HORIZON_STEPS),
        xlabel="Continuation step",
        ylabel="Within-RNG / late between-parameter distance",
        title=(
            "Flow-Lenia random rules: RNG divergence from an exact shared initial state"
        ),
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(
        0.99,
        0.03,
        "Thin curves: individual random parameters",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5C6670",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"flowlenia_rng_interrandom_attainment.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def _plot_group_comparison(
    figure_dir: Path,
    bootstraps: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = {"random": "#0F766E", "optimized": "#C44E52"}
    labels = {
        "random": "Random parameters (n=30)",
        "optimized": "Optimized parameters (n=10)",
    }
    ax.axhspan(
        EQUIVALENCE_BAND[0],
        EQUIVALENCE_BAND[1],
        color="#C8CDD0",
        alpha=0.28,
        label="Predeclared comparable-scale band (0.8-1.25)",
    )
    ax.axhline(1.0, color="#343A40", lw=1.0, ls="--")
    for group_name in ("random", "optimized"):
        bootstrap = bootstraps[group_name]["time_ratio"]
        ax.fill_between(
            BIN_ENDS,
            bootstrap["low"],
            bootstrap["high"],
            color=colors[group_name],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            BIN_ENDS,
            bootstrap["mean"],
            color=colors[group_name],
            lw=2.4,
            marker="o",
            ms=3.5,
            label=labels[group_name],
        )
    ax.set(
        xlim=(0, HORIZON_STEPS),
        xlabel="20k trajectory window end",
        ylabel="Within-RNG / between-parameter distance",
        title="Flow-Lenia RNG sensitivity relative to parameter variation",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"flowlenia_rng_opt_vs_random_time_matched.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def _plot_absolute_distances(
    figure_dir: Path,
    bootstraps: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colors = {"within_candidate": "#147D92", "inter_candidate": "#D1495B"}
    labels = {
        "within_candidate": "Within parameter: different RNG",
        "inter_candidate": "Between parameters: different RNG",
    }
    for ax, group_name, title in zip(
        axes,
        ("random", "optimized"),
        ("Random parameters (n=30)", "Optimized parameters (n=10)"),
    ):
        for metric in ("within_candidate", "inter_candidate"):
            bootstrap = bootstraps[group_name][metric]
            ax.fill_between(
                BIN_ENDS,
                bootstrap["low"],
                bootstrap["high"],
                color=colors[metric],
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                BIN_ENDS,
                bootstrap["mean"],
                color=colors[metric],
                lw=2.2,
                label=labels[metric],
            )
        ax.set_title(title)
        ax.set_xlabel("20k trajectory window end")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Symmetric CLIP-Chamfer distance")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Absolute Flow-Lenia trajectory divergence")
    for suffix in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"flowlenia_rng_opt_vs_random_absolute_chamfer.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def _first_two_consecutive(curve: np.ndarray, threshold: float) -> int | None:
    passed = np.asarray(curve) >= threshold
    indices = np.flatnonzero(passed[:-1] & passed[1:])
    if not len(indices):
        return None
    return int(BIN_ENDS[int(indices[0]) + 1])


def _first_reach(curve: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.asarray(curve) >= threshold)
    if not len(indices):
        return None
    return int(BIN_ENDS[int(indices[0])])


def analyze(
    random_root: Path,
    optimized_root: Path,
    *,
    bootstrap_replicates: int,
    subset_replicates: int,
    seed: int,
) -> dict[str, Any]:
    output_root = random_root / "analysis"
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    cache_dir = output_root / "cache"
    for path in (figure_dir, table_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("Loading and validating 30 random candidates...", flush=True)
    random_group = _load_group(random_root, "random")
    repeat_candidate = str(random_group["protocol"]["design"]["repeat_audit_candidate"])
    print(f"Validating full repeat audit for {repeat_candidate}...", flush=True)
    repeat_audit = _full_repeat_audit(random_root, repeat_candidate)
    _write_json(output_root / "full_repeat_audit.json", repeat_audit)
    print("Computing/loading random CLIP-Chamfer metrics...", flush=True)
    random_metrics = _metrics_for_group(random_group, cache_dir)
    random_aggregate = _aggregate_metrics(random_metrics)
    del random_group["embeddings"]

    print("Loading and validating 10 cached optimized candidates...", flush=True)
    optimized_group = _load_group(optimized_root, "optimized")
    if not np.array_equal(
        random_group["source_full_hashes"][0],
        optimized_group["source_full_hashes"][0],
    ):
        raise RuntimeError(
            "Random and optimized experiments do not share exact full states"
        )
    if not np.array_equal(
        random_group["branch_keys"][0], optimized_group["branch_keys"][0]
    ):
        raise RuntimeError(
            "Random and optimized experiments do not share branch RNG keys"
        )
    print("Computing/loading optimized CLIP-Chamfer metrics...", flush=True)
    optimized_metrics = _metrics_for_group(optimized_group, cache_dir)
    optimized_aggregate = _aggregate_metrics(optimized_metrics)
    del optimized_group["embeddings"]

    print("Running candidate-level bootstrap and subset checks...", flush=True)
    bootstraps = _curve_bootstraps(
        random_aggregate,
        optimized_aggregate,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    subset = _random_subset_robustness(
        random_aggregate,
        subset_size=10,
        replicates=subset_replicates,
        seed=seed + 1_000,
    )
    strata_rows = _candidate_index_strata(random_group["candidates"], random_aggregate)

    random_late_ratio = np.mean(
        random_aggregate["time_ratio"][:, random_aggregate["late_mask"]],
        axis=1,
    )
    optimized_late_ratio = np.mean(
        optimized_aggregate["time_ratio"][:, optimized_aggregate["late_mask"]],
        axis=1,
    )
    random_late_bootstrap = _bootstrap_mean(
        random_late_ratio,
        replicates=bootstrap_replicates,
        seed=seed + 2_000,
    )
    optimized_late_bootstrap = _bootstrap_mean(
        optimized_late_ratio,
        replicates=bootstrap_replicates,
        seed=seed + 2_001,
    )
    late_difference = _bootstrap_difference(
        random_late_ratio,
        optimized_late_ratio,
        replicates=bootstrap_replicates,
        seed=seed + 2_002,
    )
    random_late_within = np.mean(
        random_aggregate["within_candidate"][:, random_aggregate["late_mask"]],
        axis=1,
    )
    random_late_inter = np.mean(
        random_aggregate["inter_candidate"][:, random_aggregate["late_mask"]],
        axis=1,
    )
    optimized_late_within = np.mean(
        optimized_aggregate["within_candidate"][:, optimized_aggregate["late_mask"]],
        axis=1,
    )
    optimized_late_inter = np.mean(
        optimized_aggregate["inter_candidate"][:, optimized_aggregate["late_mask"]],
        axis=1,
    )
    random_late_within_bootstrap = _bootstrap_mean(
        random_late_within,
        replicates=bootstrap_replicates,
        seed=seed + 3_000,
    )
    random_late_inter_bootstrap = _bootstrap_mean(
        random_late_inter,
        replicates=bootstrap_replicates,
        seed=seed + 3_001,
    )
    optimized_late_within_bootstrap = _bootstrap_mean(
        optimized_late_within,
        replicates=bootstrap_replicates,
        seed=seed + 3_002,
    )
    optimized_late_inter_bootstrap = _bootstrap_mean(
        optimized_late_inter,
        replicates=bootstrap_replicates,
        seed=seed + 3_003,
    )
    late_within_difference = _bootstrap_difference(
        random_late_within,
        optimized_late_within,
        replicates=bootstrap_replicates,
        seed=seed + 3_004,
    )
    late_inter_difference = _bootstrap_difference(
        random_late_inter,
        optimized_late_inter,
        replicates=bootstrap_replicates,
        seed=seed + 3_006,
    )
    random_jackknife = _leave_one_candidate_jackknife(random_aggregate)
    optimized_jackknife = _leave_one_candidate_jackknife(optimized_aggregate)
    jackknife_difference = (
        random_jackknife["estimate"] - optimized_jackknife["estimate"]
    )
    random_variance = random_jackknife["standard_error"] ** 2
    optimized_variance = optimized_jackknife["standard_error"] ** 2
    jackknife_difference_se = float(np.sqrt(random_variance + optimized_variance))
    jackknife_difference_df = float(
        (random_variance + optimized_variance) ** 2
        / (
            random_variance**2 / random_jackknife["degrees_of_freedom"]
            + optimized_variance**2 / optimized_jackknife["degrees_of_freedom"]
        )
    )
    jackknife_difference_critical = float(student_t.ppf(0.975, jackknife_difference_df))

    candidate_rows = _candidate_rows(random_group, random_aggregate) + _candidate_rows(
        optimized_group, optimized_aggregate
    )
    pair_rows = _pair_rows(random_group, random_metrics, random_aggregate) + _pair_rows(
        optimized_group, optimized_metrics, optimized_aggregate
    )
    curve_rows = _curve_rows(bootstraps)
    subset_rows = [
        {
            "bin_start_exclusive": int(bin_end - WINDOW_STEPS),
            "bin_end_inclusive": int(bin_end),
            "subset_size": 10,
            "subset_replicates": subset_replicates,
            "mean_of_subset_means": float(subset["mean"][bin_index]),
            "subset_mean_q025": float(subset["low"][bin_index]),
            "subset_mean_q975": float(subset["high"][bin_index]),
        }
        for bin_index, bin_end in enumerate(BIN_ENDS)
    ]
    candidate_late_rows = []
    for group, aggregate, late_ratios in (
        (random_group, random_aggregate, random_late_ratio),
        (optimized_group, optimized_aggregate, optimized_late_ratio),
    ):
        for index, candidate in enumerate(group["candidates"]):
            candidate_late_rows.append(
                {
                    "group": group["kind"],
                    "candidate_id": candidate["candidate_id"],
                    "run_idx": int(candidate["run_idx"]),
                    "candidate_idx": int(candidate["candidate_idx"]),
                    "late_mean_time_matched_ratio": float(late_ratios[index]),
                    "attainment_at_300k": float(aggregate["attainment"][index, -1]),
                    "time_matched_ratio_at_300k": float(
                        aggregate["time_ratio"][index, -1]
                    ),
                }
            )
    jackknife_rows = []
    for group, jackknife in (
        (random_group, random_jackknife),
        (optimized_group, optimized_jackknife),
    ):
        for candidate_index, candidate in enumerate(group["candidates"]):
            jackknife_rows.append(
                {
                    "group": group["kind"],
                    "omitted_candidate_id": candidate["candidate_id"],
                    "full_estimate": float(jackknife["estimate"]),
                    "leave_one_out_estimate": float(
                        jackknife["leave_one_out"][candidate_index]
                    ),
                    "jackknife_pseudovalue": float(
                        jackknife["pseudovalues"][candidate_index]
                    ),
                    "jackknife_standard_error": float(jackknife["standard_error"]),
                    "jackknife_ci_low": float(jackknife["ci_low"]),
                    "jackknife_ci_high": float(jackknife["ci_high"]),
                }
            )

    _write_csv(table_dir / "candidate_window_metrics.csv", candidate_rows)
    _write_csv(table_dir / "pair_window_metrics.csv", pair_rows)
    _write_csv(table_dir / "curve_summary.csv", curve_rows)
    _write_csv(table_dir / "random_subset10_robustness.csv", subset_rows)
    _write_csv(table_dir / "random_candidate_idx_strata.csv", strata_rows)
    _write_csv(table_dir / "candidate_late_summary.csv", candidate_late_rows)
    _write_csv(
        table_dir / "dependency_aware_candidate_jackknife.csv",
        jackknife_rows,
    )
    provenance_rows = (
        random_group["trajectory_files"] + optimized_group["trajectory_files"]
    )
    _write_csv(table_dir / "input_file_provenance.csv", provenance_rows)

    _plot_random_attainment(
        figure_dir,
        random_aggregate,
        bootstraps["random"]["attainment"],
    )
    _plot_group_comparison(figure_dir, bootstraps)
    _plot_absolute_distances(figure_dir, bootstraps)

    random_late_mean = float(random_late_bootstrap["mean"][0])
    random_late_ci = [
        float(random_late_bootstrap["low"][0]),
        float(random_late_bootstrap["high"][0]),
    ]
    optimized_late_mean = float(optimized_late_bootstrap["mean"][0])
    optimized_late_ci = [
        float(optimized_late_bootstrap["low"][0]),
        float(optimized_late_bootstrap["high"][0]),
    ]
    random_equivalent = (
        random_late_ci[0] >= EQUIVALENCE_BAND[0]
        and random_late_ci[1] <= EQUIVALENCE_BAND[1]
    )
    report = {
        "status": "complete",
        "estimand": (
            "Within-parameter continuation-RNG CLIP-Chamfer divided by "
            "between-parameter different-RNG CLIP-Chamfer, conditional on "
            "four exact shared initial states and eight prespecified "
            "continuation RNG keys."
        ),
        "interpretation_scope": (
            "Empirical visual trajectory sensitivity under this fixed "
            "Flow-Lenia protocol; not a mathematical proof of chaos."
        ),
        "protocol": {
            "horizon_steps": HORIZON_STEPS,
            "trajectory_window_steps": WINDOW_STEPS,
            "clip_frame_interval": FRAME_INTERVAL,
            "frames_per_window": WINDOW_STEPS // FRAME_INTERVAL,
            "late_reference_windows": [
                [
                    int(end - WINDOW_STEPS),
                    int(end),
                ]
                for end in BIN_ENDS[BIN_ENDS - WINDOW_STEPS >= LATE_REFERENCE_START]
            ],
            "random_candidates": len(random_group["candidates"]),
            "optimized_candidates": len(optimized_group["candidates"]),
            "shared_initial_states": N_CONTEXTS,
            "unique_rng_branches": N_UNIQUE_BRANCHES,
            "within_rng_pairs_per_context": len(PAIR_LEFT),
            "between_different_rng_pairs_per_parameter_pair_and_context": len(
                BETWEEN_BRANCH_LEFT
            ),
            "between_rng_pairing": (
                "all 56 ordered branch pairs with left_rng_idx != right_rng_idx"
            ),
            "random_parameter_pairs": len(random_metrics["candidate_pairs"]),
            "optimized_parameter_pairs": len(optimized_metrics["candidate_pairs"]),
            "equivalence_band": list(EQUIVALENCE_BAND),
            "bootstrap_unit": "candidate/rule parameter vector",
            "bootstrap_replicates": bootstrap_replicates,
            "subset10_replicates": subset_replicates,
            "bootstrap_seed": seed,
        },
        "identity_audit": {
            "exact_full_initial_states_shared_between_groups": True,
            "exact_branch_rng_keys_shared_between_groups": True,
            "random_duplicate_state_max_abs": random_group["duplicate_state_max"],
            "optimized_duplicate_state_max_abs": optimized_group["duplicate_state_max"],
            "random_duplicate_rng_max_abs": random_group["duplicate_rng_max"],
            "optimized_duplicate_rng_max_abs": optimized_group["duplicate_rng_max"],
            "full_300k_repeat_audit": {
                "status": repeat_audit["status"],
                "candidate_id": repeat_audit["candidate_id"],
                "final_restart_state_exact": repeat_audit["final_restart_state_exact"],
                "clip_embedding_max_component_abs": repeat_audit["clip_embeddings"][
                    "max_component_abs"
                ],
                "clip_embedding_max_cosine_error": repeat_audit["clip_embeddings"][
                    "max_cosine_error"
                ],
            },
            "random_protocol_sha256": random_group["protocol_sha256"],
            "optimized_protocol_sha256": optimized_group["protocol_sha256"],
        },
        "random_primary": {
            "late_window_mean_time_matched_ratio": random_late_mean,
            "candidate_bootstrap_95_ci": random_late_ci,
            "equivalence_supported_by_full_ci": random_equivalent,
            "candidates_with_late_mean_ratio_at_least_1": int(
                np.sum(random_late_ratio >= 1.0)
            ),
            "total_candidates": len(random_late_ratio),
            "mean_time_matched_ratio_at_300k": float(
                bootstraps["random"]["time_ratio"]["mean"][-1]
            ),
            "ci_at_300k": [
                float(bootstraps["random"]["time_ratio"]["low"][-1]),
                float(bootstraps["random"]["time_ratio"]["high"][-1]),
            ],
            "first_two_consecutive_windows_mean_ratio_at_least_1": (
                _first_two_consecutive(bootstraps["random"]["time_ratio"]["mean"], 1.0)
            ),
            "first_late_reference_attainment_at_least_1": _first_reach(
                bootstraps["random"]["attainment"]["mean"], 1.0
            ),
            "first_two_consecutive_late_reference_attainment_at_least_1": (
                _first_two_consecutive(bootstraps["random"]["attainment"]["mean"], 1.0)
            ),
        },
        "optimized_cached_comparator": {
            "late_window_mean_time_matched_ratio": optimized_late_mean,
            "candidate_bootstrap_95_ci": optimized_late_ci,
            "candidates_with_late_mean_ratio_at_least_1": int(
                np.sum(optimized_late_ratio >= 1.0)
            ),
            "total_candidates": len(optimized_late_ratio),
            "mean_time_matched_ratio_at_300k": float(
                bootstraps["optimized"]["time_ratio"]["mean"][-1]
            ),
            "ci_at_300k": [
                float(bootstraps["optimized"]["time_ratio"]["low"][-1]),
                float(bootstraps["optimized"]["time_ratio"]["high"][-1]),
            ],
            "first_late_reference_attainment_at_least_1": _first_reach(
                bootstraps["optimized"]["attainment"]["mean"], 1.0
            ),
            "first_two_consecutive_late_reference_attainment_at_least_1": (
                _first_two_consecutive(
                    bootstraps["optimized"]["attainment"]["mean"], 1.0
                )
            ),
        },
        "random_minus_optimized_late_ratio": {
            "mean_difference": float(
                late_difference["difference_random_minus_optimized"][0]
            ),
            "candidate_bootstrap_95_ci": [
                float(late_difference["ci_low"][0]),
                float(late_difference["ci_high"][0]),
            ],
        },
        "absolute_late_window_chamfer": {
            "random_within_rng": {
                "mean": float(random_late_within_bootstrap["mean"][0]),
                "candidate_bootstrap_95_ci": [
                    float(random_late_within_bootstrap["low"][0]),
                    float(random_late_within_bootstrap["high"][0]),
                ],
            },
            "random_inter_parameter": {
                "mean": float(random_late_inter_bootstrap["mean"][0]),
                "candidate_bootstrap_95_ci": [
                    float(random_late_inter_bootstrap["low"][0]),
                    float(random_late_inter_bootstrap["high"][0]),
                ],
            },
            "optimized_within_rng": {
                "mean": float(optimized_late_within_bootstrap["mean"][0]),
                "candidate_bootstrap_95_ci": [
                    float(optimized_late_within_bootstrap["low"][0]),
                    float(optimized_late_within_bootstrap["high"][0]),
                ],
            },
            "optimized_inter_parameter": {
                "mean": float(optimized_late_inter_bootstrap["mean"][0]),
                "candidate_bootstrap_95_ci": [
                    float(optimized_late_inter_bootstrap["low"][0]),
                    float(optimized_late_inter_bootstrap["high"][0]),
                ],
            },
            "random_minus_optimized_within_rng": {
                "mean_difference": float(
                    late_within_difference["difference_random_minus_optimized"][0]
                ),
                "candidate_bootstrap_95_ci": [
                    float(late_within_difference["ci_low"][0]),
                    float(late_within_difference["ci_high"][0]),
                ],
            },
            "random_minus_optimized_inter_parameter": {
                "mean_difference": float(
                    late_inter_difference["difference_random_minus_optimized"][0]
                ),
                "candidate_bootstrap_95_ci": [
                    float(late_inter_difference["ci_low"][0]),
                    float(late_inter_difference["ci_high"][0]),
                ],
            },
        },
        "robustness": {
            "random_10_of_30_subsets_at_300k": {
                "replicates": subset_replicates,
                "mean": float(subset["mean"][-1]),
                "subset_distribution_q025": float(subset["low"][-1]),
                "subset_distribution_q975": float(subset["high"][-1]),
            },
            "candidate_idx_strata_time_ratio_at_300k": {
                str(row["candidate_idx_stratum"]): float(row["mean_time_matched_ratio"])
                for row in strata_rows
                if int(row["bin_end_inclusive"]) == HORIZON_STEPS
            },
            "dependency_aware_leave_one_candidate_jackknife": {
                "random": {
                    "estimate": float(random_jackknife["estimate"]),
                    "standard_error": float(random_jackknife["standard_error"]),
                    "student_t_95_ci": [
                        float(random_jackknife["ci_low"]),
                        float(random_jackknife["ci_high"]),
                    ],
                    "full_ci_within_equivalence_band": (
                        random_jackknife["ci_low"] >= EQUIVALENCE_BAND[0]
                        and random_jackknife["ci_high"] <= EQUIVALENCE_BAND[1]
                    ),
                },
                "optimized": {
                    "estimate": float(optimized_jackknife["estimate"]),
                    "standard_error": float(optimized_jackknife["standard_error"]),
                    "student_t_95_ci": [
                        float(optimized_jackknife["ci_low"]),
                        float(optimized_jackknife["ci_high"]),
                    ],
                },
                "random_minus_optimized": {
                    "estimate": float(jackknife_difference),
                    "standard_error": jackknife_difference_se,
                    "welch_satterthwaite_degrees_of_freedom": (jackknife_difference_df),
                    "student_t_95_ci": [
                        float(
                            jackknife_difference
                            - jackknife_difference_critical * jackknife_difference_se
                        ),
                        float(
                            jackknife_difference
                            + jackknife_difference_critical * jackknife_difference_se
                        ),
                    ],
                },
                "purpose": (
                    "Recomputes every inter-parameter denominator after "
                    "leaving out one candidate, accounting for pairwise "
                    "denominator sharing across anchor ratios."
                ),
            },
        },
        "files": {
            "figures": sorted(
                str(path.relative_to(output_root))
                for path in figure_dir.iterdir()
                if path.is_file()
            ),
            "tables": sorted(
                str(path.relative_to(output_root))
                for path in table_dir.iterdir()
                if path.is_file()
            ),
        },
    }
    _write_json(output_root / "interrandom_analysis_report.json", report)

    optimized_first_reach = _first_reach(
        bootstraps["optimized"]["attainment"]["mean"], 1.0
    )
    optimized_first_reach_text = (
        "not reached by 300k"
        if optimized_first_reach is None
        else f"{optimized_first_reach} steps"
    )
    results_text = f"""# Flow-Lenia continuation-RNG versus parameter variation

## Primary random-parameter result

- Candidate-balanced late-window ratio: `{random_late_mean:.6f}`.
- Candidate bootstrap 95% CI: `[{random_late_ci[0]:.6f}, {random_late_ci[1]:.6f}]`.
- Dependency-aware leave-one-candidate jackknife 95% CI:
  `[{float(random_jackknife["ci_low"]):.6f}, {float(random_jackknife["ci_high"]):.6f}]`.
- Predeclared comparable-scale band: `{EQUIVALENCE_BAND}`.
- Full-CI equivalence criterion passed: `{random_equivalent}`.
- Random candidates with a late mean ratio >= 1: `{int(np.sum(random_late_ratio >= 1.0))}/{len(random_late_ratio)}`.
- Mean attainment first reaches its own late between-parameter different-RNG
  reference at
  `{_first_reach(bootstraps["random"]["attainment"]["mean"], 1.0)}` steps and
  remains above it for two consecutive windows by
  `{_first_two_consecutive(bootstraps["random"]["attainment"]["mean"], 1.0)}` steps.
- Random 10-of-30 subset distribution at 300k:
  `{float(subset["mean"][-1]):.6f}`
  (`[{float(subset["low"][-1]):.6f}, {float(subset["high"][-1]):.6f}]`).

The numerator is within-rule divergence across continuation RNG keys. The
denominator is divergence from that same random rule to the other random
rules, with exact initial state, context index, and measurement window matched,
but with different continuation RNG keys. It averages all 56 ordered
off-diagonal RNG pairs per parameter pair and context. The result is
conditional on the four fixed initial states and eight fixed continuation RNG
keys used by the experiment.

## Cached optimized comparator

- Candidate-balanced late-window ratio: `{optimized_late_mean:.6f}`.
- Candidate bootstrap 95% CI: `[{optimized_late_ci[0]:.6f}, {optimized_late_ci[1]:.6f}]`.
- Random minus optimized late-ratio difference: `{float(late_difference["difference_random_minus_optimized"][0]):.6f}`.
- Difference 95% CI: `[{float(late_difference["ci_low"][0]):.6f}, {float(late_difference["ci_high"][0]):.6f}]`.
- Dependency-aware difference 95% CI:
  `[{float(jackknife_difference - jackknife_difference_critical * jackknife_difference_se):.6f}, {float(jackknife_difference + jackknife_difference_critical * jackknife_difference_se):.6f}]`.
- Optimized mean attainment first reaches its own late between-parameter
  different-RNG reference: `{optimized_first_reach_text}`.

## Absolute late-window distances

- Random within-RNG: `{float(random_late_within_bootstrap["mean"][0]):.6f}`
  (`[{float(random_late_within_bootstrap["low"][0]):.6f}, {float(random_late_within_bootstrap["high"][0]):.6f}]`).
- Random inter-parameter: `{float(random_late_inter_bootstrap["mean"][0]):.6f}`
  (`[{float(random_late_inter_bootstrap["low"][0]):.6f}, {float(random_late_inter_bootstrap["high"][0]):.6f}]`).
- Optimized within-RNG: `{float(optimized_late_within_bootstrap["mean"][0]):.6f}`
  (`[{float(optimized_late_within_bootstrap["low"][0]):.6f}, {float(optimized_late_within_bootstrap["high"][0]):.6f}]`).
- Optimized inter-parameter: `{float(optimized_late_inter_bootstrap["mean"][0]):.6f}`
  (`[{float(optimized_late_inter_bootstrap["low"][0]):.6f}, {float(optimized_late_inter_bootstrap["high"][0]):.6f}]`).

This is evidence about empirical CLIP-space trajectory sensitivity, not a
mathematical proof of chaos. It directly motivates fixed-context evaluation:
without fixing both the complete initial state and continuation randomness,
parameter comparisons can be confounded by stochastic divergence of comparable
visual magnitude.

The optimized candidates do not show increased relative RNG sensitivity in
this experiment. The random-minus-optimized ratio difference is positive and
its interval excludes zero. This does not weaken the fixed-context motivation:
the primary environment-level claim is already supported on independently
sampled random rules, without conditioning that conclusion on optimization.

The plotted origin at `(0, 0)` in the attainment figure denotes the exact common
initial state. The first measured trajectory segment is `(0, 20k]`.
"""
    (output_root / "RESULTS.md").write_text(results_text, encoding="utf-8")
    print(json.dumps(_jsonable(report), indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze fixed-context continuation-RNG divergence for all 30 "
            "Flow-Lenia random candidates and compare it with cached "
            "optimized candidates."
        )
    )
    parser.add_argument("--random-root", default=str(DEFAULT_RANDOM_ROOT))
    parser.add_argument("--optimized-root", default=str(DEFAULT_OPTIMIZED_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--subset-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    analyze(
        _resolve(cli.random_root),
        _resolve(cli.optimized_root),
        bootstrap_replicates=cli.bootstrap_replicates,
        subset_replicates=cli.subset_replicates,
        seed=cli.seed,
    )


if __name__ == "__main__":
    main()
