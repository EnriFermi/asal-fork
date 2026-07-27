#!/usr/bin/env python3
"""Paired C2-style Flow-Lenia frustration experiment.

The free arm is the existing C2 branch continuation.  The intervention arm
starts from that arm's already-perturbed step-zero state, runs in independent
wall blocks for half of the C2 horizon, then merges back to the global domain
for the remaining half.  Both arms consume the same top-level RNG stream and
the wall arm receives the exact single global mutation event used by the free
arm at every confined step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ENV_BIN = str(Path(sys.executable).resolve().parent)
if (_ENV_BIN_PATH := Path(_ENV_BIN) / "ptxas").exists():
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from flowlenia_minibang_common import list_apf_chunks, load_config
from flowlenia_minibang_resume import (
    _apply_resume_perturbation,
    _find_snapshot,
    _read_snapshot,
    _scalar,
)


PROTOCOL_VERSION = "flowlenia-c5-c2-paired-walls-half-v2"
WALL_OUTPUT_DIRNAME = "walls_to_free_rng_matched"
LEGACY_PROTOCOL_VERSION = "flowlenia-c5-c2-paired-walls-half-v1"
SIMULATION_CODE_FILES = (
    "scripts/flowlenia_c5_branch_frustration.py",
    "scripts/flowlenia_minibang_common.py",
    "scripts/flowlenia_minibang_resume.py",
    "scripts/flowlenia_minibang_resume_batch.py",
    "scripts/flowlenia_minibang_simulate.py",
    "scripts/evaluate_frustration_history_dependence.py",
    "scripts/flowlenia_c5_full_length_delta_h_one.py",
    "scripts/paper_check_frustration_batch_eval.py",
    "scripts/simulate_save_apf.py",
    "scripts/util.py",
    "substrates/__init__.py",
    "substrates/lenia_flow/lenia_flow.py",
    "substrates/lenia_flow/reintegration_tracking.py",
    "substrates/lenia_flow/utils.py",
)
DEFAULT_PAPER_CONFIG = Path(
    "experiments/paper_suite/"
    "config_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper.yaml"
)
DEFAULT_C2_ROOT = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "c2_branching"
)
DEFAULT_TRIAL_ROOT = Path(
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "frustration_simulation/trial_artifacts"
)
DEFAULT_OUTPUT_ROOT = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "flow_lenia/c5_c2_paired_walls_half"
)

HORIZON_STEPS = 20_000
WALL_STEPS = 10_000
SPLIT_N = 3
WALL_PAD = 5
EXPECTED_RUNS = 10
EXPECTED_RANDOM_PER_RUN = 3
EXPECTED_POINTS_PER_CANDIDATE = 15
EXPECTED_BRANCHES_PER_POINT = 3
SIMULATION_BATCH_SIZE = 30
OPTIMIZER_NATIVE_BATCH_SIZE = 32
JIT_MICROBATCH = 50
EXPECTED_MUTATION_PATCH_SIZE = 40
EXPECTED_MUTATION_PROBABILITY = 0.05
EXPECTED_MUTATION_SCALE = 1.0
EXPECTED_PLAN_ROWS = (
    EXPECTED_RUNS
    * (1 + EXPECTED_RANDOM_PER_RUN)
    * EXPECTED_POINTS_PER_CANDIDATE
    * EXPECTED_BRANCHES_PER_POINT
)
SIMULATION_ID_FIELDS = (
    "run_idx",
    "candidate_kind",
    "candidate_idx",
    "condition",
    "pair_id",
    "point_id",
    "step",
    "branch_id",
    "branch_seed",
)
PLAN_FIELD_ORDER = (
    "row_id",
    "run_idx",
    "trial_idx",
    "candidate_kind",
    "candidate_idx",
    "candidate_label",
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
    "horizon_steps",
    "wall_steps",
    "perturb_a_std",
    "perturb_p_std",
    "perturb_lagrangian_xy_std",
    "free_branch_dir",
    "free_provenance",
    "walls_branch_dir",
    "selection_reference_traj_id",
    "selection_semantics",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return value.resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def _simulation_code_fingerprint() -> dict[str, Any]:
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


def _simulation_config_fingerprint(path: Path) -> dict[str, Any]:
    import util

    _cfg, flat = load_config(path)
    args = SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))
    payload = {
        "substrate": str(args.substrate),
        "flow_lenia_kwargs": util.flow_lenia_kwargs_from_args(args),
        "debug_return_F": True,
    }
    return {
        "payload": payload,
        "sha256": _sha256_bytes(
            _stable_json(payload).encode("utf-8")
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for preferred in PLAN_FIELD_ORDER:
        if any(preferred in row for row in rows):
            fields.append(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plan_identity_hash(rows: Sequence[dict[str, Any]]) -> str:
    identity = [
        {key: str(row.get(key, "")) for key in PLAN_FIELD_ORDER}
        for row in rows
    ]
    return _sha256_bytes(_stable_json(identity).encode("utf-8"))


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    return float(value)


def _candidate_id(run_idx: int, kind: str, candidate_idx: int) -> str:
    if kind == "optimized":
        return f"run_{run_idx:03d}_optimized"
    return f"run_{run_idx:03d}_random_{candidate_idx:03d}"


def _run_idx_from_traj_id(traj_id: str) -> int:
    match = re.search(r"_run_(\d{3})_seed_000$", traj_id)
    if match is None:
        raise ValueError(f"Cannot parse run index from C2 traj_id={traj_id!r}")
    return int(match.group(1))


def _point_id(row: dict[str, str]) -> int:
    raw = row.get("point_id")
    if raw not in (None, ""):
        return _as_int(raw)
    condition = str(row["condition"])
    local = _as_int(row["pair_id"])
    return {"high": 0, "mid": 5, "low": 10}[condition] + local


def _load_trial_records(trial_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_path in sorted(trial_root.glob("trial_*/job_config.yaml")):
        cfg = OmegaConf.load(config_path)
        run_idx = int(cfg.meta.optimized_run_idx)
        kind = str(cfg.meta.candidate_kind)
        candidate_idx = int(cfg.meta.candidate_idx)
        source_apf_dir = _resolve(str(cfg.job.control_a_reference_apf_dir))
        source_traj_dir = source_apf_dir.parent
        source_config_path = source_traj_dir / "config.yaml"
        params_path = source_traj_dir / "params.npy"
        if (
            not source_apf_dir.exists()
            or not source_config_path.exists()
            or not params_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing exact C1 source for {config_path}: "
                f"apf={source_apf_dir}, config={source_config_path}, "
                f"params={params_path}"
            )
        simulation_config = _simulation_config_fingerprint(
            source_config_path
        )
        records.append(
            {
                "trial_idx": int(cfg.meta.trial_idx),
                "run_idx": run_idx,
                "candidate_kind": kind,
                "candidate_idx": candidate_idx,
                "candidate_label": str(cfg.meta.candidate_label),
                "candidate_id": _candidate_id(run_idx, kind, candidate_idx),
                "source_traj_id": source_traj_dir.name,
                "source_traj_dir": source_traj_dir,
                "source_config_path": source_config_path,
                "source_config_sha256": _sha256_file(source_config_path),
                "source_simulation_config_sha256": simulation_config[
                    "sha256"
                ],
                "source_simulation_config": simulation_config["payload"],
                "params_path": params_path,
                "params_sha256": _sha256_file(params_path),
                "trial_config": config_path.resolve(),
                "trial_config_sha256": _sha256_file(config_path),
            }
        )
    if len(records) != EXPECTED_RUNS * (1 + EXPECTED_RANDOM_PER_RUN):
        raise RuntimeError(
            f"Expected 40 C5 candidates, found {len(records)} under {trial_root}"
        )
    simulation_configs = {
        record["source_simulation_config_sha256"] for record in records
    }
    if len(simulation_configs) != 1:
        raise RuntimeError(
            "C5 candidates do not share one simulation-critical substrate "
            f"configuration: {sorted(simulation_configs)}"
        )
    counts = Counter((r["run_idx"], r["candidate_kind"]) for r in records)
    for run_idx in range(EXPECTED_RUNS):
        if counts[(run_idx, "optimized")] != 1:
            raise RuntimeError(f"run_{run_idx:03d} does not have one optimized trial")
        if counts[(run_idx, "random")] != EXPECTED_RANDOM_PER_RUN:
            raise RuntimeError(
                f"run_{run_idx:03d} does not have {EXPECTED_RANDOM_PER_RUN} random trials"
            )
    return records


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    c2_root = _resolve(args.c2_root)
    trial_root = _resolve(args.trial_root)
    output_root = _resolve(args.output_root)
    c2_plan_path = c2_root / "branch_plan.csv"
    c2_meta_path = c2_root / "branch_plan_meta.json"
    if not c2_plan_path.exists() or not c2_meta_path.exists():
        raise FileNotFoundError(f"Missing authoritative C2 plan under {c2_root}")

    c2_rows = _read_csv(c2_plan_path)
    if len(c2_rows) != 450:
        raise RuntimeError(f"Expected 450 C2 plan rows, found {len(c2_rows)}")
    c2_meta = json.loads(c2_meta_path.read_text())
    expected_c2 = {
        "horizon_steps": HORIZON_STEPS,
        "branches_per_time": EXPECTED_BRANCHES_PER_POINT,
        "n_high": 5,
        "n_mid": 5,
        "n_low": 5,
        "perturb_a_std": 0.02,
        "perturb_p_std": 0.02,
        "perturb_lagrangian_xy_std": 1.0,
    }
    actual_c2 = c2_meta.get("branching_config", {})
    mismatches = {
        key: {"expected": expected, "actual": actual_c2.get(key)}
        for key, expected in expected_c2.items()
        if actual_c2.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Authoritative C2 protocol mismatch: {mismatches}")

    c2_by_run: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in c2_rows:
        c2_by_run[_run_idx_from_traj_id(str(row["traj_id"]))].append(row)
    for run_idx, rows in c2_by_run.items():
        if len(rows) != EXPECTED_POINTS_PER_CANDIDATE * EXPECTED_BRANCHES_PER_POINT:
            raise RuntimeError(f"run_{run_idx:03d} has {len(rows)} C2 rows")

    trials = _load_trial_records(trial_root)
    plan_rows: list[dict[str, Any]] = []
    row_id = 0
    for trial in sorted(
        trials,
        key=lambda r: (
            r["run_idx"],
            0 if r["candidate_kind"] == "optimized" else 1,
            r["candidate_idx"],
        ),
    ):
        run_idx = int(trial["run_idx"])
        for source_row in c2_by_run[run_idx]:
            condition = str(source_row["condition"])
            pair_id = _as_int(source_row["pair_id"])
            point_id = _point_id(source_row)
            branch_id = _as_int(source_row["branch_id"])
            step = _as_int(source_row["step"])
            point_tag = (
                f"{condition}_{point_id:03d}_step_{step:06d}/"
                f"branch_{branch_id:02d}"
            )
            if trial["candidate_kind"] == "optimized":
                free_dir = _resolve(source_row["branch_dir"])
                free_provenance = "reused_authoritative_c2"
            else:
                free_dir = (
                    output_root
                    / "free_random"
                    / str(trial["candidate_id"])
                    / point_tag
                )
                free_provenance = "generated_exact_c2_protocol"
            walls_dir = (
                output_root
                / WALL_OUTPUT_DIRNAME
                / str(trial["candidate_id"])
                / point_tag
            )
            plan_rows.append(
                {
                    "row_id": row_id,
                    "run_idx": run_idx,
                    "trial_idx": int(trial["trial_idx"]),
                    "candidate_kind": str(trial["candidate_kind"]),
                    "candidate_idx": int(trial["candidate_idx"]),
                    "candidate_label": str(trial["candidate_label"]),
                    "candidate_id": str(trial["candidate_id"]),
                    "source_traj_id": str(trial["source_traj_id"]),
                    "source_traj_dir": str(trial["source_traj_dir"]),
                    "source_config_path": str(trial["source_config_path"]),
                    "source_config_sha256": str(
                        trial["source_config_sha256"]
                    ),
                    "source_simulation_config_sha256": str(
                        trial["source_simulation_config_sha256"]
                    ),
                    "params_path": str(trial["params_path"]),
                    "params_sha256": str(trial["params_sha256"]),
                    "condition": condition,
                    "pair_id": pair_id,
                    "point_id": point_id,
                    "window_index": _as_int(source_row["window_index"]),
                    "step": step,
                    "delta_h": _as_float(source_row["delta_h"]),
                    "branch_id": branch_id,
                    "branch_seed": _as_int(source_row["branch_seed"]),
                    "horizon_steps": HORIZON_STEPS,
                    "wall_steps": WALL_STEPS,
                    "perturb_a_std": _as_float(source_row["perturb_a_std"]),
                    "perturb_p_std": _as_float(source_row["perturb_p_std"]),
                    "perturb_lagrangian_xy_std": _as_float(
                        source_row["perturb_lagrangian_xy_std"]
                    ),
                    "free_branch_dir": str(free_dir),
                    "free_provenance": free_provenance,
                    "walls_branch_dir": str(walls_dir),
                    "selection_reference_traj_id": str(source_row["traj_id"]),
                    "selection_semantics": (
                        "Absolute step and C2 high/mid/low stratum selected on the "
                        "matched optimized trajectory; the same step and branch seed "
                        "are reused for all three random controls in that run."
                    ),
                }
            )
            row_id += 1
    if len(plan_rows) != EXPECTED_PLAN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PLAN_ROWS} paired rows, found {len(plan_rows)}"
        )

    plan_hash = _plan_identity_hash(plan_rows)
    code_fingerprint = _simulation_code_fingerprint()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": plan_hash,
        "simulation_code_bundle_sha256": code_fingerprint["bundle_sha256"],
        "simulation_code_files": code_fingerprint["files"],
        "authoritative_c2_plan": str(c2_plan_path),
        "authoritative_c2_plan_sha256": _sha256_file(c2_plan_path),
        "authoritative_c2_meta": str(c2_meta_path),
        "authoritative_c2_meta_sha256": _sha256_file(c2_meta_path),
        "paper_config": str(_resolve(args.paper_config)),
        "paper_config_sha256": _sha256_file(_resolve(args.paper_config)),
        "trial_root": str(trial_root),
        "trial_config_sha256": {
            str(record["trial_idx"]): record["trial_config_sha256"]
            for record in trials
        },
        "simulation_config_sha256": trials[0][
            "source_simulation_config_sha256"
        ],
        "simulation_config": trials[0]["source_simulation_config"],
        "grid_size": int(
            trials[0]["source_simulation_config"]["flow_lenia_kwargs"][
                "grid_size"
            ]
        ),
        "n_runs": EXPECTED_RUNS,
        "n_random_per_run": EXPECTED_RANDOM_PER_RUN,
        "n_candidates": len(trials),
        "n_points_per_candidate": EXPECTED_POINTS_PER_CANDIDATE,
        "n_branches_per_point": EXPECTED_BRANCHES_PER_POINT,
        "n_plan_rows": len(plan_rows),
        "outer_branch_batch_size": SIMULATION_BATCH_SIZE,
        "optimizer_native_batch_size": OPTIMIZER_NATIVE_BATCH_SIZE,
        "jit_microbatch_steps": JIT_MICROBATCH,
        "horizon_steps": HORIZON_STEPS,
        "walls_active_relative_steps": [0, WALL_STEPS],
        "walls_removed_relative_step": WALL_STEPS,
        "free_after_walls_steps": HORIZON_STEPS - WALL_STEPS,
        "grid_split": SPLIT_N,
        "wall_pad": WALL_PAD,
        "pairing": {
            "initial_state": (
                "walls arm repeats the authoritative C2 initialization code from "
                "resume_metadata: the same source APF float32 reconstruction, NumPy "
                "Gaussian perturbation, and fold_in(branch_seed). Its serialized "
                "relative-zero A/P/F must equal the free branch exactly."
            ),
            "params": "same params.npy SHA-256 in both arms",
            "rng": (
                "same folded branch RNG at relative step 0 and the same top-level "
                "split stream; during confinement one mutation event is reproduced "
                "from each selected global lane key and partitioned across wall cores"
            ),
            "selection": (
                "the optimized C2-selected absolute steps and branch seeds are "
                "reused for its three matched random controls"
            ),
        },
        "passive_tracker_policy": (
            "The full-tracker state/RNG runner is required to reproduce the free C2 "
            "arm exactly in preflight. Wall simulations then use the identical "
            "state kernels and RNG schedule without lagrangian tracking because the "
            "tracker is a passive observer and C5 uses A/P CLIP and field metrics."
        ),
        "wall_stochastic_policy": (
            "Block dynamics disable local mutation. Each confined step reproduces "
            "the free arm's single global Flow-Lenia mutation delta from the exact "
            "selected lane key on the 128x128 domain, then partitions that delta "
            "across the 3x3 block cores. Stochastic reintegration receives the exact "
            "native global PRNGKey(42) Gumbels at corresponding core coordinates. "
            "Wall padding is hard-zeroed after every transition. Volcano and food "
            "stochasticity must be off."
        ),
        "primary_metric_definition": (
            "post-release paired CLIP chamfer(walls_to_free_i, free_i), median "
            "over three branch seeds, minus the median pairwise free-branch CLIP "
            "chamfer at the same point"
        ),
        "secondary_metrics": [
            "post-release within-ensemble wall spread minus free spread",
            "full-horizon paired effect minus free spread",
            "multiscale A/P field analogues",
            "condition-stratified high/mid/low summaries",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "paired_plan.csv"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old.get("plan_sha256") != plan_hash:
            if old.get("protocol_version") != LEGACY_PROTOCOL_VERSION:
                raise RuntimeError(
                    f"Refusing to replace incompatible C5 plan under {output_root}; "
                    f"old={old.get('plan_sha256')} new={plan_hash}"
                )
            old_rows = _read_csv(plan_path)
            stable_fields = [
                key
                for key in PLAN_FIELD_ORDER
                if key
                not in {
                    "source_config_path",
                    "source_config_sha256",
                    "source_simulation_config_sha256",
                    "walls_branch_dir",
                }
            ]
            same_free_plan = (
                len(old_rows) == len(plan_rows)
                and all(
                    all(
                        str(old_row.get(key, ""))
                        == str(new_row.get(key, ""))
                        for key in stable_fields
                    )
                    for old_row, new_row in zip(
                        old_rows,
                        plan_rows,
                        strict=True,
                    )
                )
            )
            if not same_free_plan:
                raise RuntimeError(
                    "Legacy v1 plan differs outside the intentionally migrated "
                    "wall path/provenance fields"
                )
            archive = output_root / "legacy_v1_metadata"
            archive.mkdir(parents=True, exist_ok=True)
            for source in (
                plan_path,
                protocol_path,
                output_root / "plan_summary.json",
                output_root / "preflight_sham_exactness.json",
                output_root / "preflight_summary.json",
                output_root / "walls_progress.json",
                output_root / "walls_summary.json",
                output_root / "protocol_audit.json",
            ):
                if source.exists():
                    destination = archive / source.name
                    if not destination.exists():
                        shutil.copy2(source, destination)
        elif (
            old.get("protocol_version") == PROTOCOL_VERSION
            and old.get("simulation_code_bundle_sha256")
            != code_fingerprint["bundle_sha256"]
            and any(
                (output_root / WALL_OUTPUT_DIRNAME).rglob(
                    "wall_metadata.json"
                )
            )
        ):
            raise RuntimeError(
                "Simulation code changed after v2 wall outputs were created; "
                "refusing to rewrite their protocol fingerprint"
            )
    _write_csv(plan_path, plan_rows)
    _write_json(protocol_path, protocol)
    _write_json(
        output_root / "plan_summary.json",
        {
            "status": "complete",
            "plan": str(plan_path),
            "protocol": str(protocol_path),
            "plan_sha256": plan_hash,
            "n_rows": len(plan_rows),
            "candidate_counts": dict(Counter(r["candidate_kind"] for r in plan_rows)),
        },
    )
    print(
        json.dumps(
            {"status": "complete", "n_rows": len(plan_rows), "plan_sha256": plan_hash},
            indent=2,
        ),
        flush=True,
    )
    return protocol


def _load_plan(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output_root = _resolve(args.output_root)
    plan_path = output_root / "paired_plan.csv"
    protocol_path = output_root / "protocol.json"
    if not plan_path.exists() or not protocol_path.exists():
        build_plan(args)
    rows = _read_csv(plan_path)
    protocol = json.loads(protocol_path.read_text())
    found_hash = _plan_identity_hash(rows)
    if found_hash != protocol["plan_sha256"]:
        raise RuntimeError("paired_plan.csv does not match protocol.json")
    current_code = _simulation_code_fingerprint()["bundle_sha256"]
    if protocol.get("simulation_code_bundle_sha256") != current_code:
        raise RuntimeError(
            "Simulation code fingerprint differs from protocol.json; run the "
            "plan phase before creating or reusing v2 wall outputs"
        )
    return rows, protocol


def _filter_rows(
    rows: Sequence[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    selected = list(rows)
    if args.run_indices:
        wanted = {int(value) for value in args.run_indices.split(",")}
        selected = [row for row in selected if _as_int(row["run_idx"]) in wanted]
    if args.candidate_kinds != "all":
        wanted_kinds = {value.strip() for value in args.candidate_kinds.split(",")}
        selected = [row for row in selected if row["candidate_kind"] in wanted_kinds]
    if args.candidate_ids:
        wanted_ids = {value.strip() for value in args.candidate_ids.split(",")}
        selected = [row for row in selected if row["candidate_id"] in wanted_ids]
    if args.conditions != "all":
        wanted_conditions = {value.strip() for value in args.conditions.split(",")}
        selected = [row for row in selected if row["condition"] in wanted_conditions]
    if args.max_rows is not None:
        selected = selected[: int(args.max_rows)]
    if not selected:
        raise ValueError("Row filters selected no C5 jobs")
    return selected


def _require_simulation_batch_size(args: argparse.Namespace) -> None:
    if int(args.batch_size) != SIMULATION_BATCH_SIZE:
        raise RuntimeError(
            "C5 simulation batch topology is frozen by the authoritative C2 "
            f"replay protocol: --batch-size must be {SIMULATION_BATCH_SIZE}, "
            f"found {args.batch_size}"
        )


def _branch_arrays(
    branch_dir: Path,
    *,
    keys: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    chunks = list_apf_chunks(branch_dir / "apf_logs")
    if not chunks:
        raise FileNotFoundError(f"No APF chunks under {branch_dir}")
    selected_keys = None if keys is None else set(keys) | {"steps"}
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for path, _s0, _s1, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if selected_keys is not None and key not in selected_keys:
                    continue
                arr = np.asarray(data[key])
                if arr.ndim > 0 and arr.shape[0] == np.asarray(data["steps"]).shape[0]:
                    parts[key].append(arr)
    out = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    order = np.argsort(out["steps"], kind="stable")
    return {key: value[order] for key, value in out.items()}


def _expected_free_metadata(row: dict[str, str]) -> dict[str, Any]:
    return {
        "source_traj_dir": str(_resolve(row["source_traj_dir"])),
        "start_step": _as_int(row["step"]),
        "end_step": _as_int(row["step"]) + HORIZON_STEPS,
        "original_batch_size": OPTIMIZER_NATIVE_BATCH_SIZE,
        "snapshot_interval": JIT_MICROBATCH,
        "jit_microbatch": JIT_MICROBATCH,
        "branch_seed": _as_int(row["branch_seed"]),
        "perturb_a_std": _as_float(row["perturb_a_std"]),
        "perturb_p_std": _as_float(row["perturb_p_std"]),
        "perturb_lagrangian_xy_std": _as_float(
            row["perturb_lagrangian_xy_std"]
        ),
        "output_max_snapshots_per_chunk": 8,
        "batched_resume": True,
    }


def _free_output_audit(row: dict[str, str]) -> dict[str, Any]:
    branch_dir = _resolve(row["free_branch_dir"])
    metadata_path = branch_dir / "resume_metadata.json"
    result = {
        "branch_dir": str(branch_dir),
        "ready": False,
        "reason": "",
    }
    if not metadata_path.exists():
        result["reason"] = "missing resume_metadata.json"
        return result
    metadata = json.loads(metadata_path.read_text())
    expected = _expected_free_metadata(row)
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        result["reason"] = f"metadata mismatch: {mismatches}"
        return result
    try:
        arrays = _branch_arrays(branch_dir)
    except Exception as exc:
        result["reason"] = f"APF load failed: {exc}"
        return result
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    if steps.size != 8:
        result["reason"] = f"expected 8 retained C2 frames, found {steps.size}"
        return result
    if steps[0] != _as_int(row["step"]) or steps[-1] != _as_int(row["step"]) + HORIZON_STEPS:
        result["reason"] = f"wrong step endpoints: {steps.tolist()}"
        return result
    required = {
        "A",
        "P",
        "F",
        "lagrangian_xy",
        "lagrangian_c",
        "resume_batch_rng_key",
        "resume_batch_size",
        "resume_batch_index",
        "resume_jit_microbatch",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        result["reason"] = f"missing arrays: {missing}"
        return result
    native_batch_sizes = np.asarray(
        arrays["resume_batch_size"],
        dtype=np.int64,
    ).reshape(-1)
    native_batch_indices = np.asarray(
        arrays["resume_batch_index"],
        dtype=np.int64,
    ).reshape(-1)
    native_jit_microbatches = np.asarray(
        arrays["resume_jit_microbatch"],
        dtype=np.int64,
    ).reshape(-1)
    metadata_batch_index = int(metadata.get("original_batch_index", -1))
    if (
        native_batch_sizes.size != steps.size
        or not np.all(
            native_batch_sizes == OPTIMIZER_NATIVE_BATCH_SIZE
        )
        or native_batch_indices.size != steps.size
        or not np.all(native_batch_indices == metadata_batch_index)
        or metadata_batch_index < 0
        or metadata_batch_index >= OPTIMIZER_NATIVE_BATCH_SIZE
        or native_jit_microbatches.size != steps.size
        or not np.all(native_jit_microbatches == JIT_MICROBATCH)
    ):
        result["reason"] = (
            "optimizer-native batch/JIT provenance mismatch: "
            f"metadata_index={metadata_batch_index}, "
            f"sizes={np.unique(native_batch_sizes).tolist()}, "
            f"indices={np.unique(native_batch_indices).tolist()}, "
            f"jit={np.unique(native_jit_microbatches).tolist()}"
        )
        return result
    params_path = branch_dir / "params.npy"
    if not params_path.exists():
        result["reason"] = "missing params.npy"
        return result
    params_hash = _sha256_file(params_path)
    if params_hash != row["params_sha256"]:
        result["reason"] = (
            f"params hash mismatch: free={params_hash} source={row['params_sha256']}"
        )
        return result
    config_path = branch_dir / "config.yaml"
    if not config_path.exists():
        result["reason"] = "missing config.yaml"
        return result
    config_hash = _sha256_file(config_path)
    if config_hash != row["source_config_sha256"]:
        result["reason"] = (
            f"config hash mismatch: free={config_hash} "
            f"source={row['source_config_sha256']}"
        )
        return result
    chunks = list_apf_chunks(branch_dir / "apf_logs")
    if len(chunks) != 1:
        result["reason"] = f"expected one APF chunk, found {len(chunks)}"
        return result
    apf_path = chunks[0][0]
    result.update(
        {
            "ready": True,
            "reason": "",
            "steps": steps.tolist(),
            "params_sha256": params_hash,
            "config_sha256": config_hash,
            "apf_sha256": _sha256_file(apf_path),
            "start_state_sha256": {
                key: _sha256_array(arrays[key][0])
                for key in ("A", "P", "F", "lagrangian_xy", "lagrangian_c")
            },
            "start_rng_sha256": _sha256_array(arrays["resume_batch_rng_key"][0]),
            "optimizer_native_batch_size": int(
                native_batch_sizes[0]
            ),
            "optimizer_native_batch_index": int(
                native_batch_indices[0]
            ),
            "optimizer_native_batch_index_exact": True,
            "jit_microbatch_steps": int(
                native_jit_microbatches[0]
            ),
        }
    )
    return result


def generate_free_random(args: argparse.Namespace) -> dict[str, Any]:
    _require_simulation_batch_size(args)
    rows, protocol = _load_plan(args)
    selected = _filter_rows(rows, args)
    audits = [_free_output_audit(row) for row in selected]
    missing_rows = [
        row for row, audit in zip(selected, audits, strict=True) if not audit["ready"]
    ]
    invalid_reused = [
        (row, audit)
        for row, audit in zip(selected, audits, strict=True)
        if not audit["ready"] and row["free_provenance"] == "reused_authoritative_c2"
    ]
    if invalid_reused:
        raise RuntimeError(
            "Authoritative optimized C2 branch failed validation: "
            + "; ".join(
                f"row={row['row_id']} {audit['reason']}"
                for row, audit in invalid_reused[:5]
            )
        )
    jobs = []
    for row in missing_rows:
        branch_dir = _resolve(row["free_branch_dir"])
        if branch_dir.exists() and any(branch_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite incomplete free branch without manual audit: {branch_dir}"
            )
        jobs.append(
            {
                "source_traj_dir": row["source_traj_dir"],
                "step": _as_int(row["step"]),
                "additional_steps": HORIZON_STEPS,
                "output_dir": str(branch_dir),
                "branch_seed": _as_int(row["branch_seed"]),
                "perturb_a_std": _as_float(row["perturb_a_std"]),
                "perturb_p_std": _as_float(row["perturb_p_std"]),
                "perturb_lagrangian_xy_std": _as_float(
                    row["perturb_lagrangian_xy_std"]
                ),
                "output_max_snapshots_per_chunk": 8,
                "ignore_output_paths_in_simulation_signature": True,
            }
        )
    output_root = _resolve(args.output_root)
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
        command = [
            sys.executable,
            str(_REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"),
            "--jobs-json",
            str(jobs_path),
            "--batch-size",
            str(int(args.batch_size)),
        ]
        print(
            f"[free] generating {len(jobs)} missing random branches "
            f"with batch_size={args.batch_size}",
            flush=True,
        )
        subprocess.run(command, cwd=_REPO_ROOT, check=True)
    final_audits = [_free_output_audit(row) for row in selected]
    failed = [audit for audit in final_audits if not audit["ready"]]
    if failed:
        raise RuntimeError(f"{len(failed)} free branches failed validation: {failed[:3]}")
    summary = {
        "status": "complete",
        "plan_sha256": protocol["plan_sha256"],
        "n_selected": len(selected),
        "n_reused": len(selected) - len(jobs),
        "n_generated": len(jobs),
        "jobs": str(jobs_path),
    }
    _write_json(output_root / "free_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def free_cache_equivalence_audit(
    args: argparse.Namespace,
) -> dict[str, Any]:
    _require_simulation_batch_size(args)
    rows, protocol = _load_plan(args)
    random_rows = [
        row for row in rows if row["candidate_kind"] == "random"
    ]
    dated_rows = []
    for row in random_rows:
        chunks = list_apf_chunks(
            _resolve(row["free_branch_dir"]) / "apf_logs"
        )
        if len(chunks) != 1:
            continue
        dated_rows.append((chunks[0][0].stat().st_mtime_ns, row))
    dated_rows.sort(key=lambda item: (item[0], _as_int(item[1]["row_id"])))
    selected = [row for _mtime, row in dated_rows[:SIMULATION_BATCH_SIZE]]
    if len(selected) != SIMULATION_BATCH_SIZE:
        raise RuntimeError(
            "Free cache audit requires 30 complete historical random branches"
        )

    output_root = _resolve(args.output_root)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix="free_cache_equivalence_",
            dir=output_root,
        )
    )
    jobs = []
    for row in selected:
        jobs.append(
            {
                "source_traj_dir": row["source_traj_dir"],
                "step": _as_int(row["step"]),
                "additional_steps": HORIZON_STEPS,
                "output_dir": str(
                    temp_root / f"row_{_as_int(row['row_id']):04d}"
                ),
                "branch_seed": _as_int(row["branch_seed"]),
                "perturb_a_std": _as_float(row["perturb_a_std"]),
                "perturb_p_std": _as_float(row["perturb_p_std"]),
                "perturb_lagrangian_xy_std": _as_float(
                    row["perturb_lagrangian_xy_std"]
                ),
                "output_max_snapshots_per_chunk": 8,
                "ignore_output_paths_in_simulation_signature": True,
            }
        )
    jobs_path = temp_root / "jobs.json"
    _write_json(jobs_path, {"jobs": jobs})
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"),
        "--jobs-json",
        str(jobs_path),
        "--batch-size",
        str(SIMULATION_BATCH_SIZE),
    ]
    started = time.monotonic()
    try:
        subprocess.run(command, cwd=_REPO_ROOT, check=True)
        comparisons = []
        all_exact = True
        for row, job in zip(selected, jobs, strict=True):
            existing_dir = _resolve(row["free_branch_dir"])
            replay_dir = Path(job["output_dir"])
            existing = _branch_arrays(existing_dir)
            replay = _branch_arrays(replay_dir)
            keys_exact = set(existing) == set(replay)
            field_exact = {
                key: bool(np.array_equal(existing[key], replay[key]))
                for key in sorted(set(existing).intersection(replay))
            }
            exact = keys_exact and all(field_exact.values())
            all_exact = all_exact and exact
            existing_chunk = list_apf_chunks(
                existing_dir / "apf_logs"
            )[0][0]
            replay_chunk = list_apf_chunks(
                replay_dir / "apf_logs"
            )[0][0]
            comparisons.append(
                {
                    "row_id": _as_int(row["row_id"]),
                    "all_exact": exact,
                    "keys_exact": keys_exact,
                    "field_exact": field_exact,
                    "existing_apf_sha256": _sha256_file(existing_chunk),
                    "replay_apf_sha256": _sha256_file(replay_chunk),
                    "existing_mtime_ns": existing_chunk.stat().st_mtime_ns,
                }
            )
        report = {
            "status": "passed" if all_exact else "failed",
            "protocol_version": PROTOCOL_VERSION,
            "plan_sha256": protocol["plan_sha256"],
            "batch_size": SIMULATION_BATCH_SIZE,
            "selection": "30 oldest complete random free branches",
            "n_compared": len(comparisons),
            "all_exact": all_exact,
            "runner": str(
                _REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"
            ),
            "runner_sha256": _sha256_file(
                _REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"
            ),
            "elapsed_seconds": time.monotonic() - started,
            "comparisons": comparisons,
        }
        _write_json(
            output_root / "free_cache_equivalence_audit.json",
            report,
        )
        if not all_exact:
            raise RuntimeError(
                f"Free cache equivalence failed; debug data kept at {temp_root}"
            )
    except Exception:
        raise
    else:
        shutil.rmtree(temp_root)
    print(json.dumps(report, indent=2), flush=True)
    return report


@dataclass
class SimulationGeometry:
    grid_size: int
    split_n: int
    wall_pad: int
    block_size: int
    padded_grid_size: int
    crop_before: int
    crop_after: int
    n_blocks: int


@dataclass(frozen=True)
class GlobalMutationSpec:
    enabled: bool
    grid_size: int
    channels: int
    patch_size: int
    probability: float
    scale: float


def _geometry(grid_size: int) -> SimulationGeometry:
    block_size = (int(grid_size) + SPLIT_N - 1) // SPLIT_N
    if (block_size + 2 * WALL_PAD) % 2 != 0:
        block_size += 1
    padded_grid_size = block_size * SPLIT_N
    padding = padded_grid_size - int(grid_size)
    return SimulationGeometry(
        grid_size=int(grid_size),
        split_n=SPLIT_N,
        wall_pad=WALL_PAD,
        block_size=block_size,
        padded_grid_size=padded_grid_size,
        crop_before=padding // 2,
        crop_after=padding - padding // 2,
        n_blocks=SPLIT_N * SPLIT_N,
    )


def _partition_global_field(
    value: Any,
    *,
    geometry: SimulationGeometry,
) -> Any:
    import jax.numpy as jnp

    arr = jnp.asarray(value)
    suffix = arr.shape[2:]
    padded = jnp.pad(
        arr,
        (
            (geometry.crop_before, geometry.crop_after),
            (geometry.crop_before, geometry.crop_after),
        )
        + ((0, 0),) * len(suffix),
    )
    grid = padded.reshape(
        (
            geometry.split_n,
            geometry.block_size,
            geometry.split_n,
            geometry.block_size,
        )
        + suffix
    )
    axes = (0, 2, 1, 3) + tuple(range(4, grid.ndim))
    cores = jnp.transpose(grid, axes).reshape(
        (geometry.n_blocks, geometry.block_size, geometry.block_size) + suffix
    )
    return jnp.pad(
        cores,
        (
            (0, 0),
            (geometry.wall_pad, geometry.wall_pad),
            (geometry.wall_pad, geometry.wall_pad),
        )
        + ((0, 0),) * len(suffix),
    )


def _hard_wall_core_mask(geometry: SimulationGeometry) -> np.ndarray:
    size = geometry.block_size + 2 * geometry.wall_pad
    mask = np.zeros(
        (geometry.n_blocks, size, size),
        dtype=bool,
    )
    for block_idx in range(geometry.n_blocks):
        row = block_idx // geometry.split_n
        col = block_idx % geometry.split_n
        block_y0 = row * geometry.block_size
        block_x0 = col * geometry.block_size
        overlap_y0 = max(block_y0, geometry.crop_before)
        overlap_x0 = max(block_x0, geometry.crop_before)
        overlap_y1 = min(
            block_y0 + geometry.block_size,
            geometry.crop_before + geometry.grid_size,
        )
        overlap_x1 = min(
            block_x0 + geometry.block_size,
            geometry.crop_before + geometry.grid_size,
        )
        if overlap_y1 <= overlap_y0 or overlap_x1 <= overlap_x0:
            continue
        local_y0 = overlap_y0 - block_y0
        local_x0 = overlap_x0 - block_x0
        local_y1 = overlap_y1 - block_y0
        local_x1 = overlap_x1 - block_x0
        mask[
            block_idx,
            geometry.wall_pad + local_y0 : geometry.wall_pad + local_y1,
            geometry.wall_pad + local_x0 : geometry.wall_pad + local_x1,
        ] = True
    return mask


def _global_rt_gumbel(
    *,
    grid_size: int,
    dd: int,
    dtype: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    choices = (2 * int(dd) + 1) ** 2
    return jax.random.gumbel(
        jax.random.PRNGKey(42),
        (choices, int(grid_size), int(grid_size), 1),
        dtype=jnp.dtype(dtype),
    )


def _partition_rt_gumbel(
    global_gumbel: Any,
    *,
    geometry: SimulationGeometry,
) -> Any:
    import jax.numpy as jnp

    spatial_first = jnp.transpose(global_gumbel, (1, 2, 0, 3))
    blocks = _partition_global_field(
        spatial_first,
        geometry=geometry,
    )
    return jnp.transpose(blocks, (0, 3, 1, 2, 4))


def _global_mutation_delta(
    key: Any,
    *,
    spec: GlobalMutationSpec,
    dtype: Any,
) -> Any:
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    shape = (spec.grid_size, spec.grid_size, spec.channels)
    if not spec.enabled:
        return jnp.zeros(shape, dtype=dtype)
    kmut, kpos, kprob = jr.split(key, 3)
    size = max(1, min(spec.patch_size, spec.grid_size))
    mutation = (
        jnp.ones((size, size, spec.channels), dtype=dtype)
        * jr.normal(kmut, (1, 1, spec.channels), dtype=dtype)
        * jnp.asarray(spec.scale, dtype=dtype)
    )
    max_position = spec.grid_size - size
    key_i, key_j = jr.split(kpos)
    i0 = jr.randint(key_i, (), 0, max_position + 1)
    j0 = jr.randint(key_j, (), 0, max_position + 1)
    delta = jnp.zeros(shape, dtype=dtype)
    delta = jax.lax.dynamic_update_slice(delta, mutation, (i0, j0, 0))
    active = (jr.uniform(kprob, ()) < spec.probability).astype(dtype)
    return delta * active


def _stack_trees(values: Sequence[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *values)


def _unstack_tree(value: Any) -> list[Any]:
    import jax

    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return []
    return [jax.tree_util.tree_map(lambda x: x[idx], value) for idx in range(leaves[0].shape[0])]


def _assemble_block_field(
    value: Any,
    *,
    geometry: SimulationGeometry,
) -> Any:
    import jax.numpy as jnp

    arr = jnp.asarray(value)
    pad = geometry.wall_pad
    size = geometry.block_size
    arr = arr[:, pad : pad + size, pad : pad + size, ...]
    suffix = arr.shape[3:]
    grid = arr.reshape(
        (geometry.split_n, geometry.split_n, size, size) + suffix
    )
    axes = (0, 2, 1, 3) + tuple(range(4, grid.ndim))
    full = jnp.transpose(grid, axes).reshape(
        (geometry.padded_grid_size, geometry.padded_grid_size) + suffix
    )
    c0 = geometry.crop_before
    return full[
        c0 : c0 + geometry.grid_size,
        c0 : c0 + geometry.grid_size,
        ...,
    ]


def _merge_block_state(
    initial_state: dict[str, Any],
    block_state: dict[str, Any],
    *,
    geometry: SimulationGeometry,
) -> dict[str, Any]:
    import jax.numpy as jnp

    merged = dict(initial_state)
    for key in ("A", "P", "F", "Food"):
        if key in block_state:
            merged[key] = _assemble_block_field(
                block_state[key],
                geometry=geometry,
            )
    if "t" in block_state:
        merged["t"] = jnp.asarray(block_state["t"][0])
    if "mass_cycle_start" in merged or "mass_cycle_start" in block_state:
        merged["mass_cycle_start"] = jnp.sum(merged["A"])
    return merged


def _pack_particles_fixed(
    packs: Sequence[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    cap = max(int(pack["points"].shape[1]) for pack in packs)
    n_blocks = int(packs[0]["points"].shape[0])
    points = np.full(
        (len(packs), n_blocks, cap, 2),
        WALL_PAD + 0.5,
        dtype=np.float32,
    )
    channels = np.zeros((len(packs), n_blocks, cap), dtype=np.int32)
    for idx, pack in enumerate(packs):
        count = int(pack["points"].shape[1])
        points[idx, :, :count] = pack["points"]
        channels[idx, :, :count] = pack["channels"]
    return points, channels


def _runtime(config_path: Path, params_example: np.ndarray) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import substrates
    import util
    from flowlenia_minibang_simulate import _make_substrate

    _cfg, flat = load_config(config_path)
    args = SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))
    substrate = _make_substrate(args)
    _ = substrate.seed_state(jax.random.PRNGKey(0), jnp.asarray(params_example))
    geometry = _geometry(int(args.grid_size))
    flow_kwargs = util.flow_lenia_kwargs_from_args(args)
    stochastic_expected = {
        "grid_size": 128,
        "mix_rule": "stoch",
        "mutation": True,
        "mutation_patch_size": EXPECTED_MUTATION_PATCH_SIZE,
        "mutation_prob": EXPECTED_MUTATION_PROBABILITY,
        "mutation_scale": EXPECTED_MUTATION_SCALE,
        "optimize_mutation_scale": False,
        "volcano": False,
        "food_enabled": False,
        "mass_decay": 0.0,
        "mass_renorm": False,
    }
    stochastic_mismatches = {
        key: {"expected": expected, "actual": flow_kwargs.get(key)}
        for key, expected in stochastic_expected.items()
        if flow_kwargs.get(key) != expected
    }
    if stochastic_mismatches:
        raise RuntimeError(
            "C5 RNG-matched wall configuration differs from the frozen "
            f"protocol: {stochastic_mismatches}"
        )
    if bool(flow_kwargs["volcano"]):
        raise RuntimeError(
            "RNG-matched C5 walls currently require volcano=false"
        )
    if bool(flow_kwargs["food_enabled"]):
        raise RuntimeError(
            "RNG-matched C5 walls currently require food=false"
        )
    if bool(flow_kwargs["optimize_mutation_scale"]):
        raise RuntimeError(
            "RNG-matched C5 walls require optimize_mutation_scale=false"
        )
    mutation_spec = GlobalMutationSpec(
        enabled=bool(flow_kwargs["mutation"]),
        grid_size=int(args.grid_size),
        channels=int(args.k),
        patch_size=int(flow_kwargs["mutation_patch_size"]),
        probability=float(flow_kwargs["mutation_prob"]),
        scale=float(flow_kwargs["mutation_scale"]),
    )
    deterministic_kwargs = dict(flow_kwargs)
    deterministic_kwargs["mutation"] = False
    deterministic_kwargs["volcano"] = False
    deterministic_kwargs["debug_return_F"] = True
    deterministic_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate(
            "lenia_flow",
            **deterministic_kwargs,
        )
    )
    _ = deterministic_substrate.seed_state(
        jax.random.PRNGKey(0),
        jnp.asarray(params_example),
    )
    block_kwargs = dict(deterministic_kwargs)
    block_kwargs["grid_size"] = (
        geometry.block_size + 2 * geometry.wall_pad
    )
    block_substrate = substrates.FlattenSubstrateParameters(
        substrates.create_substrate("lenia_flow", **block_kwargs)
    )
    _ = block_substrate.seed_state(
        jax.random.PRNGKey(0),
        jnp.asarray(params_example),
    )
    valid_mask = _hard_wall_core_mask(geometry)
    global_rt_gumbel = _global_rt_gumbel(
        grid_size=geometry.grid_size,
        dd=int(substrate.RT.dd),
        dtype=jnp.float32,
    )
    block_rt_gumbel = _partition_rt_gumbel(
        global_rt_gumbel,
        geometry=geometry,
    )
    return {
        "args": args,
        "substrate": substrate,
        "deterministic_substrate": deterministic_substrate,
        "block_substrate": block_substrate,
        "mutation_spec": mutation_spec,
        "global_rt_gumbel": global_rt_gumbel,
        "block_rt_gumbel": block_rt_gumbel,
        "block_template_cache": {},
        "rt_noise_protocol": (
            "JAX categorical Gumbels from PRNGKey(42) are generated once at "
            "native 128x128 shape; exact spatial values are partitioned into "
            "hard-wall block cores"
        ),
        "geometry": geometry,
        "valid_mask": jnp.asarray(valid_mask, dtype=bool),
    }


def _selected_step_keys(
    rng_in: Any,
    original_batch_index: Any,
    *,
    n_steps: int,
    original_batch_size: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    def split_one(key: Any) -> Any:
        return jax.random.split(
            key,
            int(n_steps) * int(original_batch_size),
        ).reshape((int(n_steps), int(original_batch_size), 2))

    by_item = jax.vmap(split_one)(rng_in)
    by_step = jnp.swapaxes(by_item, 0, 1)
    idx = jnp.asarray(original_batch_index, dtype=jnp.int32)
    gather = jnp.broadcast_to(
        idx[None, :, None, None],
        (int(n_steps), idx.shape[0], 1, 2),
    )
    return jnp.take_along_axis(by_step, gather, axis=2)[:, :, 0, :]


def _make_block_stepper(
    block_substrate: Any,
    *,
    n_blocks: int,
    original_batch_size: int,
    valid_mask: Any,
    args: Any,
):
    import jax
    import jax.numpy as jnp
    from paper_check_frustration_batch_eval import _mask_block_spatial_state

    rt = block_substrate.RT
    flow_channel = int(getattr(args, "lagrangian_flow_channel", -1))
    flow_reduce = str(getattr(args, "lagrangian_flow_reduce", "mass_weighted"))
    channel_mode = str(getattr(args, "lagrangian_channel_mode", "resample"))
    noise_model = str(getattr(args, "lagrangian_noise_model", "rt_box"))
    diffusion_scale = float(getattr(args, "lagrangian_diffusion_scale", 1.0))

    def get(n_steps: int):
        n_steps = int(n_steps)

        @jax.jit
        def step(
            state_in: Any,
            points_in: Any,
            channels_in: Any,
            subkeys: Any,
            params_in: Any,
            original_batch_index: Any,
        ) -> tuple[Any, Any, Any]:
            selected = _selected_step_keys(
                subkeys,
                original_batch_index,
                n_steps=n_steps,
                original_batch_size=original_batch_size,
            )

            def one_lane(
                lane_key: Any,
                lane_state: Any,
                lane_points: Any,
                lane_channels: Any,
                lane_params: Any,
            ) -> tuple[Any, Any, Any]:
                block_keys = jax.random.split(lane_key, n_blocks)
                next_state = jax.vmap(
                    lambda state, key: block_substrate.step_state(
                        key, state, lane_params
                    )
                )(lane_state, block_keys)
                next_state = _mask_block_spatial_state(
                    next_state,
                    valid_mask,
                )
                lag_keys = jax.vmap(
                    lambda key: jax.random.fold_in(
                        key, jnp.uint32(0x4C4147)
                    )
                )(block_keys)

                def advect(
                    points: Any,
                    channels: Any,
                    flow: Any,
                    mass: Any,
                    lag_key: Any,
                ) -> tuple[Any, Any]:
                    return rt.advect_particles(
                        points=points,
                        F=flow,
                        A=mass,
                        channel=flow_channel,
                        reduce=flow_reduce,
                        point_channels=channels,
                        channel_mode=channel_mode,
                        key=lag_key,
                        noise_model=noise_model,
                        diffusion_scale=diffusion_scale,
                    )

                next_points, next_channels = jax.vmap(advect)(
                    lane_points,
                    lane_channels,
                    next_state["F"],
                    next_state["A"],
                    lag_keys,
                )
                return next_state, next_points, next_channels

            vmapped_lane = jax.vmap(one_lane, in_axes=(0, 0, 0, 0, 0))

            def body(carry: tuple[Any, Any, Any], keys: Any):
                state, points, channels = carry
                next_state, next_points, next_channels = vmapped_lane(
                    keys,
                    state,
                    points,
                    channels,
                    params_in,
                )
                return (next_state, next_points, next_channels), None

            return jax.lax.scan(
                body,
                (state_in, points_in, channels_in),
                selected,
            )[0]

        return step

    return get


def _make_block_state_stepper(
    block_substrate: Any,
    *,
    n_blocks: int,
    original_batch_size: int,
    valid_mask: Any,
    geometry: SimulationGeometry,
    mutation_spec: GlobalMutationSpec,
    block_rt_gumbel: Any,
):
    import jax
    import jax.numpy as jnp
    from paper_check_frustration_batch_eval import _mask_block_spatial_state

    cache: dict[int, Any] = {}

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
            selected = _selected_step_keys(
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
                mutation_delta = _global_mutation_delta(
                    lane_key,
                    spec=mutation_spec,
                    dtype=next_state["P"].dtype,
                )
                mutation_blocks = _partition_global_field(
                    mutation_delta,
                    geometry=geometry,
                )
                next_state = {
                    **next_state,
                    "P": next_state["P"] + mutation_blocks,
                }
                return _mask_block_spatial_state(next_state, valid_mask)

            vmapped_lane = jax.vmap(one_lane, in_axes=(0, 0, 0))

            def body(state: Any, keys: Any):
                return vmapped_lane(keys, state, params_in), None

            return jax.lax.scan(body, state_in, selected)[0]

        cache[n_steps] = step
        return step

    return get


def _make_global_state_stepper(
    substrate: Any,
    *,
    original_batch_size: int,
):
    import jax

    cache: dict[int, Any] = {}

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
            selected = _selected_step_keys(
                subkeys,
                original_batch_index,
                n_steps=n_steps,
                original_batch_size=original_batch_size,
            )

            def one_lane(key: Any, state: Any, lane_params: Any) -> Any:
                return substrate.step_state(key, state, lane_params)

            vmapped_lane = jax.vmap(one_lane, in_axes=(0, 0, 0))

            def body(state: Any, keys: Any):
                return vmapped_lane(keys, state, params_in), None

            return jax.lax.scan(body, state_in, selected)[0]

        cache[n_steps] = step
        return step

    return get


def _make_controlled_global_state_stepper(
    deterministic_substrate: Any,
    *,
    original_batch_size: int,
    mutation_spec: GlobalMutationSpec,
    global_rt_gumbel: Any,
):
    import jax

    cache: dict[int, Any] = {}

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
            selected = _selected_step_keys(
                subkeys,
                original_batch_index,
                n_steps=n_steps,
                original_batch_size=original_batch_size,
            )

            def one_lane(key: Any, state: Any, lane_params: Any) -> Any:
                next_state = (
                    deterministic_substrate.step_state_with_reintegration_gumbel(
                        key,
                        state,
                        lane_params,
                        global_rt_gumbel,
                    )
                )
                mutation_delta = _global_mutation_delta(
                    key,
                    spec=mutation_spec,
                    dtype=next_state["P"].dtype,
                )
                return {
                    **next_state,
                    "P": next_state["P"] + mutation_delta,
                }

            vmapped_lane = jax.vmap(one_lane, in_axes=(0, 0, 0))

            def body(state: Any, keys: Any):
                return vmapped_lane(keys, state, params_in), None

            return jax.lax.scan(body, state_in, selected)[0]

        cache[n_steps] = step
        return step

    return get


def _load_simulation_item(
    row: dict[str, str],
    substrate: Any,
    *,
    snapshot_cache: dict[tuple[str, int], tuple[Path, int, dict[str, Any]]] | None = None,
    params_cache: dict[str, np.ndarray] | None = None,
    state_template_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    branch_dir = _resolve(row["free_branch_dir"])
    params_key = str(row["params_sha256"])
    if params_cache is not None and params_key in params_cache:
        params = params_cache[params_key]
    else:
        params = np.asarray(np.load(branch_dir / "params.npy"), dtype=np.float32)
        if params_cache is not None:
            params_cache[params_key] = params
    resume_metadata = json.loads(
        (branch_dir / "resume_metadata.json").read_text()
    )
    source_traj_dir = _resolve(resume_metadata["source_traj_dir"])
    snapshot_key = (str(source_traj_dir), _as_int(row["step"]))
    cached = None if snapshot_cache is None else snapshot_cache.get(snapshot_key)
    if cached is None:
        apf_path, step, snapshot_idx = _find_snapshot(
            source_traj_dir / "apf_logs",
            _as_int(row["step"]),
        )
        snapshot = _read_snapshot(apf_path, snapshot_idx)
        if snapshot_cache is not None:
            snapshot_cache[snapshot_key] = (apf_path, int(snapshot_idx), snapshot)
    else:
        apf_path, snapshot_idx, snapshot = cached
        step = _as_int(row["step"])
    template = (
        state_template_cache.get(params_key)
        if state_template_cache is not None
        else None
    )
    if template is None:
        template = dict(
            substrate.init_state(
                jax.random.PRNGKey(0),
                jnp.asarray(params, dtype=jnp.float32),
            )
        )
        if state_template_cache is not None:
            state_template_cache[params_key] = template
    state = dict(template)
    a_value = np.asarray(snapshot["A"], dtype=np.float32)
    state["A"] = jnp.asarray(a_value)
    state["P"] = jnp.asarray(
        np.asarray(snapshot["P"], dtype=np.float32)
    )
    if "F" in snapshot:
        state["F"] = jnp.asarray(
            np.asarray(snapshot["F"], dtype=np.float32)
        )
    else:
        state["F"] = jnp.zeros(
            (
                state["A"].shape[0],
                state["A"].shape[1],
                2,
                state["A"].shape[-1],
            ),
            dtype=state["A"].dtype,
        )
    state["t"] = jnp.asarray(
        _scalar(snapshot, "state_t", 0),
        dtype=jnp.int32,
    )
    state["mass_cycle_start"] = jnp.asarray(
        _scalar(
            snapshot,
            "state_mass_cycle_start",
            float(np.sum(a_value)),
        ),
        dtype=jnp.float32,
    )
    points = np.asarray(snapshot["lagrangian_xy"], dtype=np.float32)
    rng = np.asarray(snapshot["resume_batch_rng_key"], dtype=np.uint32)
    state, points_jax, rng_jax = _apply_resume_perturbation(
        state=state,
        lag_xy=points,
        rng=rng,
        seed=_as_int(row["branch_seed"]),
        a_std=_as_float(row["perturb_a_std"]),
        p_std=_as_float(row["perturb_p_std"]),
        lag_xy_std=_as_float(row["perturb_lagrangian_xy_std"]),
        border=str(getattr(substrate.RT, "border", "wall")),
        sigma=float(getattr(substrate.RT, "sigma", 0.0)),
    )
    return {
        "row": row,
        "branch_dir": branch_dir,
        "params": params,
        "snapshot": snapshot,
        "state": state,
        "points": np.asarray(points_jax, dtype=np.float32),
        "channels": np.asarray(snapshot["lagrangian_c"], dtype=np.int32),
        "rng": np.asarray(rng_jax, dtype=np.uint32),
        "original_batch_size": int(_scalar(snapshot, "resume_batch_size", 1)),
        "original_batch_index": int(_scalar(snapshot, "resume_batch_index", 0)),
        "snapshot_interval": int(
            _scalar(snapshot, "resume_snapshot_interval", JIT_MICROBATCH)
        ),
        "jit_microbatch": int(
            _scalar(snapshot, "resume_jit_microbatch", JIT_MICROBATCH)
        ),
        "source_apf_path": str(apf_path),
        "source_snapshot_idx": int(snapshot_idx),
        "source_step": int(step),
    }


def _mutation_injection_audit(
    item: dict[str, Any],
    runtime: dict[str, Any],
    *,
    plan_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    spec: GlobalMutationSpec = runtime["mutation_spec"]
    rng = jnp.asarray(item["rng"], dtype=jnp.uint32)[None, :]
    index = jnp.asarray([item["original_batch_index"]], dtype=jnp.int32)
    keys = []
    for _ in range(4):
        rng, subkeys = _split_rng_batch(rng)
        selected = _selected_step_keys(
            subkeys,
            index,
            n_steps=JIT_MICROBATCH,
            original_batch_size=item["original_batch_size"],
        )
        keys.extend(np.asarray(jax.device_get(selected[:, 0, :]), dtype=np.uint32))

    active = []
    inactive = []
    for key in keys:
        _kmut, _kpos, kprob = jr.split(jnp.asarray(key), 3)
        is_active = bool(
            np.asarray(jax.device_get(jr.uniform(kprob, ()) < spec.probability))
        )
        (active if is_active else inactive).append(key)
    selected_keys = (
        active[:4] + inactive[:4]
        if spec.enabled
        else inactive[:8]
    )
    if spec.enabled and not active:
        raise RuntimeError(
            "Could not find an active global mutation in the first 200 "
            "optimizer-equivalent step keys"
        )

    state = item["state"]
    params = jnp.asarray(item["params"], dtype=jnp.float32)
    substrate = runtime["substrate"]
    deterministic = runtime["deterministic_substrate"]

    @jax.jit
    def compare(key: Any) -> tuple[Any, Any, Any, Any]:
        actual = substrate.step_state(key, state, params)
        native_baseline = deterministic.step_state(key, state, params)
        controlled_baseline = (
            deterministic.step_state_with_reintegration_gumbel(
                key,
                state,
                params,
                runtime["global_rt_gumbel"],
            )
        )
        delta = _global_mutation_delta(
            key,
            spec=spec,
            dtype=controlled_baseline["P"].dtype,
        )
        partitioned = _partition_global_field(
            delta,
            geometry=runtime["geometry"],
        )
        roundtrip = _assemble_block_field(
            partitioned,
            geometry=runtime["geometry"],
        )
        return (
            actual,
            native_baseline,
            controlled_baseline,
            (delta, roundtrip),
        )

    comparisons = []
    all_exact = True
    for key in selected_keys:
        (
            actual,
            native_baseline,
            controlled_baseline,
            (delta, roundtrip),
        ) = jax.device_get(
            compare(jnp.asarray(key, dtype=jnp.uint32))
        )
        rt_fields_exact = {
            name: bool(
                np.array_equal(
                    np.asarray(native_baseline[name]),
                    np.asarray(controlled_baseline[name]),
                )
            )
            for name in native_baseline
            if name in controlled_baseline
        }
        predicted_p = (
            np.asarray(controlled_baseline["P"]) + np.asarray(delta)
        )
        p_exact = bool(np.array_equal(np.asarray(actual["P"]), predicted_p))
        roundtrip_exact = bool(
            np.array_equal(np.asarray(delta), np.asarray(roundtrip))
        )
        non_p_exact = {
            name: bool(
                np.array_equal(
                    np.asarray(actual[name]),
                    np.asarray(controlled_baseline[name]),
                )
            )
            for name in actual
            if name != "P" and name in controlled_baseline
        }
        exact = (
            p_exact
            and roundtrip_exact
            and all(rt_fields_exact.values())
            and all(non_p_exact.values())
        )
        all_exact = all_exact and exact
        delta_array = np.asarray(delta)
        nonzero_spatial = np.argwhere(
            np.any(delta_array != 0, axis=-1)
        )
        mutation_bounds = (
            {
                "y_min": int(nonzero_spatial[:, 0].min()),
                "y_max": int(nonzero_spatial[:, 0].max()),
                "x_min": int(nonzero_spatial[:, 1].min()),
                "x_max": int(nonzero_spatial[:, 1].max()),
            }
            if nonzero_spatial.size
            else None
        )
        comparisons.append(
            {
                "key": np.asarray(key, dtype=np.uint32).tolist(),
                "mutation_active": bool(nonzero_spatial.size),
                "mutation_delta_sha256": _sha256_array(delta_array),
                "mutation_bounds": mutation_bounds,
                "p_exact": p_exact,
                "native_vs_controlled_rt_fields_exact": rt_fields_exact,
                "partition_roundtrip_exact": roundtrip_exact,
                "non_p_fields_exact": non_p_exact,
                "max_abs_p": float(
                    np.max(
                        np.abs(
                            np.asarray(actual["P"], dtype=np.float64)
                            - np.asarray(predicted_p, dtype=np.float64)
                        )
                    )
                ),
            }
        )
    stream_keys = jnp.asarray(
        np.stack(keys[:JIT_MICROBATCH]),
        dtype=jnp.uint32,
    )

    @jax.jit
    def stream_compare(step_keys: Any) -> tuple[Any, Any]:
        def actual_body(current: Any, key: Any) -> tuple[Any, None]:
            return substrate.step_state(key, current, params), None

        def injected_body(current: Any, key: Any) -> tuple[Any, None]:
            next_state = (
                deterministic.step_state_with_reintegration_gumbel(
                    key,
                    current,
                    params,
                    runtime["global_rt_gumbel"],
                )
            )
            delta = _global_mutation_delta(
                key,
                spec=spec,
                dtype=next_state["P"].dtype,
            )
            next_state = {
                **next_state,
                "P": next_state["P"] + delta,
            }
            return next_state, None

        actual_final = jax.lax.scan(
            actual_body,
            state,
            step_keys,
        )[0]
        injected_final = jax.lax.scan(
            injected_body,
            state,
            step_keys,
        )[0]
        return actual_final, injected_final

    actual_final, injected_final = jax.device_get(
        stream_compare(stream_keys)
    )
    stream_fields_exact = {
        name: bool(
            np.array_equal(
                np.asarray(actual_final[name]),
                np.asarray(injected_final[name]),
            )
        )
        for name in actual_final
        if name in injected_final
    }
    stream_exact = all(stream_fields_exact.values())
    all_exact = all_exact and stream_exact
    block_gumbel_spatial = jnp.transpose(
        runtime["block_rt_gumbel"],
        (0, 2, 3, 1, 4),
    )
    merged_gumbel = _assemble_block_field(
        block_gumbel_spatial,
        geometry=runtime["geometry"],
    )
    merged_gumbel = jnp.transpose(merged_gumbel, (2, 0, 1, 3))
    rt_partition_exact = bool(
        np.array_equal(
            np.asarray(jax.device_get(runtime["global_rt_gumbel"])),
            np.asarray(jax.device_get(merged_gumbel)),
        )
    )
    all_exact = all_exact and rt_partition_exact
    report = {
        "status": "passed" if all_exact else "failed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": plan_sha256,
        "row_id": _as_int(item["row"]["row_id"]),
        "n_schedule_keys_scanned": len(keys),
        "n_active_schedule_keys": len(active),
        "n_compared": len(comparisons),
        "mutation_spec": {
            "enabled": spec.enabled,
            "grid_size": spec.grid_size,
            "channels": spec.channels,
            "patch_size": spec.patch_size,
            "probability": spec.probability,
            "scale": spec.scale,
        },
        "block_local_mutation_enabled": bool(
            runtime["block_substrate"].mutation_enabled
        ),
        "stream_steps_compared": JIT_MICROBATCH,
        "stream_fields_exact": stream_fields_exact,
        "stream_exact": stream_exact,
        "native_global_rt_vs_external_gumbel_exact": all(
            all(
                comparison[
                    "native_vs_controlled_rt_fields_exact"
                ].values()
            )
            for comparison in comparisons
        ),
        "global_rt_gumbel_partition_roundtrip_exact": rt_partition_exact,
        "global_rt_gumbel_sha256": _sha256_array(
            jax.device_get(runtime["global_rt_gumbel"])
        ),
        "block_rt_gumbel_sha256": _sha256_array(
            jax.device_get(runtime["block_rt_gumbel"])
        ),
        "hard_wall_mask_sha256": _sha256_array(
            jax.device_get(runtime["valid_mask"])
        ),
        "rt_noise_protocol": runtime["rt_noise_protocol"],
        "all_exact": all_exact,
        "comparisons": comparisons,
    }
    _write_json(output_root / "preflight_mutation_exactness.json", report)
    if not all_exact or report["block_local_mutation_enabled"]:
        raise RuntimeError(
            "Global mutation injection parity failed; refusing wall simulations"
        )
    return report


def _wall_batch_topology_audit(
    item: dict[str, Any],
    runtime: dict[str, Any],
    *,
    plan_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    state_one, split_roundtrip = _prepare_block_state_batch(
        [item],
        runtime,
    )
    state_full = jax.tree_util.tree_map(
        lambda value: jnp.repeat(
            value,
            SIMULATION_BATCH_SIZE,
            axis=0,
        ),
        state_one,
    )
    rng_one = jnp.asarray(item["rng"], dtype=jnp.uint32)[None, :]
    rng_full = jnp.repeat(
        rng_one,
        SIMULATION_BATCH_SIZE,
        axis=0,
    )
    _next_one, subkeys_one = _split_rng_batch(rng_one)
    _next_full, subkeys_full = _split_rng_batch(rng_full)
    params_one = jnp.asarray(
        item["params"],
        dtype=jnp.float32,
    )[None, :]
    params_full = jnp.repeat(
        params_one,
        SIMULATION_BATCH_SIZE,
        axis=0,
    )
    index_one = jnp.asarray(
        [item["original_batch_index"]],
        dtype=jnp.int32,
    )
    index_full = jnp.repeat(
        index_one,
        SIMULATION_BATCH_SIZE,
        axis=0,
    )
    stepper = _make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=item["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
    )
    output_one = jax.device_get(
        stepper(1)(
            state_one,
            subkeys_one,
            params_one,
            index_one,
        )
    )
    output_full = jax.device_get(
        stepper(1)(
            state_full,
            subkeys_full,
            params_full,
            index_full,
        )
    )
    fields_exact = {
        key: bool(
            all(
                np.array_equal(
                    np.asarray(output_one[key][0]),
                    np.asarray(output_full[key][lane]),
                )
                for lane in range(SIMULATION_BATCH_SIZE)
            )
        )
        for key in output_one
        if key in output_full
    }
    all_exact = bool(
        split_roundtrip["all_ap_exact"]
        and fields_exact
        and all(fields_exact.values())
    )
    report = {
        "status": "passed" if all_exact else "failed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": plan_sha256,
        "n_steps": 1,
        "small_batch_size": 1,
        "frozen_batch_size": SIMULATION_BATCH_SIZE,
        "all_frozen_lanes_repeat_the_small_batch_exactly": all_exact,
        "fields_exact": fields_exact,
        "split_merge_exact": split_roundtrip["all_ap_exact"],
    }
    _write_json(
        output_root / "preflight_batch_topology_exactness.json",
        report,
    )
    if not all_exact:
        raise RuntimeError(
            "Wall outer-batch topology parity failed"
        )
    return report


def _pad_items(items: list[dict[str, Any]], batch_size: int) -> tuple[list[dict[str, Any]], int]:
    real_n = len(items)
    if real_n < 1:
        raise ValueError("Cannot pad an empty simulation batch")
    padded = list(items)
    while len(padded) < int(batch_size):
        padded.append(items[(len(padded) - real_n) % real_n])
    return padded, real_n


def _capture_global(
    states: Any,
    points: Any,
    channels: Any,
    rng: Any,
    *,
    real_n: int,
) -> list[dict[str, Any]]:
    import jax

    states_host = _unstack_tree(jax.device_get(states))
    points_host = np.asarray(jax.device_get(points), dtype=np.float32)
    channels_host = np.asarray(jax.device_get(channels), dtype=np.int32)
    rng_host = np.asarray(jax.device_get(rng), dtype=np.uint32)
    return [
        {
            "state": {key: np.asarray(value) for key, value in states_host[idx].items()},
            "points": points_host[idx],
            "channels": channels_host[idx],
            "rng": rng_host[idx],
        }
        for idx in range(real_n)
    ]


def _capture_blocks(
    block_states: Any,
    block_points: Any,
    block_channels: Any,
    rng: Any,
    *,
    initial_states: Sequence[dict[str, Any]],
    packs: Sequence[dict[str, np.ndarray]],
    geometry: SimulationGeometry,
    real_n: int,
) -> list[dict[str, Any]]:
    import jax

    block_states_host = _unstack_tree(jax.device_get(block_states))
    points_host = np.asarray(jax.device_get(block_points), dtype=np.float32)
    channels_host = np.asarray(jax.device_get(block_channels), dtype=np.int32)
    rng_host = np.asarray(jax.device_get(rng), dtype=np.uint32)
    captures = []
    for idx in range(real_n):
        merged = _merge_block_state(
            initial_states[idx],
            block_states_host[idx],
            geometry=geometry,
        )
        points_global, channels_global = _unpack_particles(
            points_host[idx],
            channels_host[idx],
            packs[idx],
            geometry=geometry,
        )
        captures.append(
            {
                "state": {key: np.asarray(value) for key, value in merged.items()},
                "points": points_global,
                "channels": channels_global,
                "rng": rng_host[idx],
            }
        )
    return captures


def _unpack_particles(
    points: np.ndarray,
    channels: np.ndarray,
    pack: dict[str, np.ndarray],
    *,
    geometry: SimulationGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    block_idx = np.asarray(pack["block_idx"], dtype=np.int32)
    slots = np.asarray(pack["slots"], dtype=np.int32)
    selected = np.asarray(points, dtype=np.float32)[block_idx, slots].copy()
    rows = block_idx // geometry.split_n
    cols = block_idx % geometry.split_n
    selected[:, 0] += (
        rows * geometry.block_size
        - geometry.wall_pad
        - geometry.crop_before
    )
    selected[:, 1] += (
        cols * geometry.block_size
        - geometry.wall_pad
        - geometry.crop_before
    )
    selected_channels = np.asarray(channels, dtype=np.int32)[block_idx, slots]
    return selected, selected_channels.copy()


def _split_rng_batch(rng: Any) -> tuple[Any, Any]:
    import jax

    split = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
    return split[:, 0, :], split[:, 1, :]


def _capture_steps_from_free(row: dict[str, str]) -> np.ndarray:
    arrays = _branch_arrays(
        _resolve(row["free_branch_dir"]),
        keys={"steps"},
    )
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    relative = steps - _as_int(row["step"])
    if relative[0] != 0 or relative[-1] != HORIZON_STEPS:
        raise RuntimeError(f"Invalid free capture steps for row {row['row_id']}")
    if np.any(relative % JIT_MICROBATCH != 0):
        raise RuntimeError(f"Free C2 capture steps are not 50-step aligned: {relative}")
    return relative


def _prepare_block_batch(
    items: Sequence[dict[str, Any]],
    runtime: dict[str, Any],
) -> tuple[Any, Any, Any, list[dict[str, np.ndarray]], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from evaluate_frustration_history_dependence import _prepare_block_template_state
    from flowlenia_c5_full_length_delta_h_one import _pack_wall_particles
    from paper_check_frustration_batch_eval import _pad_flow_spatial_state

    geometry: SimulationGeometry = runtime["geometry"]
    block_states = []
    packs: list[dict[str, np.ndarray]] = []
    roundtrip: dict[str, Any] = {"all_ap_exact": True, "items": []}
    for item in items:
        params = jnp.asarray(item["params"], dtype=jnp.float32)
        params_key = str(item["row"]["params_sha256"])
        template = runtime["block_template_cache"].get(params_key)
        if template is None:
            template = runtime["block_substrate"].init_state(
                jax.random.PRNGKey(0),
                params,
            )
            runtime["block_template_cache"][params_key] = template
        padded = _pad_flow_spatial_state(
            item["state"],
            pad_before=geometry.crop_before,
            pad_after=geometry.crop_after,
        )
        block_state = _prepare_block_template_state(
            initial_state=padded,
            block_template=template,
            split_n=geometry.split_n,
            block_size=geometry.block_size,
            pad=geometry.wall_pad,
            C=int(runtime["args"].C),
            k=int(runtime["args"].k),
        )
        block_states.append(block_state)
        pack = _pack_wall_particles(
            item["points"],
            item["channels"],
            split_n=geometry.split_n,
            block_size=geometry.block_size,
            pad=geometry.wall_pad,
            crop_before=geometry.crop_before,
        )
        packs.append(pack)
        merged = _merge_block_state(
            item["state"],
            jax.device_get(block_state),
            geometry=geometry,
        )
        item_check = {}
        for key in ("A", "P"):
            actual = np.asarray(merged[key])
            expected = np.asarray(item["state"][key])
            exact = bool(np.array_equal(actual, expected))
            item_check[key] = {
                "exact": exact,
                "max_abs": float(np.max(np.abs(actual - expected))),
            }
            roundtrip["all_ap_exact"] = roundtrip["all_ap_exact"] and exact
        roundtrip["items"].append(item_check)
    block_points, block_channels = _pack_particles_fixed(packs)
    return (
        _stack_trees(block_states),
        jnp.asarray(block_points),
        jnp.asarray(block_channels),
        packs,
        roundtrip,
    )


def _prepare_block_state_batch(
    items: Sequence[dict[str, Any]],
    runtime: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from evaluate_frustration_history_dependence import _prepare_block_template_state
    from paper_check_frustration_batch_eval import _pad_flow_spatial_state

    geometry: SimulationGeometry = runtime["geometry"]
    block_states = []
    roundtrip: dict[str, Any] = {"all_ap_exact": True, "items": []}
    for item in items:
        params = jnp.asarray(item["params"], dtype=jnp.float32)
        params_key = str(item["row"]["params_sha256"])
        template = runtime["block_template_cache"].get(params_key)
        if template is None:
            template = runtime["block_substrate"].init_state(
                jax.random.PRNGKey(0),
                params,
            )
            runtime["block_template_cache"][params_key] = template
        padded = _pad_flow_spatial_state(
            item["state"],
            pad_before=geometry.crop_before,
            pad_after=geometry.crop_after,
        )
        block_state = _prepare_block_template_state(
            initial_state=padded,
            block_template=template,
            split_n=geometry.split_n,
            block_size=geometry.block_size,
            pad=geometry.wall_pad,
            C=int(runtime["args"].C),
            k=int(runtime["args"].k),
        )
        block_states.append(block_state)
        merged = _merge_block_state(
            item["state"],
            jax.device_get(block_state),
            geometry=geometry,
        )
        item_check = {}
        for key in ("A", "P"):
            actual = np.asarray(merged[key])
            expected = np.asarray(item["state"][key])
            exact = bool(np.array_equal(actual, expected))
            item_check[key] = {
                "exact": exact,
                "max_abs": float(np.max(np.abs(actual - expected))),
            }
            roundtrip["all_ap_exact"] = roundtrip["all_ap_exact"] and exact
        roundtrip["items"].append(item_check)
    return _stack_trees(block_states), roundtrip


def _capture_global_state(
    states: Any,
    rng: Any,
    *,
    real_n: int,
) -> list[dict[str, Any]]:
    import jax

    states_host = _unstack_tree(jax.device_get(states))
    rng_host = np.asarray(jax.device_get(rng), dtype=np.uint32)
    return [
        {
            "state": {key: np.asarray(value) for key, value in states_host[idx].items()},
            "rng": rng_host[idx],
        }
        for idx in range(real_n)
    ]


def _capture_block_state(
    block_states: Any,
    rng: Any,
    *,
    initial_states: Sequence[dict[str, Any]],
    geometry: SimulationGeometry,
    real_n: int,
) -> list[dict[str, Any]]:
    import jax

    block_states_host = _unstack_tree(jax.device_get(block_states))
    rng_host = np.asarray(jax.device_get(rng), dtype=np.uint32)
    return [
        {
            "state": {
                key: np.asarray(value)
                for key, value in _merge_block_state(
                    initial_states[idx],
                    block_states_host[idx],
                    geometry=geometry,
                ).items()
            },
            "rng": rng_host[idx],
        }
        for idx in range(real_n)
    ]


def _write_wall_branch(
    item: dict[str, Any],
    captures: list[dict[str, Any]],
    relative_steps: np.ndarray,
    *,
    protocol: dict[str, Any],
    roundtrip: dict[str, Any],
    geometry: SimulationGeometry,
    batch_context: dict[str, Any],
) -> None:
    from simulate_save_apf import save_chunk

    row = item["row"]
    output_dir = _resolve(row["walls_branch_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    apf_dir = output_dir / "apf_logs"
    apf_dir.mkdir(parents=True, exist_ok=True)
    absolute_steps = [
        _as_int(row["step"]) + int(relative) for relative in relative_steps
    ]
    resume_meta = {
        "resume_batch_rng_key": [
            np.asarray(capture["rng"], dtype=np.uint32) for capture in captures
        ],
        "resume_batch_size": [
            np.asarray(item["original_batch_size"], dtype=np.int32)
            for _ in captures
        ],
        "resume_batch_index": [
            np.asarray(item["original_batch_index"], dtype=np.int32)
            for _ in captures
        ],
        "resume_selection0": [
            np.asarray(
                _scalar(item["snapshot"], "resume_selection0", 0),
                dtype=np.int32,
            )
            for _ in captures
        ],
        "resume_jit_microbatch": [
            np.asarray(item["jit_microbatch"], dtype=np.int32)
            for _ in captures
        ],
        "resume_snapshot_interval": [
            np.asarray(item["snapshot_interval"], dtype=np.int32)
            for _ in captures
        ],
        "resume_seed": [
            np.asarray(_scalar(item["snapshot"], "resume_seed", 0), dtype=np.int64)
            for _ in captures
        ],
        "resume_lagrangian_seed": [
            np.asarray(
                _scalar(item["snapshot"], "resume_lagrangian_seed", 0),
                dtype=np.int64,
            )
            for _ in captures
        ],
        "state_t": [
            np.asarray(capture["state"].get("t", 0), dtype=np.int32)
            for capture in captures
        ],
        "state_mass_cycle_start": [
            np.asarray(
                capture["state"].get(
                    "mass_cycle_start",
                    np.sum(capture["state"]["A"]),
                ),
                dtype=np.float32,
            )
            for capture in captures
        ],
    }
    save_chunk(
        str(apf_dir),
        float(getattr(item["args"], "fps", 250.0)),
        absolute_steps,
        [capture["state"]["P"] for capture in captures],
        0,
        [capture["state"]["A"] for capture in captures],
        [capture["state"]["F"] for capture in captures],
        use_fp16=True,
        compress=True,
        extra_payload={
            key: np.asarray(values) for key, values in resume_meta.items()
        },
    )
    shutil.copy2(item["branch_dir"] / "config.yaml", output_dir / "config.yaml")
    shutil.copy2(item["branch_dir"] / "params.npy", output_dir / "params.npy")
    audit_keys = {"A", "P", "F", "resume_batch_rng_key", "steps"}
    free_arrays = _branch_arrays(item["branch_dir"], keys=audit_keys)
    wall_arrays = _branch_arrays(output_dir, keys=audit_keys)
    rng_exact = bool(
        np.array_equal(
            free_arrays["resume_batch_rng_key"],
            wall_arrays["resume_batch_rng_key"],
        )
    )
    start_keys = ("A", "P", "F")
    start_exact = {
        key: bool(np.array_equal(free_arrays[key][0], wall_arrays[key][0]))
        for key in start_keys
    }
    output_chunks = list_apf_chunks(apf_dir)
    if len(output_chunks) != 1:
        raise RuntimeError(
            f"Expected one wall APF chunk for row {row['row_id']}, "
            f"found {len(output_chunks)}"
        )
    free_chunks = list_apf_chunks(item["branch_dir"] / "apf_logs")
    if len(free_chunks) != 1:
        raise RuntimeError(
            f"Expected one free APF chunk for row {row['row_id']}, "
            f"found {len(free_chunks)}"
        )
    metadata = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "simulation_config_sha256": row[
            "source_simulation_config_sha256"
        ],
        "row_id": _as_int(row["row_id"]),
        "simulation_identity": {key: row[key] for key in SIMULATION_ID_FIELDS},
        "free_branch_dir": str(item["branch_dir"]),
        "free_branch_resume_metadata_sha256": _sha256_file(
            item["branch_dir"] / "resume_metadata.json"
        ),
        "source_apf_path": item["source_apf_path"],
        "source_snapshot_idx": item["source_snapshot_idx"],
        "start_step": _as_int(row["step"]),
        "end_step": _as_int(row["step"]) + HORIZON_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "walls_removed_relative_step": WALL_STEPS,
        "wall_steps": WALL_STEPS,
        "free_after_walls_steps": HORIZON_STEPS - WALL_STEPS,
        "capture_relative_steps": relative_steps.tolist(),
        "params_sha256": _sha256_file(output_dir / "params.npy"),
        "config_sha256": _sha256_file(output_dir / "config.yaml"),
        "free_apf_sha256": _sha256_file(free_chunks[0][0]),
        "wall_apf_sha256": _sha256_file(output_chunks[0][0]),
        "outer_batch": batch_context,
        "optimizer_native_batch_size": int(item["original_batch_size"]),
        "optimizer_native_batch_index": int(item["original_batch_index"]),
        "jit_microbatch_steps": int(item["jit_microbatch"]),
        "wall_stochastic_forcing": {
            "block_local_mutation_enabled": False,
            "global_mutation_events_per_step": 1,
            "global_mutation_grid_size": geometry.grid_size,
            "event_key": "exact selected optimizer-native lane key",
            "delta_application": (
                "single global mutation delta partitioned across block cores"
            ),
            "rt_categorical_noise": (
                "native PRNGKey(42) global Gumbels partitioned exactly into "
                "corresponding hard-wall core coordinates"
            ),
        },
        "initial_start_exact_vs_free": start_exact,
        "all_initial_start_exact_vs_free": bool(all(start_exact.values())),
        "top_level_rng_stream_exact_vs_free_at_all_captures": rng_exact,
        "split_merge_roundtrip": roundtrip,
        "geometry": {
            "grid_size": geometry.grid_size,
            "grid_split": geometry.split_n,
            "wall_pad": geometry.wall_pad,
            "block_size": geometry.block_size,
            "padded_grid_size": geometry.padded_grid_size,
            "crop_before": geometry.crop_before,
            "crop_after": geometry.crop_after,
        },
        "rng_semantics": (
            "At each 50-step chunk the folded branch RNG is split exactly as in "
            "flowlenia_minibang_resume_batch. Per-step keys are selected with the "
            "original optimization batch_size/index. During confinement, block-local "
            "mutation is disabled and each selected lane key generates exactly one "
            "global mutation delta which is partitioned over the nine cores. After "
            "merge, the selected lane keys are consumed directly again."
        ),
        "passive_tracker": (
            "Disabled in the walls arm. C5 uses A/P visual and field trajectories; "
            "the lagrangian tracker is a passive observer and does not feed back "
            "into Flow-Lenia state dynamics. The full-tracker free sham is audited "
            "separately against authoritative C2 outputs."
        ),
        "wall_semantics": (
            "The 128x128 state is centered in a 132x132 partition, split into 3x3 "
            "44-cell cores with five hard-zero wall cells around each core. Padding "
            "is zeroed after every transition. At relative step 10000, block cores "
            "are assembled and cropped back to 128x128."
        ),
        "final_state_sha256": {
            key: _sha256_array(captures[-1]["state"][key])
            for key in ("A", "P", "F")
        },
    }
    if not metadata["all_initial_start_exact_vs_free"]:
        raise RuntimeError(f"Wall branch start differs from free for row {row['row_id']}")
    if not rng_exact:
        raise RuntimeError(f"Wall branch RNG stream differs from free for row {row['row_id']}")
    _write_json(output_dir / "wall_metadata.json", metadata)


def _wall_output_audit(
    row: dict[str, str],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    output_dir = _resolve(row["walls_branch_dir"])
    metadata_path = output_dir / "wall_metadata.json"
    result = {"ready": False, "branch_dir": str(output_dir), "reason": ""}
    if not metadata_path.exists():
        result["reason"] = "missing wall_metadata.json"
        return result
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "simulation_config_sha256": row[
            "source_simulation_config_sha256"
        ],
        "row_id": _as_int(row["row_id"]),
        "start_step": _as_int(row["step"]),
        "end_step": _as_int(row["step"]) + HORIZON_STEPS,
        "walls_removed_relative_step": WALL_STEPS,
        "params_sha256": row["params_sha256"],
        "config_sha256": row["source_config_sha256"],
        "optimizer_native_batch_size": OPTIMIZER_NATIVE_BATCH_SIZE,
        "jit_microbatch_steps": JIT_MICROBATCH,
        "all_initial_start_exact_vs_free": True,
        "top_level_rng_stream_exact_vs_free_at_all_captures": True,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        result["reason"] = f"metadata mismatch: {mismatches}"
        return result
    forcing = metadata.get("wall_stochastic_forcing", {})
    expected_forcing = {
        "block_local_mutation_enabled": False,
        "global_mutation_events_per_step": 1,
        "global_mutation_grid_size": int(
            protocol.get("grid_size", 128)
        ),
        "event_key": "exact selected optimizer-native lane key",
        "delta_application": (
            "single global mutation delta partitioned across block cores"
        ),
        "rt_categorical_noise": (
            "native PRNGKey(42) global Gumbels partitioned exactly into "
            "corresponding hard-wall core coordinates"
        ),
    }
    forcing_mismatches = {
        key: {"expected": value, "actual": forcing.get(key)}
        for key, value in expected_forcing.items()
        if forcing.get(key) != value
    }
    outer_batch = metadata.get("outer_batch", {})
    if forcing_mismatches:
        result["reason"] = (
            f"wall stochastic forcing mismatch: {forcing_mismatches}"
        )
        return result
    if outer_batch.get("fixed_outer_batch_size") != SIMULATION_BATCH_SIZE:
        result["reason"] = (
            f"outer batch size mismatch: {outer_batch}"
        )
        return result
    roundtrip = metadata.get("split_merge_roundtrip", {})
    if (
        roundtrip.get("hard_wall_padding_zero_at_release") is not True
        or not all(
            roundtrip.get("hard_wall_padding_zero_fields", {}).values()
        )
    ):
        result["reason"] = (
            f"hard-wall padding audit failed: {roundtrip}"
        )
        return result
    lane_index = int(outer_batch.get("lane_index", -1))
    batch_row_ids = [
        int(value) for value in outer_batch.get("real_row_ids", [])
    ]
    if (
        lane_index < 0
        or lane_index >= len(batch_row_ids)
        or batch_row_ids[lane_index] != _as_int(row["row_id"])
        or int(outer_batch.get("real_rows_in_batch", -1))
        != len(batch_row_ids)
    ):
        result["reason"] = f"outer batch membership mismatch: {outer_batch}"
        return result
    try:
        audit_keys = {
            "A",
            "P",
            "F",
            "resume_batch_rng_key",
            "resume_batch_size",
            "resume_batch_index",
            "resume_jit_microbatch",
            "steps",
        }
        arrays = _branch_arrays(output_dir, keys=audit_keys)
        free = _branch_arrays(
            _resolve(row["free_branch_dir"]),
            keys=audit_keys,
        )
    except Exception as exc:
        result["reason"] = f"APF load failed: {exc}"
        return result
    if not np.array_equal(arrays["steps"], free["steps"]):
        result["reason"] = "wall/free capture steps differ"
        return result
    if not np.array_equal(
        arrays["resume_batch_rng_key"],
        free["resume_batch_rng_key"],
    ):
        result["reason"] = "wall/free top-level RNG captures differ"
        return result
    for key in (
        "resume_batch_size",
        "resume_batch_index",
        "resume_jit_microbatch",
    ):
        if not np.array_equal(arrays[key], free[key]):
            result["reason"] = f"wall/free {key} captures differ"
            return result
    native_batch_sizes = np.asarray(
        free["resume_batch_size"],
        dtype=np.int64,
    ).reshape(-1)
    native_batch_indices = np.asarray(
        free["resume_batch_index"],
        dtype=np.int64,
    ).reshape(-1)
    native_jit_microbatches = np.asarray(
        free["resume_jit_microbatch"],
        dtype=np.int64,
    ).reshape(-1)
    metadata_batch_index = int(
        metadata.get("optimizer_native_batch_index", -1)
    )
    if (
        not np.all(
            native_batch_sizes == OPTIMIZER_NATIVE_BATCH_SIZE
        )
        or not np.all(native_batch_indices == metadata_batch_index)
        or metadata_batch_index < 0
        or metadata_batch_index >= OPTIMIZER_NATIVE_BATCH_SIZE
        or not np.all(native_jit_microbatches == JIT_MICROBATCH)
    ):
        result["reason"] = (
            "optimizer-native batch/JIT mismatch: "
            f"metadata_index={metadata_batch_index}, "
            f"sizes={np.unique(native_batch_sizes).tolist()}, "
            f"indices={np.unique(native_batch_indices).tolist()}, "
            f"jit={np.unique(native_jit_microbatches).tolist()}"
        )
        return result
    for key in ("A", "P", "F"):
        if not np.array_equal(arrays[key][0], free[key][0]):
            result["reason"] = f"wall/free relative-zero {key} differs"
            return result
    if not all(np.all(np.isfinite(arrays[key])) for key in ("A", "P", "F")):
        result["reason"] = "non-finite wall state"
        return result
    wall_chunks = list_apf_chunks(output_dir / "apf_logs")
    free_dir = _resolve(row["free_branch_dir"])
    free_chunks = list_apf_chunks(free_dir / "apf_logs")
    if len(wall_chunks) != 1 or len(free_chunks) != 1:
        result["reason"] = (
            f"expected one APF chunk per arm, found "
            f"wall={len(wall_chunks)} free={len(free_chunks)}"
        )
        return result
    artifact_hashes = {
        "params_sha256": _sha256_file(output_dir / "params.npy"),
        "config_sha256": _sha256_file(output_dir / "config.yaml"),
        "wall_apf_sha256": _sha256_file(wall_chunks[0][0]),
        "free_apf_sha256": _sha256_file(free_chunks[0][0]),
    }
    artifact_mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in artifact_hashes.items()
        if metadata.get(key) != value
    }
    if artifact_mismatches:
        result["reason"] = f"artifact hash mismatch: {artifact_mismatches}"
        return result
    result.update(
        {
            "ready": True,
            "reason": "",
            "steps": arrays["steps"].tolist(),
            "metadata_sha256": _sha256_file(metadata_path),
            "initial_state_exact": True,
            "top_level_rng_stream_exact": True,
            "global_mutation_stream_exact": True,
            "stochastic_forcing_exact": True,
            "optimizer_native_batch_size": int(
                native_batch_sizes[0]
            ),
            "optimizer_native_batch_index": int(
                native_batch_indices[0]
            ),
            "optimizer_native_batch_index_exact": True,
            "jit_microbatch_steps": int(
                native_jit_microbatches[0]
            ),
        }
    )
    return result


def _run_simulation_batch(
    raw_rows: list[dict[str, str]],
    *,
    protocol: dict[str, Any],
    batch_size: int,
    mode: str,
    output_root: Path,
    shared_engine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from flowlenia_c5_full_length_delta_h_one import _pack_wall_particles
    from flowlenia_minibang_resume_batch import _make_batched_stepper

    first_free = _resolve(raw_rows[0]["free_branch_dir"])
    if shared_engine is None:
        params_example = np.asarray(
            np.load(first_free / "params.npy"),
            dtype=np.float32,
        )
        runtime = _runtime(first_free / "config.yaml", params_example)
    else:
        runtime = shared_engine["runtime"]
    snapshot_cache = (
        shared_engine["snapshot_cache"] if shared_engine is not None else {}
    )
    params_cache = (
        shared_engine["params_cache"] if shared_engine is not None else {}
    )
    state_template_cache = (
        shared_engine["state_template_cache"]
        if shared_engine is not None
        else {}
    )
    unpadded_items = [
        _load_simulation_item(
            row,
            runtime["substrate"],
            snapshot_cache=snapshot_cache,
            params_cache=params_cache,
            state_template_cache=state_template_cache,
        )
        for row in raw_rows
    ]
    simulation_config_hashes = {
        row["source_simulation_config_sha256"] for row in raw_rows
    }
    if len(simulation_config_hashes) != 1:
        raise RuntimeError(
            "One outer batch contains multiple simulation configurations: "
            f"{sorted(simulation_config_hashes)}"
        )
    if (
        shared_engine is not None
        and next(iter(simulation_config_hashes))
        != shared_engine["simulation_config_sha256"]
    ):
        raise RuntimeError(
            "Shared wall engine simulation configuration differs from batch"
        )
    items, real_n = _pad_items(unpadded_items, batch_size)
    for item in items:
        item["args"] = runtime["args"]
    signatures = {
        (
            item["original_batch_size"],
            item["snapshot_interval"],
            item["jit_microbatch"],
            item["points"].shape,
            tuple(item["params"].shape),
        )
        for item in items
    }
    if len(signatures) != 1:
        raise RuntimeError(f"Incompatible jobs in one batch: {signatures}")
    original_batch_size = int(items[0]["original_batch_size"])
    if int(items[0]["jit_microbatch"]) != JIT_MICROBATCH:
        raise RuntimeError(
            f"Expected optimizer-equivalent jit_microbatch={JIT_MICROBATCH}, "
            f"found {items[0]['jit_microbatch']}"
        )
    capture_steps = _capture_steps_from_free(raw_rows[0])
    for row in raw_rows[1:]:
        if not np.array_equal(_capture_steps_from_free(row), capture_steps):
            raise RuntimeError("C2 free branches do not share capture offsets")

    initial_states = [item["state"] for item in items]
    state = _stack_trees(initial_states)
    points = jnp.stack([jnp.asarray(item["points"]) for item in items], axis=0)
    channels = jnp.stack(
        [jnp.asarray(item["channels"]) for item in items],
        axis=0,
    )
    rng = jnp.stack([jnp.asarray(item["rng"]) for item in items], axis=0)
    params = jnp.stack(
        [jnp.asarray(item["params"], dtype=jnp.float32) for item in items],
        axis=0,
    )
    original_indices = jnp.asarray(
        [item["original_batch_index"] for item in items],
        dtype=jnp.int32,
    )
    if shared_engine is not None:
        if int(shared_engine["original_batch_size"]) != original_batch_size:
            raise RuntimeError("Shared wall engine has the wrong original batch size")
        global_state_stepper = shared_engine["global_state_stepper"]
        block_state_stepper = shared_engine["block_state_stepper"]
        controlled_global_state_stepper = None
        global_stepper = None
    else:
        global_stepper = _make_batched_stepper(
            substrate=runtime["substrate"],
            rt=runtime["substrate"].RT,
            original_batch_size=original_batch_size,
            lag_flow_channel=int(
                getattr(runtime["args"], "lagrangian_flow_channel", -1)
            ),
            lag_flow_reduce=str(
                getattr(
                    runtime["args"],
                    "lagrangian_flow_reduce",
                    "mass_weighted",
                )
            ),
            lag_channel_mode=str(
                getattr(runtime["args"], "lagrangian_channel_mode", "resample")
            ),
            lag_noise_model=str(
                getattr(runtime["args"], "lagrangian_noise_model", "rt_box")
            ),
            lag_diffusion_scale=float(
                getattr(runtime["args"], "lagrangian_diffusion_scale", 1.0)
            ),
        )
        global_state_stepper = _make_global_state_stepper(
            runtime["substrate"],
            original_batch_size=original_batch_size,
        )
        controlled_global_state_stepper = (
            _make_controlled_global_state_stepper(
                runtime["deterministic_substrate"],
                original_batch_size=original_batch_size,
                mutation_spec=runtime["mutation_spec"],
                global_rt_gumbel=runtime["global_rt_gumbel"],
            )
        )
        block_state_stepper = _make_block_state_stepper(
            runtime["block_substrate"],
            n_blocks=runtime["geometry"].n_blocks,
            original_batch_size=original_batch_size,
            valid_mask=runtime["valid_mask"],
            geometry=runtime["geometry"],
            mutation_spec=runtime["mutation_spec"],
            block_rt_gumbel=runtime["block_rt_gumbel"],
        )
    roundtrip = {"all_ap_exact": True, "items": []}
    if mode == "walls":
        state, roundtrip = _prepare_block_state_batch(
            items,
            runtime,
        )
        if not roundtrip["all_ap_exact"]:
            raise RuntimeError("Block split/merge is not exact for A/P")
        phase = "blocks"
    elif mode == "sham":
        phase = "global"
    elif mode == "controlled_sham":
        phase = "controlled_global"
    else:
        raise ValueError(mode)

    captures: list[list[dict[str, Any]]] = [[] for _ in range(real_n)]
    if mode in {"walls", "controlled_sham"}:
        current = _capture_global_state(
            _stack_trees(initial_states),
            rng,
            real_n=real_n,
        )
    else:
        current = _capture_global(
            state,
            points,
            channels,
            rng,
            real_n=real_n,
        )
    for idx in range(real_n):
        captures[idx].append(current[idx])

    rel_step = 0
    capture_set = set(int(value) for value in capture_steps[1:])
    events = sorted(
        capture_set
        | (
            {WALL_STEPS}
            if mode in {"walls", "controlled_sham"}
            else set()
        )
    )
    start_time = time.monotonic()
    for target in events:
        while rel_step < target:
            n_steps = min(JIT_MICROBATCH, target - rel_step)
            rng, subkeys = _split_rng_batch(rng)
            if phase == "blocks":
                assert block_state_stepper is not None
                state = block_state_stepper(n_steps)(
                    state,
                    subkeys,
                    params,
                    original_indices,
                )
            elif phase == "controlled_global":
                assert controlled_global_state_stepper is not None
                state = controlled_global_state_stepper(n_steps)(
                    state,
                    subkeys,
                    params,
                    original_indices,
                )
            elif mode in {"walls", "controlled_sham"}:
                state = global_state_stepper(n_steps)(
                    state,
                    subkeys,
                    params,
                    original_indices,
                )
            else:
                assert global_stepper is not None
                state, points, channels = global_stepper(n_steps)(
                    state,
                    points,
                    channels,
                    subkeys,
                    params,
                    original_indices,
                )
            rel_step += n_steps
        if mode == "walls" and rel_step == WALL_STEPS and phase == "blocks":
            block_states_host = _unstack_tree(jax.device_get(state))
            valid_mask = np.asarray(
                jax.device_get(runtime["valid_mask"]),
                dtype=bool,
            )
            global_states = []
            for idx in range(len(items)):
                zero_fields = {}
                for key in ("A", "P", "F", "Food"):
                    if key not in block_states_host[idx]:
                        continue
                    value = np.asarray(block_states_host[idx][key])
                    if tuple(value.shape[:3]) != tuple(valid_mask.shape):
                        continue
                    expanded_mask = valid_mask.reshape(
                        valid_mask.shape
                        + (1,) * (value.ndim - valid_mask.ndim)
                    )
                    zero_fields[key] = bool(
                        np.all(np.where(expanded_mask, 0, value) == 0)
                    )
                hard_wall_zero = bool(
                    zero_fields and all(zero_fields.values())
                )
                roundtrip["items"][idx][
                    "hard_wall_padding_zero_at_release"
                ] = hard_wall_zero
                roundtrip["items"][idx][
                    "hard_wall_padding_zero_fields"
                ] = zero_fields
                if not hard_wall_zero:
                    raise RuntimeError(
                        "Hard-wall padding is nonzero before release for "
                        f"row {items[idx]['row']['row_id']}: {zero_fields}"
                    )
                global_states.append(
                    _merge_block_state(
                        initial_states[idx],
                        block_states_host[idx],
                        geometry=runtime["geometry"],
                    )
                )
            state = _stack_trees(global_states)
            phase = "global"
        elif (
            mode == "controlled_sham"
            and rel_step == WALL_STEPS
            and phase == "controlled_global"
        ):
            phase = "global"
        if target in capture_set:
            if mode == "walls" and phase == "blocks":
                current = _capture_block_state(
                    state,
                    rng,
                    initial_states=initial_states,
                    geometry=runtime["geometry"],
                    real_n=real_n,
                )
            elif mode in {"walls", "controlled_sham"}:
                current = _capture_global_state(
                    state,
                    rng,
                    real_n=real_n,
                )
            else:
                current = _capture_global(
                    state,
                    points,
                    channels,
                    rng,
                    real_n=real_n,
                )
            for idx in range(real_n):
                captures[idx].append(current[idx])
        elapsed = time.monotonic() - start_time
        rate = (rel_step * max(1, real_n)) / max(elapsed, 1e-9)
        print(
            f"[{mode}] B={real_n}/{batch_size} step={rel_step}/{HORIZON_STEPS} "
            f"lane-step/s={rate:.1f}",
            flush=True,
        )
    if rel_step != HORIZON_STEPS:
        raise RuntimeError(f"Simulation ended at {rel_step}, expected {HORIZON_STEPS}")

    if mode == "walls":
        batch_row_ids = [
            _as_int(item["row"]["row_id"]) for item in unpadded_items
        ]
        for idx, item in enumerate(unpadded_items):
            _write_wall_branch(
                item,
                captures[idx],
                capture_steps,
                protocol=protocol,
                roundtrip=roundtrip["items"][idx],
                geometry=runtime["geometry"],
                batch_context={
                    "fixed_outer_batch_size": int(batch_size),
                    "real_rows_in_batch": int(real_n),
                    "lane_index": int(idx),
                    "real_row_ids": batch_row_ids,
                    "padding": (
                        "repeat real rows cyclically to fixed outer batch size; "
                        "padded outputs are discarded"
                    ),
                },
            )
        return {
            "mode": mode,
            "n": real_n,
            "elapsed_seconds": time.monotonic() - start_time,
            "rows": [_as_int(item["row"]["row_id"]) for item in unpadded_items],
        }

    comparisons = []
    all_exact = True
    comparison_fields = (
        ("A", "P", "F")
        if mode == "controlled_sham"
        else ("A", "P", "F", "lagrangian_xy", "lagrangian_c")
    )
    for item, item_captures in zip(unpadded_items, captures, strict=True):
        expected = _branch_arrays(item["branch_dir"])
        fields: dict[str, Any] = {}
        for key in comparison_fields:
            actual = np.stack(
                [
                    capture["state"][key]
                    if key in {"A", "P", "F"}
                    else capture["points"]
                    if key == "lagrangian_xy"
                    else capture["channels"]
                    for capture in item_captures
                ],
                axis=0,
            )
            actual = actual.astype(expected[key].dtype, copy=False)
            exact = bool(np.array_equal(actual, expected[key]))
            max_abs = (
                float(
                    np.max(
                        np.abs(
                            actual.astype(np.float64)
                            - expected[key].astype(np.float64)
                        )
                    )
                )
                if np.issubdtype(actual.dtype, np.number)
                else 0.0
            )
            fields[key] = {"exact": exact, "max_abs": max_abs}
            all_exact = all_exact and exact
        actual_rng = np.stack(
            [capture["rng"] for capture in item_captures],
            axis=0,
        )
        rng_exact = bool(
            np.array_equal(actual_rng, expected["resume_batch_rng_key"])
        )
        all_exact = all_exact and rng_exact
        comparisons.append(
            {
                "row_id": _as_int(item["row"]["row_id"]),
                "fields": fields,
                "rng_exact": rng_exact,
            }
        )
    report = {
        "status": "passed" if all_exact else "failed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "mode": mode,
        "batch_size": batch_size,
        "n_real": real_n,
        "capture_relative_steps": capture_steps.tolist(),
        "all_exact": all_exact,
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    report_path = output_root / (
        "preflight_controlled_sham_exactness.json"
        if mode == "controlled_sham"
        else "preflight_sham_exactness.json"
    )
    _write_json(report_path, report)
    if not all_exact:
        raise RuntimeError(
            f"{mode} replay is not exact; refusing wall simulations"
        )
    return report


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require_simulation_batch_size(args)
    rows, protocol = _load_plan(args)
    run0_opt = [
        row
        for row in rows
        if _as_int(row["run_idx"]) == int(args.preflight_run)
        and row["candidate_kind"] == "optimized"
    ]
    run0_opt = run0_opt[: int(args.batch_size)]
    if len(run0_opt) < 1:
        raise RuntimeError("No optimized rows available for preflight")
    for row in run0_opt:
        audit = _free_output_audit(row)
        if not audit["ready"]:
            raise RuntimeError(f"Preflight free branch invalid: {audit}")
    output_root = _resolve(args.output_root)
    first_free = _resolve(run0_opt[0]["free_branch_dir"])
    params_example = np.asarray(
        np.load(first_free / "params.npy"),
        dtype=np.float32,
    )
    mutation_runtime = _runtime(
        first_free / "config.yaml",
        params_example,
    )
    mutation_item = _load_simulation_item(
        run0_opt[0],
        mutation_runtime["substrate"],
    )
    mutation_audit = _mutation_injection_audit(
        mutation_item,
        mutation_runtime,
        plan_sha256=protocol["plan_sha256"],
        output_root=output_root,
    )
    batch_topology_audit = _wall_batch_topology_audit(
        mutation_item,
        mutation_runtime,
        plan_sha256=protocol["plan_sha256"],
        output_root=output_root,
    )
    sham_path = output_root / "preflight_sham_exactness.json"
    sham = json.loads(sham_path.read_text()) if sham_path.exists() else None
    if not (
        isinstance(sham, dict)
        and sham.get("status") == "passed"
        and sham.get("protocol_version") == PROTOCOL_VERSION
        and sham.get("plan_sha256") == protocol["plan_sha256"]
        and sham.get("simulation_code_bundle_sha256")
        == protocol["simulation_code_bundle_sha256"]
        and sham.get("mode") == "sham"
        and int(sham.get("n_real", -1)) == len(run0_opt)
        and [int(row["row_id"]) for row in run0_opt]
        == [int(row["row_id"]) for row in sham.get("comparisons", [])]
    ):
        sham = _run_simulation_batch(
            run0_opt,
            protocol=protocol,
            batch_size=int(args.batch_size),
            mode="sham",
            output_root=output_root,
        )
    controlled_path = (
        output_root / "preflight_controlled_sham_exactness.json"
    )
    controlled_sham = (
        json.loads(controlled_path.read_text())
        if controlled_path.exists()
        else None
    )
    if not (
        isinstance(controlled_sham, dict)
        and controlled_sham.get("status") == "passed"
        and controlled_sham.get("protocol_version") == PROTOCOL_VERSION
        and controlled_sham.get("plan_sha256")
        == protocol["plan_sha256"]
        and controlled_sham.get("simulation_code_bundle_sha256")
        == protocol["simulation_code_bundle_sha256"]
        and controlled_sham.get("mode") == "controlled_sham"
        and int(controlled_sham.get("n_real", -1)) == len(run0_opt)
        and [int(row["row_id"]) for row in run0_opt]
        == [
            int(row["row_id"])
            for row in controlled_sham.get("comparisons", [])
        ]
    ):
        controlled_sham = _run_simulation_batch(
            run0_opt,
            protocol=protocol,
            batch_size=int(args.batch_size),
            mode="controlled_sham",
            output_root=output_root,
        )
    missing_walls = [
        row
        for row in run0_opt
        if not _wall_output_audit(
            row,
            protocol=protocol,
        )["ready"]
    ]
    wall_result = None
    if missing_walls:
        wall_result = _run_simulation_batch(
            missing_walls,
            protocol=protocol,
            batch_size=int(args.batch_size),
            mode="walls",
            output_root=output_root,
        )
    audits = [
        _wall_output_audit(row, protocol=protocol)
        for row in run0_opt
    ]
    failed = [audit for audit in audits if not audit["ready"]]
    if failed:
        raise RuntimeError(f"Wall preflight failed: {failed[:3]}")
    summary = {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "run_idx": int(args.preflight_run),
        "n_rows": len(run0_opt),
        "mutation_injection": mutation_audit,
        "batch_topology": batch_topology_audit,
        "sham": sham,
        "controlled_sham": controlled_sham,
        "walls": wall_result,
        "all_wall_outputs_valid": True,
    }
    _write_json(output_root / "preflight_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _create_wall_engine(row: dict[str, str]) -> dict[str, Any]:
    first_free = _resolve(row["free_branch_dir"])
    params = np.asarray(np.load(first_free / "params.npy"), dtype=np.float32)
    runtime = _runtime(first_free / "config.yaml", params)
    runtime_config = _simulation_config_fingerprint(
        first_free / "config.yaml"
    )
    if runtime_config["sha256"] != row["source_simulation_config_sha256"]:
        raise RuntimeError(
            "Wall engine runtime config does not match the paired plan"
        )
    snapshot_cache: dict[tuple[str, int], tuple[Path, int, dict[str, Any]]] = {}
    params_cache = {str(row["params_sha256"]): params}
    state_template_cache: dict[str, dict[str, Any]] = {}
    item = _load_simulation_item(
        row,
        runtime["substrate"],
        snapshot_cache=snapshot_cache,
        params_cache=params_cache,
        state_template_cache=state_template_cache,
    )
    original_batch_size = int(item["original_batch_size"])
    return {
        "runtime": runtime,
        "simulation_config_sha256": runtime_config["sha256"],
        "original_batch_size": original_batch_size,
        "global_state_stepper": _make_global_state_stepper(
            runtime["substrate"],
            original_batch_size=original_batch_size,
        ),
        "block_state_stepper": _make_block_state_stepper(
            runtime["block_substrate"],
            n_blocks=runtime["geometry"].n_blocks,
            original_batch_size=original_batch_size,
            valid_mask=runtime["valid_mask"],
            geometry=runtime["geometry"],
            mutation_spec=runtime["mutation_spec"],
            block_rt_gumbel=runtime["block_rt_gumbel"],
        ),
        "snapshot_cache": snapshot_cache,
        "params_cache": params_cache,
        "state_template_cache": state_template_cache,
    }


def run_walls(args: argparse.Namespace) -> dict[str, Any]:
    _require_simulation_batch_size(args)
    rows, protocol = _load_plan(args)
    preflight_path = _resolve(args.output_root) / "preflight_summary.json"
    if not preflight_path.exists():
        raise RuntimeError("Preflight is missing; run --phase preflight first")
    preflight_result = json.loads(preflight_path.read_text())
    preflight_expected = {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
    }
    preflight_mismatches = {
        key: {"expected": value, "actual": preflight_result.get(key)}
        for key, value in preflight_expected.items()
        if preflight_result.get(key) != value
    }
    mutation_preflight = preflight_result.get("mutation_injection", {})
    batch_preflight = preflight_result.get("batch_topology", {})
    controlled_preflight = preflight_result.get("controlled_sham", {})
    if (
        preflight_mismatches
        or mutation_preflight.get("status") != "passed"
        or not mutation_preflight.get("all_exact", False)
        or mutation_preflight.get("block_local_mutation_enabled") is not False
        or mutation_preflight.get(
            "native_global_rt_vs_external_gumbel_exact"
        )
        is not True
        or mutation_preflight.get(
            "global_rt_gumbel_partition_roundtrip_exact"
        )
        is not True
        or controlled_preflight.get("status") != "passed"
        or controlled_preflight.get("all_exact") is not True
        or batch_preflight.get("status") != "passed"
        or batch_preflight.get(
            "all_frozen_lanes_repeat_the_small_batch_exactly"
        )
        is not True
    ):
        raise RuntimeError(
            "Current RNG-matched preflight did not pass: "
            f"{preflight_mismatches or mutation_preflight or controlled_preflight}"
        )
    selected = _filter_rows(rows, args)
    for row in selected:
        free_audit = _free_output_audit(row)
        if not free_audit["ready"]:
            raise RuntimeError(
                f"Free branch row {row['row_id']} is not valid: {free_audit['reason']}"
            )
    pending = [
        row
        for row in selected
        if not _wall_output_audit(
            row,
            protocol=protocol,
        )["ready"]
    ]
    output_root = _resolve(args.output_root)
    progress_path = output_root / "walls_progress.json"
    completed_now = 0
    start_time = time.monotonic()
    shared_engine = _create_wall_engine(pending[0]) if pending else None
    for offset in range(0, len(pending), int(args.batch_size)):
        batch = pending[offset : offset + int(args.batch_size)]
        _run_simulation_batch(
            batch,
            protocol=protocol,
            batch_size=int(args.batch_size),
            mode="walls",
            output_root=output_root,
            shared_engine=shared_engine,
        )
        completed_now += len(batch)
        elapsed = time.monotonic() - start_time
        rate = completed_now / max(elapsed, 1e-9)
        remaining = (len(pending) - completed_now) / max(rate, 1e-9)
        progress = {
            "status": "running" if completed_now < len(pending) else "complete",
            "n_selected": len(selected),
            "n_cached_before": len(selected) - len(pending),
            "n_pending_at_start": len(pending),
            "n_completed_now": completed_now,
            "n_remaining": len(pending) - completed_now,
            "elapsed_seconds": elapsed,
            "eta_seconds": remaining,
        }
        _write_json(progress_path, progress)
        print(json.dumps(progress, indent=2), flush=True)
    audits = [
        _wall_output_audit(row, protocol=protocol)
        for row in selected
    ]
    failed = [audit for audit in audits if not audit["ready"]]
    if failed:
        raise RuntimeError(f"{len(failed)} wall branches failed final audit")
    summary = {
        "status": (
            "complete"
            if len(selected) == EXPECTED_PLAN_ROWS
            else "complete_subset"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "full_protocol_scope": len(selected) == EXPECTED_PLAN_ROWS,
        "n_selected": len(selected),
        "n_cached_before": len(selected) - len(pending),
        "n_generated": len(pending),
        "elapsed_seconds": time.monotonic() - start_time,
    }
    _write_json(output_root / "walls_summary.json", summary)
    return summary


def protocol_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows, protocol = _load_plan(args)
    selected = _filter_rows(rows, args)
    free_cache_path = (
        _resolve(args.output_root) / "free_cache_equivalence_audit.json"
    )
    free_cache_audit = (
        json.loads(free_cache_path.read_text())
        if free_cache_path.exists()
        else {}
    )
    free_cache_exact = bool(
        free_cache_audit.get("status") == "passed"
        and free_cache_audit.get("plan_sha256") == protocol["plan_sha256"]
        and int(free_cache_audit.get("batch_size", -1))
        == SIMULATION_BATCH_SIZE
        and int(free_cache_audit.get("n_compared", -1))
        == SIMULATION_BATCH_SIZE
        and free_cache_audit.get("all_exact") is True
        and free_cache_audit.get("runner_sha256")
        == _sha256_file(
            _REPO_ROOT / "scripts/flowlenia_minibang_resume_batch.py"
        )
    )
    preflight_path = _resolve(args.output_root) / "preflight_summary.json"
    preflight_report = (
        json.loads(preflight_path.read_text())
        if preflight_path.exists()
        else {}
    )
    mutation_preflight = preflight_report.get(
        "mutation_injection", {}
    )
    batch_preflight = preflight_report.get("batch_topology", {})
    controlled_preflight = preflight_report.get(
        "controlled_sham", {}
    )
    native_preflight = preflight_report.get("sham", {})
    preflight_exact = bool(
        preflight_report.get("status") == "passed"
        and preflight_report.get("protocol_version") == PROTOCOL_VERSION
        and preflight_report.get("plan_sha256") == protocol["plan_sha256"]
        and preflight_report.get("simulation_code_bundle_sha256")
        == protocol["simulation_code_bundle_sha256"]
        and mutation_preflight.get("all_exact") is True
        and mutation_preflight.get(
            "native_global_rt_vs_external_gumbel_exact"
        )
        is True
        and mutation_preflight.get(
            "global_rt_gumbel_partition_roundtrip_exact"
        )
        is True
        and native_preflight.get("all_exact") is True
        and controlled_preflight.get("all_exact") is True
        and batch_preflight.get("status") == "passed"
        and batch_preflight.get(
            "all_frozen_lanes_repeat_the_small_batch_exactly"
        )
        is True
    )
    free_audits = [_free_output_audit(row) for row in selected]
    wall_audits = [
        _wall_output_audit(row, protocol=protocol)
        for row in selected
    ]
    free_ready = sum(bool(value["ready"]) for value in free_audits)
    wall_ready = sum(bool(value["ready"]) for value in wall_audits)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "hostname": platform.node(),
    }
    try:
        import jax

        environment.update(
            {
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(value) for value in jax.devices()],
            }
        )
    except Exception as exc:
        environment["jax_error"] = str(exc)
    try:
        ptxas = subprocess.run(
            ["ptxas", "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        environment["ptxas_version"] = (
            ptxas.stdout.strip() or ptxas.stderr.strip()
        )
    except Exception as exc:
        environment["ptxas_error"] = str(exc)
    failures = {
        "free": [
            {"row_id": selected[idx]["row_id"], **audit}
            for idx, audit in enumerate(free_audits)
            if not audit["ready"]
        ],
        "walls": [
            {"row_id": selected[idx]["row_id"], **audit}
            for idx, audit in enumerate(wall_audits)
            if not audit["ready"]
        ],
    }
    all_selected_ready = (
        free_ready == len(selected)
        and wall_ready == len(selected)
        and free_cache_exact
        and preflight_exact
    )
    all_params_exact = bool(
        len(free_audits) == len(selected)
        and all(
            audit.get("ready") is True
            and audit.get("params_sha256") == row["params_sha256"]
            for row, audit in zip(
                selected,
                free_audits,
                strict=True,
            )
        )
    )
    all_initial_states_exact = bool(
        len(wall_audits) == len(selected)
        and all(
            audit.get("initial_state_exact") is True
            for audit in wall_audits
        )
    )
    all_top_level_rng_streams_exact = bool(
        len(wall_audits) == len(selected)
        and all(
            audit.get("top_level_rng_stream_exact") is True
            for audit in wall_audits
        )
    )
    all_optimizer_native_batch_indices_exact = bool(
        len(free_audits) == len(selected)
        and len(wall_audits) == len(selected)
        and all(
            free_audit.get(
                "optimizer_native_batch_index_exact"
            )
            is True
            and wall_audit.get(
                "optimizer_native_batch_index_exact"
            )
            is True
            and free_audit.get("optimizer_native_batch_index")
            == wall_audit.get("optimizer_native_batch_index")
            for free_audit, wall_audit in zip(
                free_audits,
                wall_audits,
                strict=True,
            )
        )
    )
    all_global_mutation_streams_exact = bool(
        preflight_exact
        and all_top_level_rng_streams_exact
        and all_optimizer_native_batch_indices_exact
        and len(wall_audits) == len(selected)
        and all(
            audit.get("global_mutation_stream_exact") is True
            and audit.get("stochastic_forcing_exact") is True
            for audit in wall_audits
        )
    )
    all_selected_ready = bool(
        all_selected_ready
        and all_params_exact
        and all_initial_states_exact
        and all_top_level_rng_streams_exact
        and all_optimizer_native_batch_indices_exact
        and all_global_mutation_streams_exact
    )
    full_scope = len(selected) == EXPECTED_PLAN_ROWS
    result = {
        "status": (
            "passed"
            if all_selected_ready and full_scope
            else "passed_subset"
            if all_selected_ready
            else "incomplete"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": protocol["plan_sha256"],
        "n_selected": len(selected),
        "n_free_ready": free_ready,
        "n_walls_ready": wall_ready,
        "n_optimized_rows": sum(
            row["candidate_kind"] == "optimized" for row in selected
        ),
        "n_random_rows": sum(
            row["candidate_kind"] == "random" for row in selected
        ),
        "full_protocol_scope": full_scope,
        "simulation_code_bundle_sha256": protocol[
            "simulation_code_bundle_sha256"
        ],
        "simulation_code_fingerprint_exact": (
            protocol["simulation_code_bundle_sha256"]
            == _simulation_code_fingerprint()["bundle_sha256"]
        ),
        "simulation_config_sha256": protocol[
            "simulation_config_sha256"
        ],
        "all_simulation_configs_exact": all(
            row["source_simulation_config_sha256"]
            == protocol["simulation_config_sha256"]
            for row in selected
        ),
        "all_params_exact": all_params_exact,
        "all_initial_states_exact": all_initial_states_exact,
        "all_top_level_rng_streams_exact": (
            all_top_level_rng_streams_exact
        ),
        "all_optimizer_native_batch_indices_exact": (
            all_optimizer_native_batch_indices_exact
        ),
        "all_global_mutation_streams_exact": (
            all_global_mutation_streams_exact
        ),
        "free_cache_equivalence_exact": free_cache_exact,
        "free_cache_equivalence_audit": str(free_cache_path),
        "preflight_exact": preflight_exact,
        "preflight_summary": str(preflight_path),
        "failures": failures,
        "environment": environment,
    }
    output = _resolve(args.output_root) / "protocol_audit.json"
    _write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    if args.require_complete and result["status"] != "passed":
        raise RuntimeError("Protocol audit is incomplete")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow-Lenia C5 paired C2 free versus walls-then-free experiment."
    )
    parser.add_argument(
        "--phase",
        choices=(
            "plan",
            "free",
            "free-cache-audit",
            "preflight",
            "walls",
            "audit",
        ),
        required=True,
    )
    parser.add_argument("--paper-config", type=Path, default=DEFAULT_PAPER_CONFIG)
    parser.add_argument("--c2-root", type=Path, default=DEFAULT_C2_ROOT)
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=SIMULATION_BATCH_SIZE)
    parser.add_argument("--preflight-run", type=int, default=0)
    parser.add_argument("--run-indices", type=str, default="")
    parser.add_argument(
        "--candidate-kinds",
        type=str,
        default="all",
        help="all, optimized, random, or a comma-separated subset",
    )
    parser.add_argument("--candidate-ids", type=str, default="")
    parser.add_argument("--conditions", type=str, default="all")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _enable_compilation_cache(output_root: Path) -> None:
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    cache_dir = output_root / "jax_compilation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    compilation_cache.set_cache_dir(str(cache_dir))


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    _enable_compilation_cache(_resolve(args.output_root))
    if args.phase == "plan":
        build_plan(args)
    elif args.phase == "free":
        generate_free_random(args)
    elif args.phase == "free-cache-audit":
        free_cache_equivalence_audit(args)
    elif args.phase == "preflight":
        preflight(args)
    elif args.phase == "walls":
        run_walls(args)
    elif args.phase == "audit":
        protocol_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
