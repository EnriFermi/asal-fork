from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from flowlenia_minibang_common import list_apf_chunks, load_config
from flowlenia_minibang_simulate import _make_substrate, _metric_roll_key


PROTOCOL_VERSION = "flowlenia-rng-sensitivity-v1"

C1_SCORES = _REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper/"
    "flow_lenia/checkpoint_scores.csv"
)
C1_TRAJECTORY_ROOT = _REPO_ROOT / (
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt_c1_argmax_paper/"
    "c1_lagrangian_apf_300k_train50_4seeds_exact_parallel_zip"
)
C5_PLAN = _REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "flow_lenia/c5_rng_only_mass_preserving_horizon_grid_v2/plan.csv"
)
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / (
    "analysis/results/flowlenia_rng_sensitivity_"
    "trajectory20_shared4_9branch_10k_v1"
)

SOURCE_STEPS = (50_000, 100_000, 150_000, 200_000, 250_000)
ROLLOUT_SEED_INDICES = (0, 1, 2, 3)
SHARED_RUN_SEEDS = (400_003, 400_004, 400_005, 400_006)
HORIZON_STEPS = 10_000
STEP_CHUNK = 50
METRIC_STEPS = tuple(range(0, HORIZON_STEPS + STEP_CHUNK, STEP_CHUNK))
VISUAL_STEPS = tuple(range(0, HORIZON_STEPS + 250, 250))
HORIZON_SUMMARY_STEPS = tuple(range(0, HORIZON_STEPS + 1_000, 1_000))

N_UNIQUE_BRANCHES = 9
N_BRANCHES = 10
DUPLICATE_BRANCH = 9
DUPLICATE_OF_BRANCH = 0
BRANCH_SEED_BASE = 6_700_031
CONTEXTS_PER_BATCH = 12
N_CONTEXTS = 24
SIMULATION_BATCH_SIZE = CONTEXTS_PER_BATCH * N_BRANCHES

PAIR_LEFT = np.asarray(
    [i for i in range(N_UNIQUE_BRANCHES) for j in range(i + 1, N_UNIQUE_BRANCHES)],
    dtype=np.int32,
)
PAIR_RIGHT = np.asarray(
    [j for i in range(N_UNIQUE_BRANCHES) for j in range(i + 1, N_UNIQUE_BRANCHES)],
    dtype=np.int32,
)
N_PAIRS = len(PAIR_LEFT)

PRIMARY_ARM = "trajectory"
PRIMARY_METRIC = "a_relative_l1"
PILOT_CANDIDATE = "run_000_optimized"

CODE_FILES = (
    Path(__file__),
    _REPO_ROOT / "scripts/flowlenia_minibang_simulate.py",
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def _code_fingerprint() -> dict[str, Any]:
    files = {str(path.relative_to(_REPO_ROOT)): _sha256_file(path) for path in CODE_FILES}
    return {"files": files, "bundle_sha256": _identity_sha256(files)}


def _candidate_id(row: dict[str, str]) -> str:
    run_idx = int(row["optimized_run_idx"])
    if row["candidate_kind"] == "optimized":
        return f"run_{run_idx:03d}_optimized"
    return f"run_{run_idx:03d}_random_{int(row['candidate_idx']):03d}"


def _find_source_chunk(traj_dir: Path, step: int) -> Path:
    hits = [
        path
        for path, start, end, _index in list_apf_chunks(traj_dir / "apf_logs")
        if int(start) <= int(step) <= int(end)
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one APF chunk for {traj_dir.name} step={step}, found {hits}"
        )
    return hits[0].resolve()


def _simulation_config_identity(flat_args: dict[str, Any]) -> dict[str, Any]:
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


def build_plan(output_root: Path) -> dict[str, Any]:
    for path in (C1_SCORES, C5_PLAN, C1_TRAJECTORY_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)

    c5_rows = _read_csv(C5_PLAN)
    c5_candidates: dict[str, dict[str, str]] = {}
    for row in c5_rows:
        candidate_id = row["candidate_id"]
        identity = {
            "candidate_id": candidate_id,
            "candidate_kind": row["candidate_kind"],
            "run_idx": row["run_idx"],
            "candidate_idx": row["candidate_idx"],
            "params_path": str(_resolve(row["params_path"])),
            "params_sha256": row["params_sha256"],
            "simulation_config_sha256": row["source_simulation_config_sha256"],
        }
        previous = c5_candidates.setdefault(candidate_id, identity)
        if previous != identity:
            raise RuntimeError(f"Inconsistent C5 identity for {candidate_id}")

    if len(c5_candidates) != 40:
        raise RuntimeError(f"Expected 40 candidates, found {len(c5_candidates)}")
    kinds = [row["candidate_kind"] for row in c5_candidates.values()]
    if kinds.count("optimized") != 10 or kinds.count("random") != 30:
        raise RuntimeError("Expected 10 optimized and 30 random candidates")
    sim_hashes = {row["simulation_config_sha256"] for row in c5_candidates.values()}
    if len(sim_hashes) != 1:
        raise RuntimeError(f"Candidates have different simulation configs: {sim_hashes}")

    c1_rows = _read_csv(C1_SCORES)
    c1_by_candidate: dict[str, list[dict[str, str]]] = {}
    for row in c1_rows:
        candidate_id = _candidate_id(row)
        c1_by_candidate.setdefault(candidate_id, []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for candidate_id, base in sorted(
        c5_candidates.items(),
        key=lambda item: (
            int(item[1]["run_idx"]),
            0 if item[1]["candidate_kind"] == "optimized" else 1,
            int(item[1]["candidate_idx"]),
        ),
    ):
        params_path = _resolve(base["params_path"])
        if not params_path.exists():
            raise FileNotFoundError(params_path)
        actual_params_sha = _sha256_file(params_path)
        if actual_params_sha != base["params_sha256"]:
            raise RuntimeError(
                f"Parameter hash mismatch for {candidate_id}: "
                f"{actual_params_sha} != {base['params_sha256']}"
            )
        source_rows = sorted(
            c1_by_candidate.get(candidate_id, []),
            key=lambda row: int(row["rollout_seed_idx"]),
        )
        if [int(row["rollout_seed_idx"]) for row in source_rows] != list(
            ROLLOUT_SEED_INDICES
        ):
            raise RuntimeError(f"{candidate_id} does not have exactly four C1 seeds")

        c1_mspd = np.asarray(
            [float(row["full_score_train_tau_mspd"]) for row in source_rows],
            dtype=np.float64,
        )
        candidate_rows.append(
            {
                **base,
                "params_path": str(params_path),
                "params_sha256": actual_params_sha,
                "c1_mspd_mean": float(np.mean(c1_mspd)),
                "c1_mspd_median": float(np.median(c1_mspd)),
            }
        )

        context_idx = 0
        for source in source_rows:
            seed_idx = int(source["rollout_seed_idx"])
            traj_dir = _resolve(Path(source["apf_dir"]).parent)
            source_params = traj_dir / "params.npy"
            if _sha256_file(source_params) != actual_params_sha:
                raise RuntimeError(
                    f"C1 source parameters differ for {candidate_id} seed={seed_idx}"
                )
            for anchor_idx, source_step in enumerate(SOURCE_STEPS):
                context_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "context_idx": context_idx,
                        "arm": "trajectory",
                        "rollout_seed_idx": seed_idx,
                        "anchor_idx": anchor_idx,
                        "source_step": source_step,
                        "shared_run_seed": "",
                        "source_traj_dir": str(traj_dir),
                        "source_chunk_path": str(
                            _find_source_chunk(traj_dir, source_step)
                        ),
                    }
                )
                context_idx += 1
        for shared_idx, run_seed in enumerate(SHARED_RUN_SEEDS):
            context_rows.append(
                {
                    "candidate_id": candidate_id,
                    "context_idx": context_idx,
                    "arm": "shared_state",
                    "rollout_seed_idx": shared_idx,
                    "anchor_idx": shared_idx,
                    "source_step": 0,
                    "shared_run_seed": run_seed,
                    "source_traj_dir": "",
                    "source_chunk_path": "",
                }
            )
            context_idx += 1
        if context_idx != N_CONTEXTS:
            raise RuntimeError(f"Built {context_idx} contexts for {candidate_id}")

    config_path = _resolve(
        next(
            row["source_config_path"]
            for row in c5_rows
            if row["candidate_id"] == PILOT_CANDIDATE
        )
    )
    _cfg, flat = load_config(config_path)
    flat_args = dict(OmegaConf.to_container(flat, resolve=True))
    if str(flat_args.get("substrate")) != "lenia_flow":
        raise RuntimeError("Protocol requires substrate=lenia_flow")
    if float(flat_args.get("sigma", math.nan)) != 0.2:
        raise RuntimeError(f"Expected corrected FlowLenia sigma=0.2, got {flat_args.get('sigma')}")
    if float(flat_args.get("flow_sigma", math.nan)) != 0.2:
        raise RuntimeError(
            f"Expected corrected flow_sigma=0.2, got {flat_args.get('flow_sigma')}"
        )
    if str(flat_args.get("mix_rule")) != "stoch":
        raise RuntimeError("RNG-sensitivity protocol requires mix_rule=stoch")

    code = _code_fingerprint()
    branch_rows = [
        {
            "branch_idx": branch_idx,
            "branch_seed": (
                BRANCH_SEED_BASE + branch_idx
                if branch_idx < N_UNIQUE_BRANCHES
                else BRANCH_SEED_BASE + DUPLICATE_OF_BRANCH
            ),
            "duplicate_of": (
                DUPLICATE_OF_BRANCH if branch_idx == DUPLICATE_BRANCH else ""
            ),
            "included_in_pairwise_metric": branch_idx < N_UNIQUE_BRANCHES,
        }
        for branch_idx in range(N_BRANCHES)
    ]
    plan_identity = {
        "candidates": candidate_rows,
        "contexts": context_rows,
        "branches": branch_rows,
    }
    plan_sha = _identity_sha256(plan_identity)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "plan_sha256": plan_sha,
        "code_bundle_sha256": code["bundle_sha256"],
        "code_files": code["files"],
        "input_files": {
            "c1_scores": str(C1_SCORES),
            "c1_scores_sha256": _sha256_file(C1_SCORES),
            "c5_plan": str(C5_PLAN),
            "c5_plan_sha256": _sha256_file(C5_PLAN),
            "simulation_config": str(config_path),
            "simulation_config_sha256": _sha256_file(config_path),
            "simulation_identity": _simulation_config_identity(flat_args),
        },
        "hypotheses": {
            "rng_sensitivity": (
                "Different continuation RNG streams diverge from an exactly "
                "identical FlowLenia state."
            ),
            "optimization_effect": (
                "Candidate-level RNG sensitivity is greater for optimized "
                "than independently sampled random parameter candidates."
            ),
            "terminology": (
                "This operationalizes stochastic/RNG-induced trajectory "
                "sensitivity, not deterministic Lyapunov chaos."
            ),
        },
        "design": {
            "n_candidates": 40,
            "n_optimized": 10,
            "n_random": 30,
            "trajectory_contexts_per_candidate": 20,
            "shared_state_contexts_per_candidate": 4,
            "source_steps": list(SOURCE_STEPS),
            "rollout_seed_indices": list(ROLLOUT_SEED_INDICES),
            "shared_run_seeds": list(SHARED_RUN_SEEDS),
            "horizon_steps": HORIZON_STEPS,
            "step_chunk": STEP_CHUNK,
            "metric_steps": list(METRIC_STEPS),
            "visual_steps": list(VISUAL_STEPS),
            "horizon_summary_steps": list(HORIZON_SUMMARY_STEPS),
            "n_unique_rng_branches": N_UNIQUE_BRANCHES,
            "n_total_branches": N_BRANCHES,
            "duplicate_branch": DUPLICATE_BRANCH,
            "duplicate_of_branch": DUPLICATE_OF_BRANCH,
            "external_state_perturbation": 0.0,
            "simulation_batch_size": SIMULATION_BATCH_SIZE,
            "contexts_per_batch": CONTEXTS_PER_BATCH,
        },
        "rng_protocol": {
            "description": (
                "Candidate-independent common random numbers: context key is "
                "folded with arm/seed/anchor identity, then with branch_seed."
            ),
            "branch_seed_base": BRANCH_SEED_BASE,
            "duplicate_control": "branch 9 receives exactly branch 0 RNG key",
        },
        "metrics": {
            "primary_arm": PRIMARY_ARM,
            "primary_metric": PRIMARY_METRIC,
            "primary_endpoint": (
                "For each context and time, median pairwise mass-normalized A "
                "L1 across 9 unique RNG branches; normalized linear-time AUC "
                "over 0..10000; mean over the 20 trajectory contexts."
            ),
            "secondary": [
                "mass-weighted P L1",
                "rendered RGB L1",
                "normalized flow-field L1",
                "relative total-mass difference",
                "shared-state primary metric",
                "final distance and time-to-threshold",
            ],
        },
        "statistics": {
            "unit": "parameter candidate",
            "primary_test": "one-sided exact Mann-Whitney optimized > random",
            "random_candidates": "30 independent parameter draws; no opt pairing",
            "effect_size": "rank-biserial correlation",
            "confidence_interval": "candidate bootstrap",
            "multiplicity": "trajectory A-relative-L1 AUC is the sole primary endpoint",
        },
        "pilot_candidate": PILOT_CANDIDATE,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    candidate_fields = list(candidate_rows[0].keys())
    context_fields = list(context_rows[0].keys())
    branch_fields = list(branch_rows[0].keys())
    targets = {
        output_root / "candidates.csv": (candidate_rows, candidate_fields),
        output_root / "contexts.csv": (context_rows, context_fields),
        output_root / "branches.csv": (branch_rows, branch_fields),
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                f"Existing protocol differs under {output_root}; use a new output root"
            )
        for path, (rows, fields) in targets.items():
            if not path.exists():
                raise RuntimeError(f"Protocol exists but manifest is missing: {path}")
            if _read_csv(path) != [
                {field: str(row.get(field, "")) for field in fields} for row in rows
            ]:
                raise RuntimeError(f"Existing manifest differs: {path}")
    else:
        for path, (rows, fields) in targets.items():
            _write_csv(path, rows, fields)
        _write_json(protocol_path, protocol)
    _write_json(
        output_root / "plan_audit.json",
        {
            "status": "passed",
            "plan_sha256": plan_sha,
            "code_bundle_sha256": code["bundle_sha256"],
            "n_candidates": len(candidate_rows),
            "n_contexts": len(context_rows),
            "n_branches": len(branch_rows),
            "sigma": float(flat_args["sigma"]),
            "flow_sigma": float(flat_args["flow_sigma"]),
            "mix_rule": flat_args["mix_rule"],
        },
    )
    return protocol


def load_protocol(output_root: Path) -> dict[str, Any]:
    protocol_path = output_root / "protocol.json"
    if not protocol_path.exists():
        return build_plan(output_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Unexpected protocol version in {protocol_path}")
    code = _code_fingerprint()
    if protocol.get("code_bundle_sha256") != code["bundle_sha256"]:
        raise RuntimeError(
            "Simulation code changed after protocol creation; use a new output root"
        )
    return protocol


def _scalar(snapshot: dict[str, np.ndarray], key: str, default: Any) -> Any:
    if key not in snapshot:
        return default
    value = np.asarray(snapshot[key])
    if value.size == 0:
        return default
    return value.reshape(-1)[0].item()


def _load_snapshot(path: Path, step: int) -> dict[str, np.ndarray]:
    required = (
        "steps",
        "A",
        "P",
        "F",
        "state_t",
        "state_mass_cycle_start",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(f"{path} is missing {missing}")
        steps = np.asarray(data["steps"], dtype=np.int64)
        hit = np.flatnonzero(steps == int(step))
        if hit.size != 1:
            raise RuntimeError(f"Step {step} is not unique in {path}")
        idx = int(hit[0])
        result: dict[str, np.ndarray] = {}
        for key in required[1:]:
            result[key] = np.asarray(data[key][idx])
        for key in (
            "resume_batch_rng_key",
            "resume_batch_size",
            "resume_batch_index",
            "resume_seed",
        ):
            if key in data.files:
                result[key] = np.asarray(data[key][idx])
    return result


def _physical_state_hash(snapshot: dict[str, np.ndarray]) -> str:
    return _sha256_arrays(
        np.asarray(snapshot["A"]),
        np.asarray(snapshot["P"]),
        np.asarray(snapshot["F"]),
        np.asarray(snapshot["state_t"]),
        np.asarray(snapshot["state_mass_cycle_start"]),
    )


def _tree_stack(values: list[dict[str, Any]]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *values)


def _replace_physical_state(
    latent_template: dict[str, Any], snapshot: dict[str, np.ndarray]
) -> dict[str, Any]:
    import jax.numpy as jnp

    state = dict(latent_template)
    state["A"] = jnp.asarray(snapshot["A"], dtype=jnp.float32)
    state["P"] = jnp.asarray(snapshot["P"], dtype=jnp.float32)
    state["F"] = jnp.asarray(snapshot["F"], dtype=jnp.float32)
    state["t"] = jnp.asarray(_scalar(snapshot, "state_t", 0), dtype=jnp.int32)
    state["mass_cycle_start"] = jnp.asarray(
        _scalar(snapshot, "state_mass_cycle_start", np.sum(snapshot["A"])),
        dtype=jnp.float32,
    )
    return state


def _shared_state(
    substrate: Any,
    latent_template: dict[str, Any],
    params: Any,
    run_seed: int,
    args: Any,
) -> tuple[dict[str, Any], str]:
    import jax
    import jax.numpy as jnp

    eval_key = jax.random.PRNGKey(int(run_seed))
    rng_roll = _metric_roll_key(args, eval_key)
    k_state = jax.random.split(rng_roll, 4)[0]
    seeded = substrate.seed_state(k_state, params)
    state = dict(latent_template)
    for key in ("A", "P", "F", "Food", "t"):
        if key in seeded:
            state[key] = seeded[key]
    state["mass_cycle_start"] = jnp.sum(state["A"])
    host = jax.device_get(state)
    state_hash = _sha256_arrays(
        np.asarray(host["A"]),
        np.asarray(host["P"]),
        np.asarray(host["F"]),
        np.asarray(host["t"]),
        np.asarray(host["mass_cycle_start"]),
    )
    return state, state_hash


def _context_rng_key(row: dict[str, str], branch_idx: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    arm_code = 0x5452414A if row["arm"] == "trajectory" else 0x53485244
    key = jax.random.PRNGKey(BRANCH_SEED_BASE)
    key = jax.random.fold_in(key, jnp.uint32(arm_code))
    key = jax.random.fold_in(key, int(row["rollout_seed_idx"]))
    key = jax.random.fold_in(key, int(row["anchor_idx"]))
    effective_branch = (
        DUPLICATE_OF_BRANCH if branch_idx == DUPLICATE_BRANCH else branch_idx
    )
    key = jax.random.fold_in(key, BRANCH_SEED_BASE + effective_branch)
    return np.asarray(key, dtype=np.uint32)


def _make_stepper(substrate: Any, batch_size: int):
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


def _make_metric_capture(n_contexts: int):
    import jax
    import jax.numpy as jnp

    left = jnp.asarray(PAIR_LEFT)
    right = jnp.asarray(PAIR_RIGHT)

    def capture(states, rng):
        shape_prefix = (n_contexts, N_BRANCHES)
        A = states["A"].reshape((*shape_prefix, *states["A"].shape[1:]))
        P = states["P"].reshape((*shape_prefix, *states["P"].shape[1:]))
        F = states["F"].reshape((*shape_prefix, *states["F"].shape[1:]))

        A_l, A_r = A[:, left], A[:, right]
        P_l, P_r = P[:, left], P[:, right]
        F_l, F_r = F[:, left], F[:, right]
        eps = jnp.asarray(1.0e-12, dtype=jnp.float32)

        a_num = jnp.sum(jnp.abs(A_l - A_r), axis=(-3, -2, -1))
        a_den = 0.5 * (
            jnp.sum(jnp.abs(A_l), axis=(-3, -2, -1))
            + jnp.sum(jnp.abs(A_r), axis=(-3, -2, -1))
        )
        a_relative_l1 = a_num / jnp.maximum(a_den, eps)

        mass_l = jnp.sum(A_l, axis=-1)
        mass_r = jnp.sum(A_r, axis=-1)
        weight = 0.5 * (mass_l + mass_r)
        p_num = jnp.sum(
            weight[..., None] * jnp.abs(P_l - P_r), axis=(-3, -2, -1)
        )
        p_den = jnp.sum(weight, axis=(-2, -1)) * P.shape[-1]
        p_mass_weighted_l1 = p_num / jnp.maximum(p_den, eps)

        rgb_l = jnp.clip(mass_l[..., None] * P_l[..., :3], 0.0, 1.0)
        rgb_r = jnp.clip(mass_r[..., None] * P_r[..., :3], 0.0, 1.0)
        render_l1 = jnp.mean(jnp.abs(rgb_l - rgb_r), axis=(-3, -2, -1))

        f_num = jnp.sum(jnp.abs(F_l - F_r), axis=(-4, -3, -2, -1))
        f_den = 0.5 * (
            jnp.sum(jnp.abs(F_l), axis=(-4, -3, -2, -1))
            + jnp.sum(jnp.abs(F_r), axis=(-4, -3, -2, -1))
        )
        flow_relative_l1 = f_num / jnp.maximum(f_den, eps)

        total_l = jnp.sum(A_l, axis=(-3, -2, -1))
        total_r = jnp.sum(A_r, axis=(-3, -2, -1))
        mass_relative = jnp.abs(total_l - total_r) / jnp.maximum(
            0.5 * (jnp.abs(total_l) + jnp.abs(total_r)), eps
        )

        duplicate_a = jnp.max(
            jnp.abs(A[:, DUPLICATE_OF_BRANCH] - A[:, DUPLICATE_BRANCH]),
            axis=(-3, -2, -1),
        )
        duplicate_p = jnp.max(
            jnp.abs(P[:, DUPLICATE_OF_BRANCH] - P[:, DUPLICATE_BRANCH]),
            axis=(-3, -2, -1),
        )
        duplicate_f = jnp.max(
            jnp.abs(F[:, DUPLICATE_OF_BRANCH] - F[:, DUPLICATE_BRANCH]),
            axis=(-4, -3, -2, -1),
        )
        rng_view = rng.reshape((n_contexts, N_BRANCHES, 2))
        duplicate_rng = jnp.max(
            jnp.abs(
                rng_view[:, DUPLICATE_OF_BRANCH].astype(jnp.int64)
                - rng_view[:, DUPLICATE_BRANCH].astype(jnp.int64)
            ),
            axis=-1,
        )
        return {
            "a_relative_l1": a_relative_l1,
            "p_mass_weighted_l1": p_mass_weighted_l1,
            "render_l1": render_l1,
            "flow_relative_l1": flow_relative_l1,
            "mass_relative": mass_relative,
            "duplicate_a_max_abs": duplicate_a,
            "duplicate_p_max_abs": duplicate_p,
            "duplicate_f_max_abs": duplicate_f,
            "duplicate_rng_max_abs": duplicate_rng,
        }

    return jax.jit(capture)


def _make_visual_capture(n_contexts: int, local_context_idx: int):
    import jax
    import jax.numpy as jnp

    def capture(states):
        A = states["A"].reshape(
            (n_contexts, N_BRANCHES, *states["A"].shape[1:])
        )[local_context_idx]
        P = states["P"].reshape(
            (n_contexts, N_BRANCHES, *states["P"].shape[1:])
        )[local_context_idx]
        mass = jnp.sum(A, axis=-1, keepdims=True)
        rgb = jnp.clip(mass * P[..., :3], 0.0, 1.0)
        return jnp.rint(rgb * 255.0).astype(jnp.uint8)

    return jax.jit(capture)


@dataclass
class BatchResult:
    context_indices: np.ndarray
    source_state_hashes: np.ndarray
    metrics: dict[str, np.ndarray]
    visual_context_idx: int
    visual_rgb: np.ndarray
    elapsed_seconds: float


def _simulate_context_batch(
    *,
    substrate: Any,
    args: Any,
    params: Any,
    context_rows: list[dict[str, str]],
    visual_context_idx: int,
    stepper: Any,
    metric_capture: Any,
) -> BatchResult:
    import jax
    import jax.numpy as jnp

    if len(context_rows) != CONTEXTS_PER_BATCH:
        raise RuntimeError(f"Expected {CONTEXTS_PER_BATCH} contexts")
    latent_template = dict(substrate.init_state(jax.random.PRNGKey(0), params))
    states: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    snapshot_cache: dict[tuple[Path, int], dict[str, np.ndarray]] = {}
    for row in context_rows:
        if row["arm"] == "trajectory":
            key = (_resolve(row["source_chunk_path"]), int(row["source_step"]))
            snapshot = snapshot_cache.get(key)
            if snapshot is None:
                snapshot = _load_snapshot(*key)
                snapshot_cache[key] = snapshot
            state = _replace_physical_state(latent_template, snapshot)
            state_hash = _physical_state_hash(snapshot)
        else:
            state, state_hash = _shared_state(
                substrate,
                latent_template,
                params,
                int(row["shared_run_seed"]),
                args,
            )
        states.append(state)
        source_hashes.append(state_hash)

    contexts_state = _tree_stack(states)
    batched_state = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value[:, None], N_BRANCHES, axis=1).reshape(
            (SIMULATION_BATCH_SIZE, *value.shape[1:])
        ),
        contexts_state,
    )
    rng_np = np.stack(
        [
            _context_rng_key(row, branch_idx)
            for row in context_rows
            for branch_idx in range(N_BRANCHES)
        ],
        axis=0,
    )
    rng = jnp.asarray(rng_np, dtype=jnp.uint32)
    params_batch = jnp.repeat(params[None], SIMULATION_BATCH_SIZE, axis=0)
    visual_local_idx = next(
        i
        for i, row in enumerate(context_rows)
        if int(row["context_idx"]) == int(visual_context_idx)
    )
    visual_capture = _make_visual_capture(CONTEXTS_PER_BATCH, visual_local_idx)

    metric_buffers: dict[str, list[np.ndarray]] = {}
    visual_frames: list[np.ndarray] = []
    metric_step_set = set(METRIC_STEPS)
    visual_step_set = set(VISUAL_STEPS)
    started = time.monotonic()
    for step in range(0, HORIZON_STEPS + 1, STEP_CHUNK):
        if step in metric_step_set:
            captured = jax.device_get(metric_capture(batched_state, rng))
            for key, value in captured.items():
                metric_buffers.setdefault(key, []).append(np.asarray(value))
        if step in visual_step_set:
            visual_frames.append(
                np.asarray(jax.device_get(visual_capture(batched_state)), dtype=np.uint8)
            )
        if step < HORIZON_STEPS:
            batched_state, rng = stepper(batched_state, rng, params_batch)

    elapsed = time.monotonic() - started
    metrics = {
        key: np.stack(values, axis=1) for key, values in metric_buffers.items()
    }
    initial_pair_max = max(
        float(np.max(metrics[key][:, 0]))
        for key in (
            "a_relative_l1",
            "p_mass_weighted_l1",
            "render_l1",
            "flow_relative_l1",
            "mass_relative",
        )
    )
    duplicate_max = max(
        float(np.max(metrics[key]))
        for key in (
            "duplicate_a_max_abs",
            "duplicate_p_max_abs",
            "duplicate_f_max_abs",
            "duplicate_rng_max_abs",
        )
    )
    if initial_pair_max != 0.0:
        raise RuntimeError(f"Branch states differ at t=0: {initial_pair_max}")
    if duplicate_max != 0.0:
        raise RuntimeError(f"Exact duplicate control diverged: {duplicate_max}")
    if not all(np.all(np.isfinite(value)) for value in metrics.values()):
        raise RuntimeError("Non-finite divergence metric detected")
    visual_rgb = np.stack(visual_frames, axis=1)
    return BatchResult(
        context_indices=np.asarray(
            [int(row["context_idx"]) for row in context_rows], dtype=np.int32
        ),
        source_state_hashes=np.asarray(source_hashes),
        metrics=metrics,
        visual_context_idx=int(visual_context_idx),
        visual_rgb=visual_rgb,
        elapsed_seconds=float(elapsed),
    )


def _batch_output_path(output_root: Path, candidate_id: str, batch_idx: int) -> Path:
    return output_root / "simulation" / candidate_id / f"batch_{batch_idx:02d}.npz"


def _save_batch_result(
    path: Path,
    *,
    result: BatchResult,
    candidate: dict[str, str],
    protocol: dict[str, Any],
    batch_idx: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol_version": np.asarray(PROTOCOL_VERSION),
        "plan_sha256": np.asarray(protocol["plan_sha256"]),
        "code_bundle_sha256": np.asarray(protocol["code_bundle_sha256"]),
        "candidate_id": np.asarray(candidate["candidate_id"]),
        "candidate_kind": np.asarray(candidate["candidate_kind"]),
        "params_sha256": np.asarray(candidate["params_sha256"]),
        "batch_idx": np.asarray(batch_idx, dtype=np.int32),
        "context_indices": result.context_indices,
        "source_state_hashes": result.source_state_hashes,
        "metric_steps": np.asarray(METRIC_STEPS, dtype=np.int32),
        "visual_steps": np.asarray(VISUAL_STEPS, dtype=np.int32),
        "pair_left": PAIR_LEFT,
        "pair_right": PAIR_RIGHT,
        "visual_context_idx": np.asarray(result.visual_context_idx, dtype=np.int32),
        "visual_rgb": result.visual_rgb,
        "elapsed_seconds": np.asarray(result.elapsed_seconds, dtype=np.float64),
    }
    payload.update(result.metrics)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _validate_batch_output(
    path: Path,
    *,
    candidate: dict[str, str],
    protocol: dict[str, Any],
    expected_context_indices: list[int],
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            checks = (
                str(np.asarray(data["protocol_version"]).item()) == PROTOCOL_VERSION,
                str(np.asarray(data["plan_sha256"]).item()) == protocol["plan_sha256"],
                str(np.asarray(data["code_bundle_sha256"]).item())
                == protocol["code_bundle_sha256"],
                str(np.asarray(data["candidate_id"]).item())
                == candidate["candidate_id"],
                str(np.asarray(data["params_sha256"]).item())
                == candidate["params_sha256"],
                np.array_equal(
                    np.asarray(data["context_indices"], dtype=np.int32),
                    np.asarray(expected_context_indices, dtype=np.int32),
                ),
                np.array_equal(
                    np.asarray(data["metric_steps"], dtype=np.int32),
                    np.asarray(METRIC_STEPS, dtype=np.int32),
                ),
                np.array_equal(np.asarray(data["pair_left"]), PAIR_LEFT),
                np.array_equal(np.asarray(data["pair_right"]), PAIR_RIGHT),
            )
            shapes_ok = all(
                np.asarray(data[key]).shape
                == (CONTEXTS_PER_BATCH, len(METRIC_STEPS), N_PAIRS)
                for key in (
                    "a_relative_l1",
                    "p_mass_weighted_l1",
                    "render_l1",
                    "flow_relative_l1",
                    "mass_relative",
                )
            )
            duplicate_zero = all(
                float(np.max(np.asarray(data[key]))) == 0.0
                for key in (
                    "duplicate_a_max_abs",
                    "duplicate_p_max_abs",
                    "duplicate_f_max_abs",
                    "duplicate_rng_max_abs",
                )
            )
            initial_zero = all(
                float(np.max(np.asarray(data[key])[:, 0])) == 0.0
                for key in (
                    "a_relative_l1",
                    "p_mass_weighted_l1",
                    "render_l1",
                    "flow_relative_l1",
                    "mass_relative",
                )
            )
            return all(checks) and shapes_ok and duplicate_zero and initial_zero
    except Exception:
        return False


def simulate(
    output_root: Path,
    *,
    candidate_ids: set[str] | None,
    pilot: bool,
) -> None:
    protocol = load_protocol(output_root)
    candidates = _read_csv(output_root / "candidates.csv")
    contexts = _read_csv(output_root / "contexts.csv")
    if pilot:
        candidate_ids = {PILOT_CANDIDATE}
    if candidate_ids is not None:
        unknown = candidate_ids.difference(row["candidate_id"] for row in candidates)
        if unknown:
            raise RuntimeError(f"Unknown candidates: {sorted(unknown)}")
        candidates = [row for row in candidates if row["candidate_id"] in candidate_ids]
    if not candidates:
        raise RuntimeError("No candidates selected")

    config_path = _resolve(protocol["input_files"]["simulation_config"])
    _cfg, flat = load_config(config_path)
    flat_args = dict(OmegaConf.to_container(flat, resolve=True))
    args = SimpleNamespace(**flat_args)

    import jax
    import jax.numpy as jnp

    substrate = _make_substrate(args)
    first_params = np.asarray(np.load(_resolve(candidates[0]["params_path"])), dtype=np.float32)
    _ = substrate.init_state(jax.random.PRNGKey(0), jnp.asarray(first_params))
    stepper = _make_stepper(substrate, SIMULATION_BATCH_SIZE)
    metric_capture = _make_metric_capture(CONTEXTS_PER_BATCH)

    completed_before = 0
    durations: list[float] = []
    run_started = time.monotonic()
    for candidate_pos, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["candidate_id"]
        candidate_contexts = sorted(
            [row for row in contexts if row["candidate_id"] == candidate_id],
            key=lambda row: int(row["context_idx"]),
        )
        if len(candidate_contexts) != N_CONTEXTS:
            raise RuntimeError(f"{candidate_id} has {len(candidate_contexts)} contexts")
        params = np.asarray(np.load(_resolve(candidate["params_path"])), dtype=np.float32)
        if params.ndim != 1 or params.size != int(substrate.n_params):
            raise RuntimeError(f"Unexpected parameter shape for {candidate_id}: {params.shape}")
        params_j = jnp.asarray(params)

        for batch_idx in range(N_CONTEXTS // CONTEXTS_PER_BATCH):
            batch_contexts = candidate_contexts[
                batch_idx * CONTEXTS_PER_BATCH : (batch_idx + 1) * CONTEXTS_PER_BATCH
            ]
            expected_indices = [int(row["context_idx"]) for row in batch_contexts]
            output_path = _batch_output_path(output_root, candidate_id, batch_idx)
            if _validate_batch_output(
                output_path,
                candidate=candidate,
                protocol=protocol,
                expected_context_indices=expected_indices,
            ):
                completed_before += 1
                print(f"[skip] {candidate_id} batch={batch_idx} already valid", flush=True)
                continue
            if output_path.exists():
                quarantine = output_path.with_suffix(
                    output_path.suffix + f".invalid-{int(time.time())}"
                )
                shutil.move(output_path, quarantine)
                print(f"[quarantine] {output_path} -> {quarantine}", flush=True)

            visual_context_idx = 2 if batch_idx == 0 else 20
            print(
                f"[simulate] {candidate_id} ({candidate_pos}/{len(candidates)}) "
                f"batch={batch_idx} contexts={expected_indices[0]}..{expected_indices[-1]}",
                flush=True,
            )
            result = _simulate_context_batch(
                substrate=substrate,
                args=args,
                params=params_j,
                context_rows=batch_contexts,
                visual_context_idx=visual_context_idx,
                stepper=stepper,
                metric_capture=metric_capture,
            )
            _save_batch_result(
                output_path,
                result=result,
                candidate=candidate,
                protocol=protocol,
                batch_idx=batch_idx,
            )
            if not _validate_batch_output(
                output_path,
                candidate=candidate,
                protocol=protocol,
                expected_context_indices=expected_indices,
            ):
                raise RuntimeError(f"Post-write validation failed: {output_path}")
            durations.append(result.elapsed_seconds)
            total_batches = len(candidates) * (N_CONTEXTS // CONTEXTS_PER_BATCH)
            finished_now = completed_before + len(durations)
            mean_seconds = float(np.mean(durations)) if durations else math.nan
            eta_seconds = max(0, total_batches - finished_now) * mean_seconds
            _write_json(
                output_root / "simulation_progress.json",
                {
                    "status": "running",
                    "last_candidate": candidate_id,
                    "last_batch": batch_idx,
                    "selected_candidates": len(candidates),
                    "completed_batches_in_selection": finished_now,
                    "total_batches_in_selection": total_batches,
                    "mean_new_batch_seconds": mean_seconds,
                    "eta_seconds": eta_seconds,
                    "wall_seconds": time.monotonic() - run_started,
                },
            )

    _write_json(
        output_root / ("pilot_audit.json" if pilot else "simulation_completion.json"),
        {
            "status": "passed" if pilot else "complete",
            "pilot": bool(pilot),
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "valid_batches": len(candidates) * (N_CONTEXTS // CONTEXTS_PER_BATCH),
            "reused_batches": completed_before,
            "new_batches": len(durations),
            "max_t0_pair_distance": 0.0,
            "max_duplicate_distance": 0.0,
            "metric_checkpoint_count": len(METRIC_STEPS),
            "horizon_steps": HORIZON_STEPS,
            "wall_seconds": time.monotonic() - run_started,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowLenia continuation-RNG sensitivity experiment."
    )
    parser.add_argument(
        "stage",
        choices=("plan", "pilot", "simulate"),
        help="Build immutable protocol, run one optimized pilot, or simulate selected/all candidates.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Restrict simulate stage to one or more candidate IDs.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_root = _resolve(cli.output_root)
    if cli.stage == "plan":
        protocol = build_plan(output_root)
        print(
            f"Plan ready: {output_root} plan_sha256={protocol['plan_sha256']}",
            flush=True,
        )
        return
    if cli.stage == "pilot":
        build_plan(output_root)
        simulate(output_root, candidate_ids=None, pilot=True)
        return
    simulate(
        output_root,
        candidate_ids=set(cli.candidate_id) if cli.candidate_id else None,
        pilot=False,
    )


if __name__ == "__main__":
    main()
