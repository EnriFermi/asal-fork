from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_C2_ROOT = REPO_ROOT / (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "c2_noise_horizon_sweep/full"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis/article_revision_20260722/data"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(a: np.ndarray, p: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key, value in ((b"A", a), (b"P", p)):
        contiguous = np.ascontiguousarray(value)
        digest.update(key)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _branch_start(branch_dir: Path, start_step: int) -> dict[str, np.ndarray]:
    chunks = sorted((branch_dir / "apf_logs").glob("P_steps_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {branch_dir / 'apf_logs'}")
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            matches = np.flatnonzero(steps == int(start_step))
            if matches.size == 0:
                continue
            if matches.size != 1:
                raise ValueError(f"Duplicate start step {start_step} in {path}")
            index = int(matches[0])
            return {
                "A": np.asarray(data["A"][index]).copy(),
                "P": np.asarray(data["P"][index]).copy(),
                "step": np.asarray(steps[index]),
                "path": np.asarray(str(path)),
                "index": np.asarray(index),
            }
    raise ValueError(f"Start step {start_step} is absent from {branch_dir}")


def _metadata_rows(plan: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in plan.to_dict(orient="records"):
        branch_dir = Path(str(row["branch_dir"]))
        metadata_path = branch_dir / "resume_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        start_step = int(row["step"])
        if int(metadata["start_step"]) != start_step:
            raise ValueError(f"Start-step mismatch in {metadata_path}")
        if int(metadata["branch_seed"]) != int(row["branch_seed"]):
            raise ValueError(f"Branch-seed mismatch in {metadata_path}")
        for plan_key, metadata_key in (
            ("a_std", "perturb_a_std"),
            ("p_std", "perturb_p_std"),
            ("lagrangian_xy_std", "perturb_lagrangian_xy_std"),
        ):
            if float(row[plan_key]) != 0.0 or float(metadata[metadata_key]) != 0.0:
                raise ValueError(
                    f"RNG-only row has a nonzero perturbation scale in {metadata_path}"
                )
        rows.append(
            {
                **row,
                "branch_dir_path": branch_dir,
                "metadata_path": metadata_path,
                "source_apf_path": Path(str(metadata["source_apf_path"])),
                "source_snapshot_index": int(metadata["source_snapshot_index"]),
            }
        )
    return rows


def audit(c2_root: Path, output_root: Path) -> dict[str, Any]:
    plan_path = c2_root / "sweep_plan.csv"
    plan = pd.read_csv(plan_path)
    rng_only = plan[plan["strength_tag"].astype(str) == "0"].copy()
    if len(rng_only) != 450:
        raise ValueError(f"Expected 450 RNG-only branches, found {len(rng_only)}")
    if sorted(rng_only["run_idx"].astype(int).unique().tolist()) != list(range(10)):
        raise ValueError("RNG-only plan must contain runs 000 through 009")

    rows = _metadata_rows(rng_only)
    source_cache: dict[tuple[Path, int], dict[str, np.ndarray]] = {}
    audit_rows: list[dict[str, Any]] = []
    for sequence, row in enumerate(rows, start=1):
        source_path = row["source_apf_path"]
        source_index = int(row["source_snapshot_index"])
        cache_key = (source_path, source_index)
        if cache_key not in source_cache:
            with np.load(source_path, allow_pickle=False) as source:
                source_cache[cache_key] = {
                    "A": np.asarray(source["A"][source_index]).copy(),
                    "P": np.asarray(source["P"][source_index]).copy(),
                    "step": np.asarray(source["steps"][source_index]),
                }
        source_state = source_cache[cache_key]
        branch_state = _branch_start(
            row["branch_dir_path"],
            int(row["step"]),
        )
        if int(source_state["step"]) != int(row["step"]):
            raise ValueError(
                f"Source snapshot points to step {int(source_state['step'])}, "
                f"expected {int(row['step'])}: {source_path}"
            )

        a_source = source_state["A"]
        p_source = source_state["P"]
        a_branch = branch_state["A"]
        p_branch = branch_state["P"]
        a_shape_equal = a_source.shape == a_branch.shape
        p_shape_equal = p_source.shape == p_branch.shape
        a_exact = bool(a_shape_equal and np.array_equal(a_source, a_branch))
        p_exact = bool(p_shape_equal and np.array_equal(p_source, p_branch))
        a_max_abs = (
            float(np.max(np.abs(a_source.astype(np.float32) - a_branch.astype(np.float32))))
            if a_shape_equal
            else float("nan")
        )
        p_max_abs = (
            float(np.max(np.abs(p_source.astype(np.float32) - p_branch.astype(np.float32))))
            if p_shape_equal
            else float("nan")
        )
        source_hash = _state_hash(a_source, p_source)
        branch_hash = _state_hash(a_branch, p_branch)
        audit_rows.append(
            {
                "sequence": sequence,
                "run_idx": int(row["run_idx"]),
                "traj_id": str(row["traj_id"]),
                "condition": str(row["condition"]),
                "pair_id": int(row["pair_id"]),
                "start_step": int(row["step"]),
                "branch_id": int(row["branch_id"]),
                "branch_seed": int(row["branch_seed"]),
                "a_std": float(row["a_std"]),
                "p_std": float(row["p_std"]),
                "lagrangian_xy_std": float(row["lagrangian_xy_std"]),
                "source_apf_path": str(source_path),
                "source_snapshot_index": source_index,
                "branch_apf_path": str(branch_state["path"]),
                "branch_snapshot_index": int(branch_state["index"]),
                "a_source_dtype": str(a_source.dtype),
                "a_branch_dtype": str(a_branch.dtype),
                "p_source_dtype": str(p_source.dtype),
                "p_branch_dtype": str(p_branch.dtype),
                "a_exact": a_exact,
                "p_exact": p_exact,
                "state_exact": bool(a_exact and p_exact and source_hash == branch_hash),
                "a_max_abs_diff": a_max_abs,
                "p_max_abs_diff": p_max_abs,
                "source_state_sha256": source_hash,
                "branch_state_sha256": branch_hash,
            }
        )

    audit_table = pd.DataFrame(audit_rows)
    state_keys = ["run_idx", "traj_id", "condition", "pair_id", "start_step"]
    branch_counts = audit_table.groupby(state_keys, dropna=False).size()
    if len(branch_counts) != 150 or not bool((branch_counts == 3).all()):
        raise ValueError("Expected 150 source states with exactly three branches each")

    all_exact = bool(audit_table["state_exact"].all())
    summary = {
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": _file_hash(Path(__file__).resolve()),
        "input_plan": str(plan_path.resolve()),
        "strength_tag": "0",
        "external_state_noise": 0.0,
        "n_runs": int(audit_table["run_idx"].nunique()),
        "n_source_states": int(len(branch_counts)),
        "n_branches": int(len(audit_table)),
        "branches_per_source_state": 3,
        "a_exact_count": int(audit_table["a_exact"].sum()),
        "p_exact_count": int(audit_table["p_exact"].sum()),
        "state_exact_count": int(audit_table["state_exact"].sum()),
        "all_start_states_bit_exact": all_exact,
        "max_a_abs_diff": float(audit_table["a_max_abs_diff"].max()),
        "max_p_abs_diff": float(audit_table["p_max_abs_diff"].max()),
        "audit_scope": (
            "Stored A and P at branch offset zero versus the referenced source "
            "APF snapshot; metadata and plan perturbation scales also checked."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    audit_table.to_csv(output_root / "c2_branch_point_equality_audit.csv", index=False)
    (output_root / "c2_branch_point_equality_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not all_exact:
        raise RuntimeError("At least one C2 branch start differs from its source state")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit all RNG-only Flow-Lenia C2 branch starting states."
    )
    parser.add_argument("--c2-root", type=Path, default=DEFAULT_C2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    summary = audit(args.c2_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
