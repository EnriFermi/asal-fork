from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

from paper_suite_c2_branching import _embedding_chamfer_cosine
from paper_suite_common import ensure_dir, write_csv, write_json


DEFAULT_CONFIG = (
    "experiments/paper_suite/"
    "config_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper.yaml"
)
DEFAULT_MAIN_ROOT = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper"
)
DEFAULT_SWEEP_ROOT = f"{DEFAULT_MAIN_ROOT}/c2_noise_horizon_sweep"
DEFAULT_HORIZONS = (5000, 10000, 15000, 20000, 30000)
DEFAULT_BASE_A_STD = 0.02
DEFAULT_BASE_P_STD = 0.02
DEFAULT_BASE_LAG_XY_STD = 1.0
DEFAULT_SNAPSHOT_INTERVAL = 50
DEFAULT_FRAMES_PER_HORIZON = 8
_RUN_RE = re.compile(r"run_(\d{3})(?:_|$)")


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def _parse_numbers(raw: str, cast: Any) -> list[Any]:
    values = [cast(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError(f"Empty numeric list: {raw!r}")
    return values


def _strength_tag(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^0-9A-Za-z]+", "_", text)


def _run_idx(traj_id: Any) -> int:
    match = _RUN_RE.search(str(traj_id))
    if match is None:
        raise ValueError(f"Could not parse run index from {traj_id!r}.")
    return int(match.group(1))


def _chunk_sort_key(path: Path) -> tuple[int, int, int, str]:
    match = re.search(r"P_steps_(\d+)_(\d+).*idx_(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), path.name
    return 10**18, 10**18, 10**18, path.name


def _sample_offsets(
    horizon: int,
    *,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    n_frames: int = DEFAULT_FRAMES_PER_HORIZON,
) -> list[int]:
    horizon = int(horizon)
    snapshot_interval = int(snapshot_interval)
    if horizon <= 0 or horizon % snapshot_interval != 0:
        raise ValueError(
            f"Horizon {horizon} must be a positive multiple of snapshot interval "
            f"{snapshot_interval}."
        )
    capture_count = horizon // snapshot_interval + 1
    indices = np.linspace(0, capture_count - 1, int(n_frames)).astype(np.int64)
    offsets = (indices * snapshot_interval).astype(np.int64).tolist()
    if offsets[0] != 0 or offsets[-1] != horizon or len(offsets) != int(n_frames):
        raise RuntimeError(f"Invalid frame schedule for horizon {horizon}: {offsets}")
    return [int(value) for value in offsets]


def _union_offsets(
    horizons: Iterable[int],
    *,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    n_frames: int = DEFAULT_FRAMES_PER_HORIZON,
) -> list[int]:
    return sorted(
        {
            offset
            for horizon in horizons
            for offset in _sample_offsets(
                int(horizon),
                snapshot_interval=snapshot_interval,
                n_frames=n_frames,
            )
        }
    )


def _load_main_plan(main_root: Path) -> pd.DataFrame:
    plan_path = main_root / "c2_branching" / "branch_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    plan = pd.read_csv(plan_path)
    required = {
        "traj_id",
        "pair_id",
        "condition",
        "step",
        "branch_id",
        "branch_seed",
        "source_traj_dir",
        "branch_dir",
        "delta_h",
    }
    if missing := required.difference(plan.columns):
        raise ValueError(f"{plan_path} is missing columns: {sorted(missing)}")
    if len(plan) != 450:
        raise ValueError(f"Expected 450 main C2 branch rows, found {len(plan)}.")
    key = ["traj_id", "pair_id", "condition", "branch_id"]
    if plan.duplicated(key).any():
        raise ValueError("Main C2 branch plan contains duplicate branch identities.")
    plan = plan.copy()
    plan["run_idx"] = plan["traj_id"].map(_run_idx)
    if sorted(plan["run_idx"].unique().tolist()) != list(range(10)):
        raise ValueError("Main C2 branch plan does not contain opt_000 through opt_009.")
    counts = (
        plan.groupby(["run_idx", "condition"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for condition in ("low", "mid", "high"):
        if not bool((counts[condition] == 15).all()):
            raise ValueError(f"Each run must have 15 {condition} branch rows.")
    return plan


def _load_branch_arrays(branch_dir: Path) -> dict[str, np.ndarray]:
    chunks = sorted((branch_dir / "apf_logs").glob("P_steps_*.npz"), key=_chunk_sort_key)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks in {branch_dir / 'apf_logs'}")
    parts: dict[str, list[np.ndarray]] = {}
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim < 1:
                    continue
                parts.setdefault(key, []).append(arr)
    out = {
        key: np.concatenate(values, axis=0)
        for key, values in parts.items()
        if values
    }
    if "steps" not in out:
        raise ValueError(f"Missing steps in {branch_dir}.")
    steps = np.asarray(out["steps"], dtype=np.int64)
    order = np.argsort(steps, kind="stable")
    for key, arr in list(out.items()):
        if arr.ndim > 0 and arr.shape[0] == steps.size:
            out[key] = arr[order]
    if np.unique(np.asarray(out["steps"], dtype=np.int64)).size != steps.size:
        raise ValueError(f"Duplicate APF steps in {branch_dir}.")
    return out


def _expected_absolute_steps(start_step: int, offsets: Iterable[int]) -> np.ndarray:
    return np.asarray(
        [int(start_step) + int(offset) for offset in offsets],
        dtype=np.int64,
    )


def _branch_output_status(
    branch_dir: Path,
    *,
    row: dict[str, Any],
    capture_offsets: list[int],
) -> tuple[bool, str]:
    metadata_path = branch_dir / "resume_metadata.json"
    if not metadata_path.exists():
        return False, "missing_metadata"
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False, "invalid_metadata"
    expected_meta = {
        "start_step": int(row["step"]),
        "end_step": int(row["step"]) + int(max(capture_offsets)),
        "branch_seed": int(row["branch_seed"]),
        "perturb_a_std": float(row["a_std"]),
        "perturb_p_std": float(row["p_std"]),
        "perturb_lagrangian_xy_std": float(row["lagrangian_xy_std"]),
        "capture_relative_steps": [int(value) for value in capture_offsets],
    }
    output_fields_raw = str(row.get("output_fields", "")).strip()
    output_fields = (
        [value for value in output_fields_raw.split(",") if value]
        if output_fields_raw
        else None
    )
    output_compress_raw = row.get("output_compress", None)
    if output_fields is not None:
        expected_meta["output_fields"] = ["steps", *output_fields]
    if output_compress_raw is not None and not pd.isna(output_compress_raw):
        expected_meta["output_compress"] = bool(output_compress_raw)
    for key, expected in expected_meta.items():
        actual = metadata.get(key)
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
                return False, f"metadata_{key}"
        elif actual != expected:
            return False, f"metadata_{key}"
    try:
        arrays = _load_branch_arrays(branch_dir)
    except Exception as exc:
        return False, f"invalid_apf:{type(exc).__name__}"
    expected_steps = _expected_absolute_steps(int(row["step"]), capture_offsets)
    actual_steps = np.asarray(arrays["steps"], dtype=np.int64)
    if not np.array_equal(actual_steps, expected_steps):
        return False, "step_grid"
    required_arrays = output_fields or ["A", "P", "F", "lagrangian_xy"]
    for key in required_arrays:
        if key not in arrays:
            return False, f"missing_{key}"
        if arrays[key].shape[0] != expected_steps.size:
            return False, f"length_{key}"
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float32)).all():
            return False, f"nonfinite_{key}"
    return True, "valid"


def _select_calibration_states(plan: pd.DataFrame) -> pd.DataFrame:
    specs = ((2, "high", 0), (5, "mid", 0), (7, "low", 0))
    selected: list[pd.DataFrame] = []
    for run_idx, condition, pair_id in specs:
        group = plan[
            (plan["run_idx"] == int(run_idx))
            & (plan["condition"].astype(str) == condition)
            & (plan["pair_id"].astype(int) == int(pair_id))
        ]
        if len(group) != 3:
            raise ValueError(
                f"Calibration state opt_{run_idx:03d}/{condition}/{pair_id} "
                f"has {len(group)} branches, expected 3."
            )
        selected.append(group)
    return pd.concat(selected, ignore_index=True)


def _state_id(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"opt_{int(row['run_idx']):03d}_"
        f"{str(row['condition'])}_p{int(row['pair_id']):02d}_"
        f"step_{int(row['step'])}"
    )


def _sweep_branch_dir(
    output_root: Path,
    *,
    mode: str,
    strength: float,
    row: pd.Series | dict[str, Any],
) -> Path:
    return (
        output_root
        / mode
        / "branches"
        / f"noise_{_strength_tag(float(strength))}"
        / _state_id(row)
        / f"branch_{int(row['branch_id']):02d}"
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    main_root = _resolve(args.main_root)
    output_root = ensure_dir(_resolve(args.output_root))
    main_plan = _load_main_plan(main_root)
    strengths = _parse_numbers(args.strengths, float)
    mode = str(args.mode)
    if mode == "calibration":
        selected = _select_calibration_states(main_plan)
        horizons = [int(args.calibration_horizon)]
        capture_offsets = list(
            range(
                0,
                int(args.calibration_horizon) + int(args.calibration_capture_every),
                int(args.calibration_capture_every),
            )
        )
        if capture_offsets[-1] != int(args.calibration_horizon):
            raise ValueError("Calibration horizon must be divisible by capture cadence.")
    elif mode == "full":
        horizons = _parse_numbers(args.horizons, int)
        if sorted(set(horizons)) != sorted(horizons):
            raise ValueError(f"Horizons must be unique: {horizons}")
        selected = main_plan
        if args.state_limit is not None:
            state_keys = (
                selected[["traj_id", "pair_id", "condition"]]
                .drop_duplicates()
                .sort_values(["traj_id", "condition", "pair_id"])
                .head(int(args.state_limit))
            )
            selected = selected.merge(
                state_keys,
                on=["traj_id", "pair_id", "condition"],
                how="inner",
                validate="many_to_one",
            )
        capture_offsets = _union_offsets(
            horizons,
            snapshot_interval=int(args.snapshot_interval),
            n_frames=int(args.frames_per_horizon),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    max_horizon = max(horizons)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    invalid_existing: list[str] = []
    for strength in strengths:
        for _, source in selected.iterrows():
            branch_dir = _sweep_branch_dir(
                output_root,
                mode=mode,
                strength=float(strength),
                row=source,
            )
            row = {
                "mode": mode,
                "strength": float(strength),
                "strength_tag": _strength_tag(float(strength)),
                "a_std": DEFAULT_BASE_A_STD * float(strength),
                "p_std": DEFAULT_BASE_P_STD * float(strength),
                "lagrangian_xy_std": DEFAULT_BASE_LAG_XY_STD * float(strength),
                "run_idx": int(source["run_idx"]),
                "traj_id": str(source["traj_id"]),
                "pair_id": int(source["pair_id"]),
                "condition": str(source["condition"]),
                "step": int(source["step"]),
                "delta_h": float(source["delta_h"]),
                "branch_id": int(source["branch_id"]),
                "branch_seed": int(source["branch_seed"]),
                "source_traj_dir": str(source["source_traj_dir"]),
                "reference_branch_dir": str(source["branch_dir"]),
                "branch_dir": str(branch_dir),
                "max_horizon": int(max_horizon),
                "capture_relative_steps": ",".join(
                    str(value) for value in capture_offsets
                ),
                "output_fields": "A,P" if mode == "full" else "",
                "output_compress": False if mode == "full" else None,
            }
            valid, reason = _branch_output_status(
                branch_dir,
                row=row,
                capture_offsets=capture_offsets,
            )
            row["status"] = "valid_cached" if valid else "missing_or_invalid"
            row["status_reason"] = reason
            rows.append(row)
            if valid:
                continue
            if branch_dir.exists() and any(branch_dir.iterdir()):
                invalid_existing.append(str(branch_dir))
                continue
            job = {
                    "source_traj_dir": str(source["source_traj_dir"]),
                    "step": int(source["step"]),
                    "additional_steps": int(max_horizon),
                    "output_dir": str(branch_dir),
                    "branch_seed": int(source["branch_seed"]),
                    "perturb_a_std": DEFAULT_BASE_A_STD * float(strength),
                    "perturb_p_std": DEFAULT_BASE_P_STD * float(strength),
                    "perturb_lagrangian_xy_std": (
                        DEFAULT_BASE_LAG_XY_STD * float(strength)
                    ),
                    "snapshot_interval": int(args.snapshot_interval),
                    "jit_microbatch": int(args.jit_microbatch),
                    "capture_relative_steps": capture_offsets,
                    "output_max_snapshots_per_chunk": len(capture_offsets),
                    "ignore_output_paths_in_simulation_signature": True,
                }
            if mode == "full":
                job["output_fields"] = ["steps", "A", "P"]
                job["output_compress"] = False
            jobs.append(job)
    if invalid_existing:
        preview = "\n".join(invalid_existing[:10])
        raise RuntimeError(
            "Refusing to overwrite invalid non-empty sweep outputs. Inspect or "
            f"remove these dedicated sweep paths first:\n{preview}"
        )

    mode_root = ensure_dir(output_root / mode)
    plan_path = mode_root / "sweep_plan.csv"
    jobs_path = mode_root / "pending_jobs.json"
    write_csv(plan_path, rows)
    write_json(
        jobs_path,
        {
            "jobs": jobs,
            "mode": mode,
            "strengths": strengths,
            "horizons": horizons,
            "capture_relative_steps": capture_offsets,
        },
    )
    summary = {
        "status": "ready",
        "mode": mode,
        "strengths": strengths,
        "horizons": horizons,
        "capture_relative_steps": capture_offsets,
        "n_capture_steps": len(capture_offsets),
        "n_plan_rows": len(rows),
        "n_cached_valid": int(sum(row["status"] == "valid_cached" for row in rows)),
        "n_pending_jobs": len(jobs),
        "plan": str(plan_path),
        "jobs": str(jobs_path),
    }
    write_json(mode_root / "prepare_summary.json", summary)
    return summary


def run_jobs(args: argparse.Namespace) -> dict[str, Any]:
    jobs_path = _resolve(args.jobs_json)
    payload = json.loads(jobs_path.read_text())
    jobs = payload.get("jobs", payload if isinstance(payload, list) else [])
    if not jobs:
        return {"status": "nothing_to_run", "jobs": str(jobs_path)}
    command = [
        str(_resolve(args.python)),
        str(_REPO_ROOT / "scripts" / "flowlenia_minibang_resume_batch.py"),
        "--jobs-json",
        str(jobs_path),
        "--batch-size",
        str(int(args.batch_size)),
    ]
    subprocess.run(command, cwd=_REPO_ROOT, check=True)
    return {
        "status": "completed",
        "n_jobs": len(jobs),
        "jobs": str(jobs_path),
        "command": " ".join(command),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    mode = str(args.mode)
    plan_path = output_root / mode / "sweep_plan.csv"
    plan = pd.read_csv(plan_path)
    prepare_summary = json.loads(
        (output_root / mode / "prepare_summary.json").read_text()
    )
    capture_offsets = [
        int(value) for value in prepare_summary["capture_relative_steps"]
    ]
    rows: list[dict[str, Any]] = []
    for _, source in plan.iterrows():
        valid, reason = _branch_output_status(
            Path(str(source["branch_dir"])),
            row=source.to_dict(),
            capture_offsets=capture_offsets,
        )
        rows.append(
            {
                "strength": float(source["strength"]),
                "run_idx": int(source["run_idx"]),
                "traj_id": str(source["traj_id"]),
                "pair_id": int(source["pair_id"]),
                "condition": str(source["condition"]),
                "branch_id": int(source["branch_id"]),
                "branch_dir": str(source["branch_dir"]),
                "valid": bool(valid),
                "reason": reason,
            }
        )
    audit_path = output_root / mode / "output_audit.csv"
    write_csv(audit_path, rows)
    invalid = [row for row in rows if not row["valid"]]
    summary = {
        "status": "exact" if not invalid else "incomplete",
        "mode": mode,
        "n_expected": len(rows),
        "n_valid": len(rows) - len(invalid),
        "n_invalid": len(invalid),
        "capture_relative_steps": capture_offsets,
        "audit": str(audit_path),
    }
    write_json(output_root / mode / "output_audit.json", summary)
    if invalid and args.require_complete:
        raise RuntimeError(
            f"{mode} sweep has {len(invalid)}/{len(rows)} invalid outputs."
        )
    return summary


def _render_rgb(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    pp = np.asarray(p, dtype=np.float32)
    if pp.shape[-1] < 3:
        repeats = int(math.ceil(3.0 / max(1, int(pp.shape[-1]))))
        pp = np.tile(pp, repeats)[..., :3]
    else:
        pp = pp[..., :3]
    mass = np.sum(aa, axis=-1, keepdims=True)
    return np.clip(mass * pp, 0.0, 1.0)


def calibration_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    mode_root = output_root / "calibration"
    plan = pd.read_csv(mode_root / "sweep_plan.csv")
    audit_payload = audit(
        argparse.Namespace(
            output_root=str(output_root),
            mode="calibration",
            require_complete=True,
        )
    )
    rows: list[dict[str, Any]] = []
    final_frames: dict[tuple[float, str, int], np.ndarray] = {}
    for _, source in plan.iterrows():
        arrays = _load_branch_arrays(Path(str(source["branch_dir"])))
        a = np.asarray(arrays["A"], dtype=np.float32)
        p = np.asarray(arrays["P"], dtype=np.float32)
        state_id = _state_id(source)
        final_frames[
            (float(source["strength"]), state_id, int(source["branch_id"]))
        ] = _render_rgb(a[-1], p[-1])
        rows.append(
            {
                "strength": float(source["strength"]),
                "a_std": float(source["a_std"]),
                "p_std": float(source["p_std"]),
                "lagrangian_xy_std": float(source["lagrangian_xy_std"]),
                "run_idx": int(source["run_idx"]),
                "condition": str(source["condition"]),
                "pair_id": int(source["pair_id"]),
                "branch_id": int(source["branch_id"]),
                "start_step": int(source["step"]),
                "final_relative_step": int(arrays["steps"][-1])
                - int(source["step"]),
                "A_mass_start": float(np.sum(a[0], dtype=np.float64)),
                "A_mass_final": float(np.sum(a[-1], dtype=np.float64)),
                "A_mass_ratio": float(
                    np.sum(a[-1], dtype=np.float64)
                    / max(np.sum(a[0], dtype=np.float64), 1e-12)
                ),
                "A_rms_start": float(np.sqrt(np.mean(a[0] * a[0]))),
                "A_rms_final": float(np.sqrt(np.mean(a[-1] * a[-1]))),
                "P_rms_start": float(np.sqrt(np.mean(p[0] * p[0]))),
                "P_rms_final": float(np.sqrt(np.mean(p[-1] * p[-1]))),
                "branch_dir": str(source["branch_dir"]),
            }
        )
    diagnostics = pd.DataFrame(rows)

    pair_rows: list[dict[str, Any]] = []
    state_cols = ["strength", "run_idx", "traj_id", "condition", "pair_id"]
    for key, group in plan.groupby(state_cols, sort=True):
        branch_arrays = [
            _load_branch_arrays(Path(str(row.branch_dir)))
            for row in group.sort_values("branch_id").itertuples(index=False)
        ]
        pair_values = []
        for left, right in combinations(branch_arrays, 2):
            left_rgb = _render_rgb(left["A"][-1], left["P"][-1])
            right_rgb = _render_rgb(right["A"][-1], right["P"][-1])
            pair_values.append(
                float(np.sqrt(np.mean((left_rgb - right_rgb) ** 2)))
            )
        pair_rows.append(
            {
                "strength": float(key[0]),
                "run_idx": int(key[1]),
                "traj_id": str(key[2]),
                "condition": str(key[3]),
                "pair_id": int(key[4]),
                "final_rgb_pair_rms_median": float(np.median(pair_values)),
                "final_rgb_pair_rms_mean": float(np.mean(pair_values)),
            }
        )
    pair_diagnostics = pd.DataFrame(pair_rows)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    strengths = sorted(plan["strength"].unique().tolist())
    states = (
        plan[["run_idx", "traj_id", "condition", "pair_id", "step"]]
        .drop_duplicates()
        .sort_values(["run_idx", "condition", "pair_id"])
    )
    state_ids = [_state_id(row) for _, row in states.iterrows()]
    n_cols = len(state_ids) * 3
    fig, axes = plt.subplots(
        len(strengths),
        n_cols,
        figsize=(2.05 * n_cols, 1.95 * len(strengths)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, strength in enumerate(strengths):
        for state_idx, state_id in enumerate(state_ids):
            for branch_id in range(3):
                ax = axes[row_idx, state_idx * 3 + branch_id]
                ax.imshow(final_frames[(float(strength), state_id, branch_id)])
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(
                        f"{state_id.replace('_step_', '@')}\nb{branch_id}",
                        fontsize=8,
                    )
                if state_idx == 0 and branch_id == 0:
                    ax.set_ylabel(f"scale={strength:g}", fontsize=9)
    fig.suptitle(
        f"C2 noise calibration: final frames at +{int(args.calibration_horizon)} steps",
        fontsize=13,
    )
    montage_path = mode_root / "calibration_final_frames.png"
    fig.savefig(montage_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    diagnostics_path = mode_root / "calibration_branch_diagnostics.csv"
    pair_path = mode_root / "calibration_pair_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    pair_diagnostics.to_csv(pair_path, index=False)
    summary = {
        "status": "complete",
        "audit": audit_payload,
        "strengths": strengths,
        "calibration_horizon": int(args.calibration_horizon),
        "montage": str(montage_path),
        "branch_diagnostics": str(diagnostics_path),
        "pair_diagnostics": str(pair_path),
        "median_final_rgb_pair_rms_by_strength": {
            str(float(strength)): float(
                pair_diagnostics.loc[
                    pair_diagnostics["strength"] == strength,
                    "final_rgb_pair_rms_median",
                ].median()
            )
            for strength in strengths
        },
    }
    write_json(mode_root / "calibration_report.json", summary)
    return summary


def parity(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    mode_root = output_root / "full"
    plan = pd.read_csv(mode_root / "sweep_plan.csv")
    selected = plan[np.isclose(plan["strength"].astype(float), 1.0)]
    if args.max_branches is not None:
        selected = selected.head(int(args.max_branches))
    target_offsets = _sample_offsets(
        20000,
        snapshot_interval=int(args.snapshot_interval),
        n_frames=int(args.frames_per_horizon),
    )
    rows: list[dict[str, Any]] = []
    # A and P are the complete inputs to the published CLIP rendering. The
    # sweep intentionally omits F and tracer payloads to avoid redundant I/O.
    keys = ("A", "P")
    for _, source in selected.iterrows():
        new = _load_branch_arrays(Path(str(source["branch_dir"])))
        old = _load_branch_arrays(Path(str(source["reference_branch_dir"])))
        expected_steps = _expected_absolute_steps(
            int(source["step"]),
            target_offsets,
        )
        new_idx = np.asarray(
            [
                int(np.flatnonzero(new["steps"] == step)[0])
                for step in expected_steps
            ],
            dtype=np.int64,
        )
        old_idx = np.asarray(
            [
                int(np.flatnonzero(old["steps"] == step)[0])
                for step in expected_steps
            ],
            dtype=np.int64,
        )
        for key in keys:
            if key not in new or key not in old:
                rows.append(
                    {
                        "branch_dir": str(source["branch_dir"]),
                        "reference_branch_dir": str(source["reference_branch_dir"]),
                        "key": key,
                        "array_equal": False,
                        "max_abs_diff": float("nan"),
                        "reason": "missing_key",
                    }
                )
                continue
            left = np.asarray(new[key])[new_idx]
            right = np.asarray(old[key])[old_idx]
            same = bool(np.array_equal(left, right))
            if left.shape == right.shape and np.issubdtype(left.dtype, np.number):
                max_abs_diff = float(
                    np.max(
                        np.abs(
                            left.astype(np.float64)
                            - right.astype(np.float64)
                        )
                    )
                )
            else:
                max_abs_diff = 0.0 if same else float("nan")
            rows.append(
                {
                    "branch_dir": str(source["branch_dir"]),
                    "reference_branch_dir": str(source["reference_branch_dir"]),
                    "key": key,
                    "array_equal": same,
                    "max_abs_diff": max_abs_diff,
                    "reason": "exact" if same else "different",
                }
            )
    output_path = mode_root / "noise1_horizon20k_parity.csv"
    write_csv(output_path, rows)
    mismatches = [row for row in rows if not row["array_equal"]]
    summary = {
        "status": "exact" if not mismatches else "mismatch",
        "n_branches": int(selected.shape[0]),
        "n_arrays": len(rows),
        "n_exact": len(rows) - len(mismatches),
        "n_mismatches": len(mismatches),
        "target_offsets": target_offsets,
        "details": str(output_path),
    }
    write_json(mode_root / "noise1_horizon20k_parity.json", summary)
    if mismatches and args.require_exact:
        raise RuntimeError(
            f"Noise=1 prefix parity failed for {len(mismatches)} arrays."
        )
    return summary


def _branch_source_signature(branch_dir: Path) -> str:
    records = []
    for path in sorted(
        (branch_dir / "apf_logs").glob("P_steps_*.npz"),
        key=_chunk_sort_key,
    ):
        stat = path.stat()
        records.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if not records:
        raise FileNotFoundError(f"No APF chunks in {branch_dir / 'apf_logs'}")
    return _sha256_text(
        json.dumps(records, sort_keys=True, separators=(",", ":"))
    )


def _embedding_cache_path(cache_root: Path, branch_dir: Path) -> Path:
    digest = _sha256_text(str(branch_dir.resolve()))[:24]
    return cache_root / f"{digest}.npz"


def _make_clip_batch_embedder(fm: Any, batch_size: int) -> Any:
    import jax
    import jax.numpy as jnp

    expected_batch = int(batch_size)

    @jax.jit
    def embed(frames: Any) -> Any:
        images = jnp.asarray(frames, dtype=jnp.float32)
        if images.shape[0] != expected_batch:
            raise ValueError(
                f"Expected CLIP batch {expected_batch}, got {images.shape[0]}."
            )
        if images.shape[1] != 224 or images.shape[2] != 224:
            images = jax.image.resize(
                images,
                (images.shape[0], 224, 224, images.shape[3]),
                method="bilinear",
            )
        images = (
            images
            - jnp.asarray(fm.img_mean, dtype=jnp.float32)[None, None, None, :]
        ) / jnp.asarray(fm.img_std, dtype=jnp.float32)[None, None, None, :]
        images = jnp.transpose(images, (0, 3, 1, 2))
        z = fm.clip_model.get_image_features(images)
        return z / jnp.clip(
            jnp.linalg.norm(z, axis=-1, keepdims=True),
            1e-12,
        )

    return embed


def _old_main_embeddings(
    *,
    reference_branch_dir: Path,
    main_root: Path,
) -> np.ndarray | None:
    from paper_suite_c2_branching import _clip_embedding_cache_path

    cache_root = main_root / "c2_branching" / "clip_embedding_cache"
    path = _clip_embedding_cache_path(
        reference_branch_dir,
        cache_dir=cache_root,
        foundation_model="clip",
        max_chunks=4,
        max_snapshots_per_chunk=8,
        max_frames=32,
    )
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
    return z if z.shape[0] == DEFAULT_FRAMES_PER_HORIZON else None


def _load_or_compute_sweep_embeddings(
    *,
    source: pd.Series,
    cache_root: Path,
    main_root: Path,
    embed_batch: Any,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    import jax

    branch_dir = Path(str(source["branch_dir"]))
    source_signature = _branch_source_signature(branch_dir)
    cache_path = _embedding_cache_path(cache_root, branch_dir)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_signature = str(np.asarray(data["source_signature"]).item())
            steps = np.asarray(data["steps"], dtype=np.int64)
            z = np.asarray(data["z"], dtype=np.float32)
            reused = int(
                np.asarray(data.get("reused_main_embeddings", 0)).reshape(-1)[0]
            )
        if cached_signature == source_signature and z.shape[0] == steps.size:
            return steps, z, True, reused

    arrays = _load_branch_arrays(branch_dir)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    frames = _render_rgb(arrays["A"], arrays["P"]).astype(np.float32)
    if frames.shape[0] != steps.size:
        raise ValueError(f"Frame/step count mismatch in {branch_dir}.")
    z = np.asarray(
        jax.device_get(embed_batch(frames)),
        dtype=np.float32,
    ).copy()
    reused_main = 0
    if np.isclose(float(source["strength"]), 1.0):
        old_z = _old_main_embeddings(
            reference_branch_dir=Path(str(source["reference_branch_dir"])),
            main_root=main_root,
        )
        if old_z is not None:
            old_offsets = _sample_offsets(20000)
            old_steps = _expected_absolute_steps(int(source["step"]), old_offsets)
            for step, embedding in zip(old_steps, old_z, strict=True):
                hit = np.flatnonzero(steps == int(step))
                if hit.size != 1:
                    raise ValueError(
                        f"Could not map old CLIP embedding step {step} in "
                        f"{branch_dir}."
                    )
                z[int(hit[0])] = np.asarray(embedding, dtype=np.float32)
                reused_main += 1
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        steps=steps,
        z=z,
        branch_dir=np.asarray(str(branch_dir)),
        source_signature=np.asarray(source_signature),
        reused_main_embeddings=np.asarray(reused_main, dtype=np.int32),
        cache_version=np.asarray(
            "c2_noise_horizon_clip_union_v1_main20k_reuse"
        ),
    )
    return steps, z, False, reused_main


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    result: dict[str, float | int] = {
        "n": int(x.size),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }
    if x.size < 3 or float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return result


def metrics(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    main_root = _resolve(args.main_root)
    mode_root = output_root / "full"
    audit_payload = audit(
        argparse.Namespace(
            output_root=str(output_root),
            mode="full",
            require_complete=True,
        )
    )
    plan = pd.read_csv(mode_root / "sweep_plan.csv")
    prepare_summary = json.loads((mode_root / "prepare_summary.json").read_text())
    horizons = [int(value) for value in prepare_summary["horizons"]]
    expected_capture = np.asarray(
        prepare_summary["capture_relative_steps"],
        dtype=np.int64,
    )
    if len(plan) != 3150:
        raise ValueError(f"Expected 3150 full sweep branches, found {len(plan)}.")

    import foundation_models

    fm = foundation_models.create_foundation_model("clip")
    embed_batch = _make_clip_batch_embedder(fm, int(expected_capture.size))
    cache_root = ensure_dir(mode_root / "clip_union_cache")
    embedding_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cache_hits = 0
    cache_misses = 0
    reused_main_embeddings = 0
    for index, (_, source) in enumerate(plan.iterrows(), start=1):
        steps, z, was_cached, reused = _load_or_compute_sweep_embeddings(
            source=source,
            cache_root=cache_root,
            main_root=main_root,
            embed_batch=embed_batch,
            force=bool(args.force_embeddings),
        )
        expected_steps = _expected_absolute_steps(
            int(source["step"]),
            expected_capture,
        )
        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"Unexpected embedding step grid in {source['branch_dir']}."
            )
        embedding_index[str(source["branch_dir"])] = (steps, z)
        cache_hits += int(was_cached)
        cache_misses += int(not was_cached)
        reused_main_embeddings += int(reused)
        if index == 1 or index % 50 == 0 or index == len(plan):
            print(
                f"[c2-sweep-clip] {index}/{len(plan)} "
                f"cache_hits={cache_hits} cache_misses={cache_misses}",
                flush=True,
            )

    score_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    group_cols = ["strength", "run_idx", "traj_id", "condition", "pair_id"]
    groups = list(plan.groupby(group_cols, sort=True))
    for group_index, (key, group) in enumerate(groups, start=1):
        if len(group) != 3:
            raise ValueError(f"Sweep state {key} has {len(group)} branches.")
        group = group.sort_values("branch_id")
        branch_records = list(group.itertuples(index=False))
        for horizon in horizons:
            offsets = _sample_offsets(
                horizon,
                snapshot_interval=int(args.snapshot_interval),
                n_frames=int(args.frames_per_horizon),
            )
            embeddings: list[np.ndarray] = []
            for row in branch_records:
                steps, z = embedding_index[str(row.branch_dir)]
                target_steps = _expected_absolute_steps(int(row.step), offsets)
                indices = [
                    int(np.flatnonzero(steps == int(step))[0])
                    for step in target_steps
                ]
                embeddings.append(z[np.asarray(indices, dtype=np.int64)])
            pair_values: list[float] = []
            for left_idx, right_idx in combinations(range(3), 2):
                value = float(
                    _embedding_chamfer_cosine(
                        embeddings[left_idx],
                        embeddings[right_idx],
                    )
                )
                pair_values.append(value)
                pair_rows.append(
                    {
                        "strength": float(key[0]),
                        "horizon_steps": int(horizon),
                        "run_idx": int(key[1]),
                        "traj_id": str(key[2]),
                        "condition": str(key[3]),
                        "pair_id": int(key[4]),
                        "branch_id_i": int(branch_records[left_idx].branch_id),
                        "branch_id_j": int(branch_records[right_idx].branch_id),
                        "branch_dir_i": str(branch_records[left_idx].branch_dir),
                        "branch_dir_j": str(branch_records[right_idx].branch_dir),
                        "pairwise_branching_score": value,
                    }
                )
            score_rows.append(
                {
                    "strength": float(key[0]),
                    "a_std": DEFAULT_BASE_A_STD * float(key[0]),
                    "p_std": DEFAULT_BASE_P_STD * float(key[0]),
                    "lagrangian_xy_std": DEFAULT_BASE_LAG_XY_STD
                    * float(key[0]),
                    "horizon_steps": int(horizon),
                    "run_idx": int(key[1]),
                    "traj_id": str(key[2]),
                    "condition": str(key[3]),
                    "pair_id": int(key[4]),
                    "step": int(branch_records[0].step),
                    "delta_h": float(branch_records[0].delta_h),
                    "branching_score": float(np.median(pair_values)),
                    "branching_score_pair_std": float(
                        np.std(pair_values, ddof=1)
                    ),
                    "n_branches": 3,
                    "n_branch_pairs": 3,
                    "n_frames": len(offsets),
                    "frame_offsets": ",".join(str(value) for value in offsets),
                    "branching_metric": "future_clip_chamfer_cosine",
                }
            )
        if (
            group_index == 1
            or group_index % 50 == 0
            or group_index == len(groups)
        ):
            print(
                f"[c2-sweep-scores] {group_index}/{len(groups)} states",
                flush=True,
            )

    scores = pd.DataFrame(score_rows)
    pair_details = pd.DataFrame(pair_rows)
    if len(scores) != 5250:
        raise ValueError(f"Expected 5250 sweep scores, found {len(scores)}.")

    correlation_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for (strength, horizon), grid in scores.groupby(
        ["strength", "horizon_steps"],
        sort=True,
    ):
        pooled = _correlation(
            grid["delta_h"].to_numpy(dtype=np.float64),
            grid["branching_score"].to_numpy(dtype=np.float64),
        )
        per_run: list[dict[str, Any]] = []
        for run_idx, run_group in grid.groupby("run_idx", sort=True):
            stats = _correlation(
                run_group["delta_h"].to_numpy(dtype=np.float64),
                run_group["branching_score"].to_numpy(dtype=np.float64),
            )
            row = {
                "strength": float(strength),
                "horizon_steps": int(horizon),
                "run_idx": int(run_idx),
                **stats,
            }
            per_run.append(row)
            within_rows.append(row)

        contrasts: list[float] = []
        for (run_idx, pair_id), matched in grid.groupby(
            ["run_idx", "pair_id"],
            sort=True,
        ):
            by_condition = {
                str(row.condition): float(row.branching_score)
                for row in matched.itertuples(index=False)
            }
            if "high" not in by_condition or "low" not in by_condition:
                raise ValueError(
                    f"Missing matched high/low score for {strength}/{horizon}/"
                    f"run={run_idx}/pair={pair_id}."
                )
            value = by_condition["high"] - by_condition["low"]
            contrasts.append(value)
            contrast_rows.append(
                {
                    "strength": float(strength),
                    "horizon_steps": int(horizon),
                    "run_idx": int(run_idx),
                    "pair_id": int(pair_id),
                    "high_branching_score": by_condition["high"],
                    "low_branching_score": by_condition["low"],
                    "delta_branching_score": value,
                }
            )
        from scipy.stats import binomtest

        nonzero = [value for value in contrasts if value != 0.0]
        n_positive = int(sum(value > 0.0 for value in nonzero))
        sign_p = (
            float(
                binomtest(
                    n_positive,
                    len(nonzero),
                    p=0.5,
                    alternative="greater",
                ).pvalue
            )
            if nonzero
            else float("nan")
        )
        within_pearson = np.asarray(
            [float(row["pearson_r"]) for row in per_run],
            dtype=np.float64,
        )
        within_spearman = np.asarray(
            [float(row["spearman_rho"]) for row in per_run],
            dtype=np.float64,
        )
        correlation_rows.append(
            {
                "strength": float(strength),
                "a_std": DEFAULT_BASE_A_STD * float(strength),
                "p_std": DEFAULT_BASE_P_STD * float(strength),
                "lagrangian_xy_std": DEFAULT_BASE_LAG_XY_STD
                * float(strength),
                "horizon_steps": int(horizon),
                **pooled,
                "within_pearson_mean": float(np.nanmean(within_pearson)),
                "within_pearson_median": float(np.nanmedian(within_pearson)),
                "within_pearson_n_positive": int(
                    np.sum(within_pearson > 0.0)
                ),
                "within_spearman_mean": float(np.nanmean(within_spearman)),
                "within_spearman_median": float(
                    np.nanmedian(within_spearman)
                ),
                "within_spearman_n_positive": int(
                    np.sum(within_spearman > 0.0)
                ),
                "contrast_mean": float(np.mean(contrasts)),
                "contrast_median": float(np.median(contrasts)),
                "contrast_n_positive": n_positive,
                "contrast_n_nonzero": len(nonzero),
                "contrast_sign_test_greater_p": sign_p,
            }
        )

    correlations = pd.DataFrame(correlation_rows)
    within = pd.DataFrame(within_rows)
    contrasts = pd.DataFrame(contrast_rows)
    scores_path = mode_root / "scores_clip_chamfer.csv"
    pair_path = mode_root / "pair_details_clip_chamfer.csv"
    correlations_path = mode_root / "correlation_grid.csv"
    within_path = mode_root / "within_run_correlations.csv"
    contrasts_path = mode_root / "matched_high_low_contrasts.csv"
    scores.to_csv(scores_path, index=False)
    pair_details.to_csv(pair_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    within.to_csv(within_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)

    main_scores = pd.read_csv(
        main_root
        / "c2_branching"
        / "branching_scores_clip_chamfer.csv"
    )
    persisted_scores = pd.read_csv(scores_path)
    baseline = persisted_scores[
        np.isclose(persisted_scores["strength"], 1.0)
        & (persisted_scores["horizon_steps"] == 20000)
    ]
    parity_keys = ["traj_id", "pair_id", "condition"]
    comparison = baseline.merge(
        main_scores[parity_keys + ["branching_score"]],
        on=parity_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_sweep", "_main"),
        indicator=True,
    )
    if not bool((comparison["_merge"] == "both").all()):
        raise ValueError("Sweep/main score parity keys differ.")
    score_diff = np.abs(
        comparison["branching_score_sweep"].to_numpy(dtype=np.float64)
        - comparison["branching_score_main"].to_numpy(dtype=np.float64)
    )
    baseline_parity = {
        "n": len(comparison),
        "max_abs_score_diff": float(np.max(score_diff)),
        "exact": bool(np.array_equal(
            comparison["branching_score_sweep"].to_numpy(dtype=np.float64),
            comparison["branching_score_main"].to_numpy(dtype=np.float64),
        )),
    }
    summary = {
        "status": "complete",
        "audit": audit_payload,
        "n_scores": len(scores),
        "n_pair_details": len(pair_details),
        "n_grid_cells": len(correlations),
        "embedding_cache": {
            "directory": str(cache_root),
            "hits": cache_hits,
            "misses": cache_misses,
            "reused_main_embeddings": reused_main_embeddings,
        },
        "baseline_noise1_horizon20k_parity": baseline_parity,
        "outputs": {
            "scores": str(scores_path),
            "pair_details": str(pair_path),
            "correlations": str(correlations_path),
            "within_run": str(within_path),
            "contrasts": str(contrasts_path),
        },
    }
    write_json(mode_root / "metrics_summary.json", summary)
    return summary


def _grid(
    frame: pd.DataFrame,
    *,
    value: str,
    strengths: list[float],
    horizons: list[int],
) -> np.ndarray:
    indexed = frame.set_index(["strength", "horizon_steps"])
    return np.asarray(
        [
            [
                float(indexed.loc[(strength, horizon), value])
                for horizon in horizons
            ]
            for strength in strengths
        ],
        dtype=np.float64,
    )


def plots(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    mode_root = output_root / "full"
    correlations = pd.read_csv(mode_root / "correlation_grid.csv")
    scores = pd.read_csv(mode_root / "scores_clip_chamfer.csv")
    strengths = sorted(correlations["strength"].unique().tolist())
    horizons = sorted(correlations["horizon_steps"].unique().astype(int).tolist())
    if len(strengths) != 7 or len(horizons) != 5:
        raise ValueError(
            f"Expected 7 strengths x 5 horizons, got {strengths} x {horizons}."
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figures = ensure_dir(mode_root / "figures")
    panels = (
        ("pearson_r", "Pooled Pearson r", -1.0, 1.0, "coolwarm"),
        (
            "within_spearman_median",
            "Median within-run Spearman rho",
            -1.0,
            1.0,
            "coolwarm",
        ),
        (
            "contrast_median",
            "Median high-minus-low divergence",
            None,
            None,
            "coolwarm",
        ),
        (
            "contrast_n_positive",
            "Positive matched contrasts (of 50)",
            0.0,
            50.0,
            "viridis",
        ),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.2),
        constrained_layout=True,
    )
    for ax, (value, title, vmin, vmax, cmap) in zip(
        axes.flat,
        panels,
        strict=True,
    ):
        values = _grid(
            correlations,
            value=value,
            strengths=strengths,
            horizons=horizons,
        )
        norm = None
        if value == "contrast_median":
            finite_max = float(np.nanmax(np.abs(values)))
            norm = TwoSlopeNorm(
                vmin=-max(finite_max, 1e-12),
                vcenter=0.0,
                vmax=max(finite_max, 1e-12),
            )
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
        )
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                label = (
                    f"{values[row_idx, col_idx]:.3f}"
                    if value != "contrast_n_positive"
                    else f"{int(values[row_idx, col_idx])}"
                )
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if abs(
                            (
                                values[row_idx, col_idx]
                                - np.nanmean(values)
                            )
                            / max(np.nanstd(values), 1e-12)
                        )
                        > 0.8
                        else "black"
                    ),
                )
        ax.set_xticks(
            np.arange(len(horizons)),
            [f"{value // 1000}k" for value in horizons],
        )
        ax.set_yticks(
            np.arange(len(strengths)),
            [f"{value:g}" for value in strengths],
        )
        ax.set_xlabel("branch horizon")
        ax.set_ylabel("noise scale")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    fig.suptitle(
        "Flow-Lenia C2 sensitivity to branch horizon and perturbation noise",
        fontsize=14,
    )
    heatmap_path = figures / "c2_noise_horizon_heatmaps.png"
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.6),
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(strengths)))
    for strength, color in zip(strengths, colors, strict=True):
        group = correlations[
            np.isclose(correlations["strength"], strength)
        ].sort_values("horizon_steps")
        x = group["horizon_steps"].to_numpy(dtype=np.float64) / 1000.0
        axes[0].plot(
            x,
            group["pearson_r"],
            marker="o",
            color=color,
            label=f"{strength:g}",
        )
        axes[1].plot(
            x,
            group["within_spearman_median"],
            marker="o",
            color=color,
        )
        axes[2].plot(
            x,
            group["contrast_median"],
            marker="o",
            color=color,
        )
    for ax in axes:
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.set_xlabel("branch horizon (k steps)")
        ax.grid(color="#dddddd", linewidth=0.65)
    axes[0].set_ylabel("pooled Pearson r")
    axes[1].set_ylabel("median within-run Spearman rho")
    axes[2].set_ylabel("median high-minus-low divergence")
    axes[0].legend(
        title="noise scale",
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    fig.suptitle("C2 association curves across shared-prefix horizons")
    curves_path = figures / "c2_noise_horizon_correlation_curves.png"
    fig.savefig(curves_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    aggregated = (
        scores.groupby(
            ["strength", "horizon_steps", "condition"],
            as_index=False,
        )["branching_score"]
        .agg(["median", "mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 8.4),
        constrained_layout=True,
    )
    palette = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
    for panel_idx, horizon in enumerate(horizons):
        ax = axes.flat[panel_idx]
        group = aggregated[aggregated["horizon_steps"] == horizon]
        for condition in ("low", "mid", "high"):
            condition_group = group[
                group["condition"] == condition
            ].sort_values("strength")
            ax.plot(
                np.arange(len(strengths)),
                condition_group["median"],
                marker="o",
                color=palette[condition],
                label=condition,
            )
        ax.set_xticks(
            np.arange(len(strengths)),
            [f"{value:g}" for value in strengths],
        )
        ax.set_xlabel("noise scale")
        ax.set_ylabel("median CLIP-Chamfer divergence")
        ax.set_title(f"horizon {horizon // 1000}k")
        ax.grid(color="#dddddd", linewidth=0.65)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Delta-H stratum",
        loc="lower right",
        bbox_to_anchor=(0.94, 0.08),
        frameon=False,
    )
    fig.suptitle("C2 divergence by Delta-H stratum, noise, and horizon")
    strata_path = figures / "c2_noise_horizon_divergence_by_stratum.png"
    fig.savefig(strata_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "complete",
        "figures": {
            "heatmaps": str(heatmap_path),
            "correlation_curves": str(curves_path),
            "divergence_by_stratum": str(strata_path),
        },
    }
    write_json(figures / "plot_summary.json", summary)
    return summary


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _add_common_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--main-root", default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--mode", choices=("calibration", "full"), required=True)
    parser.add_argument("--strengths", required=True)
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
    )
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--jit-microbatch", type=int, default=50)
    parser.add_argument("--frames-per-horizon", type=int, default=8)
    parser.add_argument("--calibration-horizon", type=int, default=4000)
    parser.add_argument("--calibration-capture-every", type=int, default=500)
    parser.add_argument("--state-limit", type=int, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute-efficient Flow-Lenia C2 sensitivity sweep over perturbation "
            "strength and shared-prefix branch horizons."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _add_common_prepare_args(prepare_parser)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--jobs-json", required=True)
    run_parser.add_argument(
        "--python",
        default="/home/coder/.conda/envs/torchjax/bin/python",
    )
    run_parser.add_argument("--batch-size", type=int, default=30)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)
    audit_parser.add_argument("--mode", choices=("calibration", "full"), required=True)
    audit_parser.add_argument("--require-complete", action="store_true")

    calibration_parser = subparsers.add_parser("calibration-report")
    calibration_parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)
    calibration_parser.add_argument("--calibration-horizon", type=int, default=4000)

    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)
    parity_parser.add_argument("--snapshot-interval", type=int, default=50)
    parity_parser.add_argument("--frames-per-horizon", type=int, default=8)
    parity_parser.add_argument("--max-branches", type=int, default=None)
    parity_parser.add_argument("--require-exact", action="store_true")

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--main-root", default=DEFAULT_MAIN_ROOT)
    metrics_parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)
    metrics_parser.add_argument("--snapshot-interval", type=int, default=50)
    metrics_parser.add_argument("--frames-per-horizon", type=int, default=8)
    metrics_parser.add_argument("--force-embeddings", action="store_true")

    plots_parser = subparsers.add_parser("plots")
    plots_parser.add_argument("--output-root", default=DEFAULT_SWEEP_ROOT)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run_jobs(args)
    elif args.command == "audit":
        result = audit(args)
    elif args.command == "calibration-report":
        result = calibration_report(args)
    elif args.command == "parity":
        result = parity(args)
    elif args.command == "metrics":
        result = metrics(args)
    elif args.command == "plots":
        result = plots(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
