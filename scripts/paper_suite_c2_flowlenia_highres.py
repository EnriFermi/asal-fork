from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flowlenia_minibang_common import list_apf_chunks
from paper_suite_common import (
    as_list,
    command_to_str,
    current_python,
    ensure_dir,
    load_config,
    resolve_path,
    run_subprocess,
    write_json,
)


RUN_RE = re.compile(r"run_(\d+)$")
REQUIRED_APF_KEYS = (
    "A",
    "P",
    "F",
    "lagrangian_xy",
    "lagrangian_c",
    "resume_batch_rng_key",
    "state_t",
    "state_mass_cycle_start",
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _section(cfg: Any) -> Any:
    return _get(cfg.get("c2", {}), "flow_lenia_highres", {})


def _run_idx(path: Path) -> int | None:
    match = RUN_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _root_label(root: Path) -> str:
    parent = root.parent.name if root.name == "optimization" else root.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", parent).strip("_") or "source"


def _discover_checkpoints(section: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    explicit_dirs = as_list(_get(section, "optimized_checkpoint_dirs", []))
    for raw in explicit_dirs:
        path = resolve_path(raw)
        if path is None:
            continue
        idx = _run_idx(path)
        rows.append(
            {
                "checkpoint_dir": path,
                "source_root": path.parent,
                "source_root_rank": len(rows),
                "run_idx": -1 if idx is None else int(idx),
            }
        )

    roots = as_list(_get(section, "optimized_checkpoint_roots", []))
    for root_rank, raw in enumerate(roots):
        root = resolve_path(raw)
        if root is None or not root.exists():
            continue
        for path in sorted(root.glob("run_*")):
            if not path.is_dir() or not (path / "best.pkl").exists():
                continue
            idx = _run_idx(path)
            if idx is None:
                continue
            rows.append(
                {
                    "checkpoint_dir": path,
                    "source_root": root,
                    "source_root_rank": int(root_rank),
                    "run_idx": int(idx),
                }
            )

    rows.sort(key=lambda r: (int(r["source_root_rank"]), int(r["run_idx"]), str(r["checkpoint_dir"])))
    if bool(_get(section, "dedupe_by_run_idx", True)):
        deduped: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in rows:
            idx = int(row["run_idx"])
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(row)
        rows = deduped

    max_checkpoints = _get(section, "max_checkpoints", None)
    if max_checkpoints is not None:
        rows = rows[: max(0, int(max_checkpoints))]
    return rows


def _traj_id(row: dict[str, Any]) -> str:
    idx = int(row["run_idx"])
    if idx >= 0:
        return f"flow_opt_run_{idx:03d}"
    return f"{_root_label(Path(row['source_root']))}_{Path(row['checkpoint_dir']).name}"


def _run_output_paths(output_root: Path, traj_id: str) -> dict[str, Path]:
    run_root = output_root / traj_id
    traj_dir = run_root / "traj_00000"
    return {
        "run_root": run_root,
        "traj_dir": traj_dir,
        "apf_dir": traj_dir / "apf_logs",
        "metrics_path": traj_dir / "metrics.npz",
        "config_path": traj_dir / "config.yaml",
        "params_path": traj_dir / "params.npy",
    }


def _list_chunks_safe(apf_dir: Path) -> list[tuple[Path, int, int, int]]:
    if not apf_dir.exists():
        return []
    return list_apf_chunks(apf_dir)


def _apf_status(paths: dict[str, Path]) -> tuple[bool, str]:
    missing = [name for name in ("config_path", "params_path") if not paths[name].exists()]
    if missing:
        return False, "missing " + ",".join(missing)
    chunks = _list_chunks_safe(paths["apf_dir"])
    if not chunks:
        return False, f"missing APF chunks in {paths['apf_dir']}"
    first_chunk = chunks[0][0]
    try:
        with np.load(first_chunk, allow_pickle=False) as data:
            missing_keys = [key for key in REQUIRED_APF_KEYS if key not in data.files]
    except Exception as exc:
        return False, f"cannot read APF chunk {first_chunk}: {exc}"
    if missing_keys:
        return False, f"{first_chunk} missing APF keys: {','.join(missing_keys)}"
    return True, ""


def _apf_ready(paths: dict[str, Path]) -> bool:
    return _apf_status(paths)[0]


def _command(rollout_config: Path, checkpoint_dir: Path, run_root: Path, *, force: bool) -> list[str]:
    cmd = [
        current_python(),
        "scripts/flowlenia_minibang_simulate.py",
        str(rollout_config),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-root",
        str(run_root),
        "--n-trajectories",
        "1",
        "--batch-size",
        "1",
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def _write_aggregate_manifest(
    output_root: Path,
    *,
    rollout_config: Path,
    selected: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
) -> None:
    trajectories = []
    for selection_idx, row in enumerate(selected):
        traj_id = _traj_id(row)
        paths = _run_output_paths(output_root, traj_id)
        apf_ready, apf_message = _apf_status(paths)
        item = {
            "traj_id": traj_id,
            "selection_idx": int(selection_idx),
            "source": "paper_check_flow_lenia_optimized",
            "source_checkpoint_dir": str(row["checkpoint_dir"]),
            "source_root": str(row["source_root"]),
            "source_root_rank": int(row["source_root_rank"]),
            "run_idx": int(row["run_idx"]),
            "traj_dir": str(paths["traj_dir"]),
            "apf_dir": str(paths["apf_dir"]),
            "metrics_path": str(paths["metrics_path"]),
            "config_path": str(paths["config_path"]),
            "params_path": str(paths["params_path"]),
            "apf_ready": bool(apf_ready),
            "metrics_ready": bool(paths["metrics_path"].exists()),
            "ready": bool(apf_ready),
            "apf_status": apf_message,
        }
        trajectories.append(item)
    write_json(
        output_root / "manifest.json",
        {
            "source_kind": "paper_check_flow_lenia_optimized_highres",
            "rollout_config": str(rollout_config),
            "required_apf_keys": list(REQUIRED_APF_KEYS),
            "n_trajectories": len(trajectories),
            "trajectories": trajectories,
            "commands": command_rows,
        },
    )


def run(
    config_path: str | Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path)
    section = _section(cfg)
    if not bool(_get(section, "enabled", True)):
        return {"status": "disabled"}

    output_root = resolve_path(_get(section, "output_root", _get(cfg.get("c2", {}), "trajectory_root", None)))
    if output_root is None:
        output_root = resolve_path("experiments/paper_check_flow_lenia/checkpoints/c2_highres_rollouts")
    assert output_root is not None
    output_root = ensure_dir(output_root)

    rollout_config = resolve_path(_get(section, "rollout_config", "experiments/paper_suite/c2_flowlenia_highres_rollout.yaml"))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"C2 highres rollout_config not found: {rollout_config}")

    selected = _discover_checkpoints(section)
    if not selected:
        if bool(_get(section, "required", True)):
            raise FileNotFoundError("No Flow-Lenia paper-check optimized checkpoints found for C2 highres rollouts.")
        summary = {"status": "skipped", "reason": "no optimized checkpoints found"}
        write_json(output_root / "manifest.json", {"source_kind": "paper_check_flow_lenia_optimized_highres", "trajectories": []})
        write_json(output_root / "simulation_summary.json", summary)
        return summary

    command_rows: list[dict[str, Any]] = []
    for row in selected:
        traj_id = _traj_id(row)
        paths = _run_output_paths(output_root, traj_id)
        cmd = _command(rollout_config, Path(row["checkpoint_dir"]), paths["run_root"], force=force)
        pre_ready, pre_message = _apf_status(paths)
        if pre_ready and not force:
            status = "exists"
            message = ""
        else:
            run_subprocess(cmd, dry_run=dry_run)
            post_ready, post_message = _apf_status(paths)
            status = "dry_run" if dry_run else ("exists" if post_ready else "missing_apf")
            message = pre_message if dry_run else post_message
        command_rows.append(
            {
                "traj_id": traj_id,
                "checkpoint_dir": str(row["checkpoint_dir"]),
                "run_root": str(paths["run_root"]),
                "status": status,
                "message": message,
                "command": command_to_str(cmd),
            }
        )
        _write_aggregate_manifest(output_root, rollout_config=rollout_config, selected=selected, command_rows=command_rows)

    n_apf_ready = sum(1 for row in selected if _apf_ready(_run_output_paths(output_root, _traj_id(row))))
    n_metrics_ready = sum(1 for row in selected if _run_output_paths(output_root, _traj_id(row))["metrics_path"].exists())
    summary = {
        "status": "ok" if n_apf_ready == len(selected) else ("dry_run" if dry_run else "incomplete"),
        "output_root": str(output_root),
        "rollout_config": str(rollout_config),
        "n_selected": len(selected),
        "n_ready": int(n_apf_ready),
        "n_apf_ready": int(n_apf_ready),
        "n_metrics_ready": int(n_metrics_ready),
        "ready_means": "APF/config/params ready; metrics.npz is produced by the metrics layer",
        "manifest": str(output_root / "manifest.json"),
    }
    write_json(output_root / "simulation_summary.json", summary)
    _write_aggregate_manifest(output_root, rollout_config=rollout_config, selected=selected, command_rows=command_rows)
    if not dry_run and n_apf_ready < len(selected) and bool(_get(section, "required", True)):
        raise RuntimeError(f"C2 highres APF rollout generation incomplete: {n_apf_ready}/{len(selected)} ready.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build C2 high-resolution Flow-Lenia rollouts from paper-check optimized checkpoints.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, force=args.force, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
