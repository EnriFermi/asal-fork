from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flowlenia_minibang_common import load_config as load_rollout_config
from flowlenia_minibang_simulate import select_params, simulate_batch
from paper_suite_c2_flowlenia_highres import REQUIRED_APF_KEYS, _discover_checkpoints, _root_label
from paper_suite_common import ensure_dir, load_config, log_event, resolve_path, to_plain, write_json
from paper_suite_flowlenia_arun_apf import (
    _apf_status,
    _get,
    _output_paths,
    _paper_check_control_a_seed,
    _validate_rollout_profile,
)


def _section(cfg: Any) -> Any:
    return _get(cfg.get("simulation", {}), "flow_lenia_nnopt_lagrangian_apf", {})


def _traj_id(row: dict[str, Any], *, prefix: str) -> str:
    source_idx = int(row.get("source_run_idx", row.get("run_idx", -1)))
    if source_idx >= 0 and int(row.get("source_root_rank", 0)) == 0:
        return f"{prefix}_run_{source_idx:03d}"
    if source_idx >= 0:
        return f"{prefix}_root{int(row['source_root_rank']):02d}_{_root_label(Path(row['source_root']))}_run_{source_idx:03d}"
    return (
        f"{prefix}_root{int(row['source_root_rank']):02d}_"
        f"{_root_label(Path(row['source_root']))}_{Path(row['checkpoint_dir']).name}"
    )


def _select_one_checkpoint(
    *,
    row: dict[str, Any],
    section: Any,
    rollout_flat: Any,
    suite_idx: int,
    selection_idx: int,
    traj_id: str,
) -> dict[str, Any]:
    checkpoint_dir = Path(row["checkpoint_dir"])
    selected = select_params(checkpoint_dir, rollout_flat)
    if not selected:
        raise FileNotFoundError(f"No selectable params found in {checkpoint_dir}.")
    item = dict(selected[0])
    item["traj_id"] = str(traj_id)
    item["selection_idx"] = int(selection_idx)
    item["source"] = "paper_check_flow_lenia_nnopt_control_a"
    item["source_objective"] = str(_get(section, "objective_label", "clip_oe"))
    item["source_checkpoint_dir"] = str(checkpoint_dir)
    item["source_root"] = str(row["source_root"])
    item["source_root_rank"] = int(row["source_root_rank"])
    item["source_run_idx"] = int(row.get("source_run_idx", -1))
    item["suite_run_idx"] = int(row.get("run_idx", suite_idx))
    item["run_seed"] = _paper_check_control_a_seed(section, row, suite_idx)
    item["candidate_kind"] = "optimized"
    item["candidate_idx"] = 0
    item["candidate_label"] = "optimized"
    return item


def _write_manifest(
    output_root: Path,
    *,
    rollout_config: Path,
    selected: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
    expected_steps: int | None = None,
    objective_label: str = "clip_oe",
) -> None:
    trajectories: list[dict[str, Any]] = []
    for row in selected:
        traj_id = str(row["traj_id"])
        paths = _output_paths(output_root, traj_id)
        apf_ready, apf_message = _apf_status(paths, expected_steps=expected_steps)
        trajectories.append(
            {
                "traj_id": traj_id,
                "selection_idx": int(row["selection_idx"]),
                "source": str(row.get("source", "paper_check_flow_lenia_nnopt_control_a")),
                "source_objective": str(row.get("source_objective", objective_label)),
                "source_checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                "source_root": str(row.get("source_root", "")),
                "source_root_rank": int(row.get("source_root_rank", -1)),
                "source_run_idx": int(row.get("source_run_idx", -1)),
                "suite_run_idx": int(row.get("suite_run_idx", row["selection_idx"])),
                "run_seed": int(row.get("run_seed", -1)),
                "candidate_kind": "optimized",
                "candidate_idx": 0,
                "candidate_label": "optimized",
                "traj_dir": str(paths["traj_dir"]),
                "apf_dir": str(paths["apf_dir"]),
                "metrics_path": str(paths["metrics_path"]),
                "config_path": str(paths["config_path"]),
                "params_path": str(paths["params_path"]),
                "video_path": str(paths["video_path"]),
                "frame_times_path": str(paths["frame_times_path"]),
                "apf_ready": bool(apf_ready),
                "ready": bool(apf_ready),
                "apf_status": apf_message,
            }
        )
    write_json(
        output_root / "manifest.json",
        {
            "source_kind": "paper_check_flow_lenia_nnopt_lagrangian_sparse_apf",
            "source_objective": str(objective_label),
            "rollout_config": str(rollout_config),
            "required_apf_keys": list(REQUIRED_APF_KEYS),
            "random_baselines": False,
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
        log_event("Flow-Lenia nn-opt APF disabled", component="nnopt-apf")
        return {"status": "disabled"}
    if bool(_get(section, "include_random_baselines", False)):
        raise ValueError("Flow-Lenia nn-opt APF builder is opt-only; set include_random_baselines=false.")

    output_root = resolve_path(_get(section, "output_root", "experiments/paper_check_flow_lenia/checkpoints/nnopt_lagrangian_apf_500k"))
    assert output_root is not None
    output_root = ensure_dir(output_root)
    rollout_config = resolve_path(_get(section, "rollout_config", "experiments/paper_suite/flowlenia_arun_apf_500k.yaml"))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"Flow-Lenia nn-opt APF rollout_config not found: {rollout_config}")

    expected_steps = int(_get(section, "rollout_steps", 500_000))
    rollout_cfg, rollout_flat = load_rollout_config(rollout_config, [])
    _validate_rollout_profile(rollout_flat, expected_steps=expected_steps)

    n_per_checkpoint = int(_get(section, "n_trajectories_per_checkpoint", 1))
    if n_per_checkpoint != 1:
        raise ValueError("Flow-Lenia nn-opt APF currently expects n_trajectories_per_checkpoint=1.")
    rollout_flat.n_trajectories = int(n_per_checkpoint)
    batch_size = max(1, int(_get(section, "batch_size", _get(rollout_flat, "batch_size", 1))))
    rollout_flat.batch_size = int(batch_size)
    prefix = str(_get(section, "traj_prefix", "nnopt")).strip() or "nnopt"
    objective_label = str(_get(section, "objective_label", "clip_oe"))

    checkpoints = _discover_checkpoints(section)
    log_event(
        "Flow-Lenia nn-opt APF start "
        f"force={force} dry_run={dry_run} output_root={output_root} "
        f"rollout_steps={expected_steps} batch_size={batch_size} n_checkpoints={len(checkpoints)}",
        component="nnopt-apf",
    )
    if not checkpoints:
        summary = {"status": "skipped", "reason": "no nn-opt checkpoints found", "output_root": str(output_root)}
        write_json(
            output_root / "manifest.json",
            {
                "source_kind": "paper_check_flow_lenia_nnopt_lagrangian_sparse_apf",
                "source_objective": objective_label,
                "random_baselines": False,
                "trajectories": [],
            },
        )
        write_json(output_root / "simulation_summary.json", summary)
        if bool(_get(section, "required", False)) and not dry_run:
            raise FileNotFoundError("No Flow-Lenia nn-opt checkpoints found for APF rollouts.")
        return summary

    selected: list[dict[str, Any]] = []
    for suite_idx, row in enumerate(checkpoints):
        selected.append(
            _select_one_checkpoint(
                row=row,
                section=section,
                rollout_flat=rollout_flat,
                suite_idx=suite_idx,
                selection_idx=len(selected),
                traj_id=_traj_id(row, prefix=prefix),
            )
        )

    ready_by_traj: dict[str, tuple[bool, str]] = {}
    for row in selected:
        ready, message = _apf_status(_output_paths(output_root, str(row["traj_id"])), expected_steps=expected_steps)
        ready_by_traj[str(row["traj_id"])] = (ready, message)
    pending = [row for row in selected if force or not ready_by_traj[str(row["traj_id"])][0]]
    batches_to_run = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    run_traj_ids = {str(row["traj_id"]) for batch in batches_to_run for row in batch}

    command_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        paths = _output_paths(output_root, str(row["traj_id"]))
        ready, message = ready_by_traj[str(row["traj_id"])]
        if str(row["traj_id"]) not in run_traj_ids:
            status = "exists"
            log_event(f"Flow-Lenia nn-opt APF {idx}/{len(selected)} traj={row['traj_id']} exists", component="nnopt-apf")
        else:
            status = "queued"
            reason = "force" if force else message
            log_event(
                f"Flow-Lenia nn-opt APF {idx}/{len(selected)} traj={row['traj_id']} queued pre_status={reason}",
                component="nnopt-apf",
            )
            message = reason
        command_rows.append(
            {
                "traj_id": str(row["traj_id"]),
                "checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                "run_root": str(paths["run_root"]),
                "status": status,
                "message": message,
                "command": "internal simulate_batch",
            }
        )
    _write_manifest(
        output_root,
        rollout_config=rollout_config,
        selected=selected,
        command_rows=command_rows,
        expected_steps=expected_steps,
        objective_label=objective_label,
    )

    if dry_run:
        for row in command_rows:
            if row["status"] == "queued":
                row["status"] = "dry_run"
        _write_manifest(
            output_root,
            rollout_config=rollout_config,
            selected=selected,
            command_rows=command_rows,
            expected_steps=expected_steps,
            objective_label=objective_label,
        )
        summary = {
            "status": "dry_run",
            "output_root": str(output_root),
            "rollout_config": str(rollout_config),
            "n_selected": len(selected),
            "n_to_run": len(run_traj_ids),
            "n_batches_to_run": len(batches_to_run),
            "batch_size": int(batch_size),
            "manifest": str(output_root / "manifest.json"),
        }
        write_json(output_root / "simulation_summary.json", summary)
        log_event(
            f"Flow-Lenia nn-opt APF dry-run n_batches_to_run={len(batches_to_run)} n_to_run={len(run_traj_ids)}",
            component="nnopt-apf",
        )
        return summary

    flat_dict = OmegaConf.to_container(rollout_flat, resolve=True)
    for batch_idx, batch in enumerate(batches_to_run, start=1):
        batch_ids = [str(row["traj_id"]) for row in batch]
        existing = [
            _output_paths(output_root, str(row["traj_id"]))["run_root"]
            for row in batch
            if _output_paths(output_root, str(row["traj_id"]))["run_root"].exists()
        ]
        if existing and not force:
            raise RuntimeError(
                "Refusing to overwrite incomplete Flow-Lenia nn-opt APF trajectory directories without --force. "
                "Inspect or move them first: "
                + ", ".join(str(path) for path in existing[:10])
            )
        overwrite = bool(force or existing)
        log_event(
            f"Flow-Lenia nn-opt APF running batch {batch_idx}/{len(batches_to_run)} "
            f"batch_size={len(batch)} overwrite={overwrite} traj_ids={batch_ids}",
            component="nnopt-apf",
        )
        simulate_batch(
            selected_batch=batch,
            cfg=rollout_cfg,
            flat_args=dict(flat_dict),
            output_root=output_root,
            overwrite=overwrite,
        )
        for row in command_rows:
            if row["traj_id"] in batch_ids:
                paths = _output_paths(output_root, str(row["traj_id"]))
                ready, message = _apf_status(paths, expected_steps=expected_steps)
                row["status"] = "exists" if ready else "missing_apf"
                row["message"] = message
        _write_manifest(
            output_root,
            rollout_config=rollout_config,
            selected=selected,
            command_rows=command_rows,
            expected_steps=expected_steps,
            objective_label=objective_label,
        )

    n_ready = 0
    for row in selected:
        ready, _message = _apf_status(_output_paths(output_root, str(row["traj_id"])), expected_steps=expected_steps)
        n_ready += int(ready)
    status = "ok" if n_ready == len(selected) else "incomplete"
    summary = {
        "status": status,
        "output_root": str(output_root),
        "rollout_config": str(rollout_config),
        "n_selected": len(selected),
        "n_ready": int(n_ready),
        "batch_size": int(batch_size),
        "rollout_steps": int(expected_steps),
        "manifest": str(output_root / "manifest.json"),
    }
    write_json(output_root / "simulation_summary.json", summary)
    _write_manifest(
        output_root,
        rollout_config=rollout_config,
        selected=selected,
        command_rows=command_rows,
        expected_steps=expected_steps,
        objective_label=objective_label,
    )
    if status != "ok" and bool(_get(section, "required", False)):
        raise RuntimeError(f"Flow-Lenia nn-opt APF generation incomplete: {n_ready}/{len(selected)} ready.")
    log_event(f"Flow-Lenia nn-opt APF done status={status} n_ready={n_ready}/{len(selected)}", component="nnopt-apf")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Flow-Lenia nn-opt lagrangian + sparse APF rollouts.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(to_plain(run(args.config, force=args.force, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
