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

from flowlenia_minibang_common import list_apf_chunks
from flowlenia_minibang_common import load_config as load_rollout_config
from flowlenia_minibang_simulate import select_params, simulate_batch
from paper_suite_c2_flowlenia_highres import (
    REQUIRED_APF_KEYS,
    _apf_status as _base_apf_status,
    _discover_checkpoints,
    _traj_id,
)
from paper_suite_common import (
    ensure_dir,
    load_config,
    log_event,
    resolve_path,
    to_plain,
    write_json,
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _section(cfg: Any) -> Any:
    return _get(cfg.get("simulation", {}), "flow_lenia_arun_lagrangian_apf", {})


def _output_paths(output_root: Path, traj_id: str) -> dict[str, Path]:
    traj_dir = output_root / traj_id
    return {
        "run_root": traj_dir,
        "traj_dir": traj_dir,
        "apf_dir": traj_dir / "apf_logs",
        "metrics_path": traj_dir / "metrics.npz",
        "config_path": traj_dir / "config.yaml",
        "params_path": traj_dir / "params.npy",
        "video_path": traj_dir / "video.mp4",
        "frame_times_path": traj_dir / "frame_times.csv",
    }


def _expected_rollout_signature(rollout_config: Path, flat: Any) -> dict[str, Any]:
    keys = (
        "grid_size",
        "seed_n_patches",
        "seed_patch_size",
        "rollout_steps",
        "max_steps",
        "snapshot_interval",
        "snapshots_per_file",
        "save_A",
        "save_F",
        "save_lagrangian",
        "lagrangian_n_particles",
        "lagrangian_init_mode",
        "lagrangian_flow_channel",
        "lagrangian_flow_reduce",
        "lagrangian_channel_mode",
        "lagrangian_noise_model",
        "lagrangian_diffusion_scale",
    )
    out = {"rollout_config": str(rollout_config)}
    for key in keys:
        value = _get(flat, key, None)
        if value is not None:
            out[key] = value
    return out


def _saved_rollout_signature(paths: dict[str, Path]) -> dict[str, Any]:
    cfg_path = paths["config_path"]
    if not cfg_path.exists():
        return {}
    cfg = OmegaConf.load(cfg_path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
        cfg.get("minibang", {}),
    )
    return _expected_rollout_signature(Path(str(cfg.get("rollout_config", ""))), flat)


def _signature_mismatch(saved: dict[str, Any], expected: dict[str, Any]) -> str:
    for key, expected_value in expected.items():
        if key == "rollout_config":
            continue
        if key not in saved:
            return f"saved config missing {key}"
        if saved[key] != expected_value:
            return f"saved config {key}={saved[key]!r}, expected {expected_value!r}"
    return ""


def _apf_status(
    paths: dict[str, Path],
    *,
    expected_steps: int | None = None,
    expected_signature: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    ready, message = _base_apf_status(paths)
    if not ready or expected_steps is None:
        return ready, message
    if expected_signature is not None:
        try:
            saved_signature = _saved_rollout_signature(paths)
        except Exception as exc:
            return False, f"cannot read saved config {paths['config_path']}: {exc}"
        mismatch = _signature_mismatch(saved_signature, expected_signature)
        if mismatch:
            return False, mismatch
    chunks = list_apf_chunks(paths["apf_dir"])
    if not chunks:
        return False, f"missing APF chunks in {paths['apf_dir']}"
    first_start = min(start for _path, start, _end, _idx in chunks)
    last_end = max(end for _path, _start, end, _idx in chunks)
    if first_start > 0:
        return False, f"APF coverage starts at {first_start}, expected 0"
    if last_end < int(expected_steps):
        return False, f"APF coverage ends at {last_end}, expected >= {int(expected_steps)}"
    return True, ""


def _validate_rollout_profile(flat: Any, *, expected_steps: int) -> None:
    rollout_steps = int(_get(flat, "rollout_steps", expected_steps))
    max_steps_raw = _get(flat, "max_steps", rollout_steps)
    max_steps = int(max_steps_raw) if max_steps_raw is not None else rollout_steps
    if rollout_steps != int(expected_steps) or max_steps != int(expected_steps):
        raise ValueError(
            "Flow-Lenia A-run APF rollout must be pinned to the requested limit. "
            f"Expected rollout_steps=max_steps={expected_steps}, got "
            f"rollout_steps={rollout_steps}, max_steps={max_steps}."
        )
    for key in ("save_A", "save_F", "save_lagrangian"):
        value = _get(flat, key, True)
        if str(value).strip().lower() in {"0", "false", "no", "off"}:
            raise ValueError(f"Flow-Lenia A-run APF rollout requires {key}=true.")


def _paper_check_control_a_seed(section: Any, row: dict[str, Any], suite_idx: int) -> int:
    base = int(_get(section, "run_seed_base", 400_000))
    mode = str(_get(section, "run_seed_mode", "source_run_idx")).strip().lower()
    source_run_idx = int(row.get("source_run_idx", -1))
    if mode == "source_run_idx" and source_run_idx >= 0:
        group_idx = source_run_idx
    elif mode == "suite_index":
        group_idx = int(suite_idx)
    else:
        raise ValueError("run_seed_mode must be 'source_run_idx' or 'suite_index'.")
    return int(base + 2 * group_idx)


def _candidate_traj_id(row: dict[str, Any], *, candidate_kind: str, candidate_idx: int) -> str:
    base = _traj_id(row)
    if candidate_kind == "optimized":
        return base
    return f"{base}_{candidate_kind}_{candidate_idx:03d}"


def _random_checkpoint_dir(row: dict[str, Any], random_idx: int) -> Path:
    source_root = Path(row["source_root"])
    source_run_idx = int(row.get("source_run_idx", -1))
    if source_run_idx < 0:
        raise ValueError(f"Cannot derive random checkpoint dir without source_run_idx: {row}")
    return (
        source_root.parent
        / "frustration_simulation"
        / "random_params"
        / f"group_{source_run_idx:03d}"
        / f"random_{int(random_idx):03d}"
    )


def _select_one_checkpoint(
    *,
    row: dict[str, Any],
    section: Any,
    rollout_flat: Any,
    suite_idx: int,
    selection_idx: int,
    checkpoint_dir: Path,
    traj_id: str,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
) -> dict[str, Any]:
    selected = select_params(checkpoint_dir, rollout_flat)
    if not selected:
        raise FileNotFoundError(f"No selectable params found in {checkpoint_dir}.")
    item = dict(selected[0])
    item["traj_id"] = str(traj_id)
    item["selection_idx"] = int(selection_idx)
    item["source"] = "paper_check_flow_lenia_control_a"
    item["source_checkpoint_dir"] = str(checkpoint_dir)
    item["source_root"] = str(row["source_root"])
    item["source_root_rank"] = int(row["source_root_rank"])
    item["source_run_idx"] = int(row.get("source_run_idx", -1))
    item["suite_run_idx"] = int(row.get("run_idx", suite_idx))
    item["run_seed"] = _paper_check_control_a_seed(section, row, suite_idx)
    item["candidate_kind"] = str(candidate_kind)
    item["candidate_idx"] = int(candidate_idx)
    item["candidate_label"] = str(candidate_label)
    return item


def _write_manifest(
    output_root: Path,
    *,
    rollout_config: Path,
    selected: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
    expected_steps: int | None = None,
    expected_signature: dict[str, Any] | None = None,
) -> None:
    trajectories: list[dict[str, Any]] = []
    for row in selected:
        traj_id = str(row["traj_id"])
        paths = _output_paths(output_root, traj_id)
        apf_ready, apf_message = _apf_status(
            paths,
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        trajectories.append(
            {
                "traj_id": traj_id,
                "selection_idx": int(row["selection_idx"]),
                "source": str(row.get("source", "paper_check_flow_lenia_control_a")),
                "source_checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                "source_root": str(row.get("source_root", "")),
                "source_root_rank": int(row.get("source_root_rank", -1)),
                "source_run_idx": int(row.get("source_run_idx", -1)),
                "suite_run_idx": int(row.get("suite_run_idx", row["selection_idx"])),
                "run_seed": int(row.get("run_seed", -1)),
                "candidate_kind": str(row.get("candidate_kind", "optimized")),
                "candidate_idx": int(row.get("candidate_idx", 0)),
                "candidate_label": str(row.get("candidate_label", "optimized")),
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
            "source_kind": "paper_check_flow_lenia_control_a_lagrangian_sparse_apf",
            "rollout_config": str(rollout_config),
            "expected_rollout_signature": dict(expected_signature or {}),
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
        log_event("Flow-Lenia A-run APF disabled", component="arun-apf")
        return {"status": "disabled"}

    output_root = resolve_path(_get(section, "output_root", "experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k"))
    assert output_root is not None
    output_root = ensure_dir(output_root)
    rollout_config = resolve_path(_get(section, "rollout_config", "experiments/paper_suite/flowlenia_arun_apf_500k.yaml"))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"Flow-Lenia A-run APF rollout_config not found: {rollout_config}")

    expected_steps = int(_get(section, "rollout_steps", 500_000))
    rollout_cfg, rollout_flat = load_rollout_config(rollout_config, [])
    _validate_rollout_profile(rollout_flat, expected_steps=expected_steps)
    expected_signature = _expected_rollout_signature(rollout_config, rollout_flat)

    n_per_checkpoint = int(_get(section, "n_trajectories_per_checkpoint", 1))
    if n_per_checkpoint != 1:
        raise ValueError("Flow-Lenia A-run APF currently expects n_trajectories_per_checkpoint=1.")
    rollout_flat.n_trajectories = int(n_per_checkpoint)
    batch_size = max(1, int(_get(section, "batch_size", _get(rollout_flat, "batch_size", 1))))
    rollout_flat.batch_size = int(batch_size)

    checkpoints = _discover_checkpoints(section)
    log_event(
        "Flow-Lenia A-run APF start "
        f"force={force} dry_run={dry_run} output_root={output_root} "
        f"rollout_steps={expected_steps} batch_size={batch_size} n_checkpoints={len(checkpoints)}",
        component="arun-apf",
    )
    if not checkpoints:
        summary = {"status": "skipped", "reason": "no optimized checkpoints found", "output_root": str(output_root)}
        write_json(output_root / "manifest.json", {"source_kind": "paper_check_flow_lenia_control_a_lagrangian_sparse_apf", "trajectories": []})
        write_json(output_root / "simulation_summary.json", summary)
        if bool(_get(section, "required", True)) and not dry_run:
            raise FileNotFoundError("No Flow-Lenia optimized checkpoints found for A-run APF rollouts.")
        return summary

    selected: list[dict[str, Any]] = []
    include_random = bool(_get(section, "include_random_baselines", True))
    n_random = int(_get(section, "num_random_baselines", 3)) if include_random else 0
    for suite_idx, row in enumerate(checkpoints):
        selected.append(
            _select_one_checkpoint(
                row=row,
                section=section,
                rollout_flat=rollout_flat,
                suite_idx=suite_idx,
                selection_idx=len(selected),
                checkpoint_dir=Path(row["checkpoint_dir"]),
                traj_id=_candidate_traj_id(row, candidate_kind="optimized", candidate_idx=0),
                candidate_kind="optimized",
                candidate_idx=0,
                candidate_label="optimized",
            )
        )
        for random_idx in range(n_random):
            random_dir = _random_checkpoint_dir(row, random_idx)
            if not (random_dir / "best.pkl").exists():
                raise FileNotFoundError(
                    "Missing random baseline checkpoint for Flow-Lenia A-run APF. "
                    f"Expected {random_dir / 'best.pkl'} for suite_group={suite_idx}, "
                    f"source_run_idx={int(row.get('source_run_idx', -1))}, random_idx={random_idx}."
                )
            selected.append(
                _select_one_checkpoint(
                    row=row,
                    section=section,
                    rollout_flat=rollout_flat,
                    suite_idx=suite_idx,
                    selection_idx=len(selected),
                    checkpoint_dir=random_dir,
                    traj_id=_candidate_traj_id(row, candidate_kind="random", candidate_idx=random_idx),
                    candidate_kind="random",
                    candidate_idx=random_idx,
                    candidate_label=f"random_{random_idx:03d}",
                )
            )

    ready_by_traj: dict[str, tuple[bool, str]] = {}
    for row in selected:
        ready, message = _apf_status(
            _output_paths(output_root, str(row["traj_id"])),
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        ready_by_traj[str(row["traj_id"])] = (ready, message)
    pending = [
        row
        for row in selected
        if force or not ready_by_traj[str(row["traj_id"])][0]
    ]
    batches_to_run = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

    run_traj_ids = {str(row["traj_id"]) for batch in batches_to_run for row in batch}
    command_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        paths = _output_paths(output_root, str(row["traj_id"]))
        ready, message = ready_by_traj[str(row["traj_id"])]
        if str(row["traj_id"]) not in run_traj_ids:
            status = "exists"
            log_event(f"Flow-Lenia A-run APF {idx}/{len(selected)} traj={row['traj_id']} exists", component="arun-apf")
        else:
            status = "queued"
            reason = "force" if force else message
            log_event(
                f"Flow-Lenia A-run APF {idx}/{len(selected)} traj={row['traj_id']} queued "
                f"pre_status={reason}",
                component="arun-apf",
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
        expected_signature=expected_signature,
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
            expected_signature=expected_signature,
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
            f"Flow-Lenia A-run APF dry-run n_batches_to_run={len(batches_to_run)} n_to_run={len(run_traj_ids)}",
            component="arun-apf",
        )
        return summary

    flat_dict = OmegaConf.to_container(rollout_flat, resolve=True)
    for batch_idx, batch in enumerate(batches_to_run, start=1):
        batch_ids = [str(row["traj_id"]) for row in batch]
        # Every selected row carries an explicit run_seed, so missing random
        # candidates can be filled without re-running already ready optimized
        # trajectories.
        existing = [
            (row, _output_paths(output_root, str(row["traj_id"]))["run_root"])
            for row in batch
            if _output_paths(output_root, str(row["traj_id"]))["run_root"].exists()
        ]
        protected = [
            str(path)
            for row, path in existing
            if str(row.get("candidate_kind", "optimized")) == "optimized"
        ]
        if protected and not force:
            raise RuntimeError(
                "Refusing to overwrite incomplete optimized Flow-Lenia A-run APF trajectory directories without --force. "
                "Inspect or move them first: "
                + ", ".join(protected[:10])
            )
        overwrite = bool(force or existing)
        log_event(
            f"Flow-Lenia A-run APF running batch {batch_idx}/{len(batches_to_run)} "
            f"batch_size={len(batch)} overwrite={overwrite} traj_ids={batch_ids}",
            component="arun-apf",
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
                ready, message = _apf_status(
                    paths,
                    expected_steps=expected_steps,
                    expected_signature=expected_signature,
                )
                row["status"] = "exists" if ready else "missing_apf"
                row["message"] = message
        _write_manifest(
            output_root,
            rollout_config=rollout_config,
            selected=selected,
            command_rows=command_rows,
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )

    n_ready = 0
    for row in selected:
        ready, _message = _apf_status(
            _output_paths(output_root, str(row["traj_id"])),
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
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
        expected_signature=expected_signature,
    )
    if status != "ok" and bool(_get(section, "required", True)):
        raise RuntimeError(f"Flow-Lenia A-run APF generation incomplete: {n_ready}/{len(selected)} ready.")
    log_event(f"Flow-Lenia A-run APF done status={status} n_ready={n_ready}/{len(selected)}", component="arun-apf")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Flow-Lenia control-A style lagrangian + sparse APF rollouts.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(to_plain(run(args.config, force=args.force, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
