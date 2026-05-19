from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from paper_suite_common import (
    REPO_ROOT,
    as_list,
    command_to_str,
    current_python,
    ensure_dir,
    load_config,
    log_event,
    resolve_path,
    run_subprocess,
    write_csv,
    write_json,
)
from paper_suite_synthetic import simulate as simulate_synthetic
from paper_suite_c2_branching import simulation as simulate_c2_branching


def _command_from_cfg(raw: Any, config_path: str | Path) -> list[str]:
    cmd = [str(x) for x in as_list(raw)]
    return [
        current_python()
        if x == "{python}"
        else str(config_path)
        if x == "{config}"
        else x
        for x in cmd
    ]


def _glob_paths(pattern: str) -> list[Path]:
    root_pattern = str(resolve_path(pattern) if not Path(pattern).is_absolute() else pattern)
    return [Path(p) for p in sorted(glob.glob(root_pattern))]


def _validate_npz_keys(path: Path, keys: list[str]) -> tuple[bool, str]:
    if not keys:
        return True, ""
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in keys if key not in data.files]
        if missing:
            return False, f"missing keys: {','.join(missing)}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _validate_expected(entry: Any) -> tuple[str, str]:
    required_paths = [resolve_path(x) for x in as_list(entry.get("expected_paths", []))]
    missing = [str(path) for path in required_paths if path is not None and not path.exists()]
    glob_results = []
    for pattern in as_list(entry.get("expected_globs", [])):
        matches = _glob_paths(str(pattern))
        glob_results.append((str(pattern), matches))
        if not matches:
            missing.append(f"glob:{pattern}")
    key_errors = []
    npz_keys = [str(x) for x in as_list(entry.get("expected_npz_keys", []))]
    if npz_keys:
        max_checks = int(entry.get("npz_key_check_limit", 3))
        checked = 0
        for _pattern, matches in glob_results:
            for path in matches[:max_checks]:
                ok, msg = _validate_npz_keys(path, npz_keys)
                checked += 1
                if not ok:
                    key_errors.append(f"{path}: {msg}")
        if checked == 0:
            key_errors.append("no npz files checked")
    if missing or key_errors:
        return "missing", "; ".join(missing + key_errors)
    return "ok", ""


def _task_matches(requested: str, entry_task: str) -> bool:
    if requested == "all":
        return True
    if requested == "paper_check_frustration":
        return entry_task == "paper_check"
    if requested == "paper_check":
        return entry_task in {"paper_check", "paper_check_apf", "paper_check_c1"}
    if requested == "c2":
        return entry_task in {"c2", "apf", "paper_check_apf"}
    return requested == entry_task


def run(config_path: str | Path, *, task: str = "all", smoke: bool = False, force: bool = False, allow_heavy: bool = False, dry_run: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    rows: list[dict[str, Any]] = []
    log_event(
        f"simulation layer start task={task} smoke={smoke} force={force} allow_heavy={allow_heavy} dry_run={dry_run}",
        component="simulation",
    )

    if task in {"all", "synthetic"}:
        log_event("simulation synthetic start", component="simulation")
        if dry_run:
            rows.append(
                {
                    "name": "synthetic_calibration",
                    "layer": "simulation",
                    "status": "dry_run",
                    "message": "synthetic simulation skipped by --dry-run",
                    "command": "",
                }
            )
            log_event("simulation synthetic dry run skipped", component="simulation")
        else:
            result = simulate_synthetic(config_path, smoke=smoke, force=force)
            rows.append({"name": "synthetic_calibration", "layer": "simulation", "status": "ok", "message": str(result), "command": ""})
            log_event(f"simulation synthetic done result={result}", component="simulation")

    if task in {"all", "paper_check", "paper_check_frustration", "paper_check_c1", "apf", "c2"}:
        sim_cfg = cfg.get("simulation", {})
        entries = sim_cfg.get("commands", [])
        for entry in entries:
            name = str(entry.get("name", "unnamed"))
            entry_task = str(entry.get("task", "paper_check"))
            if not _task_matches(task, entry_task):
                continue
            log_event(f"simulation command check name={name} task={entry_task}", component="simulation")
            if smoke:
                rows.append({"name": name, "layer": "simulation", "status": "smoke_skipped", "message": "real simulation command skipped in smoke mode", "command": ""})
                log_event(f"simulation {name} smoke skipped", component="simulation")
                continue
            if not bool(entry.get("enabled", True)):
                rows.append({"name": name, "layer": "simulation", "status": "disabled", "message": "", "command": ""})
                log_event(f"simulation {name} disabled", component="simulation")
                continue
            heavy = bool(entry.get("heavy", True))
            pre_status, pre_msg = _validate_expected(entry)
            always_run = bool(entry.get("always_run", False))
            force_policy = str(entry.get("force_policy", "normal")).strip().lower()
            reuse_existing_on_force = force_policy in {"reuse_existing", "skip_existing", "no_force"}
            if pre_status == "ok" and not always_run and (not force or reuse_existing_on_force):
                rows.append({"name": name, "layer": "simulation", "status": "exists", "message": "expected outputs already present", "command": ""})
                log_event(f"simulation {name} exists", component="simulation")
                continue
            cmd = _command_from_cfg(entry.get("command", []), config_path)
            if not cmd:
                rows.append({"name": name, "layer": "simulation", "status": pre_status, "message": pre_msg or "no command configured", "command": ""})
                log_event(f"simulation {name} no command status={pre_status} message={pre_msg}", component="simulation")
                continue
            if heavy and not allow_heavy:
                rows.append({"name": name, "layer": "simulation", "status": "skipped_heavy", "message": pre_msg, "command": command_to_str(cmd)})
                log_event(f"simulation {name} skipped heavy message={pre_msg}", component="simulation")
                continue
            log_event(f"simulation {name} running command", component="simulation")
            run_subprocess(cmd, dry_run=dry_run)
            if dry_run:
                rows.append({"name": name, "layer": "simulation", "status": "dry_run", "message": pre_msg, "command": command_to_str(cmd)})
                log_event(f"simulation {name} dry run recorded", component="simulation")
                continue
            post_status, post_msg = _validate_expected(entry)
            if post_status != "ok" and bool(entry.get("required", False)):
                log_event(f"simulation {name} invalid required outputs message={post_msg}", component="simulation")
                raise RuntimeError(f"Simulation command {name} finished but outputs are invalid: {post_msg}")
            rows.append({"name": name, "layer": "simulation", "status": post_status, "message": post_msg, "command": command_to_str(cmd)})
            log_event(f"simulation {name} done status={post_status} message={post_msg}", component="simulation")

    if task in {"all", "c2"}:
        log_event("simulation C2 branching pre-metrics check start", component="simulation")
        result = simulate_c2_branching(config_path, smoke=smoke, force=force, allow_heavy=allow_heavy, dry_run=dry_run)
        rows.append({"name": "c2_branching", "layer": "simulation", "status": str(result.get("status", "ok")), "message": str(result), "command": ""})
        log_event(f"simulation C2 branching pre-metrics check done result={result}", component="simulation")

    manifest = output_root / "simulation_layer_manifest.csv"
    write_csv(manifest, rows, fieldnames=["name", "layer", "status", "message", "command"])
    summary = {"n_entries": len(rows), "allow_heavy": bool(allow_heavy), "manifest": str(manifest)}
    write_json(output_root / "simulation_layer_summary.json", summary)
    log_event(f"simulation layer done manifest={manifest}", component="simulation")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-suite simulation layer.")
    parser.add_argument("config")
    parser.add_argument(
        "--task",
        choices=["all", "synthetic", "paper_check", "paper_check_frustration", "paper_check_c1", "apf", "c2"],
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, task=args.task, smoke=args.smoke, force=args.force, allow_heavy=args.allow_heavy, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
