from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text())


def _resolve_path(root: Path, raw: Any) -> str:
    if raw is None or str(raw) == "":
        return ""
    path = Path(str(raw))
    if path.is_absolute():
        return str(path)
    return str(root / path)


def _canonical_kind(row: dict[str, Any]) -> str:
    kind = str(row.get("candidate_kind", "")).strip().lower()
    label = str(row.get("candidate_label", "")).strip().lower()
    traj_id = str(row.get("traj_id", "")).strip().lower()
    text = " ".join([kind, label, traj_id])
    if "random" in text:
        return "random"
    if "optimized" in text or "flow_opt" in text:
        return "optimized"
    return kind or "other"


def _materialize_row(root: Path, row: dict[str, Any], *, selection_idx: int, optimized_run_idx: int) -> dict[str, Any]:
    out = dict(row)
    out["selection_idx"] = int(selection_idx)
    out["optimized_run_idx"] = int(optimized_run_idx)
    out["suite_run_idx"] = int(optimized_run_idx)
    out["traj_dir"] = _resolve_path(root, row.get("traj_dir"))
    out["apf_dir"] = _resolve_path(root, row.get("apf_dir"))
    out["metrics_path"] = _resolve_path(root, row.get("metrics_path"))
    out["config_path"] = _resolve_path(root, row.get("config_path"))
    out["params_path"] = _resolve_path(root, row.get("params_path"))
    out["candidate_kind"] = _canonical_kind(row)
    return out


def _check_ready(row: dict[str, Any]) -> None:
    apf_dir = Path(str(row.get("apf_dir", "")))
    if not apf_dir.exists():
        raise FileNotFoundError(f"Missing APF dir for {row.get('traj_id')}: {apf_dir}")
    if not any(apf_dir.glob("P_steps_*.npz")):
        raise FileNotFoundError(f"No P_steps_*.npz chunks for {row.get('traj_id')}: {apf_dir}")


def build(
    *,
    optimized_root: Path,
    random_root: Path,
    output_root: Path,
    expected_optimized: int | None,
    expected_random: int | None,
    force: bool,
) -> dict[str, Any]:
    output_manifest = output_root / "manifest.json"
    if output_manifest.exists() and not force:
        payload = json.loads(output_manifest.read_text())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    opt_manifest = _load_manifest(optimized_root)
    random_manifest = _load_manifest(random_root)
    opt_rows = [
        row
        for row in opt_manifest.get("trajectories", [])
        if _canonical_kind(row) == "optimized"
    ]
    random_rows = [
        row
        for row in random_manifest.get("trajectories", [])
        if _canonical_kind(row) == "random"
    ]
    if expected_optimized is not None and len(opt_rows) != int(expected_optimized):
        raise ValueError(f"Expected {expected_optimized} optimized rows, found {len(opt_rows)} in {optimized_root}.")
    if expected_random is not None and len(random_rows) != int(expected_random):
        raise ValueError(f"Expected {expected_random} random rows, found {len(random_rows)} in {random_root}.")
    if not opt_rows:
        raise ValueError(f"No optimized rows found in {optimized_root}/manifest.json.")
    if not random_rows:
        raise ValueError(f"No random rows found in {random_root}/manifest.json.")

    optimized_run_idx = int(opt_rows[0].get("suite_run_idx", opt_rows[0].get("optimized_run_idx", 0)))
    trajectories: list[dict[str, Any]] = []
    for row in opt_rows:
        materialized = _materialize_row(
            optimized_root,
            row,
            selection_idx=len(trajectories),
            optimized_run_idx=optimized_run_idx,
        )
        materialized["candidate_kind"] = "optimized"
        materialized["candidate_idx"] = int(row.get("candidate_idx", 0))
        materialized["candidate_label"] = str(row.get("candidate_label", "optimized"))
        _check_ready(materialized)
        trajectories.append(materialized)
    for random_idx, row in enumerate(random_rows):
        materialized = _materialize_row(
            random_root,
            row,
            selection_idx=len(trajectories),
            optimized_run_idx=optimized_run_idx,
        )
        materialized["candidate_kind"] = "random"
        materialized["candidate_idx"] = int(row.get("candidate_idx", random_idx))
        materialized["candidate_label"] = str(row.get("candidate_label", f"random_{random_idx:03d}"))
        _check_ready(materialized)
        trajectories.append(materialized)

    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_kind": "flowlenia_c1_combined_existing_apf_manifest",
        "optimized_root": str(optimized_root),
        "random_root": str(random_root),
        "n_optimized": len(opt_rows),
        "n_random": len(random_rows),
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
        "commands": [],
    }
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_root / "combined_manifest_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a C1 APF manifest from a new optimized root and an existing random root.")
    parser.add_argument("--optimized-root", required=True, help="APF root containing optimized trajectories.")
    parser.add_argument("--random-root", required=True, help="APF root containing already-computed random trajectories.")
    parser.add_argument("--output-root", required=True, help="Output root where combined manifest.json will be written.")
    parser.add_argument("--expected-optimized", type=int, default=None)
    parser.add_argument("--expected-random", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(
        optimized_root=Path(args.optimized_root),
        random_root=Path(args.random_root),
        output_root=Path(args.output_root),
        expected_optimized=args.expected_optimized,
        expected_random=args.expected_random,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
