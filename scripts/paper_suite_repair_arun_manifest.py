from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def repair_manifest(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("trajectories", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid manifest trajectories in {manifest_path}")

    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", ""))
        traj_id = str(row.get("traj_id", ""))
        checkpoint_dir = str(row.get("source_checkpoint_dir", ""))
        is_optimized = (
            source == "paper_check_flow_lenia_control_a"
            or "/optimization/" in checkpoint_dir
            or checkpoint_dir.endswith("/optimization")
            or traj_id.startswith("flow_opt")
        )
        if "candidate_kind" not in row and is_optimized:
            row["candidate_kind"] = "optimized"
            changed += 1
        if "candidate_idx" not in row and row.get("candidate_kind") == "optimized":
            row["candidate_idx"] = 0
            changed += 1
        if "candidate_label" not in row and row.get("candidate_kind") == "optimized":
            row["candidate_label"] = "optimized"
            changed += 1

    payload["n_trajectories"] = len(rows)
    if changed and not dry_run:
        backup = manifest_path.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(manifest_path.read_text())
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(manifest_path)
    return {"manifest": str(manifest_path), "n_trajectories": len(rows), "changed_fields": changed, "dry_run": dry_run}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair missing candidate metadata in Flow-Lenia A-run APF manifest.")
    parser.add_argument(
        "root",
        nargs="?",
        default="experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k",
        help="A-run APF root containing manifest.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(repair_manifest(_resolve(args.root), dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
