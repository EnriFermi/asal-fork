from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else _REPO_ROOT / path


def _candidate_kind(row: dict[str, Any], traj_id: str) -> str:
    raw = str(row.get("candidate_kind", "") or row.get("candidate_label", "") or traj_id).lower()
    checkpoint = str(row.get("source_checkpoint_dir", "")).lower()
    if "random" in raw or "/random" in checkpoint:
        return "random"
    if "optimized" in raw or "flow_opt" in traj_id.lower() or "/optimization/" in checkpoint:
        return "optimized"
    return "missing"


def _path_from_manifest(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _manifest_items(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        rows = payload.get("trajectories", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid trajectories in {manifest}")
        out = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            traj_id = str(row.get("traj_id", f"traj_{idx:05d}"))
            traj_dir = _path_from_manifest(root, row.get("traj_dir"), default=root / traj_id)
            out.append(
                {
                    "traj_id": traj_id,
                    "group": int(row.get("suite_run_idx", row.get("optimized_run_idx", row.get("selection_idx", idx)))),
                    "candidate_kind": _candidate_kind(row, traj_id),
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "traj_dir": traj_dir,
                    "apf_dir": _path_from_manifest(root, row.get("apf_dir"), default=traj_dir / "apf_logs"),
                    "metrics_path": _path_from_manifest(root, row.get("metrics_path"), default=traj_dir / "metrics.npz"),
                }
            )
        return out

    out = []
    for idx, traj_dir in enumerate(sorted(root.glob("flow_opt*"))):
        if not traj_dir.is_dir():
            continue
        out.append(
            {
                "traj_id": traj_dir.name,
                "group": idx,
                "candidate_kind": _candidate_kind({}, traj_dir.name),
                "candidate_idx": 0,
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
            }
        )
    return out


def _apf_step_range(apf_dir: Path) -> tuple[int | None, int | None, int]:
    chunks = sorted(apf_dir.glob("P_steps_*.npz"))
    mins: list[int] = []
    maxs: list[int] = []
    for path in chunks:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "steps" in data.files:
                    steps = np.asarray(data["steps"], dtype=np.int64).reshape(-1)
                    if steps.size:
                        mins.append(int(np.min(steps)))
                        maxs.append(int(np.max(steps)))
                        continue
        except Exception:
            pass
        match = re.search(r"P_steps_(-?\d+)_(-?\d+)", path.name)
        if match:
            mins.append(int(match.group(1)))
            maxs.append(int(match.group(2)))
    if not mins or not maxs:
        return None, None, len(chunks)
    return min(mins), max(maxs), len(chunks)


def _metric_range(metrics_path: Path) -> tuple[int | None, int | None, str]:
    if not metrics_path.exists():
        return None, None, "missing"
    try:
        with np.load(metrics_path, allow_pickle=False) as data:
            missing = [
                key
                for key in (
                    "delta_h_window_start_steps",
                    "delta_h_window_end_steps",
                    "delta_h_score_by_tau",
                    "delta_h_map",
                )
                if key not in data.files
            ]
            if missing:
                return None, None, "missing_keys:" + ",".join(missing)
            starts = np.asarray(data["delta_h_window_start_steps"], dtype=np.int64).reshape(-1)
            ends = np.asarray(data["delta_h_window_end_steps"], dtype=np.int64).reshape(-1)
            if starts.size == 0 or ends.size == 0:
                return None, None, "empty_windows"
            return int(np.min(starts)), int(np.max(ends)), "ok"
    except Exception as exc:
        return None, None, f"read_error:{type(exc).__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Flow-Lenia C1 APF coverage for optimized/random checkpoints.")
    parser.add_argument("root", nargs="?", default="experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k")
    parser.add_argument("--expected-groups", type=int, default=9)
    parser.add_argument("--random-per-group", type=int, default=3)
    parser.add_argument("--range-start", type=int, default=100000)
    parser.add_argument("--range-end", type=int, default=300000)
    args = parser.parse_args()

    root = _resolve(args.root)
    items = _manifest_items(root)
    counts = Counter(item["candidate_kind"] for item in items)
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_group[int(item["group"])].append(item)

    errors: list[str] = []
    expected_total = int(args.expected_groups) * (1 + int(args.random_per_group))
    if len(items) != expected_total:
        errors.append(f"expected {expected_total} manifest items, got {len(items)}")
    if counts.get("optimized", 0) != int(args.expected_groups):
        errors.append(f"expected {args.expected_groups} optimized items, got {counts.get('optimized', 0)}")
    expected_random = int(args.expected_groups) * int(args.random_per_group)
    if counts.get("random", 0) != expected_random:
        errors.append(f"expected {expected_random} random items, got {counts.get('random', 0)}")
    if counts.get("missing", 0):
        errors.append(f"{counts.get('missing', 0)} items have missing/unknown candidate_kind")

    print("=== C1 APF COVERAGE ===")
    print("root:", root)
    print("n_items:", len(items))
    print("candidate_counts:", dict(counts))
    print("expected_total:", expected_total)
    print()

    print("=== GROUPS ===")
    for group in range(int(args.expected_groups)):
        rows = by_group.get(group, [])
        gc = Counter(row["candidate_kind"] for row in rows)
        print(f"group {group:03d}: n={len(rows)} counts={dict(gc)}")
        if gc.get("optimized", 0) != 1 or gc.get("random", 0) != int(args.random_per_group):
            errors.append(f"group {group:03d} expected 1 optimized + {args.random_per_group} random, got {dict(gc)}")
    extra_groups = sorted(set(by_group) - set(range(int(args.expected_groups))))
    if extra_groups:
        errors.append(f"unexpected groups: {extra_groups}")
    print()

    print("=== ITEM RANGES ===")
    for item in items:
        m_start, m_end, m_status = _metric_range(Path(item["metrics_path"]))
        a_start, a_end, n_chunks = _apf_step_range(Path(item["apf_dir"]))
        print(
            f"group={item['group']:03d} kind={item['candidate_kind']:<9} "
            f"traj={item['traj_id']} metrics={m_status} metric_range={m_start}..{m_end} "
            f"apf_chunks={n_chunks} apf_range={a_start}..{a_end}"
        )
        if m_status != "ok":
            errors.append(f"{item['traj_id']}: metrics status {m_status}")
        elif m_start != int(args.range_start) or m_end != int(args.range_end):
            errors.append(f"{item['traj_id']}: metric range {m_start}..{m_end}, expected {args.range_start}..{args.range_end}")
        if n_chunks == 0:
            errors.append(f"{item['traj_id']}: no APF chunks")
        elif a_start is not None and a_end is not None and not (a_start <= int(args.range_start) and a_end >= int(args.range_end)):
            errors.append(f"{item['traj_id']}: APF range {a_start}..{a_end} does not cover {args.range_start}..{args.range_end}")

    if errors:
        print("\nFAIL:")
        for error in errors:
            print(" -", error)
        return 1
    print("\nOK: all expected C1 APF optimized/random items have APF logs and metrics over the requested range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
