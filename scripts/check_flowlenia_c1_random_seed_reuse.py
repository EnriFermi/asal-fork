from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "manifest.json"
    if path.name == "manifest.json":
        payload = json.loads(path.read_text())
        return list(payload.get("trajectories", []))
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Expected manifest.json, manifest root, or checkpoint_scores.csv, got {path}.")


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _kind(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("candidate_kind", "candidate_label", "traj_id", "trial_uid")
    ).lower()
    if "random" in text:
        return "random"
    if "optimized" in text or "flow_opt" in text:
        return "optimized"
    return str(row.get("candidate_kind", "other")).lower()


def _run_idx(row: dict[str, Any]) -> int | None:
    for key in ("source_optimized_run_idx", "optimized_run_idx", "source_run_idx", "suite_run_idx"):
        value = _as_int(row.get(key), None)
        if value is not None and value >= 0:
            return value
    return None


def _candidate_idx(row: dict[str, Any]) -> int | None:
    return _as_int(row.get("candidate_idx"), None)


def _candidate_key(row: dict[str, Any]) -> str:
    idx = _candidate_idx(row)
    if idx is not None:
        return f"idx:{idx}"
    label = str(row.get("candidate_label", "")).strip()
    if label:
        return f"label:{label}"
    traj_id = str(row.get("traj_id", row.get("trial_uid", ""))).strip()
    if "_seed_" in traj_id:
        traj_id = traj_id.rsplit("_seed_", 1)[0]
    return f"traj:{traj_id}"


def _parse_run_indices(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether existing Flow-Lenia C1 random rows can be reused for fixed-init evaluation seeds."
    )
    parser.add_argument("path", help="APF root containing manifest.json, manifest.json path, or checkpoint_scores.csv.")
    parser.add_argument("--pair-seed-base", type=int, default=400003)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--run-indices", default=None, help="Comma-separated run indices to require, e.g. 0,1,2,3,4,5,6,7,8.")
    parser.add_argument("--expected-random-candidates", type=int, default=None)
    args = parser.parse_args()

    rows = _load_rows(Path(args.path))
    random_rows = [row for row in rows if _kind(row) == "random"]
    by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in random_rows:
        run_idx = _run_idx(row)
        if run_idx is not None:
            by_run[int(run_idx)].append(row)

    required_runs = _parse_run_indices(args.run_indices)
    if required_runs is None:
        required_runs = sorted(by_run)

    print(f"rows_total={len(rows)} random_rows={len(random_rows)} runs_present={sorted(by_run)}")
    ok_all = True
    for run_idx in required_runs:
        run_rows = by_run.get(int(run_idx), [])
        candidate_keys = sorted({_candidate_key(row) for row in run_rows})
        expected_by_seed = {
            seed_idx: int(args.pair_seed_base + 2 * int(run_idx) + seed_idx)
            for seed_idx in range(int(args.n_seeds))
        }
        missing = []
        counts_by_seed = {}
        for seed_idx, expected_seed in expected_by_seed.items():
            matches = [
                row
                for row in run_rows
                if _as_int(row.get("rollout_seed_idx"), None) == seed_idx
                and _as_int(row.get("run_seed"), None) == expected_seed
            ]
            counts_by_seed[seed_idx] = len(matches)
            if not matches:
                missing.append((seed_idx, expected_seed))
        candidate_ok = (
            args.expected_random_candidates is None
            or len(candidate_keys) == int(args.expected_random_candidates)
        )
        run_ok = bool(run_rows) and not missing and candidate_ok
        ok_all = ok_all and run_ok
        status = "OK" if run_ok else "BAD"
        print(
            f"run_{run_idx:03d}: {status} rows={len(run_rows)} "
            f"random_candidates={len(candidate_keys)} counts_by_seed={counts_by_seed}"
        )
        if not candidate_ok:
            print(
                f"  expected_random_candidates={args.expected_random_candidates}, "
                f"found={len(candidate_keys)}"
            )
        if missing:
            print("  missing seed rows:")
            for seed_idx, expected_seed in missing:
                print(f"    rollout_seed_idx={seed_idx} run_seed={expected_seed}")

    if not ok_all:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
