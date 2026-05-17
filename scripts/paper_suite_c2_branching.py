from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import list_apf_chunks
from paper_suite_common import (
    REPO_ROOT,
    command_to_str,
    current_python,
    ensure_dir,
    load_config,
    read_csv,
    resolve_path,
    run_subprocess,
    sign_test_greater,
    write_csv,
    write_json,
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _output_root(cfg: Any) -> Path:
    return ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))


def _branch_cfg(cfg: Any) -> Any:
    return _get(cfg.get("c2", {}), "branching", {})


def _branch_root(cfg: Any, output_root: Path) -> Path:
    raw = _get(_branch_cfg(cfg), "branch_root", None)
    if raw is None:
        return ensure_dir(output_root / "c2_branching" / "branches")
    path = resolve_path(raw)
    return ensure_dir(path if path is not None else output_root / "c2_branching" / "branches")


def _iter_metric_paths(root: Path) -> list[Path]:
    manifest = root / "manifest.json"
    paths: list[Path] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for row in payload.get("trajectories", []):
            raw = row.get("metrics_path")
            if raw:
                path = Path(str(raw))
                if not path.is_absolute():
                    path = root / str(row.get("traj_id", "")) / path.name
                if path.exists():
                    paths.append(path)
            traj_id = row.get("traj_id")
            if traj_id:
                candidate = root / str(traj_id) / "metrics.npz"
                if candidate.exists() and candidate not in paths:
                    paths.append(candidate)
    if not paths:
        paths = sorted(root.glob("traj_*/metrics.npz"))
    return paths


def _safe_arr(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key])


def _load_delta_h(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dh = _safe_arr(data, "delta_h_best")
        if dh is None:
            dh_map = _safe_arr(data, "delta_h_map")
            if dh_map is None:
                raise ValueError(f"{path} has neither delta_h_best nor delta_h_map.")
            selected = int(np.asarray(_safe_arr(data, "delta_h_selected_tau_idx", np.asarray(0))).item())
            dh = np.asarray(dh_map[selected], dtype=np.float64)
        dh = np.asarray(dh, dtype=np.float64).reshape(-1)
        centers = _safe_arr(data, "delta_h_window_center_steps")
        if centers is None:
            starts = _safe_arr(data, "delta_h_window_start_steps", np.arange(dh.size))
            centers = np.asarray(starts, dtype=np.float64).reshape(-1)
        centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    n = min(dh.size, centers.size)
    if n == 0:
        raise ValueError(f"{path} has an empty delta-H series.")
    return centers[:n], dh[:n]


def _select_events(
    *,
    centers: np.ndarray,
    dh: np.ndarray,
    m_pairs: int,
    refractory_steps: int,
    high_quantile: float,
    low_quantile: float,
) -> list[dict[str, Any]]:
    finite = np.isfinite(dh) & np.isfinite(centers)
    if int(np.sum(finite)) < 2:
        return []
    c = centers[finite]
    h = dh[finite]
    high_thr = float(np.nanquantile(h, high_quantile))
    low_thr = float(np.nanquantile(h, low_quantile))
    high_order = np.argsort(-h)
    selected_high: list[int] = []
    for idx in high_order:
        if h[idx] < high_thr:
            continue
        step = float(c[idx])
        if any(abs(step - float(c[j])) < refractory_steps for j in selected_high):
            continue
        selected_high.append(int(idx))
        if len(selected_high) >= int(m_pairs):
            break

    low_pool = [int(i) for i in np.argsort(h) if h[int(i)] <= low_thr]
    used_low: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for pair_id, hi in enumerate(selected_high):
        if not low_pool:
            break
        candidates = [i for i in low_pool if i not in used_low]
        if not candidates:
            break
        lo = min(candidates, key=lambda i: abs(float(c[i]) - float(c[hi])))
        used_low.add(lo)
        pairs.append(
            {
                "pair_id": int(pair_id),
                "high_step": int(round(float(c[hi]))),
                "high_delta_h": float(h[hi]),
                "low_step": int(round(float(c[lo]))),
                "low_delta_h": float(h[lo]),
            }
        )
    return pairs


def _branch_output_ok(branch_dir: Path) -> bool:
    if (branch_dir / "branch_feature.npz").exists() or (branch_dir / "metrics.npz").exists():
        return True
    apf_dir = branch_dir / "apf_logs"
    return apf_dir.exists() and bool(list_apf_chunks(apf_dir))


def _make_resume_command(
    *,
    source_traj_dir: Path,
    step: int,
    horizon_steps: int,
    branch_dir: Path,
    branch_seed: int,
    perturb_a_std: float,
    perturb_p_std: float,
    perturb_lag_xy_std: float,
    force: bool,
) -> list[str]:
    cmd = [
        current_python(),
        "scripts/flowlenia_minibang_resume.py",
        str(source_traj_dir),
        "--step",
        str(int(step)),
        "--additional-steps",
        str(int(horizon_steps)),
        "--output-dir",
        str(branch_dir),
        "--branch-seed",
        str(int(branch_seed)),
        "--perturb-a-std",
        str(float(perturb_a_std)),
        "--perturb-p-std",
        str(float(perturb_p_std)),
        "--perturb-lagrangian-xy-std",
        str(float(perturb_lag_xy_std)),
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def _write_smoke_fixture(output_root: Path, branch_root: Path) -> Path:
    out_dir = ensure_dir(output_root / "c2_branching")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(123)
    for pair_id in range(2):
        for condition, step, dh, spread in (
            ("high", 100 + pair_id * 100, 1.0 + 0.2 * pair_id, 0.35),
            ("low", 150 + pair_id * 100, 0.1 + 0.05 * pair_id, 0.04),
        ):
            center = rng.normal(0.0, 0.05, size=8)
            for branch_id in range(3):
                branch_dir = ensure_dir(branch_root / "smoke_traj" / f"pair_{pair_id:03d}_{condition}" / f"branch_{branch_id:02d}")
                feature = center + rng.normal(0.0, spread, size=8)
                np.savez_compressed(branch_dir / "branch_feature.npz", feature=feature.astype(np.float32))
                rows.append(
                    {
                        "traj_id": "smoke_traj",
                        "pair_id": pair_id,
                        "condition": condition,
                        "step": step,
                        "delta_h": dh,
                        "branch_id": branch_id,
                        "source_metrics_path": "",
                        "source_traj_dir": "",
                        "branch_dir": str(branch_dir),
                        "status": "smoke_written",
                        "command": "",
                    }
                )
    path = out_dir / "branch_plan.csv"
    write_csv(path, rows)
    write_json(output_root / "c2_branching_simulation_summary.json", {"status": "ok", "n_branches": len(rows), "plan": str(path)})
    return path


def simulation(
    config_path: str | Path,
    *,
    smoke: bool = False,
    force: bool = False,
    allow_heavy: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    out_dir = ensure_dir(output_root / "c2_branching")
    bcfg = _branch_cfg(cfg)
    branch_root = _branch_root(cfg, output_root)
    if smoke:
        plan = _write_smoke_fixture(output_root, branch_root)
        return {"status": "ok", "n_branches": len(read_csv(plan)), "plan": str(plan)}

    c2_cfg = cfg.get("c2", {})
    minibang_root = resolve_path(_get(c2_cfg, "minibang_root", "experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set"))
    if minibang_root is None or not minibang_root.exists():
        summary = {"status": "skipped", "reason": f"missing minibang root {minibang_root}"}
        write_json(output_root / "c2_branching_simulation_summary.json", summary)
        return summary

    metric_paths = _iter_metric_paths(minibang_root)
    max_trajectories = int(_get(bcfg, "max_trajectories", 2))
    m_pairs = int(_get(bcfg, "m_pairs", 2))
    branches_per_time = int(_get(bcfg, "branches_per_time", 3))
    refractory_steps = int(_get(bcfg, "refractory_steps", 5000))
    high_quantile = float(_get(bcfg, "high_quantile", 0.85))
    low_quantile = float(_get(bcfg, "low_quantile", 0.35))
    horizon_steps = int(_get(bcfg, "horizon_steps", 1000))
    perturb = _get(bcfg, "perturb", {})
    perturb_a_std = float(_get(perturb, "a_std", 1e-4))
    perturb_p_std = float(_get(perturb, "p_std", 1e-4))
    perturb_lag_xy_std = float(_get(perturb, "lagrangian_xy_std", 0.01))

    ranked: list[tuple[float, Path, np.ndarray, np.ndarray]] = []
    for path in metric_paths:
        centers, dh = _load_delta_h(path)
        ranked.append((float(np.nanmax(dh)), path, centers, dh))
    ranked.sort(key=lambda x: -x[0])

    rows: list[dict[str, Any]] = []
    for _peak, metrics_path, centers, dh in ranked[:max_trajectories]:
        source_traj_dir = metrics_path.parent
        traj_id = source_traj_dir.name
        pairs = _select_events(
            centers=centers,
            dh=dh,
            m_pairs=m_pairs,
            refractory_steps=refractory_steps,
            high_quantile=high_quantile,
            low_quantile=low_quantile,
        )
        for pair in pairs:
            for condition in ("high", "low"):
                step = int(pair[f"{condition}_step"])
                delta_h = float(pair[f"{condition}_delta_h"])
                for branch_id in range(branches_per_time):
                    branch_seed = int(1000003 + 1009 * int(pair["pair_id"]) + 131 * branch_id + (0 if condition == "high" else 7919))
                    branch_dir = branch_root / traj_id / f"pair_{int(pair['pair_id']):03d}_{condition}_step_{step}" / f"branch_{branch_id:02d}"
                    cmd = _make_resume_command(
                        source_traj_dir=source_traj_dir,
                        step=step,
                        horizon_steps=horizon_steps,
                        branch_dir=branch_dir,
                        branch_seed=branch_seed,
                        perturb_a_std=perturb_a_std,
                        perturb_p_std=perturb_p_std,
                        perturb_lag_xy_std=perturb_lag_xy_std,
                        force=force,
                    )
                    if _branch_output_ok(branch_dir) and not force:
                        status = "exists"
                    elif not allow_heavy:
                        status = "skipped_heavy"
                    else:
                        run_subprocess(cmd, dry_run=dry_run)
                        status = "dry_run" if dry_run else ("exists" if _branch_output_ok(branch_dir) else "missing")
                    rows.append(
                        {
                            "traj_id": traj_id,
                            "pair_id": int(pair["pair_id"]),
                            "condition": condition,
                            "step": step,
                            "delta_h": delta_h,
                            "branch_id": branch_id,
                            "source_metrics_path": str(metrics_path),
                            "source_traj_dir": str(source_traj_dir),
                            "branch_dir": str(branch_dir),
                            "status": status,
                            "command": command_to_str(cmd),
                        }
                    )

    plan_path = out_dir / "branch_plan.csv"
    write_csv(plan_path, rows)
    summary = {
        "status": "ok",
        "n_branches": len(rows),
        "n_ready": sum(1 for row in rows if row["status"] == "exists"),
        "allow_heavy": bool(allow_heavy),
        "plan": str(plan_path),
    }
    write_json(output_root / "c2_branching_simulation_summary.json", summary)
    return summary


def _stats_feature(arr: np.ndarray) -> list[float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.quantile(finite, 0.10)),
        float(np.quantile(finite, 0.90)),
    ]


def _feature_from_metrics(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        parts: list[float] = []
        for key in ("delta_h_best", "cluster_tv"):
            arr = _safe_arr(data, key)
            if arr is not None:
                parts.extend(_stats_feature(arr))
        if not parts and "delta_h_map" in data.files:
            parts.extend(_stats_feature(np.asarray(data["delta_h_map"])))
    if not parts:
        raise ValueError(f"No metric feature arrays found in {path}")
    return np.asarray(parts, dtype=np.float64)


def _feature_from_apf(apf_dir: Path, *, max_chunks: int, max_snapshots_per_chunk: int) -> np.ndarray:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    selected = chunks[-max(1, int(max_chunks)) :]
    parts: list[float] = []
    for path, _s0, _s1, _idx in selected:
        with np.load(path, allow_pickle=False) as data:
            for key in ("A", "P", "F", "lagrangian_xy"):
                if key not in data.files:
                    continue
                arr = np.asarray(data[key])
                if arr.ndim > 0 and arr.shape[0] > max_snapshots_per_chunk:
                    idxs = np.linspace(0, arr.shape[0] - 1, max_snapshots_per_chunk).astype(int)
                    arr = arr[idxs]
                parts.extend(_stats_feature(arr))
    if not parts:
        raise ValueError(f"No usable APF arrays found in {apf_dir}")
    return np.asarray(parts, dtype=np.float64)


def _feature_from_branch(branch_dir: Path, *, max_chunks: int, max_snapshots_per_chunk: int) -> np.ndarray:
    feature_path = branch_dir / "branch_feature.npz"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as data:
            key = "feature" if "feature" in data.files else data.files[0]
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    metrics_path = branch_dir / "metrics.npz"
    if metrics_path.exists():
        return _feature_from_metrics(metrics_path)
    return _feature_from_apf(branch_dir / "apf_logs", max_chunks=max_chunks, max_snapshots_per_chunk=max_snapshots_per_chunk)


def _pairwise_mean_distance(features: list[np.ndarray]) -> float:
    if len(features) < 2:
        return float("nan")
    dists = []
    min_len = min(int(f.size) for f in features)
    for a, b in combinations(features, 2):
        aa = np.asarray(a[:min_len], dtype=np.float64)
        bb = np.asarray(b[:min_len], dtype=np.float64)
        dists.append(float(np.linalg.norm(aa - bb) / math.sqrt(max(1, min_len))))
    return float(np.mean(dists)) if dists else float("nan")


def metrics(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = _output_root(cfg)
    bcfg = _branch_cfg(cfg)
    branch_root = _branch_root(cfg, output_root)
    out_dir = ensure_dir(output_root / "c2_branching")
    plan_path = out_dir / "branch_plan.csv"
    if smoke and not plan_path.exists():
        _write_smoke_fixture(output_root, branch_root)
    if not plan_path.exists():
        summary = {"status": "skipped", "reason": f"missing branch plan {plan_path}"}
        write_json(output_root / "c2_branching_metrics_summary.json", summary)
        return summary

    max_chunks = int(_get(bcfg, "feature_max_apf_chunks", 4))
    max_snapshots = int(_get(bcfg, "feature_max_snapshots_per_chunk", 8))
    plan_rows = read_csv(plan_path)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in plan_rows:
        key = (str(row["traj_id"]), str(row["pair_id"]), str(row["condition"]))
        groups.setdefault(key, []).append(row)

    score_rows: list[dict[str, Any]] = []
    for (traj_id, pair_id, condition), rows in sorted(groups.items()):
        features = []
        used = []
        for row in rows:
            branch_dir = Path(str(row["branch_dir"]))
            try:
                features.append(_feature_from_branch(branch_dir, max_chunks=max_chunks, max_snapshots_per_chunk=max_snapshots))
                used.append(row)
            except Exception:
                continue
        if len(features) < 2:
            continue
        delta_h_vals = [float(row.get("delta_h", "nan")) for row in used]
        step_vals = [float(row.get("step", "nan")) for row in used]
        score_rows.append(
            {
                "traj_id": traj_id,
                "pair_id": int(float(pair_id)),
                "condition": condition,
                "step": float(np.nanmedian(step_vals)),
                "delta_h": float(np.nanmedian(delta_h_vals)),
                "branching_score": _pairwise_mean_distance(features),
                "n_branches": len(features),
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in score_rows:
        key = (str(row["traj_id"]), int(row["pair_id"]))
        by_pair.setdefault(key, {})[str(row["condition"])] = row
    for (traj_id, pair_id), conds in sorted(by_pair.items()):
        if "high" not in conds or "low" not in conds:
            continue
        high = conds["high"]
        low = conds["low"]
        contrast_rows.append(
            {
                "traj_id": traj_id,
                "pair_id": pair_id,
                "high_step": high["step"],
                "low_step": low["step"],
                "high_delta_h": high["delta_h"],
                "low_delta_h": low["delta_h"],
                "high_branching_score": high["branching_score"],
                "low_branching_score": low["branching_score"],
                "delta_branching_score": float(high["branching_score"]) - float(low["branching_score"]),
            }
        )

    scores_path = out_dir / "branching_scores.csv"
    contrasts_path = out_dir / "branching_pair_contrasts.csv"
    write_csv(scores_path, score_rows)
    write_csv(contrasts_path, contrast_rows)
    summary = {
        "status": "ok",
        "n_scores": len(score_rows),
        "n_pairs": len(contrast_rows),
        "branching_sign_test": sign_test_greater(row["delta_branching_score"] for row in contrast_rows),
        "scores": str(scores_path),
        "contrasts": str(contrasts_path),
    }
    write_json(output_root / "c2_branching_metrics_summary.json", summary)
    return summary


def run(
    config_path: str | Path,
    *,
    layer: str = "all",
    smoke: bool = False,
    force: bool = False,
    allow_heavy: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if layer in {"simulation", "all"}:
        out["simulation"] = simulation(config_path, smoke=smoke, force=force, allow_heavy=allow_heavy, dry_run=dry_run)
    if layer in {"metrics", "all"}:
        out["metrics"] = metrics(config_path, smoke=smoke)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2 branching-sensitivity experiment for the paper suite.")
    parser.add_argument("config")
    parser.add_argument("--layer", choices=["simulation", "metrics", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-heavy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, layer=args.layer, smoke=args.smoke, force=args.force, allow_heavy=args.allow_heavy, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
