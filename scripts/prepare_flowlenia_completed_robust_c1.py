from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from export_flowlenia_openai_es_robust_pioneer import run as export_robust_pioneer


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else _repo_root() / path


def _rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_repo_root().resolve()))
    except Exception:
        return str(path)


def _run_idx(path: Path) -> int | None:
    name = Path(path).name
    if not name.startswith("run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _get_nested(cfg: Any, path: tuple[str, ...], default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if cur is None:
            return default
        try:
            cur = cur.get(key, default)
        except Exception:
            cur = getattr(cur, key, default)
    return cur


def _configured_n_iters(run_dir: Path) -> int | None:
    cfg_path = run_dir / "optimization_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    raw = _get_nested(cfg, ("optimization", "n_iters"), None)
    if raw is None:
        raw = _get_nested(cfg, ("n_iters",), None)
    return None if raw is None else int(raw)


def _pop_n_iters(run_dir: Path) -> int | None:
    path = run_dir / "pop_traj.pkl"
    if not path.exists():
        return None
    pop = _load_pickle(path)
    params = pop.get("params") if isinstance(pop, dict) else None
    if params is None:
        return None
    return int(params.shape[0])


def _discover_completed_runs(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(source_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        idx = _run_idx(run_dir)
        if idx is None:
            continue
        n_expected = _configured_n_iters(run_dir)
        n_done = _pop_n_iters(run_dir)
        completed = bool(n_expected is not None and n_done is not None and n_done >= n_expected)
        rows.append(
            {
                "run_idx": int(idx),
                "run_dir": run_dir,
                "n_iters_expected": n_expected,
                "n_iters_done": n_done,
                "completed": completed,
                "has_best": (run_dir / "best.pkl").exists(),
                "has_pop_traj": (run_dir / "pop_traj.pkl").exists(),
            }
        )
    return [row for row in rows if row["completed"]]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_c1_config(
    *,
    output_config: Path,
    result_root: str,
    apf_root: str,
    rollout_config: str,
    selected_dirs: list[str],
    random_checkpoint_root: str,
    n_rollout_seeds: int,
    num_random_baselines: int,
    batch_size: int,
    pair_seed_base: int,
) -> None:
    cfg = {
        "meta": {
            "output_root": result_root,
            "conda_env": "torchjax",
            "wandb_mode": "disabled",
        },
        "datasets": {
            "flow_lenia": {
                "enabled": True,
                "required": True,
                "c1": {
                    "source": "apf_lagrangian_split",
                    "apf_root": apf_root,
                    "require_random": True,
                    "expected_window_start_steps": 50000,
                    "expected_window_end_steps": 300000,
                    "metric": {
                        "metric_tau_mode": "max_grid",
                        "metric_tau_grid_steps": [
                            1000,
                            2000,
                            3000,
                            4000,
                            5000,
                            6000,
                            7000,
                            8000,
                            9000,
                            10000,
                        ],
                        "metric_tau_steps": 3000,
                        "metric_window_size_steps": 20000,
                        "metric_window_step_steps": 5000,
                        "metric_range_start_steps": 0,
                        "metric_m_samples": 48,
                        "metric_m_min": 4,
                        "metric_n_proj": 16,
                        "metric_null_reps": 6,
                        "metric_particle_samples": 64,
                        "metric_periodic": False,
                        "metric_domain_y": 128.0,
                        "metric_domain_x": 128.0,
                        "metric_delta_h_floor": 0.0,
                        "metric_msc_floor": 0.01,
                        "metric_msc_term": "floor_reconstruction_error",
                        "metric_msc_normalize_by_weight_sum": True,
                    },
                },
            },
            "plife_plus": {"enabled": False, "required": False},
            "boids": {"enabled": False, "required": False},
        },
        "simulation": {
            "reuse_existing": True,
            "allow_optimization_rerun": False,
            "flow_lenia_arun_lagrangian_apf": {
                "enabled": True,
                "required": True,
                "rollout_config": rollout_config,
                "output_root": apf_root,
                "rollout_steps": 300000,
                "n_trajectories_per_checkpoint": 1,
                "n_rollout_seeds_per_checkpoint": int(n_rollout_seeds),
                "run_seed_rep_stride": 1,
                "include_random_baselines": True,
                "num_random_baselines": int(num_random_baselines),
                "random_checkpoint_root": random_checkpoint_root,
                "random_checkpoint_selection": "all_groups_flat",
                "run_seed_base": int(pair_seed_base),
                "run_seed_mode": "source_run_idx",
                "batch_size": int(batch_size),
                "optimized_checkpoint_dirs": selected_dirs,
                "dedupe_by_run_idx": False,
                "max_checkpoints": None,
            },
        },
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(output_config), resolve=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare robust-pioneer C1 config for completed Flow-Lenia OpenAI-ES runs."
    )
    parser.add_argument("--source-optimization-root", required=True)
    parser.add_argument("--selected-optimization-root", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--apf-root", required=True)
    parser.add_argument("--random-checkpoint-root", required=True)
    parser.add_argument("--rollout-config", default="experiments/paper_suite/flowlenia_arun_apf_300k_train50_grid128.yaml")
    parser.add_argument("--n-rollout-seeds", type=int, default=4)
    parser.add_argument("--num-random-baselines", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--pair-seed-base", type=int, default=400003)
    parser.add_argument("--lcb-z", type=float, default=2.0)
    parser.add_argument("--trend-quantile", type=float, default=90.0)
    parser.add_argument("--ewma-beta", type=float, default=0.85)
    parser.add_argument("--trim-frac", type=float, default=0.125)
    parser.add_argument("--min-iter", type=int, default=0)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    source_root = _resolve(args.source_optimization_root)
    selected_root = _resolve(args.selected_optimization_root)
    output_config = _resolve(args.output_config)
    result_root = str(args.result_root)
    apf_root = str(args.apf_root)
    random_checkpoint_root = str(args.random_checkpoint_root)
    rollout_config = str(args.rollout_config)

    if not source_root.exists():
        raise FileNotFoundError(f"source optimization root not found: {source_root}")
    completed = _discover_completed_runs(source_root)
    if not completed:
        raise RuntimeError(f"No completed optimization runs found under {source_root}.")

    selected_dirs: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for row in completed:
        run_idx = int(row["run_idx"])
        out_dir = selected_root / f"run_{run_idx:03d}"
        export_args = argparse.Namespace(
            run_dir=str(row["run_dir"]),
            output_dir=str(out_dir),
            lcb_z=float(args.lcb_z),
            trend_quantile=float(args.trend_quantile),
            ewma_beta=float(args.ewma_beta),
            trim_frac=float(args.trim_frac),
            min_iter=int(args.min_iter),
            force=bool(args.force_export),
        )
        meta = export_robust_pioneer(export_args)
        selected_dirs.append(_rel(out_dir))
        selected_rows.append(
            {
                "run_idx": run_idx,
                "source_run_dir": _rel(Path(row["run_dir"])),
                "selected_checkpoint_dir": _rel(out_dir),
                "n_iters_expected": row["n_iters_expected"],
                "n_iters_done": row["n_iters_done"],
                "selected_iter": meta.get("iter"),
                "selected_pop_idx": meta.get("pop_idx"),
                "selected_score_mspd": meta.get("score_mspd"),
                "selected_lcb_mspd": meta.get("seed_lcb_mspd"),
                "selected_tau_steps": (meta.get("tau") or {}).get("tau_steps"),
            }
        )

    selected_dirs = sorted(selected_dirs)
    _write_c1_config(
        output_config=output_config,
        result_root=result_root,
        apf_root=apf_root,
        rollout_config=rollout_config,
        selected_dirs=selected_dirs,
        random_checkpoint_root=random_checkpoint_root,
        n_rollout_seeds=int(args.n_rollout_seeds),
        num_random_baselines=int(args.num_random_baselines),
        batch_size=int(args.batch_size),
        pair_seed_base=int(args.pair_seed_base),
    )

    manifest_dir = output_config.parent / "generated_manifests" / output_config.stem
    _write_csv(manifest_dir / "completed_runs.csv", [{**row, "run_dir": _rel(Path(row["run_dir"]))} for row in completed])
    _write_csv(manifest_dir / "selected_robust_candidates.csv", selected_rows)
    summary = {
        "source_optimization_root": _rel(source_root),
        "selected_optimization_root": _rel(selected_root),
        "output_config": _rel(output_config),
        "result_root": result_root,
        "apf_root": apf_root,
        "random_checkpoint_root": random_checkpoint_root,
        "n_completed_runs": len(completed),
        "completed_run_indices": [int(row["run_idx"]) for row in completed],
        "n_rollout_seeds": int(args.n_rollout_seeds),
        "num_random_baselines": int(args.num_random_baselines),
        "anti_noise_selection": {
            "rule": "robust_pioneer_lcb_in_top_trend",
            "lcb_z": float(args.lcb_z),
            "trend_quantile": float(args.trend_quantile),
            "ewma_beta": float(args.ewma_beta),
            "trim_frac": float(args.trim_frac),
            "min_iter": int(args.min_iter),
        },
        "selected_candidates_csv": _rel(manifest_dir / "selected_robust_candidates.csv"),
        "completed_runs_csv": _rel(manifest_dir / "completed_runs.csv"),
    }
    _write_json(manifest_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
