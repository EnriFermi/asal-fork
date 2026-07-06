from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from paper_suite_common import load_config, resolve_path, write_json


DEFAULT_CONFIG = (
    "experiments/paper_suite/"
    "config_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random.yaml"
)
DEFAULT_SELECTED_ROOT = (
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/optimization"
)
DEFAULT_SCORES_CSV = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
    "flow_lenia/checkpoint_scores.csv"
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _flat_optimization_config(path: Path) -> Any:
    cfg = OmegaConf.load(path)
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )


def _make_substrate(args: Any):
    import substrates
    import util

    base = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    if hasattr(base, "debug_return_F"):
        base.debug_return_F = True
    return substrates.FlattenSubstrateParameters(base)


def _optimization_lagrangian_xy(
    *,
    opt_flat: Any,
    params: np.ndarray,
    run_seed: int,
    rollout_steps: int,
    sample_every_steps: int,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from flowlenia_minibang_simulate import _init_lagrangian_points_jax

    args = OmegaConf.create(OmegaConf.to_container(opt_flat, resolve=True))
    args.rollout_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    substrate = _make_substrate(args)
    params_j = jnp.asarray(np.asarray(params, dtype=np.float32))
    rng_roll, _rng_metric = jax.random.split(jax.random.PRNGKey(int(run_seed)), 2)
    k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
    s0 = substrate.init_state(k_state, params_j)
    if "F" not in s0:
        raise ValueError("Optimization-style replay requires Flow-Lenia state with F.")
    rt = substrate.RT

    lag_n = int(_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(_get(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(_get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(_get(args, "metric_lagrangian_channel_mode", "resample"))
    lag_noise_model = str(_get(args, "metric_lagrangian_noise_model", "rt_box"))
    lag_diffusion_scale = float(_get(args, "metric_lagrangian_diffusion_scale", 1.0))

    pts0 = _init_lagrangian_points_jax(
        s0["A"],
        n_particles=lag_n,
        init_mode=lag_init_mode,
        border=str(getattr(rt, "border", "wall")),
        sigma=float(getattr(rt, "sigma", 0.0)),
        key=k_pts,
    )
    if lag_channel_mode in ("fixed", "resample"):
        ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
    else:
        ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

    def step_fn(state, key_step):
        st, pts, ch = state
        st = substrate.step_state(key_step, st, params_j)
        lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
        pts, ch = rt.advect_particles(
            points=pts,
            F=st["F"],
            A=st["A"],
            channel=lag_flow_channel,
            reduce=lag_flow_reduce,
            point_channels=ch,
            channel_mode=lag_channel_mode,
            key=lag_key,
            noise_model=lag_noise_model,
            diffusion_scale=lag_diffusion_scale,
        )
        return (st, pts, ch), None

    def chunk_fn(state, key_chunk):
        state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
        return state_next, state_next[1]

    time_sampling = int(rollout_steps) // int(sample_every_steps)
    (_, _, _), xy_seq = jax.lax.scan(
        chunk_fn,
        (s0, pts0, ch0),
        jax.random.split(k_scan, time_sampling),
    )
    return np.asarray(jax.device_get(xy_seq), dtype=np.float32)


def _resolve_section(cfg: Any) -> Any:
    section = _get(_get(cfg, "simulation", {}), "flow_lenia_arun_lagrangian_apf", None)
    if section is None:
        raise ValueError("Missing simulation.flow_lenia_arun_lagrangian_apf in generated config.")
    return section


def _audit_generated_config(config_path: Path) -> dict[str, Any]:
    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    c1 = _get(_get(_get(cfg, "datasets", {}), "flow_lenia", {}), "c1", {})
    run_seed_protocol = str(_get(section, "run_seed_protocol", ""))
    metric_seed_protocol = str(_get(c1, "metric_seed_protocol", ""))
    out = {
        "config": str(config_path),
        "run_seed_protocol": run_seed_protocol,
        "metric_seed_protocol": metric_seed_protocol,
        "apf_root": str(_get(section, "output_root", "")),
        "rollout_config": str(_get(section, "rollout_config", "")),
        "run_seed_base": int(_get(section, "run_seed_base", -1)),
        "run_seed_mode": str(_get(section, "run_seed_mode", "")),
        "run_seed_rep_stride": int(_get(section, "run_seed_rep_stride", -1)),
        "n_rollout_seeds_per_checkpoint": int(_get(section, "n_rollout_seeds_per_checkpoint", -1)),
    }
    errors = []
    if run_seed_protocol != "optimization_metric":
        errors.append(f"run_seed_protocol={run_seed_protocol!r}, expected 'optimization_metric'")
    if metric_seed_protocol != "optimization_metric":
        errors.append(f"metric_seed_protocol={metric_seed_protocol!r}, expected 'optimization_metric'")
    if out["run_seed_mode"] != "source_run_idx":
        errors.append(f"run_seed_mode={out['run_seed_mode']!r}, expected 'source_run_idx'")
    if out["run_seed_rep_stride"] != 1:
        errors.append(f"run_seed_rep_stride={out['run_seed_rep_stride']!r}, expected 1")
    out["errors"] = errors
    return out


def _optimization_eval_seed_count(run_dir: Path) -> int | None:
    cfg_path = Path(run_dir) / "optimization_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    opt = cfg.get("optimization", {})
    algorithm = str(opt.get("optimizer_algorithm", opt.get("optimization_algorithm", "cma_es"))).strip().lower()
    if algorithm in {
        "mirrored_openai_es",
        "mirrored_batch_openai_es",
        "openai_es",
        "batch_openai_es",
        "mirrored_es",
        "antithetic_openai_es",
    }:
        raw = opt.get("openai_es_n_seeds", None)
    else:
        raw = opt.get("bs", None)
    return None if raw is None else int(raw)


def _audit_seed_count(config_audit: dict[str, Any], selected_root: Path) -> dict[str, Any]:
    requested = int(config_audit.get("n_rollout_seeds_per_checkpoint", -1))
    rows = []
    errors = []
    for run_dir in sorted(Path(selected_root).glob("run_*")):
        if not run_dir.is_dir():
            continue
        expected = _optimization_eval_seed_count(run_dir)
        row = {"run": run_dir.name, "optimization_eval_seed_count": expected, "c1_rollout_seed_count": requested}
        rows.append(row)
        if expected is None:
            errors.append(f"{run_dir}: missing optimization eval seed count")
        elif int(expected) != requested:
            errors.append(f"{run_dir.name}: optimization eval seed count={expected}, C1 rollout seed count={requested}")
    if not rows:
        errors.append(f"no run_* directories found under {selected_root}")
    return {"status": "ok" if not errors else "failed", "rows": rows, "errors": errors}


def _find_run(selected_root: Path, requested: str | None) -> tuple[int, Path]:
    if requested is not None:
        text = str(requested)
        run_idx = int(text[4:] if text.startswith("run_") else text)
        run_dir = selected_root / f"run_{run_idx:03d}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Selected run not found: {run_dir}")
        return run_idx, run_dir
    runs = sorted(p for p in selected_root.glob("run_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No selected run directories found under {selected_root}")
    run_dir = runs[0]
    return int(run_dir.name.split("_", 1)[1]), run_dir


def _run_replay_smoke(
    *,
    config_path: Path,
    selected_root: Path,
    run: str | None,
    seed_idx: int,
    rollout_steps: int,
    output_root: Path,
    atol: float,
) -> dict[str, Any]:
    from flowlenia_minibang_common import load_config as load_rollout_config
    from flowlenia_minibang_simulate import _load_lagrangian_series, simulate_batch

    cfg, _flat = load_config(config_path)
    section = _resolve_section(cfg)
    pair_seed_base = int(_get(section, "run_seed_base", 400003))
    run_idx, run_dir = _find_run(selected_root, run)
    run_seed = pair_seed_base + 2 * int(run_idx) + int(seed_idx)

    best = _load_pickle(run_dir / "best.pkl")
    if isinstance(best, tuple) and len(best) == 2:
        params, loss = best
    else:
        params, loss = best, float("nan")
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    opt_cfg_path = run_dir / "optimization_config.yaml"
    if not opt_cfg_path.exists():
        raise FileNotFoundError(f"Missing copied optimization_config.yaml: {opt_cfg_path}")
    opt_flat = _flat_optimization_config(opt_cfg_path)

    rollout_config = resolve_path(_get(section, "rollout_config", None))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"rollout_config not found: {rollout_config}")
    rollout_cfg, rollout_flat = load_rollout_config(rollout_config, [])
    sample_every = int(_get(rollout_flat, "sample_every_steps", _get(rollout_flat, "snapshot_interval", 50)))
    snapshot_interval = int(_get(rollout_flat, "snapshot_interval", sample_every))
    if sample_every != snapshot_interval:
        raise ValueError(f"sample_every_steps={sample_every}, snapshot_interval={snapshot_interval}; expected equality.")
    if int(rollout_steps) % sample_every != 0:
        raise ValueError(f"rollout_steps={rollout_steps} must be divisible by sample_every={sample_every}.")

    flat_dict = OmegaConf.to_container(rollout_flat, resolve=True)
    flat_dict.update(
        {
            "rollout_steps": int(rollout_steps),
            "max_steps": int(rollout_steps),
            "snapshot_interval": sample_every,
            "snapshots_per_file": max(1, int(rollout_steps) // sample_every + 1),
            "batch_size": 1,
            "n_trajectories": 1,
            "run_seed_protocol": "optimization_metric",
            "img_size": 64,
            "video_img_size": 64,
            "save_rgb": False,
            "compute_metrics": False,
            "compute_delta_h": False,
            "compute_clusters": False,
        }
    )
    if int(flat_dict.get("jit_microbatch", sample_every)) < sample_every:
        flat_dict["jit_microbatch"] = sample_every
    rollout_cfg_i = OmegaConf.create(OmegaConf.to_container(rollout_cfg, resolve=True))
    if rollout_cfg_i.get("minibang", None) is None:
        rollout_cfg_i.minibang = OmegaConf.create()
    rollout_cfg_i.minibang.run_seed_protocol = "optimization_metric"
    rollout_cfg_i.rollout.substrate.rollout_steps = int(rollout_steps)
    rollout_cfg_i.rollout.simulation.rollout_steps = int(rollout_steps)
    rollout_cfg_i.rollout.simulation.max_steps = int(rollout_steps)

    smoke_root = output_root / f"run_{run_idx:03d}_seed_{seed_idx:03d}_{rollout_steps}"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    selected_batch = [
        {
            "traj_id": f"run_{run_idx:03d}_seed_{seed_idx:03d}",
            "selection_idx": 0,
            "source_run_idx": int(run_idx),
            "run_seed": int(run_seed),
            "params": params,
            "loss": float(np.asarray(loss).reshape(-1)[0]) if np.asarray(loss).size else float("nan"),
        }
    ]
    simulate_batch(
        selected_batch=selected_batch,
        cfg=rollout_cfg_i,
        flat_args=flat_dict,
        output_root=smoke_root,
        overwrite=True,
    )
    apf_steps, apf_xy_all = _load_lagrangian_series(smoke_root / selected_batch[0]["traj_id"] / "apf_logs")
    apf_mask = np.asarray(apf_steps) > 0
    apf_xy = np.asarray(apf_xy_all[apf_mask], dtype=np.float32)
    opt_xy = _optimization_lagrangian_xy(
        opt_flat=opt_flat,
        params=params,
        run_seed=run_seed,
        rollout_steps=int(rollout_steps),
        sample_every_steps=sample_every,
    )
    if apf_xy.shape != opt_xy.shape:
        raise ValueError(f"APF/optimization xy shape mismatch: apf={apf_xy.shape}, opt={opt_xy.shape}")
    diff = np.asarray(apf_xy, dtype=np.float32) - np.asarray(opt_xy, dtype=np.float32)
    max_abs = float(np.nanmax(np.abs(diff)))
    mean_abs = float(np.nanmean(np.abs(diff)))
    ok = bool(max_abs <= float(atol))
    return {
        "status": "ok" if ok else "failed",
        "run_idx": int(run_idx),
        "seed_idx": int(seed_idx),
        "run_seed": int(run_seed),
        "rollout_steps": int(rollout_steps),
        "sample_every_steps": int(sample_every),
        "n_samples_compared": int(opt_xy.shape[0]),
        "xy_shape": list(opt_xy.shape),
        "max_abs_xy_diff": max_abs,
        "mean_abs_xy_diff": mean_abs,
        "atol": float(atol),
        "smoke_output_root": str(smoke_root),
    }


def _audit_existing_results(scores_csv: Path) -> dict[str, Any]:
    if not scores_csv.exists():
        return {"scores_csv": str(scores_csv), "status": "missing", "errors": [f"missing {scores_csv}"]}
    import pandas as pd

    df = pd.read_csv(scores_csv)
    errors: list[str] = []
    if "metric_seed_protocol" not in df.columns:
        errors.append("checkpoint_scores.csv has no metric_seed_protocol column; metrics are stale.")
    else:
        vals = sorted(set(str(x) for x in df["metric_seed_protocol"].dropna().unique()))
        if vals != ["optimization_metric"]:
            errors.append(f"metric_seed_protocol values are {vals}, expected ['optimization_metric']")
    opt = df[df.get("candidate_kind", "") == "optimized"].copy() if "candidate_kind" in df.columns else df.iloc[0:0]
    if opt.empty:
        errors.append("no optimized rows found in checkpoint_scores.csv")
    for col in ("train_tau_steps", "full_score_train_tau_mspd", "eval_score_mspd"):
        if col not in df.columns:
            errors.append(f"checkpoint_scores.csv has no {col} column")
    if "train_tau_steps" in opt.columns:
        missing_tau = int(opt["train_tau_steps"].isna().sum())
        if missing_tau:
            errors.append(f"{missing_tau} optimized rows have missing train_tau_steps")
    return {
        "scores_csv": str(scores_csv),
        "status": "ok" if not errors else "failed",
        "n_rows": int(len(df)),
        "n_optimized_rows": int(len(opt)),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast preflight for Flow-Lenia fixed-init C1 replay protocol.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--selected-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--scores-csv", default=DEFAULT_SCORES_CSV)
    parser.add_argument("--run", default=None, help="Run index or run_XXX. Defaults to first selected run.")
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--output-root", default="/tmp/asal_flowlenia_c1_replay_preflight")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-existing-results", action="store_true")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    selected_root = Path(args.selected_root)
    scores_csv = Path(args.scores_csv)
    summary: dict[str, Any] = {
        "config_audit": _audit_generated_config(config_path),
    }
    summary["seed_count_audit"] = _audit_seed_count(summary["config_audit"], selected_root)
    if not args.skip_existing_results:
        summary["existing_results_audit"] = _audit_existing_results(scores_csv)
    if not args.skip_smoke:
        summary["replay_smoke"] = _run_replay_smoke(
            config_path=config_path,
            selected_root=selected_root,
            run=args.run,
            seed_idx=int(args.seed_idx),
            rollout_steps=int(args.rollout_steps),
            output_root=Path(args.output_root),
            atol=float(args.atol),
        )

    errors: list[str] = []
    errors.extend(summary["config_audit"].get("errors", []))
    errors.extend(summary["seed_count_audit"].get("errors", []))
    if "existing_results_audit" in summary:
        errors.extend(summary["existing_results_audit"].get("errors", []))
    if "replay_smoke" in summary and summary["replay_smoke"].get("status") != "ok":
        errors.append(
            "replay_smoke failed: max_abs_xy_diff="
            f"{summary['replay_smoke'].get('max_abs_xy_diff')}"
        )
    summary["status"] = "ok" if not errors else "failed"
    summary["errors"] = errors

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
