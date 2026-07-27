from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import split

import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config
from flowlenia_exact_optimizer_rescore_one import (
    _init_lagrangian_points_jax,
    _load_flat_config,
    _load_pickle,
    _params_full_batch,
    _path_from_repo,
)


def _float_scalar(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64).reshape(()))


def _int_scalar(x: Any) -> int:
    return int(np.asarray(x).reshape(()))


def _jsonable_key(key: Any) -> list[int]:
    return [int(v) for v in np.asarray(key, dtype=np.uint32).reshape(-1)]


def _build_map_eval(args: Any):
    base_substrate = substrates.create_substrate(
        args.substrate,
        **util.substrate_kwargs_from_args(args),
    )
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    metric_space_defaults = util.metric_periodic_space_defaults(base_substrate)
    if (not hasattr(args, "metric_periodic")) or (getattr(args, "metric_periodic", None) is None):
        args.metric_periodic = bool(metric_space_defaults["periodic"])
    if (not hasattr(args, "metric_domain_y")) or (getattr(args, "metric_domain_y", None) is None):
        args.metric_domain_y = float(metric_space_defaults["domain_y"])
    if (not hasattr(args, "metric_domain_x")) or (getattr(args, "metric_domain_x", None) is None):
        args.metric_domain_x = float(metric_space_defaults["domain_x"])

    metric_cfg = resolve_metric_config(args)
    metric_loss_fn = make_metric_loss_fn(metric_cfg, include_maps=True)
    optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
    substrate_param_dims = int(substrate.n_params)
    log_clip_evolution = bool(getattr(args, "log_clip_evolution", True))
    if str(getattr(args, "metric_trajectory_source", "lagrangian")).strip().lower() != "lagrangian":
        raise ValueError("This diagnostic supports metric_trajectory_source='lagrangian' only.")

    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(args, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(args, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(args, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(args, "metric_lagrangian_diffusion_scale", 1.0))

    def split_candidate_params(params_full):
        params_sub = params_full[:substrate_param_dims]
        tau_selector = params_full[substrate_param_dims] if optimize_tau else None
        return params_sub, tau_selector

    def eval_key_parts(eval_key):
        if log_clip_evolution:
            rng_roll, rng_metric, _rng_clip = split(eval_key, 3)
        else:
            rng_roll, rng_metric = split(eval_key)
        k_state, k_pts, k_ch, k_scan = split(rng_roll, 4)
        return k_state, k_pts, k_ch, k_scan, rng_metric

    def rollout_from_parts(k_state, k_pts, k_ch, k_scan, params):
        s0 = substrate.init_state(k_state, params)
        if "F" not in s0:
            raise ValueError("Flow-Lenia state has no F; debug_return_F must be enabled.")
        rt = substrate.RT
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n_particles,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)

        def step_fn(state, key_step):
            st, pts, ch = state
            st = substrate.step_state(key_step, st, params)
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
            state_next, _ = jax.lax.scan(step_fn, state, split(key_chunk, chunk_steps))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            split(k_scan, time_sampling),
        )
        return xy_seq

    def score_maps_from_parts(parts, params_full):
        k_state, k_pts, k_ch, k_scan, rng_metric = parts
        params, tau_selector = split_candidate_params(params_full)
        xy_seq = rollout_from_parts(k_state, k_pts, k_ch, k_scan, params)
        if optimize_tau:
            loss, info = metric_loss_fn(rng_metric, xy_seq, tau_selector=tau_selector)
        else:
            loss, info = metric_loss_fn(rng_metric, xy_seq)
        return -loss, info

    score_maps_vv = jax.vmap(
        jax.vmap(score_maps_from_parts, in_axes=(0, None)),
        in_axes=(None, 0),
    )

    @jax.jit
    def score_maps_batch_from_parts(params_full_batch, parts_batch):
        return score_maps_vv(parts_batch, params_full_batch)

    @jax.jit
    def initial_ap_diff(k_state_a, k_state_b, params_full):
        params, _tau_selector = split_candidate_params(params_full)
        sa = substrate.init_state(k_state_a, params)
        sb = substrate.init_state(k_state_b, params)
        return (
            jnp.max(jnp.abs(sa["A"] - sb["A"])),
            jnp.max(jnp.abs(sa["P"] - sb["P"])),
        )

    return score_maps_batch_from_parts, initial_ap_diff, eval_key_parts, metric_cfg, substrate_param_dims, optimize_tau


def _info_small(info: dict[str, Any]) -> dict[str, float]:
    keys = (
        "score",
        "msc",
        "amp",
        "delta_h_mean",
        "delta_h_std",
        "delta_h_min",
        "delta_h_max",
        "h_real_mean",
        "h_null_mean",
        "tau_selected_idx",
        "tau_best_steps",
        "score_tau_max",
        "score_tau_mean",
        "score_tau_min",
    )
    out: dict[str, float] = {}
    for key in keys:
        if key in info:
            out[key] = _float_scalar(info[key])
    return out


def _save_map_npz(path: Path, info: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for key in (
        "delta_h_map",
        "delta_h_best",
        "delta_h_processed_map",
        "delta_h_processed_best",
        "score_by_tau",
        "amp_by_tau",
        "msc_by_tau",
        "msc_raw_by_scale_by_tau",
        "msc_by_scale_by_tau",
        "msc_raw_by_scale_best",
        "msc_by_scale_best",
        "tau_frames",
        "tau_steps",
        "window_start_frames",
        "window_start_steps",
    ):
        if key in info:
            arrays[key] = np.asarray(info[key])
    np.savez_compressed(path, **arrays)
    delta_map = np.asarray(info["delta_h_map"], dtype=np.float64)
    delta_best = np.asarray(info["delta_h_best"], dtype=np.float64)
    return {
        "map_path": str(path),
        "delta_h_map_shape": [int(x) for x in delta_map.shape],
        "delta_h_best_mean": float(np.nanmean(delta_best)),
        "delta_h_best_std": float(np.nanstd(delta_best)),
        "delta_h_best_min": float(np.nanmin(delta_best)),
        "delta_h_best_max": float(np.nanmax(delta_best)),
    }


def _stack_parts(parts_by_seed: list[tuple[Any, Any, Any, Any, Any]]) -> tuple[Any, Any, Any, Any, Any]:
    return tuple(jnp.stack([parts[i] for parts in parts_by_seed], axis=0) for i in range(5))


def _select_info_for_candidate_seed(info_batch: dict[str, Any], pop_idx: int, seed_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info_batch.items():
        arr = np.asarray(value)
        out[key] = arr[int(pop_idx), int(seed_idx)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "For one selected Flow-Lenia optimized candidate, compute two 4-seed MSPD/deltaH-map sets: "
            "exact optimizer replay and same-A/P-init replay with fresh non-init stochasticity."
        )
    )
    parser.add_argument(
        "--selected-run-dir",
        default=(
            "experiments/paper_check_flow_lenia/"
            "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
            "optimization/run_000"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
            "debug_deltaH_exact_vs_same_init/run_000"
        ),
    )
    parser.add_argument("--alt-seed-base", type=int, default=991337)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument(
        "--seed-indices",
        default=None,
        help="Optional comma-separated optimizer seed indices to score, e.g. '0' or '0,1,2,3'.",
    )
    parser.add_argument(
        "--sequential-seeds",
        action="store_true",
        help="Score selected seed indices one at a time with seed batch shape [1].",
    )
    parser.add_argument("--legacy-sigma-collision", action="store_true", default=True)
    args = parser.parse_args()

    selected_run_dir = _path_from_repo(args.selected_run_dir)
    out_dir = _path_from_repo(args.output_dir)
    maps_dir = out_dir / "maps"
    selected = json.loads((selected_run_dir / "selected_candidate.json").read_text())
    pop_path = _path_from_repo(selected["source_pop_traj"])
    source_run_dir = _path_from_repo(selected.get("source_run_dir", pop_path.parent))
    pop = _load_pickle(pop_path)

    i_iter = int(selected["iter"])
    pop_idx = int(selected["pop_idx"])
    seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)[i_iter]
    stored_scores_all = np.asarray(pop["score_by_seed"], dtype=np.float64)[i_iter, pop_idx]
    opt_mean = float(np.nanmean(stored_scores_all))
    if args.seed_indices is not None and str(args.seed_indices).strip():
        seed_indices = [int(x.strip()) for x in str(args.seed_indices).split(",") if x.strip()]
    else:
        n_seeds = min(int(args.n_seeds), int(seed_keys.shape[0]))
        seed_indices = list(range(n_seeds))
    if not seed_indices:
        raise ValueError("No seed indices selected.")
    for seed_idx in seed_indices:
        if seed_idx < 0 or seed_idx >= int(seed_keys.shape[0]):
            raise ValueError(f"seed_idx={seed_idx} out of range for seed_keys shape={seed_keys.shape}.")
    stored_scores = np.asarray([stored_scores_all[seed_idx] for seed_idx in seed_indices], dtype=np.float64)

    flat = _load_flat_config(
        source_run_dir / "optimization_config.yaml",
        legacy_sigma_collision=bool(args.legacy_sigma_collision),
    )
    (
        score_maps_batch_from_parts,
        initial_ap_diff,
        key_parts,
        metric_cfg,
        substrate_param_dims,
        optimize_tau,
    ) = _build_map_eval(flat)
    params_full_all = _params_full_batch(
        pop,
        i_iter=i_iter,
        substrate_param_dims=substrate_param_dims,
        optimize_tau=optimize_tau,
    )
    params_full_batch = jnp.asarray(params_full_all, dtype=jnp.float32)
    selected_params_full = jnp.asarray(params_full_all[pop_idx], dtype=jnp.float32)

    rows: list[dict[str, Any]] = []
    seed_chunks = [[seed_idx] for seed_idx in seed_indices] if args.sequential_seeds else [seed_indices]
    for chunk_seed_indices in seed_chunks:
        opt_keys = [jnp.asarray(seed_keys[seed_idx], dtype=jnp.uint32) for seed_idx in chunk_seed_indices]
        alt_seeds = [int(args.alt_seed_base) + int(seed_idx) for seed_idx in chunk_seed_indices]
        alt_keys = [jax.random.PRNGKey(seed) for seed in alt_seeds]
        exact_parts_by_seed = [key_parts(key) for key in opt_keys]
        alt_parts_by_seed = [key_parts(key) for key in alt_keys]
        precise_parts_batch = _stack_parts(exact_parts_by_seed)
        same_init_parts_batch = _stack_parts(
            [
                (
                    exact_parts_by_seed[local_seed_idx][0],
                    alt_parts_by_seed[local_seed_idx][1],
                    alt_parts_by_seed[local_seed_idx][2],
                    alt_parts_by_seed[local_seed_idx][3],
                    alt_parts_by_seed[local_seed_idx][4],
                )
                for local_seed_idx in range(len(chunk_seed_indices))
            ]
        )
        variants = {
            "precise": precise_parts_batch,
            "same_init_new_stochasticity": same_init_parts_batch,
        }

        for variant_name, parts_batch in variants.items():
            print(
                f"Scoring full optimizer batch variant={variant_name} "
                f"pop_shape={list(params_full_all.shape)} seed_indices={chunk_seed_indices}",
                flush=True,
            )
            score_all_j, info_all_j = score_maps_batch_from_parts(params_full_batch, parts_batch)
            score_all = np.asarray(jax.device_get(score_all_j), dtype=np.float64)
            info_all = jax.device_get(info_all_j)
            print(
                f"Done full optimizer batch variant={variant_name} "
                f"selected_scores={score_all[pop_idx, :len(chunk_seed_indices)].tolist()}",
                flush=True,
            )
            for local_seed_idx, seed_idx in enumerate(chunk_seed_indices):
                opt_key = opt_keys[local_seed_idx]
                alt_seed = alt_seeds[local_seed_idx]
                alt_key = alt_keys[local_seed_idx]
                parts_seed = tuple(parts_batch[i][local_seed_idx] for i in range(5))
                score_f = float(score_all[pop_idx, local_seed_idx])
                info = _select_info_for_candidate_seed(info_all, pop_idx, local_seed_idx)
                ap_a_diff, ap_p_diff = initial_ap_diff(
                    exact_parts_by_seed[local_seed_idx][0],
                    parts_seed[0],
                    selected_params_full,
                )
                metadata = {
                    "variant": variant_name,
                    "selected_run_dir": str(selected_run_dir),
                    "source_run_dir": str(source_run_dir),
                    "source_pop_traj": str(pop_path),
                    "iter": int(i_iter),
                    "pop_idx": int(pop_idx),
                    "seed_idx": int(seed_idx),
                    "local_seed_idx": int(local_seed_idx),
                    "opt_eval_key": _jsonable_key(opt_key),
                    "alt_seed": int(alt_seed),
                    "alt_eval_key": _jsonable_key(alt_key),
                    "resolved_args_sigma": float(flat.get("sigma")),
                    "resolved_args_flow_sigma": (
                        None if flat.get("flow_sigma", None) is None else float(flat.get("flow_sigma"))
                    ),
                    "legacy_sigma_collision": bool(args.legacy_sigma_collision),
                    "initial_A_max_abs_diff": float(np.asarray(jax.device_get(ap_a_diff), dtype=np.float64)),
                    "initial_P_max_abs_diff": float(np.asarray(jax.device_get(ap_p_diff), dtype=np.float64)),
                }
                map_path = maps_dir / f"{selected_run_dir.name}_seed{seed_idx:02d}_{variant_name}_deltaH_maps.npz"
                map_summary = _save_map_npz(map_path, info, metadata)
                row = {
                    **metadata,
                    **_info_small(info),
                    **map_summary,
                    "stored_opt_seed_score": float(stored_scores_all[seed_idx]),
                    "stored_opt_mean_score": opt_mean,
                    "recomputed_score": score_f,
                    "score_diff_vs_stored_seed": score_f - float(stored_scores_all[seed_idx]),
                    "score_abs_diff_vs_stored_seed": abs(score_f - float(stored_scores_all[seed_idx])),
                    "score_diff_vs_stored_mean": score_f - opt_mean,
                    "score_abs_diff_vs_stored_mean": abs(score_f - opt_mean),
                }
                rows.append(row)
                print(
                    f"Done variant={variant_name} seed_idx={seed_idx} score={score_f:.10g} "
                    f"diff_seed={row['score_diff_vs_stored_seed']:.3g} map={map_path}",
                    flush=True,
                )

    aggregates: dict[str, Any] = {}
    for variant_name in sorted({str(r["variant"]) for r in rows}):
        sub = [r for r in rows if r["variant"] == variant_name]
        scores = np.asarray([r["recomputed_score"] for r in sub], dtype=np.float64)
        diffs = np.asarray([r["score_diff_vs_stored_seed"] for r in sub], dtype=np.float64)
        aggregates[variant_name] = {
            "n": int(len(sub)),
            "score_mean": float(np.nanmean(scores)),
            "score_std": float(np.nanstd(scores)),
            "score_min": float(np.nanmin(scores)),
            "score_max": float(np.nanmax(scores)),
            "mean_diff_vs_stored_seed": float(np.nanmean(diffs)),
            "max_abs_diff_vs_stored_seed": float(np.nanmax(np.abs(diffs))),
            "diff_vs_stored_mean_score": float(np.nanmean(scores) - opt_mean),
            "stored_opt_mean_score": opt_mean,
        }

    payload = {
        "status": "ok",
        "selected_run_dir": str(selected_run_dir),
        "source_run_dir": str(source_run_dir),
        "source_pop_traj": str(pop_path),
        "iter": int(i_iter),
        "pop_idx": int(pop_idx),
        "seed_indices": [int(x) for x in seed_indices],
        "n_seeds": int(len(seed_indices)),
        "stored_opt_score_by_seed": [float(stored_scores_all[seed_idx]) for seed_idx in seed_indices],
        "stored_opt_score_by_seed_all": [float(x) for x in stored_scores_all],
        "stored_opt_mean_score": opt_mean,
        "metric_summary": metric_summary(metric_cfg),
        "aggregates": aggregates,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    csv_path = out_dir / "summary_rows.csv"
    csv_keys = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": str(summary_path), "summary_csv": str(csv_path), "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
