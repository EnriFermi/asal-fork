from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from flowlenia_minibang_simulate import select_params
from paper_suite_c2_flowlenia_highres import _discover_checkpoints, _root_label
from paper_suite_common import ensure_dir, load_config, log_event, resolve_path, to_plain, write_csv, write_json


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _section(cfg: Any) -> Any:
    return _get(cfg.get("simulation", {}), "plife_plus_c1_lagrangian", {})


def _load_base_config(path: Path) -> tuple[Any, Any]:
    cfg = OmegaConf.load(str(path))
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("source", {}),
        cfg.get("substrate", {}),
        cfg.get("protocol", {}),
        cfg.get("evaluation", {}),
        cfg.get("metric", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def _make_substrate(args: Any):
    import substrates
    import util

    if str(args.substrate) != "plife_plus":
        raise ValueError(f"PLife++ C1 runner requires substrate='plife_plus', got {args.substrate!r}.")
    base = substrates.create_substrate(args.substrate, **util.substrate_kwargs_from_args(args))
    return substrates.FlattenSubstrateParameters(base)


def _candidate_base_id(row: dict[str, Any]) -> str:
    source_idx = int(row.get("source_run_idx", row.get("run_idx", -1)))
    if source_idx >= 0 and int(row.get("source_root_rank", 0)) == 0:
        return f"plife_opt_run_{source_idx:03d}"
    if source_idx >= 0:
        return f"plife_opt_root{int(row['source_root_rank']):02d}_{_root_label(Path(row['source_root']))}_run_{source_idx:03d}"
    return f"plife_opt_root{int(row['source_root_rank']):02d}_{_root_label(Path(row['source_root']))}_{Path(row['checkpoint_dir']).name}"


def _candidate_uid(row: dict[str, Any], *, candidate_kind: str, candidate_idx: int, source_candidate_idx: int | None = None) -> str:
    base = _candidate_base_id(row)
    if candidate_kind == "optimized":
        if source_candidate_idx is None:
            return base
        return f"{base}_sel_{int(source_candidate_idx):03d}"
    return f"{base}_{candidate_kind}_{candidate_idx:03d}"


def _random_checkpoint_dir(row: dict[str, Any], random_group_idx: int, random_idx: int) -> Path:
    source_root = Path(row["source_root"])
    return (
        source_root.parent
        / "frustration_simulation"
        / "random_params"
        / f"group_{int(random_group_idx):03d}"
        / f"random_{int(random_idx):03d}"
    )


def _control_seed(section: Any, row: dict[str, Any], suite_idx: int, *, offset: int) -> int:
    base = int(_get(section, "run_seed_base", 400_000))
    mode = str(_get(section, "run_seed_mode", "source_run_idx")).strip().lower()
    source_run_idx = int(row.get("source_run_idx", -1))
    if mode == "source_run_idx" and source_run_idx >= 0:
        group_idx = source_run_idx
    elif mode == "suite_index":
        group_idx = int(suite_idx)
    else:
        raise ValueError("run_seed_mode must be 'source_run_idx' or 'suite_index'.")
    return int(base + 2 * group_idx + int(offset))


def _trial_paths(output_root: Path, trial_idx: int) -> dict[str, Path]:
    name = f"trial_{int(trial_idx):05d}"
    trial_dir = output_root / "trial_data"
    return {
        "json": trial_dir / f"{name}.json",
        "lagrangian": trial_dir / f"{name}_lagrangian.npz",
        "params": output_root / "params" / f"{name}_params.npy",
    }


def _lagrangian_status(path: Path, *, expected_start: int, expected_end: int, expected_samples: int) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"
    try:
        with np.load(path, allow_pickle=False) as data:
            required = [
                "sample_offsets_steps",
                "xy_late_sample_steps",
                "sample_every_steps",
                "trajectory_start_steps",
                "trajectory_end_steps",
                "trajectory_window_steps",
            ]
            missing = [key for key in required if key not in data.files]
            if missing:
                return False, "missing keys: " + ",".join(missing)
            xy_key = "xy_trajectory" if "xy_trajectory" in data.files else "xy_control_a" if "xy_control_a" in data.files else None
            if xy_key is None:
                return False, "missing trajectory key: expected xy_trajectory or legacy xy_control_a"
            start = int(np.asarray(data["trajectory_start_steps"]).item())
            end = int(np.asarray(data["trajectory_end_steps"]).item())
            xy = np.asarray(data[xy_key])
            offsets = np.asarray(data["sample_offsets_steps"]).reshape(-1)
    except Exception as exc:
        return False, f"cannot read {path}: {type(exc).__name__}: {exc}"
    if start != int(expected_start) or end != int(expected_end):
        return False, f"trajectory range {start}..{end}, expected {expected_start}..{expected_end}"
    if int(xy.shape[0]) != int(expected_samples):
        return False, f"sample count {xy.shape[0]}, expected {expected_samples}"
    if int(offsets.size) != int(expected_samples):
        return False, f"offset count {offsets.size}, expected {expected_samples}"
    return True, ""


def _unwrap_sampled_xy(xy_seq: np.ndarray, *, domain_y: float, domain_x: float) -> np.ndarray:
    xy = np.asarray(xy_seq, dtype=np.float32)
    if xy.shape[0] <= 1:
        return xy
    dxy = xy[1:] - xy[:-1]
    if domain_y > 0:
        dxy[..., 0] = (dxy[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
    if domain_x > 0:
        dxy[..., 1] = (dxy[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
    increments = np.cumsum(dxy, axis=0)
    return np.concatenate((xy[:1], xy[:1] + increments), axis=0)


def _select_candidate(
    *,
    source_row: dict[str, Any],
    section: Any,
    checkpoint_dir: Path,
    suite_idx: int,
    trial_idx: int,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
    selected_item: dict[str, Any],
) -> dict[str, Any]:
    item = dict(selected_item)
    item["trial_idx"] = int(trial_idx)
    source_candidate_idx = item.get("selection_idx", None)
    item["trial_uid"] = _candidate_uid(
        source_row,
        candidate_kind=candidate_kind,
        candidate_idx=candidate_idx,
        source_candidate_idx=None if source_candidate_idx is None else int(source_candidate_idx),
    )
    item["optimized_run_idx"] = int(suite_idx)
    item["source_optimized_run_idx"] = int(source_row.get("source_run_idx", -1))
    item["source_root"] = str(source_row["source_root"])
    item["source_root_rank"] = int(source_row["source_root_rank"])
    item["source_checkpoint_dir"] = str(checkpoint_dir)
    item["source_candidate_idx"] = -1 if source_candidate_idx is None else int(source_candidate_idx)
    item["source_candidate_source"] = str(item.get("source", ""))
    item["source_candidate_iter"] = int(item.get("iter", -1))
    item["source_candidate_pop_idx"] = int(item.get("pop_idx", -1))
    item["candidate_kind"] = str(candidate_kind)
    item["candidate_idx"] = int(candidate_idx)
    item["candidate_label"] = str(candidate_label)
    item["seed_x"] = _control_seed(section, source_row, suite_idx, offset=0)
    item["seed_x1"] = _control_seed(section, source_row, suite_idx, offset=1)
    metric_seed_base = int(_get(section, "metric_seed_base", 800_000))
    item["metric_seed"] = int(metric_seed_base + int(trial_idx))
    return item


def _build_selected(checkpoints: list[dict[str, Any]], section: Any, rollout_flat: Any) -> list[dict[str, Any]]:
    include_random = bool(_get(section, "include_random_baselines", True))
    n_random = int(_get(section, "num_random_baselines", 3)) if include_random else 0
    n_per_checkpoint = int(_get(section, "n_trajectories_per_checkpoint", _get(section, "n_optimized_candidates_per_checkpoint", 1)))
    if n_per_checkpoint < 1:
        raise ValueError(f"n_trajectories_per_checkpoint must be >= 1, got {n_per_checkpoint}.")
    opt_select_args = OmegaConf.create(OmegaConf.to_container(rollout_flat, resolve=True))
    opt_select_args.n_trajectories = int(n_per_checkpoint)
    random_select_args = OmegaConf.create(OmegaConf.to_container(rollout_flat, resolve=True))
    random_select_args.n_trajectories = 1
    for key in (
        "selection_mode",
        "selection_seed",
        "include_population",
        "include_best_traj",
        "include_final_best",
        "selection_keep_top_n",
        "selection_loss_bias_gamma",
        "selection_loss_jitter_frac",
    ):
        value = _get(section, key, None)
        if value is not None:
            opt_select_args[key] = value
            random_select_args[key] = value
    selected: list[dict[str, Any]] = []
    suite_idx = 0
    for row in checkpoints:
        optimized_candidates = select_params(Path(row["checkpoint_dir"]), opt_select_args)
        if len(optimized_candidates) < n_per_checkpoint:
            raise ValueError(
                f"Checkpoint {row['checkpoint_dir']} yielded only {len(optimized_candidates)} optimized candidates, "
                f"expected {n_per_checkpoint}."
            )
        for opt_candidate in optimized_candidates[:n_per_checkpoint]:
            group_base = int(suite_idx) * (1 + n_random)
            source_run_idx = int(row.get("source_run_idx", -1))
            random_group_idx = source_run_idx if n_per_checkpoint == 1 and source_run_idx >= 0 else suite_idx
            opt_item = _select_candidate(
                source_row=row,
                section=section,
                checkpoint_dir=Path(row["checkpoint_dir"]),
                suite_idx=suite_idx,
                trial_idx=group_base,
                candidate_kind="optimized",
                candidate_idx=0,
                candidate_label="optimized",
                selected_item=opt_candidate,
            )
            opt_item["trial_uid"] = f"{opt_item['trial_uid']}_group_{suite_idx:03d}"
            selected.append(opt_item)
            for random_idx in range(n_random):
                random_dir = _random_checkpoint_dir(row, random_group_idx, random_idx)
                if not (random_dir / "best.pkl").exists():
                    raise FileNotFoundError(
                        "Missing PLife++ random baseline checkpoint. "
                        f"Expected {random_dir / 'best.pkl'} for suite_group={suite_idx}, random_group={random_group_idx}, "
                        f"source_run_idx={int(row.get('source_run_idx', -1))}, random_idx={random_idx}."
                    )
                random_selected = select_params(random_dir, random_select_args)
                if not random_selected:
                    raise FileNotFoundError(f"No selectable random params found in {random_dir}.")
                random_item = _select_candidate(
                    source_row=row,
                    section=section,
                    checkpoint_dir=random_dir,
                    suite_idx=suite_idx,
                    trial_idx=group_base + random_idx + 1,
                    candidate_kind="random",
                    candidate_idx=random_idx,
                    candidate_label=f"random_{random_idx:03d}",
                    selected_item=random_selected[0],
                )
                random_item["trial_uid"] = f"{_candidate_base_id(row)}_group_{suite_idx:03d}_random_{random_idx:03d}"
                selected.append(random_item)
            suite_idx += 1
    return selected


def _init_batched_state(substrate, params_batch: jax.Array, seeds: list[int]):
    init_pairs = [jax.random.split(jax.random.PRNGKey(int(seed)), 2) for seed in seeds]
    rng_batch = jnp.stack([pair[0] for pair in init_pairs], axis=0)
    init_keys = jnp.stack([pair[1] for pair in init_pairs], axis=0)
    states = jax.jit(lambda keys, params: jax.vmap(substrate.init_state)(keys, params))(init_keys, params_batch)
    return rng_batch, states


def _simulate_xy_batch(
    *,
    substrate,
    params_batch: np.ndarray,
    seeds: list[int],
    total_steps: int,
    late_start: int,
    late_end: int,
    sample_every: int,
    jit_microbatch: int,
) -> tuple[np.ndarray, np.ndarray]:
    params_np = np.asarray(params_batch, dtype=np.float32)
    params_lanes = params_np
    seeds = [int(seed) for seed in seeds]
    params_j = jnp.asarray(params_lanes)
    rng_batch, states = _init_batched_state(substrate, params_j, seeds)
    lane_count = int(params_lanes.shape[0])
    cache: dict[tuple[int, bool], Any] = {}
    chunk_steps = int(sample_every)
    chunks_per_call = max(1, int(jit_microbatch) // max(1, chunk_steps))

    def get_stepper(n_chunks: int, collect: bool):
        key = (int(n_chunks), bool(collect))
        if key in cache:
            return cache[key]

        def advance(rng_in, states_in, params_in):
            def body(carry, _unused):
                rng_cur, st_cur = carry
                split = jax.vmap(lambda k: jax.random.split(k, 2))(rng_cur)
                rng_next = split[:, 0]
                chunk_keys = split[:, 1]
                step_keys = jax.vmap(lambda k: jax.random.split(k, chunk_steps))(chunk_keys)
                step_keys = jnp.swapaxes(step_keys, 0, 1)

                def step_body(st_inner, step_keys_lanes):
                    return jax.vmap(substrate.step_state)(step_keys_lanes, st_inner, params_in), None

                st_next, _ = jax.lax.scan(step_body, st_cur, step_keys)
                return (rng_next, st_next), st_next["x"]

            (rng_out, states_out), xs = jax.lax.scan(body, (rng_in, states_in), None, length=int(n_chunks))
            if collect:
                return rng_out, states_out, xs
            return rng_out, states_out, jnp.zeros((0, lane_count, 1, 2), dtype=jnp.float32)

        cache[key] = jax.jit(advance)
        return cache[key]

    chunks: list[np.ndarray] = []
    steps_chunks: list[np.ndarray] = []
    steps_done = 0
    while steps_done < int(total_steps):
        target = int(late_end) if steps_done >= int(late_start) else int(late_start)
        target = min(int(total_steps), target)
        if steps_done >= int(late_end):
            break
        n_chunks = min(chunks_per_call, (target - steps_done) // chunk_steps)
        if n_chunks < 1:
            raise ValueError(
                "PLife++ C1 simulation range is not aligned to sample_every_steps: "
                f"steps_done={steps_done}, target={target}, sample_every={chunk_steps}."
            )
        collect = bool(steps_done >= int(late_start))
        rng_batch, states, xs = get_stepper(n_chunks, collect)(rng_batch, states, params_j)
        n = int(n_chunks * chunk_steps)
        if collect:
            step_values = steps_done + chunk_steps * np.arange(1, n_chunks + 1, dtype=np.int32)
            keep = (step_values > int(late_start)) & (step_values <= int(late_end))
            if np.any(keep):
                xs_np = np.asarray(jax.device_get(xs), dtype=np.float32)
                chunks.append(xs_np[keep])
                steps_chunks.append(step_values[keep])
        steps_done += n
    if not chunks:
        raise RuntimeError("No PLife++ C1 trajectory samples were collected.")
    xy = np.concatenate(chunks, axis=0)
    steps = np.concatenate(steps_chunks, axis=0)
    return xy, steps


def _write_trial_artifacts(
    *,
    output_root: Path,
    row: dict[str, Any],
    xy: np.ndarray,
    sample_steps: np.ndarray,
    base_cfg: Any,
    unwrap: bool,
    domain_y: float,
    domain_x: float,
) -> dict[str, Any]:
    trial_idx = int(row["trial_idx"])
    paths = _trial_paths(output_root, trial_idx)
    ensure_dir(paths["json"].parent)
    ensure_dir(paths["params"].parent)
    np.save(paths["params"], np.asarray(row["params"], dtype=np.float32))
    late_start = int(base_cfg.evaluation.late_window_start_steps)
    late_end = int(base_cfg.evaluation.late_window_end_steps)
    sample_every = int(base_cfg.metric.sample_every_steps)
    if unwrap:
        xy = _unwrap_sampled_xy(xy, domain_y=domain_y, domain_x=domain_x)
    offsets = np.asarray(sample_steps, dtype=np.int32) - int(late_start)
    np.savez_compressed(
        paths["lagrangian"],
        xy_trajectory=np.asarray(xy, dtype=np.float32),
        # Legacy alias: old posthoc/frustration readers expect xy_control_a.
        xy_control_a=np.asarray(xy, dtype=np.float32),
        sample_offsets_steps=offsets.astype(np.int32),
        xy_late_sample_steps=np.asarray(sample_steps, dtype=np.int32),
        sample_every_steps=np.asarray(sample_every, dtype=np.int32),
        trajectory_start_steps=np.asarray(late_start, dtype=np.int32),
        trajectory_end_steps=np.asarray(late_end, dtype=np.int32),
        trajectory_window_steps=np.asarray(late_end - late_start, dtype=np.int32),
        metric_window_size_steps=np.asarray(int(base_cfg.metric.metric_window_size_steps), dtype=np.int32),
        metric_window_step_steps=np.asarray(int(base_cfg.metric.metric_window_step_steps), dtype=np.int32),
        metric_tau_steps=np.asarray(int(base_cfg.metric.metric_tau_steps), dtype=np.int32),
        params_path=str(paths["params"].relative_to(output_root)),
    )
    json_row = {
        "trial_idx": int(trial_idx),
        "trial_uid": str(row["trial_uid"]),
        "optimized_run_idx": int(row["optimized_run_idx"]),
        "source_optimized_run_idx": int(row["source_optimized_run_idx"]),
        "source_root_rank": int(row["source_root_rank"]),
        "source_root": str(row["source_root"]),
        "source_checkpoint_dir": str(row["source_checkpoint_dir"]),
        "source_candidate_idx": int(row.get("source_candidate_idx", -1)),
        "source_candidate_source": str(row.get("source_candidate_source", "")),
        "source_candidate_iter": int(row.get("source_candidate_iter", -1)),
        "source_candidate_pop_idx": int(row.get("source_candidate_pop_idx", -1)),
        "candidate_kind": str(row["candidate_kind"]),
        "candidate_idx": int(row["candidate_idx"]),
        "candidate_label": str(row["candidate_label"]),
        "trajectory_seed": int(row["seed_x"]),
        "seed_x": int(row["seed_x"]),
        "seed_x1": int(row["seed_x1"]),
        "metric_seed": int(row["metric_seed"]),
        "lagrangian_path": str(paths["lagrangian"].relative_to(output_root)),
        "params_path": str(paths["params"].relative_to(output_root)),
        "trajectory_start_steps": int(late_start),
        "trajectory_end_steps": int(late_end),
        "sample_every_steps": int(sample_every),
    }
    paths["json"].write_text(json.dumps(json_row, indent=2) + "\n")
    return json_row


def _write_manifest(
    output_root: Path,
    *,
    selected: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
    base_config: Path,
    expected_start: int,
    expected_end: int,
    expected_samples: int,
) -> None:
    trajectories = []
    for row in selected:
        paths = _trial_paths(output_root, int(row["trial_idx"]))
        ready, message = _lagrangian_status(
            paths["lagrangian"],
            expected_start=expected_start,
            expected_end=expected_end,
            expected_samples=expected_samples,
        )
        trajectories.append(
            {
                "trial_idx": int(row["trial_idx"]),
                "trial_uid": str(row["trial_uid"]),
                "optimized_run_idx": int(row["optimized_run_idx"]),
                "source_optimized_run_idx": int(row["source_optimized_run_idx"]),
                "source_root_rank": int(row["source_root_rank"]),
                "source_root": str(row["source_root"]),
                "source_checkpoint_dir": str(row["source_checkpoint_dir"]),
                "source_candidate_idx": int(row.get("source_candidate_idx", -1)),
                "source_candidate_source": str(row.get("source_candidate_source", "")),
                "source_candidate_iter": int(row.get("source_candidate_iter", -1)),
                "source_candidate_pop_idx": int(row.get("source_candidate_pop_idx", -1)),
                "candidate_kind": str(row["candidate_kind"]),
                "candidate_idx": int(row["candidate_idx"]),
                "candidate_label": str(row["candidate_label"]),
                "seed_x": int(row["seed_x"]),
                "seed_x1": int(row["seed_x1"]),
                "metric_seed": int(row["metric_seed"]),
                "lagrangian_path": str(paths["lagrangian"]),
                "params_path": str(paths["params"]),
                "ready": bool(ready),
                "status": message,
            }
        )
    write_json(
        output_root / "manifest.json",
        {
            "source_kind": "plife_plus_c1_control_ab_lagrangian",
            "base_config": str(base_config),
            "trajectory_start_steps": int(expected_start),
            "trajectory_end_steps": int(expected_end),
            "n_trajectories": len(trajectories),
            "trajectories": trajectories,
            "commands": command_rows,
        },
    )


def _smoke_selected(output_root: Path, base_cfg: Any, rollout_flat: Any, section: Any) -> list[dict[str, Any]]:
    args = SimpleNamespace(**OmegaConf.to_container(rollout_flat, resolve=True))
    substrate = _make_substrate(args)
    params = np.asarray(substrate.default_params(jax.random.PRNGKey(123)), dtype=np.float32)
    selected = []
    for idx, kind in enumerate(["optimized", "random"]):
        selected.append(
            {
                "source": "smoke_default",
                "iter": 0,
                "pop_idx": -1,
                "loss": float(idx),
                "params": params + np.float32(idx) * 1e-4,
                "param_hash": f"smoke_{idx}",
                "saturation_T": 0.0,
                "trial_idx": idx,
                "trial_uid": f"plife_smoke_{kind}",
                "optimized_run_idx": 0,
                "source_optimized_run_idx": 0,
                "source_root": str(output_root),
                "source_root_rank": 0,
                "source_checkpoint_dir": str(output_root),
                "candidate_kind": kind,
                "candidate_idx": 0,
                "candidate_label": kind if kind == "optimized" else "random_000",
                "seed_x": _control_seed(section, {"source_run_idx": 0}, 0, offset=0),
                "seed_x1": _control_seed(section, {"source_run_idx": 0}, 0, offset=1),
                "metric_seed": int(_get(section, "metric_seed_base", 800_000)) + idx,
            }
        )
    return selected


def run(
    config_path: str | Path,
    *,
    smoke: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    section = _section(cfg)
    if not bool(_get(section, "enabled", True)):
        log_event("PLife++ C1 lagrangian simulation disabled", component="plife-c1")
        return {"status": "disabled"}

    output_root = resolve_path(_get(section, "output_root", "experiments/paper_check_plife_plus/checkpoints/c1_lagrangian_24k"))
    assert output_root is not None
    if smoke:
        output_root = resolve_path(_get(section, "smoke_output_root", "analysis/results/paper_suite_smoke/plife_plus_c1_lagrangian"))
        assert output_root is not None
    output_root = ensure_dir(output_root)

    base_config = resolve_path(_get(section, "base_config", "experiments/paper_check_plife_plus/frustration_simulation/config.yaml"))
    if base_config is None or not base_config.exists():
        raise FileNotFoundError(f"PLife++ C1 base_config not found: {base_config}")
    base_cfg, rollout_flat = _load_base_config(base_config)
    if smoke:
        base_cfg.substrate.rollout_steps = 48
        base_cfg.substrate.n_particles = 32
        base_cfg.protocol.total_steps = 48
        base_cfg.protocol.warmup_steps = 16
        base_cfg.evaluation.late_window_start_steps = 24
        base_cfg.evaluation.late_window_end_steps = 48
        base_cfg.metric.sample_every_steps = 1
        base_cfg.metric.metric_window_size_steps = 8
        base_cfg.metric.metric_window_step_steps = 4
        base_cfg.metric.metric_tau_steps = 2
        rollout_flat = OmegaConf.merge(
            base_cfg.get("meta", {}),
            base_cfg.get("source", {}),
            base_cfg.get("substrate", {}),
            base_cfg.get("protocol", {}),
            base_cfg.get("evaluation", {}),
            base_cfg.get("metric", {}),
            base_cfg.get("logging", {}),
        )

    expected_start = int(base_cfg.evaluation.late_window_start_steps)
    expected_end = int(base_cfg.evaluation.late_window_end_steps)
    sample_every = int(base_cfg.metric.sample_every_steps)
    total_steps = int(base_cfg.protocol.total_steps)
    if expected_start < 0 or expected_end <= expected_start:
        raise ValueError(f"Invalid PLife++ C1 late window: {expected_start}..{expected_end}.")
    if expected_end > total_steps:
        raise ValueError(f"PLife++ C1 late window end {expected_end} exceeds total_steps={total_steps}.")
    if sample_every <= 0:
        raise ValueError(f"PLife++ C1 sample_every_steps must be positive, got {sample_every}.")
    if expected_start % sample_every != 0 or expected_end % sample_every != 0:
        raise ValueError(
            "PLife++ C1 late window boundaries must be divisible by sample_every_steps: "
            f"start={expected_start}, end={expected_end}, sample_every={sample_every}."
        )
    if (expected_end - expected_start) % sample_every != 0:
        raise ValueError(
            "PLife++ C1 late window must be divisible by sample_every_steps: "
            f"range={expected_end - expected_start}, sample_every={sample_every}."
        )
    expected_samples = int((expected_end - expected_start) // sample_every)
    if expected_samples < 2:
        raise ValueError(f"PLife++ C1 expected_samples must be >= 2, got {expected_samples}.")

    rollout_flat.n_trajectories = 1
    rollout_flat.selection_mode = str(_get(section, "selection_mode", "loss"))
    batch_size = max(1, int(_get(section, "batch_size", 9)))
    if smoke:
        batch_size = min(batch_size, 2)
    jit_microbatch = max(1, int(_get(section, "jit_microbatch", 256 if not smoke else 8)))

    if smoke:
        selected = _smoke_selected(output_root, base_cfg, rollout_flat, section)
    else:
        checkpoints = _discover_checkpoints(section)
        log_event(
            f"PLife++ C1 lagrangian discovered n_checkpoints={len(checkpoints)}",
            component="plife-c1",
        )
        if not checkpoints:
            raise FileNotFoundError("No PLife++ optimized checkpoints found for C1 lagrangian simulation.")
        selected = _build_selected(checkpoints, section, rollout_flat)
        expected_groups_raw = _get(section, "expected_optimized_groups", None)
        if expected_groups_raw is not None:
            expected_groups = int(expected_groups_raw)
            actual_groups = sum(1 for row in selected if str(row.get("candidate_kind")) == "optimized")
            if actual_groups != expected_groups:
                raise ValueError(
                    "PLife++ C1 lagrangian optimized group count mismatch: "
                    f"found {actual_groups}, expected {expected_groups}. "
                    "Check simulation.plife_plus_c1_lagrangian.optimized_checkpoint_roots."
                )

    ready_by_trial: dict[int, tuple[bool, str]] = {}
    for row in selected:
        paths = _trial_paths(output_root, int(row["trial_idx"]))
        ready, message = _lagrangian_status(
            paths["lagrangian"],
            expected_start=expected_start,
            expected_end=expected_end,
            expected_samples=expected_samples,
        )
        ready_by_trial[int(row["trial_idx"])] = (ready, message)
    pending = [row for row in selected if force or not ready_by_trial[int(row["trial_idx"])][0]]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    command_rows = []
    pending_ids = {int(row["trial_idx"]) for row in pending}
    for row in selected:
        trial_idx = int(row["trial_idx"])
        ready, message = ready_by_trial[trial_idx]
        status = "queued" if trial_idx in pending_ids else "exists"
        if force and trial_idx in pending_ids:
            message = "force"
        command_rows.append(
            {
                "trial_idx": trial_idx,
                "trial_uid": str(row["trial_uid"]),
                "candidate_kind": str(row["candidate_kind"]),
                "candidate_idx": int(row["candidate_idx"]),
                "optimized_run_idx": int(row["optimized_run_idx"]),
                "status": status,
                "message": message,
                "lagrangian_path": str(_trial_paths(output_root, trial_idx)["lagrangian"]),
            }
        )

    _write_manifest(
        output_root,
        selected=selected,
        command_rows=command_rows,
        base_config=base_config,
        expected_start=expected_start,
        expected_end=expected_end,
        expected_samples=expected_samples,
    )

    if dry_run:
        for row in command_rows:
            if row["status"] == "queued":
                row["status"] = "dry_run"
        _write_manifest(
            output_root,
            selected=selected,
            command_rows=command_rows,
            base_config=base_config,
            expected_start=expected_start,
            expected_end=expected_end,
            expected_samples=expected_samples,
        )
        summary = {
            "status": "dry_run",
            "output_root": str(output_root),
            "n_selected": len(selected),
            "n_to_run": len(pending),
            "n_batches_to_run": len(batches),
            "batch_size": int(batch_size),
            "manifest": str(output_root / "manifest.json"),
        }
        write_json(output_root / "simulation_summary.json", summary)
        return summary

    args = SimpleNamespace(**OmegaConf.to_container(rollout_flat, resolve=True))
    substrate = _make_substrate(args)
    expected_dim = int(substrate.n_params)
    rows_out = []
    for batch_idx, batch in enumerate(batches, start=1):
        existing_protected = [
            str(_trial_paths(output_root, int(row["trial_idx"]))["lagrangian"])
            for row in batch
            if str(row.get("candidate_kind", "optimized")) == "optimized"
            and _trial_paths(output_root, int(row["trial_idx"]))["lagrangian"].exists()
            and not force
        ]
        if existing_protected:
            raise RuntimeError(
                "Refusing to overwrite incomplete optimized PLife++ C1 trajectories without --force: "
                + ", ".join(existing_protected[:10])
            )
        params_batch = np.stack([np.asarray(row["params"], dtype=np.float32) for row in batch], axis=0)
        if params_batch.shape[1] != expected_dim:
            raise ValueError(f"Loaded params have dim {params_batch.shape[1]}, substrate expects {expected_dim}.")
        log_event(
            "PLife++ C1 lagrangian running batch "
            f"{batch_idx}/{len(batches)} batch_size={len(batch)} trials={[int(row['trial_idx']) for row in batch]}",
            component="plife-c1",
        )
        xy_batch, sample_steps = _simulate_xy_batch(
            substrate=substrate,
            params_batch=params_batch,
            seeds=[int(row["seed_x"]) for row in batch],
            total_steps=total_steps,
            late_start=expected_start,
            late_end=expected_end,
            sample_every=sample_every,
            jit_microbatch=jit_microbatch,
        )
        unwrap = bool(_get(base_cfg.metric, "metric_unwrap_state_x", True)) and str(_get(base_cfg.substrate, "border", "wall")) == "torus"
        domain = float(_get(base_cfg.substrate, "world_size", 1.0))
        for i, row in enumerate(batch):
            rows_out.append(
                _write_trial_artifacts(
                    output_root=output_root,
                    row=row,
                    xy=xy_batch[:, i],
                    sample_steps=sample_steps,
                    base_cfg=base_cfg,
                    unwrap=unwrap,
                    domain_y=domain,
                    domain_x=domain,
                )
            )
        for command in command_rows:
            if int(command["trial_idx"]) in {int(row["trial_idx"]) for row in batch}:
                ready, message = _lagrangian_status(
                    Path(command["lagrangian_path"]),
                    expected_start=expected_start,
                    expected_end=expected_end,
                    expected_samples=expected_samples,
                )
                command["status"] = "exists" if ready else "missing_lagrangian"
                command["message"] = message
        _write_manifest(
            output_root,
            selected=selected,
            command_rows=command_rows,
            base_config=base_config,
            expected_start=expected_start,
            expected_end=expected_end,
            expected_samples=expected_samples,
        )

    all_rows = []
    for row in selected:
        json_path = _trial_paths(output_root, int(row["trial_idx"]))["json"]
        if json_path.exists():
            all_rows.append(json.loads(json_path.read_text()))
    if all_rows:
        write_csv(output_root / "trial_results.csv", sorted(all_rows, key=lambda x: int(x["trial_idx"])))
    n_ready = 0
    for row in selected:
        paths = _trial_paths(output_root, int(row["trial_idx"]))
        ready, _message = _lagrangian_status(
            paths["lagrangian"],
            expected_start=expected_start,
            expected_end=expected_end,
            expected_samples=expected_samples,
        )
        n_ready += int(ready)
    status = "ok" if n_ready == len(selected) else "incomplete"
    summary = {
        "status": status,
        "output_root": str(output_root),
        "n_selected": len(selected),
        "n_ready": int(n_ready),
        "batch_size": int(batch_size),
        "jit_microbatch": int(jit_microbatch),
        "trajectory_start_steps": int(expected_start),
        "trajectory_end_steps": int(expected_end),
        "manifest": str(output_root / "manifest.json"),
    }
    write_json(output_root / "simulation_summary.json", summary)
    _write_manifest(
        output_root,
        selected=selected,
        command_rows=command_rows,
        base_config=base_config,
        expected_start=expected_start,
        expected_end=expected_end,
        expected_samples=expected_samples,
    )
    if status != "ok" and bool(_get(section, "required", True)):
        raise RuntimeError(f"PLife++ C1 lagrangian generation incomplete: {n_ready}/{len(selected)} ready.")
    log_event(f"PLife++ C1 lagrangian done status={status} n_ready={n_ready}/{len(selected)}", component="plife-c1")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build reusable PLife++ C1 control-A/control-B lagrangian trajectories.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(to_plain(run(args.config, smoke=args.smoke, force=args.force, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
