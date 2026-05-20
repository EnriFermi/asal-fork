from __future__ import annotations

import argparse
import json
import shutil
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
from tqdm import tqdm

from flowlenia_minibang_common import load_config, resolve_path, write_json
from flowlenia_minibang_resume import (
    _apply_resume_perturbation,
    _find_snapshot,
    _flush,
    _get,
    _read_snapshot,
    _reconstruct_state,
    _scalar,
)
from flowlenia_minibang_simulate import _make_substrate


BUFFER_KEYS = (
    "steps",
    "P",
    "A",
    "F",
    "lagrangian_xy",
    "lagrangian_c",
    "resume_batch_rng_key",
    "resume_batch_size",
    "resume_batch_index",
    "resume_selection0",
    "resume_jit_microbatch",
    "resume_snapshot_interval",
    "resume_seed",
    "resume_lagrangian_seed",
    "state_t",
    "state_mass_cycle_start",
)


def _jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        jobs = payload.get("jobs", [])
    else:
        jobs = payload
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{path} must contain a non-empty job list.")
    return [dict(job) for job in jobs]


def _resolve_job_path(raw: dict[str, Any], key: str, base: Path | None = None) -> Path:
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Branch job is missing {key!r}: {raw}")
    path = resolve_path(str(value), base)
    if path is None:
        raise ValueError(f"Could not resolve {key}={value!r}")
    return path


def _prepare_job(raw: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    traj_dir = _resolve_job_path(raw, "source_traj_dir")
    if not traj_dir.exists():
        raise FileNotFoundError(f"source_traj_dir not found: {traj_dir}")
    config_path = resolve_path(raw.get("config"), traj_dir) if raw.get("config") else traj_dir / "config.yaml"
    params_path = resolve_path(raw.get("params"), traj_dir) if raw.get("params") else traj_dir / "params.npy"
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if params_path is None or not params_path.exists():
        raise FileNotFoundError(f"params not found: {params_path}")

    cfg, flat = load_config(config_path)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    args = SimpleNamespace(**flat_args)
    apf_path, start_step, snapshot_idx = _find_snapshot(traj_dir / "apf_logs", int(raw["step"]) if raw.get("step") is not None else None)
    snapshot = _read_snapshot(apf_path, snapshot_idx)
    params = np.asarray(np.load(params_path), dtype=np.float32)

    total_default = int(_get(args, "rollout_steps", _get(args, "max_steps", 0)))
    if _get(args, "max_steps", None) is not None:
        total_default = min(total_default, int(_get(args, "max_steps")))
    end_step = int(raw["end_step"]) if raw.get("end_step") is not None else int(total_default)
    if raw.get("additional_steps") is not None:
        end_step = int(start_step) + int(raw["additional_steps"])
    if end_step <= int(start_step):
        raise ValueError(f"end_step must be > start_step, got {end_step} <= {start_step}")

    output_dir = _resolve_job_path(raw, "output_dir", traj_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists. Use --overwrite.")
    apf_out = output_dir / "apf_logs"
    apf_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    np.save(output_dir / "params.npy", params)

    original_batch_size = int(_scalar(snapshot, "resume_batch_size", 1))
    original_batch_index = int(_scalar(snapshot, "resume_batch_index", 0))
    snapshot_interval = int(_scalar(snapshot, "resume_snapshot_interval", _get(args, "snapshot_interval", 100)))
    jit_microbatch = int(_scalar(snapshot, "resume_jit_microbatch", _get(args, "jit_microbatch", snapshot_interval)))
    horizon_steps = int(end_step) - int(start_step)
    config_sig = _stable_json(flat_args)
    shape_sig = {
        "params": tuple(params.shape),
        "A": tuple(np.asarray(snapshot["A"]).shape),
        "P": tuple(np.asarray(snapshot["P"]).shape),
        "F": tuple(np.asarray(snapshot["F"]).shape) if "F" in snapshot else (),
        "lagrangian_xy": tuple(np.asarray(snapshot["lagrangian_xy"]).shape),
        "lagrangian_c": tuple(np.asarray(snapshot["lagrangian_c"]).shape),
    }
    group_key = (
        config_sig,
        tuple(sorted(shape_sig.items())),
        int(original_batch_size),
        int(snapshot_interval),
        int(jit_microbatch),
        int(horizon_steps),
        int(start_step) % max(1, int(snapshot_interval)),
    )
    return {
        "raw": raw,
        "traj_dir": traj_dir,
        "config_path": config_path,
        "params_path": params_path,
        "output_dir": output_dir,
        "apf_out": apf_out,
        "cfg": cfg,
        "flat_args": flat_args,
        "args": args,
        "apf_path": apf_path,
        "snapshot": snapshot,
        "snapshot_idx": int(snapshot_idx),
        "params": params,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "horizon_steps": int(horizon_steps),
        "original_batch_size": int(original_batch_size),
        "original_batch_index": int(original_batch_index),
        "snapshot_interval": int(snapshot_interval),
        "jit_microbatch": int(jit_microbatch),
        "group_key": group_key,
    }


def _make_batched_stepper(
    *,
    substrate: Any,
    rt: Any,
    original_batch_size: int,
    lag_flow_channel: int,
    lag_flow_reduce: str,
    lag_channel_mode: str,
    lag_noise_model: str,
    lag_diffusion_scale: float,
):
    def step_n(state_in, lag_xy_in, lag_ch_in, rng_in, params_in, original_batch_index_in, n_steps: int):
        def split_one(key):
            return jax.random.split(key, int(n_steps) * int(original_batch_size)).reshape(
                (int(n_steps), int(original_batch_size), 2)
            )

        rngs_by_item = jax.vmap(split_one)(rng_in)
        rngs_by_step = jnp.swapaxes(rngs_by_item, 0, 1)
        idx = jnp.asarray(original_batch_index_in, dtype=jnp.int32)
        gather_idx = jnp.broadcast_to(idx[None, :, None, None], (int(n_steps), idx.shape[0], 1, 2))
        rngs = jnp.take_along_axis(rngs_by_step, gather_idx, axis=2)[:, :, 0, :]

        def one_item(key_i, st_i, pts_i, ch_i, params_i):
            st_next = substrate.step_state(key_i, st_i, params_i)
            lag_key = jax.random.fold_in(key_i, jnp.uint32(0x4C4147))
            pts_next, ch_next = rt.advect_particles(
                points=pts_i,
                F=st_next["F"],
                A=st_next["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=ch_i,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return st_next, pts_next, ch_next

        vmapped_one = jax.vmap(one_item, in_axes=(0, 0, 0, 0, 0))

        def scan_body(carry, keys_i):
            st, pts, ch = carry
            st_next, pts_next, ch_next = vmapped_one(keys_i, st, pts, ch, params_in)
            return (st_next, pts_next, ch_next), None

        (state_out, lag_xy_out, lag_ch_out), _ = jax.lax.scan(scan_body, (state_in, lag_xy_in, lag_ch_in), rngs)
        return state_out, lag_xy_out, lag_ch_out

    cache: dict[int, Any] = {}

    def get(n_steps: int):
        n = int(n_steps)
        if n not in cache:
            cache[n] = jax.jit(lambda st, xy, ch, rng, params, idx: step_n(st, xy, ch, rng, params, idx, n))
        return cache[n]

    return get


def _new_out(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "apf_dir": job["apf_out"],
        "file_idx": 0,
        "sim_fps": float(_get(job["args"], "fps", 250.0)),
        "buffers": {key: [] for key in BUFFER_KEYS},
    }


def _resume_meta(job: dict[str, Any]) -> dict[str, np.ndarray]:
    snapshot = job["snapshot"]
    args = job["args"]
    return {
        "resume_batch_size": np.asarray(job["original_batch_size"], dtype=np.int32),
        "resume_batch_index": np.asarray(job["original_batch_index"], dtype=np.int32),
        "resume_selection0": np.asarray(_scalar(snapshot, "resume_selection0", 0), dtype=np.int32),
        "resume_jit_microbatch": np.asarray(job["jit_microbatch"], dtype=np.int32),
        "resume_snapshot_interval": np.asarray(job["snapshot_interval"], dtype=np.int32),
        "resume_seed": np.asarray(_scalar(snapshot, "resume_seed", _get(args, "seed", 0)), dtype=np.int64),
        "resume_lagrangian_seed": np.asarray(
            _scalar(snapshot, "resume_lagrangian_seed", _get(args, "lagrangian_seed", _get(args, "seed", 0))),
            dtype=np.int64,
        ),
    }


def _capture_apf_only(
    *,
    outs: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    state: dict[str, Any],
    lag_xy: jax.Array,
    lag_ch: jax.Array,
    rng: jax.Array,
    rel_step: int,
    resume_meta: list[dict[str, np.ndarray]],
) -> None:
    state_np = jax.device_get(state)
    lag_xy_np, lag_ch_np, rng_np = jax.device_get((lag_xy, lag_ch, rng))
    for i, (out, job, meta) in enumerate(zip(outs, jobs, resume_meta, strict=True)):
        b = out["buffers"]
        step = int(job["start_step"]) + int(rel_step)
        b["steps"].append(step)
        b["P"].append(np.asarray(state_np["P"][i]))
        b["A"].append(np.asarray(state_np["A"][i]))
        b["F"].append(np.asarray(state_np["F"][i]))
        b["lagrangian_xy"].append(np.asarray(lag_xy_np[i]))
        b["lagrangian_c"].append(np.asarray(lag_ch_np[i]))
        b["resume_batch_rng_key"].append(np.asarray(rng_np[i], dtype=np.uint32))
        for key, value in meta.items():
            b[key].append(np.asarray(value))
        state_t = np.asarray(state_np.get("t", 0))
        mass_cycle_start = np.asarray(state_np.get("mass_cycle_start", np.sum(state_np["A"], axis=tuple(range(1, state_np["A"].ndim)))))
        b["state_t"].append(np.asarray(state_t[i] if state_t.ndim > 0 else state_t, dtype=np.int32))
        b["state_mass_cycle_start"].append(
            np.asarray(mass_cycle_start[i] if mass_cycle_start.ndim > 0 else mass_cycle_start, dtype=np.float32)
        )


def _write_metadata(job: dict[str, Any]) -> None:
    raw = job["raw"]
    metadata = {
        "source_traj_dir": str(job["traj_dir"]),
        "source_apf_path": str(job["apf_path"]),
        "source_snapshot_index": int(job["snapshot_idx"]),
        "start_step": int(job["start_step"]),
        "end_step": int(job["end_step"]),
        "original_batch_size": int(job["original_batch_size"]),
        "original_batch_index": int(job["original_batch_index"]),
        "snapshot_interval": int(job["snapshot_interval"]),
        "jit_microbatch": int(job["jit_microbatch"]),
        "branch_seed": int(raw.get("branch_seed", -1)),
        "perturb_a_std": float(raw.get("perturb_a_std", 0.0)),
        "perturb_p_std": float(raw.get("perturb_p_std", 0.0)),
        "perturb_lagrangian_xy_std": float(raw.get("perturb_lagrangian_xy_std", 0.0)),
        "batched_resume": True,
    }
    write_json(job["output_dir"] / "resume_metadata.json", metadata)


def _process_prepared_batch(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        return
    first = jobs[0]
    args = first["args"]
    substrate = _make_substrate(args)
    _ = substrate.seed_state(jax.random.PRNGKey(0), jnp.asarray(first["params"], dtype=jnp.float32))
    rt = substrate.RT

    states = []
    lag_xys = []
    lag_chs = []
    rngs = []
    params = []
    for job in jobs:
        state = _reconstruct_state(substrate, job["params"], job["snapshot"])
        lag_xy = jnp.asarray(np.asarray(job["snapshot"]["lagrangian_xy"], dtype=np.float32))
        rng = jnp.asarray(np.asarray(job["snapshot"]["resume_batch_rng_key"], dtype=np.uint32))
        state, lag_xy, rng = _apply_resume_perturbation(
            state=state,
            lag_xy=lag_xy,
            rng=rng,
            seed=int(job["raw"].get("branch_seed", -1)),
            a_std=float(job["raw"].get("perturb_a_std", 0.0)),
            p_std=float(job["raw"].get("perturb_p_std", 0.0)),
            lag_xy_std=float(job["raw"].get("perturb_lagrangian_xy_std", 0.0)),
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
        )
        states.append(state)
        lag_xys.append(lag_xy)
        lag_chs.append(jnp.asarray(np.asarray(job["snapshot"]["lagrangian_c"], dtype=np.int32)))
        rngs.append(rng)
        params.append(jnp.asarray(job["params"], dtype=jnp.float32))

    state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
    lag_xy = jnp.stack(lag_xys, axis=0)
    lag_ch = jnp.stack(lag_chs, axis=0)
    rng = jnp.stack(rngs, axis=0)
    params_j = jnp.stack(params, axis=0)
    original_batch_index = jnp.asarray([job["original_batch_index"] for job in jobs], dtype=jnp.int32)

    lag_flow_channel = int(_get(args, "lagrangian_flow_channel", _get(args, "metric_lagrangian_flow_channel", -1)))
    lag_flow_reduce = str(_get(args, "lagrangian_flow_reduce", _get(args, "metric_lagrangian_flow_reduce", "mass_weighted")))
    lag_channel_mode = str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample")))
    lag_noise_model = str(_get(args, "lagrangian_noise_model", _get(args, "metric_lagrangian_noise_model", "rt_box")))
    lag_diffusion_scale = float(_get(args, "lagrangian_diffusion_scale", _get(args, "metric_lagrangian_diffusion_scale", 1.0)))
    stepper = _make_batched_stepper(
        substrate=substrate,
        rt=rt,
        original_batch_size=int(first["original_batch_size"]),
        lag_flow_channel=lag_flow_channel,
        lag_flow_reduce=lag_flow_reduce,
        lag_channel_mode=lag_channel_mode,
        lag_noise_model=lag_noise_model,
        lag_diffusion_scale=lag_diffusion_scale,
    )

    outs = [_new_out(job) for job in jobs]
    metas = [_resume_meta(job) for job in jobs]
    for job in jobs:
        _write_metadata(job)

    horizon_steps = int(first["horizon_steps"])
    snapshot_interval = int(first["snapshot_interval"])
    jit_microbatch = int(first["jit_microbatch"])
    snapshots_per_file = max(1, int(_get(args, "snapshots_per_file", 50)))
    rel_done = 0
    desc = f"batched resume B={len(jobs)} {horizon_steps} steps"
    pbar = tqdm(total=horizon_steps * len(jobs), desc=desc)
    try:
        _capture_apf_only(outs=outs, jobs=jobs, state=state, lag_xy=lag_xy, lag_ch=lag_ch, rng=rng, rel_step=rel_done, resume_meta=metas)
        while rel_done < horizon_steps:
            start0 = int(first["start_step"])
            target_abs = min(start0 + horizon_steps, ((start0 + rel_done) // snapshot_interval + 1) * snapshot_interval)
            target_rel = int(target_abs) - start0
            while rel_done < target_rel:
                n = min(jit_microbatch, target_rel - rel_done)
                split = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
                rng = split[:, 0, :]
                subkey = split[:, 1, :]
                state, lag_xy, lag_ch = stepper(n)(state, lag_xy, lag_ch, subkey, params_j, original_batch_index)
                rel_done += int(n)
                pbar.update(int(n) * len(jobs))
            _capture_apf_only(outs=outs, jobs=jobs, state=state, lag_xy=lag_xy, lag_ch=lag_ch, rng=rng, rel_step=rel_done, resume_meta=metas)
            for out in outs:
                if len(out["buffers"]["steps"]) >= snapshots_per_file:
                    _flush(out, args)
    finally:
        pbar.close()
        for out in outs:
            _flush(out, args)


def _process_jobs(raw_jobs: list[dict[str, Any]], *, batch_size: int, overwrite: bool) -> None:
    batch_size = max(1, int(batch_size))
    total = len(raw_jobs)
    done = 0
    for start in range(0, total, batch_size):
        raw_chunk = raw_jobs[start : start + batch_size]
        prepared = [_prepare_job(raw, overwrite=overwrite) for raw in raw_chunk]
        groups: dict[Any, list[dict[str, Any]]] = {}
        for job in prepared:
            groups.setdefault(job["group_key"], []).append(job)
        for group_jobs in groups.values():
            _process_prepared_batch(group_jobs)
            done += len(group_jobs)
            print(f"[batched-resume] completed {done}/{total} jobs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched Flow-Lenia minibang resume runner.")
    parser.add_argument("--jobs-json", required=True, help="JSON file containing branch resume jobs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum compatible branch jobs per JAX batch.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate output directories if they exist.")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    jobs_path = resolve_path(cli.jobs_json)
    if jobs_path is None or not jobs_path.exists():
        raise FileNotFoundError(f"jobs-json not found: {cli.jobs_json}")
    jobs = _load_jobs(jobs_path)
    print(f"[batched-resume] loaded {len(jobs)} jobs batch_size={int(cli.batch_size)}")
    _process_jobs(jobs, batch_size=int(cli.batch_size), overwrite=bool(cli.overwrite))
    print("[batched-resume] done")


if __name__ == "__main__":
    main()
