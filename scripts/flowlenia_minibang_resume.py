from __future__ import annotations

import argparse
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

from flowlenia_minibang_common import list_apf_chunks, load_config, resolve_path, write_json
from flowlenia_minibang_simulate import _as_bool, _frame_u8, _get, _make_substrate, _write_frame_times


def _find_snapshot(apf_dir: Path, step: int | None) -> tuple[Path, int, int]:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    if step is None:
        path = chunks[-1][0]
        with np.load(path) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            if steps.size == 0:
                raise ValueError(f"APF chunk has no steps: {path}")
            return path, int(steps[-1]), int(steps.size - 1)
    for path, _s0, _s1, _idx in chunks:
        with np.load(path) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            hit = np.flatnonzero(steps == int(step))
            if hit.size:
                return path, int(step), int(hit[0])
    raise ValueError(f"Step {step} not found in APF chunks under {apf_dir}")


def _read_snapshot(path: Path, idx: int) -> dict[str, Any]:
    with np.load(path) as data:
        required = ("A", "P", "lagrangian_xy", "lagrangian_c", "resume_batch_rng_key")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(
                f"{path} is not resume-capable; missing keys: {missing}. "
                "Regenerate APF logs with the updated minibang runner."
            )
        out = {key: np.asarray(data[key][idx]) for key in data.files if np.asarray(data[key]).ndim > 0}
        out["_available_keys"] = list(data.files)
    return out


def _scalar(snapshot: dict[str, Any], key: str, default: Any) -> Any:
    if key not in snapshot:
        return default
    arr = np.asarray(snapshot[key])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _reconstruct_state(substrate: Any, params: np.ndarray, snapshot: dict[str, Any]) -> dict[str, Any]:
    params_j = jnp.asarray(params, dtype=jnp.float32)
    state = dict(substrate.seed_state(jax.random.PRNGKey(0), params_j))
    A_np = np.asarray(snapshot["A"], dtype=np.float32)
    state["A"] = jnp.asarray(A_np)
    state["P"] = jnp.asarray(np.asarray(snapshot["P"], dtype=np.float32))
    if "F" in snapshot:
        state["F"] = jnp.asarray(np.asarray(snapshot["F"], dtype=np.float32))
    else:
        A = state["A"]
        state["F"] = jnp.zeros((A.shape[0], A.shape[1], 2, A.shape[-1]), dtype=A.dtype)
    state["t"] = jnp.asarray(_scalar(snapshot, "state_t", 0), dtype=jnp.int32)
    state["mass_cycle_start"] = jnp.asarray(
        _scalar(snapshot, "state_mass_cycle_start", float(np.sum(A_np))),
        dtype=jnp.float32,
    )
    return state


def _make_stepper(
    *,
    substrate: Any,
    rt: Any,
    params: jax.Array,
    original_batch_size: int,
    original_batch_index: int,
    lag_flow_channel: int,
    lag_flow_reduce: str,
    lag_channel_mode: str,
    lag_noise_model: str,
    lag_diffusion_scale: float,
):
    def step_n(state_in, lag_xy_in, lag_ch_in, rng_in, n_steps: int):
        rngs_batch = jax.random.split(rng_in, int(n_steps) * int(original_batch_size)).reshape(
            (int(n_steps), int(original_batch_size), 2)
        )
        rngs = rngs_batch[:, int(original_batch_index)]

        def scan_body(carry, key_i):
            st, pts, ch = carry
            st_next = substrate.step_state(key_i, st, params)
            lag_key = jax.random.fold_in(key_i, jnp.uint32(0x4C4147))
            pts_next, ch_next = rt.advect_particles(
                points=pts,
                F=st_next["F"],
                A=st_next["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=ch,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return (st_next, pts_next, ch_next), None

        (state_out, lag_xy_out, lag_ch_out), _ = jax.lax.scan(scan_body, (state_in, lag_xy_in, lag_ch_in), rngs)
        return state_out, lag_xy_out, lag_ch_out

    cache: dict[int, Any] = {}

    def get(n_steps: int):
        n = int(n_steps)
        if n not in cache:
            cache[n] = jax.jit(lambda st, xy, ch, rng: step_n(st, xy, ch, rng, n))
        return cache[n]

    return get


def _capture(
    *,
    out: dict[str, Any],
    state: dict[str, Any],
    params: jax.Array,
    lag_xy: jax.Array,
    lag_ch: jax.Array,
    rng: jax.Array,
    step: int,
    substrate: Any,
    args: Any,
    resume_meta: dict[str, Any],
) -> None:
    rgb = jax.device_get(substrate.render_state(state, params, img_size=int(_get(args, "img_size", 224))))
    state_np = jax.device_get(state)
    lag_xy_np, lag_ch_np = jax.device_get((lag_xy, lag_ch))

    frame = _frame_u8(rgb)
    out["writer"].append_data(frame)
    out["frame_rows"].append(
        dict(
            frame_idx=int(out["frame_idx"]),
            step=int(step),
            video_sec=float(out["frame_idx"]) / float(out["video_fps"]),
            sim_sec=float(step) / float(out["sim_fps"]),
        )
    )
    out["frame_idx"] += 1

    b = out["buffers"]
    b["steps"].append(int(step))
    b["P"].append(np.asarray(state_np["P"]))
    b["A"].append(np.asarray(state_np["A"]))
    b["F"].append(np.asarray(state_np["F"]))
    b["lagrangian_xy"].append(np.asarray(lag_xy_np))
    b["lagrangian_c"].append(np.asarray(lag_ch_np))
    b["resume_batch_rng_key"].append(np.asarray(rng, dtype=np.uint32))
    for key, value in resume_meta.items():
        b[key].append(np.asarray(value))
    b["state_t"].append(np.asarray(state_np.get("t", 0), dtype=np.int32))
    b["state_mass_cycle_start"].append(np.asarray(state_np.get("mass_cycle_start", np.sum(state_np["A"])), dtype=np.float32))


def _flush(out: dict[str, Any], args: Any) -> None:
    from simulate_save_apf import save_chunk

    b = out["buffers"]
    if not b["steps"]:
        return
    extra = {
        key: np.asarray(b[key])
        for key in (
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
        if b.get(key)
    }
    out["file_idx"] = save_chunk(
        str(out["apf_dir"]),
        float(out["sim_fps"]),
        b["steps"],
        b["P"],
        int(out["file_idx"]),
        b["A"],
        b["F"],
        use_fp16=True,
        snaps_lagrangian=b["lagrangian_xy"],
        snaps_lagrangian_c=b["lagrangian_c"],
        compress=_as_bool(_get(args, "compress", True), True),
        extra_payload=extra,
    )
    for key in b:
        b[key] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume a minibang trajectory from a resume-capable APF snapshot.")
    parser.add_argument("traj_dir", help="Trajectory directory containing config.yaml, params.npy and apf_logs.")
    parser.add_argument("--step", type=int, default=None, help="APF snapshot step to resume from. Defaults to the last snapshot.")
    parser.add_argument("--end-step", type=int, default=None, help="Absolute simulation step to stop at. Defaults to config rollout/max.")
    parser.add_argument("--additional-steps", type=int, default=None, help="Run this many steps after the APF snapshot.")
    parser.add_argument("--output-dir", default=None, help="Default: <traj_dir>/resume_from_<step>_to_<end_step>.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate output directory if it exists.")
    parser.add_argument("--config", default=None, help="Override config path. Defaults to <traj_dir>/config.yaml.")
    parser.add_argument("--params", default=None, help="Override params path. Defaults to <traj_dir>/params.npy.")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    traj_dir = resolve_path(cli.traj_dir)
    if traj_dir is None or not traj_dir.exists():
        raise FileNotFoundError(f"traj_dir not found: {cli.traj_dir}")
    config_path = resolve_path(cli.config, traj_dir) if cli.config else traj_dir / "config.yaml"
    params_path = resolve_path(cli.params, traj_dir) if cli.params else traj_dir / "params.npy"
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if params_path is None or not params_path.exists():
        raise FileNotFoundError(f"params not found: {params_path}")

    cfg, flat = load_config(config_path)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    args = SimpleNamespace(**flat_args)

    apf_path, start_step, snapshot_idx = _find_snapshot(traj_dir / "apf_logs", cli.step)
    snapshot = _read_snapshot(apf_path, snapshot_idx)
    params = np.asarray(np.load(params_path), dtype=np.float32)

    total_default = int(_get(args, "rollout_steps", _get(args, "max_steps", 0)))
    if _get(args, "max_steps", None) is not None:
        total_default = min(total_default, int(_get(args, "max_steps")))
    end_step = int(cli.end_step) if cli.end_step is not None else int(total_default)
    if cli.additional_steps is not None:
        end_step = int(start_step) + int(cli.additional_steps)
    if end_step <= int(start_step):
        raise ValueError(f"end_step must be > start_step, got {end_step} <= {start_step}")

    output_dir = resolve_path(cli.output_dir, traj_dir) if cli.output_dir else traj_dir / f"resume_from_{start_step}_to_{end_step}"
    assert output_dir is not None
    if output_dir.exists() and cli.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists. Use --overwrite.")
    apf_out = output_dir / "apf_logs"
    apf_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    np.save(output_dir / "params.npy", params)

    substrate = _make_substrate(args)
    _ = substrate.seed_state(jax.random.PRNGKey(0), jnp.asarray(params, dtype=jnp.float32))
    rt = substrate.RT
    state = _reconstruct_state(substrate, params, snapshot)
    lag_xy = jnp.asarray(np.asarray(snapshot["lagrangian_xy"], dtype=np.float32))
    lag_ch = jnp.asarray(np.asarray(snapshot["lagrangian_c"], dtype=np.int32))
    rng = jnp.asarray(np.asarray(snapshot["resume_batch_rng_key"], dtype=np.uint32))
    params_j = jnp.asarray(params, dtype=jnp.float32)

    original_batch_size = int(_scalar(snapshot, "resume_batch_size", 1))
    original_batch_index = int(_scalar(snapshot, "resume_batch_index", 0))
    snapshot_interval = int(_scalar(snapshot, "resume_snapshot_interval", _get(args, "snapshot_interval", 100)))
    jit_microbatch = int(_scalar(snapshot, "resume_jit_microbatch", _get(args, "jit_microbatch", snapshot_interval)))
    lag_flow_channel = int(_get(args, "lagrangian_flow_channel", _get(args, "metric_lagrangian_flow_channel", -1)))
    lag_flow_reduce = str(_get(args, "lagrangian_flow_reduce", _get(args, "metric_lagrangian_flow_reduce", "mass_weighted")))
    lag_channel_mode = str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample")))
    lag_noise_model = str(_get(args, "lagrangian_noise_model", _get(args, "metric_lagrangian_noise_model", "rt_box")))
    lag_diffusion_scale = float(_get(args, "lagrangian_diffusion_scale", _get(args, "metric_lagrangian_diffusion_scale", 1.0)))

    stepper = _make_stepper(
        substrate=substrate,
        rt=rt,
        params=params_j,
        original_batch_size=original_batch_size,
        original_batch_index=original_batch_index,
        lag_flow_channel=lag_flow_channel,
        lag_flow_reduce=lag_flow_reduce,
        lag_channel_mode=lag_channel_mode,
        lag_noise_model=lag_noise_model,
        lag_diffusion_scale=lag_diffusion_scale,
    )

    import imageio

    video_fps = float(_get(args, "video_fps", 30.0))
    sim_fps = float(_get(args, "fps", 250.0))
    out = dict(
        apf_dir=apf_out,
        writer=imageio.get_writer(
            str(output_dir / "video.mp4"),
            fps=video_fps,
            codec=str(_get(args, "codec", "libx264")),
            macro_block_size=_get(args, "macro_block_size", 1),
        ),
        file_idx=0,
        frame_idx=0,
        sim_fps=sim_fps,
        video_fps=video_fps,
        frame_rows=[],
        buffers={key: [] for key in (
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
        )},
    )
    resume_meta = {
        "resume_batch_size": np.asarray(original_batch_size, dtype=np.int32),
        "resume_batch_index": np.asarray(original_batch_index, dtype=np.int32),
        "resume_selection0": np.asarray(_scalar(snapshot, "resume_selection0", 0), dtype=np.int32),
        "resume_jit_microbatch": np.asarray(jit_microbatch, dtype=np.int32),
        "resume_snapshot_interval": np.asarray(snapshot_interval, dtype=np.int32),
        "resume_seed": np.asarray(_scalar(snapshot, "resume_seed", _get(args, "seed", 0)), dtype=np.int64),
        "resume_lagrangian_seed": np.asarray(_scalar(snapshot, "resume_lagrangian_seed", _get(args, "lagrangian_seed", _get(args, "seed", 0))), dtype=np.int64),
    }

    metadata = {
        "source_traj_dir": str(traj_dir),
        "source_apf_path": str(apf_path),
        "source_snapshot_index": int(snapshot_idx),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "original_batch_size": int(original_batch_size),
        "original_batch_index": int(original_batch_index),
        "snapshot_interval": int(snapshot_interval),
        "jit_microbatch": int(jit_microbatch),
    }
    write_json(output_dir / "resume_metadata.json", metadata)

    steps_done = int(start_step)
    pbar = tqdm(total=end_step - steps_done, desc=f"resume {traj_dir.name} {start_step}->{end_step}")
    try:
        _capture(out=out, state=state, params=params_j, lag_xy=lag_xy, lag_ch=lag_ch, rng=rng, step=steps_done, substrate=substrate, args=args, resume_meta=resume_meta)
        while steps_done < end_step:
            target_next_snapshot = min(end_step, ((steps_done // snapshot_interval) + 1) * snapshot_interval)
            while steps_done < target_next_snapshot:
                n = min(jit_microbatch, target_next_snapshot - steps_done)
                rng, subkey = jax.random.split(rng)
                state, lag_xy, lag_ch = stepper(n)(state, lag_xy, lag_ch, subkey)
                steps_done += n
                pbar.update(n)
            _capture(out=out, state=state, params=params_j, lag_xy=lag_xy, lag_ch=lag_ch, rng=rng, step=steps_done, substrate=substrate, args=args, resume_meta=resume_meta)
            if len(out["buffers"]["steps"]) >= max(1, int(_get(args, "snapshots_per_file", 50))):
                _flush(out, args)
    finally:
        pbar.close()
        _flush(out, args)
        out["writer"].close()
        _write_frame_times(output_dir / "frame_times.csv", out["frame_rows"])

    print(f"Done. Resumed trajectory written to {output_dir}")


if __name__ == "__main__":
    main()
