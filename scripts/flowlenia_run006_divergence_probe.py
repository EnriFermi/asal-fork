from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_COMMIT = "2e2152ff6d56481d922804a74b90556c39ce94cc"
DEFAULT_RUN_DIR = (
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization/run_006"
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _activate_source_root(source_root: Path) -> None:
    source_root = source_root.resolve()
    source_scripts = source_root / "scripts"
    this_root = Path(__file__).resolve().parent.parent
    this_scripts = this_root / "scripts"
    remove = {str(this_root), str(this_scripts)}
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) not in remove]
    for path in (str(source_scripts), str(source_root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _run_text(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None
    output = result.stdout.strip()
    return output or None


def _environment_payload(source_root: Path) -> dict[str, Any]:
    import jax
    import jaxlib

    package_names = (
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "equinox",
        "evosax",
        "omegaconf",
    )
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    source_files = (
        "scripts/main_opt_msc.py",
        "scripts/util.py",
        "scripts/clip_deltah_msc_metric.py",
        "substrates/lenia_flow/lenia_flow.py",
        "substrates/lenia_flow/reintegration_tracking.py",
    )
    source_sha256 = {
        name: _sha256_file(source_root / name)
        for name in source_files
        if (source_root / name).exists()
    }
    env_prefixes = ("JAX_", "XLA_", "CUDA_", "NVIDIA_", "TF_")
    selected_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(env_prefixes)
    }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "source_root": str(source_root),
        "git_head": _run_text(["git", "rev-parse", "HEAD"], cwd=source_root),
        "git_status": _run_text(["git", "status", "--short"], cwd=source_root),
        "source_sha256": source_sha256,
        "nvidia_smi": _run_text(["nvidia-smi", "-q"]),
        "ptxas_version": _run_text(["ptxas", "--version"]),
        "environment": selected_env,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
    }


def _flat_config(path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    # This deliberately reproduces commit 2e2152f's legacy sigma collision.
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )


def _init_lagrangian_points_jax(
    A0,
    *,
    n_particles: int,
    init_mode: str,
    border: str,
    sigma: float,
    key,
):
    import jax
    import jax.numpy as jnp

    sx, sy = int(A0.shape[0]), int(A0.shape[1])
    init_mode = str(init_mode).strip().lower()
    if init_mode == "uniform":
        k0, k1 = jax.random.split(key)
        y = jax.random.uniform(k0, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(k1, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif init_mode == "mass":
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(
            total > 0.0,
            flat / jnp.maximum(total, 1e-12),
            jnp.ones_like(flat) / flat.size,
        )
        k_idx, k_jit = jax.random.split(key)
        idx = jax.random.choice(k_idx, flat.size, shape=(n_particles,), replace=True, p=probs)
        iy, ix = idx // sy, idx % sy
        jitter = jax.random.uniform(k_jit, (n_particles, 2), minval=-0.49, maxval=0.49)
        pts = jnp.stack((iy.astype(jnp.float32) + 0.5, ix.astype(jnp.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unsupported lagrangian init mode: {init_mode!r}")

    if border == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
    else:
        y = jnp.clip(pts[:, 0], float(sigma), float(sx - sigma))
        x = jnp.clip(pts[:, 1], float(sigma), float(sy - sigma))
    return jnp.stack((y, x), axis=-1).astype(jnp.float32)


def _build_evaluators(flat, *, trace_candidate: int, trace_seed_index: int):
    import jax
    import jax.numpy as jnp
    from jax.random import split

    import substrates
    import util
    from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config

    base_substrate = substrates.create_substrate(
        flat.substrate,
        **util.substrate_kwargs_from_args(flat),
    )
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    substrate = substrates.FlattenSubstrateParameters(base_substrate)
    if getattr(flat, "rollout_steps", None) is None:
        flat.rollout_steps = substrate.rollout_steps

    defaults = util.metric_periodic_space_defaults(base_substrate)
    if getattr(flat, "metric_periodic", None) is None:
        flat.metric_periodic = bool(defaults["periodic"])
    if getattr(flat, "metric_domain_y", None) is None:
        flat.metric_domain_y = float(defaults["domain_y"])
    if getattr(flat, "metric_domain_x", None) is None:
        flat.metric_domain_x = float(defaults["domain_x"])

    metric_cfg = resolve_metric_config(flat)
    optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
    metric_fn = make_metric_loss_fn(metric_cfg, include_maps=False)
    metric_trace_fn = make_metric_loss_fn(metric_cfg, include_maps=True)
    substrate_param_dims = int(substrate.n_params)
    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(flat, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(flat, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(flat, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(flat, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(flat, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(flat, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(flat, "metric_lagrangian_diffusion_scale", 1.0))

    def split_candidate(params_full):
        params = params_full[:substrate_param_dims]
        tau = params_full[substrate_param_dims] if optimize_tau else None
        return params, tau

    def rollout_xy(rng, params):
        k_state, k_pts, k_ch, k_scan = split(rng, 4)
        state0 = substrate.init_state(k_state, params)
        rt = substrate.RT
        pts0 = _init_lagrangian_points_jax(
            state0["A"],
            n_particles=lag_n_particles,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            channels0 = rt.sample_point_channels(pts0, state0["A"], k_ch)
        else:
            channels0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)

        def step_fn(carry, key_step):
            state, points, channels = carry
            state = substrate.step_state(key_step, state, params)
            lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
            points, channels = rt.advect_particles(
                points=points,
                F=state["F"],
                A=state["A"],
                channel=lag_flow_channel,
                reduce=lag_flow_reduce,
                point_channels=channels,
                channel_mode=lag_channel_mode,
                key=lag_key,
                noise_model=lag_noise_model,
                diffusion_scale=lag_diffusion_scale,
            )
            return (state, points, channels), None

        def chunk_fn(carry, key_chunk):
            next_carry, _ = jax.lax.scan(step_fn, carry, split(key_chunk, chunk_steps))
            return next_carry, next_carry[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (state0, pts0, channels0),
            split(k_scan, time_sampling),
        )
        return xy_seq

    def calc_loss(rng, params_full, *, include_maps: bool):
        params, tau = split_candidate(params_full)
        rng_roll, rng_metric = split(rng)
        xy_seq = rollout_xy(rng_roll, params)
        fn = metric_trace_fn if include_maps else metric_fn
        if optimize_tau:
            loss, info = fn(rng_metric, xy_seq, tau_selector=tau)
        else:
            loss, info = fn(rng_metric, xy_seq)
        return loss, info, xy_seq

    def calc_loss_no_trace(rng, params_full):
        loss, info, _xy_seq = calc_loss(rng, params_full, include_maps=False)
        return loss, info

    def calc_loss_with_xy(rng, params_full):
        loss, info, xy_seq = calc_loss(rng, params_full, include_maps=False)
        return loss, info, xy_seq

    calc_loss_vv = jax.vmap(jax.vmap(calc_loss_no_trace, in_axes=(0, None)), in_axes=(None, 0))
    calc_loss_xy_vv = jax.vmap(jax.vmap(calc_loss_with_xy, in_axes=(0, None)), in_axes=(None, 0))

    @jax.jit
    def full_eval_with_selected_xy(params_full, seed_keys):
        loss, info, xy_all = calc_loss_xy_vv(seed_keys, params_full)
        return loss, info, xy_all[trace_candidate, trace_seed_index]

    @jax.jit
    def trace_metric_from_xy(params_full, seed_key, xy_seq):
        _rng_roll, rng_metric = split(seed_key)
        _params, tau = split_candidate(params_full)
        if optimize_tau:
            return metric_trace_fn(rng_metric, xy_seq, tau_selector=tau)
        return metric_trace_fn(rng_metric, xy_seq)

    return full_eval_with_selected_xy, trace_metric_from_xy, metric_cfg, substrate_param_dims, optimize_tau


def _params_full(pop: dict[str, Any], i_iter: int, substrate_dims: int, optimize_tau: bool) -> np.ndarray:
    params = np.asarray(pop["params"], dtype=np.float32)[i_iter]
    if not optimize_tau:
        return params
    if params.shape[1] == substrate_dims + 1:
        return params
    tau = np.asarray(pop["tau_selector_raw"], dtype=np.float32)[i_iter]
    return np.concatenate((params, tau[:, None]), axis=1).astype(np.float32)


def capture(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _activate_source_root(source_root)

    import jax

    config_path = run_dir / "optimization_config.yaml"
    pop = _load_pickle(run_dir / "pop_traj.pkl")
    flat = _flat_config(config_path)
    full_eval, trace_metric_from_xy, metric_cfg, substrate_dims, optimize_tau = _build_evaluators(
        flat,
        trace_candidate=args.candidate,
        trace_seed_index=args.seed_index,
    )
    params_full = _params_full(pop, args.iter, substrate_dims, optimize_tau)
    seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)[args.iter]
    selected_params = params_full[args.candidate]
    selected_key = seed_keys[args.seed_index]

    np.savez(
        output_dir / "inputs.npz",
        params_full=params_full,
        seed_keys=seed_keys,
        selected_params=selected_params,
        selected_key=selected_key,
    )
    _write_json(output_dir / "environment.json", _environment_payload(source_root))

    full_started = time.monotonic()
    print("[probe] phase 1/2: full 8x4 evaluation with selected xy started", flush=True)
    loss_by_seed, info_by_seed, xy_seq = full_eval(params_full, seed_keys)
    loss_np = np.asarray(jax.device_get(loss_by_seed), dtype=np.float32)
    print(
        f"[probe] phase 1/2 complete after {time.monotonic() - full_started:.1f}s",
        flush=True,
    )
    info_np = jax.device_get(info_by_seed)
    full_arrays: dict[str, np.ndarray] = {"loss_by_seed": loss_np, "score_by_seed": -loss_np}
    if isinstance(info_np, dict):
        for key, value in info_np.items():
            full_arrays[f"info__{key}"] = np.asarray(value)
    np.savez(output_dir / "full_eval.npz", **full_arrays)

    metric_started = time.monotonic()
    print("[probe] phase 2/2: selected metric maps and artifact write started", flush=True)
    trace_loss, trace_info = trace_metric_from_xy(selected_params, selected_key, xy_seq)
    trace_loss_np = np.asarray(jax.device_get(trace_loss), dtype=np.float32)
    trace_info_np = jax.device_get(trace_info)
    xy_np = np.asarray(jax.device_get(xy_seq), dtype=np.float32)
    np.save(output_dir / "trace_xy.npy", xy_np, allow_pickle=False)
    trace_arrays: dict[str, np.ndarray] = {"loss": trace_loss_np}
    if isinstance(trace_info_np, dict):
        for key, value in trace_info_np.items():
            trace_arrays[key] = np.asarray(value)
    np.savez(output_dir / "trace_metric.npz", **trace_arrays)

    stored_scores = np.asarray(pop["score_by_seed"], dtype=np.float32)[args.iter]
    selected_recomputed = float(-loss_np[args.candidate, args.seed_index])
    selected_stored = float(stored_scores[args.candidate, args.seed_index])
    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "source_root": str(source_root),
        "source_commit_expected": args.source_commit,
        "iter": int(args.iter),
        "candidate": int(args.candidate),
        "seed_index": int(args.seed_index),
        "selected_seed_key": [int(x) for x in selected_key],
        "selected_stored_score": selected_stored,
        "selected_recomputed_score": selected_recomputed,
        "selected_score_diff": selected_recomputed - selected_stored,
        "full_score_max_abs_diff_vs_stored": float(np.max(np.abs((-loss_np) - stored_scores))),
        "params_sha256": _sha256_bytes(selected_params),
        "seed_key_sha256": _sha256_bytes(selected_key),
        "xy_shape": [int(x) for x in xy_np.shape],
        "xy_sha256": _sha256_bytes(xy_np),
        "trace_execution_shape": "selected xy returned from full 8x4 vmap",
        "metric_objective": str(metric_cfg.get("objective")),
        "effective_legacy_flow_sigma": float(flat.sigma),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        f"[probe] phase 2/2 complete after {time.monotonic() - metric_started:.1f}s",
        flush=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _compare_npz(path_a: Path, path_b: Path) -> dict[str, Any]:
    a, b = np.load(path_a), np.load(path_b)
    keys_a, keys_b = set(a.files), set(b.files)
    fields: dict[str, Any] = {}
    for key in sorted(keys_a & keys_b):
        av, bv = np.asarray(a[key]), np.asarray(b[key])
        if av.shape != bv.shape:
            fields[key] = {"shape_a": list(av.shape), "shape_b": list(bv.shape)}
            continue
        equal = bool(np.array_equal(av, bv))
        item: dict[str, Any] = {"equal": equal, "shape": list(av.shape)}
        if not equal and np.issubdtype(av.dtype, np.number):
            diff = np.abs(av.astype(np.float64) - bv.astype(np.float64))
            item["max_abs_diff"] = float(np.nanmax(diff))
            item["different_values"] = int(np.count_nonzero(av != bv))
        fields[key] = item
    return {
        "only_a": sorted(keys_a - keys_b),
        "only_b": sorted(keys_b - keys_a),
        "fields": fields,
    }


def _compare_xy(path_a: Path, path_b: Path, block_frames: int = 32) -> dict[str, Any]:
    a = np.load(path_a, mmap_mode="r")
    b = np.load(path_b, mmap_mode="r")
    if a.shape != b.shape:
        return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    first_frame: int | None = None
    max_abs = 0.0
    different_values = 0
    for start in range(0, a.shape[0], block_frames):
        end = min(a.shape[0], start + block_frames)
        av = np.asarray(a[start:end])
        bv = np.asarray(b[start:end])
        neq = av != bv
        count = int(np.count_nonzero(neq))
        if count:
            if first_frame is None:
                frame_has_diff = np.any(neq.reshape((end - start, -1)), axis=1)
                first_frame = int(start + np.flatnonzero(frame_has_diff)[0])
            different_values += count
            max_abs = max(max_abs, float(np.max(np.abs(av.astype(np.float64) - bv.astype(np.float64)))))
    return {
        "equal": first_frame is None,
        "shape": [int(x) for x in a.shape],
        "first_different_sample_frame": first_frame,
        "first_different_nominal_step": None if first_frame is None else int((first_frame + 1) * 50),
        "different_values": different_values,
        "max_abs_diff": max_abs,
    }


def compare(args: argparse.Namespace) -> int:
    dir_a, dir_b = Path(args.capture_a).resolve(), Path(args.capture_b).resolve()
    inputs = _compare_npz(dir_a / "inputs.npz", dir_b / "inputs.npz")
    full_eval = _compare_npz(dir_a / "full_eval.npz", dir_b / "full_eval.npz")
    trace_metric = _compare_npz(dir_a / "trace_metric.npz", dir_b / "trace_metric.npz")
    xy = _compare_xy(dir_a / "trace_xy.npy", dir_b / "trace_xy.npy")
    env_a = json.loads((dir_a / "environment.json").read_text())
    env_b = json.loads((dir_b / "environment.json").read_text())

    input_fields = inputs["fields"]
    inputs_equal = all(item.get("equal", False) for item in input_fields.values())
    score_field = full_eval["fields"].get("score_by_seed", {})
    scores_equal = bool(score_field.get("equal", False))
    metric_equal = all(
        item.get("equal", False)
        for item in trace_metric["fields"].values()
        if "equal" in item
    )
    if not inputs_equal:
        verdict = "INPUT_MISMATCH"
    elif not xy.get("equal", False):
        verdict = "ROLLOUT_DIVERGENCE"
    elif not metric_equal:
        verdict = "METRIC_DIVERGENCE_WITH_IDENTICAL_XY"
    elif not scores_equal:
        verdict = "FULL_BATCH_ONLY_DIVERGENCE"
    else:
        verdict = "BITWISE_MATCH"

    environment_differences = {
        key: {"a": env_a.get(key), "b": env_b.get(key)}
        for key in ("hostname", "platform", "python", "packages", "jax_devices", "ptxas_version")
        if env_a.get(key) != env_b.get(key)
    }
    result = {
        "verdict": verdict,
        "capture_a": str(dir_a),
        "capture_b": str(dir_b),
        "inputs": inputs,
        "full_eval": full_eval,
        "trace_xy": xy,
        "trace_metric": trace_metric,
        "environment_differences": environment_differences,
    }
    output = Path(args.output).resolve()
    _write_json(output, result)
    print(json.dumps({
        "verdict": verdict,
        "trace_xy": xy,
        "score_by_seed": score_field,
        "environment_differences": environment_differences,
        "full_report": str(output),
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture and compare exact Flow-Lenia run_006 divergence probes.")
    sub = parser.add_subparsers(dest="command", required=True)

    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    capture_parser.add_argument("--source-root", required=True)
    capture_parser.add_argument("--source-commit", default=DEFAULT_COMMIT)
    capture_parser.add_argument("--output-dir", required=True)
    capture_parser.add_argument("--iter", type=int, default=0)
    capture_parser.add_argument("--candidate", type=int, default=2)
    capture_parser.add_argument("--seed-index", type=int, default=0)
    capture_parser.set_defaults(func=capture)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("capture_a")
    compare_parser.add_argument("capture_b")
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
