from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import flowlenia_run006_divergence_probe as base_probe


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hlo_text(lowered: Any) -> tuple[str, str]:
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_module = lowered.compiler_ir(dialect="hlo")
    if hasattr(hlo_module, "as_hlo_text"):
        hlo = hlo_module.as_hlo_text()
    else:
        hlo = str(hlo_module)
    return stablehlo, hlo


def _build_staged_window_probe(metric_cfg: dict[str, Any], tau_index: int, window_index: int):
    import jax
    import jax.numpy as jnp

    starts = jnp.asarray(metric_cfg["starts"], dtype=jnp.int32)
    win = int(metric_cfg["window_size_frames"])
    tau_frames_list = [int(x) for x in metric_cfg["tau_frames_list"]]
    tseg_list = [int(x) for x in metric_cfg["tseg_list"]]
    m_count_list = [int(x) for x in metric_cfg["m_count_list"]]
    tau_count = len(tau_frames_list)
    W = int(metric_cfg["W"])
    tau = tau_frames_list[tau_index]
    tseg = tseg_list[tau_index]
    m_count = m_count_list[tau_index]
    particle_samples = int(metric_cfg["particle_samples"])
    null_reps = int(metric_cfg["null_reps"])
    n_proj = int(metric_cfg["n_proj"])
    sample_stride_steps = float(metric_cfg["sample_stride_steps"])
    dirs_seed = int(metric_cfg["dirs_seed"])
    periodic = bool(metric_cfg["periodic"])
    positions_unwrapped = bool(metric_cfg.get("positions_unwrapped", False))
    domain_y = float(metric_cfg["domain_y"])
    domain_x = float(metric_cfg["domain_x"])

    if not (0 <= tau_index < tau_count):
        raise ValueError(f"tau_index must be in [0, {tau_count}), got {tau_index}")
    if not (0 <= window_index < W):
        raise ValueError(f"window_index must be in [0, {W}), got {window_index}")

    dir_key = jax.random.PRNGKey(dirs_seed)

    def normalize_dirs(dtype):
        raw = jax.random.normal(dir_key, (n_proj, 2), dtype=dtype)
        normalized = raw / jnp.maximum(jnp.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        return raw, normalized

    def signature(v_s, dirs):
        proj = jnp.einsum("msd,ld->msl", v_s, dirs)
        proj_sorted = jnp.sort(proj, axis=0)
        sig = jnp.transpose(proj_sorted, (1, 2, 0)).reshape(v_s.shape[1], -1)
        return proj, proj_sorted, sig

    def pairwise_l1(sig):
        n = sig.shape[0]
        d = jnp.mean(jnp.abs(sig[:, None, :] - sig[None, :, :]), axis=2)
        mask = jnp.triu(jnp.ones((n, n), dtype=sig.dtype), k=1)
        denom = jnp.array(n * (n - 1) // 2, dtype=sig.dtype)
        value = jnp.sum(d * mask) / jnp.maximum(denom, jnp.array(1.0, dtype=sig.dtype))
        return d, value

    @jax.jit
    def staged(rng_metric, xy_seq):
        keys_tau = jax.random.split(rng_metric, tau_count)
        keys_w = jax.random.split(keys_tau[tau_index], W)
        key_k, key_p, key_null = jax.random.split(keys_w[window_index], 3)
        start = starts[window_index]
        X_w = jax.lax.dynamic_slice(
            xy_seq,
            (start, 0, 0),
            (win, xy_seq.shape[1], 2),
        )
        n_particles = X_w.shape[1]
        s_count = min(particle_samples, n_particles)

        if m_count >= tseg:
            k_idx = jnp.arange(m_count, dtype=jnp.int32)
        else:
            k_idx = jnp.sort(jax.random.choice(key_k, tseg, shape=(m_count,), replace=False))
        if s_count >= n_particles:
            p_idx = jnp.arange(n_particles, dtype=jnp.int32)
        else:
            p_idx = jnp.sort(
                jax.random.choice(key_p, n_particles, shape=(s_count,), replace=False)
            )

        X0 = X_w[k_idx][:, p_idx, :]
        X1 = X_w[k_idx + tau][:, p_idx, :]
        dx = X1 - X0
        if periodic and not positions_unwrapped:
            if domain_y > 0:
                dy = (dx[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
                dx = dx.at[..., 0].set(dy)
            if domain_x > 0:
                ddx = (dx[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
                dx = dx.at[..., 1].set(ddx)
        dt = jnp.maximum(
            jnp.asarray(float(tau) * sample_stride_steps, dtype=xy_seq.dtype),
            jnp.asarray(1e-12, dtype=xy_seq.dtype),
        )
        v_s = dx / dt

        dirs_raw, dirs = normalize_dirs(xy_seq.dtype)
        proj, proj_sorted, sig = signature(v_s, dirs)
        pairwise_real, h_real = pairwise_l1(sig)

        pool = v_s.reshape((-1, 2))
        pool_n = pool.shape[0]
        null_keys = jax.random.split(key_null, null_reps)

        def one_null(key):
            idx = jax.random.randint(key, (m_count, s_count), 0, pool_n)
            v0 = pool[idx]
            proj0, proj0_sorted, sig0 = signature(v0, dirs)
            pairwise0, h0 = pairwise_l1(sig0)
            return idx, proj0, proj0_sorted, sig0, pairwise0, h0

        null_idx, null_proj, null_proj_sorted, null_sig, null_pairwise, h0 = jax.vmap(
            one_null
        )(null_keys)
        h_null = jnp.median(h0)
        return {
            "dir_key": dir_key,
            "dirs_raw": dirs_raw,
            "dirs": dirs,
            "key_k": key_k,
            "key_p": key_p,
            "key_null": key_null,
            "k_idx": k_idx,
            "p_idx": p_idx,
            "X0": X0,
            "X1": X1,
            "dx": dx,
            "dt": dt,
            "v_s": v_s,
            "proj": proj,
            "proj_sorted": proj_sorted,
            "sig": sig,
            "pairwise_real": pairwise_real,
            "h_real": h_real,
            "null_keys": null_keys,
            "null_idx": null_idx,
            "null_proj": null_proj,
            "null_proj_sorted": null_proj_sorted,
            "null_sig": null_sig,
            "null_pairwise": null_pairwise,
            "h0": h0,
            "h_null": h_null,
            "delta_h": h_real - h_null,
        }

    @jax.jit
    def dirs_only():
        return normalize_dirs(jnp.float32)

    @jax.jit
    def projection_only(v_s, dirs):
        return signature(v_s, dirs)

    @jax.jit
    def pairwise_only(sig):
        return pairwise_l1(sig)

    return staged, dirs_only, projection_only, pairwise_only


def capture(args: argparse.Namespace) -> int:
    import jax

    source_root = Path(args.source_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    capture_dir = Path(args.capture_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_probe._activate_source_root(source_root)

    flat = base_probe._flat_config(run_dir / "optimization_config.yaml")
    _, trace_metric_from_xy, metric_cfg, _, _ = base_probe._build_evaluators(
        flat,
        trace_candidate=int(args.candidate),
        trace_seed_index=int(args.seed_index),
    )

    with np.load(capture_dir / "inputs.npz", allow_pickle=False) as inputs:
        selected_params = np.asarray(inputs["selected_params"], dtype=np.float32)
        selected_key = np.asarray(inputs["selected_key"], dtype=np.uint32)
    xy_host = np.load(capture_dir / "trace_xy.npy", mmap_mode="r", allow_pickle=False)
    xy = jax.device_put(np.asarray(xy_host))
    params = jax.device_put(selected_params)
    seed_key = jax.device_put(selected_key)

    print("[forensics] lowering exact metric", flush=True)
    lowered = trace_metric_from_xy.lower(params, seed_key, xy)
    stablehlo, hlo = _hlo_text(lowered)
    stablehlo = stablehlo.replace(str(source_root), "<SOURCE_ROOT>")
    hlo = hlo.replace(str(source_root), "<SOURCE_ROOT>")
    stablehlo_path = output_dir / "exact_metric.stablehlo.mlir"
    hlo_path = output_dir / "exact_metric.hlo.txt"
    stablehlo_path.write_text(stablehlo)
    hlo_path.write_text(hlo)
    exact_loss_value: float | None = None
    if not args.skip_exact_execution:
        compiled = lowered.compile()
        print("[forensics] executing exact metric on saved xy", flush=True)
        exact_loss, exact_info = jax.device_get(compiled(params, seed_key, xy))
        exact_arrays = {"loss": np.asarray(exact_loss)}
        exact_arrays.update({key: np.asarray(value) for key, value in exact_info.items()})
        np.savez(output_dir / "exact_metric.npz", **exact_arrays)
        exact_loss_value = float(np.asarray(exact_loss))

    _rng_roll, rng_metric = jax.random.split(seed_key)
    staged, dirs_only, projection_only, pairwise_only = _build_staged_window_probe(
        metric_cfg,
        int(args.tau_index),
        int(args.window_index),
    )
    print("[forensics] executing staged first-window metric", flush=True)
    staged_result = {
        key: np.asarray(value)
        for key, value in jax.device_get(staged(rng_metric, xy)).items()
    }
    np.savez(output_dir / "staged_window.npz", **staged_result)

    dirs_raw_only, dirs_only_value = jax.device_get(dirs_only())
    proj_only, proj_sorted_only, sig_only = jax.device_get(
        projection_only(
            jax.device_put(staged_result["v_s"]),
            jax.device_put(staged_result["dirs"]),
        )
    )
    pairwise_only_value, h_real_only = jax.device_get(
        pairwise_only(jax.device_put(staged_result["sig"]))
    )
    np.savez(
        output_dir / "isolated_primitives.npz",
        dirs_raw=np.asarray(dirs_raw_only),
        dirs=np.asarray(dirs_only_value),
        proj=np.asarray(proj_only),
        proj_sorted=np.asarray(proj_sorted_only),
        sig=np.asarray(sig_only),
        pairwise_real=np.asarray(pairwise_only_value),
        h_real=np.asarray(h_real_only),
    )

    with np.load(capture_dir / "trace_metric.npz", allow_pickle=False) as baseline:
        baseline_delta = float(baseline["delta_h_map"][args.tau_index, args.window_index])
        baseline_loss = float(baseline["loss"])
    staged_delta = float(staged_result["delta_h"])
    summary = {
        "status": "complete",
        "source_root": str(source_root),
        "capture_dir": str(capture_dir),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "stablehlo_sha256": _sha256_file(stablehlo_path),
        "hlo_sha256": _sha256_file(hlo_path),
        "xy_sha256": _sha256_array(np.asarray(xy_host)),
        "tau_index": int(args.tau_index),
        "window_index": int(args.window_index),
        "baseline_loss": baseline_loss,
        "exact_loss": exact_loss_value,
        "exact_loss_matches_baseline": None
        if exact_loss_value is None
        else bool(
            np.asarray(exact_loss_value, dtype=np.float32)
            == np.asarray(baseline_loss, dtype=np.float32)
        ),
        "baseline_delta_h": baseline_delta,
        "staged_delta_h": staged_delta,
        "staged_delta_matches_baseline": bool(
            np.asarray(staged_delta, dtype=np.float32)
            == np.asarray(baseline_delta, dtype=np.float32)
        ),
        "dirs_staged_match_isolated": bool(
            np.array_equal(staged_result["dirs"], np.asarray(dirs_only_value))
        ),
        "projection_staged_matches_isolated": bool(
            np.array_equal(staged_result["proj"], np.asarray(proj_only))
        ),
        "pairwise_staged_matches_isolated": bool(
            np.array_equal(staged_result["pairwise_real"], np.asarray(pairwise_only_value))
        ),
        "h_real_staged_matches_isolated": bool(
            np.array_equal(staged_result["h_real"], np.asarray(h_real_only))
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forensic metric-only JIT probe for the run_006 A100/H100 divergence."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", type=int, default=2)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--tau-index", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--skip-exact-execution", action="store_true")
    return capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
