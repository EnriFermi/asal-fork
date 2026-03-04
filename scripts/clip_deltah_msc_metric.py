from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _maybe_list(x: Any) -> list[Any] | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        return [p.strip() for p in s.split(",") if p.strip()]
    if isinstance(x, Sequence):
        return list(x)
    return None


def _resolve_frames(
    *,
    frames: Any,
    steps: Any,
    sample_stride_steps: int,
    name: str,
) -> int:
    if frames is not None:
        val = int(frames)
    elif steps is not None:
        val = int(max(1, round(float(steps) / float(sample_stride_steps))))
    else:
        raise ValueError(f"{name}: either *_frames or *_steps must be provided.")
    if val < 1:
        raise ValueError(f"{name} must be >= 1, got {val}.")
    return val


def _resolve_optional_steps(x: Any, *, name: str) -> int | None:
    if x is None:
        return None
    val = int(x)
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}.")
    return val


def _resolve_scales(
    *,
    W: int,
    scales_raw: Any,
    weights_raw: Any,
) -> tuple[list[int], dict[int, float], list[tuple[int, float]]]:
    scales_list = _maybe_list(scales_raw)
    if scales_list is None or len(scales_list) == 0:
        scales: list[int] = []
        r = 1
        while r <= (W // 2):
            scales.append(r)
            r *= 2
    else:
        scales = sorted({int(x) for x in scales_list if int(x) > 0})

    if not scales:
        return [], {}, []

    weight_map: dict[int, float] = {r: 1.0 for r in scales}
    if weights_raw is not None:
        if isinstance(weights_raw, Mapping):
            for k, v in weights_raw.items():
                rk = int(k)
                if rk in weight_map:
                    weight_map[rk] = float(v)
        else:
            w_list = _maybe_list(weights_raw)
            if w_list is None:
                weight_scalar = float(weights_raw)
                for r in scales:
                    weight_map[r] = weight_scalar
            elif len(w_list) == len(scales):
                for r, w in zip(scales, w_list):
                    weight_map[r] = float(w)
            elif len(w_list) == 1:
                weight_scalar = float(w_list[0])
                for r in scales:
                    weight_map[r] = weight_scalar
            else:
                raise ValueError(
                    "metric_scale_weights must be either a scalar, a dict, "
                    f"or a list of length len(metric_scales)={len(scales)}."
                )

    scales_set = set(scales)
    pairs: list[tuple[int, float]] = []
    for r in scales:
        if (2 * r) in scales_set:
            pairs.append((int(r), float(weight_map[r])))
    return scales, weight_map, pairs


def resolve_metric_config(args: Any) -> dict[str, Any]:
    rollout_steps = int(args.rollout_steps)
    sample_every_raw = getattr(args, "sample_every_steps", None)
    time_sampling_raw = getattr(args, "time_sampling", None)

    if sample_every_raw is None and time_sampling_raw is None:
        raise ValueError(
            "Specify either sample_every_steps (recommended) or time_sampling."
        )

    if sample_every_raw is not None:
        sample_stride_steps = int(sample_every_raw)
        if sample_stride_steps < 1:
            raise ValueError(f"sample_every_steps must be >= 1, got {sample_stride_steps}.")
        if rollout_steps % sample_stride_steps != 0:
            raise ValueError(
                "rollout_steps must be divisible by sample_every_steps. "
                f"Got rollout_steps={rollout_steps}, sample_every_steps={sample_stride_steps}."
            )
        time_sampling = rollout_steps // sample_stride_steps
        if time_sampling_raw is not None and int(time_sampling_raw) != time_sampling:
            raise ValueError(
                "time_sampling conflicts with sample_every_steps: "
                f"time_sampling={int(time_sampling_raw)} but expected {time_sampling} "
                f"for rollout_steps={rollout_steps} and sample_every_steps={sample_stride_steps}."
            )
    else:
        time_sampling = int(time_sampling_raw)
        if time_sampling < 1:
            raise ValueError(f"time_sampling must be >= 1, got {time_sampling}.")
        if rollout_steps % time_sampling != 0:
            raise ValueError(
                "rollout_steps must be divisible by time_sampling for "
                "window/tau conversion in metric."
            )
        sample_stride_steps = rollout_steps // time_sampling

    win_size_frames = _resolve_frames(
        frames=getattr(args, "metric_window_size_frames", None),
        steps=getattr(args, "metric_window_size_steps", 20_000),
        sample_stride_steps=sample_stride_steps,
        name="metric_window_size",
    )
    win_step_frames = _resolve_frames(
        frames=getattr(args, "metric_window_step_frames", None),
        steps=getattr(args, "metric_window_step_steps", 20_000),
        sample_stride_steps=sample_stride_steps,
        name="metric_window_step",
    )
    tau_frames = _resolve_frames(
        frames=getattr(args, "metric_tau_frames", None),
        steps=getattr(args, "metric_tau_steps", 3_000),
        sample_stride_steps=sample_stride_steps,
        name="metric_tau",
    )

    if tau_frames >= win_size_frames:
        raise ValueError(
            f"metric_tau_frames ({tau_frames}) must be < metric_window_size_frames ({win_size_frames})."
        )

    T = int(time_sampling)
    if T < win_size_frames:
        raise ValueError(
            f"time_sampling ({T}) is too small for metric_window_size_frames ({win_size_frames})."
        )

    range_start_steps = _resolve_optional_steps(
        getattr(args, "metric_range_start_steps", None),
        name="metric_range_start_steps",
    )
    range_end_steps = _resolve_optional_steps(
        getattr(args, "metric_range_end_steps", None),
        name="metric_range_end_steps",
    )
    if range_start_steps is None:
        range_start_steps = 0
    if range_end_steps is None:
        range_end_steps = int(rollout_steps)
    if range_end_steps <= range_start_steps:
        raise ValueError(
            "metric_range_end_steps must be > metric_range_start_steps, got "
            f"{range_end_steps} <= {range_start_steps}."
        )

    starts_all = np.arange(0, T - win_size_frames + 1, win_step_frames, dtype=np.int32)
    starts_steps = starts_all.astype(np.int64) * int(sample_stride_steps)
    win_size_steps_eff = int(win_size_frames) * int(sample_stride_steps)
    ends_steps = starts_steps + win_size_steps_eff
    mask = (starts_steps >= range_start_steps) & (ends_steps <= range_end_steps)
    starts = starts_all[mask]
    W = int(starts.size)
    if W < 1:
        raise ValueError(
            "No valid windows produced for metric after range filtering; check "
            "window/tau/time_sampling or metric_range_start_steps/metric_range_end_steps."
        )

    tseg = win_size_frames - tau_frames
    if tseg < 1:
        raise ValueError(
            f"metric_window_size_frames ({win_size_frames}) - metric_tau_frames ({tau_frames}) must be >= 1."
        )

    m_samples_raw = int(getattr(args, "metric_m_samples", 48))
    m_count = tseg if m_samples_raw <= 0 else min(tseg, m_samples_raw)
    m_min = int(getattr(args, "metric_m_min", 4))
    if m_count < m_min:
        raise ValueError(
            f"Too few lagged samples per window for DeltaH: m_count={m_count}, m_min={m_min}. "
            "Increase window size / decrease tau / increase time_sampling."
        )

    n_proj = int(getattr(args, "metric_n_proj", 16))
    if n_proj < 2:
        raise ValueError(f"metric_n_proj must be >= 2, got {n_proj}.")

    null_reps = int(getattr(args, "metric_null_reps", 6))
    if null_reps < 0:
        raise ValueError(f"metric_null_reps must be >= 0, got {null_reps}.")
    particle_samples = int(
        getattr(args, "metric_particle_samples", getattr(args, "metric_spatial_samples", 64))
    )
    if particle_samples < 2:
        raise ValueError(f"metric_particle_samples must be >= 2, got {particle_samples}.")

    mode = str(getattr(args, "metric_preprocess_mode", "clip")).strip().lower()
    if mode not in {"clip", "shift", "none"}:
        raise ValueError(
            f"metric_preprocess_mode must be one of ['clip','shift','none'], got {mode!r}."
        )

    scales, weight_map, pairs = _resolve_scales(
        W=W,
        scales_raw=getattr(args, "metric_scales", None),
        weights_raw=getattr(args, "metric_scale_weights", None),
    )

    periodic_raw = getattr(args, "metric_periodic", False)
    domain_y_raw = getattr(args, "metric_domain_y", 0.0)
    domain_x_raw = getattr(args, "metric_domain_x", 0.0)

    cfg = dict(
        rollout_steps=rollout_steps,
        time_sampling=int(time_sampling),
        sample_every_steps=int(sample_stride_steps),
        sample_stride_steps=sample_stride_steps,
        window_size_frames=win_size_frames,
        window_step_frames=win_step_frames,
        tau_frames=tau_frames,
        starts=starts,
        range_start_steps=int(range_start_steps),
        range_end_steps=int(range_end_steps),
        W=W,
        tseg=tseg,
        m_count=int(m_count),
        n_proj=n_proj,
        null_reps=null_reps,
        particle_samples=particle_samples,
        preprocess_mode=mode,
        scales=scales,
        scale_weights={int(k): float(v) for k, v in weight_map.items()},
        scale_pairs=pairs,
        alpha=float(getattr(args, "metric_alpha", 1.0)),
        beta=float(getattr(args, "metric_beta", 1.0)),
        eps=float(getattr(args, "metric_eps", 1e-12)),
        dirs_seed=int(getattr(args, "metric_dirs_seed", 123)),
        periodic=bool(periodic_raw) if periodic_raw is not None else False,
        domain_y=0.0 if domain_y_raw is None else float(domain_y_raw),
        domain_x=0.0 if domain_x_raw is None else float(domain_x_raw),
    )
    return cfg


def metric_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(
        rollout_steps=int(cfg["rollout_steps"]),
        time_sampling=int(cfg["time_sampling"]),
        sample_every_steps=int(cfg["sample_every_steps"]),
        sample_stride_steps=int(cfg["sample_stride_steps"]),
        window_size_frames=int(cfg["window_size_frames"]),
        window_step_frames=int(cfg["window_step_frames"]),
        tau_frames=int(cfg["tau_frames"]),
        range_start_steps=int(cfg["range_start_steps"]),
        range_end_steps=int(cfg["range_end_steps"]),
        n_windows=int(cfg["W"]),
        tseg=int(cfg["tseg"]),
        m_count=int(cfg["m_count"]),
        n_proj=int(cfg["n_proj"]),
        null_reps=int(cfg["null_reps"]),
        particle_samples=int(cfg["particle_samples"]),
        periodic=bool(cfg["periodic"]),
        domain_y=float(cfg["domain_y"]),
        domain_x=float(cfg["domain_x"]),
        preprocess_mode=str(cfg["preprocess_mode"]),
        scales=list(cfg["scales"]),
        scale_pairs=[(int(r), float(w)) for r, w in cfg["scale_pairs"]],
        alpha=float(cfg["alpha"]),
        beta=float(cfg["beta"]),
    )


def _mean_pairwise_l1(sig: jnp.ndarray) -> jnp.ndarray:
    n = sig.shape[0]
    if n < 2:
        return jnp.array(0.0, dtype=sig.dtype)
    d = jnp.mean(jnp.abs(sig[:, None, :] - sig[None, :, :]), axis=2)
    mask = jnp.triu(jnp.ones((n, n), dtype=sig.dtype), k=1)
    denom = jnp.array(n * (n - 1) // 2, dtype=sig.dtype)
    return jnp.sum(d * mask) / jnp.maximum(denom, jnp.array(1.0, dtype=sig.dtype))


def make_metric_loss_fn(cfg: dict[str, Any]):
    starts = jnp.asarray(cfg["starts"], dtype=jnp.int32)
    W = int(cfg["W"])
    win = int(cfg["window_size_frames"])
    tau = int(cfg["tau_frames"])
    tseg = int(cfg["tseg"])
    m_count = int(cfg["m_count"])
    n_proj = int(cfg["n_proj"])
    null_reps = int(cfg["null_reps"])
    particle_samples = int(cfg["particle_samples"])
    mode = str(cfg["preprocess_mode"])
    scale_pairs = [(int(r), float(w)) for r, w in cfg["scale_pairs"]]
    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    eps = float(cfg["eps"])
    dirs_seed = int(cfg["dirs_seed"])
    periodic = bool(cfg["periodic"])
    domain_y = float(cfg["domain_y"])
    domain_x = float(cfg["domain_x"])
    use_all_lags = (m_count >= tseg)
    base_k_idx = jnp.arange(m_count, dtype=jnp.int32)
    dir_key = jax.random.PRNGKey(dirs_seed)

    def _preprocess(h: jnp.ndarray) -> jnp.ndarray:
        if mode == "clip":
            return jnp.maximum(h, 0.0)
        if mode == "shift":
            return h - jnp.min(h)
        return h

    def _signature_from_increments(v_s: jnp.ndarray, dirs: jnp.ndarray) -> jnp.ndarray:
        # v_s: (m_count, S, 2)
        proj = jnp.einsum("msd,ld->msl", v_s, dirs)  # (m_count, S, L)
        proj = jnp.sort(proj, axis=0)  # (m_count, S, L)
        sig = jnp.transpose(proj, (1, 2, 0)).reshape(v_s.shape[1], -1)  # (S, L*m_count)
        return sig

    def _delta_periodic(dx: jnp.ndarray) -> jnp.ndarray:
        if periodic:
            if domain_y > 0:
                dy = (dx[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
                dx = dx.at[..., 0].set(dy)
            if domain_x > 0:
                ddx = (dx[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
                dx = dx.at[..., 1].set(ddx)
        return dx

    def _delta_h_window(xy_seq: jnp.ndarray, start: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        key_k, key_p, key_null = jax.random.split(key, 3)
        X_w = jax.lax.dynamic_slice(
            xy_seq,
            (start, 0, 0),
            (win, xy_seq.shape[1], 2),
        )  # (win, N, 2)
        n_particles = X_w.shape[1]
        s_count = min(particle_samples, n_particles)
        if use_all_lags:
            k_idx = base_k_idx
        else:
            k_idx = jax.random.choice(key_k, tseg, shape=(m_count,), replace=False)
            k_idx = jnp.sort(k_idx)

        if s_count >= n_particles:
            p_idx = jnp.arange(n_particles, dtype=jnp.int32)
        else:
            p_idx = jax.random.choice(key_p, n_particles, shape=(s_count,), replace=False)
            p_idx = jnp.sort(p_idx)

        X0 = X_w[k_idx][:, p_idx, :]
        X1 = X_w[k_idx + tau][:, p_idx, :]
        dx = _delta_periodic(X1 - X0)
        dt = jnp.maximum(
            jnp.asarray(float(tau) * float(cfg["sample_stride_steps"]), dtype=xy_seq.dtype),
            jnp.asarray(1e-12, dtype=xy_seq.dtype),
        )
        v_s = dx / dt  # (m_count, S, 2)

        dirs = jax.random.normal(dir_key, (n_proj, 2), dtype=xy_seq.dtype)
        dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        sig = _signature_from_increments(v_s, dirs)
        h_real = _mean_pairwise_l1(sig)

        if null_reps <= 0:
            h_null = jnp.array(0.0, dtype=xy_seq.dtype)
        else:
            pool = v_s.reshape((-1, 2))
            pool_n = pool.shape[0]

            def _one_null(k: jax.Array) -> jnp.ndarray:
                idx = jax.random.randint(k, (m_count, s_count), 0, pool_n)
                v0 = pool[idx]  # (m_count, S, 2)
                sig0 = _signature_from_increments(v0, dirs)
                return _mean_pairwise_l1(sig0)

            null_keys = jax.random.split(key_null, null_reps)
            h0 = jax.vmap(_one_null)(null_keys)
            h_null = jnp.median(h0)

        return h_real - h_null

    def metric_loss_fn(rng_metric: jax.Array, xy_seq: jnp.ndarray):
        keys_w = jax.random.split(rng_metric, W)
        h = jax.vmap(lambda s, k: _delta_h_window(xy_seq, s, k))(starts, keys_w)  # (W,)
        h_pos = _preprocess(h)

        amp = jnp.mean(h_pos)
        msc = jnp.array(0.0, dtype=xy_seq.dtype)
        for r, wr in scale_pairs:
            U_r = W // r
            U_2r = W // (2 * r)
            g_r = jnp.mean(h_pos[: U_r * r].reshape(U_r, r), axis=1)
            g_2r = jnp.mean(h_pos[: U_2r * (2 * r)].reshape(U_2r, 2 * r), axis=1)
            # Compare only on common support: if U_r is odd, the last fine block
            # has no paired coarse block from level 2r and is dropped.
            U_cmp = min(U_r, 2 * U_2r)
            g_r_cmp = g_r[:U_cmp]
            up = jnp.repeat(g_2r, 2)[:U_cmp]
            overlap = jnp.sum(g_r_cmp * up)
            power = jnp.sum(g_r_cmp * g_r_cmp)
            d_r = 1.0 - overlap / (power + eps)
            msc = msc + (wr * d_r)

        score = alpha * amp + beta * msc
        loss = -score
        return loss, dict(
            score=score,
            amp=amp,
            msc=msc,
            delta_h_mean=jnp.mean(h),
            delta_h_std=jnp.std(h),
            delta_h_min=jnp.min(h),
            delta_h_max=jnp.max(h),
        )

    return metric_loss_fn
