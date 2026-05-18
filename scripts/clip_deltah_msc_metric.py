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


def _resolve_tau_grid_frames(
    *,
    sample_stride_steps: int,
    tau_grid_frames_raw: Any,
    tau_grid_steps_raw: Any,
) -> list[int]:
    if tau_grid_frames_raw is not None:
        vals = _maybe_list(tau_grid_frames_raw)
        if vals is None:
            vals = [tau_grid_frames_raw]
        frames = [int(v) for v in vals]
    elif tau_grid_steps_raw is not None:
        vals = _maybe_list(tau_grid_steps_raw)
        if vals is None:
            vals = [tau_grid_steps_raw]
        frames = [int(max(1, round(float(v) / float(sample_stride_steps)))) for v in vals]
    else:
        return []

    out: list[int] = []
    seen: set[int] = set()
    for v in frames:
        if v < 1:
            raise ValueError(f"All tau grid values must be >= 1, got {v}.")
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _tau_index_from_latent_jax(raw_tau: jax.Array | None, n_tau: int) -> jax.Array:
    if n_tau <= 1:
        return jnp.array(0, dtype=jnp.int32)
    if raw_tau is None:
        return jnp.array(0, dtype=jnp.int32)
    raw = jnp.ravel(jnp.asarray(raw_tau, dtype=jnp.float32))[0]
    u = jax.nn.sigmoid(raw)
    idx = jnp.rint(u * float(n_tau - 1)).astype(jnp.int32)
    return jnp.clip(idx, 0, n_tau - 1)


def tau_selection_from_latent(cfg: dict[str, Any], raw_tau: float | np.ndarray | None) -> dict[str, Any]:
    tau_frames_list = [int(x) for x in cfg.get("tau_frames_list", [cfg["tau_frames"]])]
    tau_steps_list = [
        int(x)
        for x in cfg.get(
            "tau_steps_list",
            [int(t) * int(cfg["sample_stride_steps"]) for t in tau_frames_list],
        )
    ]
    n_tau = len(tau_frames_list)
    if n_tau <= 1:
        idx = 0
        raw_val = 0.0 if raw_tau is None else float(np.ravel(np.asarray(raw_tau, dtype=np.float64))[0])
    else:
        raw_val = 0.0 if raw_tau is None else float(np.ravel(np.asarray(raw_tau, dtype=np.float64))[0])
        u = 1.0 / (1.0 + np.exp(-raw_val))
        idx = int(np.clip(np.rint(u * float(n_tau - 1)), 0, n_tau - 1))
    return dict(
        tau_selector_raw=float(raw_val),
        tau_idx=int(idx),
        tau_frames=int(tau_frames_list[idx]),
        tau_steps=int(tau_steps_list[idx]),
    )


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
    tau_mode = str(getattr(args, "metric_tau_mode", "fixed")).strip().lower()
    if tau_mode not in {"fixed", "max_grid", "trainable_grid"}:
        raise ValueError(
            f"metric_tau_mode must be one of ['fixed','max_grid','trainable_grid'], got {tau_mode!r}."
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

    m_samples_raw = int(getattr(args, "metric_m_samples", 48))
    m_min = int(getattr(args, "metric_m_min", 4))
    tau_frames_fixed = _resolve_frames(
        frames=getattr(args, "metric_tau_frames", None),
        steps=getattr(args, "metric_tau_steps", 3_000),
        sample_stride_steps=sample_stride_steps,
        name="metric_tau",
    )
    tau_frames_grid = _resolve_tau_grid_frames(
        sample_stride_steps=sample_stride_steps,
        tau_grid_frames_raw=getattr(args, "metric_tau_grid_frames", None),
        tau_grid_steps_raw=getattr(args, "metric_tau_grid_steps", None),
    )
    if tau_mode == "fixed":
        tau_frames_list = [int(tau_frames_fixed)]
    else:
        tau_frames_list = tau_frames_grid if tau_frames_grid else [int(tau_frames_fixed)]
    if tau_mode == "trainable_grid" and len(tau_frames_list) < 2:
        raise ValueError(
            "metric_tau_mode='trainable_grid' requires at least 2 tau values in "
            "metric_tau_grid_steps or metric_tau_grid_frames."
        )

    for tau_frames in tau_frames_list:
        if tau_frames >= win_size_frames:
            raise ValueError(
                f"metric tau ({tau_frames}) must be < metric_window_size_frames ({win_size_frames})."
            )

    tseg_list: list[int] = []
    m_count_list: list[int] = []
    for tau_frames in tau_frames_list:
        tseg = win_size_frames - tau_frames
        if tseg < 1:
            raise ValueError(
                f"metric_window_size_frames ({win_size_frames}) - tau ({tau_frames}) must be >= 1."
            )
        m_count = tseg if m_samples_raw <= 0 else min(tseg, m_samples_raw)
        if m_count < m_min:
            raise ValueError(
                f"Too few lagged samples per window for tau={tau_frames}: "
                f"m_count={m_count}, m_min={m_min}. "
                "Increase window size / decrease tau / increase time_sampling."
            )
        tseg_list.append(int(tseg))
        m_count_list.append(int(m_count))

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
    delta_h_floor = float(getattr(args, "metric_delta_h_floor", 0.0) or 0.0)
    if delta_h_floor < 0.0:
        raise ValueError(f"metric_delta_h_floor must be >= 0, got {delta_h_floor}.")

    scales, weight_map, pairs = _resolve_scales(
        W=W,
        scales_raw=getattr(args, "metric_scales", None),
        weights_raw=getattr(args, "metric_scale_weights", None),
    )

    periodic_raw = getattr(args, "metric_periodic", False)
    domain_y_raw = getattr(args, "metric_domain_y", 0.0)
    domain_x_raw = getattr(args, "metric_domain_x", 0.0)

    tau_frames = int(tau_frames_list[0])
    tseg = int(tseg_list[0])
    m_count = int(m_count_list[0])

    cfg = dict(
        rollout_steps=rollout_steps,
        time_sampling=int(time_sampling),
        sample_every_steps=int(sample_stride_steps),
        sample_stride_steps=sample_stride_steps,
        window_size_frames=win_size_frames,
        window_step_frames=win_step_frames,
        tau_mode=tau_mode,
        tau_frames=tau_frames,
        tau_steps=int(tau_frames * sample_stride_steps),
        tau_frames_list=[int(x) for x in tau_frames_list],
        tau_steps_list=[int(x) * int(sample_stride_steps) for x in tau_frames_list],
        starts=starts,
        range_start_steps=int(range_start_steps),
        range_end_steps=int(range_end_steps),
        W=W,
        tseg=tseg,
        tseg_list=[int(x) for x in tseg_list],
        m_count=int(m_count),
        m_count_list=[int(x) for x in m_count_list],
        n_proj=n_proj,
        null_reps=null_reps,
        particle_samples=particle_samples,
        preprocess_mode=mode,
        delta_h_floor=delta_h_floor,
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
        tau_mode=str(cfg.get("tau_mode", "fixed")),
        tau_frames=int(cfg["tau_frames"]),
        tau_steps=int(cfg.get("tau_steps", int(cfg["tau_frames"]) * int(cfg["sample_stride_steps"]))),
        tau_frames_list=[int(x) for x in cfg.get("tau_frames_list", [cfg["tau_frames"]])],
        tau_steps_list=[
            int(x)
            for x in cfg.get(
                "tau_steps_list",
                [int(cfg["tau_frames"]) * int(cfg["sample_stride_steps"])],
            )
        ],
        range_start_steps=int(cfg["range_start_steps"]),
        range_end_steps=int(cfg["range_end_steps"]),
        n_windows=int(cfg["W"]),
        tseg=int(cfg["tseg"]),
        tseg_list=[int(x) for x in cfg.get("tseg_list", [cfg["tseg"]])],
        m_count=int(cfg["m_count"]),
        m_count_list=[int(x) for x in cfg.get("m_count_list", [cfg["m_count"]])],
        n_proj=int(cfg["n_proj"]),
        null_reps=int(cfg["null_reps"]),
        particle_samples=int(cfg["particle_samples"]),
        periodic=bool(cfg["periodic"]),
        positions_unwrapped=bool(cfg.get("positions_unwrapped", False)),
        domain_y=float(cfg["domain_y"]),
        domain_x=float(cfg["domain_x"]),
        preprocess_mode=str(cfg["preprocess_mode"]),
        delta_h_floor=float(cfg.get("delta_h_floor", 0.0)),
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


def make_metric_loss_fn(cfg: dict[str, Any], *, include_maps: bool = False):
    starts = jnp.asarray(cfg["starts"], dtype=jnp.int32)
    W = int(cfg["W"])
    win = int(cfg["window_size_frames"])
    tau_mode = str(cfg.get("tau_mode", "fixed"))
    tau_frames_list = [int(x) for x in cfg.get("tau_frames_list", [cfg["tau_frames"]])]
    tau_steps_list = [
        int(x)
        for x in cfg.get(
            "tau_steps_list",
            [int(t) * int(cfg["sample_stride_steps"]) for t in tau_frames_list],
        )
    ]
    tseg_list = [int(x) for x in cfg.get("tseg_list", [cfg["tseg"]])]
    m_count_list = [int(x) for x in cfg.get("m_count_list", [cfg["m_count"]])]
    if not (len(tau_frames_list) == len(tseg_list) == len(m_count_list) == len(tau_steps_list)):
        raise ValueError("Tau list config mismatch: tau/tseg/m_count lengths must match.")
    n_proj = int(cfg["n_proj"])
    null_reps = int(cfg["null_reps"])
    particle_samples = int(cfg["particle_samples"])
    mode = str(cfg["preprocess_mode"])
    delta_h_floor = float(cfg.get("delta_h_floor", 0.0))
    scale_pairs = [(int(r), float(w)) for r, w in cfg["scale_pairs"]]
    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    eps = float(cfg["eps"])
    dirs_seed = int(cfg["dirs_seed"])
    periodic = bool(cfg["periodic"])
    positions_unwrapped = bool(cfg.get("positions_unwrapped", False))
    domain_y = float(cfg["domain_y"])
    domain_x = float(cfg["domain_x"])
    sample_stride_steps = float(cfg["sample_stride_steps"])
    dir_key = jax.random.PRNGKey(dirs_seed)

    def _preprocess(h: jnp.ndarray) -> jnp.ndarray:
        if mode == "clip":
            out = jnp.maximum(h, 0.0)
        elif mode == "shift":
            out = h - jnp.min(h)
        else:
            out = h
        if delta_h_floor > 0.0:
            floor = jnp.asarray(delta_h_floor, dtype=out.dtype)
            out = jnp.where(out >= floor, out, jnp.zeros_like(out))
        return out

    def _signature_from_increments(v_s: jnp.ndarray, dirs: jnp.ndarray) -> jnp.ndarray:
        # v_s: (m_count, S, 2)
        proj = jnp.einsum("msd,ld->msl", v_s, dirs)  # (m_count, S, L)
        proj = jnp.sort(proj, axis=0)  # (m_count, S, L)
        sig = jnp.transpose(proj, (1, 2, 0)).reshape(v_s.shape[1], -1)  # (S, L*m_count)
        return sig

    def _delta_periodic(dx: jnp.ndarray) -> jnp.ndarray:
        if periodic and not positions_unwrapped:
            if domain_y > 0:
                dy = (dx[..., 0] + 0.5 * domain_y) % domain_y - 0.5 * domain_y
                dx = dx.at[..., 0].set(dy)
            if domain_x > 0:
                ddx = (dx[..., 1] + 0.5 * domain_x) % domain_x - 0.5 * domain_x
                dx = dx.at[..., 1].set(ddx)
        return dx

    def _delta_h_window(
        xy_seq: jnp.ndarray,
        start: jnp.ndarray,
        key: jax.Array,
        *,
        tau: int,
        tseg: int,
        m_count: int,
    ) -> jnp.ndarray:
        key_k, key_p, key_null = jax.random.split(key, 3)
        X_w = jax.lax.dynamic_slice(
            xy_seq,
            (start, 0, 0),
            (win, xy_seq.shape[1], 2),
        )  # (win, N, 2)
        n_particles = X_w.shape[1]
        s_count = min(particle_samples, n_particles)
        use_all_lags = (m_count >= tseg)
        base_k_idx = jnp.arange(m_count, dtype=jnp.int32)
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
        X_sample = X_w[:, p_idx, :]
        dx = _delta_periodic(X1 - X0)
        dt = jnp.maximum(
            jnp.asarray(float(tau) * sample_stride_steps, dtype=xy_seq.dtype),
            jnp.asarray(1e-12, dtype=xy_seq.dtype),
        )
        v_s = dx / dt  # (m_count, S, 2)
        dx_norm = jnp.linalg.norm(dx, axis=-1)
        speed_norm = jnp.linalg.norm(v_s, axis=-1)
        pos_flat = X_sample.reshape((-1, 2))

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

        h_delta = h_real - h_null
        return dict(
            delta_h=h_delta,
            h_real=h_real,
            h_null=h_null,
            dx_norm_mean=jnp.mean(dx_norm),
            dx_norm_std=jnp.std(dx_norm),
            dx_norm_max=jnp.max(dx_norm),
            speed_norm_mean=jnp.mean(speed_norm),
            speed_norm_std=jnp.std(speed_norm),
            speed_norm_max=jnp.max(speed_norm),
            speed_component_std_mean=jnp.mean(jnp.std(v_s.reshape((-1, 2)), axis=0)),
            position_std_mean=jnp.mean(jnp.std(pos_flat, axis=0)),
            position_range_mean=jnp.mean(jnp.max(pos_flat, axis=0) - jnp.min(pos_flat, axis=0)),
        )

    def _score_from_h(h: jnp.ndarray, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h_pos = _preprocess(h)
        amp = jnp.mean(h_pos)
        msc = jnp.array(0.0, dtype=dtype)
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
            d_r = jnp.where(
                power > eps,
                1.0 - overlap / (power + eps),
                jnp.array(0.0, dtype=dtype),
            )
            msc = msc + (wr * d_r)
        score = alpha * amp + beta * msc
        return score, amp, msc, h_pos

    def metric_loss_fn(rng_metric: jax.Array, xy_seq: jnp.ndarray, tau_selector: jax.Array | None = None):
        tau_count = len(tau_frames_list)
        keys_tau = jax.random.split(rng_metric, tau_count)

        h_list = []
        h_real_list = []
        h_null_list = []
        dx_norm_mean_list = []
        dx_norm_std_list = []
        dx_norm_max_list = []
        speed_norm_mean_list = []
        speed_norm_std_list = []
        speed_norm_max_list = []
        speed_component_std_mean_list = []
        position_std_mean_list = []
        position_range_mean_list = []
        h_processed_list = []
        score_list = []
        amp_list = []
        msc_list = []
        for i in range(tau_count):
            tau = int(tau_frames_list[i])
            tseg = int(tseg_list[i])
            m_count = int(m_count_list[i])
            keys_w = jax.random.split(keys_tau[i], W)
            window_diag = jax.vmap(
                lambda s, k: _delta_h_window(
                    xy_seq,
                    s,
                    k,
                    tau=tau,
                    tseg=tseg,
                    m_count=m_count,
                )
            )(starts, keys_w)
            h = window_diag["delta_h"]
            score, amp, msc, h_processed = _score_from_h(h, xy_seq.dtype)
            h_list.append(h)
            h_real_list.append(window_diag["h_real"])
            h_null_list.append(window_diag["h_null"])
            dx_norm_mean_list.append(window_diag["dx_norm_mean"])
            dx_norm_std_list.append(window_diag["dx_norm_std"])
            dx_norm_max_list.append(window_diag["dx_norm_max"])
            speed_norm_mean_list.append(window_diag["speed_norm_mean"])
            speed_norm_std_list.append(window_diag["speed_norm_std"])
            speed_norm_max_list.append(window_diag["speed_norm_max"])
            speed_component_std_mean_list.append(window_diag["speed_component_std_mean"])
            position_std_mean_list.append(window_diag["position_std_mean"])
            position_range_mean_list.append(window_diag["position_range_mean"])
            h_processed_list.append(h_processed)
            score_list.append(score)
            amp_list.append(amp)
            msc_list.append(msc)

        h_all = jnp.stack(h_list, axis=0)  # (Ktau, W)
        h_processed_all = jnp.stack(h_processed_list, axis=0)
        h_real_all = jnp.stack(h_real_list, axis=0)
        h_null_all = jnp.stack(h_null_list, axis=0)
        dx_norm_mean_all = jnp.stack(dx_norm_mean_list, axis=0)
        dx_norm_std_all = jnp.stack(dx_norm_std_list, axis=0)
        dx_norm_max_all = jnp.stack(dx_norm_max_list, axis=0)
        speed_norm_mean_all = jnp.stack(speed_norm_mean_list, axis=0)
        speed_norm_std_all = jnp.stack(speed_norm_std_list, axis=0)
        speed_norm_max_all = jnp.stack(speed_norm_max_list, axis=0)
        speed_component_std_mean_all = jnp.stack(speed_component_std_mean_list, axis=0)
        position_std_mean_all = jnp.stack(position_std_mean_list, axis=0)
        position_range_mean_all = jnp.stack(position_range_mean_list, axis=0)
        score_all = jnp.stack(score_list, axis=0)  # (Ktau,)
        amp_all = jnp.stack(amp_list, axis=0)
        msc_all = jnp.stack(msc_list, axis=0)

        if tau_mode == "max_grid":
            best_idx = jnp.argmax(score_all)
        elif tau_mode == "trainable_grid":
            best_idx = _tau_index_from_latent_jax(tau_selector, tau_count)
        else:
            best_idx = jnp.array(0, dtype=jnp.int32)

        score = score_all[best_idx]
        amp = amp_all[best_idx]
        msc = msc_all[best_idx]
        h_best = h_all[best_idx]
        h_processed_best = h_processed_all[best_idx]
        h_real_best = h_real_all[best_idx]
        h_null_best = h_null_all[best_idx]
        dx_norm_mean_best = dx_norm_mean_all[best_idx]
        dx_norm_std_best = dx_norm_std_all[best_idx]
        dx_norm_max_best = dx_norm_max_all[best_idx]
        speed_norm_mean_best = speed_norm_mean_all[best_idx]
        speed_norm_std_best = speed_norm_std_all[best_idx]
        speed_norm_max_best = speed_norm_max_all[best_idx]
        speed_component_std_mean_best = speed_component_std_mean_all[best_idx]
        position_std_mean_best = position_std_mean_all[best_idx]
        position_range_mean_best = position_range_mean_all[best_idx]

        tau_frames_arr = jnp.asarray(tau_frames_list, dtype=jnp.int32)
        tau_steps_arr = jnp.asarray(tau_steps_list, dtype=jnp.int32)
        tau_best_frames = tau_frames_arr[best_idx]
        tau_best_steps = tau_steps_arr[best_idx]
        tau_selector_raw = (
            jnp.asarray(jnp.ravel(jnp.asarray(tau_selector, dtype=xy_seq.dtype))[0], dtype=xy_seq.dtype)
            if tau_selector is not None
            else jnp.asarray(0.0, dtype=xy_seq.dtype)
        )
        tau_selected_idx = best_idx.astype(xy_seq.dtype)

        loss = -score
        info = dict(
            score=score,
            amp=amp,
            msc=msc,
            delta_h_mean=jnp.mean(h_best),
            delta_h_std=jnp.std(h_best),
            delta_h_min=jnp.min(h_best),
            delta_h_max=jnp.max(h_best),
            delta_h_abs_mean=jnp.mean(jnp.abs(h_best)),
            delta_h_positive_frac=jnp.mean((h_best > 0.0).astype(xy_seq.dtype)),
            delta_h_processed_mean=jnp.mean(h_processed_best),
            delta_h_processed_std=jnp.std(h_processed_best),
            delta_h_processed_min=jnp.min(h_processed_best),
            delta_h_processed_max=jnp.max(h_processed_best),
            delta_h_processed_positive_frac=jnp.mean((h_processed_best > 0.0).astype(xy_seq.dtype)),
            h_real_mean=jnp.mean(h_real_best),
            h_real_std=jnp.std(h_real_best),
            h_real_min=jnp.min(h_real_best),
            h_real_max=jnp.max(h_real_best),
            h_null_mean=jnp.mean(h_null_best),
            h_null_std=jnp.std(h_null_best),
            h_null_min=jnp.min(h_null_best),
            h_null_max=jnp.max(h_null_best),
            h_real_minus_null_mean=jnp.mean(h_real_best - h_null_best),
            h_real_over_null_mean=jnp.mean(h_real_best / jnp.maximum(jnp.abs(h_null_best), eps)),
            h_delta_over_real_mean=jnp.mean(h_best / jnp.maximum(jnp.abs(h_real_best), eps)),
            dx_norm_mean=jnp.mean(dx_norm_mean_best),
            dx_norm_std_mean=jnp.mean(dx_norm_std_best),
            dx_norm_max=jnp.max(dx_norm_max_best),
            speed_norm_mean=jnp.mean(speed_norm_mean_best),
            speed_norm_std_mean=jnp.mean(speed_norm_std_best),
            speed_norm_max=jnp.max(speed_norm_max_best),
            speed_component_std_mean=jnp.mean(speed_component_std_mean_best),
            position_std_mean=jnp.mean(position_std_mean_best),
            position_range_mean=jnp.mean(position_range_mean_best),
            delta_h_tau_mean=jnp.mean(h_all),
            delta_h_tau_abs_mean=jnp.mean(jnp.abs(h_all)),
            delta_h_processed_tau_mean=jnp.mean(h_processed_all),
            h_real_tau_mean=jnp.mean(h_real_all),
            h_null_tau_mean=jnp.mean(h_null_all),
            dx_norm_tau_mean=jnp.mean(dx_norm_mean_all),
            speed_norm_tau_mean=jnp.mean(speed_norm_mean_all),
            tau_selected_idx=tau_selected_idx,
            tau_best_frames=tau_best_frames.astype(xy_seq.dtype),
            tau_best_steps=tau_best_steps.astype(xy_seq.dtype),
            tau_selector_raw=tau_selector_raw,
            score_tau_max=jnp.max(score_all),
            score_tau_min=jnp.min(score_all),
            score_tau_mean=jnp.mean(score_all),
            msc_tau_max=jnp.max(msc_all),
            msc_tau_min=jnp.min(msc_all),
            msc_tau_mean=jnp.mean(msc_all),
            amp_tau_max=jnp.max(amp_all),
            amp_tau_min=jnp.min(amp_all),
            amp_tau_mean=jnp.mean(amp_all),
        )
        if include_maps:
            info.update(
                delta_h_map=h_all,
                delta_h_best=h_best,
                delta_h_processed_map=h_processed_all,
                delta_h_processed_best=h_processed_best,
                score_by_tau=score_all,
                amp_by_tau=amp_all,
                msc_by_tau=msc_all,
                tau_frames=tau_frames_arr,
                tau_steps=tau_steps_arr,
                window_start_frames=starts.astype(jnp.int32),
                window_start_steps=(starts * int(cfg["sample_stride_steps"])).astype(jnp.int32),
            )
        return loss, info

    return metric_loss_fn
