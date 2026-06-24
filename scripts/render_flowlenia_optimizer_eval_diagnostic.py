from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from replay_flowlenia_optimizer_eval_with_source import (
    _activate_source_root,
    _best_candidate_and_keys,
    _flat_config,
    _init_lagrangian_points_jax,
)


def _build_context(flat: Any, *, include_maps: bool):
    import substrates
    import util
    from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config

    base_substrate = substrates.create_substrate(flat.substrate, **util.substrate_kwargs_from_args(flat))
    if hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    if hasattr(base_substrate, "render_mode") and getattr(flat, "render_mode", None) is not None:
        base_substrate.render_mode = str(flat.render_mode)
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
    metric_loss_fn = make_metric_loss_fn(metric_cfg, include_maps=include_maps)
    return substrate, metric_cfg, metric_loss_fn


def _select(arr: Any, local_idx: int, rep_idx: int) -> np.ndarray:
    x = np.asarray(jax.device_get(arr))
    if x.ndim >= 2:
        return np.asarray(x[local_idx, rep_idx])
    return x


def _metric_chunk_raw(
    *,
    flat: Any,
    substrate: Any,
    metric_cfg: dict[str, Any],
    metric_loss_fn: Any,
    candidate: dict[str, Any],
    substrate_param_dims: int,
    tau_extra_dims: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    chunk_steps = int(metric_cfg["sample_every_steps"])
    time_sampling = int(metric_cfg["time_sampling"])
    lag_n_particles = int(getattr(flat, "metric_lagrangian_n_particles", 256))
    lag_init_mode = str(getattr(flat, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(getattr(flat, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(getattr(flat, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(getattr(flat, "metric_lagrangian_channel_mode", "mix"))
    lag_noise_model = str(getattr(flat, "metric_lagrangian_noise_model", "none"))
    lag_diffusion_scale = float(getattr(flat, "metric_lagrangian_diffusion_scale", 1.0))
    log_clip_evolution = bool(getattr(flat, "log_clip_evolution", True))

    def rollout_metric_xy(rng, params):
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng, 4)
        s0 = substrate.init_state(k_state, params)
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
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, chunk_steps))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, time_sampling),
        )
        return xy_seq

    def calc_loss(rng, params_full):
        params = params_full[:substrate_param_dims]
        tau_selector = params_full[substrate_param_dims] if tau_extra_dims else None
        if log_clip_evolution:
            rng_roll, rng_metric, _rng_clip = jax.random.split(rng, 3)
        else:
            rng_roll, rng_metric = jax.random.split(rng)
        xy = rollout_metric_xy(rng_roll, params)
        if tau_extra_dims:
            return metric_loss_fn(rng_metric, xy, tau_selector=tau_selector)
        return metric_loss_fn(rng_metric, xy)

    calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))

    @jax.jit
    def eval_chunk_raw(rng, params_chunk):
        _rng_next, rng_metric_parent = jax.random.split(rng)
        return calc_loss_vv(jax.random.split(rng_metric_parent, int(flat.bs)), params_chunk)

    params_chunk = jnp.asarray(np.asarray(candidate["params_chunk"], dtype=np.float32), dtype=jnp.float32)
    loss_raw, info_raw = eval_chunk_raw(candidate["rng_eval"], params_chunk)
    info_np = {k: np.asarray(jax.device_get(v)) for k, v in info_raw.items()}
    return np.asarray(jax.device_get(loss_raw)), info_np


def _render_state_video(
    *,
    flat: Any,
    substrate: Any,
    params: np.ndarray,
    rep_key: Any,
    output: Path,
    img_size: int,
    fps: int,
    codec: str,
    stride_steps: int,
    max_steps: int,
    frame_batch_size: int,
    resize_method: str,
) -> dict[str, Any]:
    chunk_steps = int(getattr(flat, "sample_every_steps", 50))
    time_sampling = int(int(getattr(flat, "rollout_steps")) // chunk_steps)
    if stride_steps % chunk_steps != 0:
        raise ValueError(f"stride_steps must be divisible by sample_every_steps={chunk_steps}, got {stride_steps}.")
    stride_chunks = max(1, int(stride_steps // chunk_steps))
    max_chunks = min(time_sampling, int(max_steps // chunk_steps))
    n_frames = int(max_chunks // stride_chunks)
    if n_frames < 1:
        raise ValueError("Video would have zero frames. Increase max_steps or reduce stride_steps.")

    log_clip_evolution = bool(getattr(flat, "log_clip_evolution", True))
    if log_clip_evolution:
        rng_roll, _rng_metric, _rng_clip = jax.random.split(rep_key, 3)
    else:
        rng_roll, _rng_metric = jax.random.split(rep_key)
    k_state, _k_pts, _k_ch, k_scan = jax.random.split(rng_roll, 4)
    params_j = jnp.asarray(np.asarray(params, dtype=np.float32), dtype=jnp.float32)
    state = substrate.init_state(k_state, params_j)
    key_groups = jax.random.split(k_scan, time_sampling)[: n_frames * stride_chunks]
    key_groups = key_groups.reshape((n_frames, stride_chunks, 2))

    def run_groups(state_in, keys_group_batch):
        def one_group(st, keys_group):
            def one_metric_chunk(st2, key_chunk):
                def one_step(st3, key_step):
                    return substrate.step_state(key_step, st3, params_j), None

                st_next, _ = jax.lax.scan(one_step, st2, jax.random.split(key_chunk, chunk_steps))
                return st_next, None

            st_after, _ = jax.lax.scan(one_metric_chunk, st, keys_group)
            frame = substrate.render_state(st_after, params_j, img_size=None)
            if img_size is not None and int(img_size) > 0 and int(frame.shape[0]) != int(img_size):
                frame = jax.image.resize(frame, (int(img_size), int(img_size), 3), method=str(resize_method))
            return st_after, frame

        return jax.lax.scan(one_group, state_in, keys_group_batch)

    run_groups_jit = jax.jit(run_groups)
    frames_out: list[np.ndarray] = []
    for start in range(0, n_frames, int(frame_batch_size)):
        batch = key_groups[start : start + int(frame_batch_size)]
        state, frames = run_groups_jit(state, batch)
        frames_out.append(np.asarray(jax.device_get(frames)))
    video = np.concatenate(frames_out, axis=0)
    video_u8 = (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, video_u8, fps=int(fps), codec=str(codec), macro_block_size=None)
    return {
        "video_path": str(output),
        "video_frames": int(video_u8.shape[0]),
        "video_stride_steps": int(stride_steps),
        "video_max_steps_rendered": int(video_u8.shape[0] * stride_steps),
        "video_fps": int(fps),
        "video_resize_method": str(resize_method),
    }


def _draw_label(frames_u8: np.ndarray, label: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frames_u8

    out = frames_u8.copy()
    font = ImageFont.load_default()
    for i in range(out.shape[0]):
        img = Image.fromarray(out[i])
        draw = ImageDraw.Draw(img, "RGBA")
        bbox = draw.textbbox((0, 0), label, font=font)
        w = int(bbox[2] - bbox[0] + 12)
        h = int(bbox[3] - bbox[1] + 8)
        draw.rectangle((4, 4, 4 + w, 4 + h), fill=(0, 0, 0, 150))
        draw.text((10, 8), label, fill=(255, 255, 255, 255), font=font)
        out[i] = np.asarray(img)
    return out


def _render_state_grid_video(
    *,
    flat: Any,
    substrate: Any,
    params: np.ndarray,
    rep_keys: Any,
    rep_scores: np.ndarray,
    output: Path,
    img_size: int,
    fps: int,
    codec: str,
    stride_steps: int,
    max_steps: int,
    frame_batch_size: int,
    resize_method: str,
) -> dict[str, Any]:
    chunk_steps = int(getattr(flat, "sample_every_steps", 50))
    time_sampling = int(int(getattr(flat, "rollout_steps")) // chunk_steps)
    if stride_steps % chunk_steps != 0:
        raise ValueError(f"stride_steps must be divisible by sample_every_steps={chunk_steps}, got {stride_steps}.")
    stride_chunks = max(1, int(stride_steps // chunk_steps))
    max_chunks = min(time_sampling, int(max_steps // chunk_steps))
    n_frames = int(max_chunks // stride_chunks)
    if n_frames < 1:
        raise ValueError("Grid video would have zero frames. Increase max_steps or reduce stride_steps.")

    params_j = jnp.asarray(np.asarray(params, dtype=np.float32), dtype=jnp.float32)
    log_clip_evolution = bool(getattr(flat, "log_clip_evolution", True))

    states = []
    key_groups_by_rep = []
    for rep_key in list(rep_keys):
        if log_clip_evolution:
            rng_roll, _rng_metric, _rng_clip = jax.random.split(rep_key, 3)
        else:
            rng_roll, _rng_metric = jax.random.split(rep_key)
        k_state, _k_pts, _k_ch, k_scan = jax.random.split(rng_roll, 4)
        states.append(substrate.init_state(k_state, params_j))
        key_groups = jax.random.split(k_scan, time_sampling)[: n_frames * stride_chunks]
        key_groups_by_rep.append(key_groups.reshape((n_frames, stride_chunks, 2)))

    def run_groups(state_in, keys_group_batch):
        def one_group(st, keys_group):
            def one_metric_chunk(st2, key_chunk):
                def one_step(st3, key_step):
                    return substrate.step_state(key_step, st3, params_j), None

                st_next, _ = jax.lax.scan(one_step, st2, jax.random.split(key_chunk, chunk_steps))
                return st_next, None

            st_after, _ = jax.lax.scan(one_metric_chunk, st, keys_group)
            frame = substrate.render_state(st_after, params_j, img_size=None)
            if img_size is not None and int(img_size) > 0 and int(frame.shape[0]) != int(img_size):
                frame = jax.image.resize(frame, (int(img_size), int(img_size), 3), method=str(resize_method))
            return st_after, frame

        return jax.lax.scan(one_group, state_in, keys_group_batch)

    run_groups_jit = jax.jit(run_groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"rep {i}  MSPD={float(rep_scores[i]):.6g}" for i in range(len(rep_scores))]
    with imageio.get_writer(str(output), fps=int(fps), codec=str(codec), macro_block_size=None) as writer:
        for start in range(0, n_frames, int(frame_batch_size)):
            end = min(n_frames, start + int(frame_batch_size))
            rep_frames = []
            for i_rep in range(len(states)):
                states[i_rep], frames = run_groups_jit(states[i_rep], key_groups_by_rep[i_rep][start:end])
                frames_u8 = (np.clip(np.asarray(jax.device_get(frames)), 0.0, 1.0) * 255).astype(np.uint8)
                rep_frames.append(_draw_label(frames_u8, labels[i_rep]))
            while len(rep_frames) < 4:
                rep_frames.append(np.zeros_like(rep_frames[0]))
            top = np.concatenate([rep_frames[0], rep_frames[1]], axis=2)
            bottom = np.concatenate([rep_frames[2], rep_frames[3]], axis=2)
            grid = np.concatenate([top, bottom], axis=1)
            for frame in grid:
                writer.append_data(frame)

    return {
        "all_reps_video_path": str(output),
        "all_reps_video_frames": int(n_frames),
        "all_reps_video_stride_steps": int(stride_steps),
        "all_reps_video_max_steps_rendered": int(n_frames * stride_steps),
        "all_reps_video_fps": int(fps),
        "all_reps_video_panel_size": int(img_size),
        "all_reps_video_size": [int(img_size) * 2, int(img_size) * 2],
        "all_reps_video_resize_method": str(resize_method),
    }


def _plot_maps(
    *,
    out_dir: Path,
    delta_h_map: np.ndarray,
    processed_map: np.ndarray | None,
    tau_steps: np.ndarray,
    window_start_steps: np.ndarray,
    score_by_tau: np.ndarray,
    selected_tau_steps: int,
    selected_score: float,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def heatmap(path: Path, data: np.ndarray, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[
                float(window_start_steps[0]),
                float(window_start_steps[-1]),
                float(tau_steps[0]),
                float(tau_steps[-1]),
            ],
            cmap="viridis",
        )
        ax.axhline(float(selected_tau_steps), color="white", linewidth=1.4, linestyle="--", alpha=0.9)
        ax.set_xlabel("window start step")
        ax.set_ylabel("tau steps")
        ax.set_title(title)
        ax.set_yticks(tau_steps)
        fig.colorbar(im, ax=ax, label="Delta-H")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    heatmap(
        out_dir / "delta_h_map.png",
        np.asarray(delta_h_map, dtype=np.float64),
        f"Delta-H map, selected tau={selected_tau_steps}, MSPD={selected_score:.6g}",
    )
    paths["delta_h_map_png"] = str(out_dir / "delta_h_map.png")

    if processed_map is not None:
        heatmap(
            out_dir / "delta_h_processed_map.png",
            np.asarray(processed_map, dtype=np.float64),
            f"Processed Delta-H map, selected tau={selected_tau_steps}, MSPD={selected_score:.6g}",
        )
        paths["delta_h_processed_map_png"] = str(out_dir / "delta_h_processed_map.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(tau_steps, score_by_tau, marker="o", color="#2a6fbb", linewidth=1.8)
    ax.axvline(float(selected_tau_steps), color="#d14b3a", linestyle="--", linewidth=1.3)
    ax.set_xlabel("tau steps")
    ax.set_ylabel("MSPD")
    ax.set_title("MSPD by tau for selected optimizer rep")
    fig.tight_layout()
    fig.savefig(out_dir / "mspd_by_tau.png", dpi=180)
    plt.close(fig)
    paths["mspd_by_tau_png"] = str(out_dir / "mspd_by_tau.png")
    return paths


def _metric_range_start(metric_cfg: dict[str, Any], flat: Any) -> int:
    return int(metric_cfg.get("range_start_steps", getattr(flat, "metric_range_start_steps", 0)))


def _metric_range_end(metric_cfg: dict[str, Any], flat: Any) -> int:
    return int(metric_cfg.get("range_end_steps", getattr(flat, "metric_range_end_steps", getattr(flat, "rollout_steps", 0))))


def _plot_all_rep_maps(
    *,
    out_dir: Path,
    delta_h_maps: np.ndarray,
    processed_maps: np.ndarray | None,
    tau_steps: np.ndarray,
    window_start_steps: np.ndarray,
    score_by_tau: np.ndarray,
    rep_scores: np.ndarray,
    selected_tau_steps: np.ndarray,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def heatmap_grid(path: Path, data: np.ndarray, title: str, colorbar_label: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True, sharey=True)
        vmax = float(np.nanmax(data))
        vmin = float(np.nanmin(data))
        im = None
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(
                data[i],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[
                    float(window_start_steps[0]),
                    float(window_start_steps[-1]),
                    float(tau_steps[0]),
                    float(tau_steps[-1]),
                ],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.axhline(float(selected_tau_steps[i]), color="white", linewidth=1.2, linestyle="--", alpha=0.9)
            ax.set_title(f"rep {i}: MSPD={float(rep_scores[i]):.6g}")
            ax.set_yticks(tau_steps)
        for ax in axes[-1, :]:
            ax.set_xlabel("window start step")
        for ax in axes[:, 0]:
            ax.set_ylabel("tau steps")
        fig.suptitle(title)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), label=colorbar_label, shrink=0.92)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    heatmap_grid(out_dir / "all_reps_delta_h_maps.png", delta_h_maps, "Delta-H maps for all optimizer launches", "Delta-H")
    paths["all_reps_delta_h_maps_png"] = str(out_dir / "all_reps_delta_h_maps.png")

    if processed_maps is not None and processed_maps.size > 0:
        heatmap_grid(
            out_dir / "all_reps_delta_h_processed_maps.png",
            processed_maps,
            "Processed Delta-H maps for all optimizer launches",
            "processed Delta-H",
        )
        paths["all_reps_delta_h_processed_maps_png"] = str(out_dir / "all_reps_delta_h_processed_maps.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i in range(score_by_tau.shape[0]):
        ax.plot(tau_steps, score_by_tau[i], marker="o", linewidth=1.5, label=f"rep {i}: {float(rep_scores[i]):.3g}")
    ax.set_xlabel("tau steps")
    ax.set_ylabel("MSPD")
    ax.set_title("MSPD by tau for all optimizer launches")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "all_reps_mspd_by_tau.png", dpi=180)
    plt.close(fig)
    paths["all_reps_mspd_by_tau_png"] = str(out_dir / "all_reps_mspd_by_tau.png")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    _activate_source_root(Path(args.source_root))
    flat = _flat_config(run_dir / "optimization_config.yaml")
    if args.metric_range_start_steps is not None:
        flat.metric_range_start_steps = int(args.metric_range_start_steps)
    if args.metric_range_end_steps is not None:
        flat.metric_range_end_steps = int(args.metric_range_end_steps)
    if args.render_mode is not None:
        flat.render_mode = str(args.render_mode)
    substrate, metric_cfg, metric_loss_fn = _build_context(flat, include_maps=True)
    tau_extra_dims = 1 if str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid" else 0
    substrate_param_dims = int(substrate.n_params)
    candidate = _best_candidate_and_keys(run_dir, flat, substrate_param_dims, tau_extra_dims)
    loss_raw, info_raw = _metric_chunk_raw(
        flat=flat,
        substrate=substrate,
        metric_cfg=metric_cfg,
        metric_loss_fn=metric_loss_fn,
        candidate=candidate,
        substrate_param_dims=substrate_param_dims,
        tau_extra_dims=tau_extra_dims,
    )

    local_idx = int(candidate["chunk_local_idx"])
    per_rep_score = np.asarray(info_raw["score"][local_idx], dtype=np.float64)
    rep_mode = str(args.rep_index).strip().lower()
    if rep_mode == "all":
        rep_idx = -1
    elif rep_mode == "best":
        rep_idx = int(np.nanargmax(per_rep_score))
    else:
        rep_idx = int(args.rep_index)
    if rep_idx != -1 and (rep_idx < 0 or rep_idx >= per_rep_score.shape[0]):
        raise ValueError(f"rep-index must be in [0, {per_rep_score.shape[0] - 1}] or 'best', got {args.rep_index!r}.")

    if rep_idx == -1:
        delta_h_maps = np.asarray(info_raw["delta_h_map"][local_idx], dtype=np.float32)
        processed_maps = (
            np.asarray(info_raw["delta_h_processed_map"][local_idx], dtype=np.float32)
            if "delta_h_processed_map" in info_raw
            else None
        )
        tau_steps = np.asarray(info_raw["tau_steps"][local_idx, 0], dtype=np.int64)
        window_start_steps = np.asarray(info_raw["window_start_steps"][local_idx, 0], dtype=np.int64)
        score_by_tau = np.asarray(info_raw["score_by_tau"][local_idx], dtype=np.float64)
        selected_tau_steps_all = np.asarray(info_raw["tau_best_steps"][local_idx], dtype=np.int64)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "optimizer_eval_all_reps_metric_maps.npz",
            delta_h_map_by_rep=delta_h_maps,
            delta_h_processed_map_by_rep=processed_maps if processed_maps is not None else np.asarray([]),
            tau_steps=tau_steps,
            window_start_steps=window_start_steps,
            score_by_tau_by_rep=score_by_tau,
            per_rep_score=per_rep_score,
            loss_raw=np.asarray(loss_raw[local_idx], dtype=np.float32),
            selected_tau_steps=selected_tau_steps_all,
        )
        plot_paths = _plot_all_rep_maps(
            out_dir=out_dir,
            delta_h_maps=delta_h_maps,
            processed_maps=processed_maps,
            tau_steps=tau_steps,
            window_start_steps=window_start_steps,
            score_by_tau=score_by_tau,
            rep_scores=per_rep_score,
            selected_tau_steps=selected_tau_steps_all,
        )
        for i_rep in range(per_rep_score.shape[0]):
            _plot_maps(
                out_dir=out_dir / f"rep_{i_rep}",
                delta_h_map=delta_h_maps[i_rep],
                processed_map=processed_maps[i_rep] if processed_maps is not None else None,
                tau_steps=tau_steps,
                window_start_steps=window_start_steps,
                score_by_tau=score_by_tau[i_rep],
                selected_tau_steps=int(selected_tau_steps_all[i_rep]),
                selected_score=float(per_rep_score[i_rep]),
            )

        video_info: dict[str, Any] = {}
        if not args.skip_video:
            params = np.asarray(candidate["params"], dtype=np.float32)
            video_info = _render_state_grid_video(
                flat=flat,
                substrate=substrate,
                params=params,
                rep_keys=candidate["keys"],
                rep_scores=per_rep_score,
                output=out_dir / "optimizer_eval_all_reps_state_grid.mp4",
                img_size=int(args.img_size),
                fps=int(args.fps),
                codec=str(args.codec),
                stride_steps=int(args.video_stride_steps),
                max_steps=int(args.video_max_steps),
                frame_batch_size=int(args.frame_batch_size),
                resize_method=str(args.video_resize_method),
            )

        summary = {
            "run_dir": str(run_dir),
            "source_root": str(args.source_root),
            "output_dir": str(out_dir),
            "best_iter": int(candidate["best_iter"]),
            "pop_idx": int(candidate["pop_idx"]),
            "chunk_start": int(candidate["chunk_start"]),
            "chunk_local_idx": int(candidate["chunk_local_idx"]),
            "rep_mode": "all",
            "per_rep_score": [float(x) for x in per_rep_score],
            "optimizer_mean_score": float(np.mean(per_rep_score)),
            "min_rep_idx": int(np.nanargmin(per_rep_score)),
            "max_rep_idx": int(np.nanargmax(per_rep_score)),
            "metric_range_start_steps": _metric_range_start(metric_cfg, flat),
            "metric_range_end_steps": _metric_range_end(metric_cfg, flat),
            "metric_n_windows": int(delta_h_maps.shape[-1]),
            "metric_maps_npz": str(out_dir / "optimizer_eval_all_reps_metric_maps.npz"),
            **plot_paths,
            **video_info,
        }
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        return summary

    score = float(info_raw["score"][local_idx, rep_idx])
    selected_tau_steps = int(info_raw["tau_best_steps"][local_idx, rep_idx])
    delta_h_map = _select(info_raw["delta_h_map"], local_idx, rep_idx)
    processed_map = _select(info_raw["delta_h_processed_map"], local_idx, rep_idx) if "delta_h_processed_map" in info_raw else None
    tau_steps = _select(info_raw["tau_steps"], local_idx, rep_idx).astype(np.int64)
    window_start_steps = _select(info_raw["window_start_steps"], local_idx, rep_idx).astype(np.int64)
    score_by_tau = _select(info_raw["score_by_tau"], local_idx, rep_idx).astype(np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "optimizer_eval_metric_maps.npz",
        delta_h_map=np.asarray(delta_h_map, dtype=np.float32),
        delta_h_processed_map=np.asarray(processed_map, dtype=np.float32) if processed_map is not None else np.asarray([]),
        tau_steps=tau_steps,
        window_start_steps=window_start_steps,
        score_by_tau=score_by_tau,
        per_rep_score=per_rep_score,
        loss_raw=np.asarray(loss_raw[local_idx], dtype=np.float32),
        selected_rep_idx=np.asarray(rep_idx, dtype=np.int32),
        selected_tau_steps=np.asarray(selected_tau_steps, dtype=np.int32),
    )
    plot_paths = _plot_maps(
        out_dir=out_dir,
        delta_h_map=delta_h_map,
        processed_map=processed_map,
        tau_steps=tau_steps,
        window_start_steps=window_start_steps,
        score_by_tau=score_by_tau,
        selected_tau_steps=selected_tau_steps,
        selected_score=score,
    )

    params = np.asarray(candidate["params"], dtype=np.float32)
    video_info: dict[str, Any] = {}
    if not args.skip_video:
        video_info = _render_state_video(
            flat=flat,
            substrate=substrate,
            params=params,
            rep_key=candidate["keys"][rep_idx],
            output=out_dir / "optimizer_eval_state_video.mp4",
            img_size=int(args.img_size),
            fps=int(args.fps),
            codec=str(args.codec),
            stride_steps=int(args.video_stride_steps),
            max_steps=int(args.video_max_steps),
            frame_batch_size=int(args.frame_batch_size),
            resize_method=str(args.video_resize_method),
        )

    summary = {
        "run_dir": str(run_dir),
        "source_root": str(args.source_root),
        "output_dir": str(out_dir),
        "best_iter": int(candidate["best_iter"]),
        "pop_idx": int(candidate["pop_idx"]),
        "chunk_start": int(candidate["chunk_start"]),
        "chunk_local_idx": int(candidate["chunk_local_idx"]),
        "selected_rep_idx": int(rep_idx),
        "per_rep_score": [float(x) for x in per_rep_score],
        "optimizer_mean_score": float(np.mean(per_rep_score)),
        "selected_rep_score": float(score),
        "selected_tau_steps": int(selected_tau_steps),
        "metric_range_start_steps": _metric_range_start(metric_cfg, flat),
        "metric_range_end_steps": _metric_range_end(metric_cfg, flat),
        "metric_n_windows": int(delta_h_map.shape[-1]),
        "metric_maps_npz": str(out_dir / "optimizer_eval_metric_maps.npz"),
        **plot_paths,
        **video_info,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render exact Flow-Lenia optimizer evaluation video and Delta-H map.")
    parser.add_argument("run_dir")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rep-index", default="best", help="'best', 'all', or integer rep index in optimization bs.")
    parser.add_argument("--render-mode", default="Pcolor")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--video-stride-steps", type=int, default=1000)
    parser.add_argument("--video-max-steps", type=int, default=300000)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--metric-range-start-steps", type=int, default=None)
    parser.add_argument("--metric-range-end-steps", type=int, default=None)
    parser.add_argument("--skip-video", action="store_true", help="Compute maps/scores only.")
    parser.add_argument(
        "--video-resize-method",
        default="linear",
        choices=("nearest", "linear", "cubic", "lanczos3", "lanczos5"),
        help="Resize method used only for video rendering; metric computation is unchanged.",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
