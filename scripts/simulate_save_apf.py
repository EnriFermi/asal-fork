import os
import signal
import sys
from collections import deque
import colorsys
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import substrates
import util


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/simulate_save_apf.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def _parse_optional_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in ("none", ""):
        return None
    return int(x)


def _parse_log_step_range(args) -> Tuple[Optional[int], Optional[int]]:
    step_range = getattr(args, "log_step_range", None)
    if step_range is not None:
        vals = list(OmegaConf.to_container(step_range, resolve=True) if OmegaConf.is_config(step_range) else step_range)
        if len(vals) == 0:
            return None, None
        if len(vals) != 2:
            raise ValueError(f"log_step_range must be [start, end]; got {vals}")
        start, end = vals
        return _parse_optional_int(start), _parse_optional_int(end)
    start = _parse_optional_int(getattr(args, "log_step_start", None))
    end = _parse_optional_int(getattr(args, "log_step_end", None))
    return start, end


def _in_step_range(step: int, start: Optional[int], end: Optional[int]) -> bool:
    if start is not None and step < start:
        return False
    if end is not None and step > end:
        return False
    return True


def _project_root() -> str:
    # scripts/.. -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_path(path: str, base_dir: str) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _init_lagrangian_points(
    A0: np.ndarray,
    *,
    n_particles: int,
    init_mode: str,
    seed: int,
    rt: Any,
) -> np.ndarray:
    if n_particles < 1:
        raise ValueError(f"lagrangian_n_particles must be >= 1, got {n_particles}.")

    sx, sy = int(A0.shape[0]), int(A0.shape[1])
    rng = np.random.default_rng(seed)
    mode = init_mode.strip().lower()

    if mode == "uniform":
        pts = np.empty((n_particles, 2), dtype=np.float32)
        pts[:, 0] = rng.uniform(0.5, sx - 0.5, size=n_particles)
        pts[:, 1] = rng.uniform(0.5, sy - 0.5, size=n_particles)
    elif mode == "mass":
        mass = np.clip(np.asarray(A0, dtype=np.float32).sum(axis=-1), 0.0, np.inf)
        flat = mass.reshape(-1)
        total = float(flat.sum())
        probs = None if total <= 0.0 else (flat / total)
        idx = rng.choice(flat.size, size=n_particles, replace=True, p=probs)
        iy = idx // sy
        ix = idx % sy
        jitter = rng.uniform(-0.49, 0.49, size=(n_particles, 2)).astype(np.float32)
        pts = np.stack((iy.astype(np.float32) + 0.5, ix.astype(np.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unknown lagrangian_init_mode={init_mode!r}. Use 'mass' or 'uniform'.")

    if str(getattr(rt, "border", "wall")) == "torus":
        pts[:, 0] = np.mod(pts[:, 0] - 0.5, sx) + 0.5
        pts[:, 1] = np.mod(pts[:, 1] - 0.5, sy) + 0.5
    else:
        lo = float(getattr(rt, "sigma", 0.0))
        hi_y = float(sx - getattr(rt, "sigma", 0.0))
        hi_x = float(sy - getattr(rt, "sigma", 0.0))
        pts[:, 0] = np.clip(pts[:, 0], lo, hi_y)
        pts[:, 1] = np.clip(pts[:, 1], lo, hi_x)

    return pts.astype(np.float32, copy=False)


def _make_distinct_colors_bgr(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cols = []
    for i in range(n):
        h = (i / float(max(1, n))) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for cv2 drawing
    return np.asarray(cols, dtype=np.uint8)


def _lagrangian_to_xy_pixels(points_yx: np.ndarray, h: int, w: int) -> np.ndarray:
    y = np.clip(np.rint(points_yx[:, 0] - 0.5), 0, h - 1).astype(np.int32)
    x = np.clip(np.rint(points_yx[:, 1] - 0.5), 0, w - 1).astype(np.int32)
    return np.stack([x, y], axis=-1)


def _frame_u8_from_snapshot(
    p_frame: np.ndarray,
    a_frame: Optional[np.ndarray],
    rgb_frame: Optional[np.ndarray],
) -> np.ndarray:
    if rgb_frame is not None:
        frame = np.asarray(rgb_frame)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        return frame

    p = np.asarray(p_frame, dtype=np.float32)
    if p.shape[-1] >= 3:
        p3 = p[..., :3]
    else:
        reps = int(np.ceil(3 / p.shape[-1]))
        p3 = np.tile(p, (1, 1, reps))[..., :3]

    if a_frame is not None:
        a = np.asarray(a_frame, dtype=np.float32)
        inten = np.sum(a, axis=-1, keepdims=True)
        rgb = np.clip(inten * p3, 0.0, 1.0)
    else:
        mn = float(np.min(p3))
        mx = float(np.max(p3))
        if mx <= mn:
            rgb = np.zeros_like(p3, dtype=np.float32)
        else:
            rgb = (p3 - mn) / (mx - mn + 1e-8)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def save_chunk(
    out_dir: str,
    fps: float,
    steps: List[int],
    snaps_P: List[np.ndarray],
    file_idx: int,
    snaps_A: Optional[List[np.ndarray]] = None,
    snaps_F: Optional[List[np.ndarray]] = None,
    use_fp16: bool = True,
    snaps_rgb: Optional[List[np.ndarray]] = None,
    snaps_lagrangian: Optional[List[np.ndarray]] = None,
    snaps_lagrangian_c: Optional[List[np.ndarray]] = None,
    compress: bool = True,
) -> int:
    if not steps:
        return file_idx
    start_step = int(steps[0])
    end_step = int(steps[-1])
    start_sec = start_step / fps
    end_sec = end_step / fps

    dtype = np.float16 if use_fp16 else np.float32
    arrP = np.stack(snaps_P, axis=0)
    if arrP.dtype != dtype:
        arrP = arrP.astype(dtype, copy=False)
    meta = {
        "steps": np.array(steps, dtype=np.int64),
        "fps": np.array(fps, dtype=np.float32),
    }
    fname = f"P_steps_{start_step}_{end_step}__secs_{start_sec:.3f}_{end_sec:.3f}__idx_{file_idx:04d}.npz"
    path = os.path.join(out_dir, fname)

    payload = {"P": arrP, **meta}
    if snaps_A is not None:
        arrA = np.stack(snaps_A, axis=0)
        if arrA.dtype != dtype:
            arrA = arrA.astype(dtype, copy=False)
        payload["A"] = arrA
    if snaps_F is not None:
        arrF = np.stack(snaps_F, axis=0)
        if arrF.dtype != dtype:
            arrF = arrF.astype(dtype, copy=False)
        payload["F"] = arrF
    if snaps_rgb is not None:
        payload["rgb"] = np.stack(snaps_rgb, axis=0).astype(np.uint8)
    if snaps_lagrangian is not None:
        payload["lagrangian_xy"] = np.stack(snaps_lagrangian, axis=0).astype(np.float32, copy=False)
    if snaps_lagrangian_c is not None:
        payload["lagrangian_c"] = np.stack(snaps_lagrangian_c, axis=0).astype(np.int32, copy=False)

    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    saved = ["P"]
    if snaps_A is not None:
        saved.append("A")
    if snaps_F is not None:
        saved.append("F")
    if snaps_rgb is not None:
        saved.append("rgb")
    if snaps_lagrangian is not None:
        saved.append("lagrangian_xy")
    if snaps_lagrangian_c is not None:
        saved.append("lagrangian_c")
    print(f"Saved {len(steps)} snapshots ({','.join(saved)}) to {path}")
    return file_idx + 1


def main(cfg, args):
    proj_root = _project_root()
    save_dir = _resolve_path(getattr(args, "save_dir"), proj_root)

    best_path = os.path.join(save_dir, "best.pkl")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best.pkl not found in {save_dir}. Ensure main_opt.py saved results with --save_dir.")

    best_obj = util.load_pkl(save_dir, "best")
    if isinstance(best_obj, tuple) and len(best_obj) == 2:
        best_member, best_fitness = best_obj
    else:
        best_member, best_fitness = best_obj, None
    best_member = jnp.asarray(best_member)

    # Optional override: take params from best_traj.pkl at a specific iteration
    traj_iter = getattr(args, "traj_iter", None)
    if traj_iter is not None:
        traj_iter = int(traj_iter)
        traj_path = os.path.join(save_dir, "best_traj.pkl")
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"traj_iter={traj_iter} requested but best_traj.pkl not found in {args.save_dir}. "
                f"Re-run main_opt.py with code that saves best_traj.pkl."
            )
        traj = util.load_pkl(save_dir, "best_traj")
        params_arr = traj.get("params", None)
        if params_arr is None:
            raise ValueError(f"best_traj.pkl in {args.save_dir} does not contain 'params'.")
        n_iters_available = int(np.asarray(params_arr).shape[0])
        if traj_iter < 0 or traj_iter >= n_iters_available:
            raise ValueError(f"traj_iter {traj_iter} out of range [0, {n_iters_available-1}]")
        best_member = jnp.asarray(params_arr[traj_iter])

        loss_arr = traj.get("loss", None)
        if loss_arr is not None and np.asarray(loss_arr).shape[0] == n_iters_available:
            best_fitness = np.asarray(loss_arr)[traj_iter]

    save_F = bool(getattr(args, "save_F", True))
    save_lagrangian = bool(getattr(args, "save_lagrangian", False))
    need_F_from_state = bool(save_F or save_lagrangian)
    if args.substrate == "lenia_flow":
        kw = util.flow_lenia_kwargs_from_args(args)
        kw["debug_return_F"] = need_F_from_state
        substrate = substrates.create_substrate(args.substrate, **kw)
    else:
        if need_F_from_state:
            raise ValueError("save_F/save_lagrangian are only supported for substrate='lenia_flow'.")
        substrate = substrates.create_substrate(args.substrate)

    substrate = substrates.FlattenSubstrateParameters(substrate)

    # Validate parameter length (common source of silent mismatches)
    param_len = int(np.asarray(best_member).size)
    expected_len = int(np.asarray(substrate.default_params(jax.random.PRNGKey(0))).size)
    if param_len != expected_len:
        raise ValueError(
            f"Loaded parameter length {param_len} does not match substrate expectation {expected_len}. "
            f"Check that training and simulation use the same substrate configuration."
        )

    total_steps = int(getattr(args, "rollout_steps", substrate.rollout_steps) or substrate.rollout_steps)
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    seed = int(getattr(args, "seed", 0))
    rng = jax.random.PRNGKey(seed)
    s = substrate.init_state(rng, best_member)

    # Logging config
    snapshot_interval = max(1, int(getattr(args, "snapshot_interval", 100)))
    chunk_size = max(1, int(getattr(args, "snapshots_per_file", 50)))
    fps = float(getattr(args, "fps", 250))
    batch_steps = int(getattr(args, "batch_steps", 256))
    jit_microbatch = int(getattr(args, "jit_microbatch", 64))

    log_start, log_end = _parse_log_step_range(args)

    out_dir_cfg = getattr(args, "output_dir", None)
    if out_dir_cfg is None or (isinstance(out_dir_cfg, str) and out_dir_cfg.strip().lower() in ("none", "")):
        out_dir = os.path.join(save_dir, "snapshots_P")
    else:
        out_dir_cfg = str(out_dir_cfg)
        out_dir = _resolve_path(out_dir_cfg, proj_root)
    os.makedirs(out_dir, exist_ok=True)

    save_A = bool(getattr(args, "save_A", True))
    save_rgb = bool(getattr(args, "save_rgb", False))
    save_fp16 = bool(getattr(args, "save_fp16", True))
    compress = bool(getattr(args, "compress", True))
    save_dtype = np.float16 if save_fp16 else np.float32

    save_lagrangian_video = bool(getattr(args, "save_lagrangian_video", False))
    lagrangian_video_output = getattr(args, "lagrangian_video_output", None)
    lagrangian_video_codec = str(getattr(args, "lagrangian_video_codec", "libx264"))
    lagrangian_video_macro_block_size = int(getattr(args, "lagrangian_video_macro_block_size", 1))
    lagrangian_video_trail_length = int(getattr(args, "lagrangian_video_trail_length", 40))
    lagrangian_video_line_thickness = int(getattr(args, "lagrangian_video_line_thickness", 1))
    lagrangian_video_draw_points = bool(getattr(args, "lagrangian_video_draw_points", True))
    lagrangian_video_point_radius = int(getattr(args, "lagrangian_video_point_radius", 2))
    lagrangian_video_draw_step_text = bool(getattr(args, "lagrangian_video_draw_step_text", False))
    lagrangian_video_particle_stride = int(getattr(args, "lagrangian_video_particle_stride", 1))
    lagrangian_video_max_particles = _parse_optional_int(getattr(args, "lagrangian_video_max_particles", None))
    lagrangian_video_fps_raw = getattr(args, "lagrangian_video_fps", None)
    if isinstance(lagrangian_video_fps_raw, str) and lagrangian_video_fps_raw.strip().lower() in ("none", ""):
        lagrangian_video_fps_raw = None
    lagrangian_video_fps = (
        float(lagrangian_video_fps_raw)
        if lagrangian_video_fps_raw is not None
        else float(fps) / float(max(1, snapshot_interval))
    )

    lagrangian_n_particles = int(getattr(args, "lagrangian_n_particles", 256))
    lagrangian_seed = int(getattr(args, "lagrangian_seed", seed))
    lagrangian_init_mode = str(getattr(args, "lagrangian_init_mode", "mass")).strip().lower()
    lagrangian_flow_reduce = str(getattr(args, "lagrangian_flow_reduce", "mass_weighted")).strip().lower()
    lagrangian_flow_channel = _parse_optional_int(getattr(args, "lagrangian_flow_channel", -1))
    lagrangian_channel_mode = str(getattr(args, "lagrangian_channel_mode", "mix")).strip().lower()
    lagrangian_noise_model = str(getattr(args, "lagrangian_noise_model", "none")).strip().lower()
    lagrangian_diffusion_scale = float(getattr(args, "lagrangian_diffusion_scale", 1.0))
    save_lagrangian_channels = bool(getattr(args, "save_lagrangian_channels", False))
    lagrangian_unique_check_every_steps = int(getattr(args, "lagrangian_unique_check_every_steps", 0))
    if lagrangian_flow_channel is None:
        lagrangian_flow_channel = -1
    if lagrangian_flow_channel < -1:
        raise ValueError(
            f"lagrangian_flow_channel must be >= -1 or null, got {lagrangian_flow_channel}."
        )
    if lagrangian_channel_mode not in ("mix", "fixed", "resample"):
        raise ValueError(
            f"lagrangian_channel_mode must be one of ('mix','fixed','resample'), got {lagrangian_channel_mode!r}."
        )
    if lagrangian_flow_reduce not in ("mass_weighted", "mean"):
        raise ValueError(
            f"lagrangian_flow_reduce must be 'mass_weighted' or 'mean', got {lagrangian_flow_reduce!r}."
        )
    if lagrangian_noise_model not in ("none", "rt_box", "gaussian"):
        raise ValueError(
            f"lagrangian_noise_model must be one of ('none','rt_box','gaussian'), got {lagrangian_noise_model!r}."
        )
    if lagrangian_diffusion_scale < 0:
        raise ValueError(f"lagrangian_diffusion_scale must be >= 0, got {lagrangian_diffusion_scale}.")
    if lagrangian_unique_check_every_steps < 0:
        raise ValueError(
            f"lagrangian_unique_check_every_steps must be >= 0, got {lagrangian_unique_check_every_steps}."
        )
    if save_lagrangian_video and not save_lagrangian:
        raise ValueError("save_lagrangian_video=true requires save_lagrangian=true.")
    if lagrangian_video_trail_length < 1:
        raise ValueError(
            f"lagrangian_video_trail_length must be >= 1, got {lagrangian_video_trail_length}."
        )
    if lagrangian_video_line_thickness < 1:
        raise ValueError(
            f"lagrangian_video_line_thickness must be >= 1, got {lagrangian_video_line_thickness}."
        )
    if lagrangian_video_point_radius < 1:
        raise ValueError(
            f"lagrangian_video_point_radius must be >= 1, got {lagrangian_video_point_radius}."
        )
    if lagrangian_video_particle_stride < 1:
        raise ValueError(
            f"lagrangian_video_particle_stride must be >= 1, got {lagrangian_video_particle_stride}."
        )
    if lagrangian_video_fps <= 0:
        raise ValueError(f"lagrangian_video_fps must be > 0, got {lagrangian_video_fps}.")

    smooth_F = bool(getattr(args, "smooth_F", False))
    smooth_F_window = int(getattr(args, "smooth_F_window", 1))
    if smooth_F_window < 1:
        raise ValueError(f"smooth_F_window must be >= 1, got {smooth_F_window}.")
    if smooth_F_window % 2 == 0:
        raise ValueError(f"smooth_F_window must be odd for centered smoothing, got {smooth_F_window}.")
    smooth_F_enabled = bool(save_F and smooth_F and smooth_F_window > 1)
    if smooth_F and not save_F:
        print("smooth_F=true ignored because save_F=false.")
    if smooth_F_enabled:
        print(
            f"Enabled centered F smoothing: window={smooth_F_window} snapshots "
            f"(radius={smooth_F_window // 2}), snapshot_interval={snapshot_interval}"
        )

    need_A = save_A or save_rgb

    lag_xy = jnp.zeros((1, 2), dtype=jnp.float32)
    lag_ch = jnp.zeros((1,), dtype=jnp.int32)
    lag_unique_initial = 0
    lag_unique_next_check_step = lagrangian_unique_check_every_steps
    rt = None
    if save_lagrangian:
        base_substrate = substrate.substrate if hasattr(substrate, "substrate") else substrate
        rt = getattr(base_substrate, "RT", None)
        if rt is None or not hasattr(rt, "advect_particles"):
            raise RuntimeError(
                "Lagrangian tracking requires ReintegrationTracking.advect_particles, "
                "but substrate RT is unavailable."
            )
        lag_xy0 = _init_lagrangian_points(
            np.asarray(s["A"]),
            n_particles=lagrangian_n_particles,
            init_mode=lagrangian_init_mode,
            seed=lagrangian_seed,
            rt=rt,
        )
        lag_xy = jnp.asarray(lag_xy0, dtype=jnp.float32)
        lag_unique_initial = int(np.unique(lag_xy0, axis=0).shape[0])
        lag_ch0 = np.zeros((lagrangian_n_particles,), dtype=np.int32)
        if lagrangian_channel_mode in ("fixed", "resample") and int(s["F"].shape[-1]) > 1:
            lag_ch_key = jax.random.PRNGKey(lagrangian_seed + 7919)
            lag_ch = rt.sample_point_channels(lag_xy, s["A"], lag_ch_key)
            lag_ch0 = np.asarray(lag_ch, dtype=np.int32)
        else:
            lag_ch = jnp.zeros((lagrangian_n_particles,), dtype=jnp.int32)

        meta_path = os.path.join(out_dir, "lagrangian_meta.npz")
        np.savez(
            meta_path,
            initial_xy=lag_xy0,
            particle_ids=np.arange(lagrangian_n_particles, dtype=np.int32),
            initial_channels=lag_ch0,
            init_mode=np.array(lagrangian_init_mode),
            flow_reduce=np.array(lagrangian_flow_reduce),
            flow_channel=np.array(lagrangian_flow_channel, dtype=np.int32),
            channel_mode=np.array(lagrangian_channel_mode),
            noise_model=np.array(lagrangian_noise_model),
            diffusion_scale=np.array(lagrangian_diffusion_scale, dtype=np.float32),
            seed=np.array(lagrangian_seed, dtype=np.int64),
            snapshot_interval=np.array(snapshot_interval, dtype=np.int32),
        )
        print(
            f"Enabled explicit Lagrangian tracking: n={lagrangian_n_particles}, "
            f"init={lagrangian_init_mode}, channel_mode={lagrangian_channel_mode}, "
            f"reduce={lagrangian_flow_reduce}, channel={lagrangian_flow_channel}, "
            f"noise={lagrangian_noise_model}, dscale={lagrangian_diffusion_scale}, "
            f"meta={meta_path}"
        )
        if lagrangian_unique_check_every_steps > 0:
            print(
                f"Enabled lagrangian uniqueness monitor: every {lagrangian_unique_check_every_steps} steps."
            )

    video_writer = None
    video_cv2 = None
    video_history: List[np.ndarray] = []
    video_keep_ids = np.zeros((0,), dtype=np.int32)
    video_colors_bgr = np.zeros((0, 3), dtype=np.uint8)
    video_out_path = None
    video_frames_written = 0
    if save_lagrangian_video:
        try:
            import imageio  # type: ignore
        except Exception as e:
            raise ImportError(
                "save_lagrangian_video=true requires imageio. Install with `pip install imageio`."
            ) from e
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise ImportError(
                "save_lagrangian_video=true requires opencv-python for overlay drawing."
            ) from e

        if lagrangian_video_output is None or (
            isinstance(lagrangian_video_output, str) and lagrangian_video_output.strip().lower() in ("none", "")
        ):
            video_out_path = os.path.join(out_dir, "lagrangian_overlay.mp4")
        else:
            video_out_path = _resolve_path(str(lagrangian_video_output), proj_root)
        os.makedirs(os.path.dirname(video_out_path), exist_ok=True)

        video_keep_ids = np.arange(lagrangian_n_particles, dtype=np.int32)[::lagrangian_video_particle_stride]
        if lagrangian_video_max_particles is not None:
            video_keep_ids = video_keep_ids[: max(0, int(lagrangian_video_max_particles))]
        if video_keep_ids.size == 0:
            raise ValueError("No particles selected for lagrangian video. Check stride/max_particles settings.")

        video_colors_bgr = _make_distinct_colors_bgr(int(video_keep_ids.size))
        video_cv2 = cv2
        video_writer = imageio.get_writer(
            video_out_path,
            fps=lagrangian_video_fps,
            codec=lagrangian_video_codec,
            macro_block_size=lagrangian_video_macro_block_size,
        )
        print(
            f"Enabled lagrangian overlay video: {video_out_path}, fps={lagrangian_video_fps:.3f}, "
            f"trail={lagrangian_video_trail_length}, draw={video_keep_ids.size}/{lagrangian_n_particles} particles."
        )

    # Jitted microbatch stepper:
    # returns (
    #   state_next, lag_xy_next, lag_ch_next,
    #   Pbuf, Abuf_or_dummy, Fbuf_or_dummy, Lbuf_or_dummy, Cbuf_or_dummy
    # )
    def build_batch_stepper(mb: int):
        def run_batch(state, lag_points, lag_channels, rng_in):
            # Keep simulation RNG stream identical to pre-lagrangian implementation.
            rngs = jax.random.split(rng_in, mb)
            # Derive an independent stream for lagrangian stochastic updates.
            lag_root = jax.random.fold_in(rng_in, jnp.uint32(0x4C4147))
            lag_rngs = jax.random.split(lag_root, mb)
            P0 = jnp.zeros((mb, *state["P"].shape), dtype=state["P"].dtype)
            if need_A:
                A0 = jnp.zeros((mb, *state["A"].shape), dtype=state["A"].dtype)
            else:
                A0 = jnp.zeros((1,), dtype=jnp.float32)
            if need_F_from_state:
                F0 = jnp.zeros((mb, *state["F"].shape), dtype=state["F"].dtype)
            else:
                F0 = jnp.zeros((1,), dtype=jnp.float32)
            if save_lagrangian:
                L0 = jnp.zeros((mb, lag_points.shape[0], 2), dtype=lag_points.dtype)
            else:
                L0 = jnp.zeros((1,), dtype=jnp.float32)
            if save_lagrangian and save_lagrangian_channels:
                C0 = jnp.zeros((mb, lag_channels.shape[0]), dtype=lag_channels.dtype)
            else:
                C0 = jnp.zeros((1,), dtype=jnp.int32)

            def body(i, carry):
                st, lag_xy_i, lag_ch_i, Pbuf, Abuf, Fbuf, Lbuf, Cbuf = carry
                st = substrate.step_state(rngs[i], st, best_member)
                Pbuf = Pbuf.at[i].set(st["P"])
                if need_A:
                    Abuf = Abuf.at[i].set(st["A"])
                if need_F_from_state:
                    Fbuf = Fbuf.at[i].set(st["F"])
                if save_lagrangian:
                    lag_xy_i, lag_ch_i = rt.advect_particles(
                        points=lag_xy_i,
                        F=st["F"],
                        A=st["A"],
                        channel=lagrangian_flow_channel,
                        reduce=lagrangian_flow_reduce,
                        point_channels=lag_ch_i,
                        channel_mode=lagrangian_channel_mode,
                        key=lag_rngs[i],
                        noise_model=lagrangian_noise_model,
                        diffusion_scale=lagrangian_diffusion_scale,
                    )
                    Lbuf = Lbuf.at[i].set(lag_xy_i)
                    if save_lagrangian_channels:
                        Cbuf = Cbuf.at[i].set(lag_ch_i)
                return (st, lag_xy_i, lag_ch_i, Pbuf, Abuf, Fbuf, Lbuf, Cbuf)

            state_next, lag_next, lag_ch_next, Pbuf, Abuf, Fbuf, Lbuf, Cbuf = jax.lax.fori_loop(
                0, mb, body, (state, lag_points, lag_channels, P0, A0, F0, L0, C0)
            )
            return state_next, lag_next, lag_ch_next, Pbuf, Abuf, Fbuf, Lbuf, Cbuf

        return jax.jit(run_batch)

    _stepper_cache = {}

    def get_stepper(mb: int):
        mb = int(mb)
        if mb not in _stepper_cache:
            _stepper_cache[mb] = build_batch_stepper(mb)
        return _stepper_cache[mb]

    steps_buf: List[int] = []
    snaps_P_buf: List[np.ndarray] = []
    snaps_A_buf: List[np.ndarray] = []
    snaps_F_buf: List[np.ndarray] = []
    snaps_rgb_buf: List[np.ndarray] = []
    snaps_lagrangian_buf: List[np.ndarray] = []
    snaps_lagrangian_c_buf: List[np.ndarray] = []
    file_idx = 0

    # Streaming state for centered smoothing over snapshot index.
    smooth_radius = smooth_F_window // 2
    smooth_items = deque()
    smooth_start_idx = 0
    smooth_next_emit_idx = 0
    smooth_seen = 0

    def flush_chunk_if_needed():
        nonlocal file_idx, steps_buf, snaps_P_buf, snaps_A_buf, snaps_F_buf, snaps_rgb_buf, snaps_lagrangian_buf, snaps_lagrangian_c_buf
        if len(snaps_P_buf) < chunk_size:
            return
        file_idx = save_chunk(
            out_dir,
            fps,
            steps_buf,
            snaps_P_buf,
            file_idx,
            snaps_A_buf if save_A else None,
            snaps_F_buf if save_F else None,
            use_fp16=save_fp16,
            snaps_rgb=snaps_rgb_buf if save_rgb else None,
            snaps_lagrangian=snaps_lagrangian_buf if save_lagrangian else None,
            snaps_lagrangian_c=snaps_lagrangian_c_buf if (save_lagrangian and save_lagrangian_channels) else None,
            compress=compress,
        )
        steps_buf = []
        snaps_P_buf = []
        snaps_A_buf = []
        snaps_F_buf = []
        snaps_rgb_buf = []
        snaps_lagrangian_buf = []
        snaps_lagrangian_c_buf = []

    def emit_snapshot(
        step: int,
        p_frame: np.ndarray,
        a_frame: Optional[np.ndarray],
        f_frame: Optional[np.ndarray],
        rgb_frame: Optional[np.ndarray],
        lag_frame: Optional[np.ndarray],
        lag_c_frame: Optional[np.ndarray],
    ):
        nonlocal video_frames_written
        steps_buf.append(step)
        snaps_P_buf.append(p_frame.astype(save_dtype, copy=False))
        if save_A:
            assert a_frame is not None
            snaps_A_buf.append(a_frame.astype(save_dtype, copy=False))
        if save_F:
            assert f_frame is not None
            snaps_F_buf.append(f_frame.astype(save_dtype, copy=False))
        if save_rgb:
            assert rgb_frame is not None
            snaps_rgb_buf.append(rgb_frame)
        if save_lagrangian:
            assert lag_frame is not None
            snaps_lagrangian_buf.append(lag_frame.astype(np.float32, copy=False))
            if save_lagrangian_channels:
                assert lag_c_frame is not None
                snaps_lagrangian_c_buf.append(lag_c_frame.astype(np.int32, copy=False))

        if video_writer is not None:
            assert lag_frame is not None
            assert video_cv2 is not None
            frame_u8 = _frame_u8_from_snapshot(p_frame, a_frame, rgb_frame)
            h, w = frame_u8.shape[:2]
            canvas = video_cv2.cvtColor(frame_u8, video_cv2.COLOR_RGB2BGR)

            lag_now = np.asarray(lag_frame, dtype=np.float32)
            lag_sel = lag_now[video_keep_ids]
            video_history.append(lag_sel)
            if len(video_history) > lagrangian_video_trail_length:
                del video_history[0]

            if len(video_history) > 1:
                hist = np.stack(video_history, axis=0)  # (L, Ndraw, 2)
                for i in range(hist.shape[1]):
                    poly = _lagrangian_to_xy_pixels(hist[:, i, :], h, w).reshape(-1, 1, 2)
                    video_cv2.polylines(
                        canvas,
                        [poly],
                        isClosed=False,
                        color=tuple(int(c) for c in video_colors_bgr[i]),
                        thickness=lagrangian_video_line_thickness,
                    )

            if lagrangian_video_draw_points:
                pts = _lagrangian_to_xy_pixels(lag_sel, h, w)
                for i, pt in enumerate(pts):
                    video_cv2.circle(
                        canvas,
                        (int(pt[0]), int(pt[1])),
                        lagrangian_video_point_radius,
                        tuple(int(c) for c in video_colors_bgr[i]),
                        -1,
                    )

            if lagrangian_video_draw_step_text:
                video_cv2.putText(
                    canvas,
                    f"step={int(step)}",
                    (12, 26),
                    video_cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    video_cv2.LINE_AA,
                )

            frame_rgb = video_cv2.cvtColor(canvas, video_cv2.COLOR_BGR2RGB)
            video_writer.append_data(frame_rgb)
            video_frames_written += 1
        flush_chunk_if_needed()

    def emit_smoothed_at_index(snapshot_idx: int, max_available_idx: int):
        nonlocal smooth_next_emit_idx, smooth_start_idx
        win_start = max(0, snapshot_idx - smooth_radius)
        win_end = min(max_available_idx, snapshot_idx + smooth_radius)
        center_off = snapshot_idx - smooth_start_idx
        start_off = win_start - smooth_start_idx
        end_off = win_end - smooth_start_idx
        if center_off < 0 or end_off >= len(smooth_items):
            raise RuntimeError(
                "F smoothing buffer underflow/overflow. "
                f"snapshot_idx={snapshot_idx}, smooth_start_idx={smooth_start_idx}, "
                f"len={len(smooth_items)}, win=[{win_start},{win_end}], max={max_available_idx}"
            )

        step, p_frame, a_frame, f_center, rgb_frame, lag_frame, lag_c_frame = smooth_items[center_off]
        assert f_center is not None
        f_mean = np.zeros_like(f_center, dtype=np.float32)
        n_acc = 0
        for off in range(start_off, end_off + 1):
            f_raw = smooth_items[off][3]
            assert f_raw is not None
            f_mean += f_raw.astype(np.float32, copy=False)
            n_acc += 1
        f_mean /= max(1, n_acc)

        emit_snapshot(step, p_frame, a_frame, f_mean, rgb_frame, lag_frame, lag_c_frame)
        smooth_next_emit_idx += 1

        min_needed_idx = max(0, smooth_next_emit_idx - smooth_radius)
        while smooth_start_idx < min_needed_idx and smooth_items:
            smooth_items.popleft()
            smooth_start_idx += 1

    def process_snapshot(
        step: int,
        p_frame: np.ndarray,
        a_frame: Optional[np.ndarray],
        f_frame: Optional[np.ndarray],
        rgb_frame: Optional[np.ndarray],
        lag_frame: Optional[np.ndarray],
        lag_c_frame: Optional[np.ndarray],
    ):
        nonlocal smooth_seen
        if not smooth_F_enabled:
            emit_snapshot(step, p_frame, a_frame, f_frame, rgb_frame, lag_frame, lag_c_frame)
            return

        snapshot_idx = smooth_seen
        smooth_seen += 1
        smooth_items.append((step, p_frame, a_frame, f_frame, rgb_frame, lag_frame, lag_c_frame))

        # Emit any index that already has enough future snapshots for centered window.
        while smooth_next_emit_idx <= snapshot_idx - smooth_radius:
            emit_smoothed_at_index(smooth_next_emit_idx, snapshot_idx)

    outputs_flushed = False

    def flush_pending_outputs():
        nonlocal file_idx, outputs_flushed
        if outputs_flushed:
            return
        outputs_flushed = True

        if smooth_F_enabled and smooth_seen > 0:
            tail_last_idx = smooth_seen - 1
            # Tail frames: use truncated centered window because future snapshots are unavailable.
            while smooth_next_emit_idx < smooth_seen:
                emit_smoothed_at_index(smooth_next_emit_idx, tail_last_idx)

        if snaps_P_buf:
            file_idx = save_chunk(
                out_dir,
                fps,
                steps_buf,
                snaps_P_buf,
                file_idx,
                snaps_A_buf if save_A else None,
                snaps_F_buf if save_F else None,
                use_fp16=save_fp16,
                snaps_rgb=snaps_rgb_buf if save_rgb else None,
                snaps_lagrangian=snaps_lagrangian_buf if save_lagrangian else None,
                snaps_lagrangian_c=snaps_lagrangian_c_buf if (save_lagrangian and save_lagrangian_channels) else None,
                compress=compress,
            )

    old_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _interrupt_handler)

    interrupted = False
    pbar = None
    try:
        steps_done = 0
        pbar = tqdm(total=total_steps, desc="Simulating")
        if save_lagrangian and lagrangian_unique_check_every_steps > 0 and lag_unique_initial > 0:
            pbar.set_postfix_str(f"Luniq={lag_unique_initial}")

        while steps_done < total_steps:
            outer_b = min(batch_steps, total_steps - steps_done)
            remaining = outer_b
            while remaining > 0:
                mb = jit_microbatch if remaining >= jit_microbatch else remaining
                rng, _rng = split(rng)
                step_micro = get_stepper(mb)
                s, lag_xy, lag_ch, batch_P, batch_A, batch_F, batch_L, batch_C = step_micro(s, lag_xy, lag_ch, _rng)

                base_step = steps_done
                idxs = [
                    i
                    for i in range(mb)
                    if (base_step + i) % snapshot_interval == 0
                    and _in_step_range(base_step + i, log_start, log_end)
                ]
                if idxs:
                    sel_idx = jnp.array(idxs)
                    selP = np.asarray(jnp.take(batch_P, sel_idx, axis=0))
                    selA = np.asarray(jnp.take(batch_A, sel_idx, axis=0)) if need_A else None
                    selF = np.asarray(jnp.take(batch_F, sel_idx, axis=0)) if save_F else None
                    selL = np.asarray(jnp.take(batch_L, sel_idx, axis=0)) if save_lagrangian else None
                    selC = (
                        np.asarray(jnp.take(batch_C, sel_idx, axis=0))
                        if (save_lagrangian and save_lagrangian_channels)
                        else None
                    )

                    selRGB = None
                    if save_rgb:
                        assert selA is not None
                        a_sum = np.sum(selA, axis=-1, keepdims=True)
                        if selP.shape[-1] >= 3:
                            p3 = selP[..., :3]
                        else:
                            reps = int(np.ceil(3 / selP.shape[-1]))
                            p3 = np.tile(selP, (1, 1, 1, reps))[..., :3]
                        rgb = np.clip(a_sum * p3, 0.0, 1.0)
                        selRGB = (rgb * 255).astype(np.uint8)

                    for i_local, i_global in enumerate(idxs):
                        global_step = base_step + i_global
                        a_frame = selA[i_local] if save_A else None
                        f_frame = selF[i_local] if save_F else None
                        rgb_frame = selRGB[i_local] if save_rgb else None
                        lag_frame = selL[i_local] if save_lagrangian else None
                        lag_c_frame = selC[i_local] if (save_lagrangian and save_lagrangian_channels) else None
                        process_snapshot(global_step, selP[i_local], a_frame, f_frame, rgb_frame, lag_frame, lag_c_frame)

                remaining -= mb
                steps_done += mb
                pbar.update(mb)

                if (
                    save_lagrangian
                    and lagrangian_unique_check_every_steps > 0
                    and lag_unique_initial > 0
                    and steps_done >= lag_unique_next_check_step
                ):
                    lag_now = np.asarray(lag_xy)
                    lag_unique_now = int(np.unique(lag_now, axis=0).shape[0])
                    pbar.set_postfix_str(f"Luniq={lag_unique_now}")
                    while lag_unique_next_check_step <= steps_done:
                        lag_unique_next_check_step += lagrangian_unique_check_every_steps

        flush_pending_outputs()

    except KeyboardInterrupt as e:
        interrupted = True
        print(f"Interrupted early ({e}); finalizing partial logs/video...")
        flush_pending_outputs()
    finally:
        if pbar is not None:
            pbar.close()
        signal.signal(signal.SIGTERM, old_sigterm)
        if video_writer is not None:
            try:
                video_writer.close()
                print(f"Saved lagrangian overlay video ({video_frames_written} frames) to {video_out_path}")
            except Exception as e:
                print(f"Failed to close lagrangian video writer: {e}")

    if interrupted:
        print("Simulation interrupted before requested total_steps.")
    print(f"Finished simulation. Saved {file_idx} chunk files to {out_dir}")
    print(f"Best fitness: {np.array(best_fitness).item() if best_fitness is not None else None}")


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
