import os
import sys
from typing import List, Optional, Tuple

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
) -> int:
    if not steps:
        return file_idx
    start_step = int(steps[0])
    end_step = int(steps[-1])
    start_sec = start_step / fps
    end_sec = end_step / fps

    dtype = np.float16 if use_fp16 else np.float32
    arrP = np.stack(snaps_P, axis=0).astype(dtype)
    meta = {
        "steps": np.array(steps, dtype=np.int64),
        "fps": np.array(fps, dtype=np.float32),
    }
    fname = f"P_steps_{start_step}_{end_step}__secs_{start_sec:.3f}_{end_sec:.3f}__idx_{file_idx:04d}.npz"
    path = os.path.join(out_dir, fname)

    payload = {"P": arrP, **meta}
    if snaps_A is not None:
        payload["A"] = np.stack(snaps_A, axis=0).astype(dtype)
    if snaps_F is not None:
        payload["F"] = np.stack(snaps_F, axis=0).astype(dtype)
    if snaps_rgb is not None:
        payload["rgb"] = np.stack(snaps_rgb, axis=0).astype(np.uint8)

    np.savez_compressed(path, **payload)
    saved = ["P"]
    if snaps_A is not None:
        saved.append("A")
    if snaps_F is not None:
        saved.append("F")
    if snaps_rgb is not None:
        saved.append("rgb")
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
    if args.substrate == "lenia_flow":
        kw = util.flow_lenia_kwargs_from_args(args)
        kw["debug_return_F"] = save_F
        substrate = substrates.create_substrate(args.substrate, **kw)
    else:
        if save_F:
            raise ValueError("save_F is only supported for substrate='lenia_flow'.")
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

    need_A = save_A or save_rgb

    # Jitted microbatch stepper: returns (state_next, Pbuf, Abuf_or_dummy, Fbuf_or_dummy)
    def build_batch_stepper(mb: int):
        def run_batch(state, rng_in):
            rngs = jax.random.split(rng_in, mb)
            P0 = jnp.zeros((mb, *state["P"].shape), dtype=state["P"].dtype)
            if need_A:
                A0 = jnp.zeros((mb, *state["A"].shape), dtype=state["A"].dtype)
            else:
                A0 = jnp.zeros((1,), dtype=jnp.float32)
            if save_F:
                F0 = jnp.zeros((mb, *state["F"].shape), dtype=state["F"].dtype)
            else:
                F0 = jnp.zeros((1,), dtype=jnp.float32)

            def body(i, carry):
                st, Pbuf, Abuf, Fbuf = carry
                st = substrate.step_state(rngs[i], st, best_member)
                Pbuf = Pbuf.at[i].set(st["P"])
                if need_A:
                    Abuf = Abuf.at[i].set(st["A"])
                if save_F:
                    Fbuf = Fbuf.at[i].set(st["F"])
                return (st, Pbuf, Abuf, Fbuf)

            state_next, Pbuf, Abuf, Fbuf = jax.lax.fori_loop(0, mb, body, (state, P0, A0, F0))
            return state_next, Pbuf, Abuf, Fbuf

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
    file_idx = 0

    steps_done = 0
    pbar = tqdm(total=total_steps, desc="Simulating")
    while steps_done < total_steps:
        outer_b = min(batch_steps, total_steps - steps_done)
        remaining = outer_b
        while remaining > 0:
            mb = jit_microbatch if remaining >= jit_microbatch else remaining
            rng, _rng = split(rng)
            step_micro = get_stepper(mb)
            s, batch_P, batch_A, batch_F = step_micro(s, _rng)

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

                dtype = np.float16 if save_fp16 else np.float32
                for i_local, i_global in enumerate(idxs):
                    global_step = base_step + i_global
                    steps_buf.append(global_step)
                    snaps_P_buf.append(selP[i_local].astype(dtype))
                    if save_A:
                        assert selA is not None
                        snaps_A_buf.append(selA[i_local].astype(dtype))
                    if save_F:
                        assert selF is not None
                        snaps_F_buf.append(selF[i_local].astype(dtype))
                    if save_rgb:
                        assert selRGB is not None
                        snaps_rgb_buf.append(selRGB[i_local])

                    if len(snaps_P_buf) >= chunk_size:
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
                        )
                        steps_buf = []
                        snaps_P_buf = []
                        snaps_A_buf = []
                        snaps_F_buf = []
                        snaps_rgb_buf = []

            remaining -= mb
            steps_done += mb
            pbar.update(mb)
    pbar.close()
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
        )

    print(f"Finished simulation. Saved {file_idx} chunk files to {out_dir}")
    print(f"Best fitness: {np.array(best_fitness).item() if best_fitness is not None else None}")


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
