"""
flow_drift_metrics.py

GPU-friendly (JAX) drift metrics for Flow-Lenia using logged A/F/P.

This module supports two common logging formats:
1) Per-timestep pickle files:
   - f"{save_pth}/{t}.pickle"
   - f"{save_pth}/{t}.zip"  (gzip-compressed pickle despite the ".zip" suffix)
2) Chunked NPZ snapshot files (like scripts/simulate_save_apf.py):
   - P_steps_<start>_<end>__secs_<t0>_<t1>__idx_<NNNN>.npz with arrays:
       steps: (T,), P: (T,H,W,K), optional A: (T,H,W,C), optional F: (T,H,W,2,C)

Jupyter usage example
---------------------
```python
import matplotlib.pyplot as plt
from flow_drift_metrics import compute_drift_timeseries

# For NPZ chunks logged by simulate_save_apf.py
out = compute_drift_timeseries(
    "experiments/log_apf/checkpoints/2602021501",
    t1=0,
    t2=20000,
    log_format="npz",   # or "auto"
    device="gpu",       # or "cpu", or "gpu:0"
    csv_path="drift.csv",
)

plt.figure(figsize=(10,4))
plt.plot(out["t"], out["drift_fast"], label="fast")
plt.plot(out["t"], out["drift_slow"], label="slow")
plt.legend()
plt.tight_layout()
```
"""

from __future__ import annotations

import csv
import gzip
import os
import pickle
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax


_NPZ_PATTERN = re.compile(
    r"P_steps_(\d+)_(\d+)__secs_([0-9.]+)_([0-9.]+)__idx_(\d+)\.npz$"
)


def resolve_device(device: Union[str, int, jax.Device, None] = "gpu") -> jax.Device:
    """
    Resolve a user-friendly device selector to a concrete JAX device.

    device examples:
      - "gpu", "gpu:0", "cpu", "tpu"
      - 0 (global device index)
      - jax.Device instance
      - None / "default" -> jax.devices()[0]
    """
    if isinstance(device, jax.Device):
        return device
    if device is None or device == "default":
        return jax.devices()[0]
    if isinstance(device, int):
        return jax.devices()[int(device)]
    if not isinstance(device, str):
        raise TypeError(f"device must be str|int|jax.Device|None, got {type(device)}")

    s = device.strip().lower()
    if ":" in s:
        plat, idx_s = s.split(":", 1)
        idx = int(idx_s)
    else:
        plat, idx = s, 0
    if plat in ("cuda",):
        plat = "gpu"

    devs = jax.devices(plat) if plat in ("cpu", "gpu", "tpu") else jax.devices()
    if not devs:
        raise ValueError(f"No JAX devices found for device={device!r}. Available: {jax.devices()!r}")
    if idx < 0 or idx >= len(devs):
        raise ValueError(f"device index {idx} out of range for platform {plat!r} (n={len(devs)})")
    return devs[idx]


def _to_jnp(x: np.ndarray, dev: jax.Device, dtype=jnp.float32) -> jax.Array:
    return jax.device_put(jnp.asarray(x, dtype=dtype), dev)


# -----------------------------------------------------------------------------
# Loader: per-timestep pickle / gzip-pickle
# -----------------------------------------------------------------------------
def load_state(save_pth: str, t: int) -> Dict:
    """
    Load a dict-like state from {save_pth}/{t}.pickle or {t}.zip (gzip pickle).
    """
    p_pickle = os.path.join(save_pth, f"{int(t)}.pickle")
    p_zip = os.path.join(save_pth, f"{int(t)}.zip")

    if os.path.exists(p_pickle):
        with open(p_pickle, "rb") as f:
            return pickle.load(f)
    if os.path.exists(p_zip):
        with gzip.open(p_zip, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"No state file for t={t} in {save_pth!r} (expected .pickle or .zip).")


def extract_AFP(state: Dict, *, t: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract (A, F, P) from a loaded state dict.

    If F is missing, raise a clear error (do not reconstruct).
    """
    if "A" not in state:
        raise KeyError(f"'A' not found in state at t={t}. Keys: {list(state.keys())}")
    if "F" not in state:
        raise ValueError(
            f"F not found in state at t={t}. "
            f"Log F during simulation (store the F used in reintegration tracking)."
        )
    A = np.asarray(state["A"])
    F = np.asarray(state["F"])
    P = np.asarray(state["P"]) if "P" in state else None
    return A, F, P


# -----------------------------------------------------------------------------
# Loader: chunked npz snapshots (simulate_save_apf.py compatible)
# -----------------------------------------------------------------------------
def _list_npz_chunks(base_dir: str) -> List[Tuple[str, int, int, int]]:
    chunks: List[Tuple[str, int, int, int]] = []
    for fn in os.listdir(base_dir):
        m = _NPZ_PATTERN.match(fn)
        if not m:
            continue
        s0, s1, _t0, _t1, idx = m.groups()
        chunks.append((os.path.join(base_dir, fn), int(s0), int(s1), int(idx)))
    chunks.sort(key=lambda d: (d[1], d[3]))
    return chunks


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def iter_npz_snapshots(
    base_dir: str,
    t1: int,
    t2: int,
    *,
    fields: Sequence[str] = ("A", "F", "P"),
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """
    Yield per-snapshot dicts from chunked npz logs for steps in [t1, t2].
    Each yield is (t, {"A":..., "F":..., "P":...}) depending on requested fields.
    """
    want = {str(k) for k in fields}
    for path, start_step, end_step, _idx in _list_npz_chunks(base_dir):
        if not _overlaps(start_step, end_step, int(t1), int(t2)):
            continue
        data = np.load(path)
        steps = np.asarray(data["steps"], dtype=np.int64)
        mask = (steps >= int(t1)) & (steps <= int(t2))
        if not np.any(mask):
            continue

        # Keep arrays on host, per-frame extraction below.
        payload: Dict[str, np.ndarray] = {}
        if "P" in want:
            payload["P"] = np.asarray(data["P"])
        if "A" in want:
            if "A" not in data.files:
                raise ValueError(f"A not found in npz chunk {path}. Re-run logging with save_A=true.")
            payload["A"] = np.asarray(data["A"])
        if "F" in want:
            if "F" not in data.files:
                raise ValueError(f"F not found in npz chunk {path}. Re-run logging with save_F=true.")
            payload["F"] = np.asarray(data["F"])

        idxs = np.nonzero(mask)[0]
        for i in idxs:
            out: Dict[str, np.ndarray] = {}
            if "A" in payload:
                out["A"] = payload["A"][i]
            if "F" in payload:
                out["F"] = payload["F"][i]
            if "P" in payload:
                out["P"] = payload["P"][i]
            yield int(steps[i]), out


def load_npz_range(
    base_dir: str,
    t1: int,
    t2: int,
    *,
    fields: Sequence[str] = ("A", "F", "P"),
    device: Union[str, int, jax.Device, None] = "gpu",
    dtype=jnp.float32,
) -> Dict[str, jax.Array]:
    """
    Load a step range from chunked npz logs and return stacked JAX arrays on `device`.
    Returns a dict containing "t" plus requested fields (A/F/P).
    """
    dev = resolve_device(device)
    want = {str(k) for k in fields}

    ts: List[int] = []
    A_list: List[np.ndarray] = []
    F_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []

    for t, sample in iter_npz_snapshots(base_dir, t1, t2, fields=fields):
        ts.append(int(t))
        if "A" in want:
            A_list.append(sample["A"])
        if "F" in want:
            F_list.append(sample["F"])
        if "P" in want:
            P_list.append(sample["P"])

    if not ts:
        raise ValueError(f"No snapshots found in {base_dir!r} for steps [{t1}, {t2}].")

    out: Dict[str, jax.Array] = {"t": _to_jnp(np.asarray(ts, dtype=np.int64), dev, dtype=jnp.int64)}
    if "A" in want:
        out["A"] = _to_jnp(np.stack(A_list, axis=0), dev, dtype=dtype)
    if "F" in want:
        out["F"] = _to_jnp(np.stack(F_list, axis=0), dev, dtype=dtype)
    if "P" in want:
        out["P"] = _to_jnp(np.stack(P_list, axis=0), dev, dtype=dtype)
    return out


def load_pickle_range(
    save_pth: str,
    t1: int,
    t2: int,
    *,
    fields: Sequence[str] = ("A", "F", "P"),
    device: Union[str, int, jax.Device, None] = "gpu",
    dtype=jnp.float32,
) -> Dict[str, jax.Array | None]:
    """
    Load a timestep range from per-timestep pickles and return stacked JAX arrays on `device`.
    Returns a dict containing "t" plus requested fields (A/F/P).
    """
    dev = resolve_device(device)
    want = {str(k) for k in fields}

    ts: List[int] = []
    A_list: List[np.ndarray] = []
    F_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []
    all_have_P = True

    for t in range(int(t1), int(t2) + 1):
        state = load_state(save_pth, t)
        A, F, P = extract_AFP(state, t=t)
        ts.append(int(t))
        if "A" in want:
            A_list.append(A)
        if "F" in want:
            F_list.append(F)
        if "P" in want:
            if P is None:
                all_have_P = False
            else:
                P_list.append(P)

    out: Dict[str, jax.Array | None] = {"t": _to_jnp(np.asarray(ts, dtype=np.int64), dev, dtype=jnp.int64)}
    if "A" in want:
        out["A"] = _to_jnp(np.stack(A_list, axis=0), dev, dtype=dtype)
    if "F" in want:
        out["F"] = _to_jnp(np.stack(F_list, axis=0), dev, dtype=dtype)
    if "P" in want:
        if not all_have_P:
            out["P"] = None
        else:
            out["P"] = _to_jnp(np.stack(P_list, axis=0), dev, dtype=dtype)
    return out


def infer_log_format(save_pth: str) -> str:
    """
    Infer logging format for `save_pth`: "npz" or "pickle".
    """
    if os.path.isdir(save_pth):
        for fn in os.listdir(save_pth):
            if _NPZ_PATTERN.match(fn):
                return "npz"
    return "pickle"


def load_range(
    save_pth: str,
    t1: int,
    t2: int,
    *,
    fields: Sequence[str] = ("A", "F", "P"),
    device: Union[str, int, jax.Device, None] = "gpu",
    dtype=jnp.float32,
    log_format: str = "auto",
) -> Dict[str, jax.Array | None]:
    """
    Convenience: load a range from either npz chunks or per-timestep pickles.
    """
    fmt = log_format.strip().lower()
    if fmt == "auto":
        fmt = infer_log_format(save_pth)
    if fmt == "npz":
        return load_npz_range(save_pth, t1, t2, fields=fields, device=device, dtype=dtype)
    if fmt == "pickle":
        return load_pickle_range(save_pth, t1, t2, fields=fields, device=device, dtype=dtype)
    raise ValueError(f"Unknown log_format={log_format!r}. Use 'auto'|'npz'|'pickle'.")


# -----------------------------------------------------------------------------
# Metric core (JAX, jitted)
# -----------------------------------------------------------------------------
def _make_box_avg(r: int):
    r = int(r)
    shifts = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    denom = float((2 * r + 1) ** 2)

    def box_avg(x: jax.Array) -> jax.Array:
        acc = jnp.zeros_like(x)
        for dy, dx in shifts:
            acc = acc + jnp.roll(x, (dy, dx), axis=(0, 1))
        return acc / denom

    return box_avg


def _central_grad_periodic(m: jax.Array) -> jax.Array:
    dm_dx = 0.5 * (jnp.roll(m, -1, axis=1) - jnp.roll(m, +1, axis=1))
    dm_dy = 0.5 * (jnp.roll(m, -1, axis=0) - jnp.roll(m, +1, axis=0))
    return jnp.stack([dm_dy, dm_dx], axis=-1)  # (H,W,2) with (y,x)


def _norm2(v: jax.Array, eps: float) -> jax.Array:
    # Keep eps out of the sqrt to match the metric definition (||v|| + eps).
    _ = eps
    return jnp.sqrt(jnp.sum(v * v, axis=-1))


def _quantile_masked(vals: jax.Array, mask: jax.Array, q: float) -> jax.Array:
    flat = vals.reshape((-1,))
    mflat = mask.reshape((-1,)).astype(jnp.int32)
    n_total = flat.shape[0]
    n_mask = jnp.sum(mflat)

    def do_quantile(_):
        filled = jnp.where(mflat.astype(bool), flat, -jnp.inf)
        s = jnp.sort(filled)
        n_unmasked = n_total - n_mask
        k = jnp.floor(jnp.asarray(q) * (n_mask - 1)).astype(jnp.int32)
        idx = (n_unmasked + k).astype(jnp.int32)
        idx = jnp.clip(idx, 0, n_total - 1)
        return s[idx]

    return lax.cond(n_mask > 0, do_quantile, lambda _: jnp.array(0.0, dtype=flat.dtype), operand=None)

def make_frame_step(
    *,
    eps: float = 1e-6,
    r_pool: int = 3,
    beta_t: float = 0.01,
    q_top: float = 0.995,
    m_thr_frac: float = 0.02,
    mp_thr_frac: float = 0.01,
    kappa_thr: float = 0.2,
):
    """
    Create a jitted per-frame step function:
      (ema_Jp, ema_Mp, metrics...) = step(A, F, ema_Jp, ema_Mp)

    ema_Jp is the EMA of Jp with shape (H,W,2).
    ema_Mp is the EMA of Mp with shape (H,W).
    """
    box_avg = _make_box_avg(r_pool)
    eps = float(eps)
    beta_t = float(beta_t)
    q_top = float(q_top)
    m_thr_frac = float(m_thr_frac)
    mp_thr_frac = float(mp_thr_frac)
    kappa_thr = float(kappa_thr)

    @jax.jit
    def step(A: jax.Array, F: jax.Array, ema_Jp: jax.Array, ema_Mp: jax.Array):
        A = A.astype(jnp.float32)
        F = F.astype(jnp.float32)

        # m: (H,W), J: (H,W,2)
        m = jnp.sum(A, axis=-1)
        J = jnp.sum(F * A[..., None, :], axis=-1)

        g = _central_grad_periodic(m)  # (H,W,2)
        n = g / (jnp.sqrt(jnp.sum(g * g, axis=-1, keepdims=True)) + eps)

        q = jnp.sum(J * n, axis=-1)  # (H,W)
        J_perp = q[..., None] * n  # (H,W,2)
        mag_perp = jnp.abs(q)  # (H,W)

        Jp = box_avg(J_perp)
        Mp = box_avg(mag_perp)

        norm_Jp = _norm2(Jp, eps)  # (H,W)
        kappa_fast = norm_Jp / (Mp + eps)

        # Slow EMA for both numerator and denominator.
        ema_Jp_next = (1.0 - beta_t) * ema_Jp + beta_t * Jp
        ema_Mp_next = (1.0 - beta_t) * ema_Mp + beta_t * Mp

        norm_ema = _norm2(ema_Jp_next, eps)
        kappa_slow = norm_ema / (ema_Mp_next + eps)

        S_fast = kappa_fast * norm_Jp
        S_slow = kappa_slow * norm_ema

        # Robustness to transient intruders (gliders) / jitter:
        #   persist ~ 1 for persistent drift, small for transient.
        #   cosang gates direction flips.
        persist = norm_ema / (norm_Jp + eps)
        cosang = jnp.sum(Jp * ema_Jp_next, axis=-1) / (norm_Jp * norm_ema + eps)

        max_m = jnp.max(m)
        max_mp = jnp.max(Mp)
        m_thr = m_thr_frac * max_m
        mp_thr = mp_thr_frac * max_mp
        mask = (
            (m > m_thr)
            & (Mp > mp_thr)
            & (kappa_slow > kappa_thr)
            & (persist > 0.3)
            & (cosang > 0.2)
        )
        n_mask = jnp.sum(mask.astype(jnp.int32))

        drift_fast = _quantile_masked(S_fast, mask, q_top)
        drift_slow = _quantile_masked(S_slow, mask, q_top)

        # Dominant direction from slow signal: mean EMA_Jp over top pixels
        top_mask = mask & (S_slow >= drift_slow)
        n_top = jnp.sum(top_mask.astype(jnp.int32))
        mean_vec = jnp.sum(ema_Jp_next * top_mask[..., None], axis=(0, 1)) / (n_top.astype(jnp.float32) + eps)
        mean_norm = jnp.sqrt(jnp.sum(mean_vec * mean_vec)) + eps
        dir_slow = mean_vec / mean_norm
        dir_slow = lax.cond(n_top > 0, lambda v: v, lambda _: jnp.zeros_like(dir_slow), operand=dir_slow)

        return ema_Jp_next, ema_Mp_next, drift_fast, drift_slow, dir_slow, n_mask, max_m, max_mp

    return step


def make_jperp_fn(*, eps: float = 1e-6):
    """
    Create a jitted per-frame function that computes the normal-projected momentum.

    Returns: (m, J, J_perp, q, n)
      m:      (H, W)       total mass
      J:      (H, W, 2)    momentum sum_c A_c * F_c
      J_perp: (H, W, 2)    normal-projected momentum q*n
      q:      (H, W)       normal scalar flux
      n:      (H, W, 2)    unit normal from grad(m)
    """
    eps = float(eps)

    @jax.jit
    def fn(A: jax.Array, F: jax.Array):
        A = A.astype(jnp.float32)
        F = F.astype(jnp.float32)
        m = jnp.sum(A, axis=-1)
        J = jnp.sum(F * A[..., None, :], axis=-1)
        g = _central_grad_periodic(m)
        n = g / (jnp.sqrt(jnp.sum(g * g, axis=-1, keepdims=True)) + eps)
        q = jnp.sum(J * n, axis=-1)
        J_perp = q[..., None] * n
        return m, J, J_perp, q, n

    return fn


# -----------------------------------------------------------------------------
# Streaming timeseries + CSV
# -----------------------------------------------------------------------------
def compute_drift_timeseries(
    save_pth: str,
    t1: int,
    t2: int,
    *,
    device: Union[str, int, jax.Device, None] = "gpu",
    log_format: str = "auto",
    csv_path: Optional[str] = None,
    eps: float = 1e-6,
    r_pool: int = 3,
    beta_t: float = 0.01,
    q_top: float = 0.995,
    m_thr_frac: float = 0.02,
    mp_thr_frac: float = 0.01,
    kappa_thr: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Compute drift metrics for frames in [t1, t2] (inclusive).

    Returns numpy arrays:
      t, drift_fast, drift_slow, dir_slow_y, dir_slow_x, n_mask, max_m, max_mp
    """
    dev = resolve_device(device)
    step_fn = make_frame_step(
        eps=eps,
        r_pool=r_pool,
        beta_t=beta_t,
        q_top=q_top,
        m_thr_frac=m_thr_frac,
        mp_thr_frac=mp_thr_frac,
        kappa_thr=kappa_thr,
    )

    fmt = log_format.strip().lower()
    if fmt == "auto":
        fmt = infer_log_format(save_pth)

    if fmt == "npz":
        iterator: Iterable[Tuple[int, Dict[str, np.ndarray]]] = iter_npz_snapshots(
            save_pth, t1, t2, fields=("A", "F")
        )
    elif fmt == "pickle":
        def _iter_pickle():
            for t in range(int(t1), int(t2) + 1):
                st = load_state(save_pth, t)
                A, F, _P = extract_AFP(st, t=t)
                yield t, {"A": A, "F": F}
        iterator = _iter_pickle()
    else:
        raise ValueError(f"Unknown log_format={log_format!r}. Use 'auto'|'npz'|'pickle'.")

    # CSV setup
    csv_f = None
    csv_w = None
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t", "drift_fast", "drift_slow", "dir_slow_y", "dir_slow_x", "n_mask", "max_m", "max_mp"])

    ts: List[int] = []
    drift_fast: List[float] = []
    drift_slow: List[float] = []
    dir_y: List[float] = []
    dir_x: List[float] = []
    n_mask_list: List[int] = []
    max_m_list: List[float] = []
    max_mp_list: List[float] = []

    ema_Jp: Optional[jax.Array] = None
    ema_Mp: Optional[jax.Array] = None
    try:
        for t, sample in iterator:
            A_np = np.asarray(sample["A"])
            F_np = np.asarray(sample["F"])

            A = _to_jnp(A_np, dev, dtype=jnp.float32)
            F = _to_jnp(F_np, dev, dtype=jnp.float32)

            if ema_Jp is None or ema_Mp is None:
                H, W = int(A_np.shape[0]), int(A_np.shape[1])
                ema_Jp = jax.device_put(jnp.zeros((H, W, 2), dtype=jnp.float32), dev)
                ema_Mp = jax.device_put(jnp.zeros((H, W), dtype=jnp.float32), dev)

            ema_Jp, ema_Mp, df, ds, dvec, nmask, mm, mp = step_fn(A, F, ema_Jp, ema_Mp)

            df_f = float(np.asarray(df))
            ds_f = float(np.asarray(ds))
            dvec_np = np.asarray(dvec)
            nmask_i = int(np.asarray(nmask))
            mm_f = float(np.asarray(mm))
            mp_f = float(np.asarray(mp))

            ts.append(int(t))
            drift_fast.append(df_f)
            drift_slow.append(ds_f)
            dir_y.append(float(dvec_np[0]))
            dir_x.append(float(dvec_np[1]))
            n_mask_list.append(nmask_i)
            max_m_list.append(mm_f)
            max_mp_list.append(mp_f)

            if csv_w is not None:
                csv_w.writerow([int(t), df_f, ds_f, float(dvec_np[0]), float(dvec_np[1]), nmask_i, mm_f, mp_f])
    finally:
        if csv_f is not None:
            csv_f.close()

    return {
        "t": np.asarray(ts, dtype=np.int64),
        "drift_fast": np.asarray(drift_fast, dtype=np.float32),
        "drift_slow": np.asarray(drift_slow, dtype=np.float32),
        "dir_slow_y": np.asarray(dir_y, dtype=np.float32),
        "dir_slow_x": np.asarray(dir_x, dtype=np.float32),
        "n_mask": np.asarray(n_mask_list, dtype=np.int32),
        "max_m": np.asarray(max_m_list, dtype=np.float32),
        "max_mp": np.asarray(max_mp_list, dtype=np.float32),
    }
