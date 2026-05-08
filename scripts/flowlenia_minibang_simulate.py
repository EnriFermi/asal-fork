from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# This script may spawn long JAX rollouts. Avoid grabbing most GPU memory upfront.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from flowlenia_minibang_common import (
    list_apf_chunks,
    load_config,
    project_root,
    resolve_path,
    to_plain,
    write_json,
)


def _ensure_jax() -> None:
    if "jax" not in globals():
        import jax as _jax
        import jax.numpy as _jnp

        globals()["jax"] = _jax
        globals()["jnp"] = _jnp


def _ensure_flow_modules() -> None:
    if "substrates" not in globals():
        import substrates as _substrates
        import util as _util

        globals()["substrates"] = _substrates
        globals()["util"] = _util


def _ensure_metric_modules() -> None:
    if "make_metric_loss_fn" not in globals():
        from clip_deltah_msc_metric import make_metric_loss_fn as _make_metric_loss_fn
        from clip_deltah_msc_metric import resolve_metric_config as _resolve_metric_config

        globals()["make_metric_loss_fn"] = _make_metric_loss_fn
        globals()["resolve_metric_config"] = _resolve_metric_config


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if hasattr(obj, name) else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _load_pkl_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _param_hash(params: np.ndarray) -> str:
    arr = np.asarray(params, dtype=np.float32)
    rounded = np.round(arr, decimals=6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def _loss_sort_key(row: dict[str, Any]) -> tuple[bool, float]:
    loss = row.get("loss", np.nan)
    finite = np.isfinite(loss)
    return (not bool(finite), float(loss) if finite else float("inf"))


def _candidate_allowed(row: dict[str, Any], args: Any) -> bool:
    source = str(row.get("source", ""))
    if source == "pop":
        return _as_bool(_get(args, "include_population", True), True)
    if source == "best_traj":
        return _as_bool(_get(args, "include_best_traj", True), True)
    if source == "best_final":
        return _as_bool(_get(args, "include_final_best", True), True)
    return True


def load_checkpoint_candidates(checkpoint_dir: Path, args: Any) -> tuple[list[dict[str, Any]], int, int]:
    candidates: list[dict[str, Any]] = []
    n_iters = 0
    param_dim = 0

    pop_obj = _load_pkl_if_exists(checkpoint_dir / "pop_traj.pkl")
    if isinstance(pop_obj, dict) and "params" in pop_obj:
        pop_params = np.asarray(pop_obj["params"], dtype=np.float32)
        if pop_params.ndim != 3:
            raise ValueError(f"pop_traj['params'] must be (n_iters, pop_size, n_params), got {pop_params.shape}.")
        pop_loss = np.asarray(pop_obj.get("loss", np.full(pop_params.shape[:2], np.nan)), dtype=np.float64)
        if pop_loss.shape != pop_params.shape[:2]:
            raise ValueError(f"pop_traj['loss'] shape {pop_loss.shape} does not match {pop_params.shape[:2]}.")
        n_iters = max(n_iters, int(pop_params.shape[0]))
        param_dim = int(pop_params.shape[-1])
        for i_iter in range(pop_params.shape[0]):
            finite_losses = pop_loss[i_iter][np.isfinite(pop_loss[i_iter])]
            if finite_losses.size:
                order = np.argsort(pop_loss[i_iter])
                rank = np.empty_like(order)
                rank[order] = np.arange(order.size)
                denom = max(1, order.size - 1)
            else:
                rank = np.zeros((pop_params.shape[1],), dtype=np.int64)
                denom = 1
            for j_pop in range(pop_params.shape[1]):
                candidates.append(
                    dict(
                        source="pop",
                        iter=int(i_iter),
                        pop_idx=int(j_pop),
                        loss=float(pop_loss[i_iter, j_pop]),
                        iter_loss_rank_frac=float(rank[j_pop] / denom),
                        params=pop_params[i_iter, j_pop],
                    )
                )

    best_traj = _load_pkl_if_exists(checkpoint_dir / "best_traj.pkl")
    if isinstance(best_traj, dict) and "params" in best_traj:
        best_params = np.asarray(best_traj["params"], dtype=np.float32)
        if best_params.ndim != 2:
            raise ValueError(f"best_traj['params'] must be (n_iters, n_params), got {best_params.shape}.")
        best_loss = np.asarray(best_traj.get("loss", np.full((best_params.shape[0],), np.nan)), dtype=np.float64)
        n_iters = max(n_iters, int(best_params.shape[0]))
        param_dim = int(best_params.shape[-1])
        for i_iter in range(best_params.shape[0]):
            candidates.append(
                dict(
                    source="best_traj",
                    iter=int(i_iter),
                    pop_idx=-1,
                    loss=float(best_loss[i_iter]) if i_iter < best_loss.shape[0] else float("nan"),
                    iter_loss_rank_frac=0.0,
                    params=best_params[i_iter],
                )
            )

    best_obj = _load_pkl_if_exists(checkpoint_dir / "best.pkl")
    if best_obj is not None:
        if isinstance(best_obj, tuple) and len(best_obj) == 2:
            best_params, best_loss = best_obj
        else:
            best_params, best_loss = best_obj, np.nan
        best_params = np.asarray(best_params, dtype=np.float32)
        if best_params.ndim != 1:
            best_params = best_params.reshape(-1)
        param_dim = int(best_params.size)
        candidates.append(
            dict(
                source="best_final",
                iter=max(0, n_iters - 1),
                pop_idx=-1,
                loss=float(np.asarray(best_loss).reshape(-1)[0]) if np.asarray(best_loss).size else float("nan"),
                iter_loss_rank_frac=0.0,
                params=best_params,
            )
        )

    candidates = [c for c in candidates if _candidate_allowed(c, args)]
    if not candidates:
        raise FileNotFoundError(
            f"No selectable params found in {checkpoint_dir}. Expected pop_traj.pkl, best_traj.pkl, or best.pkl."
        )
    if n_iters <= 0:
        n_iters = max(int(c.get("iter", 0)) for c in candidates) + 1
    return candidates, n_iters, param_dim


def select_params(checkpoint_dir: Path, args: Any) -> list[dict[str, Any]]:
    candidates, n_iters, _param_dim = load_checkpoint_candidates(checkpoint_dir, args)
    target_n = int(_get(args, "n_trajectories", 50))
    if target_n < 1:
        raise ValueError(f"n_trajectories must be >= 1, got {target_n}.")

    rng = np.random.default_rng(int(_get(args, "selection_seed", 0)))
    selection_mode = str(_get(args, "selection_mode", "iter_bins")).strip().lower().replace("-", "_")

    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    def add_candidate(row: dict[str, Any]) -> bool:
        h = _param_hash(np.asarray(row["params"]))
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        row_out = dict(row)
        row_out.pop("params", None)
        row_out["param_hash"] = h
        row_out["saturation_T"] = float(row_out.get("iter", 0)) / float(max(1, n_iters - 1))
        row_out["params"] = np.asarray(row["params"], dtype=np.float32)
        selected.append(row_out)
        return True

    if selection_mode in {"loss", "loss_top", "global_loss"}:
        for row in sorted(candidates, key=_loss_sort_key):
            if len(selected) >= target_n:
                break
            add_candidate(row)
    elif selection_mode in {"loss_quantile_biased", "quantile_biased", "loss_biased"}:
        unique_rows: list[dict[str, Any]] = []
        unique_hashes: set[str] = set()
        for row in sorted(candidates, key=_loss_sort_key):
            h = _param_hash(np.asarray(row["params"]))
            if h in unique_hashes:
                continue
            unique_hashes.add(h)
            unique_rows.append(row)
        sorted_rows = unique_rows
        top_n = max(0, int(_get(args, "selection_keep_top_n", 5)))
        bias_gamma = float(_get(args, "selection_loss_bias_gamma", 2.5))
        if not np.isfinite(bias_gamma) or bias_gamma <= 0.0:
            raise ValueError(f"selection_loss_bias_gamma must be > 0, got {bias_gamma}.")
        jitter_frac = float(_get(args, "selection_loss_jitter_frac", 0.15))
        jitter_frac = float(np.clip(jitter_frac, 0.0, 1.0))

        for row in sorted_rows[:top_n]:
            if len(selected) >= target_n:
                break
            add_candidate(row)

        remaining = max(0, target_n - len(selected))
        if remaining > 0 and sorted_rows:
            n = len(sorted_rows)
            if remaining == 1:
                base_u = np.asarray([0.5], dtype=np.float64)
            else:
                base_u = (np.arange(remaining, dtype=np.float64) + 0.5) / float(remaining)
            q = np.power(base_u, bias_gamma)
            base_rank = q * float(max(0, n - 1))
            bucket_width = float(max(1, n - 1)) / float(max(1, remaining))
            jitter = rng.uniform(-0.5, 0.5, size=remaining) * jitter_frac * bucket_width
            rank_targets = np.clip(np.rint(base_rank + jitter).astype(np.int64), 0, max(0, n - 1))

            for rank in rank_targets:
                if len(selected) >= target_n:
                    break
                row = sorted_rows[int(rank)]
                if add_candidate(row):
                    continue
                # If the quantile target lands on an already selected row, walk outward
                # around that rank before moving to the next target.
                for offset in range(1, n):
                    lo = int(rank) - offset
                    hi = int(rank) + offset
                    added = False
                    if lo >= 0:
                        added = add_candidate(sorted_rows[lo])
                    if added:
                        break
                    if hi < n:
                        added = add_candidate(sorted_rows[hi])
                    if added:
                        break
    elif selection_mode in {"iter_bins", "iteration_bins", "stratified"}:
        iter_bins = int(_get(args, "selection_iter_bins", min(10, target_n)))
        iter_bins = max(1, min(iter_bins, target_n, max(1, n_iters)))
        elite_fraction = float(_get(args, "selection_elite_fraction", 0.35))
        elite_fraction = float(np.clip(elite_fraction, 0.0, 1.0))
        pool_min = int(_get(args, "selection_pool_min_per_bin", 8))
        best_traj_per_bin = _as_bool(_get(args, "selection_best_traj_per_bin", True), True)

        edges = np.linspace(0, n_iters, iter_bins + 1, dtype=np.int64)
        quotas = np.full((iter_bins,), target_n // iter_bins, dtype=np.int64)
        quotas[: target_n % iter_bins] += 1

        for i_bin in range(iter_bins):
            lo = int(edges[i_bin])
            hi = int(edges[i_bin + 1])
            if i_bin == iter_bins - 1:
                hi = max(hi, n_iters)
            in_bin = [c for c in candidates if lo <= int(c.get("iter", 0)) < hi]
            if not in_bin:
                continue

            added_best = False
            if best_traj_per_bin:
                best_rows = sorted([c for c in in_bin if c.get("source") == "best_traj"], key=_loss_sort_key)
                if best_rows and len(selected) < target_n:
                    added_best = add_candidate(best_rows[0])

            in_bin_sorted = sorted(in_bin, key=_loss_sort_key)
            pool_n = max(pool_min, int(np.ceil(elite_fraction * len(in_bin_sorted))))
            pool = in_bin_sorted[: min(len(in_bin_sorted), pool_n)]
            quota_remaining = max(0, int(quotas[i_bin]) - (1 if added_best else 0))
            if quota_remaining <= 0:
                continue
            if len(pool) <= quota_remaining:
                chosen = pool
            else:
                idx = rng.choice(len(pool), size=quota_remaining, replace=False)
                chosen = [pool[int(i)] for i in idx]
            for row in chosen:
                if len(selected) >= target_n:
                    break
                add_candidate(row)
    else:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. Use 'loss' or 'iter_bins'."
        )

    if len(selected) < target_n:
        for row in sorted(candidates, key=lambda r: (int(r.get("iter", 0)), *_loss_sort_key(r))):
            if len(selected) >= target_n:
                break
            add_candidate(row)

    if len(selected) < target_n:
        print(f"Warning: requested {target_n} trajectories, selected only {len(selected)} unique parameter sets.")

    selected = selected[:target_n]
    for idx, row in enumerate(selected):
        row["traj_id"] = f"traj_{idx:05d}"
        row["selection_idx"] = int(idx)
    return selected


def _make_substrate(args: Any):
    _ensure_jax()
    _ensure_flow_modules()
    if str(args.substrate) != "lenia_flow":
        raise ValueError(f"This minibang runner supports substrate='lenia_flow', got {args.substrate!r}.")
    kwargs = util.flow_lenia_kwargs_from_args(args)
    kwargs["debug_return_F"] = True
    base = substrates.create_substrate(args.substrate, **kwargs)
    return substrates.FlattenSubstrateParameters(base)


def _init_lagrangian_points_jax(
    A0: jax.Array,
    *,
    n_particles: int,
    init_mode: str,
    border: str,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    _ensure_jax()
    sx = int(A0.shape[0])
    sy = int(A0.shape[1])
    mode = str(init_mode).strip().lower()
    if mode == "uniform":
        ky, kx = jax.random.split(key)
        y = jax.random.uniform(ky, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(kx, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif mode == "mass":
        k_idx, k_jit = jax.random.split(key)
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(total > 0.0, flat / jnp.maximum(total, 1e-12), jnp.ones_like(flat) / flat.size)
        idx = jax.random.choice(k_idx, flat.size, shape=(n_particles,), replace=True, p=probs)
        iy = idx // sy
        ix = idx % sy
        jitter = jax.random.uniform(k_jit, (n_particles, 2), minval=-0.49, maxval=0.49)
        pts = jnp.stack((iy.astype(jnp.float32) + 0.5, ix.astype(jnp.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unknown lagrangian_init_mode={init_mode!r}. Use 'mass' or 'uniform'.")

    if str(border) == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
        return jnp.stack((y, x), axis=-1).astype(jnp.float32)
    lo = float(sigma)
    y = jnp.clip(pts[:, 0], lo, sx - lo)
    x = jnp.clip(pts[:, 1], lo, sy - lo)
    return jnp.stack((y, x), axis=-1).astype(jnp.float32)


def _frame_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _write_frame_times(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "step", "video_sec", "sim_sec"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flush_run_buffers(run: dict[str, Any], args: Any) -> None:
    from simulate_save_apf import save_chunk

    buffers = run["buffers"]
    if not buffers["steps"]:
        return
    run["file_idx"] = save_chunk(
        str(run["apf_dir"]),
        float(run["sim_fps"]),
        buffers["steps"],
        buffers["P"],
        int(run["file_idx"]),
        buffers["A"] if _as_bool(_get(args, "save_A", True), True) else None,
        buffers["F"] if _as_bool(_get(args, "save_F", True), True) else None,
        use_fp16=_as_bool(_get(args, "save_fp16", True), True),
        snaps_rgb=buffers["rgb"] if _as_bool(_get(args, "save_rgb", False), False) else None,
        snaps_lagrangian=buffers["lagrangian_xy"] if _as_bool(_get(args, "save_lagrangian", True), True) else None,
        snaps_lagrangian_c=(
            buffers["lagrangian_c"]
            if _as_bool(_get(args, "save_lagrangian", True), True)
            and _as_bool(_get(args, "save_lagrangian_channels", True), True)
            else None
        ),
        compress=_as_bool(_get(args, "compress", True), True),
    )
    for key in buffers:
        buffers[key] = []


def _init_run_dirs(
    selected: list[dict[str, Any]],
    *,
    output_root: Path,
    cfg: Any,
    args: Any,
    overwrite: bool,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    sim_fps = float(_get(args, "fps", 250.0))
    video_fps = float(_get(args, "video_fps", max(1.0, min(30.0, sim_fps / max(1, int(_get(args, "snapshot_interval", 50)))))))

    for row in selected:
        traj_dir = output_root / str(row["traj_id"])
        if traj_dir.exists() and overwrite:
            shutil.rmtree(traj_dir)
        if traj_dir.exists() and not overwrite and any(traj_dir.iterdir()):
            raise FileExistsError(f"{traj_dir} already exists. Use --overwrite to regenerate it.")
        apf_dir = traj_dir / "apf_logs"
        ckpt_dir = traj_dir / "checkpoint"
        apf_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        params = np.asarray(row["params"], dtype=np.float32)
        np.save(traj_dir / "params.npy", params)
        with (ckpt_dir / "best.pkl").open("wb") as f:
            pickle.dump((params, float(row.get("loss", np.nan))), f)
        meta = {k: v for k, v in row.items() if k != "params"}
        write_json(traj_dir / "selection.json", meta)

        cfg_i = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if cfg_i.get("meta") is None:
            cfg_i.meta = {}
        if cfg_i.get("simulation") is None:
            cfg_i.simulation = {}
        cfg_i.meta.save_dir = str(ckpt_dir)
        cfg_i.meta.output_dir = str(apf_dir)
        cfg_i.simulation.output = str(traj_dir / "video.mp4")
        OmegaConf.save(config=cfg_i, f=str(traj_dir / "config.yaml"))

        import imageio

        writer = imageio.get_writer(
            str(traj_dir / "video.mp4"),
            fps=video_fps,
            codec=str(_get(args, "codec", "libx264")),
            macro_block_size=_get(args, "macro_block_size", 1),
        )
        runs.append(
            dict(
                traj_id=str(row["traj_id"]),
                selection={k: v for k, v in row.items() if k != "params"},
                params=params,
                traj_dir=traj_dir,
                apf_dir=apf_dir,
                writer=writer,
                file_idx=0,
                frame_idx=0,
                sim_fps=sim_fps,
                video_fps=video_fps,
                frame_rows=[],
                buffers=dict(
                    steps=[],
                    P=[],
                    A=[],
                    F=[],
                    rgb=[],
                    lagrangian_xy=[],
                    lagrangian_c=[],
                ),
            )
        )
    return runs


def _close_runs(runs: list[dict[str, Any]]) -> None:
    for run in runs:
        try:
            run["writer"].close()
        finally:
            _write_frame_times(run["traj_dir"] / "frame_times.csv", run["frame_rows"])


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(np.sum(closest))
        if not np.isfinite(total) or total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=closest / total))
        centers[i] = X[idx]
        dist = np.sum((X - centers[i]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return centers


def _assign_kmeans(X: np.ndarray, centers: np.ndarray, chunk_size: int = 200_000) -> np.ndarray:
    labels = np.empty((X.shape[0],), dtype=np.int32)
    for start in range(0, X.shape[0], chunk_size):
        end = min(X.shape[0], start + chunk_size)
        dist = np.sum((X[start:end, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(dist, axis=1).astype(np.int32)
    return labels


def _fit_kmeans(X: np.ndarray, k: int, *, rng: np.random.Generator, n_iter: int, restarts: int) -> tuple[np.ndarray, np.ndarray, float]:
    best_centers = None
    best_labels = None
    best_inertia = float("inf")
    restarts = max(1, int(restarts))
    for _ in range(restarts):
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros((X.shape[0],), dtype=np.int32)
        for _iter in range(max(1, int(n_iter))):
            labels = _assign_kmeans(X, centers)
            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centers[c] = X[mask].mean(axis=0)
            shift = float(np.max(np.sum((new_centers - centers) ** 2, axis=1)))
            centers = new_centers
            if shift < 1e-8:
                break
        dist = np.sum((X - centers[labels]) ** 2, axis=1)
        inertia = float(np.mean(dist))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()
    assert best_centers is not None and best_labels is not None
    return best_centers, best_labels, best_inertia


def _fit_dpmeans_weighted(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    lam: float,
    n_iter: int,
    max_clusters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    from evolutionary_metrics import dpmeans_weighted

    X = np.asarray(X, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if X.shape[0] == 0:
        raise ValueError("DP-means needs at least one sample.")
    if weights.shape[0] != X.shape[0]:
        raise ValueError(f"weights shape {weights.shape} does not match X shape {X.shape}.")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)

    centers_arr = dpmeans_weighted(
        X,
        weights.astype(np.float32),
        lam=float(lam),
        iters=int(n_iter),
        max_clusters=int(max_clusters),
    )
    labels_final = _assign_kmeans(X, centers_arr)
    d2 = np.sum((X - centers_arr[labels_final]) ** 2, axis=1)
    inertia = float(np.average(d2, weights=weights))
    return centers_arr, labels_final, inertia


def _p_rgb_features(p_flat: np.ndarray) -> np.ndarray:
    p = np.asarray(p_flat, dtype=np.float32)
    if p.shape[-1] >= 3:
        return p[:, :3].astype(np.float32, copy=False)
    reps = int(np.ceil(3 / max(1, p.shape[-1])))
    return np.tile(p, (1, reps))[:, :3].astype(np.float32)


def _rendered_pcolor_features(p_flat: np.ndarray, mass_flat: np.ndarray | None) -> np.ndarray:
    rgb = _p_rgb_features(p_flat)
    if mass_flat is None:
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)
    inten = np.asarray(mass_flat, dtype=np.float32).reshape(-1, 1)
    return np.clip(inten * rgb, 0.0, 1.0).astype(np.float32)


def _chroma_features(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    denom = np.sum(rgb, axis=1, keepdims=True)
    return np.where(denom > 1e-8, rgb / np.maximum(denom, 1e-8), 0.0).astype(np.float32)


def _cluster_features_from_apf(p_flat: np.ndarray, mass_flat: np.ndarray | None, *, cluster_space: str) -> np.ndarray:
    mode = str(cluster_space).strip().lower()
    if mode in {"p", "p_full", "full"}:
        return np.asarray(p_flat, dtype=np.float32)
    if mode in {"p_rgb", "prgb", "raw_rgb"}:
        return _p_rgb_features(p_flat)
    if mode in {"pcolor", "p_color", "render", "rendered", "rendered_rgb", "video_rgb", "rgb"}:
        return _rendered_pcolor_features(p_flat, mass_flat)
    if mode in {"pcolor_chroma", "rendered_chroma", "video_chroma", "chroma", "hue"}:
        return _chroma_features(_rendered_pcolor_features(p_flat, mass_flat))
    raise ValueError(f"Unsupported cluster_space={cluster_space!r}. Use 'p', 'p_rgb', 'pcolor', or 'pcolor_chroma'.")


def _cluster_center_rgb(centers_raw: np.ndarray, *, cluster_space: str) -> np.ndarray:
    centers = np.asarray(centers_raw, dtype=np.float32)
    mode = str(cluster_space).strip().lower()
    if mode in {"pcolor_chroma", "rendered_chroma", "video_chroma", "chroma", "hue"}:
        rgb = centers[:, :3] if centers.shape[1] >= 3 else np.tile(centers, (1, int(np.ceil(3 / centers.shape[1]))))[:, :3]
        rgb = np.clip(rgb, 0.0, 1.0)
        maxc = np.max(rgb, axis=1, keepdims=True)
        rgb = np.where(maxc > 1e-8, rgb / np.maximum(maxc, 1e-8), 0.0)
    elif mode in {"pcolor", "p_color", "p_rgb", "rgb"}:
        rgb = centers[:, :3] if centers.shape[1] >= 3 else np.tile(centers, (1, int(np.ceil(3 / centers.shape[1]))))[:, :3]
    else:
        rgb = centers[:, :3] if centers.shape[1] >= 3 else np.tile(centers, (1, int(np.ceil(3 / centers.shape[1]))))[:, :3]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _weighted_group_mean(values: np.ndarray, labels: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sums = np.zeros((int(k), vals.shape[1]), dtype=np.float64)
    sw = np.zeros((int(k),), dtype=np.float64)
    np.add.at(sums, lab, vals * w[:, None])
    np.add.at(sw, lab, w)
    out = np.zeros_like(sums, dtype=np.float64)
    keep = sw > 0.0
    out[keep] = sums[keep] / sw[keep, None]
    if not np.all(keep):
        out[~keep] = vals[:1]
    return out.astype(np.float32)


def _compact_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique = np.unique(lab)
    remap = {int(old): i for i, old in enumerate(unique.tolist())}
    compact = np.asarray([remap[int(x)] for x in lab], dtype=np.int32)
    return compact, int(unique.size)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    mx = np.max(arr, axis=1)
    mn = np.min(arr, axis=1)
    delta = mx - mn
    hue = np.zeros((arr.shape[0],), dtype=np.float64)
    nz = delta > 1e-12
    mask = nz & (mx == r)
    hue[mask] = (60.0 * ((g[mask] - b[mask]) / delta[mask]) + 360.0) % 360.0
    mask = nz & (mx == g)
    hue[mask] = 60.0 * ((b[mask] - r[mask]) / delta[mask] + 2.0)
    mask = nz & (mx == b)
    hue[mask] = 60.0 * ((r[mask] - g[mask]) / delta[mask] + 4.0)
    sat = np.where(mx > 1e-12, delta / np.maximum(mx, 1e-12), 0.0)
    return hue.astype(np.float32), sat.astype(np.float32), mx.astype(np.float32)


def _color_family_labels(
    centers_rgb: np.ndarray,
    *,
    min_saturation: float,
    min_value: float,
) -> np.ndarray:
    hue, sat, val = _rgb_to_hsv(centers_rgb)
    labels = np.zeros((hue.shape[0],), dtype=np.int32)
    low_value = val < float(min_value)
    low_sat = (~low_value) & (sat < float(min_saturation))
    labels[low_value] = 0
    labels[low_sat] = 1
    chroma = ~(low_value | low_sat)
    h = hue[chroma]
    fam = np.empty_like(h, dtype=np.int32)
    fam[(h < 30.0) | (h >= 330.0)] = 2
    fam[(h >= 30.0) & (h < 75.0)] = 3
    fam[(h >= 75.0) & (h < 165.0)] = 4
    fam[(h >= 165.0) & (h < 200.0)] = 5
    fam[(h >= 200.0) & (h < 265.0)] = 6
    fam[(h >= 265.0) & (h < 330.0)] = 7
    labels[chroma] = fam
    return labels


def _merge_centers_by_rgb(
    centers_z: np.ndarray,
    centers_raw: np.ndarray,
    centers_rgb: np.ndarray,
    sample_labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    threshold: float,
    merge_color_families: bool,
    hue_threshold_deg: float,
    min_saturation: float,
    min_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if centers_rgb.shape[0] <= 1:
        identity = np.arange(centers_rgb.shape[0], dtype=np.int32)
        return centers_z, centers_raw, centers_rgb, identity

    old_k = int(centers_rgb.shape[0])
    old_weights = np.bincount(
        np.asarray(sample_labels, dtype=np.int32),
        weights=np.asarray(sample_weights, dtype=np.float64),
        minlength=old_k,
    ).astype(np.float32)
    old_weights = np.where(old_weights > 0.0, old_weights, 1.0).astype(np.float32)

    if merge_color_families:
        merge_labels, new_k = _compact_labels(
            _color_family_labels(
                centers_rgb,
                min_saturation=float(min_saturation),
                min_value=float(min_value),
            )
        )
    elif float(hue_threshold_deg) > 0.0:
        from evolutionary_metrics import dpmeans_weighted

        hue, sat, val = _rgb_to_hsv(centers_rgb)
        saturated = (sat >= float(min_saturation)) & (val >= float(min_value))
        merge_labels = np.arange(old_k, dtype=np.int32)
        next_label = old_k
        if np.any(saturated):
            theta = np.deg2rad(hue[saturated].astype(np.float64))
            hue_points = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
            hue_lam = float(2.0 * np.sin(np.deg2rad(float(hue_threshold_deg)) * 0.5))
            hue_centers = dpmeans_weighted(
                hue_points,
                old_weights[saturated],
                lam=hue_lam,
                iters=6,
                max_clusters=int(np.sum(saturated)),
            )
            sat_labels = _assign_kmeans(hue_points, hue_centers)
            merge_labels[saturated] = sat_labels
            if np.any(~saturated):
                merge_labels[~saturated] = np.arange(next_label, next_label + int(np.sum(~saturated)), dtype=np.int32)
            merge_labels, new_k = _compact_labels(merge_labels)
        else:
            merge_labels, new_k = _compact_labels(merge_labels)
    elif float(threshold) > 0.0:
        from evolutionary_metrics import dpmeans_weighted

        merge_rgb = dpmeans_weighted(
            np.asarray(centers_rgb, dtype=np.float32),
            old_weights,
            lam=float(threshold),
            iters=6,
            max_clusters=old_k,
        )
        merge_labels = _assign_kmeans(np.asarray(centers_rgb, dtype=np.float32), merge_rgb)
        new_k = int(merge_rgb.shape[0])
    else:
        identity = np.arange(old_k, dtype=np.int32)
        return centers_z, centers_raw, centers_rgb, identity

    merged_z = _weighted_group_mean(centers_z, merge_labels, old_weights, new_k)
    merged_raw = _weighted_group_mean(centers_raw, merge_labels, old_weights, new_k)
    merged_rgb = _weighted_group_mean(centers_rgb, merge_labels, old_weights, new_k)
    merged_rgb = np.clip(merged_rgb, 0.0, 1.0).astype(np.float32)
    return merged_z, merged_raw, merged_rgb, merge_labels.astype(np.int32)


def _count_snapshots(apf_dir: Path) -> int:
    total = 0
    for path, _s0, _s1, _idx in list_apf_chunks(apf_dir):
        with np.load(path) as data:
            total += int(np.asarray(data["steps"]).shape[0])
    return total


def _compute_cluster_metrics(apf_dir: Path, args: Any, *, seed: int) -> dict[str, np.ndarray]:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}.")
    total_snapshots = _count_snapshots(apf_dir)
    if total_snapshots <= 0:
        raise ValueError(f"No snapshots found in {apf_dir}.")

    rng = np.random.default_rng(seed)
    cluster_method = str(_get(args, "cluster_method", "kmeans")).strip().lower().replace("-", "_")
    if cluster_method in {"dp", "dp_means"}:
        cluster_method = "dpmeans"
    if cluster_method not in {"kmeans", "dpmeans"}:
        raise ValueError(f"Unsupported cluster_method={cluster_method!r}. Use 'kmeans' or 'dpmeans'.")
    cluster_space = str(_get(args, "cluster_space", "p")).strip().lower()
    cluster_k = int(_get(args, "cluster_k", 8))
    max_samples = int(_get(args, "cluster_max_samples", 200_000))
    per_snapshot_raw = _get(args, "cluster_samples_per_snapshot", None)
    if per_snapshot_raw is None:
        samples_per_snapshot = max(1, max_samples // max(1, total_snapshots))
    else:
        samples_per_snapshot = max(1, int(per_snapshot_raw))
    min_mass = float(_get(args, "cluster_min_mass", 1e-8))

    samples: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    for path, _s0, _s1, _idx in chunks:
        with np.load(path) as data:
            if "P" not in data.files or "A" not in data.files:
                raise ValueError(f"Chunk {path} must contain P and A for cluster mass metrics.")
            P = np.asarray(data["P"], dtype=np.float32)
            A = np.asarray(data["A"], dtype=np.float32)
            for i in range(P.shape[0]):
                mass = np.sum(A[i], axis=-1).reshape(-1).astype(np.float64)
                p_flat = P[i].reshape((-1, P.shape[-1]))
                finite = np.all(np.isfinite(p_flat), axis=1) & np.isfinite(mass) & (mass > min_mass)
                valid_idx = np.flatnonzero(finite)
                if valid_idx.size == 0:
                    continue
                weights = mass[valid_idx]
                weights_sum = float(weights.sum())
                n_take = min(samples_per_snapshot, max(1, valid_idx.size))
                if weights_sum > 0.0 and np.isfinite(weights_sum):
                    probs = weights / weights_sum
                    chosen = rng.choice(valid_idx, size=n_take, replace=True, p=probs)
                else:
                    chosen = rng.choice(valid_idx, size=n_take, replace=True)
                samples.append(_cluster_features_from_apf(p_flat[chosen], mass[chosen], cluster_space=cluster_space))
                sample_weights.append(np.ones((chosen.shape[0],), dtype=np.float32))

    if not samples:
        raise ValueError(f"Could not sample any mass-weighted P vectors from {apf_dir}.")
    X_raw = np.concatenate(samples, axis=0)
    W_raw = np.concatenate(sample_weights, axis=0)
    if X_raw.shape[0] > max_samples:
        keep = rng.choice(X_raw.shape[0], size=max_samples, replace=False)
        X_raw = X_raw[keep]
        W_raw = W_raw[keep]

    standardize = _as_bool(_get(args, "cluster_standardize", True), True)
    if standardize:
        center = np.median(X_raw, axis=0).astype(np.float32)
        q25, q75 = np.percentile(X_raw, [25, 75], axis=0)
        scale = (q75 - q25).astype(np.float32)
        std = np.std(X_raw, axis=0).astype(np.float32)
        scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0)).astype(np.float32)
        X = ((X_raw - center) / scale).astype(np.float32)
    else:
        center = np.zeros((X_raw.shape[1],), dtype=np.float32)
        scale = np.ones((X_raw.shape[1],), dtype=np.float32)
        X = X_raw

    if cluster_method == "dpmeans":
        centers_z, sample_labels, inertia = _fit_dpmeans_weighted(
            X,
            W_raw,
            lam=float(_get(args, "cluster_dp_lambda", 1.5)),
            n_iter=int(_get(args, "cluster_dp_iters", 8)),
            max_clusters=int(_get(args, "cluster_dp_max_clusters", 64)),
        )
        k_eff = int(centers_z.shape[0])
    else:
        k_eff = max(1, min(cluster_k, X.shape[0]))
        centers_z, sample_labels, inertia = _fit_kmeans(
            X,
            k_eff,
            rng=rng,
            n_iter=int(_get(args, "cluster_kmeans_iters", 40)),
            restarts=int(_get(args, "cluster_kmeans_restarts", 2)),
        )
    centers_raw = centers_z * scale[None, :] + center[None, :]
    centers_rgb = _cluster_center_rgb(centers_raw, cluster_space=cluster_space)
    premerge_k_eff = int(k_eff)
    merge_rgb_threshold = float(_get(args, "cluster_merge_rgb_threshold", 0.0))
    merge_hue_threshold_deg = float(_get(args, "cluster_merge_hue_threshold_deg", 0.0))
    merge_color_families = _as_bool(_get(args, "cluster_merge_color_families", False), False)
    merge_min_saturation = float(_get(args, "cluster_merge_min_saturation", 0.2))
    merge_min_value = float(_get(args, "cluster_merge_min_value", 0.05))
    centers_z, centers_raw, centers_rgb, cluster_merge_map = _merge_centers_by_rgb(
        centers_z,
        centers_raw,
        centers_rgb,
        sample_labels,
        W_raw,
        threshold=merge_rgb_threshold,
        merge_color_families=merge_color_families,
        hue_threshold_deg=merge_hue_threshold_deg,
        min_saturation=merge_min_saturation,
        min_value=merge_min_value,
    )
    k_eff = int(centers_z.shape[0])

    steps_all: list[np.ndarray] = []
    mass_rows: list[np.ndarray] = []
    total_mass_rows: list[np.ndarray] = []
    for path, _s0, _s1, _idx in chunks:
        with np.load(path) as data:
            P = np.asarray(data["P"], dtype=np.float32)
            A = np.asarray(data["A"], dtype=np.float32)
            steps = np.asarray(data["steps"], dtype=np.int64)
            for i in range(P.shape[0]):
                mass = np.sum(A[i], axis=-1).reshape(-1).astype(np.float64)
                X_i = _cluster_features_from_apf(P[i].reshape((-1, P.shape[-1])), mass, cluster_space=cluster_space)
                X_i = ((X_i - center) / scale).astype(np.float32)
                labels_i = _assign_kmeans(X_i, centers_z)
                row = np.bincount(labels_i, weights=mass, minlength=k_eff).astype(np.float64)
                mass_rows.append(row)
                total_mass_rows.append(np.asarray([float(np.sum(mass))], dtype=np.float64))
            steps_all.append(steps)

    steps_arr = np.concatenate(steps_all).astype(np.int64)
    mass_by_cluster = np.stack(mass_rows, axis=0)
    total_mass = np.concatenate(total_mass_rows).reshape(-1)
    denom = np.maximum(total_mass[:, None], 1e-12)
    mass_prob = mass_by_cluster / denom
    entropy = -np.sum(np.where(mass_prob > 0.0, mass_prob * np.log(mass_prob + 1e-12), 0.0), axis=1)
    entropy_norm = entropy / max(np.log(float(k_eff)), 1e-12)
    tv_step = np.zeros((mass_prob.shape[0],), dtype=np.float64)
    if mass_prob.shape[0] > 1:
        tv_step[1:] = 0.5 * np.sum(np.abs(mass_prob[1:] - mass_prob[:-1]), axis=1)

    snapshot_interval = int(_get(args, "snapshot_interval", 1))
    lag_steps_raw = _get(args, "cluster_change_lag_steps", None)
    if lag_steps_raw is None:
        lag_frames = int(_get(args, "cluster_change_lag_frames", 4))
    else:
        lag_frames = max(1, int(round(float(lag_steps_raw) / float(max(1, snapshot_interval)))))
    lag_frames = max(1, lag_frames)
    tv_lag = np.zeros((mass_prob.shape[0],), dtype=np.float64)
    if mass_prob.shape[0] > lag_frames:
        tv_lag[lag_frames:] = 0.5 * np.sum(np.abs(mass_prob[lag_frames:] - mass_prob[:-lag_frames]), axis=1)

    return dict(
        cluster_steps=steps_arr,
        cluster_mass=mass_by_cluster.astype(np.float32),
        cluster_mass_prob=mass_prob.astype(np.float32),
        cluster_total_mass=total_mass.astype(np.float32),
        cluster_entropy=entropy.astype(np.float32),
        cluster_entropy_norm=entropy_norm.astype(np.float32),
        cluster_tv_step=tv_step.astype(np.float32),
        cluster_tv_lag=tv_lag.astype(np.float32),
        cluster_change_lag_frames=np.asarray(lag_frames, dtype=np.int32),
        cluster_centers_raw=centers_raw.astype(np.float32),
        cluster_centers_z=centers_z.astype(np.float32),
        cluster_centers_rgb=centers_rgb.astype(np.float32),
        cluster_standardize_center=center.astype(np.float32),
        cluster_standardize_scale=scale.astype(np.float32),
        cluster_k=np.asarray(k_eff, dtype=np.int32),
        cluster_premerge_k=np.asarray(premerge_k_eff, dtype=np.int32),
        cluster_merge_rgb_threshold=np.asarray(merge_rgb_threshold, dtype=np.float32),
        cluster_merge_hue_threshold_deg=np.asarray(merge_hue_threshold_deg, dtype=np.float32),
        cluster_merge_color_families=np.asarray(merge_color_families),
        cluster_merge_min_saturation=np.asarray(merge_min_saturation, dtype=np.float32),
        cluster_merge_min_value=np.asarray(merge_min_value, dtype=np.float32),
        cluster_merge_map=cluster_merge_map.astype(np.int32),
        cluster_method=np.asarray(cluster_method),
        cluster_space=np.asarray(cluster_space),
        cluster_kmeans_inertia=np.asarray(inertia, dtype=np.float32),
        cluster_samples_used=np.asarray(X_raw.shape[0], dtype=np.int32),
    )


def _load_lagrangian_series(apf_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    steps_all: list[np.ndarray] = []
    lag_all: list[np.ndarray] = []
    for path, _s0, _s1, _idx in list_apf_chunks(apf_dir):
        with np.load(path) as data:
            if "lagrangian_xy" not in data.files:
                raise ValueError(f"{path} has no lagrangian_xy. Enable save_lagrangian=true.")
            steps_all.append(np.asarray(data["steps"], dtype=np.int64))
            lag_all.append(np.asarray(data["lagrangian_xy"], dtype=np.float32))
    if not lag_all:
        raise FileNotFoundError(f"No lagrangian snapshots found in {apf_dir}.")
    return np.concatenate(steps_all), np.concatenate(lag_all, axis=0)


def _prepare_metric_args(flat_args: dict[str, Any], *, rollout_steps: int, sample_every_steps: int) -> SimpleNamespace:
    data = dict(flat_args)
    data["rollout_steps"] = int(rollout_steps)
    data["sample_every_steps"] = int(sample_every_steps)
    data["time_sampling"] = None

    # Defaults chosen for detection, not for optimizing the objective.
    defaults = {
        "metric_tau_mode": "max_grid",
        "metric_window_size_steps": max(sample_every_steps * 8, min(20_000, rollout_steps // 5)),
        "metric_m_samples": 48,
        "metric_m_min": 4,
        "metric_n_proj": 16,
        "metric_null_reps": 6,
        "metric_particle_samples": min(256, int(data.get("lagrangian_n_particles", 256))),
        "metric_preprocess_mode": "clip",
        "metric_alpha": 1.0,
        "metric_beta": 1.0,
        "metric_eps": 1e-12,
        "metric_dirs_seed": 123,
    }
    for key, value in defaults.items():
        if data.get(key, None) is None:
            data[key] = value
    if data.get("minibang_metric_tau_mode", None) is not None:
        data["metric_tau_mode"] = str(data["minibang_metric_tau_mode"])
    elif str(data.get("metric_tau_mode", "")).strip().lower() == "trainable_grid":
        data["metric_tau_mode"] = "max_grid"
    if data.get("metric_window_step_steps", None) is None:
        data["metric_window_step_steps"] = max(sample_every_steps, int(data["metric_window_size_steps"]) // 4)
    if data.get("metric_tau_steps", None) is None:
        data["metric_tau_steps"] = max(sample_every_steps, int(data["metric_window_size_steps"]) // 5)

    window = int(data.get("metric_window_size_steps"))
    if window >= rollout_steps:
        window = max(sample_every_steps * 2, rollout_steps // 2)
        data["metric_window_size_steps"] = max(sample_every_steps, window)
    if int(data.get("metric_tau_steps", sample_every_steps)) >= int(data["metric_window_size_steps"]):
        data["metric_tau_steps"] = max(sample_every_steps, int(data["metric_window_size_steps"]) // 4)

    grid = data.get("metric_tau_grid_steps", None)
    if grid is not None:
        grid_list = [int(x) for x in (OmegaConf.to_container(grid, resolve=True) if OmegaConf.is_config(grid) else grid)]
        grid_list = [x for x in grid_list if 0 < x < int(data["metric_window_size_steps"])]
        data["metric_tau_grid_steps"] = grid_list if grid_list else None

    if data.get("metric_range_start_steps", None) is None:
        data["metric_range_start_steps"] = 0
    if data.get("metric_range_end_steps", None) is None or int(data["metric_range_end_steps"]) > rollout_steps:
        data["metric_range_end_steps"] = int(rollout_steps)
    if int(data["metric_range_start_steps"]) >= int(data["metric_range_end_steps"]):
        data["metric_range_start_steps"] = 0
        data["metric_range_end_steps"] = int(rollout_steps)
    return SimpleNamespace(**data)


def _compute_delta_h_metrics(apf_dir: Path, flat_args: dict[str, Any], *, seed: int) -> dict[str, np.ndarray]:
    _ensure_jax()
    _ensure_metric_modules()
    steps, lag = _load_lagrangian_series(apf_dir)
    if lag.shape[0] < 4:
        raise ValueError("Need at least 4 lagrangian snapshots for deltaH.")
    diffs = np.diff(steps)
    positive = diffs[diffs > 0]
    sample_every = int(np.median(positive)) if positive.size else int(flat_args.get("snapshot_interval", 1))
    if steps[0] == 0 and lag.shape[0] > 1:
        xy = lag[1:]
        step_offset = int(sample_every)
    else:
        xy = lag
        step_offset = int(steps[0])
    rollout_steps = int(xy.shape[0] * sample_every)
    metric_args = _prepare_metric_args(flat_args, rollout_steps=rollout_steps, sample_every_steps=sample_every)
    metric_cfg = resolve_metric_config(metric_args)
    metric_eval = make_metric_loss_fn(metric_cfg, include_maps=True)
    _loss, info = metric_eval(jax.random.PRNGKey(seed), jnp.asarray(xy, dtype=jnp.float32))
    info_np = jax.device_get(info)

    window_start_steps = np.asarray(info_np["window_start_steps"], dtype=np.int64) + max(0, step_offset - sample_every)
    tau_steps = np.asarray(info_np["tau_steps"], dtype=np.int64)
    delta_h_map = np.asarray(info_np["delta_h_map"], dtype=np.float32)
    selected_idx = int(np.asarray(info_np["tau_selected_idx"]).item())
    selected_tau = int(np.asarray(info_np["tau_best_steps"]).item())
    window_size = int(metric_cfg["window_size_frames"]) * int(metric_cfg["sample_every_steps"])
    window_end_steps = window_start_steps + int(window_size)
    window_center_steps = window_start_steps + int(window_size // 2)

    return dict(
        delta_h_map=delta_h_map,
        delta_h_best=np.asarray(info_np["delta_h_best"], dtype=np.float32),
        delta_h_score_by_tau=np.asarray(info_np["score_by_tau"], dtype=np.float32),
        delta_h_amp_by_tau=np.asarray(info_np["amp_by_tau"], dtype=np.float32),
        delta_h_msc_by_tau=np.asarray(info_np["msc_by_tau"], dtype=np.float32),
        delta_h_tau_steps=tau_steps.astype(np.int32),
        delta_h_tau_frames=np.asarray(info_np["tau_frames"], dtype=np.int32),
        delta_h_selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
        delta_h_selected_tau_steps=np.asarray(selected_tau, dtype=np.int32),
        delta_h_window_start_steps=window_start_steps.astype(np.int64),
        delta_h_window_end_steps=window_end_steps.astype(np.int64),
        delta_h_window_center_steps=window_center_steps.astype(np.int64),
        delta_h_window_size_steps=np.asarray(window_size, dtype=np.int32),
        delta_h_sample_every_steps=np.asarray(sample_every, dtype=np.int32),
        delta_h_score_scalar=np.asarray(info_np["score"], dtype=np.float32),
        delta_h_amp_scalar=np.asarray(info_np["amp"], dtype=np.float32),
        delta_h_msc_scalar=np.asarray(info_np["msc"], dtype=np.float32),
    )


def compute_metrics_for_run(run: dict[str, Any], flat_args: dict[str, Any]) -> None:
    args = SimpleNamespace(**flat_args)
    metrics: dict[str, np.ndarray] = {}
    metrics_errors: dict[str, str] = {}
    strict = _as_bool(_get(args, "metrics_strict", True), True)
    seed = int(_get(args, "metrics_seed", 12345)) + int(run["selection"]["selection_idx"])

    if _as_bool(_get(args, "compute_clusters", True), True):
        try:
            metrics.update(_compute_cluster_metrics(run["apf_dir"], args, seed=seed + 17))
        except Exception as exc:
            if strict:
                raise
            metrics_errors["clusters"] = str(exc)
            print(f"[{run['traj_id']}] cluster metrics failed: {exc}")

    if _as_bool(_get(args, "compute_delta_h", True), True):
        try:
            metrics.update(_compute_delta_h_metrics(run["apf_dir"], flat_args, seed=seed + 31))
        except Exception as exc:
            if strict:
                raise
            metrics_errors["delta_h"] = str(exc)
            print(f"[{run['traj_id']}] deltaH metrics failed: {exc}")

    metrics["traj_selection_idx"] = np.asarray(run["selection"]["selection_idx"], dtype=np.int32)
    metrics["optimization_iter"] = np.asarray(run["selection"].get("iter", -1), dtype=np.int32)
    metrics["saturation_T"] = np.asarray(run["selection"].get("saturation_T", np.nan), dtype=np.float32)

    out_path = run["traj_dir"] / "metrics.npz"
    np.savez_compressed(out_path, **metrics)

    summary = {
        "traj_id": run["traj_id"],
        "metrics_path": str(out_path),
        "errors": metrics_errors,
        "keys": sorted(metrics.keys()),
    }
    if "delta_h_best" in metrics:
        dh = np.asarray(metrics["delta_h_best"], dtype=np.float64)
        summary["delta_h_best_max"] = float(np.nanmax(dh)) if dh.size else None
        summary["delta_h_best_mean"] = float(np.nanmean(dh)) if dh.size else None
    if "cluster_tv_lag" in metrics:
        tv = np.asarray(metrics["cluster_tv_lag"], dtype=np.float64)
        summary["cluster_tv_lag_max"] = float(np.nanmax(tv)) if tv.size else None
        summary["cluster_tv_lag_mean"] = float(np.nanmean(tv)) if tv.size else None
    write_json(run["traj_dir"] / "metrics_summary.json", summary)


def _capture_snapshot(
    *,
    step: int,
    states: Any,
    lag_xy: jax.Array,
    lag_ch: jax.Array,
    params_batch: jax.Array,
    substrate: Any,
    runs: list[dict[str, Any]],
    args: Any,
    capture_fn_cache: dict[tuple[int, int], Any],
) -> None:
    _ensure_jax()
    B = int(params_batch.shape[0])
    img_size = int(_get(args, "img_size", _get(args, "video_img_size", 224)))
    cache_key = (B, img_size)
    if cache_key not in capture_fn_cache:
        def capture_fn(states_in, params_in, lag_xy_in, lag_ch_in):
            rgb = jax.vmap(lambda st, p: substrate.render_state(st, p, img_size=img_size))(states_in, params_in)
            return states_in["P"], states_in["A"], states_in["F"], rgb, lag_xy_in, lag_ch_in

        capture_fn_cache[cache_key] = jax.jit(capture_fn)

    P, A, F, rgb, lag_np, lag_ch_np = jax.device_get(capture_fn_cache[cache_key](states, params_batch, lag_xy, lag_ch))
    chunk_size = max(1, int(_get(args, "snapshots_per_file", 200)))
    save_A = _as_bool(_get(args, "save_A", True), True)
    save_F = _as_bool(_get(args, "save_F", True), True)
    save_rgb = _as_bool(_get(args, "save_rgb", False), False)
    save_lagrangian = _as_bool(_get(args, "save_lagrangian", True), True)
    save_lagrangian_channels = _as_bool(_get(args, "save_lagrangian_channels", True), True)

    for i, run in enumerate(runs):
        frame = _frame_u8(rgb[i])
        run["writer"].append_data(frame)
        run["frame_rows"].append(
            dict(
                frame_idx=int(run["frame_idx"]),
                step=int(step),
                video_sec=float(run["frame_idx"]) / float(run["video_fps"]),
                sim_sec=float(step) / float(run["sim_fps"]),
            )
        )
        run["frame_idx"] += 1

        buffers = run["buffers"]
        buffers["steps"].append(int(step))
        buffers["P"].append(np.asarray(P[i]))
        if save_A:
            buffers["A"].append(np.asarray(A[i]))
        if save_F:
            buffers["F"].append(np.asarray(F[i]))
        if save_rgb:
            buffers["rgb"].append(frame)
        if save_lagrangian:
            buffers["lagrangian_xy"].append(np.asarray(lag_np[i]))
            if save_lagrangian_channels:
                buffers["lagrangian_c"].append(np.asarray(lag_ch_np[i]))
        if len(buffers["steps"]) >= chunk_size:
            _flush_run_buffers(run, args)


def simulate_batch(
    *,
    selected_batch: list[dict[str, Any]],
    cfg: Any,
    flat_args: dict[str, Any],
    output_root: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    _ensure_jax()
    args = SimpleNamespace(**flat_args)
    substrate = _make_substrate(args)
    params_np = np.stack([np.asarray(row["params"], dtype=np.float32) for row in selected_batch], axis=0)
    expected = int(substrate.n_params)
    if params_np.shape[1] != expected:
        raise ValueError(f"Loaded params have dim {params_np.shape[1]}, substrate expects {expected}.")

    # Build RT before jitted lagrangian helpers close over it.
    _ = substrate.init_state(jax.random.PRNGKey(0), jnp.asarray(params_np[0]))
    rt = substrate.RT

    runs = _init_run_dirs(selected_batch, output_root=output_root, cfg=cfg, args=args, overwrite=overwrite)
    B = len(selected_batch)
    params_batch = jnp.asarray(params_np)
    seed0 = int(_get(args, "seed", 0))
    selection0 = int(selected_batch[0]["selection_idx"])
    init_keys = jax.random.split(jax.random.PRNGKey(seed0 + 1009 * (selection0 + 1)), B)
    lag_keys = jax.random.split(jax.random.PRNGKey(int(_get(args, "lagrangian_seed", seed0)) + 9173 * (selection0 + 1)), B)
    ch_keys = jax.random.split(jax.random.PRNGKey(int(_get(args, "lagrangian_seed", seed0)) + 1877 * (selection0 + 1)), B)

    init_states = jax.jit(lambda keys, params: jax.vmap(substrate.init_state)(keys, params))
    states = init_states(init_keys, params_batch)

    lag_n = int(_get(args, "lagrangian_n_particles", _get(args, "metric_lagrangian_n_particles", 2048)))
    lag_init_mode = str(_get(args, "lagrangian_init_mode", _get(args, "metric_lagrangian_init_mode", "mass")))
    lag_flow_channel = int(_get(args, "lagrangian_flow_channel", _get(args, "metric_lagrangian_flow_channel", -1)))
    lag_flow_reduce = str(_get(args, "lagrangian_flow_reduce", _get(args, "metric_lagrangian_flow_reduce", "mass_weighted")))
    lag_channel_mode = str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample")))
    lag_noise_model = str(_get(args, "lagrangian_noise_model", _get(args, "metric_lagrangian_noise_model", "rt_box")))
    lag_diffusion_scale = float(_get(args, "lagrangian_diffusion_scale", _get(args, "metric_lagrangian_diffusion_scale", 1.0)))

    def init_lag_one(A0, key_pts, key_ch):
        pts = _init_lagrangian_points_jax(
            A0,
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=key_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch = rt.sample_point_channels(pts, A0, key_ch)
        else:
            ch = jnp.zeros((lag_n,), dtype=jnp.int32)
        return pts, ch

    lag_xy, lag_ch = jax.jit(lambda A0, kp, kc: jax.vmap(init_lag_one)(A0, kp, kc))(states["A"], lag_keys, ch_keys)

    stepper_cache: dict[tuple[int, int], Any] = {}

    def get_stepper(n_steps: int):
        key = (int(n_steps), B)
        if key in stepper_cache:
            return stepper_cache[key]

        def advance(states_in, lag_xy_in, lag_ch_in, params_in, rng_in):
            rngs = jax.random.split(rng_in, int(n_steps) * B).reshape((int(n_steps), B, 2))

            def scan_body(carry, keys_step):
                st, pts, ch = carry

                def one_step(key_i, st_i, pts_i, ch_i, params_i):
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

                return jax.vmap(one_step)(keys_step, st, pts, ch, params_in), None

            (st_out, pts_out, ch_out), _ = jax.lax.scan(scan_body, (states_in, lag_xy_in, lag_ch_in), rngs)
            return st_out, pts_out, ch_out

        stepper_cache[key] = jax.jit(advance)
        return stepper_cache[key]

    capture_cache: dict[tuple[int, int], Any] = {}
    snapshot_interval = max(1, int(_get(args, "snapshot_interval", 50)))
    total_steps_raw = _get(args, "rollout_steps", None)
    if total_steps_raw is None:
        total_steps_raw = _get(args, "max_steps", substrate.rollout_steps)
    if total_steps_raw is None:
        total_steps_raw = substrate.rollout_steps
    total_steps = int(total_steps_raw)
    max_steps = _get(args, "max_steps", None)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    jit_microbatch = max(1, int(_get(args, "jit_microbatch", min(64, snapshot_interval))))

    _capture_snapshot(
        step=0,
        states=states,
        lag_xy=lag_xy,
        lag_ch=lag_ch,
        params_batch=params_batch,
        substrate=substrate,
        runs=runs,
        args=args,
        capture_fn_cache=capture_cache,
    )

    rng = jax.random.PRNGKey(seed0 + 991 * (selection0 + 1))
    steps_done = 0
    pbar = tqdm(total=total_steps, desc=f"batch {runs[0]['traj_id']}..{runs[-1]['traj_id']}")
    try:
        while steps_done < total_steps:
            target_next_snapshot = min(total_steps, ((steps_done // snapshot_interval) + 1) * snapshot_interval)
            while steps_done < target_next_snapshot:
                n = min(jit_microbatch, target_next_snapshot - steps_done)
                rng, subkey = jax.random.split(rng)
                states, lag_xy, lag_ch = get_stepper(n)(states, lag_xy, lag_ch, params_batch, subkey)
                steps_done += n
                pbar.update(n)
            _capture_snapshot(
                step=steps_done,
                states=states,
                lag_xy=lag_xy,
                lag_ch=lag_ch,
                params_batch=params_batch,
                substrate=substrate,
                runs=runs,
                args=args,
                capture_fn_cache=capture_cache,
            )
    finally:
        pbar.close()
        for run in runs:
            _flush_run_buffers(run, args)
        _close_runs(runs)

    flat_plain = dict(flat_args)
    for run in runs:
        if _as_bool(_get(args, "compute_metrics", True), True):
            compute_metrics_for_run(run, flat_plain)
    return runs


def _write_manifest(
    output_root: Path,
    selected: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    cfg_path: Path,
    checkpoint_dir: Path,
    flat_args: Any,
) -> None:
    manifest_rows: list[dict[str, Any]] = []
    run_by_id = {run["traj_id"]: run for run in runs}
    for row in selected:
        traj_id = str(row["traj_id"])
        run = run_by_id.get(traj_id)
        item = {k: v for k, v in row.items() if k != "params"}
        if run is not None:
            item.update(
                traj_dir=str(run["traj_dir"]),
                apf_dir=str(run["apf_dir"]),
                video_path=str(run["traj_dir"] / "video.mp4"),
                metrics_path=str(run["traj_dir"] / "metrics.npz"),
                frame_times_path=str(run["traj_dir"] / "frame_times.csv"),
            )
        manifest_rows.append(item)

    write_json(
        output_root / "manifest.json",
        dict(
            config_path=str(cfg_path),
            checkpoint_dir=str(checkpoint_dir),
            n_trajectories=len(manifest_rows),
            detect_start_step=_get(flat_args, "detect_start_step", None),
            detect_end_step=_get(flat_args, "detect_end_step", None),
            detect_max_duration_steps=_get(flat_args, "detect_max_duration_steps", None),
            trajectories=manifest_rows,
        ),
    )

    csv_path = output_root / "manifest.csv"
    if manifest_rows:
        fieldnames = sorted({k for row in manifest_rows for k in row.keys()})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow({k: to_plain(row.get(k, "")) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FlowLenia minibang candidate dataset from optimization checkpoints."
    )
    parser.add_argument("config", help="Base YAML config with substrate/simulation/logging/metric/minibang sections.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing best.pkl/pop_traj.pkl/best_traj.pkl.")
    parser.add_argument("--output-root", default=None, help="Output dataset directory.")
    parser.add_argument("--n-trajectories", type=int, default=None, help="Override minibang.n_trajectories.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override minibang.batch_size.")
    parser.add_argument("--select-only", action="store_true", help="Only write selected_params.json/csv, do not simulate.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing traj_XXXXX directories.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def main() -> None:
    cli = parse_args()
    root = project_root()
    cfg_path = resolve_path(cli.config, root)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cli.config}")
    cfg, flat = load_config(cfg_path, cli.overrides)
    if cli.n_trajectories is not None:
        flat.n_trajectories = int(cli.n_trajectories)
    if cli.batch_size is not None:
        flat.batch_size = int(cli.batch_size)

    checkpoint_dir_raw = cli.checkpoint_dir or _get(flat, "checkpoint_dir", None) or _get(flat, "save_dir", None)
    checkpoint_dir = resolve_path(checkpoint_dir_raw, root)
    if checkpoint_dir is None or not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir_raw}")

    output_root_raw = cli.output_root or _get(flat, "output_root", None)
    if output_root_raw is None:
        output_root = checkpoint_dir / "minibang_dataset"
    else:
        output_root = resolve_path(output_root_raw, root)
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    selected = select_params(checkpoint_dir, flat)
    write_json(
        output_root / "selected_params.json",
        [{k: v for k, v in row.items() if k != "params"} for row in selected],
    )
    with (output_root / "selected_params.csv").open("w", newline="") as f:
        fieldnames = sorted({k for row in selected for k in row.keys() if k != "params"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: to_plain(row.get(k, "")) for k in fieldnames})

    print(f"Selected {len(selected)} trajectories from {checkpoint_dir}")
    print(f"Output root: {output_root}")
    if cli.select_only:
        print("select-only requested; stopping before simulation.")
        _write_manifest(output_root, selected, [], cfg_path, checkpoint_dir, flat)
        return

    batch_size = max(1, int(_get(flat, "batch_size", 2)))
    flat_dict = OmegaConf.to_container(flat, resolve=True)
    all_runs: list[dict[str, Any]] = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        runs = simulate_batch(
            selected_batch=batch,
            cfg=cfg,
            flat_args=dict(flat_dict),
            output_root=output_root,
            overwrite=bool(cli.overwrite),
        )
        all_runs.extend(runs)
        _write_manifest(output_root, selected, all_runs, cfg_path, checkpoint_dir, flat)

    _write_manifest(output_root, selected, all_runs, cfg_path, checkpoint_dir, flat)
    print(f"Done. Dataset manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
