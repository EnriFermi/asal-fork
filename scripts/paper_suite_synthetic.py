from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np

from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config
from paper_suite_common import ensure_dir, load_config, log_event, resolve_path, sign_test_greater, to_plain, write_csv, write_json


FAMILIES = ("S0", "S1", "S3", "S4", "S5", "S6", "S7")


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    value = cfg.get(key, default) if cfg is not None else default
    return int(default if value is None else value)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    value = cfg.get(key, default) if cfg is not None else default
    return float(default if value is None else value)


def _tau_grid(cfg: Any) -> list[int]:
    vals = cfg.get("tau_grid_steps", [1, 2, 4, 8, 16, 32, 64]) if cfg is not None else [1, 2, 4, 8, 16, 32, 64]
    return [int(x) for x in vals]


def _periodic_delta(dx: np.ndarray, domain: float = 1.0) -> np.ndarray:
    return (dx + 0.5 * domain) % domain - 0.5 * domain


def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack((np.sin(theta), np.cos(theta)), axis=-1).astype(np.float32)


def _simulate_s0(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    x0 = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.repeat(x0[None, :, :], T, axis=0)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "static_null"}}


def _simulate_s1(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    sigma = 0.006
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        x = np.mod(x + rng.normal(0.0, sigma, size=(N, 2)).astype(np.float32), L)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "homogeneous_motion_null"}}


def _simulate_s3(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    radius = 0.08
    direction = _random_unit_vectors(rng, 1)[0]
    speed = 0.002
    center0 = rng.uniform(0.2, 0.8, size=(2,)).astype(np.float32)
    offsets = _random_unit_vectors(rng, N) * rng.uniform(0.0, radius, size=(N, 1)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        center = np.mod(center0 + direction * speed * t, L)
        jitter = rng.normal(0.0, 0.0015, size=(N, 2)).astype(np.float32)
        xy[t] = np.mod(center[None, :] + offsets + jitter, L)
    return {"xy": xy, "labels": np.zeros(N, dtype=np.int32), "metadata": {"expected": "coherent_motion_low_complexity"}}


def _simulate_s4(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    labels = np.arange(N, dtype=np.int32) % 2
    rng.shuffle(labels)
    v = np.zeros((N, 2), dtype=np.float32)
    v[labels == 0] = np.asarray([0.0008, 0.0], dtype=np.float32)
    v[labels == 1] = np.asarray([0.0, 0.0050], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        x = np.mod(x + v + rng.normal(0.0, 0.0008, size=(N, 2)).astype(np.float32), L)
    return {"xy": xy, "labels": labels, "metadata": {"expected": "two_role_positive_control"}}


def _simulate_s5(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    t0 = int(0.5 * T)
    v0 = np.asarray([0.0015, 0.0], dtype=np.float32)
    v1 = np.asarray([0.0, 0.0015], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        xy[t] = x
        v = v0 if t < t0 else v1
        x = np.mod(x + v + rng.normal(0.0, 0.0009, size=(N, 2)).astype(np.float32), L)
    return {
        "xy": xy,
        "labels": np.zeros(N, dtype=np.int32),
        "metadata": {"expected": "global_switch_not_generic_changepoint", "event_interval": [t0, t0]},
    }


def _simulate_s6(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    t0 = int(0.40 * T)
    t1 = int(0.65 * T)
    switch_times = rng.integers(t0, max(t0 + 1, t1), size=N)
    v_old = np.asarray([0.0010, 0.0], dtype=np.float32)
    v_new = np.asarray([0.0, 0.0050], dtype=np.float32)
    x = rng.uniform(0.0, L, size=(N, 2)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    labels_t = np.empty((T, N), dtype=np.int8)
    for t in range(T):
        switched = t >= switch_times
        labels_t[t] = switched.astype(np.int8)
        xy[t] = x
        v = np.where(switched[:, None], v_new[None, :], v_old[None, :]).astype(np.float32)
        x = np.mod(x + v + rng.normal(0.0, 0.0008, size=(N, 2)).astype(np.float32), L)
    labels_mid = labels_t[(t0 + t1) // 2].astype(np.int32)
    return {
        "xy": xy,
        "labels": labels_mid,
        "labels_t": labels_t,
        "metadata": {"expected": "partial_transition_positive_control", "event_interval": [t0, t1]},
    }


def _simulate_s7(rng: np.random.Generator, *, T: int, N: int, L: float) -> dict[str, Any]:
    specs = [
        {"radius": 0.11, "speed": 0.0016, "direction": np.asarray([1.0, 0.2], dtype=np.float32)},
        {"radius": 0.04, "speed": 0.0055, "direction": np.asarray([-0.2, 1.0], dtype=np.float32)},
        {"radius": 0.025, "speed": 0.0100, "direction": np.asarray([0.8, -0.6], dtype=np.float32)},
    ]
    n_groups = len(specs)
    labels = np.arange(N, dtype=np.int32) % n_groups
    rng.shuffle(labels)
    centers0 = rng.uniform(0.15, 0.85, size=(n_groups, 2)).astype(np.float32)
    offsets = np.zeros((N, 2), dtype=np.float32)
    for group, spec in enumerate(specs):
        idx = np.flatnonzero(labels == group)
        if idx.size:
            offsets[idx] = _random_unit_vectors(rng, idx.size) * rng.uniform(0.0, spec["radius"], size=(idx.size, 1)).astype(np.float32)
    xy = np.empty((T, N, 2), dtype=np.float32)
    for t in range(T):
        frame = np.empty((N, 2), dtype=np.float32)
        for group, spec in enumerate(specs):
            direction = spec["direction"] / np.linalg.norm(spec["direction"])
            center = np.mod(centers0[group] + direction * float(spec["speed"]) * t, L)
            idx = np.flatnonzero(labels == group)
            if idx.size:
                frame[idx] = center[None, :] + offsets[idx] + rng.normal(0.0, 0.001, size=(idx.size, 2)).astype(np.float32)
        xy[t] = np.mod(frame, L)
    crosses = [float(spec["radius"]) / max(float(spec["speed"]), 1e-12) for spec in specs]
    return {
        "xy": xy,
        "labels": labels,
        "metadata": {
            "expected": "multiscale_tau_calibration",
            "scale_range": [float(min(crosses)), float(max(crosses))],
            "crossing_times": crosses,
        },
    }


SIMULATORS = {
    "S0": _simulate_s0,
    "S1": _simulate_s1,
    "S3": _simulate_s3,
    "S4": _simulate_s4,
    "S5": _simulate_s5,
    "S6": _simulate_s6,
    "S7": _simulate_s7,
}


def _synthetic_dirs(cfg: Any) -> dict[str, Path]:
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    root = ensure_dir(output_root / "synthetic_calibration")
    return {
        "root": root,
        "simulation": ensure_dir(root / "simulation"),
        "metrics": ensure_dir(root / "metrics"),
    }


def _cached_simulation_matches(path: Path, *, family: str, seed: int, T: int, N: int, L: float) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            xy_shape = tuple(np.asarray(data["xy"]).shape)
            metadata = __import__("json").loads(str(np.asarray(data["metadata_json"]).item()))
    except Exception:
        return False
    return (
        xy_shape == (int(T), int(N), 2)
        and str(metadata.get("family")) == str(family)
        and int(metadata.get("seed", -1)) == int(seed)
        and int(metadata.get("time_steps", -1)) == int(T)
        and int(metadata.get("n_particles", -1)) == int(N)
        and abs(float(metadata.get("domain_size", float("nan"))) - float(L)) < 1e-12
    )


def simulate(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    syn = cfg.get("synthetic", {})
    dirs = _synthetic_dirs(cfg)
    T = _cfg_int(syn, "time_steps", 2000)
    N = _cfg_int(syn, "n_particles", 256)
    L = _cfg_float(syn, "domain_size", 1.0)
    seeds = _cfg_int(syn, "seeds", 3)
    families = [str(x) for x in (syn.get("families", list(FAMILIES)) or list(FAMILIES))]
    manifest_rows = []
    log_event(
        f"synthetic simulation start smoke={smoke} force={force} families={families} seeds={seeds} T={T} N={N}",
        component="synthetic",
    )
    for family in families:
        if family not in SIMULATORS:
            raise ValueError(f"Unknown synthetic family {family!r}. Expected one of {sorted(SIMULATORS)}.")
        for seed in range(seeds):
            out_path = dirs["simulation"] / f"{family}_seed_{seed:03d}.npz"
            had_existing = out_path.exists()
            if out_path.exists() and not force and _cached_simulation_matches(out_path, family=family, seed=seed, T=T, N=N, L=L):
                status = "exists"
            else:
                rng = np.random.default_rng(_cfg_int(syn, "seed_base", 100) + seed + 1009 * families.index(family))
                payload = SIMULATORS[family](rng, T=T, N=N, L=L)
                metadata = dict(payload.get("metadata", {}))
                metadata.update({"family": family, "seed": int(seed), "time_steps": int(T), "n_particles": int(N), "domain_size": float(L)})
                save_payload = {
                    "xy": np.asarray(payload["xy"], dtype=np.float32),
                    "labels": np.asarray(payload.get("labels", np.zeros(N, dtype=np.int32)), dtype=np.int32),
                    "metadata_json": np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
                }
                if "labels_t" in payload:
                    save_payload["labels_t"] = np.asarray(payload["labels_t"], dtype=np.int8)
                np.savez_compressed(out_path, **save_payload)
                status = "rewritten_stale" if had_existing and not force else "written"
            manifest_rows.append({"family": family, "seed": seed, "path": str(out_path), "status": status})
            log_event(f"synthetic simulation {family} seed={seed} status={status}", component="synthetic")
    write_csv(dirs["root"] / "simulation_manifest.csv", manifest_rows)
    write_json(dirs["root"] / "simulation_summary.json", {"n_runs": len(manifest_rows), "time_steps": T, "n_particles": N})
    log_event(f"synthetic simulation done n_runs={len(manifest_rows)} manifest={dirs['root'] / 'simulation_manifest.csv'}", component="synthetic")
    return {"simulation_manifest": str(dirs["root"] / "simulation_manifest.csv"), "n_runs": len(manifest_rows)}


def _build_metric_cfg(syn_cfg: Any, T: int) -> dict[str, Any]:
    args = SimpleNamespace(
        rollout_steps=int(T),
        sample_every_steps=1,
        time_sampling=None,
        metric_window_size_steps=_cfg_int(syn_cfg, "metric_window_size_steps", 200),
        metric_window_step_steps=_cfg_int(syn_cfg, "metric_window_step_steps", 50),
        metric_tau_mode="max_grid",
        metric_tau_steps=_tau_grid(syn_cfg)[0],
        metric_tau_grid_steps=_tau_grid(syn_cfg),
        metric_window_size_frames=None,
        metric_window_step_frames=None,
        metric_tau_frames=None,
        metric_tau_grid_frames=None,
        metric_range_start_steps=_cfg_int(syn_cfg, "metric_range_start_steps", 0),
        metric_range_end_steps=None,
        metric_m_samples=_cfg_int(syn_cfg, "metric_m_samples", 32),
        metric_m_min=_cfg_int(syn_cfg, "metric_m_min", 4),
        metric_n_proj=_cfg_int(syn_cfg, "metric_n_proj", 12),
        metric_null_reps=_cfg_int(syn_cfg, "metric_null_reps", 4),
        metric_particle_samples=_cfg_int(syn_cfg, "metric_particle_samples", 64),
        metric_dirs_seed=_cfg_int(syn_cfg, "metric_dirs_seed", 123),
        metric_periodic=True,
        metric_domain_y=_cfg_float(syn_cfg, "domain_size", 1.0),
        metric_domain_x=_cfg_float(syn_cfg, "domain_size", 1.0),
        metric_preprocess_mode="clip",
        metric_scales=None,
        metric_scale_weights=None,
        metric_alpha=0.0,
        metric_beta=1.0,
        metric_eps=1e-12,
    )
    return resolve_metric_config(args)


def _kmeans(features: np.ndarray, k: int, *, seed: int, n_iters: int = 40) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"kmeans expects 2D features, got {x.shape}.")
    n = x.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=min(k, n), replace=False)].copy()
    if centers.shape[0] < k:
        centers = np.concatenate([centers, np.repeat(centers[-1:], k - centers.shape[0], axis=0)], axis=0)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iters):
        d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(d, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = np.mean(x[labels == j], axis=0)
    return labels


def _adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.int64).reshape(-1)
    y = np.asarray(b, dtype=np.int64).reshape(-1)
    if x.size != y.size:
        raise ValueError("ARI label vectors must have the same length.")
    n = int(x.size)
    if n < 2:
        return float("nan")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    contingency = np.zeros((xi.max() + 1, yi.max() + 1), dtype=np.int64)
    for i, j in zip(xi, yi):
        contingency[i, j] += 1

    def comb2(v: np.ndarray) -> float:
        vv = np.asarray(v, dtype=np.float64)
        return float(np.sum(vv * (vv - 1.0) / 2.0))

    sum_comb = comb2(contingency)
    row_comb = comb2(np.sum(contingency, axis=1))
    col_comb = comb2(np.sum(contingency, axis=0))
    total = n * (n - 1.0) / 2.0
    expected = row_comb * col_comb / total if total > 0 else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb - expected) < 1e-12 else 0.0
    return float((sum_comb - expected) / denom)


def _role_recovery(xy: np.ndarray, labels: np.ndarray, tau: int, *, seed: int, domain: float) -> dict[str, Any] | None:
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    tau = int(max(1, min(tau, xy.shape[0] - 1)))
    dx = _periodic_delta(xy[tau:] - xy[:-tau], domain=domain)
    speed = np.linalg.norm(dx, axis=-1)
    feats = np.concatenate(
        [
            np.mean(dx, axis=0),
            np.std(dx, axis=0),
            np.mean(speed, axis=0, keepdims=True).T,
            np.std(speed, axis=0, keepdims=True).T,
        ],
        axis=1,
    )
    pred = _kmeans(feats, int(unique.size), seed=seed)
    return {"ari": _adjusted_rand_index(labels, pred), "n_roles": int(unique.size)}


def _event_error(window_centers: np.ndarray, values: np.ndarray, interval: list[int]) -> dict[str, Any]:
    idx = int(np.nanargmax(values))
    peak = float(window_centers[idx])
    lo, hi = float(interval[0]), float(interval[1])
    if lo <= peak <= hi:
        error = 0.0
    else:
        error = min(abs(peak - lo), abs(peak - hi))
    return {"peak_step": peak, "event_start": lo, "event_end": hi, "event_error_steps": float(error)}


def _metric_cache_payload(*, metric_cfg: dict[str, Any], trajectory_path: Path, trajectory_metadata: dict[str, Any], xy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "_paper_suite_cache_version": np.asarray(1, dtype=np.int32),
        "_metric_config_json": np.asarray(json.dumps(to_plain(metric_cfg), sort_keys=True)),
        "_trajectory_path": np.asarray(str(trajectory_path)),
        "_trajectory_metadata_json": np.asarray(json.dumps(to_plain(trajectory_metadata), sort_keys=True)),
        "_trajectory_shape": np.asarray(xy.shape, dtype=np.int32),
    }


def _metric_cache_matches(cached: np.lib.npyio.NpzFile, *, metric_cfg: dict[str, Any], trajectory_path: Path, trajectory_metadata: dict[str, Any], xy: np.ndarray) -> bool:
    required = {
        "_paper_suite_cache_version",
        "_metric_config_json",
        "_trajectory_path",
        "_trajectory_metadata_json",
        "_trajectory_shape",
    }
    if not required.issubset(set(cached.files)):
        return False
    try:
        return (
            int(np.asarray(cached["_paper_suite_cache_version"]).item()) == 1
            and str(np.asarray(cached["_metric_config_json"]).item()) == json.dumps(to_plain(metric_cfg), sort_keys=True)
            and str(np.asarray(cached["_trajectory_path"]).item()) == str(trajectory_path)
            and str(np.asarray(cached["_trajectory_metadata_json"]).item()) == json.dumps(to_plain(trajectory_metadata), sort_keys=True)
            and tuple(np.asarray(cached["_trajectory_shape"], dtype=np.int32).tolist()) == tuple(int(x) for x in xy.shape)
        )
    except Exception:
        return False


def metrics(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    syn = cfg.get("synthetic", {})
    dirs = _synthetic_dirs(cfg)
    families = [str(x) for x in (syn.get("families", list(FAMILIES)) or list(FAMILIES))]
    seeds = _cfg_int(syn, "seeds", 3)
    log_event(f"synthetic metrics start smoke={smoke} force={force} families={families} seeds={seeds}", component="synthetic")
    sim_files = [dirs["simulation"] / f"{family}_seed_{seed:03d}.npz" for family in families for seed in range(seeds)]
    missing = [str(path) for path in sim_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} configured synthetic simulation files in {dirs['simulation']}. "
            "Run layer=simulation/task=synthetic first. First missing file: "
            f"{missing[0]}"
        )
    with np.load(sim_files[0], allow_pickle=False) as first:
        T = int(np.asarray(first["xy"]).shape[0])
    metric_cfg = _build_metric_cfg(syn, T)
    metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))
    domain = _cfg_float(syn, "domain_size", 1.0)

    score_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for idx, path in enumerate(sim_files, start=1):
        with np.load(path, allow_pickle=False) as data:
            xy = np.asarray(data["xy"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int32)
            metadata = __import__("json").loads(str(np.asarray(data["metadata_json"]).item()))
        family = str(metadata["family"])
        seed = int(metadata["seed"])
        metrics_path = dirs["metrics"] / f"{family}_seed_{seed:03d}_metrics.npz"
        if metrics_path.exists() and not force:
            with np.load(metrics_path, allow_pickle=False) as cached:
                if _metric_cache_matches(cached, metric_cfg=metric_cfg, trajectory_path=path, trajectory_metadata=metadata, xy=xy):
                    info_np = {key: np.asarray(cached[key]) for key in cached.files}
                else:
                    info_np = {}
        else:
            info_np = {}
        if not info_np:
            log_event(f"synthetic metrics computing {idx}/{len(sim_files)} {family} seed={seed} from {path.name}", component="synthetic")
            rng = jax.random.PRNGKey(_cfg_int(syn, "metric_seed", 12345) + seed)
            _loss, info = metric_eval(rng, jnp.asarray(xy))
            info_np = {key: np.asarray(jax.device_get(value)) for key, value in info.items()}
            info_np.update(_metric_cache_payload(metric_cfg=metric_cfg, trajectory_path=path, trajectory_metadata=metadata, xy=xy))
            np.savez_compressed(metrics_path, **info_np)
        else:
            log_event(f"synthetic metrics exists {idx}/{len(sim_files)} {family} seed={seed} metrics={metrics_path}", component="synthetic")

        tau_steps = np.asarray(info_np["tau_steps"], dtype=np.int32)
        score_by_tau = np.asarray(info_np["score_by_tau"], dtype=np.float64)
        delta_h_map = np.asarray(info_np["delta_h_map"], dtype=np.float64)
        best_idx = int(np.asarray(info_np["tau_selected_idx"]).item())
        best_tau = int(tau_steps[best_idx])
        win_starts = np.asarray(info_np["window_start_steps"], dtype=np.float64)
        win_centers = win_starts + 0.5 * int(metric_cfg["window_size_frames"])

        score_rows.append(
            {
                "family": family,
                "seed": seed,
                "score": float(np.asarray(info_np["score"]).item()),
                "msc": float(np.asarray(info_np["msc"]).item()),
                "amp": float(np.asarray(info_np["amp"]).item()),
                "delta_h_mean": float(np.asarray(info_np["delta_h_mean"]).item()),
                "delta_h_std": float(np.asarray(info_np["delta_h_std"]).item()),
                "tau_best_steps": best_tau,
                "metrics_path": str(metrics_path),
                "trajectory_path": str(path),
            }
        )
        for i, tau in enumerate(tau_steps):
            tau_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "tau_steps": int(tau),
                    "score_by_tau": float(score_by_tau[i]),
                    "delta_h_median": float(np.nanmedian(delta_h_map[i])),
                    "delta_h_mean": float(np.nanmean(delta_h_map[i])),
                    "selected": bool(i == best_idx),
                }
            )

        rec = _role_recovery(xy, labels, best_tau, seed=seed, domain=domain)
        if rec is not None:
            role_rows.append({"family": family, "seed": seed, "tau_steps": best_tau, **rec})

        event_interval = metadata.get("event_interval")
        if event_interval is not None:
            event_rows.append({"family": family, "seed": seed, "tau_steps": best_tau, **_event_error(win_centers, delta_h_map[best_idx], event_interval)})

        scale_range = metadata.get("scale_range")
        if scale_range is not None:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            role_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "tau_steps": best_tau,
                    "ari": "" if rec is None else rec["ari"],
                    "n_roles": "" if rec is None else rec["n_roles"],
                    "scale_low": lo,
                    "scale_high": hi,
                    "tau_in_scale_range": bool(lo <= best_tau <= hi),
                }
            )

    write_csv(dirs["root"] / "per_family_scores.csv", score_rows)
    write_csv(dirs["root"] / "tau_profiles.csv", tau_rows)
    write_csv(dirs["root"] / "role_recovery.csv", role_rows)
    write_csv(dirs["root"] / "event_localization.csv", event_rows)

    summary: dict[str, Any] = {"n_runs": len(score_rows), "families": {}}
    for family in sorted({row["family"] for row in score_rows}):
        vals = [row["score"] for row in score_rows if row["family"] == family]
        tau_vals = [row["tau_best_steps"] for row in score_rows if row["family"] == family]
        summary["families"][family] = {
            "n": len(vals),
            "score_median": float(np.median(vals)) if vals else float("nan"),
            "tau_best_median": float(np.median(tau_vals)) if tau_vals else float("nan"),
        }
    if role_rows:
        aris = [float(row["ari"]) for row in role_rows if row.get("ari") not in ("", None)]
        summary["role_recovery"] = sign_test_greater(aris)
    write_json(dirs["root"] / "synthetic_calibration_summary.json", summary)
    log_event(f"synthetic metrics done n_runs={len(score_rows)} summary={dirs['root'] / 'synthetic_calibration_summary.json'}", component="synthetic")
    return {"n_runs": len(score_rows), "summary_path": str(dirs["root"] / "synthetic_calibration_summary.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mandatory synthetic MSPD calibration S0-S7.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--layer", choices=["simulation", "metrics", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.layer in {"simulation", "all"}:
        print(simulate(args.config, smoke=args.smoke, force=args.force))
    if args.layer in {"metrics", "all"}:
        print(metrics(args.config, smoke=args.smoke, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
