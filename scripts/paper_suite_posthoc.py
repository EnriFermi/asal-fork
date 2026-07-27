from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
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
import pandas as pd
from omegaconf import OmegaConf

from analysis.history_dependence.paper_check_metric_stats import (
    augment_rows_with_history_dependence_distances,
    history_distance_base_names,
)
from clip_deltah_msc_metric import make_metric_loss_fn, resolve_metric_config
from flowlenia_minibang_simulate import _init_lagrangian_points_jax, _load_lagrangian_series, _make_substrate
from paper_suite_common import (
    REPO_ROOT,
    as_list,
    cfg_get,
    dataset_items,
    ensure_dir,
    load_config,
    log_event,
    nanmedian,
    resolve_path,
    safe_float,
    sign_test_greater,
    write_csv,
    write_json,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _source_root_label(root: Path) -> str:
    root = Path(root)
    label = root.parent.name if root.name == "frustration_simulation" else root.name
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label) or "root"


def _resolve_frustration_root(root: Path) -> Path:
    root = Path(root)
    if (root / "trial_results.csv").exists() or (root / "trial_data").exists():
        return root
    if (root / "frustration_simulation").exists():
        return root / "frustration_simulation"
    return root


def _load_trial_rows(root: Path) -> pd.DataFrame:
    fs_root = _resolve_frustration_root(root)
    csv_path = fs_root / "trial_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        trial_dir = fs_root / "trial_data"
        if not trial_dir.exists():
            raise FileNotFoundError(f"Expected {csv_path} or {trial_dir}.")
        rows = [json.loads(path.read_text()) for path in sorted(trial_dir.glob("trial_*.json")) if not path.name.endswith("_summary.json")]
        if not rows:
            raise FileNotFoundError(f"No trial JSON rows found in {trial_dir}.")
        df = pd.DataFrame(rows)
    df = df.copy()
    df["frustration_root"] = str(fs_root)
    df["source_root"] = str(root)
    df["source_root_name"] = _source_root_label(root)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    if "candidate_kind" not in df.columns:
        df["candidate_kind"] = "other"
    df["candidate_kind_canon"] = [
        _canonicalize_kind(kind, label)
        for kind, label in zip(df.get("candidate_kind", ""), df.get("candidate_label", ""))
    ]
    return df.sort_values(["optimized_run_idx", "candidate_kind_canon", "candidate_idx"], na_position="last").reset_index(drop=True)


def _dedupe_trial_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    if "candidate_idx" not in out.columns:
        out["candidate_idx"] = 0
    if "optimized_run_idx" not in out.columns:
        out["optimized_run_idx"] = out.get("trial_idx", np.arange(out.shape[0]))
    if "source_root_rank" not in out.columns:
        out["source_root_rank"] = 0
    if "source_optimized_run_idx" not in out.columns:
        out["source_optimized_run_idx"] = out["optimized_run_idx"]
    key_cols = ["source_root_rank", "source_optimized_run_idx", "candidate_kind_canon", "candidate_idx"]
    for col in key_cols:
        if col not in out.columns:
            return out
    out["_dedupe_key"] = [tuple(row[col] for col in key_cols) for _, row in out.iterrows()]
    out = out.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    return out.reset_index(drop=True)


def _reindex_optimized_groups(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    if "source_optimized_run_idx" not in out.columns:
        out["source_optimized_run_idx"] = out.get("optimized_run_idx", np.arange(out.shape[0]))
    keys = (
        out[["source_root_rank", "source_optimized_run_idx"]]
        .drop_duplicates()
        .sort_values(["source_root_rank", "source_optimized_run_idx"], na_position="last")
        .reset_index(drop=True)
    )
    group_map = {
        (row["source_root_rank"], row["source_optimized_run_idx"]): int(i)
        for i, row in keys.iterrows()
    }
    out["suite_optimized_run_idx"] = [
        group_map[(row["source_root_rank"], row["source_optimized_run_idx"])]
        for _, row in out.iterrows()
    ]
    out["optimized_run_idx"] = out["suite_optimized_run_idx"]
    return out.reset_index(drop=True)


def _load_trial_rows_many(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for rank, root in enumerate(roots):
        try:
            frame = _load_trial_rows(root)
        except Exception as exc:
            errors.append(f"{root}: {exc}")
            continue
        frame["source_root_rank"] = int(rank)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No usable paper-check roots found. Errors: " + "; ".join(errors))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["source_optimized_run_idx"] = merged["optimized_run_idx"]
    rows_before = int(merged.shape[0])
    source_groups_before = int(merged[["source_root_rank", "source_optimized_run_idx"]].drop_duplicates().shape[0])
    merged = merged.sort_values(["source_root_rank", "optimized_run_idx", "candidate_kind_canon", "candidate_idx"], na_position="last")
    merged = _dedupe_trial_rows(merged)
    rows_after_dedupe = int(merged.shape[0])
    merged = _reindex_optimized_groups(merged)
    groups_after = int(merged["optimized_run_idx"].nunique()) if "optimized_run_idx" in merged.columns else 0
    log_event(
        "merged paper-check roots "
        f"rows_before={rows_before} source_groups_before={source_groups_before} "
        f"rows_after_root_local_dedupe={rows_after_dedupe} suite_groups={groups_after}",
        component="posthoc",
    )
    merged["trial_uid"] = [
        f"root{int(row['source_root_rank']):02d}_{row.get('source_root_name', _source_root_label(Path(str(row['source_root']))))}_trial_{int(row['trial_idx']):05d}"
        if "trial_idx" in row and not pd.isna(row["trial_idx"])
        else f"root{int(row['source_root_rank']):02d}_{row.get('source_root_name', _source_root_label(Path(str(row['source_root']))))}_{idx:05d}"
        for idx, row in merged.iterrows()
    ]
    return merged.reset_index(drop=True)


def _canonicalize_kind(kind: Any, label: Any = None) -> str:
    text = f"{'' if pd.isna(kind) else kind} {'' if label is None or pd.isna(label) else label}".lower()
    if "optimized" in text or "best" in text or "opt" in text:
        return "optimized"
    if "random" in text or "rand" in text:
        return "random"
    if "init" in text or "initial" in text:
        return "init"
    return "other"


def _resolve_artifact(row: pd.Series | dict[str, Any], field: str) -> Path | None:
    raw = dict(row).get(field)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = Path(str(dict(row).get("frustration_root", ".")))
    return root / path


def _primary_lagrangian_xy_key(data: np.lib.npyio.NpzFile, preferred: Any = None) -> str:
    if preferred is not None and str(preferred).strip():
        key = str(preferred)
        if key not in data.files:
            raise KeyError(f"missing configured trajectory key {key!r}; available keys={list(data.files)}")
        return key
    for key in ("xy_trajectory", "xy", "xy_control_a"):
        if key in data.files:
            return key
    raise KeyError(f"missing primary trajectory key; expected one of xy_trajectory, xy, xy_control_a; available keys={list(data.files)}")


def _cfg_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return as_list(value)


def _infer_lagrangian_timing(data: np.lib.npyio.NpzFile, xy: np.ndarray) -> tuple[int, int]:
    scalar_sample_every = int(np.asarray(data["sample_every_steps"]).item()) if "sample_every_steps" in data.files else None
    inferred_sample_every: int | None = None
    sample_steps: np.ndarray | None = None
    for key in ("xy_late_sample_steps", "sample_offsets_steps"):
        if key not in data.files:
            continue
        arr = np.asarray(data[key], dtype=np.int64).reshape(-1)
        if arr.size != int(xy.shape[0]):
            continue
        diffs = np.diff(arr)
        positive = diffs[diffs > 0]
        if positive.size:
            inferred_sample_every = int(round(float(np.median(positive))))
            sample_steps = arr
            break

    sample_every = int(inferred_sample_every or scalar_sample_every or 1)
    if "trajectory_window_steps" in data.files:
        rollout_steps = int(np.asarray(data["trajectory_window_steps"]).item())
    elif sample_steps is not None and sample_steps.size:
        rollout_steps = int(sample_steps[-1] - sample_steps[0] + sample_every)
    else:
        rollout_steps = int(xy.shape[0] * sample_every)
    return sample_every, rollout_steps


def _safe_metric_timing(metric_cfg_raw: Any, *, rollout_steps: int, sample_every: int) -> dict[str, Any]:
    sample_every = max(1, int(sample_every))
    rollout_steps = max(sample_every, int(rollout_steps))
    time_sampling = max(1, rollout_steps // sample_every)
    m_min = max(1, int(cfg_get(metric_cfg_raw, "metric_m_min", 4)))

    win_frames_raw = cfg_get(metric_cfg_raw, "metric_window_size_frames", None)
    if win_frames_raw is not None:
        win_frames = int(win_frames_raw)
    else:
        win_steps_raw = int(cfg_get(metric_cfg_raw, "metric_window_size_steps", 20000))
        win_frames = int(max(1, round(float(win_steps_raw) / float(sample_every))))
    win_frames = max(1, min(int(win_frames), int(time_sampling)))
    win_steps = int(win_frames * sample_every)

    step_frames_raw = cfg_get(metric_cfg_raw, "metric_window_step_frames", None)
    if step_frames_raw is not None:
        step_frames = int(step_frames_raw)
    else:
        step_steps_raw = int(cfg_get(metric_cfg_raw, "metric_window_step_steps", 5000))
        step_frames = int(max(1, round(float(step_steps_raw) / float(sample_every))))
    step_frames = max(1, min(int(step_frames), int(win_frames)))
    step_steps = int(step_frames * sample_every)

    max_tau_frames = max(1, int(win_frames) - int(m_min))
    tau_frames_raw = cfg_get(metric_cfg_raw, "metric_tau_frames", None)
    if tau_frames_raw is not None:
        tau_frames = int(tau_frames_raw)
    else:
        tau_steps_raw = int(cfg_get(metric_cfg_raw, "metric_tau_steps", 3000))
        tau_frames = int(max(1, round(float(tau_steps_raw) / float(sample_every))))
    tau_frames = max(1, min(int(tau_frames), int(max_tau_frames)))
    tau_steps = int(tau_frames * sample_every)

    grid_frames_raw = cfg_get(metric_cfg_raw, "metric_tau_grid_frames", None)
    if grid_frames_raw is not None:
        grid_frames = [int(x) for x in (_cfg_list(grid_frames_raw) or [])]
    else:
        grid_steps_raw = cfg_get(metric_cfg_raw, "metric_tau_grid_steps", None)
        grid_frames = [
            int(max(1, round(float(x) / float(sample_every))))
            for x in (_cfg_list(grid_steps_raw) or [])
        ]
    grid_frames = sorted({int(x) for x in grid_frames if 0 < int(x) <= int(max_tau_frames)})
    if not grid_frames:
        grid_frames = [tau_frames]
    grid_steps = [int(x * sample_every) for x in grid_frames]

    range_end_raw = cfg_get(metric_cfg_raw, "metric_range_end_steps", None)
    range_end = None if range_end_raw is None else min(int(range_end_raw), int(rollout_steps))

    return {
        "metric_window_size_steps": win_steps,
        "metric_window_step_steps": step_steps,
        "metric_tau_steps": tau_steps,
        "metric_tau_grid_steps": grid_steps,
        "metric_window_size_frames": None,
        "metric_window_step_frames": None,
        "metric_tau_frames": None,
        "metric_tau_grid_frames": None,
        "metric_range_end_steps": range_end,
    }


def _metric_config_from_lagrangian(path: Path, metric_cfg_raw: Any) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as data:
            xy_key = _primary_lagrangian_xy_key(data)
            xy = np.asarray(data[xy_key], dtype=np.float32)
            sample_every, rollout_steps = _infer_lagrangian_timing(data, xy)
    except Exception as exc:
        raise ValueError(f"invalid lagrangian npz used for metric config path={path} {_file_probe(path)} error={type(exc).__name__}: {exc}") from exc
    return _metric_config_from_timing(metric_cfg_raw, rollout_steps=rollout_steps, sample_every=sample_every)


def _metric_config_from_timing(metric_cfg_raw: Any, *, rollout_steps: int, sample_every: int) -> dict[str, Any]:
    timing = _safe_metric_timing(metric_cfg_raw, rollout_steps=rollout_steps, sample_every=sample_every)
    args = SimpleNamespace(
        rollout_steps=rollout_steps,
        sample_every_steps=sample_every,
        time_sampling=None,
        metric_window_size_steps=timing["metric_window_size_steps"],
        metric_window_step_steps=timing["metric_window_step_steps"],
        metric_tau_mode=str(cfg_get(metric_cfg_raw, "metric_tau_mode", "max_grid")),
        metric_tau_steps=timing["metric_tau_steps"],
        metric_tau_grid_steps=timing["metric_tau_grid_steps"],
        metric_window_size_frames=timing["metric_window_size_frames"],
        metric_window_step_frames=timing["metric_window_step_frames"],
        metric_tau_frames=timing["metric_tau_frames"],
        metric_tau_grid_frames=timing["metric_tau_grid_frames"],
        metric_range_start_steps=int(cfg_get(metric_cfg_raw, "metric_range_start_steps", 0)),
        metric_range_end_steps=timing["metric_range_end_steps"],
        metric_m_samples=int(cfg_get(metric_cfg_raw, "metric_m_samples", 48)),
        metric_m_min=int(cfg_get(metric_cfg_raw, "metric_m_min", 4)),
        metric_n_proj=int(cfg_get(metric_cfg_raw, "metric_n_proj", 16)),
        metric_null_reps=int(cfg_get(metric_cfg_raw, "metric_null_reps", 6)),
        metric_particle_samples=int(cfg_get(metric_cfg_raw, "metric_particle_samples", 64)),
        metric_dirs_seed=int(cfg_get(metric_cfg_raw, "metric_dirs_seed", 123)),
        metric_periodic=bool(cfg_get(metric_cfg_raw, "metric_periodic", False)),
        metric_domain_y=float(cfg_get(metric_cfg_raw, "metric_domain_y", 0.0)),
        metric_domain_x=float(cfg_get(metric_cfg_raw, "metric_domain_x", 0.0)),
        metric_preprocess_mode=str(cfg_get(metric_cfg_raw, "metric_preprocess_mode", "clip")),
        metric_scales=cfg_get(metric_cfg_raw, "metric_scales", None),
        metric_scale_weights=cfg_get(metric_cfg_raw, "metric_scale_weights", None),
        metric_delta_h_floor=float(cfg_get(metric_cfg_raw, "metric_delta_h_floor", 0.0)),
        metric_msc_floor=cfg_get(metric_cfg_raw, "metric_msc_floor", 0.01),
        metric_msc_term=str(cfg_get(metric_cfg_raw, "metric_msc_term", "floor_reconstruction_error")),
        metric_msc_normalize_by_weight_sum=bool(cfg_get(metric_cfg_raw, "metric_msc_normalize_by_weight_sum", True)),
        metric_alpha=float(cfg_get(metric_cfg_raw, "metric_alpha", 0.0)),
        metric_beta=float(cfg_get(metric_cfg_raw, "metric_beta", 1.0)),
        metric_eps=float(cfg_get(metric_cfg_raw, "metric_eps", 1e-12)),
    )
    return resolve_metric_config(args)


def _score_maps(metric_eval, metric_seed: Any, xy: np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(metric_seed, (int, np.integer)):
        key = jax.random.PRNGKey(int(metric_seed))
    else:
        key = jnp.asarray(metric_seed, dtype=jnp.uint32)
        if tuple(key.shape) != (2,):
            raise ValueError(f"metric_seed key must have shape (2,), got {tuple(key.shape)}.")
    _loss, info = metric_eval(key, jnp.asarray(xy, dtype=jnp.float32))
    return {key: np.asarray(jax.device_get(value)) for key, value in info.items()}


def _normalize_metric_seed_protocol(value: Any) -> str:
    text = str(value if value is not None else "posthoc_index").strip().lower().replace("-", "_")
    if text in {"", "posthoc_index", "posthoc", "legacy", "selection_idx"}:
        return "posthoc_index"
    if text in {"optimization_metric", "optimization", "main_opt_msc", "opt_metric"}:
        return "optimization_metric"
    raise ValueError(
        f"Unknown C1 metric_seed_protocol={value!r}. Use 'posthoc_index' or 'optimization_metric'."
    )


def _c1_metric_seed(c1_cfg: Any, item: dict[str, Any], idx: int) -> tuple[Any, str]:
    protocol = _normalize_metric_seed_protocol(cfg_get(c1_cfg, "metric_seed_protocol", "posthoc_index"))
    if protocol == "optimization_metric":
        run_seed = item.get("run_seed", None)
        if run_seed is None:
            raise ValueError(
                f"C1 metric_seed_protocol='optimization_metric' requires run_seed in manifest item {item.get('traj_id')!r}."
            )
        log_clip_evolution = bool(
            cfg_get(
                c1_cfg,
                "metric_log_clip_evolution",
                cfg_get(c1_cfg, "log_clip_evolution", False),
            )
        )
        if log_clip_evolution:
            _rng_roll, rng_metric, _rng_clip = jax.random.split(jax.random.PRNGKey(int(run_seed)), 3)
            split_label = "split3"
        else:
            _rng_roll, rng_metric = jax.random.split(jax.random.PRNGKey(int(run_seed)), 2)
            split_label = "split2"
        key_np = np.asarray(rng_metric, dtype=np.uint32)
        return rng_metric, f"optimization_metric:{split_label}:{int(key_np[0])},{int(key_np[1])}"
    metric_seed = 10_000_000 + int(item.get("selection_idx", idx - 1))
    return int(metric_seed), f"posthoc_index:{metric_seed}"


def _checkpoint_train_tau_steps(checkpoint_dir_raw: Any) -> int | None:
    if checkpoint_dir_raw is None or str(checkpoint_dir_raw) == "":
        return None
    path = Path(str(checkpoint_dir_raw))
    if not path.is_absolute():
        path = REPO_ROOT / path

    for tau_path in (path / "best_tau.json", path / "selected_candidate.json"):
        if not tau_path.exists():
            continue
        try:
            payload = json.loads(tau_path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        tau_payload = payload.get("tau", payload)
        if not isinstance(tau_payload, dict):
            tau_payload = payload
        value = tau_payload.get("tau_steps")
        if value is None:
            continue
        try:
            out = int(value)
        except Exception:
            continue
        if out > 0:
            return out
    return None


def _item_train_tau_steps(item: dict[str, Any]) -> int | None:
    for key in ("optimizer_native_tau_steps", "train_tau_steps"):
        value = item.get(key, None)
        if value in (None, ""):
            continue
        try:
            out = int(value)
        except Exception:
            continue
        if out > 0:
            return out
    return _checkpoint_train_tau_steps(item.get("source_checkpoint_dir", None))


def _float32_ulp_distance(lhs: float, rhs: float) -> int | None:
    if not (np.isfinite(lhs) and np.isfinite(rhs)):
        return None

    def ordered(value: float) -> int:
        bits = int(np.asarray(np.float32(value)).view(np.uint32).item())
        if bits & 0x80000000:
            return 0x80000000 - (bits & 0x7FFFFFFF)
        return 0x80000000 + bits

    return abs(ordered(lhs) - ordered(rhs))


def _select_score_tau_idx(
    *,
    c1_cfg: Any,
    metric_cfg_raw: Any,
    sel_score_by_tau: np.ndarray,
    train_tau_idx: int,
    context: str,
) -> int:
    source = str(
        cfg_get(
            c1_cfg,
            "score_tau_source",
            cfg_get(metric_cfg_raw, "score_tau_source", "selection"),
        )
    ).strip().lower()
    if source in {"selection", "selection_holdout", "max_selection", "max_grid"}:
        return int(np.nanargmax(sel_score_by_tau))
    if source in {"train", "train_tau", "optimizer_train_tau", "optimizer_native_tau"}:
        if int(train_tau_idx) < 0:
            raise ValueError(
                f"{context}: score_tau_source={source!r} requires train tau metadata, "
                "but no matching train_tau_idx was found."
            )
        return int(train_tau_idx)
    raise ValueError(
        f"{context}: unsupported score_tau_source={source!r}; use 'selection' or 'train_tau'."
    )


def _file_probe(path: Path) -> str:
    try:
        size = path.stat().st_size
    except Exception as exc:
        return f"stat_error={type(exc).__name__}: {exc}"
    try:
        with path.open("rb") as f:
            head = f.read(16)
        ascii_head = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
        return f"size_bytes={size} first16_hex={head.hex()} first16_ascii={ascii_head!r}"
    except Exception as exc:
        return f"size_bytes={size} read_error={type(exc).__name__}: {exc}"


def _load_npz_arrays(path: Path, keys: list[str], *, context: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in keys if key not in data.files]
            if missing:
                raise KeyError(f"missing keys {missing}; available keys={list(data.files)}")
            return {key: np.asarray(data[key]) for key in keys}
    except Exception as exc:
        raise ValueError(f"{context}: invalid npz artifact path={path} {_file_probe(path)} error={type(exc).__name__}: {exc}") from exc


def _c1_source_mode(ds_cfg: Any) -> str:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
    return str(cfg_get(c1_cfg, "source", "frustration_lagrangian")).strip().lower()


def _c1_uses_apf(ds_cfg: Any) -> bool:
    return _c1_source_mode(ds_cfg) in {
        "apf",
        "apf_metrics",
        "arun",
        "arun_lagrangian_apf",
        "apf_lagrangian_split",
        "apf_temporal_holdout",
    }


def _c1_uses_apf_lagrangian_split(ds_cfg: Any) -> bool:
    return _c1_source_mode(ds_cfg) in {"apf_lagrangian_split", "apf_temporal_holdout"}


def _c1_uses_lagrangian_root(ds_cfg: Any) -> bool:
    return _c1_source_mode(ds_cfg) in {"lagrangian_root", "c1_lagrangian", "trajectory_root", "trajectories"}


def _c1_lagrangian_roots(ds_cfg: Any) -> list[Path]:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
    roots_raw = cfg_get(c1_cfg, "lagrangian_roots", None)
    if roots_raw is None:
        root_raw = cfg_get(c1_cfg, "lagrangian_root", None) or cfg_get(c1_cfg, "trajectory_root", None)
        roots_raw = [] if root_raw is None else [root_raw]
    roots = []
    for raw in as_list(roots_raw):
        path = resolve_path(raw)
        if path is not None and Path(path).exists():
            roots.append(Path(path))
    return roots


def _manifest_path(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _iter_apf_metric_items(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            traj_id = str(row.get("traj_id", f"flow_opt_{idx:03d}"))
            traj_dir = _manifest_path(root, row.get("traj_dir"), default=root / traj_id)
            apf_dir = _manifest_path(root, row.get("apf_dir"), default=traj_dir / "apf_logs")
            metrics_path = _manifest_path(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            candidate_kind = _canonicalize_kind(row.get("candidate_kind", row.get("candidate_label", "optimized")), row.get("candidate_label", None))
            items.append(
                {
                    "traj_id": traj_id,
                    "selection_idx": int(row.get("selection_idx", idx)),
                    "optimized_run_idx": int(row.get("suite_run_idx", row.get("selection_idx", idx))),
                    "source_optimized_run_idx": int(row.get("source_run_idx", row.get("suite_run_idx", idx))),
                    "source_root_rank": int(row.get("source_root_rank", -1)),
                    "source_root": str(row.get("source_root", "")),
                    "source_root_name": str(row.get("source_root_name", "")),
                    "source_checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                    "base_run_seed": int(row.get("base_run_seed", row.get("run_seed", -1))),
                    "run_seed": int(row.get("run_seed", -1)),
                    "rollout_seed_idx": int(row.get("rollout_seed_idx", 0)),
                    "rollout_seed_count": int(row.get("rollout_seed_count", 1)),
                    "candidate_kind": candidate_kind,
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "candidate_label": str(row.get("candidate_label", candidate_kind)),
                    "traj_dir": traj_dir,
                    "apf_dir": apf_dir,
                    "params_path": _manifest_path(root, row.get("params_path"), default=traj_dir / "params.npy"),
                    "metrics_path": metrics_path,
                    **{
                        key: row[key]
                        for key in (
                            "optimizer_native_source_pop_traj",
                            "optimizer_native_source_run_dir",
                            "optimizer_native_iter",
                            "optimizer_native_pop_idx",
                            "optimizer_native_tau_idx",
                            "optimizer_native_tau_steps",
                            "optimizer_native_tau_frames",
                            "optimizer_native_tau_selector_raw",
                            "optimizer_native_score_mspd",
                            "optimizer_native_score_by_seed_mspd",
                            "optimizer_native_legacy_sigma_collision",
                            "optimizer_native_params_source",
                            "optimizer_native_use_row_params",
                            "theta_source_checkpoint",
                        )
                        if key in row
                    },
                }
            )
    if items:
        return items
    for idx, traj_dir in enumerate(sorted(root.glob("flow_opt_*"))):
        if not traj_dir.is_dir():
            continue
        metrics_path = traj_dir / "metrics.npz"
        items.append(
            {
                "traj_id": traj_dir.name,
                "selection_idx": idx,
                "optimized_run_idx": idx,
                "source_optimized_run_idx": idx,
                "source_root_rank": -1,
                "source_root": "",
                "source_root_name": "",
                "base_run_seed": -1,
                "run_seed": -1,
                "rollout_seed_idx": 0,
                "rollout_seed_count": 1,
                "candidate_kind": _canonicalize_kind(traj_dir.name, traj_dir.name),
                "candidate_idx": 0,
                "candidate_label": traj_dir.name,
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": metrics_path,
            }
        )
    return items


def _apf_sample_every(steps: np.ndarray) -> int:
    diffs = np.diff(np.asarray(steps, dtype=np.int64).reshape(-1))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError("Cannot infer APF lagrangian sample interval from fewer than two increasing steps.")
    return int(round(float(np.median(positive))))


def _slice_apf_lagrangian(
    steps: np.ndarray,
    lagrangian_xy: np.ndarray,
    *,
    start_steps: int,
    end_steps: int,
    sample_every: int,
    context: str,
) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.int64).reshape(-1)
    xy = np.asarray(lagrangian_xy, dtype=np.float32)
    if steps.shape[0] != xy.shape[0]:
        raise ValueError(f"{context}: steps/lagrangian length mismatch: steps={steps.shape[0]}, xy={xy.shape[0]}.")
    start_steps = int(start_steps)
    end_steps = int(end_steps)
    if end_steps <= start_steps:
        raise ValueError(f"{context}: invalid APF holdout range {start_steps}..{end_steps}.")
    if (end_steps - start_steps) % int(sample_every) != 0:
        raise ValueError(
            f"{context}: APF holdout range {start_steps}..{end_steps} is not divisible by sample_every={sample_every}."
        )
    mask = (steps > start_steps) & (steps <= end_steps)
    out = xy[mask]
    expected = int((end_steps - start_steps) // int(sample_every))
    if int(out.shape[0]) != expected:
        available_min = int(np.nanmin(steps)) if steps.size else None
        available_max = int(np.nanmax(steps)) if steps.size else None
        raise ValueError(
            f"{context}: APF holdout range {start_steps}..{end_steps} has {out.shape[0]} samples, expected {expected}; "
            f"available step span is {available_min}..{available_max}."
        )
    return out


def _path_from_repo(raw: Any) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _flat_optimization_config(path: Path, *, legacy_sigma_collision: bool = False) -> Any:
    cfg = OmegaConf.load(path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    substrate_cfg = cfg.get("substrate", {})
    if legacy_sigma_collision:
        if flat.get("flow_sigma", None) is not None:
            flat.flow_sigma = None
    elif str(substrate_cfg.get("substrate", "")).strip().lower() == "lenia_flow":
        flow_sigma = substrate_cfg.get("flow_sigma", substrate_cfg.get("sigma", None))
        if flow_sigma is not None:
            flat.flow_sigma = flow_sigma
    return flat


def _metric_roll_key_for_optimizer(args: Any, eval_key: Any) -> Any:
    if bool(cfg_get(args, "log_clip_evolution", True)):
        rng_roll, _rng_metric, _rng_clip = jax.random.split(eval_key, 3)
    else:
        rng_roll, _rng_metric = jax.random.split(eval_key, 2)
    return rng_roll


def _seed_idx_for_run_seed(seed_keys: np.ndarray, run_seed: int) -> int:
    expected = np.asarray(jax.random.PRNGKey(int(run_seed)), dtype=np.uint32).reshape(2)
    seed_keys = np.asarray(seed_keys, dtype=np.uint32)
    matches = np.flatnonzero(np.all(seed_keys == expected[None, :], axis=1))
    if matches.size != 1:
        keys = [[int(x) for x in np.asarray(key).reshape(-1)] for key in seed_keys]
        raise ValueError(f"run_seed={run_seed} does not match exactly one optimizer seed key; seed_keys={keys}")
    return int(matches[0])


def _optimizer_native_metadata(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("optimizer_native_source_pop_traj", None) not in (None, ""):
        pop_path = _path_from_repo(item.get("optimizer_native_source_pop_traj"))
        source_run_dir = _path_from_repo(item.get("optimizer_native_source_run_dir", pop_path.parent))
        selected = {
            "iter": int(item.get("optimizer_native_iter", -1)),
            "pop_idx": int(item.get("optimizer_native_pop_idx", -1)),
        }
    else:
        checkpoint_dir_raw = item.get("source_checkpoint_dir", "")
        if not checkpoint_dir_raw:
            raise ValueError(f"{item.get('traj_id')}: optimizer-native replay requires source_checkpoint_dir.")
        checkpoint_dir = _path_from_repo(checkpoint_dir_raw)
        selected_path = checkpoint_dir / "selected_candidate.json"
        if not selected_path.exists():
            raise FileNotFoundError(f"{item.get('traj_id')}: missing selected_candidate.json at {selected_path}")
        selected = json.loads(selected_path.read_text())
        pop_path = _path_from_repo(selected.get("source_pop_traj", ""))
        source_run_dir = _path_from_repo(selected.get("source_run_dir", pop_path.parent))
    if not pop_path.exists():
        raise FileNotFoundError(f"{item.get('traj_id')}: source_pop_traj not found: {pop_path}")
    opt_config_path = source_run_dir / "optimization_config.yaml"
    if not opt_config_path.exists():
        raise FileNotFoundError(f"{item.get('traj_id')}: optimization_config.yaml not found: {opt_config_path}")

    with pop_path.open("rb") as f:
        pop = pickle.load(f)
    params = np.asarray(pop["params"], dtype=np.float32)
    seed_keys = np.asarray(pop["seed_keys"], dtype=np.uint32)
    i_iter = int(selected.get("iter", -1))
    pop_idx = int(selected.get("pop_idx", -1))
    if params.ndim != 3 or seed_keys.ndim != 3:
        raise ValueError(f"{item.get('traj_id')}: invalid pop_traj shapes params={params.shape} seed_keys={seed_keys.shape}")
    if i_iter < 0 or i_iter >= params.shape[0] or pop_idx < 0 or pop_idx >= params.shape[1]:
        raise ValueError(f"{item.get('traj_id')}: selected iter/pop_idx out of range: iter={i_iter} pop_idx={pop_idx}")
    init_params_batch = np.asarray(params[i_iter], dtype=np.float32)
    params_batch = np.asarray(init_params_batch, dtype=np.float32)
    use_row_params = str(item.get("optimizer_native_use_row_params", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    } or item.get("optimizer_native_use_row_params") is True
    if use_row_params:
        params_path = _path_from_repo(item.get("params_path", ""))
        if not params_path.exists():
            raise FileNotFoundError(f"{item.get('traj_id')}: row params override not found: {params_path}")
        row_params = np.asarray(np.load(params_path), dtype=np.float32).reshape(-1)
        expected = np.asarray(params_batch[pop_idx], dtype=np.float32).reshape(-1)
        if row_params.shape != expected.shape:
            raise ValueError(
                f"{item.get('traj_id')}: row params shape {row_params.shape} "
                f"does not match optimizer params shape {expected.shape}."
            )
        params_batch = np.array(params_batch, copy=True)
        params_batch[pop_idx] = row_params
    seed_idx = _seed_idx_for_run_seed(seed_keys[i_iter], int(item["run_seed"]))
    return {
        "optimization_config": opt_config_path,
        "source_pop_traj": pop_path,
        "iter": i_iter,
        "pop_idx": pop_idx,
        "seed_idx": seed_idx,
        "params_batch": params_batch,
        "init_params_batch": init_params_batch,
        "seed_keys": np.asarray(seed_keys[i_iter], dtype=np.uint32),
        "row_params_override": bool(use_row_params),
        "params_source": str(item.get("optimizer_native_params_source", "pop_traj")),
    }


def _make_optimizer_native_selected_xy_fn(
    *,
    opt_config_path: Path,
    rollout_steps: int,
    sample_every_steps: int,
    legacy_sigma_collision: bool = False,
    use_init_params_override: bool = False,
):
    args = _flat_optimization_config(opt_config_path, legacy_sigma_collision=bool(legacy_sigma_collision))
    args.rollout_steps = int(rollout_steps)
    args.sample_every_steps = int(sample_every_steps)
    substrate = _make_substrate(args)
    substrate_param_dims = int(substrate.n_params)

    # Force RT construction before jitted rollout helpers close over it, as in
    # main_opt_msc and the preflight diagnostics.
    probe_params = substrate.default_params(jax.random.PRNGKey(17))
    _ = substrate.init_state(jax.random.PRNGKey(0), probe_params)
    rt = substrate.RT

    lag_n = int(cfg_get(args, "metric_lagrangian_n_particles", 8192))
    lag_init_mode = str(cfg_get(args, "metric_lagrangian_init_mode", "mass"))
    lag_flow_channel = int(cfg_get(args, "metric_lagrangian_flow_channel", -1))
    lag_flow_reduce = str(cfg_get(args, "metric_lagrangian_flow_reduce", "mass_weighted"))
    lag_channel_mode = str(cfg_get(args, "metric_lagrangian_channel_mode", "resample"))
    lag_noise_model = str(cfg_get(args, "metric_lagrangian_noise_model", "rt_box"))
    lag_diffusion_scale = float(cfg_get(args, "metric_lagrangian_diffusion_scale", 1.0))
    n_chunks = int(rollout_steps) // int(sample_every_steps)
    if int(rollout_steps) % int(sample_every_steps) != 0:
        raise ValueError(
            f"optimizer-native replay requires rollout_steps divisible by sample_every_steps; "
            f"got {rollout_steps} and {sample_every_steps}."
        )

    def rollout_one(eval_key, params_full):
        params = params_full[:substrate_param_dims]
        rng_roll = _metric_roll_key_for_optimizer(args, eval_key)
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
        s0 = substrate.init_state(k_state, params)
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

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
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, n_chunks),
        )
        return xy_seq

    def rollout_one_with_opt_init(eval_key, params_full, init_params_full):
        params = params_full[:substrate_param_dims]
        init_params = init_params_full[:substrate_param_dims]
        rng_roll = _metric_roll_key_for_optimizer(args, eval_key)
        k_state, k_pts, k_ch, k_scan = jax.random.split(rng_roll, 4)
        s0 = substrate.init_state(k_state, params)
        init_s0 = substrate.init_state(k_state, init_params)
        s0 = dict(s0)
        for key in ("A", "P", "Food", "t", "F"):
            if key in init_s0:
                s0[key] = init_s0[key]
        pts0 = _init_lagrangian_points_jax(
            s0["A"],
            n_particles=lag_n,
            init_mode=lag_init_mode,
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
            key=k_pts,
        )
        if lag_channel_mode in ("fixed", "resample"):
            ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
        else:
            ch0 = jnp.zeros((lag_n,), dtype=jnp.int32)

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
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, int(sample_every_steps)))
            return state_next, state_next[1]

        (_, _, _), xy_seq = jax.lax.scan(
            chunk_fn,
            (s0, pts0, ch0),
            jax.random.split(k_scan, n_chunks),
        )
        return xy_seq

    if bool(use_init_params_override):
        def selected_xy(seed_keys_in, params_batch_in, init_params_batch_in, pop_idx_in, seed_idx_in):
            # The caller needs one selected candidate/seed, so avoid replaying the full
            # optimizer population and all evaluation seeds before indexing the result.
            params = params_batch_in[pop_idx_in]
            init_params = init_params_batch_in[pop_idx_in]
            key = seed_keys_in[seed_idx_in]
            return rollout_one_with_opt_init(key, params, init_params)
    else:
        def selected_xy(seed_keys_in, params_batch_in, init_params_batch_in, pop_idx_in, seed_idx_in):
            params = params_batch_in[pop_idx_in]
            key = seed_keys_in[seed_idx_in]
            return rollout_one(key, params)

    return jax.jit(selected_xy)


def _optimizer_native_lagrangian_xy(
    item: dict[str, Any],
    *,
    rollout_steps: int,
    sample_every_steps: int,
    fn_cache: dict[tuple[str, int, int, bool], Any],
    legacy_sigma_collision: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta = _optimizer_native_metadata(item)
    legacy_sigma_collision = bool(legacy_sigma_collision)
    row_params_override = bool(meta.get("row_params_override", False))
    cache_key = (
        str(meta["optimization_config"]),
        int(rollout_steps),
        int(sample_every_steps),
        legacy_sigma_collision,
        row_params_override,
    )
    if cache_key not in fn_cache:
        fn_cache[cache_key] = _make_optimizer_native_selected_xy_fn(
            opt_config_path=Path(meta["optimization_config"]),
            rollout_steps=int(rollout_steps),
            sample_every_steps=int(sample_every_steps),
            legacy_sigma_collision=legacy_sigma_collision,
            use_init_params_override=row_params_override,
        )
    fn = fn_cache[cache_key]
    flat = _flat_optimization_config(
        Path(meta["optimization_config"]),
        legacy_sigma_collision=legacy_sigma_collision,
    )
    meta = dict(
        meta,
        legacy_sigma_collision=legacy_sigma_collision,
        row_params_override=bool(meta.get("row_params_override", False)),
        params_source=str(meta.get("params_source", "pop_traj")),
        resolved_args_sigma=None if cfg_get(flat, "sigma", None) is None else float(cfg_get(flat, "sigma")),
        resolved_args_flow_sigma=(
            None if cfg_get(flat, "flow_sigma", None) is None else float(cfg_get(flat, "flow_sigma"))
        ),
    )
    xy = fn(
        jnp.asarray(meta["seed_keys"], dtype=jnp.uint32),
        jnp.asarray(meta["params_batch"], dtype=jnp.float32),
        jnp.asarray(meta.get("init_params_batch", meta["params_batch"]), dtype=jnp.float32),
        jnp.asarray(int(meta["pop_idx"]), dtype=jnp.int32),
        jnp.asarray(int(meta["seed_idx"]), dtype=jnp.int32),
    )
    return np.asarray(jax.device_get(xy), dtype=np.float32), meta


def _item_optimizer_native_legacy_sigma_collision(item: dict[str, Any], default: bool) -> bool:
    raw = item.get("optimizer_native_legacy_sigma_collision", None)
    if raw in (None, ""):
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _sample_every_for_c1_apf_split(root: Path, c1_cfg: Any, items: list[dict[str, Any]]) -> int:
    for key in ("sample_every_steps", "snapshot_interval"):
        raw = cfg_get(c1_cfg, key, None)
        if raw is not None:
            return int(raw)
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        sig = payload.get("expected_rollout_signature", {})
        for key in ("sample_every_steps", "snapshot_interval"):
            raw = sig.get(key, None)
            if raw is not None:
                return int(raw)
    for item in items:
        apf_dir = Path(item["apf_dir"])
        if apf_dir.exists():
            try:
                steps, _lag = _load_lagrangian_series(apf_dir)
                return _apf_sample_every(steps)
            except Exception:
                continue
    raise ValueError(f"Could not infer C1 APF sample interval from config or APF root: {root}")


def _absolute_window_steps(info: dict[str, np.ndarray], *, range_start_steps: int, window_size_steps: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(info["window_start_steps"], dtype=np.int64).reshape(-1) + int(range_start_steps)
    ends = starts + int(window_size_steps)
    return starts, ends


def _interleaved_window_masks(n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(n_windows), dtype=np.int64)
    selection = (idx % 2) == 0
    evaluation = ~selection
    if not np.any(selection) or not np.any(evaluation):
        raise ValueError(f"Need at least one selection and one eval window for interleaved C1 split, got W={n_windows}.")
    return selection, evaluation


def _preprocess_delta_h_for_score(metric_cfg: dict[str, Any], h: np.ndarray) -> np.ndarray:
    mode = str(metric_cfg.get("preprocess_mode", "clip"))
    out = np.asarray(h, dtype=np.float64)
    if mode == "clip":
        out = np.maximum(out, 0.0)
    elif mode == "shift":
        out = out - np.nanmin(out)
    elif mode != "none":
        raise ValueError(f"Unknown metric preprocess mode: {mode!r}")
    floor = float(metric_cfg.get("delta_h_floor", 0.0) or 0.0)
    if floor > 0.0:
        out = np.where(out >= floor, out, 0.0)
    return out


def _score_processed_h(metric_cfg: dict[str, Any], h_pos: np.ndarray) -> tuple[float, float, float]:
    h_pos = np.asarray(h_pos, dtype=np.float64).reshape(-1)
    if h_pos.size == 0:
        raise ValueError("Cannot score empty Delta-H window subset.")
    amp = float(np.nanmean(h_pos))
    eps = float(metric_cfg.get("eps", 1e-12))
    msc_floor = float(metric_cfg.get("msc_floor", metric_cfg.get("delta_h_floor", 0.0)) or 0.0)
    msc_term = str(metric_cfg.get("msc_term", "overlap"))
    valid_pairs = []
    for r_raw, w_raw in metric_cfg.get("scale_pairs", []):
        r = int(r_raw)
        if r <= 0 or h_pos.size // (2 * r) < 1:
            continue
        valid_pairs.append((r, float(w_raw)))
    if str(metric_cfg.get("scale_normalization", "none")) == "sum_weight_r":
        weight_denom = float(sum(w for _r, w in valid_pairs)) + eps
    else:
        weight_denom = 1.0
    msc = 0.0
    for r, wr in valid_pairs:
        U_r = h_pos.size // r
        U_2r = h_pos.size // (2 * r)
        if U_r < 1 or U_2r < 1:
            continue
        g_r = np.mean(h_pos[: U_r * r].reshape(U_r, r), axis=1)
        g_2r = np.mean(h_pos[: U_2r * (2 * r)].reshape(U_2r, 2 * r), axis=1)
        U_cmp = min(g_r.shape[0], 2 * g_2r.shape[0])
        if U_cmp < 1:
            continue
        g_r_cmp = g_r[:U_cmp]
        up = np.repeat(g_2r, 2)[:U_cmp]
        if msc_term == "floor_reconstruction_error":
            numerator = float(np.nanmean((g_r_cmp - up) ** 2))
            denominator = float(np.nanmean(g_r_cmp * g_r_cmp)) + msc_floor * msc_floor + eps
            d_r = numerator / denominator
        else:
            overlap = float(np.nansum(g_r_cmp * up))
            power = float(np.nansum(g_r_cmp * g_r_cmp))
            d_r = 1.0 - overlap / (power + eps) if power > eps else 0.0
        msc += (wr * d_r) / weight_denom
    score = float(metric_cfg.get("alpha", 0.0)) * amp + float(metric_cfg.get("beta", 1.0)) * float(msc)
    return float(score), float(amp), float(msc)


def _score_delta_h_map_subset(
    metric_cfg: dict[str, Any],
    delta_h_map: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full = np.asarray(delta_h_map, dtype=np.float64)
    if full.ndim != 2:
        raise ValueError(f"Expected delta_h_map with shape (n_tau, n_windows), got {full.shape}.")
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != full.shape[1]:
        raise ValueError(f"Window mask length {mask.size} does not match delta_h_map W={full.shape[1]}.")
    subset_raw = full[:, mask]
    scores, amps, mscs = [], [], []
    for row in subset_raw:
        h_pos = _preprocess_delta_h_for_score(metric_cfg, row)
        score, amp, msc = _score_processed_h(metric_cfg, h_pos)
        scores.append(score)
        amps.append(amp)
        mscs.append(msc)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(amps, dtype=np.float64),
        np.asarray(mscs, dtype=np.float64),
        subset_raw,
    )


def _compute_c1_from_apf_lagrangian_split(
    dataset_name: str,
    ds_cfg: Any,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
    metric_cfg_raw = cfg_get(c1_cfg, "metric", {})
    root = resolve_path(cfg_get(c1_cfg, "apf_root", None))
    if root is None or not root.exists():
        raise FileNotFoundError(f"{dataset_name}: C1 apf_root not found: {root}")
    items = _iter_apf_metric_items(root)
    if not items:
        raise FileNotFoundError(f"{dataset_name}: no APF trajectory items found under {root}")

    range_start_raw = cfg_get(c1_cfg, "expected_window_start_steps", None)
    range_end_raw = cfg_get(c1_cfg, "expected_window_end_steps", None)
    if range_start_raw is None or range_end_raw is None:
        raise ValueError(
            f"{dataset_name}: apf_lagrangian_split requires c1.expected_window_start_steps and c1.expected_window_end_steps."
        )
    range_start = int(range_start_raw)
    range_end = int(range_end_raw)
    if range_end <= range_start:
        raise ValueError(f"{dataset_name}: invalid C1 APF range {range_start}..{range_end}.")
    require_random = bool(cfg_get(c1_cfg, "require_random", True))

    maps_dir = ensure_dir(output_dir / "c1_delta_h_maps")
    if force:
        for stale in maps_dir.glob("*_c1_maps.npz"):
            stale.unlink()

    sample_every = _sample_every_for_c1_apf_split(root, c1_cfg, items)
    full_len = range_end - range_start
    metric_cfg = _metric_config_from_timing(metric_cfg_raw, rollout_steps=full_len, sample_every=sample_every)
    metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))
    window_size_steps = int(metric_cfg["window_size_frames"]) * int(metric_cfg["sample_every_steps"])
    optimized_replay_source = str(cfg_get(c1_cfg, "optimized_replay_source", "apf")).strip().lower()
    use_optimizer_native_for_optimized = optimized_replay_source in {
        "optimizer_native",
        "optimization_native",
        "optimizer_nested_jit",
        "nested_jit",
    }
    if use_optimizer_native_for_optimized:
        raise RuntimeError(
            f"{dataset_name}: optimized_replay_source={optimized_replay_source!r} is disabled. "
            "C1 posthoc must score the saved APF trajectories instead of launching a new replay."
        )
    optimized_replay_legacy_sigma_collision = bool(
        cfg_get(c1_cfg, "optimized_replay_legacy_sigma_collision", False)
    )
    optimizer_native_fn_cache: dict[tuple[str, int, int, bool], Any] = {}

    score_rows: list[dict[str, Any]] = []
    log_event(
        f"{dataset_name}: C1 APF interleaved holdout start n_items={len(items)} root={root} "
        f"range={range_start}..{range_end} selection_windows=2k eval_windows=2k+1 "
        f"optimized_replay_source={optimized_replay_source} "
        f"optimized_replay_legacy_sigma_collision={optimized_replay_legacy_sigma_collision}",
        component="posthoc",
    )
    for idx, item in enumerate(items, start=1):
        apf_dir = Path(item["apf_dir"])
        if idx == 1 or idx == len(items) or idx % 5 == 0:
            log_event(
                f"{dataset_name}: C1 APF interleaved scoring {idx}/{len(items)} traj={item['traj_id']} apf={apf_dir}",
                component="posthoc",
            )
        replay_source = "apf_lagrangian"
        optimizer_native_meta: dict[str, Any] = {}
        try:
            has_optimizer_native_manifest = item.get("optimizer_native_source_pop_traj", None) not in (None, "")
            if use_optimizer_native_for_optimized and (
                str(item.get("candidate_kind", "")) == "optimized" or has_optimizer_native_manifest
            ):
                item_legacy_sigma_collision = _item_optimizer_native_legacy_sigma_collision(
                    item,
                    optimized_replay_legacy_sigma_collision,
                )
                native_xy, optimizer_native_meta = _optimizer_native_lagrangian_xy(
                    item,
                    rollout_steps=range_end,
                    sample_every_steps=sample_every,
                    fn_cache=optimizer_native_fn_cache,
                    legacy_sigma_collision=item_legacy_sigma_collision,
                )
                native_steps = np.arange(sample_every, range_end + sample_every, sample_every, dtype=np.int64)
                xy_full = _slice_apf_lagrangian(
                    native_steps,
                    native_xy,
                    start_steps=range_start,
                    end_steps=range_end,
                    sample_every=sample_every,
                    context=f"{dataset_name}: {item['traj_id']} optimizer_native_full_range",
                )
                replay_source = (
                    (
                        "optimizer_native_nested_jit_row_params_legacy_sigma_collision"
                        if item_legacy_sigma_collision
                        else "optimizer_native_nested_jit_row_params"
                    )
                    if optimizer_native_meta.get("row_params_override", False)
                    else (
                        "optimizer_native_nested_jit_legacy_sigma_collision"
                        if item_legacy_sigma_collision
                        else "optimizer_native_nested_jit"
                    )
                )
            else:
                steps, lag = _load_lagrangian_series(apf_dir)
                sample_every_i = _apf_sample_every(steps)
                if sample_every_i != sample_every:
                    raise ValueError(f"sample_every mismatch: got {sample_every_i}, expected {sample_every}")
                xy_full = _slice_apf_lagrangian(
                    steps,
                    lag,
                    start_steps=range_start,
                    end_steps=range_end,
                    sample_every=sample_every,
                    context=f"{dataset_name}: {item['traj_id']} full_range",
                )
        except Exception as exc:
            raise ValueError(
                f"{dataset_name}: invalid C1 replay for traj={item['traj_id']} apf={apf_dir} "
                f"replay_source={replay_source} error={type(exc).__name__}: {exc}"
            ) from exc

        metric_seed, metric_seed_label = _c1_metric_seed(c1_cfg, item, idx)
        info = _score_maps(metric_eval, metric_seed, xy_full)
        full_map = np.asarray(info["delta_h_map"], dtype=np.float64)
        selection_mask, eval_mask = _interleaved_window_masks(full_map.shape[1])
        sel_score_by_tau, sel_amp_by_tau, sel_msc_by_tau, sel_map = _score_delta_h_map_subset(
            metric_cfg,
            full_map,
            selection_mask,
        )
        eval_score_by_tau, eval_amp_by_tau, eval_msc_by_tau, eval_map = _score_delta_h_map_subset(
            metric_cfg,
            full_map,
            eval_mask,
        )
        full_score_by_tau = np.asarray(info.get("score_by_tau", []), dtype=np.float64).reshape(-1)
        full_amp_by_tau = np.asarray(info.get("amp_by_tau", []), dtype=np.float64).reshape(-1)
        full_msc_by_tau = np.asarray(info.get("msc_by_tau", []), dtype=np.float64).reshape(-1)
        tau_steps = np.asarray(info["tau_steps"], dtype=np.int32)
        if full_score_by_tau.shape[0] != tau_steps.shape[0]:
            full_mask = np.ones(full_map.shape[1], dtype=bool)
            full_score_by_tau, full_amp_by_tau, full_msc_by_tau, _full_map = _score_delta_h_map_subset(
                metric_cfg,
                full_map,
                full_mask,
            )
        full_selected_idx = int(np.nanargmax(full_score_by_tau))
        train_tau_steps = _item_train_tau_steps(item)
        train_tau_idx = -1
        if train_tau_steps is not None:
            matches = np.where(tau_steps == int(train_tau_steps))[0]
            if matches.size:
                train_tau_idx = int(matches[0])
        train_tau_score = float(full_score_by_tau[train_tau_idx]) if train_tau_idx >= 0 else float("nan")
        reference_seed_score = float("nan")
        reference_seed_scores_raw = item.get("optimizer_native_score_by_seed_mspd", None)
        if reference_seed_scores_raw not in (None, ""):
            reference_seed_scores = np.asarray(reference_seed_scores_raw, dtype=np.float32).reshape(-1)
            rollout_seed_idx = int(item.get("rollout_seed_idx", 0))
            if 0 <= rollout_seed_idx < reference_seed_scores.size:
                reference_seed_score = float(reference_seed_scores[rollout_seed_idx])
        train_mspd_abs_error = (
            abs(float(np.float32(train_tau_score)) - float(np.float32(reference_seed_score)))
            if np.isfinite(train_tau_score) and np.isfinite(reference_seed_score)
            else float("nan")
        )
        train_mspd_ulp_distance = _float32_ulp_distance(train_tau_score, reference_seed_score)
        train_mspd_exact_match = bool(
            np.isfinite(train_tau_score)
            and np.isfinite(reference_seed_score)
            and np.asarray(np.float32(train_tau_score)).view(np.uint32).item()
            == np.asarray(np.float32(reference_seed_score)).view(np.uint32).item()
        )
        cross_hardware_source_runs = {
            int(value)
            for value in as_list(
                cfg_get(c1_cfg, "optimizer_reference_cross_hardware_source_run_indices", [])
            )
        }
        cross_hardware_max_ulps = int(
            cfg_get(c1_cfg, "optimizer_reference_cross_hardware_max_ulps", 0)
        )
        source_run_idx = int(item.get("source_optimized_run_idx", -1))
        cross_hardware_exception_used = bool(
            not train_mspd_exact_match
            and source_run_idx in cross_hardware_source_runs
            and train_mspd_ulp_distance is not None
            and train_mspd_ulp_distance <= cross_hardware_max_ulps
        )
        train_mspd_validation_passed = bool(
            train_mspd_exact_match or cross_hardware_exception_used
        )
        if train_mspd_exact_match:
            train_mspd_validation = "bit_exact"
        elif cross_hardware_exception_used:
            train_mspd_validation = "known_cross_hardware_ulp"
        elif train_mspd_ulp_distance is None:
            train_mspd_validation = "not_available"
        else:
            train_mspd_validation = "failed"
        if (
            bool(cfg_get(c1_cfg, "require_exact_optimized_train_mspd", False))
            and str(item.get("candidate_kind", "")) == "optimized"
            and not train_mspd_validation_passed
        ):
            raise RuntimeError(
                f"{dataset_name}: exact optimizer MSPD replay failed for {item['traj_id']}: "
                f"APF full_score_train_tau={train_tau_score:.17g}, "
                f"optimizer score_by_seed={reference_seed_score:.17g}, "
                f"float32_abs_error={train_mspd_abs_error:.17g}, "
                f"float32_ulp_distance={train_mspd_ulp_distance}, "
                f"source_run_idx={source_run_idx}, "
                f"allowed_cross_hardware_runs={sorted(cross_hardware_source_runs)}, "
                f"cross_hardware_max_ulps={cross_hardware_max_ulps}."
            )
        selected_idx = _select_score_tau_idx(
            c1_cfg=c1_cfg,
            metric_cfg_raw=metric_cfg_raw,
            sel_score_by_tau=sel_score_by_tau,
            train_tau_idx=train_tau_idx,
            context=f"{dataset_name}: {item['traj_id']}",
        )
        eval_values = eval_map[selected_idx]
        full_window_start, full_window_end = _absolute_window_steps(
            info,
            range_start_steps=range_start,
            window_size_steps=window_size_steps,
        )
        sel_window_start = full_window_start[selection_mask]
        sel_window_end = full_window_end[selection_mask]
        eval_window_start = full_window_start[eval_mask]
        eval_window_end = full_window_end[eval_mask]

        trial_uid = str(item["traj_id"])
        maps_path = maps_dir / f"{trial_uid}_c1_maps.npz"
        np.savez_compressed(
            maps_path,
            delta_h_selection=sel_map,
            delta_h_eval=eval_map,
            tau_steps=tau_steps,
            window_start_steps=eval_window_start.astype(np.int64),
            window_end_steps=eval_window_end.astype(np.int64),
            selection_window_start_steps=sel_window_start.astype(np.int64),
            selection_window_end_steps=sel_window_end.astype(np.int64),
            eval_window_start_steps=eval_window_start.astype(np.int64),
            eval_window_end_steps=eval_window_end.astype(np.int64),
            selection_score_by_tau=sel_score_by_tau,
            eval_score_by_tau=eval_score_by_tau,
            full_score_by_tau=full_score_by_tau,
            selection_amp_by_tau=sel_amp_by_tau,
            eval_amp_by_tau=eval_amp_by_tau,
            full_amp_by_tau=full_amp_by_tau,
            selection_msc_by_tau=sel_msc_by_tau,
            eval_msc_by_tau=eval_msc_by_tau,
            full_msc_by_tau=full_msc_by_tau,
            selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
            full_selected_tau_idx=np.asarray(full_selected_idx, dtype=np.int32),
            train_tau_idx=np.asarray(train_tau_idx, dtype=np.int32),
            train_tau_steps=np.asarray(-1 if train_tau_steps is None else int(train_tau_steps), dtype=np.int32),
            apf_dir=str(apf_dir),
            c1_replay_source=np.asarray(replay_source),
        )
        score_rows.append(
            {
                "dataset": dataset_name,
                "trial_idx": idx - 1,
                "trial_uid": trial_uid,
                "source_root": str(item.get("source_root", root)),
                "source_root_rank": int(item.get("source_root_rank", -1)),
                "source_root_name": str(item.get("source_root_name", "arun_lagrangian_apf_500k")),
                "source_optimized_run_idx": int(item.get("source_optimized_run_idx", idx - 1)),
                "optimized_run_idx": int(item.get("optimized_run_idx", idx - 1)),
                "base_run_seed": int(item.get("base_run_seed", item.get("run_seed", -1))),
                "run_seed": int(item.get("run_seed", -1)),
                "metric_seed_protocol": _normalize_metric_seed_protocol(
                    cfg_get(c1_cfg, "metric_seed_protocol", "posthoc_index")
                ),
                "metric_seed": metric_seed_label,
                "score_tau_source": str(
                    cfg_get(
                        c1_cfg,
                        "score_tau_source",
                        cfg_get(metric_cfg_raw, "score_tau_source", "selection"),
                    )
                ),
                "rollout_seed_idx": int(item.get("rollout_seed_idx", 0)),
                "rollout_seed_count": int(item.get("rollout_seed_count", 1)),
                "candidate_kind": str(item.get("candidate_kind", "optimized")),
                "candidate_idx": int(item.get("candidate_idx", 0)),
                "candidate_label": str(item.get("candidate_label", item.get("candidate_kind", "optimized"))),
                "c1_replay_source": replay_source,
                "optimizer_native_iter": int(optimizer_native_meta.get("iter", -1)),
                "optimizer_native_pop_idx": int(optimizer_native_meta.get("pop_idx", -1)),
                "optimizer_native_seed_idx": int(optimizer_native_meta.get("seed_idx", -1)),
                "optimizer_native_legacy_sigma_collision": bool(
                    optimizer_native_meta.get("legacy_sigma_collision", False)
                ),
                "optimizer_native_params_source": str(optimizer_native_meta.get("params_source", "")),
                "optimizer_native_row_params_override": bool(
                    optimizer_native_meta.get("row_params_override", False)
                ),
                "optimizer_native_resolved_sigma": safe_float(
                    optimizer_native_meta.get("resolved_args_sigma", np.nan)
                ),
                "optimizer_native_resolved_flow_sigma": safe_float(
                    optimizer_native_meta.get("resolved_args_flow_sigma", np.nan)
                ),
                "selected_tau_idx": selected_idx,
                "selected_tau_steps": int(tau_steps[selected_idx]),
                "full_selected_tau_idx": full_selected_idx,
                "full_selected_tau_steps": int(tau_steps[full_selected_idx]),
                "train_tau_idx": train_tau_idx,
                "train_tau_steps": int(train_tau_steps) if train_tau_steps is not None else np.nan,
                "selection_score_mspd": float(sel_score_by_tau[selected_idx]),
                "selection_amp": float(sel_amp_by_tau[selected_idx]),
                "selection_msc": float(sel_msc_by_tau[selected_idx]),
                "eval_score_mspd": float(eval_score_by_tau[selected_idx]),
                "eval_amp": float(eval_amp_by_tau[selected_idx]),
                "eval_msc": float(eval_msc_by_tau[selected_idx]),
                "full_score_selected_mspd": float(full_score_by_tau[selected_idx]),
                "full_score_max_mspd": float(full_score_by_tau[full_selected_idx]),
                "full_score_train_tau_mspd": train_tau_score,
                "optimizer_reference_seed_score_mspd": reference_seed_score,
                "optimizer_reference_train_mspd_abs_error": train_mspd_abs_error,
                "optimizer_reference_train_mspd_ulp_distance": train_mspd_ulp_distance,
                "optimizer_reference_train_mspd_exact_match": train_mspd_exact_match,
                "optimizer_reference_train_mspd_validation": train_mspd_validation,
                "optimizer_reference_train_mspd_validation_passed": train_mspd_validation_passed,
                "optimizer_reference_cross_hardware_exception_used": cross_hardware_exception_used,
                "optimizer_reference_cross_hardware_max_ulps": cross_hardware_max_ulps,
                "full_amp_max": float(full_amp_by_tau[full_selected_idx]) if full_amp_by_tau.size else np.nan,
                "full_msc_max": float(full_msc_by_tau[full_selected_idx]) if full_msc_by_tau.size else np.nan,
                "eval_score_max_mspd": float(np.nanmax(eval_score_by_tau)),
                "selection_delta_h_mean": float(np.nanmean(sel_map[selected_idx])),
                "selection_delta_h_median": float(np.nanmedian(sel_map[selected_idx])),
                "selection_delta_h_std": float(np.nanstd(sel_map[selected_idx])),
                "eval_delta_h_mean": float(np.nanmean(eval_values)),
                "eval_delta_h_median": float(np.nanmedian(eval_values)),
                "eval_delta_h_std": float(np.nanstd(eval_values)),
                "window_start_min_steps": int(np.nanmin(eval_window_start)),
                "window_end_max_steps": int(np.nanmax(eval_window_end)),
                "selection_window_start_min_steps": int(np.nanmin(sel_window_start)),
                "selection_window_end_max_steps": int(np.nanmax(sel_window_end)),
                "eval_window_start_min_steps": int(np.nanmin(eval_window_start)),
                "eval_window_end_max_steps": int(np.nanmax(eval_window_end)),
                "apf_dir": str(apf_dir),
                "maps_path": str(maps_path),
                "c1_source": "apf_lagrangian_split",
                "c1_eval_mode": "interleaved_windows_2k_vs_2k_plus_1",
            }
        )

    score_df = pd.DataFrame(score_rows)
    contrast_df = _group_contrasts(score_df, "eval_score_mspd")
    n_random_rows = int((score_df["candidate_kind"] == "random").sum()) if "candidate_kind" in score_df.columns else 0
    if require_random and n_random_rows == 0:
        for stale in (output_dir / "group_contrasts.csv", output_dir / "checkpoint_scores.csv"):
            if stale.exists():
                stale.unlink()
        raise ValueError(
            f"{dataset_name}: C1 APF interleaved holdout source {root} has no candidate_kind=random rows. "
            "Refusing to report optimized-only C1 as a matched contrast."
        )
    if require_random and contrast_df.empty:
        raise ValueError(f"{dataset_name}: C1 APF interleaved holdout source {root} has random rows but no matched groups.")
    if contrast_df.empty:
        contrast_df = pd.DataFrame(
            columns=[
                "optimized_run_idx",
                "eval_score_mspd__optimized",
                "eval_score_mspd__optimized_median",
                "eval_score_mspd__optimized_mean",
                "eval_score_mspd__random_median",
                "eval_score_mspd__random_mean",
                "n_optimized",
                "n_random",
                "delta_vs_random_median",
            ]
        )
    score_df.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    contrast_df.to_csv(output_dir / "group_contrasts.csv", index=False)
    summary = sign_test_greater(contrast_df["delta_vs_random_median"].tolist() if "delta_vs_random_median" in contrast_df.columns else [])
    if not require_random and len(contrast_df) == 0:
        opt_scores = np.asarray(
            [safe_float(v) for v in score_df.get("eval_score_mspd", [])],
            dtype=np.float64,
        )
        opt_scores = opt_scores[np.isfinite(opt_scores)]
        summary.update(
            {
                "comparison_note": "optimized_only_diagnostic_no_random_contrast",
                "optimized_score_median": float(np.nanmedian(opt_scores)) if opt_scores.size else float("nan"),
                "optimized_score_mean": float(np.nanmean(opt_scores)) if opt_scores.size else float("nan"),
            }
        )
    summary.update(
        {
            "source": "apf_lagrangian_split",
            "apf_root": str(root),
            "range_steps": [int(range_start), int(range_end)],
            "holdout_split": "interleaved_windows_2k_vs_2k_plus_1",
            "n_scores": int(len(score_df)),
            "n_contrasts": int(len(contrast_df)),
            "n_random_rows": int(n_random_rows),
        }
    )
    log_event(
        f"{dataset_name}: C1 APF interleaved holdout done n_scores={len(score_df)} n_contrasts={len(contrast_df)} output={output_dir}",
        component="posthoc",
    )
    return score_df, contrast_df, summary


def _compute_c1_from_apf_metrics(
    dataset_name: str,
    ds_cfg: Any,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
    metric_cfg_raw = cfg_get(c1_cfg, "metric", {})
    root = resolve_path(cfg_get(c1_cfg, "apf_root", None))
    if root is None or not root.exists():
        raise FileNotFoundError(f"{dataset_name}: C1 apf_root not found: {root}")
    items = _iter_apf_metric_items(root)
    if not items:
        raise FileNotFoundError(f"{dataset_name}: no APF metric items found under {root}")
    require_random = bool(cfg_get(c1_cfg, "require_random", True))

    expected_start_raw = cfg_get(c1_cfg, "expected_window_start_steps", None)
    expected_end_raw = cfg_get(c1_cfg, "expected_window_end_steps", None)
    expected_start = None if expected_start_raw is None else int(expected_start_raw)
    expected_end = None if expected_end_raw is None else int(expected_end_raw)

    maps_dir = ensure_dir(output_dir / "c1_delta_h_maps")
    if force:
        for stale in maps_dir.glob("*_c1_maps.npz"):
            stale.unlink()

    score_rows: list[dict[str, Any]] = []
    log_event(f"{dataset_name}: C1 APF metrics start n_items={len(items)} root={root}", component="posthoc")
    for idx, item in enumerate(items, start=1):
        metrics_path = Path(item["metrics_path"])
        if idx == 1 or idx == len(items) or idx % 5 == 0:
            log_event(
                f"{dataset_name}: C1 APF scoring {idx}/{len(items)} traj={item['traj_id']} metrics={metrics_path}",
                component="posthoc",
            )
        try:
            with np.load(metrics_path, allow_pickle=False) as data:
                required = [
                    "delta_h_map",
                    "delta_h_score_by_tau",
                    "delta_h_amp_by_tau",
                    "delta_h_msc_by_tau",
                    "delta_h_tau_steps",
                    "delta_h_selected_tau_idx",
                    "delta_h_window_start_steps",
                    "delta_h_window_end_steps",
                ]
                missing = [key for key in required if key not in data.files]
                if missing:
                    raise KeyError(f"missing keys {missing}; available keys={list(data.files)}")
                delta_h_map = np.asarray(data["delta_h_map"], dtype=np.float64)
                score_by_tau = np.asarray(data["delta_h_score_by_tau"], dtype=np.float64).reshape(-1)
                amp_by_tau = np.asarray(data["delta_h_amp_by_tau"], dtype=np.float64).reshape(-1)
                msc_by_tau = np.asarray(data["delta_h_msc_by_tau"], dtype=np.float64).reshape(-1)
                tau_steps = np.asarray(data["delta_h_tau_steps"], dtype=np.int32).reshape(-1)
                selected_idx = int(np.asarray(data["delta_h_selected_tau_idx"]).reshape(-1)[0])
                window_start_steps = np.asarray(data["delta_h_window_start_steps"], dtype=np.int64).reshape(-1)
                window_end_steps = np.asarray(data["delta_h_window_end_steps"], dtype=np.int64).reshape(-1)
                if "delta_h_sample_every_steps" in data.files:
                    sample_every_steps = int(np.asarray(data["delta_h_sample_every_steps"]).reshape(-1)[0])
                else:
                    diffs = np.diff(window_start_steps)
                    positive = diffs[diffs > 0]
                    sample_every_steps = int(np.gcd.reduce(positive.astype(np.int64))) if positive.size else 1
        except Exception as exc:
            raise ValueError(f"{dataset_name}: invalid C1 APF metrics path={metrics_path} {_file_probe(metrics_path)} error={type(exc).__name__}: {exc}") from exc

        if expected_start is not None and int(np.nanmin(window_start_steps)) != expected_start:
            raise ValueError(
                f"{dataset_name}: C1 APF metrics range mismatch for {metrics_path}: "
                f"first_start={int(np.nanmin(window_start_steps))}, expected {expected_start}."
            )
        if expected_end is not None and int(np.nanmax(window_end_steps)) != expected_end:
            raise ValueError(
                f"{dataset_name}: C1 APF metrics range mismatch for {metrics_path}: "
                f"last_end={int(np.nanmax(window_end_steps))}, expected {expected_end}."
            )

        rollout_steps = int(np.nanmax(window_end_steps) - np.nanmin(window_start_steps))
        metric_cfg = _metric_config_from_timing(metric_cfg_raw, rollout_steps=rollout_steps, sample_every=max(1, sample_every_steps))
        selection_mask, eval_mask = _interleaved_window_masks(delta_h_map.shape[1])
        sel_score_by_tau, sel_amp_by_tau, sel_msc_by_tau, sel_map = _score_delta_h_map_subset(metric_cfg, delta_h_map, selection_mask)
        eval_score_by_tau, eval_amp_by_tau, eval_msc_by_tau, eval_map = _score_delta_h_map_subset(metric_cfg, delta_h_map, eval_mask)
        selected_idx = int(np.nanargmax(sel_score_by_tau))
        sel_window_start = window_start_steps[selection_mask]
        sel_window_end = window_end_steps[selection_mask]
        eval_window_start = window_start_steps[eval_mask]
        eval_window_end = window_end_steps[eval_mask]
        trial_uid = str(item["traj_id"])
        maps_path = maps_dir / f"{trial_uid}_c1_maps.npz"
        np.savez_compressed(
            maps_path,
            delta_h_selection=sel_map,
            delta_h_eval=eval_map,
            tau_steps=tau_steps,
            window_start_steps=eval_window_start.astype(np.int64),
            window_end_steps=eval_window_end.astype(np.int64),
            selection_window_start_steps=sel_window_start.astype(np.int64),
            selection_window_end_steps=sel_window_end.astype(np.int64),
            eval_window_start_steps=eval_window_start.astype(np.int64),
            eval_window_end_steps=eval_window_end.astype(np.int64),
            selection_score_by_tau=sel_score_by_tau,
            eval_score_by_tau=eval_score_by_tau,
            selection_amp_by_tau=sel_amp_by_tau,
            eval_amp_by_tau=eval_amp_by_tau,
            selection_msc_by_tau=sel_msc_by_tau,
            eval_msc_by_tau=eval_msc_by_tau,
            selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
            source_metrics_path=str(metrics_path),
        )
        eval_values = eval_map[selected_idx]
        score_rows.append(
            {
                "dataset": dataset_name,
                "trial_idx": idx - 1,
                "trial_uid": trial_uid,
                "source_root": str(item.get("source_root", root)),
                "source_root_rank": int(item.get("source_root_rank", -1)),
                "source_root_name": str(item.get("source_root_name", "arun_lagrangian_apf_500k")),
                "source_optimized_run_idx": int(item.get("source_optimized_run_idx", idx - 1)),
                "optimized_run_idx": int(item.get("optimized_run_idx", idx - 1)),
                "candidate_kind": str(item.get("candidate_kind", "optimized")),
                "candidate_idx": int(item.get("candidate_idx", 0)),
                "candidate_label": str(item.get("candidate_label", item.get("candidate_kind", "optimized"))),
                "selected_tau_idx": selected_idx,
                "selected_tau_steps": int(tau_steps[selected_idx]),
                "selection_score_mspd": float(sel_score_by_tau[selected_idx]),
                "selection_amp": float(sel_amp_by_tau[selected_idx]),
                "selection_msc": float(sel_msc_by_tau[selected_idx]),
                "eval_score_mspd": float(eval_score_by_tau[selected_idx]),
                "eval_amp": float(eval_amp_by_tau[selected_idx]),
                "eval_msc": float(eval_msc_by_tau[selected_idx]),
                "eval_delta_h_mean": float(np.nanmean(eval_values)),
                "eval_delta_h_median": float(np.nanmedian(eval_values)),
                "eval_delta_h_std": float(np.nanstd(eval_values)),
                "window_start_min_steps": int(np.nanmin(eval_window_start)),
                "window_end_max_steps": int(np.nanmax(eval_window_end)),
                "selection_window_start_min_steps": int(np.nanmin(sel_window_start)),
                "selection_window_end_max_steps": int(np.nanmax(sel_window_end)),
                "eval_window_start_min_steps": int(np.nanmin(eval_window_start)),
                "eval_window_end_max_steps": int(np.nanmax(eval_window_end)),
                "metrics_path": str(metrics_path),
                "maps_path": str(maps_path),
                "c1_source": "apf_metrics",
                "c1_eval_mode": "interleaved_windows_2k_vs_2k_plus_1_from_cached_apf_metrics",
            }
        )

    score_df = pd.DataFrame(score_rows)
    contrast_df = _group_contrasts(score_df, "eval_score_mspd")
    n_random_rows = int((score_df["candidate_kind"] == "random").sum()) if "candidate_kind" in score_df.columns else 0
    if require_random and n_random_rows == 0:
        for stale in (
            output_dir / "group_contrasts.csv",
            output_dir / "checkpoint_scores.csv",
        ):
            if stale.exists():
                stale.unlink()
        raise ValueError(
            f"{dataset_name}: C1 APF source {root} has no candidate_kind=random metrics. "
            "Refusing to fabricate random=0 baselines. Generate/add random APF metrics "
            "under the same manifest, or set c1.require_random=false only for optimized-only diagnostics."
        )
    if require_random and contrast_df.empty:
        raise ValueError(f"{dataset_name}: C1 APF source {root} has random rows but no matched optimized-vs-random groups.")
    if contrast_df.empty:
        contrast_df = pd.DataFrame(
            columns=["optimized_run_idx", "eval_score_mspd__optimized", "eval_score_mspd__random_median", "n_random", "delta_vs_random_median"]
        )
    score_df.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    contrast_df.to_csv(output_dir / "group_contrasts.csv", index=False)
    summary = sign_test_greater(contrast_df["delta_vs_random_median"].tolist() if "delta_vs_random_median" in contrast_df.columns else [])
    summary.update(
        {
            "source": "apf_metrics",
            "apf_root": str(root),
            "n_scores": int(len(score_df)),
            "n_contrasts": int(len(contrast_df)),
            "comparison_note": (
                "Flow-Lenia C1 reads matched optimized/random APF metrics."
                if len(contrast_df)
                else "Flow-Lenia C1 reads optimized A-run APF metrics; matched random contrasts are unavailable in this source."
            ),
        }
    )
    log_event(
        f"{dataset_name}: C1 APF metrics done n_scores={len(score_df)} n_contrasts={len(contrast_df)} output={output_dir}",
        component="posthoc",
    )
    return score_df, contrast_df, summary


def _compute_c1(
    dataset_name: str,
    rows: pd.DataFrame,
    ds_cfg: Any,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
    metric_cfg_raw = cfg_get(c1_cfg, "metric", {})
    expected_start_raw = cfg_get(c1_cfg, "expected_trajectory_start_steps", None)
    expected_end_raw = cfg_get(c1_cfg, "expected_trajectory_end_steps", None)
    expected_start = None if expected_start_raw is None else int(expected_start_raw)
    expected_end = None if expected_end_raw is None else int(expected_end_raw)
    require_random = bool(cfg_get(c1_cfg, "require_random", False))
    available = rows.copy()
    available["lagrangian_abs_path"] = [_resolve_artifact(row, "lagrangian_path") for _, row in available.iterrows()]
    available = available[[path is not None and Path(path).exists() for path in available["lagrangian_abs_path"]]].reset_index(drop=True)
    if available.empty:
        raise FileNotFoundError(f"{dataset_name}: no lagrangian artifacts found for C1.")

    metric_cfg = _metric_config_from_lagrangian(Path(available.iloc[0]["lagrangian_abs_path"]), metric_cfg_raw)
    metric_eval = jax.jit(make_metric_loss_fn(metric_cfg, include_maps=True))
    maps_dir = ensure_dir(output_dir / "c1_delta_h_maps")
    if force:
        for stale in maps_dir.glob("*_c1_maps.npz"):
            stale.unlink()
    score_rows: list[dict[str, Any]] = []
    total = int(available.shape[0])
    log_event(f"{dataset_name}: C1 start n_lagrangian={total}", component="posthoc")
    for idx, (_, row) in enumerate(available.iterrows(), start=1):
        lag_path = Path(row["lagrangian_abs_path"])
        if idx == 1 or idx == total or idx % 5 == 0:
            log_event(
                f"{dataset_name}: C1 scoring {idx}/{total} trial_idx={row.get('trial_idx', 'na')} path={lag_path}",
                component="posthoc",
            )
        trajectory_key_cfg = cfg_get(c1_cfg, "trajectory_key", None)
        try:
            with np.load(lag_path, allow_pickle=False) as data:
                trajectory_key = _primary_lagrangian_xy_key(data, trajectory_key_cfg)
                xy_primary = np.asarray(data[trajectory_key], dtype=np.float32)
                if expected_start is not None:
                    if "trajectory_start_steps" not in data.files:
                        raise ValueError(f"{dataset_name}: C1 {lag_path} missing trajectory_start_steps.")
                    actual_start = int(np.asarray(data["trajectory_start_steps"]).item())
                    if actual_start != expected_start:
                        raise ValueError(
                            f"{dataset_name}: C1 trajectory range mismatch for {lag_path}: "
                            f"start={actual_start}, expected {expected_start}."
                        )
                if expected_end is not None:
                    if "trajectory_end_steps" not in data.files:
                        raise ValueError(f"{dataset_name}: C1 {lag_path} missing trajectory_end_steps.")
                    actual_end = int(np.asarray(data["trajectory_end_steps"]).item())
                    if actual_end != expected_end:
                        raise ValueError(
                            f"{dataset_name}: C1 trajectory range mismatch for {lag_path}: "
                            f"end={actual_end}, expected {expected_end}."
                        )
        except Exception as exc:
            raise ValueError(
                f"{dataset_name}: invalid C1 lagrangian trajectory path={lag_path} {_file_probe(lag_path)} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        metric_seed = int(row.get("metric_seed", row.get("seed_x", 0) + 10_000_000))
        info = _score_maps(metric_eval, metric_seed, xy_primary)
        full_map = np.asarray(info["delta_h_map"], dtype=np.float64)
        selection_mask, eval_mask = _interleaved_window_masks(full_map.shape[1])
        sel_score_by_tau, sel_amp_by_tau, sel_msc_by_tau, sel_map = _score_delta_h_map_subset(
            metric_cfg,
            full_map,
            selection_mask,
        )
        eval_score_by_tau, eval_amp_by_tau, eval_msc_by_tau, eval_map = _score_delta_h_map_subset(
            metric_cfg,
            full_map,
            eval_mask,
        )
        selected_idx = int(np.nanargmax(sel_score_by_tau))
        tau_steps = np.asarray(info["tau_steps"], dtype=np.int32)
        eval_values = eval_map[selected_idx]
        eval_score_mspd = float(eval_score_by_tau[selected_idx])
        full_window_start = np.asarray(info["window_start_steps"], dtype=np.int64).reshape(-1)
        sel_window_start = full_window_start[selection_mask]
        eval_window_start = full_window_start[eval_mask]
        trial_uid = str(row.get("trial_uid", f"trial_{int(row['trial_idx']):05d}"))
        maps_path = maps_dir / f"{trial_uid}_c1_maps.npz"
        np.savez_compressed(
            maps_path,
            delta_h_selection=sel_map,
            delta_h_eval=eval_map,
            tau_steps=tau_steps,
            window_start_steps=eval_window_start.astype(np.int64),
            selection_window_start_steps=sel_window_start.astype(np.int64),
            eval_window_start_steps=eval_window_start.astype(np.int64),
            selection_score_by_tau=sel_score_by_tau,
            eval_score_by_tau=eval_score_by_tau,
            selection_amp_by_tau=sel_amp_by_tau,
            eval_amp_by_tau=eval_amp_by_tau,
            selection_msc_by_tau=sel_msc_by_tau,
            eval_msc_by_tau=eval_msc_by_tau,
            selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
            trajectory_key=np.asarray(str(trajectory_key)),
        )
        score_rows.append(
            {
                "dataset": dataset_name,
                "trial_idx": int(row["trial_idx"]),
                "trial_uid": trial_uid,
                "source_root": str(row.get("source_root", "")),
                "source_root_rank": int(row.get("source_root_rank", -1)),
                "source_root_name": str(row.get("source_root_name", "")),
                "source_optimized_run_idx": int(row.get("source_optimized_run_idx", row["optimized_run_idx"])),
                "optimized_run_idx": int(row["optimized_run_idx"]),
                "candidate_kind": row["candidate_kind_canon"],
                "candidate_idx": int(row.get("candidate_idx", 0)),
                "candidate_label": str(row.get("candidate_label", row["candidate_kind_canon"])),
                "selected_tau_idx": selected_idx,
                "selected_tau_steps": int(tau_steps[selected_idx]),
                "selection_score_mspd": float(sel_score_by_tau[selected_idx]),
                "selection_amp": float(sel_amp_by_tau[selected_idx]),
                "selection_msc": float(sel_msc_by_tau[selected_idx]),
                "eval_score_mspd": eval_score_mspd,
                "eval_amp": float(eval_amp_by_tau[selected_idx]),
                "eval_msc": float(eval_msc_by_tau[selected_idx]),
                "selection_delta_h_mean": float(np.nanmean(sel_map[selected_idx])),
                "selection_delta_h_median": float(np.nanmedian(sel_map[selected_idx])),
                "selection_delta_h_std": float(np.nanstd(sel_map[selected_idx])),
                "eval_delta_h_mean": float(np.nanmean(eval_values)),
                "eval_delta_h_median": float(np.nanmedian(eval_values)),
                "eval_delta_h_std": float(np.nanstd(eval_values)),
                "selection_window_start_min_steps": int(np.nanmin(sel_window_start)),
                "selection_window_start_max_steps": int(np.nanmax(sel_window_start)),
                "eval_window_start_min_steps": int(np.nanmin(eval_window_start)),
                "eval_window_start_max_steps": int(np.nanmax(eval_window_start)),
                "maps_path": str(maps_path),
                "trajectory_key": str(trajectory_key),
                "c1_eval_mode": "single_trajectory_interleaved_windows_2k_vs_2k_plus_1",
            }
        )
    score_df = pd.DataFrame(score_rows)
    contrast_df = _group_contrasts(score_df, "eval_score_mspd")
    n_random_rows = int((score_df["candidate_kind"] == "random").sum()) if "candidate_kind" in score_df.columns else 0
    if require_random and n_random_rows == 0:
        for stale in (
            output_dir / "group_contrasts.csv",
            output_dir / "checkpoint_scores.csv",
        ):
            if stale.exists():
                stale.unlink()
        raise ValueError(
            f"{dataset_name}: C1 lagrangian source has no candidate_kind=random rows. "
            "Refusing to report optimized-only C1/C6 as a matched contrast."
        )
    if require_random and contrast_df.empty:
        raise ValueError(f"{dataset_name}: C1 lagrangian source has random rows but no matched optimized-vs-random groups.")
    score_df.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    contrast_df.to_csv(output_dir / "group_contrasts.csv", index=False)
    log_event(
        f"{dataset_name}: C1 done n_scores={len(score_df)} n_contrasts={len(contrast_df)} output={output_dir}",
        component="posthoc",
    )
    summary = sign_test_greater(contrast_df["delta_vs_random_median"].tolist())
    summary.update(
        {
            "source": _c1_source_mode(ds_cfg),
            "holdout_split": "interleaved_windows_2k_vs_2k_plus_1",
            "n_scores": int(len(score_df)),
            "n_contrasts": int(len(contrast_df)),
            "n_random_rows": int(n_random_rows),
        }
    )
    return score_df, contrast_df, summary


def _group_contrasts(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for group_idx, group in frame.groupby("optimized_run_idx"):
        opt = group[group["candidate_kind"] == "optimized"]
        randoms = group[group["candidate_kind"] == "random"]
        if opt.empty or randoms.empty:
            continue
        opt_values = [safe_float(v) for v in opt[metric]]
        rand_values = [safe_float(v) for v in randoms[metric]]
        opt_arr = np.asarray(opt_values, dtype=np.float64)
        rand_arr = np.asarray(rand_values, dtype=np.float64)
        opt_finite = opt_arr[np.isfinite(opt_arr)]
        rand_finite = rand_arr[np.isfinite(rand_arr)]
        if opt_finite.size == 0 or rand_finite.size == 0:
            continue
        opt_value = float(np.nanmedian(opt_finite))
        rand_median = nanmedian(rand_values)
        rows.append(
            {
                "optimized_run_idx": int(group_idx),
                f"{metric}__optimized": opt_value,
                f"{metric}__optimized_median": opt_value,
                f"{metric}__optimized_mean": float(np.nanmean(opt_finite)),
                f"{metric}__random_median": rand_median,
                f"{metric}__random_mean": float(np.nanmean(rand_finite)),
                "n_optimized": int(opt_finite.size),
                "n_random": int(len(rand_values)),
                "delta_vs_random_median": float(opt_value - rand_median),
            }
        )
    return pd.DataFrame(rows)


def _c1_claim_label(dataset_name: str) -> str:
    return "C6.1" if str(dataset_name) == "plife_plus" else "C1"


def _c5_claim_label(dataset_name: str) -> str:
    return "C6.5" if str(dataset_name) == "plife_plus" else "C5"


def _is_c6_transfer_dataset(dataset_name: str) -> bool:
    return str(dataset_name) in {"plife_plus", "boids"}


def _append_existing_cross_rows(
    *,
    dataset_name: str,
    ds_out: Path,
    cross_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = ds_out / "dataset_summary.json"
    if not summary_path.exists():
        return {"status": "skipped", "reason": "not a C6 transfer dataset and no existing dataset_summary.json"}
    try:
        status = json.loads(summary_path.read_text())
    except Exception as exc:
        return {"status": "skipped", "reason": f"could not read existing dataset_summary.json: {exc}"}
    c1_summary = status.get("c1")
    if isinstance(c1_summary, dict) and c1_summary:
        cross_rows.append({"dataset": dataset_name, "claim": _c1_claim_label(dataset_name), "metric": "eval_score_mspd", **c1_summary})
    c5_summary = status.get("c5")
    if isinstance(c5_summary, dict) and c5_summary:
        cross_rows.append({"dataset": dataset_name, "claim": _c5_claim_label(dataset_name), "metric": c5_summary.get("metric", "frustration"), **c5_summary})
    status = dict(status)
    status["status"] = status.get("status", "ok")
    status["reused_for_c6"] = True
    return status


def _write_analysis_config(output_dir: Path, ds_cfg: Any, frustration_root: Path) -> Path:
    explicit = cfg_get(ds_cfg, "analysis_config", None)
    if explicit is not None:
        path = resolve_path(explicit)
        if path is not None and path.exists():
            return path
    c5_metric = cfg_get(cfg_get(ds_cfg, "c5", {}), "metric", {})
    payload = {
        "source": {"type": "history_dependence_eval", "path": str(frustration_root.relative_to(REPO_ROOT)) if str(frustration_root).startswith(str(REPO_ROOT)) else str(frustration_root)},
        "output": {"dir": str(output_dir / "history_distance_analysis")},
        "embeddings": {
            "enabled": True,
            "normalize": True,
            "synced_metrics": ["cosine", "euclidean"],
            "cloud_metrics": ["cosine"],
            "cloud_method": "chamfer",
            "primary_metric": "embedding_cloud_chamfer_cosine",
        },
        "trajectories": {
            "enabled": True,
            "metric_tau_mode": cfg_get(c5_metric, "metric_tau_mode", "fixed"),
            "metric_tau_steps": cfg_get(c5_metric, "metric_tau_steps", 3000),
            "metric_tau_grid_steps": cfg_get(c5_metric, "metric_tau_grid_steps", None),
            "metric_window_size_steps": cfg_get(c5_metric, "metric_window_size_steps", 20000),
            "metric_window_step_steps": cfg_get(c5_metric, "metric_window_step_steps", 5000),
            "metric_range_start_steps": cfg_get(c5_metric, "metric_range_start_steps", 0),
            "metric_range_end_steps": cfg_get(c5_metric, "metric_range_end_steps", None),
            "metric_m_samples": cfg_get(c5_metric, "metric_m_samples", 48),
            "metric_m_min": cfg_get(c5_metric, "metric_m_min", 4),
            "metric_n_proj": cfg_get(c5_metric, "metric_n_proj", 16),
            "metric_null_reps": cfg_get(c5_metric, "metric_null_reps", 6),
            "metric_particle_samples": cfg_get(c5_metric, "metric_particle_samples", 64),
            "metric_dirs_seed": cfg_get(c5_metric, "metric_dirs_seed", 123),
            "metric_periodic": cfg_get(c5_metric, "metric_periodic", False),
            "metric_domain_y": cfg_get(c5_metric, "metric_domain_y", 0.0),
            "metric_domain_x": cfg_get(c5_metric, "metric_domain_x", 0.0),
            "metric_preprocess_mode": cfg_get(c5_metric, "metric_preprocess_mode", "clip"),
            "metric_scales": cfg_get(c5_metric, "metric_scales", None),
            "metric_scale_weights": cfg_get(c5_metric, "metric_scale_weights", None),
            "metric_delta_h_floor": cfg_get(c5_metric, "metric_delta_h_floor", 0.0),
            "metric_msc_floor": cfg_get(c5_metric, "metric_msc_floor", 0.01),
            "metric_msc_term": cfg_get(c5_metric, "metric_msc_term", "floor_reconstruction_error"),
            "metric_msc_normalize_by_weight_sum": cfg_get(c5_metric, "metric_msc_normalize_by_weight_sum", True),
            "metric_alpha": cfg_get(c5_metric, "metric_alpha", 0.0),
            "metric_beta": cfg_get(c5_metric, "metric_beta", 1.0),
            "metric_eps": cfg_get(c5_metric, "metric_eps", 1e-12),
            "occupancy_bins": 64,
            "pairwise_map_metrics": ["l2", "mean_abs"],
            "fixed_tau_distribution_steps": cfg_get(c5_metric, "fixed_tau_distribution_steps", cfg_get(c5_metric, "metric_tau_steps", 3000)),
            "pairwise_distribution_metrics": ["wasserstein", "ks", "energy"],
        },
        "progress": {"enabled": False, "show_inner": False},
    }
    path = output_dir / "generated_history_analysis_config.yaml"
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True))
    return path


def _derive_anchor_columns(rows: pd.DataFrame, base_names: list[str]) -> pd.DataFrame:
    out = rows.copy()
    if "anchor_effect_minus_baseline" not in out.columns and {"walls_effect_distance_ctrl_a", "baseline_distance"}.issubset(out.columns):
        out["anchor_effect_minus_baseline"] = out["walls_effect_distance_ctrl_a"] - out["baseline_distance"]
    if "anchor_effect_over_baseline_ratio" not in out.columns and {"walls_effect_distance_ctrl_a", "baseline_distance"}.issubset(out.columns):
        out["anchor_effect_over_baseline_ratio"] = out["walls_effect_distance_ctrl_a"] / (
            out["baseline_distance"] + 1e-12
        )
    if "clip_oe_loss_walls_minus_control_a" not in out.columns and {"clip_oe_loss_walls", "clip_oe_loss_control_a"}.issubset(out.columns):
        out["clip_oe_loss_walls_minus_control_a"] = out["clip_oe_loss_walls"] - out["clip_oe_loss_control_a"]
    if "msc_score_walls_minus_control_a" not in out.columns and {"msc_score_walls", "msc_score_control_a"}.issubset(out.columns):
        out["msc_score_walls_minus_control_a"] = out["msc_score_walls"] - out["msc_score_control_a"]
    for base in base_names:
        ctrl = f"{base}__walls_effect_distance_ctrl_a"
        baseline = f"{base}__baseline_distance"
        anchor = f"{base}__anchor_effect_minus_baseline"
        if anchor not in out.columns and {ctrl, baseline}.issubset(out.columns):
            out[anchor] = out[ctrl] - out[baseline]
        anchor_ratio = f"{base}__anchor_effect_over_baseline_ratio"
        if anchor_ratio not in out.columns and {ctrl, baseline}.issubset(out.columns):
            out[anchor_ratio] = out[ctrl] / (out[baseline] + 1e-12)
    return out


def _compute_c5(dataset_name: str, rows: pd.DataFrame, ds_cfg: Any, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fs_root = Path(str(rows.iloc[0]["frustration_root"]))
    analysis_config = _write_analysis_config(output_dir, ds_cfg, fs_root)
    log_event(
        f"{dataset_name}: C5 start n_trials={len(rows)} root={fs_root} analysis_config={analysis_config}",
        component="posthoc",
    )
    augmented, _effect_cols, base_names = augment_rows_with_history_dependence_distances(
        rows,
        analysis_config_path=analysis_config,
        show_progress=True,
    )
    augmented = _derive_anchor_columns(augmented, base_names)
    augmented.to_csv(output_dir / "frustration_trial_metrics.csv", index=False)

    preferred = [
        "anchor_effect_minus_baseline",
        "anchor_effect_over_baseline_ratio",
        "walls_effect_distance_ctrl_a",
        "clip_oe_loss_walls_minus_control_a",
        "msc_score_walls_minus_control_a",
        "embedding_cloud_chamfer_cosine__anchor_effect_minus_baseline",
        "embedding_cloud_chamfer_cosine__anchor_effect_over_baseline_ratio",
        "embedding_synced_cosine__anchor_effect_minus_baseline",
        "delta_h_l2__anchor_effect_minus_baseline",
        "delta_h_mean_abs__anchor_effect_minus_baseline",
        "msc_score_anchor_absdiff_minus_baseline",
    ]
    metric_cols = [col for col in preferred if col in augmented.columns]
    run_rows = []
    for group_idx, group in augmented.groupby("optimized_run_idx"):
        opt = group[group["candidate_kind_canon"] == "optimized"]
        randoms = group[group["candidate_kind_canon"] == "random"]
        if opt.empty or randoms.empty:
            continue
        opt0 = opt.iloc[0]
        row_out = {
            "dataset": dataset_name,
            "optimized_run_idx": int(group_idx),
            "source_root_rank": int(opt0.get("source_root_rank", -1)),
            "source_root_name": str(opt0.get("source_root_name", "")),
            "source_optimized_run_idx": int(opt0.get("source_optimized_run_idx", group_idx)),
            "n_random": int(randoms.shape[0]),
        }
        for col in metric_cols:
            opt_value = safe_float(opt.iloc[0][col])
            rand_median = nanmedian(randoms[col].tolist())
            row_out[f"{col}__optimized"] = opt_value
            row_out[f"{col}__random_median"] = rand_median
            row_out[f"{col}__delta_vs_random_median"] = float(opt_value - rand_median)
        run_rows.append(row_out)
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(output_dir / "frustration_run_level.csv", index=False)
    summary_rows = []
    for col in metric_cols:
        delta_col = f"{col}__delta_vs_random_median"
        if delta_col not in run_df.columns:
            continue
        summary = sign_test_greater(run_df[delta_col].tolist())
        summary_rows.append({"dataset": dataset_name, "metric": col, **summary})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "frustration_metric_summary.csv", index=False)
    primary_metric = "embedding_cloud_chamfer_cosine__anchor_effect_minus_baseline"
    primary = summary_df[summary_df["metric"] == primary_metric]
    primary_summary = primary.iloc[0].to_dict() if not primary.empty else (summary_rows[0] if summary_rows else {})
    log_event(
        f"{dataset_name}: C5 done n_run_rows={len(run_df)} n_summary_rows={len(summary_df)} primary_metric={primary_summary.get('metric', 'none')}",
        component="posthoc",
    )
    return run_df, summary_df, primary_summary


def _write_smoke_dataset(output_root: Path, dataset_name: str) -> dict[str, Any]:
    root = ensure_dir(output_root / "smoke_inputs" / dataset_name / "frustration_simulation")
    trial_data = ensure_dir(root / "trial_data")
    stable_name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(dataset_name))
    rng = np.random.default_rng(700 + stable_name_seed % 1000)
    T, N = 72, 36

    def make_xy(kind: str, seed: int) -> np.ndarray:
        rr = np.random.default_rng(seed)
        x = rr.uniform(0.0, 1.0, size=(N, 2)).astype(np.float32)
        xy = np.empty((T, N, 2), dtype=np.float32)
        labels = np.arange(N) % 2
        for t in range(T):
            xy[t] = x
            if kind == "optimized":
                v = np.where(labels[:, None] == 0, np.asarray([0.002, 0.0]), np.asarray([0.0, 0.010])).astype(np.float32)
            else:
                v = rr.normal(0.0, 0.002, size=(N, 2)).astype(np.float32)
            x = np.mod(x + v + rr.normal(0.0, 0.0008, size=(N, 2)).astype(np.float32), 1.0)
        return xy

    rows = []
    specs = [("optimized", 0), ("random", 0), ("random", 1), ("random", 2)]
    for trial_idx, (kind, cand_idx) in enumerate(specs):
        xy_a = make_xy(kind, 1000 + trial_idx)
        xy_b = make_xy(kind, 2000 + trial_idx)
        xy_w = xy_a.copy()
        xy_w[20:] = np.mod(xy_w[20:] + (0.025 if kind == "optimized" else 0.006), 1.0)
        z_a = rng.normal(0, 1, size=(8, 16)).astype(np.float32)
        z_b = z_a + rng.normal(0, 0.03, size=z_a.shape).astype(np.float32)
        z_w = z_a + rng.normal(0, 0.15 if kind == "optimized" else 0.05, size=z_a.shape).astype(np.float32)
        lag_path = trial_data / f"trial_{trial_idx:05d}_lagrangian.npz"
        emb_path = trial_data / f"trial_{trial_idx:05d}_embeddings.npz"
        np.savez_compressed(
            lag_path,
            xy_control_a=xy_a,
            xy_control_b=xy_b,
            xy_walls=xy_w,
            sample_every_steps=np.asarray(1, dtype=np.int32),
            trajectory_start_steps=np.asarray(0, dtype=np.int32),
            trajectory_end_steps=np.asarray(T, dtype=np.int32),
            trajectory_window_steps=np.asarray(T, dtype=np.int32),
            metric_window_size_steps=np.asarray(24, dtype=np.int32),
            metric_window_step_steps=np.asarray(12, dtype=np.int32),
            metric_tau_steps=np.asarray(4, dtype=np.int32),
        )
        np.savez_compressed(emb_path, z_control_a=z_a, z_control_b=z_b, z_walls=z_w)
        row = {
            "trial_idx": trial_idx,
            "optimized_run_idx": 0,
            "candidate_kind": kind,
            "candidate_idx": cand_idx,
            "candidate_label": "optimized" if kind == "optimized" else f"random_{cand_idx:03d}",
            "seed_x": 10 + trial_idx,
            "seed_x1": 100 + trial_idx,
            "metric_seed": 1000 + trial_idx,
            "embeddings_path": str(emb_path.relative_to(root)),
            "lagrangian_path": str(lag_path.relative_to(root)),
        }
        (trial_data / f"trial_{trial_idx:05d}.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "trial_results.csv", index=False)
    return {"path": str(root)}


def run(config_path: str | Path, *, task: str = "all", smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    datasets = dataset_items(cfg)
    log_event(
        f"posthoc start task={task} smoke={smoke} output_root={output_root} n_config_datasets={len(datasets)}",
        component="posthoc",
    )
    if smoke:
        datasets = []
        for name in ("flow_lenia", "plife_plus", "boids"):
            fake = _write_smoke_dataset(output_root, name)
            ds = OmegaConf.create(
                {
                    "enabled": True,
                    "required": True,
                    "frustration_root": fake["path"],
                    "c1": {
                        "metric": {
                            "metric_tau_mode": "max_grid",
                            "metric_tau_grid_steps": [1, 2, 4, 8],
                            "metric_tau_steps": 4,
                            "metric_window_size_steps": 24,
                            "metric_window_step_steps": 12,
                            "metric_m_samples": 12,
                            "metric_null_reps": 2,
                            "metric_particle_samples": 24,
                            "metric_n_proj": 6,
                            "metric_periodic": True,
                            "metric_domain_y": 1.0,
                            "metric_domain_x": 1.0,
                        }
                    },
                    "c5": {
                        "metric": {
                            "metric_tau_mode": "max_grid",
                            "metric_tau_grid_steps": [1, 2, 4, 8],
                            "metric_tau_steps": 4,
                            "fixed_tau_distribution_steps": 4,
                            "metric_window_size_steps": 24,
                            "metric_window_step_steps": 12,
                            "metric_m_samples": 12,
                            "metric_null_reps": 2,
                            "metric_particle_samples": 24,
                            "metric_n_proj": 6,
                        }
                    },
                }
            )
            datasets.append((name, ds))

    cross_rows = []
    overview: dict[str, Any] = {}
    for dataset_name, ds in datasets:
        ds_out = ensure_dir(output_root / dataset_name)
        log_event(f"{dataset_name}: dataset start output={ds_out}", component="posthoc")
        if task == "c6" and not _is_c6_transfer_dataset(dataset_name):
            status = _append_existing_cross_rows(dataset_name=dataset_name, ds_out=ds_out, cross_rows=cross_rows)
            overview[dataset_name] = status
            log_event(
                f"{dataset_name}: skipped C6 recompute for non-transfer dataset status={status.get('status')} "
                f"reason={status.get('reason', 'reused existing summary')}",
                component="posthoc",
            )
            continue
        c1_apf_source = _c1_uses_apf(ds)
        c1_lagrangian_source = _c1_uses_lagrangian_root(ds)
        needs_frustration_rows = task in {"all", "c5", "c6"} or (
            task in {"c1"} and not (c1_apf_source or c1_lagrangian_source)
        )
        fs_roots: list[Path] = []
        if needs_frustration_rows:
            roots_raw = cfg_get(ds, "frustration_roots", None)
            if roots_raw is not None:
                fs_roots = [resolve_path(x) for x in as_list(roots_raw)]
                fs_roots = [Path(x) for x in fs_roots if x is not None and Path(x).exists()]
            else:
                root_raw = cfg_get(ds, "frustration_root", None) or cfg_get(ds, "paper_check_root", None)
                if root_raw is None:
                    root_raw = cfg_get(ds, "checkpoint_root", None)
                fs_root = resolve_path(root_raw) if root_raw is not None else None
                fs_roots = [Path(fs_root)] if fs_root is not None and Path(fs_root).exists() else []
            if not fs_roots:
                if bool(cfg_get(ds, "required", False)):
                    raise ValueError(f"{dataset_name}: no existing frustration_root/frustration_roots configured.")
                overview[dataset_name] = {"status": "skipped", "reason": "no root configured"}
                log_event(f"{dataset_name}: skipped no root configured", component="posthoc")
                continue
        try:
            if fs_roots:
                log_event(f"{dataset_name}: loading roots {[str(x) for x in fs_roots]}", component="posthoc")
                rows = _load_trial_rows_many(fs_roots)
                log_event(f"{dataset_name}: loaded n_trials={rows.shape[0]}", component="posthoc")
            else:
                rows = pd.DataFrame()
            status = {"status": "ok", "n_trials": int(rows.shape[0]), "frustration_roots": [str(x) for x in fs_roots]}
            if task in {"all", "c1", "c6"}:
                if _c1_uses_apf_lagrangian_split(ds):
                    _scores, contrasts, c1_summary = _compute_c1_from_apf_lagrangian_split(dataset_name, ds, ds_out, force=force)
                elif c1_apf_source:
                    _scores, contrasts, c1_summary = _compute_c1_from_apf_metrics(dataset_name, ds, ds_out, force=force)
                elif c1_lagrangian_source:
                    c1_roots = _c1_lagrangian_roots(ds)
                    if not c1_roots:
                        raise ValueError(f"{dataset_name}: C1 lagrangian_root/lagrangian_roots not found.")
                    log_event(f"{dataset_name}: loading C1 lagrangian roots {[str(x) for x in c1_roots]}", component="posthoc")
                    c1_rows = _load_trial_rows_many(c1_roots)
                    log_event(f"{dataset_name}: loaded C1 lagrangian n_trials={c1_rows.shape[0]}", component="posthoc")
                    status["c1_lagrangian_roots"] = [str(x) for x in c1_roots]
                    status["c1_lagrangian_trials"] = int(c1_rows.shape[0])
                    _scores, contrasts, c1_summary = _compute_c1(dataset_name, c1_rows, ds, ds_out, force=force)
                else:
                    _scores, contrasts, c1_summary = _compute_c1(dataset_name, rows, ds, ds_out, force=force)
                status["c1"] = c1_summary
                cross_rows.append({"dataset": dataset_name, "claim": _c1_claim_label(dataset_name), "metric": "eval_score_mspd", **c1_summary})
            if task in {"all", "c5", "c6"}:
                _run_df, _summary_df, c5_summary = _compute_c5(dataset_name, rows, ds, ds_out)
                status["c5"] = c5_summary
                if c5_summary:
                    cross_rows.append({"dataset": dataset_name, "claim": _c5_claim_label(dataset_name), "metric": c5_summary.get("metric", "frustration"), **c5_summary})
            write_json(ds_out / "dataset_summary.json", status)
            overview[dataset_name] = status
            log_event(f"{dataset_name}: dataset done summary={ds_out / 'dataset_summary.json'}", component="posthoc")
        except Exception as exc:
            if bool(cfg_get(ds, "required", False)):
                log_event(f"{dataset_name}: failed required dataset error={exc}", component="posthoc")
                raise
            overview[dataset_name] = {"status": "skipped", "reason": str(exc)}
            log_event(f"{dataset_name}: skipped optional dataset error={exc}", component="posthoc")
    if cross_rows:
        write_csv(output_root / "cross_substrate_summary.csv", cross_rows)
    write_json(output_root / "paper_suite_metrics_summary.json", overview)
    log_event(
        f"posthoc done n_cross_rows={len(cross_rows)} summary={output_root / 'paper_suite_metrics_summary.json'}",
        component="posthoc",
    )
    return overview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Posthoc C1/C5/C6 paper-suite metrics from saved artifacts.")
    parser.add_argument("config")
    parser.add_argument("--task", choices=["all", "c1", "c5", "c6"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, task=args.task, smoke=args.smoke, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
