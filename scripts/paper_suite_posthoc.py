from __future__ import annotations

import argparse
import json
import os
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
            if "xy_control_a" not in data.files:
                raise KeyError(f"missing xy_control_a; available keys={list(data.files)}")
            xy = np.asarray(data["xy_control_a"], dtype=np.float32)
            sample_every, rollout_steps = _infer_lagrangian_timing(data, xy)
    except Exception as exc:
        raise ValueError(f"invalid lagrangian npz used for metric config path={path} {_file_probe(path)} error={type(exc).__name__}: {exc}") from exc
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
        metric_alpha=float(cfg_get(metric_cfg_raw, "metric_alpha", 0.0)),
        metric_beta=float(cfg_get(metric_cfg_raw, "metric_beta", 1.0)),
        metric_eps=float(cfg_get(metric_cfg_raw, "metric_eps", 1e-12)),
    )
    return resolve_metric_config(args)


def _score_maps(metric_eval, metric_seed: int, xy: np.ndarray) -> dict[str, np.ndarray]:
    _loss, info = metric_eval(jax.random.PRNGKey(int(metric_seed)), jnp.asarray(xy, dtype=jnp.float32))
    return {key: np.asarray(jax.device_get(value)) for key, value in info.items()}


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
            metrics_path = _manifest_path(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            candidate_kind = _canonicalize_kind(row.get("candidate_kind", row.get("candidate_label", "optimized")), row.get("candidate_label", None))
            items.append(
                {
                    "traj_id": traj_id,
                    "optimized_run_idx": int(row.get("suite_run_idx", row.get("selection_idx", idx))),
                    "source_optimized_run_idx": int(row.get("source_run_idx", row.get("suite_run_idx", idx))),
                    "source_root_rank": int(row.get("source_root_rank", -1)),
                    "source_root": str(row.get("source_root", "")),
                    "source_root_name": str(row.get("source_root_name", "")),
                    "run_seed": int(row.get("run_seed", -1)),
                    "candidate_kind": candidate_kind,
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "candidate_label": str(row.get("candidate_label", candidate_kind)),
                    "traj_dir": traj_dir,
                    "metrics_path": metrics_path,
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
                "optimized_run_idx": idx,
                "source_optimized_run_idx": idx,
                "source_root_rank": -1,
                "source_root": "",
                "source_root_name": "",
                "run_seed": -1,
                "candidate_kind": _canonicalize_kind(traj_dir.name, traj_dir.name),
                "candidate_idx": 0,
                "candidate_label": traj_dir.name,
                "traj_dir": traj_dir,
                "metrics_path": metrics_path,
            }
        )
    return items


def _compute_c1_from_apf_metrics(
    dataset_name: str,
    ds_cfg: Any,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c1_cfg = cfg_get(ds_cfg, "c1", {})
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
        except Exception as exc:
            raise ValueError(f"{dataset_name}: invalid C1 APF metrics path={metrics_path} {_file_probe(metrics_path)} error={type(exc).__name__}: {exc}") from exc

        if selected_idx < 0 or selected_idx >= int(score_by_tau.shape[0]):
            selected_idx = int(np.nanargmax(score_by_tau))
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

        trial_uid = str(item["traj_id"])
        maps_path = maps_dir / f"{trial_uid}_c1_maps.npz"
        np.savez_compressed(
            maps_path,
            delta_h_selection=delta_h_map,
            delta_h_eval=delta_h_map,
            tau_steps=tau_steps,
            window_start_steps=window_start_steps.astype(np.int64),
            window_end_steps=window_end_steps.astype(np.int64),
            selection_score_by_tau=score_by_tau,
            eval_score_by_tau=score_by_tau,
            selection_amp_by_tau=amp_by_tau,
            eval_amp_by_tau=amp_by_tau,
            selection_msc_by_tau=msc_by_tau,
            eval_msc_by_tau=msc_by_tau,
            selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
            source_metrics_path=str(metrics_path),
        )
        eval_values = delta_h_map[selected_idx]
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
                "selection_score_mspd": float(score_by_tau[selected_idx]),
                "selection_amp": float(amp_by_tau[selected_idx]),
                "selection_msc": float(msc_by_tau[selected_idx]),
                "eval_score_mspd": float(score_by_tau[selected_idx]),
                "eval_amp": float(amp_by_tau[selected_idx]),
                "eval_msc": float(msc_by_tau[selected_idx]),
                "eval_delta_h_mean": float(np.nanmean(eval_values)),
                "eval_delta_h_median": float(np.nanmedian(eval_values)),
                "eval_delta_h_std": float(np.nanstd(eval_values)),
                "window_start_min_steps": int(np.nanmin(window_start_steps)),
                "window_end_max_steps": int(np.nanmax(window_end_steps)),
                "metrics_path": str(metrics_path),
                "maps_path": str(maps_path),
                "c1_source": "apf_metrics",
                "c1_eval_mode": "single_arun_trajectory",
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
    metric_cfg_raw = cfg_get(cfg_get(ds_cfg, "c1", {}), "metric", {})
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
        arrays = _load_npz_arrays(
            lag_path,
            ["xy_control_a", "xy_control_b"],
            context=f"{dataset_name}: C1 trial_idx={row.get('trial_idx', 'na')} row={idx}/{total}",
        )
        xy_sel = np.asarray(arrays["xy_control_a"], dtype=np.float32)
        xy_eval = np.asarray(arrays["xy_control_b"], dtype=np.float32)
        metric_seed = int(row.get("metric_seed", row.get("seed_x", 0) + 10_000_000))
        sel_info = _score_maps(metric_eval, metric_seed, xy_sel)
        eval_info = _score_maps(metric_eval, metric_seed + 1, xy_eval)
        sel_map = np.asarray(sel_info["delta_h_map"], dtype=np.float64)
        eval_map = np.asarray(eval_info["delta_h_map"], dtype=np.float64)
        sel_score_by_tau = np.asarray(sel_info["score_by_tau"], dtype=np.float64).reshape(-1)
        eval_score_by_tau = np.asarray(eval_info["score_by_tau"], dtype=np.float64).reshape(-1)
        sel_amp_by_tau = np.asarray(sel_info["amp_by_tau"], dtype=np.float64).reshape(-1)
        eval_amp_by_tau = np.asarray(eval_info["amp_by_tau"], dtype=np.float64).reshape(-1)
        sel_msc_by_tau = np.asarray(sel_info["msc_by_tau"], dtype=np.float64).reshape(-1)
        eval_msc_by_tau = np.asarray(eval_info["msc_by_tau"], dtype=np.float64).reshape(-1)
        selected_idx = int(np.nanargmax(sel_score_by_tau))
        tau_steps = np.asarray(sel_info["tau_steps"], dtype=np.int32)
        eval_values = eval_map[selected_idx]
        eval_score_mspd = float(eval_score_by_tau[selected_idx])
        trial_uid = str(row.get("trial_uid", f"trial_{int(row['trial_idx']):05d}"))
        maps_path = maps_dir / f"{trial_uid}_c1_maps.npz"
        np.savez_compressed(
            maps_path,
            delta_h_selection=sel_map,
            delta_h_eval=eval_map,
            tau_steps=tau_steps,
            window_start_steps=np.asarray(sel_info["window_start_steps"], dtype=np.int32),
            selection_score_by_tau=sel_score_by_tau,
            eval_score_by_tau=eval_score_by_tau,
            selection_amp_by_tau=sel_amp_by_tau,
            eval_amp_by_tau=eval_amp_by_tau,
            selection_msc_by_tau=sel_msc_by_tau,
            eval_msc_by_tau=eval_msc_by_tau,
            selected_tau_idx=np.asarray(selected_idx, dtype=np.int32),
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
                "selected_tau_idx": selected_idx,
                "selected_tau_steps": int(tau_steps[selected_idx]),
                "selection_score_mspd": float(sel_score_by_tau[selected_idx]),
                "selection_amp": float(sel_amp_by_tau[selected_idx]),
                "selection_msc": float(sel_msc_by_tau[selected_idx]),
                "eval_score_mspd": eval_score_mspd,
                "eval_amp": float(eval_amp_by_tau[selected_idx]),
                "eval_msc": float(eval_msc_by_tau[selected_idx]),
                "eval_delta_h_mean": float(np.nanmean(eval_values)),
                "eval_delta_h_median": float(np.nanmedian(eval_values)),
                "eval_delta_h_std": float(np.nanstd(eval_values)),
                "maps_path": str(maps_path),
            }
        )
    score_df = pd.DataFrame(score_rows)
    contrast_df = _group_contrasts(score_df, "eval_score_mspd")
    score_df.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    contrast_df.to_csv(output_dir / "group_contrasts.csv", index=False)
    log_event(
        f"{dataset_name}: C1 done n_scores={len(score_df)} n_contrasts={len(contrast_df)} output={output_dir}",
        component="posthoc",
    )
    return score_df, contrast_df, sign_test_greater(contrast_df["delta_vs_random_median"].tolist())


def _group_contrasts(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for group_idx, group in frame.groupby("optimized_run_idx"):
        opt = group[group["candidate_kind"] == "optimized"]
        randoms = group[group["candidate_kind"] == "random"]
        if opt.empty or randoms.empty:
            continue
        opt_value = safe_float(opt.iloc[0][metric])
        rand_values = [safe_float(v) for v in randoms[metric]]
        rand_median = nanmedian(rand_values)
        rows.append(
            {
                "optimized_run_idx": int(group_idx),
                f"{metric}__optimized": opt_value,
                f"{metric}__random_median": rand_median,
                "n_random": int(len(rand_values)),
                "delta_vs_random_median": float(opt_value - rand_median),
            }
        )
    return pd.DataFrame(rows)


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
            "metric_alpha": 0.0,
            "metric_beta": 1.0,
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
        c1_apf_source = _c1_source_mode(ds) in {"apf", "apf_metrics", "arun", "arun_lagrangian_apf"}
        needs_frustration_rows = task in {"all", "c5", "c6"} or (task in {"c1"} and not c1_apf_source)
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
                if c1_apf_source:
                    _scores, contrasts, c1_summary = _compute_c1_from_apf_metrics(dataset_name, ds, ds_out, force=force)
                else:
                    _scores, contrasts, c1_summary = _compute_c1(dataset_name, rows, ds, ds_out, force=force)
                status["c1"] = c1_summary
                cross_rows.append({"dataset": dataset_name, "claim": "C1/C6", "metric": "eval_score_mspd", **c1_summary})
            if task in {"all", "c5", "c6"}:
                _run_df, _summary_df, c5_summary = _compute_c5(dataset_name, rows, ds, ds_out)
                status["c5"] = c5_summary
                if c5_summary:
                    cross_rows.append({"dataset": dataset_name, "claim": "C5/C6", "metric": c5_summary.get("metric", "frustration"), **c5_summary})
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
