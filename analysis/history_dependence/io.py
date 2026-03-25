from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import REPO_ROOT, resolve_config_path, resolve_path


@dataclass
class RunCollection:
    runs: pd.DataFrame
    source_summary: dict[str, Any]
    metric_summary: dict[str, Any] | None
    source_dir: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _read_trial_rows(eval_dir: Path) -> pd.DataFrame:
    trial_csv = eval_dir / "trial_results.csv"
    if trial_csv.exists():
        frame = pd.read_csv(trial_csv)
        if "trial_idx" not in frame.columns:
            raise ValueError(f"{trial_csv} does not contain 'trial_idx'.")
        return frame.sort_values("trial_idx").reset_index(drop=True)

    trial_dir = eval_dir / "trial_data"
    if not trial_dir.exists():
        raise FileNotFoundError(
            f"Could not find either {trial_csv} or {trial_dir}."
        )

    rows = []
    for trial_json in sorted(trial_dir.glob("trial_*.json")):
        rows.append(_load_json(trial_json))
    if not rows:
        raise FileNotFoundError(f"No trial JSON files found under {trial_dir}.")
    return pd.DataFrame(rows).sort_values("trial_idx").reset_index(drop=True)


def _resolve_optional_file(path_value: Any, root: Path) -> str | None:
    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return None
    path = resolve_path(str(path_value), root)
    return None if path is None else str(path)


def _runs_from_history_dependence_eval(source_cfg: dict[str, Any], config_dir: Path) -> RunCollection:
    eval_dir = resolve_config_path(source_cfg.get("path"), config_dir)
    if eval_dir is None:
        raise ValueError("source.path must be set for source.type=history_dependence_eval.")
    if not eval_dir.exists():
        raise FileNotFoundError(f"History-dependence evaluation directory does not exist: {eval_dir}")

    source_summary_path = eval_dir / "summary.json"
    metric_summary_path = eval_dir / "msc_metric_summary.json"
    source_summary = _load_json(source_summary_path) if source_summary_path.exists() else {}
    metric_summary = _load_json(metric_summary_path) if metric_summary_path.exists() else None
    trials = _read_trial_rows(eval_dir)

    run_rows: list[dict[str, Any]] = []
    variants = (
        ("control_a", "free", "z_control_a", "xy_control_a"),
        ("control_b", "free", "z_control_b", "xy_control_b"),
        ("walls", "wall", "z_walls", "xy_walls"),
    )

    for trial in trials.to_dict(orient="records"):
        trial_idx = int(trial["trial_idx"])
        pair_group = f"trial_{trial_idx:05d}"
        embeddings_path = _resolve_optional_file(trial.get("embeddings_path"), eval_dir)
        lagrangian_path = _resolve_optional_file(trial.get("lagrangian_path"), eval_dir)
        for variant, condition, embeddings_key, lagrangian_key in variants:
            run_rows.append(
                {
                    "run_id": f"{pair_group}__{variant}",
                    "run_label": f"{pair_group} {variant}",
                    "pair_group_id": pair_group,
                    "trial_idx": trial_idx,
                    "variant": variant,
                    "condition": condition,
                    "embeddings_path": embeddings_path,
                    "embeddings_key": embeddings_key,
                    "lagrangian_path": lagrangian_path,
                    "lagrangian_key": lagrangian_key,
                    "frame_path": None,
                    "has_embeddings": bool(embeddings_path),
                    "has_lagrangian": bool(lagrangian_path),
                    "source_type": "history_dependence_eval",
                }
            )

    runs = pd.DataFrame(run_rows).sort_values(["condition", "trial_idx", "variant"]).reset_index(drop=True)
    return RunCollection(
        runs=runs,
        source_summary=source_summary,
        metric_summary=metric_summary,
        source_dir=eval_dir,
    )


def _runs_from_manifest(source_cfg: dict[str, Any], config_dir: Path) -> RunCollection:
    runs_raw = source_cfg.get("runs")
    if not runs_raw:
        raise ValueError("source.runs must be provided for source.type=run_manifest.")

    run_rows = []
    for idx, item in enumerate(runs_raw):
        run_id = str(item.get("run_id") or item.get("id") or f"run_{idx:03d}")
        condition = str(item.get("condition", "")).strip().lower()
        if condition not in {"free", "wall"}:
            raise ValueError(f"Run {run_id} has invalid condition={condition!r}; expected 'free' or 'wall'.")
        embeddings_path = resolve_config_path(item.get("embeddings_path"), config_dir)
        lagrangian_path = resolve_config_path(item.get("lagrangian_path"), config_dir)
        frame_path = resolve_config_path(item.get("frame_path"), config_dir)
        run_rows.append(
            {
                "run_id": run_id,
                "run_label": str(item.get("run_label", run_id)),
                "pair_group_id": str(item.get("pair_group_id", run_id)),
                "trial_idx": int(item.get("trial_idx", idx)),
                "variant": str(item.get("variant", condition)),
                "condition": condition,
                "embeddings_path": None if embeddings_path is None else str(embeddings_path),
                "embeddings_key": str(item.get("embeddings_key", "embeddings")),
                "lagrangian_path": None if lagrangian_path is None else str(lagrangian_path),
                "lagrangian_key": str(item.get("lagrangian_key", "xy")),
                "frame_path": None if frame_path is None else str(frame_path),
                "has_embeddings": embeddings_path is not None,
                "has_lagrangian": lagrangian_path is not None,
                "source_type": "run_manifest",
            }
        )

    runs = pd.DataFrame(run_rows).sort_values(["condition", "run_id"]).reset_index(drop=True)
    return RunCollection(
        runs=runs,
        source_summary={},
        metric_summary=None,
        source_dir=resolve_config_path(source_cfg.get("base_dir"), config_dir) or config_dir,
    )


def build_run_collection(cfg: dict[str, Any]) -> RunCollection:
    config_dir = Path(cfg["_config_dir"])
    source_cfg = dict(cfg.get("source", {}))
    source_type = str(source_cfg.get("type", "history_dependence_eval")).strip().lower()

    if source_type == "history_dependence_eval":
        return _runs_from_history_dependence_eval(source_cfg, config_dir)
    if source_type == "run_manifest":
        return _runs_from_manifest(source_cfg, config_dir)
    raise ValueError(f"Unsupported source.type={source_type!r}.")


@lru_cache(maxsize=None)
def _load_npz_key(path: str, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain array {key!r}; available keys: {sorted(data.files)}")
        return np.asarray(data[key])


def load_embeddings(run_row: pd.Series | dict[str, Any]) -> np.ndarray:
    row = dict(run_row)
    path = row.get("embeddings_path")
    key = row.get("embeddings_key")
    if not path:
        raise FileNotFoundError(f"Run {row.get('run_id')} does not have embeddings_path.")
    return np.asarray(_load_npz_key(str(path), str(key)), dtype=np.float64)


def load_lagrangian(run_row: pd.Series | dict[str, Any]) -> np.ndarray:
    row = dict(run_row)
    path = row.get("lagrangian_path")
    key = row.get("lagrangian_key")
    if not path:
        raise FileNotFoundError(f"Run {row.get('run_id')} does not have lagrangian_path.")
    return np.asarray(_load_npz_key(str(path), str(key)), dtype=np.float64)


def load_optional_npz_scalar(path: str | None, key: str) -> int | float | None:
    if not path:
        return None
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        arr = np.asarray(data[key])
        return arr.item() if arr.shape == () else None


def infer_lagrangian_metadata(run_collection: RunCollection) -> dict[str, Any]:
    for row in run_collection.runs.to_dict(orient="records"):
        lag_path = row.get("lagrangian_path")
        if not lag_path:
            continue
        with np.load(lag_path, allow_pickle=False) as data:
            meta = {}
            for key in (
                "sample_every_steps",
                "trajectory_start_steps",
                "trajectory_end_steps",
                "trajectory_window_steps",
                "metric_window_size_steps",
                "metric_window_step_steps",
                "metric_tau_steps",
            ):
                if key in data.files:
                    arr = np.asarray(data[key])
                    if arr.shape == ():
                        meta[key] = arr.item()
            xy_key = str(row["lagrangian_key"])
            if xy_key in data.files:
                meta["time_sampling"] = int(np.asarray(data[xy_key]).shape[0])
            if meta:
                return meta
    return {}
