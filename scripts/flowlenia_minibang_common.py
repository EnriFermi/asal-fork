from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


CHUNK_RE = re.compile(
    r"P_steps_(\d+)_(\d+)__secs_([0-9.]+)_([0-9.]+)__idx_(\d+)\.npz$"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | os.PathLike[str] | None, base_dir: Path | None = None) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    root = project_root() if base_dir is None else Path(base_dir)
    return (root / path).resolve()


def ensure_env_resolver() -> None:
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key, default=None: os.getenv(key, default))


def _apply_rollout_overrides(cfg):
    rollout = cfg.get("rollout", None)
    if rollout is None:
        return cfg, OmegaConf.create()

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    scalar_rollout = OmegaConf.create()
    section_names = {"meta", "substrate", "simulation", "logging", "metric", "minibang"}
    for key, value in rollout.items():
        if value is None:
            continue
        key_s = str(key)
        if key_s in section_names:
            if cfg.get(key_s, None) is None:
                cfg[key_s] = OmegaConf.create()
            cfg[key_s] = OmegaConf.merge(cfg.get(key_s, {}), value)
        else:
            scalar_rollout[key_s] = value
    return cfg, scalar_rollout


def load_config(config_path: Path, overrides: list[str] | None = None):
    ensure_env_resolver()
    cfg = OmegaConf.load(str(config_path))
    base_config_raw = cfg.get("base_config", None)
    if base_config_raw is not None:
        base_path = Path(str(base_config_raw))
        if not base_path.is_absolute():
            candidate = project_root() / base_path
            if candidate.exists():
                base_path = candidate
            else:
                base_path = config_path.parent / base_path
        if not base_path.exists():
            raise FileNotFoundError(f"base_config not found: {base_config_raw}")
        base_cfg = OmegaConf.load(str(base_path))
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg, scalar_rollout = _apply_rollout_overrides(cfg)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
        scalar_rollout,
        cfg.get("minibang", {}),
    )
    return cfg, flat


def to_plain(obj: Any) -> Any:
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True))
    os.replace(tmp, path)


def list_apf_chunks(apf_dir: Path) -> list[tuple[Path, int, int, int]]:
    chunks: list[tuple[Path, int, int, int]] = []
    for path in Path(apf_dir).iterdir():
        match = CHUNK_RE.match(path.name)
        if match is None:
            continue
        s0, s1, _t0, _t1, idx = match.groups()
        chunks.append((path, int(s0), int(s1), int(idx)))
    chunks.sort(key=lambda x: (x[1], x[3]))
    return chunks


def robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(finite):
        return out
    med = np.nanmedian(arr[finite])
    mad = np.nanmedian(np.abs(arr[finite] - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(arr[finite])
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    out[finite] = (arr[finite] - med) / scale
    return out


def intervals_from_mask(
    steps: np.ndarray,
    mask: np.ndarray,
    *,
    pad_steps: int = 0,
) -> list[tuple[int, int, int, int]]:
    steps_i = np.asarray(steps, dtype=np.int64)
    mask_b = np.asarray(mask, dtype=bool)
    if steps_i.size != mask_b.size:
        raise ValueError("steps and mask must have the same length.")
    intervals: list[tuple[int, int, int, int]] = []
    i = 0
    while i < mask_b.size:
        if not mask_b[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask_b.size and mask_b[j + 1]:
            j += 1
        start = int(steps_i[i]) - int(pad_steps)
        end = int(steps_i[j]) + int(pad_steps)
        intervals.append((max(0, start), max(0, end), i, j))
        i = j + 1
    return intervals


def merge_intervals(
    rows: list[dict[str, Any]],
    *,
    gap_steps: int,
    max_duration_steps: int | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: (int(r["start_step"]), int(r["end_step"])))
    merged: list[dict[str, Any]] = []
    cur = dict(rows_sorted[0])
    cur["reasons"] = list(cur.get("reasons", []))
    for row in rows_sorted[1:]:
        start = int(row["start_step"])
        end = int(row["end_step"])
        merged_end = max(int(cur["end_step"]), end)
        merged_duration = merged_end - int(cur["start_step"])
        can_merge = start <= int(cur["end_step"]) + int(gap_steps)
        if max_duration_steps is not None:
            can_merge = can_merge and merged_duration <= int(max_duration_steps)
        if can_merge:
            cur["end_step"] = merged_end
            cur["score"] = max(float(cur.get("score", 0.0)), float(row.get("score", 0.0)))
            cur["delta_h_z_max"] = max(
                float(cur.get("delta_h_z_max", 0.0)),
                float(row.get("delta_h_z_max", 0.0)),
            )
            cur["mass_shift_z_max"] = max(
                float(cur.get("mass_shift_z_max", 0.0)),
                float(row.get("mass_shift_z_max", 0.0)),
            )
            reasons = set(cur.get("reasons", []))
            reasons.update(row.get("reasons", []))
            cur["reasons"] = sorted(reasons)
        else:
            merged.append(cur)
            cur = dict(row)
            cur["reasons"] = list(cur.get("reasons", []))
    merged.append(cur)
    return merged


def load_frame_times(traj_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = traj_dir / "frame_times.csv"
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return None
    data = np.atleast_1d(data)
    return np.asarray(data["step"], dtype=np.float64), np.asarray(data["video_sec"], dtype=np.float64)


def step_to_video_sec(step: float, frame_times: tuple[np.ndarray, np.ndarray] | None) -> float:
    if frame_times is None:
        return float("nan")
    steps, secs = frame_times
    if steps.size == 0:
        return float("nan")
    return float(np.interp(float(step), steps, secs, left=secs[0], right=secs[-1]))
