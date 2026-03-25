from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path | None, root: Path | None = None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root or REPO_ROOT) / path


def resolve_config_path(path_like: str | Path | None, config_dir: Path | None = None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {".", ".."}:
        return (config_dir or REPO_ROOT) / path
    return REPO_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "item"


def pair_type(condition_a: str, condition_b: str) -> str:
    pair = tuple(sorted((str(condition_a), str(condition_b))))
    if pair == ("free", "free"):
        return "free-free"
    if pair == ("wall", "wall"):
        return "wall-wall"
    return "free-wall"


def write_json(path: str | Path, payload: Any) -> None:
    def _jsonify(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonify(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(_jsonify(payload), f, indent=2)


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)


def save_matrix(path: str | Path, matrix: pd.DataFrame) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out)


def save_npz(path: str | Path, **payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        np.savez_compressed(f, **payload)


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    arr = np.asarray(matrix.to_numpy(dtype=np.float64))
    tri = np.triu_indices_from(arr, k=1)
    return arr[tri]
