from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]


class _NullProgress:
    def __init__(self, iterable=None):
        self._iterable = iterable

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


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


def progress(iterable=None, *, total: int | None = None, desc: str | None = None, enabled: bool = True, leave: bool = False):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)
    return _NullProgress(iterable=iterable)


def progress_bar(*, total: int | None = None, desc: str | None = None, enabled: bool = True, leave: bool = False):
    if enabled and _tqdm is not None:
        return _tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True)
    return _NullProgress()


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
