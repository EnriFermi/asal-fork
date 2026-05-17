from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_env_resolver() -> None:
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key, default=None: os.getenv(key, default))


def load_config(path_like: str | Path, *, smoke: bool = False) -> tuple[DictConfig, Path]:
    ensure_env_resolver()
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = OmegaConf.load(path)
    if smoke and cfg.get("smoke") is not None:
        cfg = OmegaConf.merge(cfg, cfg.get("smoke", {}))
    return cfg, path


def resolve_path(path_like: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return (REPO_ROOT if base_dir is None else base_dir) / path


def ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def write_json(path_like: str | Path, payload: Any) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path_like: str | Path) -> dict[str, Any]:
    with Path(path_like).open("r") as f:
        return json.load(f)


def write_csv(path_like: str | Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path = Path(path_like)
    rows_l = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows_l:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_l:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path_like: str | Path) -> list[dict[str, str]]:
    with Path(path_like).open("r", newline="") as f:
        return list(csv.DictReader(f))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if OmegaConf.is_list(value):
        return list(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def dataset_items(cfg: DictConfig) -> list[tuple[str, DictConfig]]:
    datasets = cfg.get("datasets", {})
    out = []
    for name in datasets.keys():
        ds = datasets.get(name)
        if bool(ds.get("enabled", True)):
            out.append((str(name), ds))
    return out


def command_to_str(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
    print(f"[paper-suite] command: {command_to_str(cmd)}")
    if dry_run:
        return 0
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return 0


def current_python() -> str:
    return sys.executable


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def nanmedian(values: Iterable[Any]) -> float:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def sign_test_greater(values: Iterable[Any]) -> dict[str, Any]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[np.abs(arr) > 1e-12]
    n = int(nonzero.size)
    k = int(np.sum(nonzero > 0))
    p = float("nan")
    if n > 0:
        try:
            from scipy import stats as scipy_stats

            p = float(scipy_stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
        except Exception:
            p = float(sum(_binom_pmf(n, i) for i in range(k, n + 1)))
    return {
        "n": int(arr.size),
        "n_nonzero": n,
        "n_positive": k,
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "sign_test_greater_p": p,
    }


def _binom_pmf(n: int, k: int) -> float:
    import math

    return math.comb(n, k) * (0.5 ** n)

