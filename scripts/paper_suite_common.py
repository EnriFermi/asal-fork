from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
_SUBPROCESS_COUNTER = 0


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


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value[:96] or "command"


def _default_log_dir() -> Path:
    raw = os.environ.get("PAPER_SUITE_LOG_DIR")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return ensure_dir(path)
    return ensure_dir(REPO_ROOT / "analysis" / "results" / "paper_suite" / "logs")


def log_event(message: str, *, component: str = "paper-suite") -> None:
    line = f"{_timestamp()} [{component}] {message}"
    print(line, flush=True)
    master = os.environ.get("PAPER_SUITE_MASTER_LOG")
    if not master:
        return
    try:
        path = Path(master)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def init_suite_logging(config_path: str | Path, *, smoke: bool = False, layer: str = "all", task: str = "all") -> Path:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or (
        REPO_ROOT / "analysis" / "results" / "paper_suite"
    )
    log_dir = ensure_dir(output_root / "logs")
    run_id = os.environ.get("PAPER_SUITE_RUN_ID") or _run_stamp()
    master_log = log_dir / f"{run_id}_master.log"
    os.environ["PAPER_SUITE_RUN_ID"] = run_id
    os.environ["PAPER_SUITE_LOG_DIR"] = str(log_dir)
    os.environ["PAPER_SUITE_MASTER_LOG"] = str(master_log)
    os.environ.setdefault("PAPER_SUITE_LOG_PROGRESS", "plain")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    log_event(
        f"logging initialized run_id={run_id} layer={layer} task={task} log_dir={log_dir}",
        component="runner",
    )
    return master_log


def _command_log_path(cmd: list[str]) -> Path:
    global _SUBPROCESS_COUNTER
    _SUBPROCESS_COUNTER += 1
    run_id = os.environ.get("PAPER_SUITE_RUN_ID") or _run_stamp()
    script = next((Path(str(part)).name for part in cmd if str(part).endswith(".py")), Path(str(cmd[0])).name if cmd else "command")
    name = _safe_name(script.replace(".py", ""))
    return _default_log_dir() / f"{run_id}_{os.getpid()}_{_SUBPROCESS_COUNTER:03d}_{name}.log"


def run_subprocess(cmd: list[str], *, dry_run: bool = False) -> int:
    log_path = _command_log_path(cmd)
    log_event(f"command start log={log_path} cmd={command_to_str(cmd)}")
    if dry_run:
        log_path.write_text(f"{_timestamp()} [dry-run] {command_to_str(cmd)}\n")
        log_event(f"command dry-run log={log_path}")
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("ab") as log_f:
        log_f.write(f"{_timestamp()} [command] {command_to_str(cmd)}\n".encode("utf-8"))
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=env,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_f.write(chunk)
            log_f.flush()
        returncode = proc.wait()
        log_f.write(f"{_timestamp()} [exit] returncode={returncode}\n".encode("utf-8"))
        log_f.flush()
    if returncode != 0:
        log_event(f"command failed returncode={returncode} log={log_path}")
        raise subprocess.CalledProcessError(returncode, cmd)
    log_event(f"command done log={log_path}")
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
