from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_like: str | Path | None, root: str | Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return Path(root) / path


def load_yaml(path_like: str | Path) -> DictConfig:
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key, default=None: os.getenv(key, default))
    return OmegaConf.load(str(path_like))


def load_paper_check_config(path_like: str | Path) -> tuple[DictConfig, Path]:
    config_path = Path(path_like)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    cfg = load_yaml(config_path)
    return cfg, config_path


def load_stage_base_config(section_cfg: DictConfig, config_dir: str | Path) -> tuple[DictConfig, Path]:
    base_path = resolve_path(section_cfg.get("base_config"), config_dir)
    if base_path is None:
        raise ValueError("stage section must define base_config.")
    base_cfg = load_yaml(base_path)
    overrides = section_cfg.get("overrides")
    if overrides is None:
        return base_cfg, base_path
    return OmegaConf.merge(base_cfg, overrides), base_path


def validate_machine_config(cfg: DictConfig) -> tuple[int, int]:
    meta = cfg.get("meta", {})
    machine_idx = int(meta.get("machine_idx", 0))
    num_machines = int(meta.get("num_machines", 1))
    if num_machines < 1:
        raise ValueError(f"meta.num_machines must be >= 1, got {num_machines}.")
    if machine_idx < 0 or machine_idx >= num_machines:
        raise ValueError(
            f"meta.machine_idx must satisfy 0 <= machine_idx < num_machines, got "
            f"machine_idx={machine_idx}, num_machines={num_machines}."
        )
    return machine_idx, num_machines


def shard_indices(total: int, machine_idx: int, num_machines: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}.")
    return list(range(int(machine_idx), int(total), int(num_machines)))


def write_resolved_yaml(path_like: str | Path, cfg: DictConfig) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def as_int(cfg: DictConfig | dict, key: str, default: int) -> int:
    if cfg is None:
        return int(default)
    value = cfg.get(key, default)
    return int(default if value is None else value)


def sorted_existing_files(paths: Iterable[Path]) -> list[Path]:
    return sorted(path for path in paths if path.exists())
