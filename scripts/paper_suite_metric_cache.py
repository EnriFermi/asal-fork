from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_CACHE_VERSION = "mspd_metric_cache_v2_floor_reconstruction"
METRIC_CODE_VERSION = "mspd_avg_floor_reconstruction_msc_v2"


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metric_config_hash(metric_cfg: dict[str, Any]) -> str:
    return sha256_text(stable_json(metric_cfg))


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def chunks_input_identity(apf_dir: Path, chunks: list[tuple[Path, int, int, int]]) -> dict[str, Any]:
    return {
        "kind": "apf_chunks",
        "apf_dir": str(apf_dir.resolve()),
        "chunks": [
            {
                **file_identity(Path(path)),
                "start_step": int(start),
                "end_step": int(end),
                "chunk_index": int(idx),
            }
            for path, start, end, idx in chunks
        ],
    }


def build_metric_cache_metadata(metric_cfg: dict[str, Any], input_identity: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_json = stable_json(metric_cfg)
    identity_json = stable_json(input_identity)
    out: dict[str, Any] = {
        "metric_cache_version": METRIC_CACHE_VERSION,
        "metric_code_version": METRIC_CODE_VERSION,
        "metric_config_json": cfg_json,
        "metric_config_hash": sha256_text(cfg_json),
        "metric_input_identity_json": identity_json,
        "metric_input_identity_hash": sha256_text(identity_json),
    }
    if extra:
        for key, value in extra.items():
            out[str(key)] = _plain(value)
    return out


def metadata_npz_payload(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            value = stable_json(value)
        payload[str(key)] = np.asarray(str(value))
    return payload


def _npz_str(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def expected_metric_metadata(metric_cfg: dict[str, Any], input_identity: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_metric_cache_metadata(metric_cfg, input_identity, extra=extra)


def compare_metrics_npz_metadata(path: Path, metric_cfg: dict[str, Any], input_identity: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    expected = expected_metric_metadata(metric_cfg, input_identity)
    try:
        with np.load(path, allow_pickle=False) as data:
            found_version = _npz_str(data, "metric_cache_version")
            found_code_version = _npz_str(data, "metric_code_version")
            found_cfg_hash = _npz_str(data, "metric_config_hash")
            found_identity_hash = _npz_str(data, "metric_input_identity_hash")
    except Exception as exc:
        return False, f"cannot read metrics cache metadata: {type(exc).__name__}: {exc}", expected

    missing = [
        key
        for key, value in (
            ("metric_cache_version", found_version),
            ("metric_code_version", found_code_version),
            ("metric_config_hash", found_cfg_hash),
            ("metric_input_identity_hash", found_identity_hash),
        )
        if value is None
    ]
    if missing:
        return False, f"missing metric cache metadata keys: {missing}", expected
    if found_version != METRIC_CACHE_VERSION:
        return False, f"metric cache version mismatch: found {found_version}, expected {METRIC_CACHE_VERSION}", expected
    if found_code_version != METRIC_CODE_VERSION:
        return False, f"metric code version mismatch: found {found_code_version}, expected {METRIC_CODE_VERSION}", expected
    if found_cfg_hash != expected["metric_config_hash"]:
        return False, f"metric config hash mismatch: found {found_cfg_hash}, expected {expected['metric_config_hash']}", expected
    if found_identity_hash != expected["metric_input_identity_hash"]:
        return (
            False,
            f"metric input identity hash mismatch: found {found_identity_hash}, expected {expected['metric_input_identity_hash']}",
            expected,
        )
    return True, "ok", expected


def require_fresh_metrics_npz(path: Path, metric_cfg: dict[str, Any], input_identity: dict[str, Any], *, context: str) -> dict[str, Any]:
    ok, reason, expected = compare_metrics_npz_metadata(path, metric_cfg, input_identity)
    if not ok:
        raise ValueError(f"{context}: stale or invalid metrics cache {path}: {reason}. Recompute the metrics layer with --force.")
    return expected
