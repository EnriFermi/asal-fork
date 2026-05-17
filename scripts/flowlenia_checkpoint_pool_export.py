from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flowlenia_minibang_common import list_apf_chunks, load_config, resolve_path, to_plain, write_json


SCHEMA_VERSION = "flow_lenia_checkpoint_pool_v1"
CHECKPOINT_FILE_SCHEMA = "flow_lenia_simulation_state_npz_v1"
CHECKPOINT_KIND = "simulation_state"

SOURCE_REQUIRED_KEYS = (
    "A",
    "P",
    "F",
    "lagrangian_xy",
    "lagrangian_c",
    "resume_batch_rng_key",
    "resume_batch_size",
    "resume_batch_index",
    "resume_selection0",
    "resume_jit_microbatch",
    "resume_snapshot_interval",
    "resume_seed",
    "resume_lagrangian_seed",
    "state_t",
    "state_mass_cycle_start",
)

EXPORTED_REQUIRED_KEYS = SOURCE_REQUIRED_KEYS + (
    "params",
    "config_yaml",
    "flat_config_json",
    "metadata_json",
    "checkpoint_file_schema",
    "checkpoint_kind",
    "case_id",
    "checkpoint_id",
    "step",
)

FLOAT16_STATE_KEYS = {
    "A",
    "P",
    "F",
    "lagrangian_xy",
    "params",
    "fps",
    "state_mass_cycle_start",
}

SUBSTRATE_CONFIG_KEYS = (
    "substrate",
    "grid_size",
    "C",
    "k",
    "kernel_components",
    "M",
    "dd",
    "dt",
    "sigma",
    "border",
    "mix_rule",
    "base_seed",
    "seed_patch_size",
    "seed_n_patches",
    "seed_mode",
    "p_constant_per_patch",
    "render_mode",
    "clip1",
    "clip2",
    "mutations",
    "mutation_sz",
    "mutation_p",
    "mutation_scale",
    "optimize_mutation_scale",
    "volcano",
    "volcano_sz",
    "volcano_p",
    "volcano_delta",
    "food",
    "mass_decay",
    "mass_clip_eps",
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _param_hash(params: np.ndarray) -> str:
    arr = np.asarray(params, dtype=np.float16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _default_case_id(traj_id: str, *, mode: str) -> str:
    if mode == "traj":
        return str(traj_id)
    m = re.match(r"^(?:traj|case)_(\d+)$", str(traj_id))
    if m:
        return f"case_{int(m.group(1)):05d}"
    return str(traj_id)


def _remote_join(base: str, rel: str) -> str:
    base = str(base).rstrip("/")
    rel = str(rel).lstrip("/")
    return f"{base}/{rel}"


def _run_json(cmd: list[str]) -> Any:
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = proc.stdout.strip()
    return json.loads(text) if text else None


def _rclone_file_id(rclone: str, remote_path: str) -> str | None:
    cmds = [
        [rclone, "lsjson", remote_path, "--files-only", "--hash", "--hash-type", "SHA-256"],
    ]
    if "/" in remote_path:
        parent, name = remote_path.rsplit("/", 1)
        cmds.append([rclone, "lsjson", parent, "--files-only", "--hash", "--hash-type", "SHA-256"])
    else:
        name = remote_path

    for cmd in cmds:
        try:
            payload = _run_json(cmd)
        except Exception:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_name = str(item.get("Name", item.get("Path", "")))
            item_path = str(item.get("Path", ""))
            if item_name == name or item_path == name or len(items) == 1:
                file_id = item.get("ID") or item.get("Id") or item.get("id")
                if file_id:
                    return str(file_id)
    return None


def _upload_with_rclone(
    local_path: Path,
    *,
    remote_base: str,
    remote_relpath: str,
    rclone: str,
    dry_run: bool,
    allow_path_uri: bool,
) -> dict[str, Any]:
    remote_path = _remote_join(remote_base, remote_relpath)
    cmd = [rclone, "copyto", str(local_path), remote_path]
    if dry_run:
        return {
            "uri": f"rclone://{remote_path}",
            "remote_path": remote_path,
            "gdrive_file_id": None,
            "dry_run": True,
        }

    subprocess.run(cmd, check=True)
    file_id = _rclone_file_id(rclone, remote_path)
    if file_id:
        uri = f"gdrive://{file_id}"
    elif allow_path_uri:
        uri = f"rclone://{remote_path}"
    else:
        raise RuntimeError(
            f"Uploaded {local_path} to {remote_path}, but rclone did not expose a Drive file ID. "
            "Pass --allow-rclone-uri to write rclone:// paths instead."
        )
    return {
        "uri": uri,
        "remote_path": remote_path,
        "gdrive_file_id": file_id,
        "dry_run": False,
    }


def _slice_npz_value(data: Any, key: str, idx: int, n_steps: int) -> np.ndarray:
    arr = np.asarray(data[key])
    if arr.ndim > 0 and arr.shape[0] == n_steps:
        arr = arr[idx]
    return np.asarray(arr)


def _to_pool_array(key: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if key in FLOAT16_STATE_KEYS or np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float16, copy=False)
    return arr


def _source_scalar(snapshot: dict[str, np.ndarray], key: str, default: Any) -> Any:
    if key not in snapshot:
        return default
    arr = np.asarray(snapshot[key])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _validate_source(data: Any, source_path: Path) -> None:
    missing = [key for key in SOURCE_REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(
            f"{source_path} is not checkpoint-pool-ready; missing keys: {missing}. "
            "Regenerate APF logs with the updated minibang simulation pipeline."
        )


def _validate_exported_npz(path: Path) -> None:
    with np.load(path) as data:
        missing = [key for key in EXPORTED_REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"Exported checkpoint {path} is missing required keys: {missing}")
        for key in FLOAT16_STATE_KEYS:
            if key in data.files and np.asarray(data[key]).dtype != np.float16:
                raise ValueError(f"{path} key {key!r} must be float16, got {np.asarray(data[key]).dtype}.")


def _selected_indices(steps: np.ndarray, *, start_step: int, stride_steps: int, end_step: int | None) -> np.ndarray:
    steps_i = np.asarray(steps, dtype=np.int64)
    mask = steps_i >= int(start_step)
    if end_step is not None:
        mask &= steps_i <= int(end_step)
    mask &= ((steps_i - int(start_step)) % int(stride_steps)) == 0
    return np.flatnonzero(mask)


def _case_dirs_from_manifest(dataset_root: Path, *, case_id_mode: str) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.json"
    rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        rows = list(manifest.get("trajectories", []))

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        traj_id = str(row.get("traj_id") or row.get("case_id") or "")
        traj_dir_raw = row.get("traj_dir")
        traj_dir = resolve_path(traj_dir_raw) if traj_dir_raw else None
        if (traj_dir is None or not traj_dir.exists()) and traj_id:
            candidate = dataset_root / traj_id
            if candidate.exists():
                traj_dir = candidate
        if traj_dir is None or not traj_dir.exists():
            continue
        if not traj_id:
            traj_id = traj_dir.name
        if traj_id in seen:
            continue
        seen.add(traj_id)
        item = dict(row)
        item["traj_id"] = traj_id
        item["case_id"] = str(row.get("case_id") or _default_case_id(traj_id, mode=case_id_mode))
        item["traj_dir"] = str(traj_dir)
        cases.append(item)

    if not cases:
        for traj_dir in sorted(dataset_root.glob("traj_*")):
            if not traj_dir.is_dir():
                continue
            traj_id = traj_dir.name
            cases.append(
                {
                    "traj_id": traj_id,
                    "case_id": _default_case_id(traj_id, mode=case_id_mode),
                    "traj_dir": str(traj_dir),
                }
            )

    cases.sort(key=lambda x: str(x["traj_id"]))
    return cases


def _load_case_static(case: dict[str, Any], *, allow_food_state: bool) -> dict[str, Any]:
    traj_dir = Path(str(case["traj_dir"]))
    config_path = traj_dir / "config.yaml"
    params_path = traj_dir / "params.npy"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml for {traj_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.npy for {traj_dir}")

    cfg, flat = load_config(config_path)
    flat_plain = OmegaConf.to_container(flat, resolve=True)
    food_enabled = _as_bool(flat_plain.get("food", flat_plain.get("food_enabled", False)), False)
    if food_enabled and not allow_food_state:
        raise ValueError(
            f"{traj_dir} has food enabled, but APF checkpoint pool export intentionally does not store Food. "
            "Pass --allow-food-state only if you accept non-exact food resume."
        )
    params = np.asarray(np.load(params_path), dtype=np.float32)
    config_text = config_path.read_text(encoding="utf-8")
    substrate_config = {key: to_plain(flat_plain[key]) for key in SUBSTRATE_CONFIG_KEYS if key in flat_plain}
    return {
        "config_path": config_path,
        "params_path": params_path,
        "config_text": config_text,
        "flat_config": to_plain(flat_plain),
        "substrate_config": substrate_config,
        "params": params,
    }


def _build_payload(
    *,
    data: Any,
    source_path: Path,
    source_index: int,
    n_steps: int,
    case: dict[str, Any],
    static: dict[str, Any],
    step: int,
    checkpoint_id: str,
) -> dict[str, np.ndarray]:
    snapshot: dict[str, np.ndarray] = {}
    for key in SOURCE_REQUIRED_KEYS:
        snapshot[key] = _to_pool_array(key, _slice_npz_value(data, key, source_index, n_steps))
    if "fps" in data.files:
        snapshot["fps"] = _to_pool_array("fps", np.asarray(data["fps"]))

    metadata = {
        "source_apf_chunk": str(source_path),
        "source_apf_index": int(source_index),
        "source_traj_dir": str(case["traj_dir"]),
        "source_config_path": str(static["config_path"]),
        "source_params_path": str(static["params_path"]),
        "traj_id": str(case["traj_id"]),
        "case_id": str(case["case_id"]),
        "checkpoint_id": str(checkpoint_id),
        "step": int(step),
        "saturation": case.get("saturation", case.get("saturation_T", None)),
        "loss": case.get("loss", None),
        "optimization_iter": case.get("iter", case.get("optimization_iter", None)),
        "seed": int(_source_scalar(snapshot, "resume_seed", static["flat_config"].get("seed", 0))),
        "resume_contract": {
            "runner": "scripts/flowlenia_minibang_resume.py",
            "requires_params": True,
            "requires_config": True,
            "rng_key": "resume_batch_rng_key",
            "batch_key_contract": "split rng by original batch size and select original batch index",
            "food_state_saved": False,
            "float_storage": "float16",
        },
    }

    payload: dict[str, np.ndarray] = {
        **snapshot,
        "params": _to_pool_array("params", np.asarray(static["params"])),
        "step": np.asarray(int(step), dtype=np.int64),
        "checkpoint_file_schema": np.asarray(CHECKPOINT_FILE_SCHEMA),
        "checkpoint_kind": np.asarray(CHECKPOINT_KIND),
        "case_id": np.asarray(str(case["case_id"])),
        "traj_id": np.asarray(str(case["traj_id"])),
        "checkpoint_id": np.asarray(str(checkpoint_id)),
        "config_yaml": np.asarray(str(static["config_text"])),
        "flat_config_json": np.asarray(json.dumps(to_plain(static["flat_config"]), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(to_plain(metadata), sort_keys=True)),
    }
    return payload


def _write_checkpoint_npz(path: Path, payload: dict[str, np.ndarray], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    if compress:
        np.savez_compressed(tmp, **payload)
    else:
        np.savez(tmp, **payload)
    os.replace(tmp, path)
    _validate_exported_npz(path)


def _entry_from_checkpoint(
    *,
    path: Path,
    upload: dict[str, Any],
    sha256: str,
    case: dict[str, Any],
    static: dict[str, Any],
    step: int,
    checkpoint_id: str,
    source_path: Path,
    source_index: int,
) -> dict[str, Any]:
    seed = None
    with np.load(path) as data:
        if "resume_seed" in data.files:
            seed = int(np.asarray(data["resume_seed"]).reshape(-1)[0])
    return {
        "checkpoint_id": checkpoint_id,
        "case_id": str(case["case_id"]),
        "traj_id": str(case["traj_id"]),
        "step": int(step),
        "saturation": case.get("saturation", case.get("saturation_T", None)),
        "seed": seed if seed is not None else static["flat_config"].get("seed", None),
        "uri": upload["uri"],
        "sha256": sha256,
        "media_type": "application/x-npz",
        "format": "npz",
        "bytes": int(path.stat().st_size),
        "float_storage": "float16",
        "local_path": str(path),
        "remote_path": upload.get("remote_path"),
        "gdrive_file_id": upload.get("gdrive_file_id"),
        "source_apf_chunk": str(source_path),
        "source_apf_index": int(source_index),
        "loss": case.get("loss", None),
        "optimization_iter": case.get("iter", case.get("optimization_iter", None)),
        "param_sha256_fp16": _param_hash(static["params"]),
    }


def _export_case(
    *,
    case: dict[str, Any],
    static: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    traj_dir = Path(str(case["traj_dir"]))
    apf_dir = traj_dir / "apf_logs"
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")

    entries: list[dict[str, Any]] = []
    for source_path, _s0, _s1, _idx in chunks:
        with np.load(source_path) as data:
            _validate_source(data, source_path)
            steps = np.asarray(data["steps"], dtype=np.int64)
            indices = _selected_indices(
                steps,
                start_step=int(args.start_step),
                stride_steps=int(args.stride_steps),
                end_step=args.end_step,
            )
            for source_index in indices.tolist():
                step = int(steps[source_index])
                checkpoint_id = f"{case['case_id']}_t{step}"
                file_name = f"{checkpoint_id}.npz"
                local_path = output_dir / "checkpoints" / file_name
                if args.overwrite or not local_path.exists():
                    payload = _build_payload(
                        data=data,
                        source_path=source_path,
                        source_index=int(source_index),
                        n_steps=int(steps.shape[0]),
                        case=case,
                        static=static,
                        step=step,
                        checkpoint_id=checkpoint_id,
                    )
                    _write_checkpoint_npz(local_path, payload, compress=not args.no_compress)
                else:
                    _validate_exported_npz(local_path)

                sha256 = _sha256_file(local_path)
                if args.gdrive_remote:
                    upload = _upload_with_rclone(
                        local_path,
                        remote_base=str(args.gdrive_remote),
                        remote_relpath=f"checkpoints/{file_name}",
                        rclone=str(args.rclone_binary),
                        dry_run=bool(args.dry_run),
                        allow_path_uri=bool(args.allow_rclone_uri),
                    )
                else:
                    upload = {
                        "uri": f"file://{local_path}",
                        "remote_path": None,
                        "gdrive_file_id": None,
                        "dry_run": bool(args.dry_run),
                    }
                entries.append(
                    _entry_from_checkpoint(
                        path=local_path,
                        upload=upload,
                        sha256=sha256,
                        case=case,
                        static=static,
                        step=step,
                        checkpoint_id=checkpoint_id,
                        source_path=source_path,
                        source_index=int(source_index),
                    )
                )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export resume-capable FlowLenia APF checkpoints into a sparse checkpoint pool."
    )
    parser.add_argument("dataset_root", help="Golden-set directory containing manifest.json and traj_*/apf_logs.")
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/checkpoint_pool_5k.")
    parser.add_argument("--pool-id", default="flow_lenia_pool_v1")
    parser.add_argument("--stride-steps", type=int, default=5000, help="Checkpoint stride in simulation steps.")
    parser.add_argument("--start-step", type=int, default=0, help="First exact simulation step to export.")
    parser.add_argument("--end-step", type=int, default=None, help="Optional last exact simulation step to export.")
    parser.add_argument("--case-id-mode", choices=["case", "traj"], default="case")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional case/traj ids to export.")
    parser.add_argument("--case-limit", type=int, default=None, help="Debug limit after case filtering.")
    parser.add_argument("--gdrive-remote", default=None, help="rclone Drive destination, e.g. gdrive:flow_lenia_pool_v1.")
    parser.add_argument("--rclone-binary", default="rclone")
    parser.add_argument("--allow-rclone-uri", action="store_true", help="Allow rclone:// URIs if Drive file IDs are unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Build local files but do not upload to Drive.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite local checkpoint npz files if they already exist.")
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--allow-food-state", action="store_true", help="Do not fail if config has food=true.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.stride_steps) <= 0:
        raise ValueError(f"stride_steps must be > 0, got {args.stride_steps}")

    dataset_root = resolve_path(args.dataset_root)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {args.dataset_root}")
    output_dir = resolve_path(args.output_dir, dataset_root) if args.output_dir else dataset_root / "checkpoint_pool_5k"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _case_dirs_from_manifest(dataset_root, case_id_mode=str(args.case_id_mode))
    if args.case_ids:
        keep = {str(x) for x in args.case_ids}
        cases = [c for c in cases if str(c["case_id"]) in keep or str(c["traj_id"]) in keep]
    if args.case_limit is not None:
        cases = cases[: max(0, int(args.case_limit))]
    if not cases:
        raise ValueError(f"No cases found under {dataset_root}")

    files: list[dict[str, Any]] = []
    substrate_config: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []

    try:
        from tqdm import tqdm

        iterator = tqdm(cases, desc="export checkpoint pool")
    except Exception:
        iterator = cases

    for case in iterator:
        try:
            static = _load_case_static(case, allow_food_state=bool(args.allow_food_state))
            if substrate_config is None:
                substrate_config = dict(static["substrate_config"])
            entries = _export_case(case=case, static=static, output_dir=output_dir, args=args)
            files.extend(entries)
        except Exception as exc:
            errors.append({"case_id": case.get("case_id"), "traj_id": case.get("traj_id"), "error": str(exc)})
            raise

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pool_id": str(args.pool_id),
        "storage": "gdrive" if args.gdrive_remote else "local",
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_stride_steps": int(args.stride_steps),
        "checkpoint_start_step": int(args.start_step),
        "checkpoint_end_step": args.end_step,
        "created_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "source_dataset_root": str(dataset_root),
        "substrate_config": substrate_config or {},
        "resume_contract": {
            "runner": "scripts/flowlenia_minibang_resume.py",
            "source_required_keys": list(SOURCE_REQUIRED_KEYS),
            "exported_required_keys": list(EXPORTED_REQUIRED_KEYS),
            "float_storage": "float16",
            "food_state_saved": False,
            "notes": "Dynamics tensors fK/m/s/fcr are reconstructed from params and config; RNG stream state is stored.",
        },
        "n_cases": int(len(cases)),
        "n_files": int(len(files)),
        "files": files,
        "errors": errors,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    upload_summary = None
    if args.gdrive_remote:
        manifest_sha = _sha256_file(manifest_path)
        upload_summary = _upload_with_rclone(
            manifest_path,
            remote_base=str(args.gdrive_remote),
            remote_relpath="manifest.json",
            rclone=str(args.rclone_binary),
            dry_run=bool(args.dry_run),
            allow_path_uri=bool(args.allow_rclone_uri),
        )
        upload_summary["sha256"] = manifest_sha
        upload_summary["local_path"] = str(manifest_path)
        write_json(output_dir / "manifest_upload.json", upload_summary)

    print(f"Exported {len(files)} checkpoint files from {len(cases)} cases.")
    print(f"Local manifest: {manifest_path}")
    if upload_summary:
        print(f"Uploaded manifest URI: {upload_summary['uri']}")


if __name__ == "__main__":
    main()
