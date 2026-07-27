from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flowlenia_minibang_common import list_apf_chunks
from flowlenia_minibang_common import load_config as load_rollout_config
from flowlenia_minibang_simulate import select_params, simulate_batch, simulate_optimizer_native_selected_batch
from paper_suite_c2_flowlenia_highres import (
    REQUIRED_APF_KEYS,
    _apf_status as _base_apf_status,
    _discover_checkpoints,
    _traj_id,
)
from paper_suite_common import (
    ensure_dir,
    load_config,
    log_event,
    resolve_path,
    to_plain,
    write_json,
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _section(cfg: Any, section_key: str = "flow_lenia_arun_lagrangian_apf") -> Any:
    return _get(cfg.get("simulation", {}), section_key, {})


def _output_paths(output_root: Path, traj_id: str) -> dict[str, Path]:
    traj_dir = output_root / traj_id
    return {
        "run_root": traj_dir,
        "traj_dir": traj_dir,
        "apf_dir": traj_dir / "apf_logs",
        "metrics_path": traj_dir / "metrics.npz",
        "config_path": traj_dir / "config.yaml",
        "params_path": traj_dir / "params.npy",
        "video_path": traj_dir / "video.mp4",
        "frame_times_path": traj_dir / "frame_times.csv",
    }


def _expected_rollout_signature(rollout_config: Path, flat: Any) -> dict[str, Any]:
    keys = (
        "grid_size",
        "sigma",
        "flow_sigma",
        "seed_n_patches",
        "seed_patch_size",
        "rollout_steps",
        "max_steps",
        "snapshot_interval",
        "snapshots_per_file",
        "save_A",
        "save_F",
        "save_lagrangian",
        "lagrangian_n_particles",
        "lagrangian_init_mode",
        "lagrangian_flow_channel",
        "lagrangian_flow_reduce",
        "lagrangian_channel_mode",
        "lagrangian_noise_model",
        "lagrangian_diffusion_scale",
        "log_clip_evolution",
        "run_seed_protocol",
    )
    out = {"rollout_config": str(rollout_config)}
    for key in keys:
        value = _get(flat, key, None)
        if value is not None:
            out[key] = value
    return out


def _saved_rollout_signature(paths: dict[str, Path]) -> dict[str, Any]:
    cfg_path = paths["config_path"]
    if not cfg_path.exists():
        return {}
    cfg = OmegaConf.load(cfg_path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
        cfg.get("minibang", {}),
    )
    return _expected_rollout_signature(Path(str(cfg.get("rollout_config", ""))), flat)


def _apply_section_rollout_overrides(rollout_cfg: Any, rollout_flat: Any, section: Any) -> tuple[Any, Any]:
    passthrough_keys = (
        "run_seed_protocol",
    )
    cfg_out = OmegaConf.create(OmegaConf.to_container(rollout_cfg, resolve=False))
    flat_out = OmegaConf.create(OmegaConf.to_container(rollout_flat, resolve=False))
    if cfg_out.get("minibang", None) is None:
        cfg_out.minibang = OmegaConf.create()
    section_names = {"meta", "substrate", "simulation", "logging", "metric", "minibang"}
    rollout_overrides = _get(section, "rollout_overrides", None)
    if rollout_overrides is not None:
        for section_name, values in rollout_overrides.items():
            name = str(section_name)
            if name not in section_names:
                raise ValueError(
                    "flow_lenia_arun_lagrangian_apf.rollout_overrides keys must be one of "
                    f"{sorted(section_names)}, got {name!r}."
                )
            if cfg_out.get(name, None) is None:
                cfg_out[name] = OmegaConf.create()
            cfg_out[name] = OmegaConf.merge(cfg_out.get(name, {}), values)
            if values is not None:
                flat_out = OmegaConf.merge(flat_out, values)
    for key in passthrough_keys:
        value = _get(section, key, None)
        if value is None:
            continue
        flat_out[key] = value
        cfg_out.minibang[key] = value
    return cfg_out, flat_out


def _signature_mismatch(saved: dict[str, Any], expected: dict[str, Any]) -> str:
    for key, expected_value in expected.items():
        if key == "rollout_config":
            continue
        if key not in saved:
            return f"saved config missing {key}"
        if saved[key] != expected_value:
            return f"saved config {key}={saved[key]!r}, expected {expected_value!r}"
    return ""


def _apf_status(
    paths: dict[str, Path],
    *,
    expected_steps: int | None = None,
    expected_signature: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    ready, message = _base_apf_status(paths)
    if not ready or expected_steps is None:
        return ready, message
    if expected_signature is not None:
        try:
            saved_signature = _saved_rollout_signature(paths)
        except Exception as exc:
            return False, f"cannot read saved config {paths['config_path']}: {exc}"
        mismatch = _signature_mismatch(saved_signature, expected_signature)
        if mismatch:
            return False, mismatch
    chunks = list_apf_chunks(paths["apf_dir"])
    if not chunks:
        return False, f"missing APF chunks in {paths['apf_dir']}"
    first_start = min(start for _path, start, _end, _idx in chunks)
    last_end = max(end for _path, _start, end, _idx in chunks)
    if first_start > 0:
        return False, f"APF coverage starts at {first_start}, expected 0"
    if last_end < int(expected_steps):
        return False, f"APF coverage ends at {last_end}, expected >= {int(expected_steps)}"
    return True, ""


def _validate_rollout_profile(flat: Any, *, expected_steps: int) -> None:
    rollout_steps = int(_get(flat, "rollout_steps", expected_steps))
    max_steps_raw = _get(flat, "max_steps", rollout_steps)
    max_steps = int(max_steps_raw) if max_steps_raw is not None else rollout_steps
    if rollout_steps != int(expected_steps) or max_steps != int(expected_steps):
        raise ValueError(
            "Flow-Lenia A-run APF rollout must be pinned to the requested limit. "
            f"Expected rollout_steps=max_steps={expected_steps}, got "
            f"rollout_steps={rollout_steps}, max_steps={max_steps}."
        )
    for key in ("save_A", "save_F", "save_lagrangian"):
        value = _get(flat, key, True)
        if str(value).strip().lower() in {"0", "false", "no", "off"}:
            raise ValueError(f"Flow-Lenia A-run APF rollout requires {key}=true.")


def _paper_check_control_a_seed(section: Any, row: dict[str, Any], suite_idx: int) -> int:
    base = int(_get(section, "run_seed_base", 400_000))
    mode = str(_get(section, "run_seed_mode", "source_run_idx")).strip().lower()
    source_run_idx = int(row.get("source_run_idx", -1))
    if mode == "source_run_idx" and source_run_idx >= 0:
        group_idx = source_run_idx
    elif mode == "suite_index":
        group_idx = int(suite_idx)
    else:
        raise ValueError("run_seed_mode must be 'source_run_idx' or 'suite_index'.")
    return int(base + 2 * group_idx)


def _rollout_run_seed(section: Any, base_seed: int, rollout_seed_idx: int) -> int:
    stride = int(_get(section, "run_seed_rep_stride", 1))
    if stride < 1:
        raise ValueError(f"run_seed_rep_stride must be >= 1, got {stride}.")
    return int(base_seed + stride * int(rollout_seed_idx))


def _candidate_traj_id(row: dict[str, Any], *, candidate_kind: str, candidate_idx: int) -> str:
    base = _traj_id(row)
    if candidate_kind == "optimized":
        return base
    return f"{base}_{candidate_kind}_{candidate_idx:03d}"


def _rollout_traj_id(base_traj_id: str, *, rollout_seed_idx: int, n_rollout_seeds: int) -> str:
    if int(n_rollout_seeds) == 1:
        return str(base_traj_id)
    return f"{base_traj_id}_seed_{int(rollout_seed_idx):03d}"


def _random_selection_mode(section: Any) -> str:
    return str(_get(section, "random_checkpoint_selection", "per_source_group")).strip().lower()


def _uses_optimizer_iter0_random(section: Any) -> bool:
    return _random_selection_mode(section) in {
        "optimization_iter0",
        "optimizer_iter0",
        "pop_traj_iter0",
        "source_pop_traj_iter0",
    }


def _uses_random_theta_optimizer_context(section: Any) -> bool:
    return _random_selection_mode(section) in {
        "per_source_group_optimizer_context",
        "per_source_group_optimizer_init",
        "random_params_optimizer_context",
        "random_params_optimizer_init",
    }


def _params_hash(params: Any) -> str:
    arr = np.asarray(params, dtype=np.float32).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _optimizer_native_random_pop_indices(section: Any, n_random: int) -> list[int]:
    raw = _get(section, "random_optimizer_native_pop_indices", None)
    if raw not in (None, ""):
        values = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
        out = [int(x) for x in values]
    else:
        out = list(range(int(n_random)))
    if len(out) < int(n_random):
        raise ValueError(
            f"Need at least {int(n_random)} random_optimizer_native_pop_indices, got {out}."
        )
    return out[: int(n_random)]


def _mapping_get_bool(raw: Any, key: int, default: bool) -> bool:
    if raw in (None, ""):
        return bool(default)
    values = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    if not isinstance(values, dict):
        return bool(default)
    keys = (key, str(key), f"run_{int(key):03d}", f"{int(key):03d}")
    for candidate in keys:
        if candidate in values:
            return bool(values[candidate])
    return bool(default)


def _optimizer_native_legacy_sigma_collision_for_row(section: Any, row: dict[str, Any]) -> bool:
    default = bool(_get(section, "optimized_native_legacy_sigma_collision", False))
    mapping = _get(section, "optimizer_native_legacy_sigma_collision_by_source_run_idx", None)
    run_idx = int(row.get("source_run_idx", row.get("run_idx", -1)))
    if run_idx < 0:
        return bool(default)
    return _mapping_get_bool(mapping, run_idx, default)


def _optimizer_source_from_selected_checkpoint(checkpoint_dir: Path) -> tuple[Path, Path]:
    selected_path = checkpoint_dir / "selected_candidate.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"optimization_iter0 random controls require {selected_path}")
    selected = json.loads(selected_path.read_text())
    pop_path = Path(str(selected.get("source_pop_traj", "")))
    if not pop_path.is_absolute():
        pop_path = (Path.cwd() / pop_path).resolve()
    source_run_dir = Path(str(selected.get("source_run_dir", pop_path.parent)))
    if not source_run_dir.is_absolute():
        source_run_dir = (Path.cwd() / source_run_dir).resolve()
    if not pop_path.exists():
        raise FileNotFoundError(f"source_pop_traj not found: {pop_path}")
    if not (source_run_dir / "optimization_config.yaml").exists():
        raise FileNotFoundError(f"optimization_config.yaml not found under {source_run_dir}")
    return pop_path, source_run_dir


def _attach_selected_checkpoint_tau_metadata(item: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    selected_path = checkpoint_dir / "selected_candidate.json"
    if not selected_path.exists():
        return item
    try:
        selected = json.loads(selected_path.read_text())
    except Exception:
        return item
    tau = selected.get("tau", {})
    if not isinstance(tau, dict):
        tau = {}

    def get_tau_value(key: str) -> Any:
        if key in tau:
            return tau[key]
        return selected.get(key)

    mappings = (
        ("optimizer_native_tau_idx", "tau_idx", int),
        ("optimizer_native_tau_steps", "tau_steps", int),
        ("optimizer_native_tau_frames", "tau_frames", int),
        ("optimizer_native_tau_selector_raw", "tau_selector_raw", float),
    )
    for out_key, source_key, cast in mappings:
        value = get_tau_value(source_key)
        if value in (None, ""):
            continue
        if out_key not in item:
            item[out_key] = cast(value)
    return item


def _common_selected_metadata(
    item: dict[str, Any],
    *,
    row: dict[str, Any],
    section: Any,
    suite_idx: int,
    selection_idx: int,
    checkpoint_dir: Path,
    traj_id: str,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
    rollout_seed_idx: int,
    n_rollout_seeds: int,
) -> dict[str, Any]:
    item["traj_id"] = str(traj_id)
    item["selection_idx"] = int(selection_idx)
    item["source"] = "paper_check_flow_lenia_control_a"
    item["source_checkpoint_dir"] = str(checkpoint_dir)
    item["source_root"] = str(row["source_root"])
    item["source_root_rank"] = int(row["source_root_rank"])
    item["source_run_idx"] = int(row.get("source_run_idx", -1))
    item["suite_run_idx"] = int(row.get("run_idx", suite_idx))
    base_seed = _paper_check_control_a_seed(section, row, suite_idx)
    item["base_run_seed"] = int(base_seed)
    item["run_seed"] = _rollout_run_seed(section, base_seed, rollout_seed_idx)
    item["rollout_seed_idx"] = int(rollout_seed_idx)
    item["rollout_seed_count"] = int(n_rollout_seeds)
    item["candidate_kind"] = str(candidate_kind)
    item["candidate_idx"] = int(candidate_idx)
    item["candidate_label"] = str(candidate_label)
    item["optimizer_native_legacy_sigma_collision"] = _optimizer_native_legacy_sigma_collision_for_row(
        section,
        row,
    )
    return item


def _random_checkpoint_dir(section: Any, row: dict[str, Any], random_idx: int) -> Path:
    source_run_idx = int(row.get("source_run_idx", -1))
    if source_run_idx < 0:
        raise ValueError(f"Cannot derive random checkpoint dir without source_run_idx: {row}")
    random_root_raw = _get(section, "random_checkpoint_root", None)
    if random_root_raw not in (None, ""):
        random_root = resolve_path(random_root_raw)
        if random_root is None:
            raise FileNotFoundError(f"random_checkpoint_root could not be resolved: {random_root_raw}")
        return (
            random_root
            / f"group_{source_run_idx:03d}"
            / f"random_{int(random_idx):03d}"
        )
    source_root = Path(row["source_root"])
    return (
        source_root.parent
        / "frustration_simulation"
        / "random_params"
        / f"group_{source_run_idx:03d}"
        / f"random_{int(random_idx):03d}"
    )


def _random_checkpoint_dirs(section: Any, row: dict[str, Any], n_random: int) -> list[Path]:
    mode = str(_get(section, "random_checkpoint_selection", "per_source_group")).strip().lower()
    random_root_raw = _get(section, "random_checkpoint_root", None)
    if random_root_raw not in (None, "") and mode in {"all_groups_flat", "global_flat", "flat"}:
        random_root = resolve_path(random_root_raw)
        if random_root is None or not random_root.exists():
            raise FileNotFoundError(f"random_checkpoint_root not found: {random_root_raw}")

        def sort_key(best_path: Path) -> tuple[int, int, str]:
            group_match = re.match(r"group_(\d+)$", best_path.parent.parent.name)
            random_match = re.match(r"random_(\d+)$", best_path.parent.name)
            group_idx = int(group_match.group(1)) if group_match else 10**9
            random_idx = int(random_match.group(1)) if random_match else 10**9
            return group_idx, random_idx, str(best_path)

        dirs = [path.parent for path in sorted(random_root.glob("group_*/random_*/best.pkl"), key=sort_key)]
        if len(dirs) < int(n_random):
            raise FileNotFoundError(
                f"Need {int(n_random)} random checkpoints under {random_root}/group_*/random_*/best.pkl, "
                f"found {len(dirs)}."
            )
        return dirs[: int(n_random)]
    return [_random_checkpoint_dir(section, row, random_idx) for random_idx in range(int(n_random))]


def _select_one_checkpoint(
    *,
    row: dict[str, Any],
    section: Any,
    rollout_flat: Any,
    suite_idx: int,
    selection_idx: int,
    checkpoint_dir: Path,
    traj_id: str,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
    rollout_seed_idx: int = 0,
    n_rollout_seeds: int = 1,
) -> dict[str, Any]:
    selected = select_params(checkpoint_dir, rollout_flat)
    if not selected:
        raise FileNotFoundError(f"No selectable params found in {checkpoint_dir}.")
    item = dict(selected[0])
    item = _common_selected_metadata(
        item,
        row=row,
        section=section,
        suite_idx=suite_idx,
        selection_idx=selection_idx,
        checkpoint_dir=checkpoint_dir,
        traj_id=traj_id,
        candidate_kind=candidate_kind,
        candidate_idx=candidate_idx,
        candidate_label=candidate_label,
        rollout_seed_idx=rollout_seed_idx,
        n_rollout_seeds=n_rollout_seeds,
    )
    return _attach_selected_checkpoint_tau_metadata(item, checkpoint_dir)


def _attach_optimizer_native_context(
    item: dict[str, Any],
    *,
    checkpoint_dir: Path,
    use_row_params: bool = False,
    params_source: str = "optimizer_native",
    theta_source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    pop_path, source_run_dir = _optimizer_source_from_selected_checkpoint(checkpoint_dir)
    selected_path = checkpoint_dir / "selected_candidate.json"
    selected = json.loads(selected_path.read_text())
    optimizer_iter = int(selected.get("iter", -1))
    optimizer_pop_idx = int(selected.get("pop_idx", -1))
    item["optimizer_native_source_pop_traj"] = str(pop_path)
    item["optimizer_native_source_run_dir"] = str(source_run_dir)
    item["optimizer_native_iter"] = int(optimizer_iter)
    item["optimizer_native_pop_idx"] = int(optimizer_pop_idx)
    item["optimizer_native_params_source"] = str(params_source)
    item["optimizer_native_use_row_params"] = bool(use_row_params)
    if theta_source_checkpoint is not None:
        item["theta_source_checkpoint"] = str(theta_source_checkpoint)

    with pop_path.open("rb") as f:
        pop = pickle.load(f)
    if optimizer_iter < 0 or optimizer_pop_idx < 0:
        return item
    if "objective_score" in pop:
        objective_score = np.asarray(pop["objective_score"], dtype=np.float32)
        item["optimizer_native_score_mspd"] = float(objective_score[optimizer_iter, optimizer_pop_idx])
        if not use_row_params:
            item["fitness"] = float(objective_score[optimizer_iter, optimizer_pop_idx])
    if "score_by_seed" in pop:
        score_by_seed = np.asarray(pop["score_by_seed"], dtype=np.float32)
        item["optimizer_native_score_by_seed_mspd"] = [
            float(x) for x in score_by_seed[optimizer_iter, optimizer_pop_idx].reshape(-1)
        ]
    if "tau_idx" in pop:
        item["optimizer_native_tau_idx"] = int(np.asarray(pop["tau_idx"])[optimizer_iter, optimizer_pop_idx])
    if "tau_steps" in pop:
        item["optimizer_native_tau_steps"] = int(np.asarray(pop["tau_steps"])[optimizer_iter, optimizer_pop_idx])
    if "tau_frames" in pop:
        item["optimizer_native_tau_frames"] = int(np.asarray(pop["tau_frames"])[optimizer_iter, optimizer_pop_idx])
    if "tau_selector_raw" in pop:
        item["optimizer_native_tau_selector_raw"] = float(
            np.asarray(pop["tau_selector_raw"], dtype=np.float32)[optimizer_iter, optimizer_pop_idx]
        )
    return item


def _select_optimizer_native_pop_candidate(
    *,
    row: dict[str, Any],
    section: Any,
    suite_idx: int,
    selection_idx: int,
    traj_id: str,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
    rollout_seed_idx: int,
    n_rollout_seeds: int,
    optimizer_iter: int,
    optimizer_pop_idx: int,
) -> dict[str, Any]:
    checkpoint_dir = Path(row["checkpoint_dir"])
    pop_path, source_run_dir = _optimizer_source_from_selected_checkpoint(checkpoint_dir)
    with pop_path.open("rb") as f:
        pop = pickle.load(f)
    params = np.asarray(pop["params"], dtype=np.float32)
    if params.ndim != 3:
        raise ValueError(f"invalid pop_traj params shape in {pop_path}: {params.shape}")
    optimizer_iter = int(optimizer_iter)
    optimizer_pop_idx = int(optimizer_pop_idx)
    if optimizer_iter < 0 or optimizer_iter >= params.shape[0]:
        raise ValueError(f"optimizer_iter={optimizer_iter} out of range for {pop_path} shape={params.shape}")
    if optimizer_pop_idx < 0 or optimizer_pop_idx >= params.shape[1]:
        raise ValueError(f"optimizer_pop_idx={optimizer_pop_idx} out of range for {pop_path} shape={params.shape}")

    item: dict[str, Any] = {
        "params": np.asarray(params[optimizer_iter, optimizer_pop_idx], dtype=np.float32),
        "fitness": 0.0,
        "iter": int(optimizer_iter),
        "optimizer_native_source_pop_traj": str(pop_path),
        "optimizer_native_source_run_dir": str(source_run_dir),
        "optimizer_native_iter": int(optimizer_iter),
        "optimizer_native_pop_idx": int(optimizer_pop_idx),
    }
    if "objective_score" in pop:
        objective_score = np.asarray(pop["objective_score"], dtype=np.float32)
        item["optimizer_native_score_mspd"] = float(objective_score[optimizer_iter, optimizer_pop_idx])
        item["fitness"] = float(objective_score[optimizer_iter, optimizer_pop_idx])
    if "score_by_seed" in pop:
        score_by_seed = np.asarray(pop["score_by_seed"], dtype=np.float32)
        item["optimizer_native_score_by_seed_mspd"] = [
            float(x) for x in score_by_seed[optimizer_iter, optimizer_pop_idx].reshape(-1)
        ]
    if "tau_idx" in pop:
        item["optimizer_native_tau_idx"] = int(np.asarray(pop["tau_idx"])[optimizer_iter, optimizer_pop_idx])
    if "tau_steps" in pop:
        item["optimizer_native_tau_steps"] = int(np.asarray(pop["tau_steps"])[optimizer_iter, optimizer_pop_idx])
    if "tau_frames" in pop:
        item["optimizer_native_tau_frames"] = int(np.asarray(pop["tau_frames"])[optimizer_iter, optimizer_pop_idx])
    if "tau_selector_raw" in pop:
        item["optimizer_native_tau_selector_raw"] = float(
            np.asarray(pop["tau_selector_raw"], dtype=np.float32)[optimizer_iter, optimizer_pop_idx]
        )

    return _common_selected_metadata(
        item,
        row=row,
        section=section,
        suite_idx=suite_idx,
        selection_idx=selection_idx,
        checkpoint_dir=source_run_dir,
        traj_id=traj_id,
        candidate_kind=candidate_kind,
        candidate_idx=candidate_idx,
        candidate_label=candidate_label,
        rollout_seed_idx=rollout_seed_idx,
        n_rollout_seeds=n_rollout_seeds,
    )


def _write_manifest(
    output_root: Path,
    *,
    rollout_config: Path,
    selected: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
    expected_steps: int | None = None,
    expected_signature: dict[str, Any] | None = None,
) -> None:
    trajectories: list[dict[str, Any]] = []
    for row in selected:
        traj_id = str(row["traj_id"])
        paths = _output_paths(output_root, traj_id)
        apf_ready, apf_message = _apf_status(
            paths,
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        trajectories.append(
            {
                "traj_id": traj_id,
                "selection_idx": int(row["selection_idx"]),
                "source": str(row.get("source", "paper_check_flow_lenia_control_a")),
                "source_checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                "source_root": str(row.get("source_root", "")),
                "source_root_rank": int(row.get("source_root_rank", -1)),
                "source_run_idx": int(row.get("source_run_idx", -1)),
                "suite_run_idx": int(row.get("suite_run_idx", row["selection_idx"])),
                "base_run_seed": int(row.get("base_run_seed", row.get("run_seed", -1))),
                "run_seed": int(row.get("run_seed", -1)),
                "rollout_seed_idx": int(row.get("rollout_seed_idx", 0)),
                "rollout_seed_count": int(row.get("rollout_seed_count", 1)),
                "candidate_kind": str(row.get("candidate_kind", "optimized")),
                "candidate_idx": int(row.get("candidate_idx", 0)),
                "candidate_label": str(row.get("candidate_label", "optimized")),
                "traj_dir": str(paths["traj_dir"]),
                "apf_dir": str(paths["apf_dir"]),
                "metrics_path": str(paths["metrics_path"]),
                "config_path": str(paths["config_path"]),
                "params_path": str(paths["params_path"]),
                "video_path": str(paths["video_path"]),
                "frame_times_path": str(paths["frame_times_path"]),
                "apf_ready": bool(apf_ready),
                "ready": bool(apf_ready),
                "apf_status": apf_message,
                **{
                    key: to_plain(row[key])
                    for key in (
                        "optimizer_native_source_pop_traj",
                        "optimizer_native_source_run_dir",
                        "optimizer_native_iter",
                        "optimizer_native_pop_idx",
                        "optimizer_native_tau_idx",
                        "optimizer_native_tau_steps",
                        "optimizer_native_tau_frames",
                        "optimizer_native_tau_selector_raw",
                        "optimizer_native_score_mspd",
                        "optimizer_native_score_by_seed_mspd",
                        "optimizer_native_legacy_sigma_collision",
                        "optimizer_native_params_source",
                        "optimizer_native_use_row_params",
                        "theta_source_checkpoint",
                    )
                    if key in row
                },
            }
        )
    write_json(
        output_root / "manifest.json",
        {
            "source_kind": "paper_check_flow_lenia_control_a_lagrangian_sparse_apf",
            "rollout_config": str(rollout_config),
            "expected_rollout_signature": dict(expected_signature or {}),
            "required_apf_keys": list(REQUIRED_APF_KEYS),
            "n_trajectories": len(trajectories),
            "trajectories": trajectories,
            "commands": command_rows,
        },
    )


def run(
    config_path: str | Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    section_key: str = "flow_lenia_arun_lagrangian_apf",
) -> dict[str, Any]:
    cfg, _ = load_config(config_path)
    section = _section(cfg, section_key)
    if not bool(_get(section, "enabled", True)):
        log_event(f"Flow-Lenia A-run APF disabled section={section_key}", component="arun-apf")
        return {"status": "disabled", "section": section_key}

    output_root = resolve_path(_get(section, "output_root", "experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k"))
    assert output_root is not None
    output_root = ensure_dir(output_root)
    rollout_config = resolve_path(_get(section, "rollout_config", "experiments/paper_suite/flowlenia_arun_apf_500k.yaml"))
    if rollout_config is None or not rollout_config.exists():
        raise FileNotFoundError(f"Flow-Lenia A-run APF rollout_config not found: {rollout_config}")

    expected_steps = int(_get(section, "rollout_steps", 500_000))
    rollout_cfg, rollout_flat = load_rollout_config(rollout_config, [])
    rollout_cfg, rollout_flat = _apply_section_rollout_overrides(rollout_cfg, rollout_flat, section)
    _validate_rollout_profile(rollout_flat, expected_steps=expected_steps)
    expected_signature = _expected_rollout_signature(rollout_config, rollout_flat)

    n_rollout_seeds = int(
        _get(section, "n_rollout_seeds_per_checkpoint", _get(section, "n_trajectories_per_checkpoint", 1))
    )
    if n_rollout_seeds < 1:
        raise ValueError(f"n_rollout_seeds_per_checkpoint must be >= 1, got {n_rollout_seeds}.")
    # Parameter selection remains one parameter vector per checkpoint. Multiple
    # trajectories here mean repeated stochastic rollouts of that same vector.
    rollout_flat.n_trajectories = 1
    batch_size = max(1, int(_get(section, "batch_size", _get(rollout_flat, "batch_size", 1))))
    rollout_flat.batch_size = int(batch_size)
    optimized_apf_source = str(_get(section, "optimized_apf_source", "minibang")).strip().lower()
    use_optimizer_native_optimized = optimized_apf_source in {
        "optimizer_native",
        "optimization_native",
        "optimizer_nested_jit",
        "native",
    }
    optimized_native_legacy_sigma_collision = bool(
        _get(section, "optimized_native_legacy_sigma_collision", False)
    )

    checkpoints = _discover_checkpoints(section)
    log_event(
        "Flow-Lenia A-run APF start "
        f"section={section_key} force={force} dry_run={dry_run} output_root={output_root} "
        f"rollout_steps={expected_steps} batch_size={batch_size} n_checkpoints={len(checkpoints)} "
        f"n_rollout_seeds_per_checkpoint={n_rollout_seeds}",
        component="arun-apf",
    )
    if not checkpoints:
        summary = {"status": "skipped", "section": section_key, "reason": "no optimized checkpoints found", "output_root": str(output_root)}
        write_json(output_root / "manifest.json", {"source_kind": "paper_check_flow_lenia_control_a_lagrangian_sparse_apf", "trajectories": []})
        write_json(output_root / "simulation_summary.json", summary)
        if bool(_get(section, "required", True)) and not dry_run:
            raise FileNotFoundError("No Flow-Lenia optimized checkpoints found for A-run APF rollouts.")
        return summary

    selected: list[dict[str, Any]] = []
    include_random = bool(_get(section, "include_random_baselines", True))
    n_random = int(_get(section, "num_random_baselines", 3)) if include_random else 0
    use_optimizer_iter0_random = bool(include_random and _uses_optimizer_iter0_random(section))
    use_random_theta_optimizer_context = bool(include_random and _uses_random_theta_optimizer_context(section))
    random_optimizer_iter = int(_get(section, "random_optimizer_native_iter", 0))
    random_optimizer_pop_indices = _optimizer_native_random_pop_indices(section, n_random) if use_optimizer_iter0_random else []
    for suite_idx, row in enumerate(checkpoints):
        optimized_base_traj_id = _candidate_traj_id(row, candidate_kind="optimized", candidate_idx=0)
        for rollout_seed_idx in range(n_rollout_seeds):
            optimized_item = _select_one_checkpoint(
                row=row,
                section=section,
                rollout_flat=rollout_flat,
                suite_idx=suite_idx,
                selection_idx=len(selected),
                checkpoint_dir=Path(row["checkpoint_dir"]),
                traj_id=_rollout_traj_id(
                    optimized_base_traj_id,
                    rollout_seed_idx=rollout_seed_idx,
                    n_rollout_seeds=n_rollout_seeds,
                ),
                candidate_kind="optimized",
                candidate_idx=0,
                candidate_label="optimized",
                rollout_seed_idx=rollout_seed_idx,
                n_rollout_seeds=n_rollout_seeds,
            )
            if use_optimizer_native_optimized:
                optimized_item = _attach_optimizer_native_context(
                    optimized_item,
                    checkpoint_dir=Path(row["checkpoint_dir"]),
                )
            selected.append(optimized_item)
        if use_optimizer_iter0_random:
            for random_idx, pop_idx in enumerate(random_optimizer_pop_indices):
                random_base_traj_id = _candidate_traj_id(row, candidate_kind="random", candidate_idx=random_idx)
                for rollout_seed_idx in range(n_rollout_seeds):
                    selected.append(
                        _select_optimizer_native_pop_candidate(
                            row=row,
                            section=section,
                            suite_idx=suite_idx,
                            selection_idx=len(selected),
                            traj_id=_rollout_traj_id(
                                random_base_traj_id,
                                rollout_seed_idx=rollout_seed_idx,
                                n_rollout_seeds=n_rollout_seeds,
                            ),
                            candidate_kind="random",
                            candidate_idx=random_idx,
                            candidate_label=f"random_{random_idx:03d}_iter{random_optimizer_iter:03d}_pop{int(pop_idx):03d}",
                            rollout_seed_idx=rollout_seed_idx,
                            n_rollout_seeds=n_rollout_seeds,
                            optimizer_iter=random_optimizer_iter,
                            optimizer_pop_idx=int(pop_idx),
                        )
                    )
        else:
            for random_idx, random_dir in enumerate(_random_checkpoint_dirs(section, row, n_random)):
                if not (random_dir / "best.pkl").exists():
                    raise FileNotFoundError(
                        "Missing random baseline checkpoint for Flow-Lenia A-run APF. "
                        f"Expected {random_dir / 'best.pkl'} for suite_group={suite_idx}, "
                        f"source_run_idx={int(row.get('source_run_idx', -1))}, random_idx={random_idx}."
                    )
                random_base_traj_id = _candidate_traj_id(row, candidate_kind="random", candidate_idx=random_idx)
                for rollout_seed_idx in range(n_rollout_seeds):
                    random_item = _select_one_checkpoint(
                        row=row,
                        section=section,
                        rollout_flat=rollout_flat,
                        suite_idx=suite_idx,
                        selection_idx=len(selected),
                        checkpoint_dir=random_dir,
                        traj_id=_rollout_traj_id(
                            random_base_traj_id,
                            rollout_seed_idx=rollout_seed_idx,
                            n_rollout_seeds=n_rollout_seeds,
                        ),
                        candidate_kind="random",
                        candidate_idx=random_idx,
                        candidate_label=f"random_{random_idx:03d}",
                        rollout_seed_idx=rollout_seed_idx,
                        n_rollout_seeds=n_rollout_seeds,
                    )
                    if use_random_theta_optimizer_context:
                        random_item = _attach_optimizer_native_context(
                            random_item,
                            checkpoint_dir=Path(row["checkpoint_dir"]),
                            use_row_params=True,
                            params_source="external_random_checkpoint",
                            theta_source_checkpoint=random_dir,
                        )
                    selected.append(random_item)

    ready_by_traj: dict[str, tuple[bool, str]] = {}
    for row in selected:
        ready, message = _apf_status(
            _output_paths(output_root, str(row["traj_id"])),
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        ready_by_traj[str(row["traj_id"])] = (ready, message)
    pending = [
        row
        for row in selected
        if force or not ready_by_traj[str(row["traj_id"])][0]
    ]
    batches_to_run: list[dict[str, Any]] = []
    if use_optimizer_native_optimized:
        native_pending = [
            row
            for row in pending
            if str(row.get("candidate_kind", "optimized")) == "optimized"
            or row.get("optimizer_native_source_pop_traj", None) not in (None, "")
        ]
        minibang_pending = [
            row
            for row in pending
            if str(row.get("candidate_kind", "optimized")) != "optimized"
            and row.get("optimizer_native_source_pop_traj", None) in (None, "")
        ]
        by_checkpoint: dict[str, list[dict[str, Any]]] = {}
        for row in native_pending:
            row_legacy = bool(row.get("optimizer_native_legacy_sigma_collision", optimized_native_legacy_sigma_collision))
            if row.get("optimizer_native_source_pop_traj", None) not in (None, ""):
                key = "|".join(
                    (
                        str(Path(str(row["optimizer_native_source_pop_traj"])).resolve()),
                        str(int(row.get("optimizer_native_iter", -1))),
                        f"legacy={int(row_legacy)}",
                    )
                )
            else:
                key = "|".join(
                    (
                        str(Path(row["source_checkpoint_dir"]).resolve()),
                        f"legacy={int(row_legacy)}",
                    )
                )
            by_checkpoint.setdefault(key, []).append(row)
        for rows in by_checkpoint.values():
            rows.sort(key=lambda r: int(r.get("rollout_seed_idx", 0)))
            legacy_values = {
                bool(row.get("optimizer_native_legacy_sigma_collision", optimized_native_legacy_sigma_collision))
                for row in rows
            }
            if len(legacy_values) != 1:
                raise ValueError(f"optimizer-native batch has mixed sigma protocols: {legacy_values}")
            batches_to_run.append(
                {
                    "kind": "optimizer_native",
                    "rows": rows,
                    "legacy_sigma_collision": bool(next(iter(legacy_values))),
                }
            )
        for start in range(0, len(minibang_pending), batch_size):
            batches_to_run.append({"kind": "minibang", "rows": minibang_pending[start : start + batch_size]})
    else:
        batches_to_run = [
            {"kind": "minibang", "rows": pending[start : start + batch_size]}
            for start in range(0, len(pending), batch_size)
        ]

    run_traj_ids = {str(row["traj_id"]) for batch in batches_to_run for row in batch["rows"]}
    command_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        paths = _output_paths(output_root, str(row["traj_id"]))
        ready, message = ready_by_traj[str(row["traj_id"])]
        if str(row["traj_id"]) not in run_traj_ids:
            status = "exists"
            log_event(
                f"Flow-Lenia A-run APF {idx}/{len(selected)} traj={row['traj_id']} exists "
                f"run_root={paths['run_root']} apf_status={message or 'ready'}",
                component="arun-apf",
            )
        else:
            status = "queued"
            reason = "force" if force else message
            log_event(
                f"Flow-Lenia A-run APF {idx}/{len(selected)} traj={row['traj_id']} queued "
                f"run_root={paths['run_root']} pre_status={reason}",
                component="arun-apf",
            )
            message = reason
        command_rows.append(
            {
                "traj_id": str(row["traj_id"]),
                "checkpoint_dir": str(row.get("source_checkpoint_dir", "")),
                "run_root": str(paths["run_root"]),
                "status": status,
                "message": message,
                "command": "internal simulate_batch",
            }
        )
    _write_manifest(
        output_root,
        rollout_config=rollout_config,
        selected=selected,
        command_rows=command_rows,
        expected_steps=expected_steps,
        expected_signature=expected_signature,
    )

    if dry_run:
        for row in command_rows:
            if row["status"] == "queued":
                row["status"] = "dry_run"
        _write_manifest(
            output_root,
            rollout_config=rollout_config,
            selected=selected,
            command_rows=command_rows,
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        summary = {
            "status": "dry_run",
            "section": section_key,
            "output_root": str(output_root),
            "rollout_config": str(rollout_config),
            "n_selected": len(selected),
            "n_to_run": len(run_traj_ids),
            "n_batches_to_run": len(batches_to_run),
            "batch_size": int(batch_size),
            "manifest": str(output_root / "manifest.json"),
        }
        write_json(output_root / "simulation_summary.json", summary)
        log_event(
            f"Flow-Lenia A-run APF dry-run n_batches_to_run={len(batches_to_run)} n_to_run={len(run_traj_ids)}",
            component="arun-apf",
        )
        return summary

    flat_dict = OmegaConf.to_container(rollout_flat, resolve=True)
    for batch_idx, batch_info in enumerate(batches_to_run, start=1):
        batch = batch_info["rows"]
        batch_kind = str(batch_info["kind"])
        batch_ids = [str(row["traj_id"]) for row in batch]
        # Every selected row carries an explicit run_seed, so missing random
        # candidates can be filled without re-running already ready optimized
        # trajectories.
        existing = [
            (row, _output_paths(output_root, str(row["traj_id"]))["run_root"])
            for row in batch
            if _output_paths(output_root, str(row["traj_id"]))["run_root"].exists()
        ]
        protected = [
            str(path)
            for row, path in existing
            if str(row.get("candidate_kind", "optimized")) == "optimized"
        ]
        if protected and not force:
            raise RuntimeError(
                "Refusing to overwrite incomplete optimized Flow-Lenia A-run APF trajectory directories without --force. "
                "Inspect or move them first: "
                + ", ".join(protected[:10])
            )
        overwrite = bool(force or existing)
        log_event(
            f"Flow-Lenia A-run APF running batch {batch_idx}/{len(batches_to_run)} "
            f"kind={batch_kind} batch_size={len(batch)} overwrite={overwrite} traj_ids={batch_ids}",
            component="arun-apf",
        )
        if batch_kind == "optimizer_native":
            batch_legacy_sigma_collision = bool(
                batch_info.get("legacy_sigma_collision", optimized_native_legacy_sigma_collision)
            )
            simulate_optimizer_native_selected_batch(
                selected_batch=batch,
                cfg=rollout_cfg,
                flat_args=dict(flat_dict),
                output_root=output_root,
                overwrite=overwrite,
                legacy_sigma_collision=batch_legacy_sigma_collision,
            )
        else:
            simulate_batch(
                selected_batch=batch,
                cfg=rollout_cfg,
                flat_args=dict(flat_dict),
                output_root=output_root,
                overwrite=overwrite,
            )
        for row in command_rows:
            if row["traj_id"] in batch_ids:
                paths = _output_paths(output_root, str(row["traj_id"]))
                ready, message = _apf_status(
                    paths,
                    expected_steps=expected_steps,
                    expected_signature=expected_signature,
                )
                row["status"] = "exists" if ready else "missing_apf"
                row["message"] = message
        _write_manifest(
            output_root,
            rollout_config=rollout_config,
            selected=selected,
            command_rows=command_rows,
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )

    n_ready = 0
    for row in selected:
        ready, _message = _apf_status(
            _output_paths(output_root, str(row["traj_id"])),
            expected_steps=expected_steps,
            expected_signature=expected_signature,
        )
        n_ready += int(ready)
    status = "ok" if n_ready == len(selected) else "incomplete"
    summary = {
        "status": status,
        "section": section_key,
        "output_root": str(output_root),
        "rollout_config": str(rollout_config),
        "n_selected": len(selected),
        "n_ready": int(n_ready),
        "batch_size": int(batch_size),
        "rollout_steps": int(expected_steps),
        "manifest": str(output_root / "manifest.json"),
    }
    write_json(output_root / "simulation_summary.json", summary)
    _write_manifest(
        output_root,
        rollout_config=rollout_config,
        selected=selected,
        command_rows=command_rows,
        expected_steps=expected_steps,
        expected_signature=expected_signature,
    )
    if status != "ok" and bool(_get(section, "required", True)):
        raise RuntimeError(f"Flow-Lenia A-run APF generation incomplete: {n_ready}/{len(selected)} ready.")
    log_event(f"Flow-Lenia A-run APF done status={status} n_ready={n_ready}/{len(selected)}", component="arun-apf")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Flow-Lenia control-A style lagrangian + sparse APF rollouts.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--section-key",
        default="flow_lenia_arun_lagrangian_apf",
        help="simulation section to read from the paper-suite config",
    )
    args = parser.parse_args(argv)
    print(to_plain(run(args.config, force=args.force, dry_run=args.dry_run, section_key=args.section_key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
