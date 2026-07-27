from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

import substrates
import util
from generate_random_best import _sample_params_sep_cma_es_ask
from paper_check_common import (
    ensure_dir,
    load_paper_check_config,
    load_stage_base_config,
    repo_root,
    resolve_path,
    shard_indices,
    validate_machine_config,
    write_resolved_yaml,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for non-interactive environments.
    tqdm = None


def _flat_opt_args(cfg):
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _trial_name(trial_idx: int) -> str:
    return f"trial_{int(trial_idx):05d}"


def _chunk_list(items: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _make_progress_bar(*, total: int, desc: str):
    if tqdm is None or total <= 0:
        return None
    return tqdm(
        total=int(total),
        desc=desc,
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
        ascii=True,
    )


def _group_pair_seeds(paper_cfg, group_idx: int) -> tuple[int, int]:
    base = int(paper_cfg.get("paper_check", {}).get("pair_seed_base", 200_000))
    seed_x = base + 2 * int(group_idx)
    seed_x1 = base + 2 * int(group_idx) + 1
    return seed_x, seed_x1


def _metric_seed(paper_cfg, trial_idx: int) -> int:
    base = int(paper_cfg.get("paper_check", {}).get("metric_seed_base", 2_000_000))
    return base + int(trial_idx)


def _resolve_manifest_path(root: Path, raw) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _load_training_references(manifest_path: Path) -> dict[tuple[int, str, int, int], dict]:
    payload = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    references: dict[tuple[int, str, int, int], dict] = {}
    for row in payload.get("trajectories", []):
        source_run_idx = int(row.get("source_run_idx", row.get("suite_run_idx", -1)))
        candidate_kind = str(row.get("candidate_kind", "optimized")).strip().lower()
        candidate_idx = int(row.get("candidate_idx", 0))
        run_seed = int(row.get("run_seed", -1))
        key = (source_run_idx, candidate_kind, candidate_idx, run_seed)
        if key in references:
            raise ValueError(f"Duplicate C5 training reference key {key} in {manifest_path}.")
        references[key] = {
            **dict(row),
            "apf_dir": str(_resolve_manifest_path(root, row["apf_dir"])),
            "params_path": str(_resolve_manifest_path(root, row["params_path"])),
        }
    return references


def _training_reference(
    references: dict[tuple[int, str, int, int], dict],
    *,
    group_idx: int,
    candidate_kind: str,
    candidate_idx: int,
    run_seed: int,
) -> dict:
    key = (int(group_idx), str(candidate_kind).strip().lower(), int(candidate_idx), int(run_seed))
    if key not in references:
        raise KeyError(f"Missing C5 training reference {key}.")
    return references[key]


def _optimizer_native_resume_metadata(reference: dict, *, step: int) -> dict[str, int | str]:
    apf_dir = Path(str(reference["apf_dir"]))
    for path in sorted(apf_dir.glob("P_steps_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            matches = np.flatnonzero(steps == int(step))
            if matches.size != 1:
                continue
            idx = int(matches[0])
            batch_size = int(np.asarray(data["resume_batch_size"])[idx])
            batch_index = int(np.asarray(data["resume_batch_index"])[idx])
            seed_count = int(reference.get("rollout_seed_count", 0))
            if seed_count < 1 or batch_size % seed_count != 0:
                raise ValueError(
                    f"Invalid optimizer-native batch metadata in {path}: "
                    f"batch_size={batch_size}, seed_count={seed_count}."
                )
            return {
                "reference_apf_file": str(path),
                "batch_size": batch_size,
                "batch_index": batch_index,
                "population_size": int(batch_size // seed_count),
                "seed_count": seed_count,
                "execution_pop_idx": int(batch_index // seed_count),
                "seed_idx": int(batch_index % seed_count),
            }
    raise FileNotFoundError(
        f"No optimizer-native resume metadata for step={step} under {apf_dir}."
    )


def _optimizer_native_init_identity(reference: dict) -> tuple[Path, int, int]:
    raw_pop_path = reference.get("optimizer_native_source_pop_traj", None)
    if raw_pop_path in (None, ""):
        raise ValueError(
            f"Training reference {reference.get('traj_id')} has no optimizer_native_source_pop_traj."
        )
    pop_path = Path(str(raw_pop_path)).resolve()
    optimizer_iter = int(reference.get("optimizer_native_iter", -1))
    optimizer_pop_idx = int(reference.get("optimizer_native_pop_idx", -1))
    if optimizer_iter < 0 or optimizer_pop_idx < 0:
        raise ValueError(
            f"Training reference {reference.get('traj_id')} has invalid optimizer-native "
            f"iter/pop_idx={optimizer_iter}/{optimizer_pop_idx}."
        )
    return pop_path, optimizer_iter, optimizer_pop_idx


def _materialize_optimizer_native_init_params(
    *,
    control_a_reference: dict,
    control_b_reference: dict,
    trial_artifact_dir: Path,
    cache: dict[tuple[Path, int, int], np.ndarray],
) -> Path:
    identity_a = _optimizer_native_init_identity(control_a_reference)
    identity_b = _optimizer_native_init_identity(control_b_reference)
    if identity_a != identity_b:
        raise ValueError(
            "Control A/B references disagree on optimizer-native init identity: "
            f"{identity_a} != {identity_b}."
        )
    pop_path, optimizer_iter, optimizer_pop_idx = identity_a
    if not pop_path.exists():
        raise FileNotFoundError(f"Optimizer-native source pop_traj not found: {pop_path}.")
    if identity_a not in cache:
        with pop_path.open("rb") as f:
            pop = pickle.load(f)
        params = np.asarray(pop["params"], dtype=np.float32)
        if optimizer_iter >= params.shape[0] or optimizer_pop_idx >= params.shape[1]:
            raise IndexError(
                f"Optimizer-native init index [{optimizer_iter}, {optimizer_pop_idx}] "
                f"is outside params shape {params.shape} in {pop_path}."
            )
        cache[identity_a] = np.asarray(
            params[optimizer_iter, optimizer_pop_idx],
            dtype=np.float32,
        ).copy()

    init_params = cache[identity_a]
    init_params_path = trial_artifact_dir / "optimizer_native_init_params.npy"
    if init_params_path.exists():
        existing = np.asarray(np.load(init_params_path), dtype=np.float32)
        if not np.array_equal(existing, init_params):
            raise RuntimeError(
                f"Existing optimizer-native init params disagree with source: {init_params_path}."
            )
    else:
        np.save(init_params_path, init_params)
    _write_json(
        trial_artifact_dir / "optimizer_native_init_params_provenance.json",
        {
            "source_pop_traj": str(pop_path),
            "optimizer_iter": int(optimizer_iter),
            "optimizer_pop_idx": int(optimizer_pop_idx),
            "control_a_reference": str(control_a_reference.get("traj_id", "")),
            "control_b_reference": str(control_b_reference.get("traj_id", "")),
            "n_params": int(init_params.size),
        },
    )
    return init_params_path


def _canonicalize_random_mean_init(name) -> str:
    normalized = str(name or "strategy_default").strip().lower().replace("-", "_")
    aliases = {
        "strategy_default": "strategy_default",
        "optimizer_default": "strategy_default",
        "default": "strategy_default",
        "substrate_default": "substrate_default",
        "default_params": "substrate_default",
        "smart": "substrate_default",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown params_init {name!r}. Use 'strategy_default' or 'substrate_default'."
        )
    return aliases[normalized]


def _build_trial_artifact_dir(save_root: Path, trial_idx: int) -> Path:
    return ensure_dir(save_root / "trial_artifacts" / _trial_name(trial_idx))


def _ensure_random_checkpoint(
    *,
    random_dir: Path,
    substrate,
    sigma_init: float,
    pop_size: int,
    param_seed: int,
    member_idx: int,
    group_idx: int,
    random_idx: int,
    mean_init_mode: str = "strategy_default",
) -> None:
    best_path = random_dir / "best.pkl"
    if best_path.exists():
        return
    ensure_dir(random_dir)
    params = _sample_params_sep_cma_es_ask(
        substrate,
        seed=int(param_seed),
        sigma_init=float(sigma_init),
        pop_size=int(pop_size),
        member_idx=int(member_idx),
        mean_init_mode=str(mean_init_mode),
    )
    util.save_pkl(str(random_dir), "best", (np.asarray(params, dtype=np.float32), 0.0))
    _write_json(
        random_dir / "metadata.json",
        {
            "group_idx": int(group_idx),
            "random_idx": int(random_idx),
            "param_seed": int(param_seed),
            "member_idx": int(member_idx),
            "pop_size": int(pop_size),
            "sigma_init": float(sigma_init),
            "mean_init_mode": str(mean_init_mode),
        },
    )


def _rebuild_local_summary(save_root: Path) -> None:
    rows = []
    trial_data_dir = save_root / "trial_data"
    if trial_data_dir.exists():
        for path in sorted(trial_data_dir.glob("trial_*.json")):
            with path.open("r") as f:
                rows.append(json.load(f))
    rows = sorted(rows, key=lambda item: int(item["trial_idx"]))
    _save_csv(save_root / "trial_results.csv", rows)
    summary = {
        "n_trials_local": int(len(rows)),
        "save_dir": str(save_root),
        "n_optimized_local": int(sum(row.get("candidate_kind") == "optimized" for row in rows)),
        "n_random_local": int(sum(row.get("candidate_kind") == "random" for row in rows)),
    }
    _write_json(save_root / "summary.json", summary)


def _cfg_get(cfg, path: tuple[str, ...]):
    node = cfg
    for key in path:
        node = node.get(key, None)
        if node is None:
            return None
    return node


def _completed_trial_matches_current_config(
    *,
    save_root: Path,
    trial_idx: int,
    expected_cfg,
) -> tuple[bool, str]:
    resolved_path = save_root / "trial_artifacts" / _trial_name(trial_idx) / "resolved_config.yaml"
    if not resolved_path.exists():
        return False, f"missing resolved config {resolved_path}"
    try:
        saved_cfg = OmegaConf.load(resolved_path)
    except Exception as exc:
        return False, f"cannot read resolved config {resolved_path}: {exc}"
    keys = (
        ("substrate", "substrate"),
        ("substrate", "grid_size"),
        ("substrate", "seed_n_patches"),
        ("substrate", "seed_patch_size"),
        ("protocol", "grid_split"),
        ("protocol", "warmup_steps"),
        ("protocol", "total_steps"),
        ("evaluation", "late_window_start_steps"),
        ("evaluation", "late_window_end_steps"),
        ("metric", "sample_every_steps"),
        ("metric", "metric_window_size_steps"),
        ("metric", "metric_tau_steps"),
    )
    for path in keys:
        saved = _cfg_get(saved_cfg, path)
        expected = _cfg_get(expected_cfg, path)
        if saved != expected:
            dotted = ".".join(path)
            return False, f"{dotted}={saved!r}, expected {expected!r}"
    return True, ""


def _build_job_config(
    *,
    paper_cfg,
    config_path: Path,
    base_cfg,
    save_root_rel: Path,
    param_checkpoint_rel: Path,
    trial_idx: int,
    group_idx: int,
    candidate_kind: str,
    candidate_idx: int,
    candidate_label: str,
    init_params_path: Path,
    seed_x: int,
    seed_x1: int,
    metric_seed: int,
    control_a_reference: dict | None = None,
    control_b_reference: dict | None = None,
    training_reference_step: int | None = None,
    random_param_seed: int | None = None,
    random_member_idx: int | None = None,
):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if cfg.get("meta") is None:
        cfg.meta = OmegaConf.create()
    if cfg.get("source") is None:
        cfg.source = OmegaConf.create()
    if cfg.get("evaluation") is None:
        cfg.evaluation = OmegaConf.create()
    if cfg.get("logging") is None:
        cfg.logging = OmegaConf.create()
    if cfg.get("job") is None:
        cfg.job = OmegaConf.create()

    stage_cfg = paper_cfg.get("frustration_simulation", {})
    cfg.meta.save_dir = str(save_root_rel)
    cfg.meta.trial_idx = int(trial_idx)
    cfg.meta.optimized_run_idx = int(group_idx)
    cfg.meta.candidate_kind = str(candidate_kind)
    cfg.meta.candidate_idx = int(candidate_idx)
    cfg.meta.candidate_label = str(candidate_label)

    cfg.source.checkpoint_dir = str(param_checkpoint_rel)
    cfg.source.params_name = "best"
    cfg.source.params_path = None
    cfg.source.init_params_path = str(init_params_path)

    cfg.evaluation.resume = bool(stage_cfg.get("resume", True))
    cfg.evaluation.checkpoint_every_steps = int(stage_cfg.get("checkpoint_every_steps", 5_000))
    cfg.evaluation.full_embedding_sample_every_steps = int(
        stage_cfg.get("full_embedding_sample_every_steps", cfg.get("metric", {}).get("sample_every_steps", 1_000))
    )
    cfg.evaluation.continuation_full_embedding_sample_every_steps = int(
        stage_cfg.get(
            "continuation_full_embedding_sample_every_steps",
            cfg.evaluation.full_embedding_sample_every_steps,
        )
    )
    cfg.evaluation.log_full_embeddings_for_b = bool(stage_cfg.get("log_full_embeddings_for_b", False))
    cfg.evaluation.run_seed_protocol = str(stage_cfg.get("run_seed_protocol", "legacy"))
    cfg.evaluation.training_horizon_steps = int(
        stage_cfg.get("training_horizon_steps", training_reference_step or 0)
    )
    cfg.evaluation.require_training_reference_match = bool(
        stage_cfg.get("require_training_reference_match", False)
    )
    cfg.evaluation.training_reference_only = bool(
        stage_cfg.get("training_reference_only", False)
    )
    bootstrap_cache_root = stage_cfg.get("bootstrap_cache_root", None)
    if bootstrap_cache_root is not None:
        cfg.evaluation.bootstrap_cache_root = str(bootstrap_cache_root)
    cfg.evaluation.wall_video_enabled = bool(stage_cfg.get("wall_video_enabled", False))
    cfg.evaluation.wall_video_sample_every_steps = int(
        stage_cfg.get("wall_video_sample_every_steps", 5_000)
    )
    cfg.evaluation.wall_video_img_size = int(stage_cfg.get("wall_video_img_size", 256))
    cfg.evaluation.wall_video_fps = float(stage_cfg.get("wall_video_fps", 24.0))
    cfg.evaluation.wall_video_codec = str(stage_cfg.get("wall_video_codec", "libx264"))
    cfg.evaluation.wall_video_keep_frames = bool(stage_cfg.get("wall_video_keep_frames", False))

    wandb_project = paper_cfg.get("meta", {}).get("wandb_project", None)
    if wandb_project is not None:
        cfg.logging.wandb_project = str(wandb_project)
    wandb_mode = paper_cfg.get("meta", {}).get("wandb_mode", None)
    if wandb_mode is not None:
        cfg.logging.wandb_mode = str(wandb_mode)

    cfg.job.seed_x = int(seed_x)
    cfg.job.seed_x1 = int(seed_x1)
    cfg.job.metric_seed = int(metric_seed)
    if control_a_reference is not None:
        cfg.job.control_a_reference_apf_dir = str(control_a_reference["apf_dir"])
        cfg.job.control_a_reference_params_path = str(control_a_reference["params_path"])
    if control_b_reference is not None:
        cfg.job.control_b_reference_apf_dir = str(control_b_reference["apf_dir"])
        cfg.job.control_b_reference_params_path = str(control_b_reference["params_path"])
    if training_reference_step is not None:
        cfg.job.training_reference_step = int(training_reference_step)
    if control_a_reference is not None and control_b_reference is not None and training_reference_step is not None:
        resume_a = _optimizer_native_resume_metadata(
            control_a_reference,
            step=training_reference_step,
        )
        resume_b = _optimizer_native_resume_metadata(
            control_b_reference,
            step=training_reference_step,
        )
        identity_a = _optimizer_native_init_identity(control_a_reference)
        identity_b = _optimizer_native_init_identity(control_b_reference)
        if identity_a != identity_b:
            raise ValueError(
                f"Control references disagree on optimizer-native source identity: "
                f"{identity_a} != {identity_b}."
            )
        if (
            int(resume_a["population_size"]) != int(resume_b["population_size"])
            or int(resume_a["seed_count"]) != int(resume_b["seed_count"])
            or int(resume_a["execution_pop_idx"]) != int(resume_b["execution_pop_idx"])
        ):
            raise ValueError(
                "Control references disagree on optimizer-native execution context: "
                f"A={resume_a}, B={resume_b}."
            )
        cfg.job.optimizer_native_source_pop_traj = str(identity_a[0])
        cfg.job.optimizer_native_iter = int(identity_a[1])
        cfg.job.optimizer_native_source_pop_idx = int(identity_a[2])
        cfg.job.optimizer_native_use_row_params = bool(
            control_a_reference.get("optimizer_native_use_row_params", False)
        )
        cfg.job.optimizer_native_population_size = int(resume_a["population_size"])
        cfg.job.optimizer_native_seed_count = int(resume_a["seed_count"])
        cfg.job.optimizer_native_execution_pop_idx = int(resume_a["execution_pop_idx"])
        cfg.job.control_a_optimizer_native_seed_idx = int(resume_a["seed_idx"])
        cfg.job.control_b_optimizer_native_seed_idx = int(resume_b["seed_idx"])
    if random_param_seed is not None:
        cfg.job.random_param_seed = int(random_param_seed)
    if random_member_idx is not None:
        cfg.job.random_member_idx = int(random_member_idx)

    trial_artifact_dir = _build_trial_artifact_dir(resolve_path(save_root_rel, repo_root()), trial_idx)
    resolved_config_path = trial_artifact_dir / "job_config.yaml"
    return write_resolved_yaml(resolved_config_path, cfg)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_paper_check_frustration.py <paper_check_config.yaml>")

    paper_cfg, config_path = load_paper_check_config(sys.argv[1])
    machine_idx, num_machines = validate_machine_config(paper_cfg)
    paper_section = paper_cfg.get("paper_check", {})
    total_groups = int(paper_section.get("num_optimizations", 1))
    num_random = int(paper_section.get("num_random_baselines", 1))
    if total_groups < 1:
        raise ValueError(f"paper_check.num_optimizations must be >= 1, got {total_groups}.")
    if num_random < 0:
        raise ValueError(f"paper_check.num_random_baselines must be >= 0, got {num_random}.")

    assigned_groups = shard_indices(total_groups, machine_idx, num_machines)
    print(
        f"[paper_check/frustration] machine_idx={machine_idx} num_machines={num_machines} "
        f"assigned_groups={assigned_groups}"
    )

    repo = repo_root()
    batch_eval_script = repo / "scripts" / "paper_check_frustration_batch_eval.py"

    stage_cfg = paper_cfg.get("frustration_simulation", {})
    save_root_rel = Path(str(stage_cfg.get("save_root", "experiments/paper_check_flow_lenia/checkpoints/frustration_simulation")))
    save_root_abs = resolve_path(save_root_rel, repo)
    ensure_dir(save_root_abs)
    trial_batch_size = int(stage_cfg.get("trial_batch_size", 1))
    if trial_batch_size < 1:
        raise ValueError(f"frustration_simulation.trial_batch_size must be >= 1, got {trial_batch_size}.")

    base_hist_cfg, _ = load_stage_base_config(stage_cfg, config_path.parent)
    opt_stage_cfg = paper_cfg.get("optimization", {})
    opt_save_root_rel = Path(str(opt_stage_cfg.get("save_root", "experiments/paper_check_flow_lenia/checkpoints/optimization")))
    opt_save_root_abs = resolve_path(opt_save_root_rel, repo)

    opt_base_cfg, _ = load_stage_base_config(opt_stage_cfg, config_path.parent)
    opt_flat = _flat_opt_args(opt_base_cfg)
    opt_args = SimpleNamespace(**OmegaConf.to_container(opt_flat, resolve=True))
    substrate = substrates.create_substrate(
        opt_args.substrate,
        **util.substrate_kwargs_from_args(opt_args),
    )
    substrate = substrates.FlattenSubstrateParameters(substrate)
    external_random_root_raw = stage_cfg.get("random_checkpoint_root", None)
    if external_random_root_raw is None:
        random_root = ensure_dir(save_root_abs / "random_params")
    else:
        random_root = resolve_path(external_random_root_raw, repo)
        if random_root is None or not random_root.exists():
            raise FileNotFoundError(f"Configured random_checkpoint_root does not exist: {random_root}.")
    require_existing_random = bool(stage_cfg.get("require_existing_random_checkpoints", False))
    training_manifest_raw = stage_cfg.get("training_reference_manifest", None)
    training_references: dict[tuple[int, str, int, int], dict] = {}
    training_reference_step = None
    if training_manifest_raw is not None:
        training_manifest_path = resolve_path(training_manifest_raw, repo)
        if training_manifest_path is None or not training_manifest_path.exists():
            raise FileNotFoundError(f"Configured training_reference_manifest does not exist: {training_manifest_path}.")
        training_references = _load_training_references(training_manifest_path)
        training_reference_step = int(stage_cfg.get("training_reference_step", 300_000))
    random_seed_base = int(paper_section.get("random_param_seed_base", 500_000))
    opt_pop_size = int(getattr(opt_args, "pop_size"))
    opt_sigma = float(getattr(opt_args, "sigma"))
    opt_params_init = _canonicalize_random_mean_init(getattr(opt_args, "params_init", "strategy_default"))

    pending_job_cfgs: list[Path] = []
    init_params_cache: dict[tuple[Path, int, int], np.ndarray] = {}
    for group_idx in assigned_groups:
        seed_x, seed_x1 = _group_pair_seeds(paper_cfg, group_idx)
        optimized_checkpoint_rel = opt_save_root_rel / f"run_{int(group_idx):03d}"
        optimized_checkpoint_abs = resolve_path(optimized_checkpoint_rel, repo)
        if optimized_checkpoint_abs is None or not (optimized_checkpoint_abs / "best.pkl").exists():
            raise FileNotFoundError(
                f"Optimized checkpoint missing for group {group_idx}: expected {optimized_checkpoint_abs / 'best.pkl'}."
            )
        fallback_init_params_path = optimized_checkpoint_abs / "params.npy"
        if not fallback_init_params_path.exists():
            raise FileNotFoundError(
                "Fallback initialization params missing for group "
                f"{group_idx}: {fallback_init_params_path}."
            )

        candidate_specs = [
            dict(
                trial_idx=int(group_idx) * (num_random + 1),
                candidate_kind="optimized",
                candidate_idx=0,
                candidate_label="optimized",
                checkpoint_rel=optimized_checkpoint_rel,
                random_param_seed=None,
                random_member_idx=None,
            )
        ]

        group_random_root = ensure_dir(random_root / f"group_{int(group_idx):03d}")
        for random_idx in range(num_random):
            pop_round = int(random_idx // opt_pop_size)
            member_idx = int(random_idx % opt_pop_size)
            param_seed = int(random_seed_base + group_idx * 10_000 + pop_round)
            random_dir_abs = group_random_root / f"random_{int(random_idx):03d}"
            if require_existing_random:
                if not (random_dir_abs / "best.pkl").exists():
                    raise FileNotFoundError(
                        f"Required existing random checkpoint missing: {random_dir_abs / 'best.pkl'}."
                    )
            else:
                _ensure_random_checkpoint(
                    random_dir=random_dir_abs,
                    substrate=substrate,
                    sigma_init=opt_sigma,
                    pop_size=opt_pop_size,
                    param_seed=param_seed,
                    member_idx=member_idx,
                    group_idx=group_idx,
                    random_idx=random_idx,
                    mean_init_mode=opt_params_init,
                )
            checkpoint_rel = Path(str(random_dir_abs.relative_to(repo)))
            candidate_specs.append(
                dict(
                    trial_idx=int(group_idx) * (num_random + 1) + random_idx + 1,
                    candidate_kind="random",
                    candidate_idx=int(random_idx),
                    candidate_label=f"random_{int(random_idx):03d}",
                    checkpoint_rel=checkpoint_rel,
                    random_param_seed=param_seed,
                    random_member_idx=member_idx,
                )
            )

        for spec in candidate_specs:
            trial_row_json = save_root_abs / "trial_data" / f"{_trial_name(spec['trial_idx'])}.json"
            if trial_row_json.exists():
                matches, reason = _completed_trial_matches_current_config(
                    save_root=save_root_abs,
                    trial_idx=int(spec["trial_idx"]),
                    expected_cfg=base_hist_cfg,
                )
                if not matches:
                    raise RuntimeError(
                        "Refusing to reuse stale Flow-Lenia frustration trial "
                        f"trial_idx={spec['trial_idx']} at {trial_row_json}: {reason}. "
                        "Use a fresh save_root or remove the stale trial artifacts."
                    )
                print(f"[paper_check/frustration] skipping completed trial_idx={spec['trial_idx']}")
                continue
            control_a_reference = None
            control_b_reference = None
            if training_references:
                control_a_reference = _training_reference(
                    training_references,
                    group_idx=group_idx,
                    candidate_kind=str(spec["candidate_kind"]),
                    candidate_idx=int(spec["candidate_idx"]),
                    run_seed=seed_x,
                )
                control_b_reference = _training_reference(
                    training_references,
                    group_idx=group_idx,
                    candidate_kind=str(spec["candidate_kind"]),
                    candidate_idx=int(spec["candidate_idx"]),
                    run_seed=seed_x1,
                )
                init_params_path = _materialize_optimizer_native_init_params(
                    control_a_reference=control_a_reference,
                    control_b_reference=control_b_reference,
                    trial_artifact_dir=_build_trial_artifact_dir(
                        save_root_abs,
                        int(spec["trial_idx"]),
                    ),
                    cache=init_params_cache,
                )
            else:
                init_params_path = fallback_init_params_path
            resolved_job_cfg = _build_job_config(
                paper_cfg=paper_cfg,
                config_path=config_path,
                base_cfg=base_hist_cfg,
                save_root_rel=save_root_rel,
                param_checkpoint_rel=spec["checkpoint_rel"],
                trial_idx=int(spec["trial_idx"]),
                group_idx=int(group_idx),
                candidate_kind=str(spec["candidate_kind"]),
                candidate_idx=int(spec["candidate_idx"]),
                candidate_label=str(spec["candidate_label"]),
                init_params_path=init_params_path,
                seed_x=int(seed_x),
                seed_x1=int(seed_x1),
                metric_seed=_metric_seed(paper_cfg, int(spec["trial_idx"])),
                control_a_reference=control_a_reference,
                control_b_reference=control_b_reference,
                training_reference_step=training_reference_step,
                random_param_seed=spec["random_param_seed"],
                random_member_idx=spec["random_member_idx"],
            )
            print(
                f"[paper_check/frustration] queued group_idx={group_idx} "
                f"trial_idx={spec['trial_idx']} label={spec['candidate_label']}"
            )
            pending_job_cfgs.append(resolved_job_cfg)

    batches = _chunk_list(pending_job_cfgs, trial_batch_size)
    print(
        f"[paper_check/frustration] pending_trials={len(pending_job_cfgs)} "
        f"trial_batch_size={trial_batch_size} n_batches={len(batches)}"
    )
    pbar = _make_progress_bar(total=len(pending_job_cfgs), desc="frustration trials")
    try:
        for batch_idx, job_cfg_batch in enumerate(batches):
            trial_ids = [int(OmegaConf.load(path).get("meta", {}).get("trial_idx")) for path in job_cfg_batch]
            print(
                f"[paper_check/frustration] starting batch_idx={batch_idx} "
                f"batch_size={len(job_cfg_batch)} trial_ids={trial_ids}"
            )
            subprocess.run(
                [sys.executable, str(batch_eval_script), *[str(path) for path in job_cfg_batch]],
                cwd=str(repo),
                check=True,
            )
            if pbar is not None:
                pbar.update(len(job_cfg_batch))
            _rebuild_local_summary(save_root_abs)
    finally:
        if pbar is not None:
            pbar.close()

    _rebuild_local_summary(save_root_abs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
