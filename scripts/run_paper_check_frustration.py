from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _group_pair_seeds(paper_cfg, group_idx: int) -> tuple[int, int]:
    base = int(paper_cfg.get("paper_check", {}).get("pair_seed_base", 200_000))
    seed_x = base + 2 * int(group_idx)
    seed_x1 = base + 2 * int(group_idx) + 1
    return seed_x, seed_x1


def _metric_seed(paper_cfg, trial_idx: int) -> int:
    base = int(paper_cfg.get("paper_check", {}).get("metric_seed_base", 2_000_000))
    return base + int(trial_idx)


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
    seed_x: int,
    seed_x1: int,
    metric_seed: int,
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

    cfg.evaluation.resume = bool(stage_cfg.get("resume", True))
    cfg.evaluation.checkpoint_every_steps = int(stage_cfg.get("checkpoint_every_steps", 5_000))
    cfg.evaluation.full_embedding_sample_every_steps = int(
        stage_cfg.get("full_embedding_sample_every_steps", cfg.get("metric", {}).get("sample_every_steps", 1_000))
    )
    cfg.evaluation.log_full_embeddings_for_b = bool(stage_cfg.get("log_full_embeddings_for_b", False))

    wandb_project = paper_cfg.get("meta", {}).get("wandb_project", None)
    if wandb_project is not None:
        cfg.logging.wandb_project = str(wandb_project)
    wandb_mode = paper_cfg.get("meta", {}).get("wandb_mode", None)
    if wandb_mode is not None:
        cfg.logging.wandb_mode = str(wandb_mode)

    cfg.job.seed_x = int(seed_x)
    cfg.job.seed_x1 = int(seed_x1)
    cfg.job.metric_seed = int(metric_seed)
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
    save_root_rel = Path(str(stage_cfg.get("save_root", "experiments/paper_check/checkpoints/frustration_simulation")))
    save_root_abs = resolve_path(save_root_rel, repo)
    ensure_dir(save_root_abs)
    trial_batch_size = int(stage_cfg.get("trial_batch_size", 1))
    if trial_batch_size < 1:
        raise ValueError(f"frustration_simulation.trial_batch_size must be >= 1, got {trial_batch_size}.")

    base_hist_cfg, _ = load_stage_base_config(stage_cfg, config_path.parent)
    opt_stage_cfg = paper_cfg.get("optimization", {})
    opt_save_root_rel = Path(str(opt_stage_cfg.get("save_root", "experiments/paper_check/checkpoints/optimization")))
    opt_save_root_abs = resolve_path(opt_save_root_rel, repo)

    opt_base_cfg, _ = load_stage_base_config(opt_stage_cfg, config_path.parent)
    opt_flat = _flat_opt_args(opt_base_cfg)
    opt_args = SimpleNamespace(**OmegaConf.to_container(opt_flat, resolve=True))
    substrate = substrates.create_substrate(
        opt_args.substrate,
        **util.substrate_kwargs_from_args(opt_args),
    )
    substrate = substrates.FlattenSubstrateParameters(substrate)
    random_root = ensure_dir(save_root_abs / "random_params")
    random_seed_base = int(paper_section.get("random_param_seed_base", 500_000))
    opt_pop_size = int(getattr(opt_args, "pop_size"))
    opt_sigma = float(getattr(opt_args, "sigma"))
    opt_params_init = _canonicalize_random_mean_init(getattr(opt_args, "params_init", "strategy_default"))

    pending_job_cfgs: list[Path] = []
    for group_idx in assigned_groups:
        seed_x, seed_x1 = _group_pair_seeds(paper_cfg, group_idx)
        optimized_checkpoint_rel = opt_save_root_rel / f"run_{int(group_idx):03d}"
        optimized_checkpoint_abs = resolve_path(optimized_checkpoint_rel, repo)
        if optimized_checkpoint_abs is None or not (optimized_checkpoint_abs / "best.pkl").exists():
            raise FileNotFoundError(
                f"Optimized checkpoint missing for group {group_idx}: expected {optimized_checkpoint_abs / 'best.pkl'}."
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
                print(f"[paper_check/frustration] skipping completed trial_idx={spec['trial_idx']}")
                continue
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
                seed_x=int(seed_x),
                seed_x1=int(seed_x1),
                metric_seed=_metric_seed(paper_cfg, int(spec["trial_idx"])),
                random_param_seed=spec["random_param_seed"],
                random_member_idx=spec["random_member_idx"],
            )
            print(
                f"[paper_check/frustration] queued group_idx={group_idx} "
                f"trial_idx={spec['trial_idx']} label={spec['candidate_label']}"
            )
            pending_job_cfgs.append(resolved_job_cfg)

    for batch_idx, job_cfg_batch in enumerate(_chunk_list(pending_job_cfgs, trial_batch_size)):
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
        _rebuild_local_summary(save_root_abs)

    _rebuild_local_summary(save_root_abs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
