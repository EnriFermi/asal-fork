from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

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


def _build_run_config(paper_cfg, config_path: Path, run_idx: int):
    stage_cfg = paper_cfg.get("optimization", {})
    base_cfg, _ = load_stage_base_config(stage_cfg, config_path.parent)
    if base_cfg.get("meta") is None:
        base_cfg.meta = OmegaConf.create()
    if base_cfg.get("logging") is None:
        base_cfg.logging = OmegaConf.create()

    paper_section = paper_cfg.get("paper_check", {})
    run_seed = int(paper_section.get("optimization_seed_base", 0)) + int(run_idx)

    save_root_rel = Path(str(stage_cfg.get("save_root", "experiments/paper_check/checkpoints/optimization")))
    run_save_dir_rel = save_root_rel / f"run_{int(run_idx):03d}"
    run_save_dir_abs = resolve_path(run_save_dir_rel, repo_root())
    ensure_dir(run_save_dir_abs)

    base_cfg.meta.seed = int(run_seed)
    base_cfg.meta.save_dir = str(run_save_dir_rel)
    base_cfg.meta.resume = bool(stage_cfg.get("resume", True))

    wandb_project = paper_cfg.get("meta", {}).get("wandb_project", None)
    if wandb_project is not None:
        base_cfg.logging.wandb_project = str(wandb_project)

    save_every = stage_cfg.get("save_every", None)
    if save_every is not None:
        base_cfg.logging.save_every = int(save_every)

    resolved_config_path = run_save_dir_abs / "optimization_config.yaml"
    return base_cfg, write_resolved_yaml(resolved_config_path, base_cfg)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_paper_check_optimization.py <paper_check_config.yaml>")

    paper_cfg, config_path = load_paper_check_config(sys.argv[1])
    machine_idx, num_machines = validate_machine_config(paper_cfg)
    total_runs = int(paper_cfg.get("paper_check", {}).get("num_optimizations", 1))
    if total_runs < 1:
        raise ValueError(f"paper_check.num_optimizations must be >= 1, got {total_runs}.")

    assigned = shard_indices(total_runs, machine_idx, num_machines)
    print(
        f"[paper_check/optimization] machine_idx={machine_idx} num_machines={num_machines} "
        f"assigned_runs={assigned}"
    )

    repo = repo_root()
    script_path = repo / "scripts" / "main_opt_msc.py"
    for run_idx in assigned:
        _, resolved_config_path = _build_run_config(paper_cfg, config_path, run_idx)
        print(
            f"[paper_check/optimization] starting run_idx={run_idx} "
            f"config={resolved_config_path}"
        )
        subprocess.run(
            [sys.executable, str(script_path), str(resolved_config_path)],
            cwd=str(repo),
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
