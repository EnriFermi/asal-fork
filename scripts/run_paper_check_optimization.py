from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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


def _explicit_run_indices(paper_section) -> list[int] | None:
    raw = paper_section.get("run_indices", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    out = [int(v) for v in values]
    if not out:
        raise ValueError("paper_check.run_indices was provided but is empty.")
    if len(set(out)) != len(out):
        raise ValueError(f"paper_check.run_indices contains duplicates: {out}")
    if any(v < 0 for v in out):
        raise ValueError(f"paper_check.run_indices must be non-negative, got {out}")
    return out


def _entrypoint_command(stage_cfg, repo: Path, resolved_config_path: Path) -> list[str]:
    raw = stage_cfg.get("entrypoint", stage_cfg.get("objective", "msc"))
    name = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "msc": "msc",
        "main_opt_msc": "msc",
        "delta_h": "msc",
        "deltah": "msc",
        "clip": "clip_oe",
        "clip_oe": "clip_oe",
        "oe": "clip_oe",
        "oe_loss": "clip_oe",
        "main_opt": "clip_oe",
        "asal": "clip_oe",
    }
    if name not in aliases:
        raise ValueError(
            "Unknown optimization.entrypoint/objective "
            f"{raw!r}. Use 'msc' or 'clip_oe'."
        )
    entrypoint = aliases[name]
    if entrypoint == "msc":
        return [
            sys.executable,
            str(repo / "scripts" / "main_opt_msc.py"),
            str(resolved_config_path),
        ]
    return [
        sys.executable,
        str(repo / "scripts" / "run_main_opt_from_yaml.py"),
        str(resolved_config_path),
    ]


def _build_run_config(paper_cfg, config_path: Path, run_idx: int):
    stage_cfg = paper_cfg.get("optimization", {})
    base_cfg, _ = load_stage_base_config(stage_cfg, config_path.parent)
    overrides = stage_cfg.get("overrides", None)
    if overrides:
        base_cfg = OmegaConf.merge(base_cfg, overrides)
    if base_cfg.get("meta") is None:
        base_cfg.meta = OmegaConf.create()
    if base_cfg.get("logging") is None:
        base_cfg.logging = OmegaConf.create()

    paper_section = paper_cfg.get("paper_check", {})
    run_seed = int(paper_section.get("optimization_seed_base", 0)) + int(run_idx)

    save_root_rel = Path(str(stage_cfg.get("save_root", "experiments/paper_check_flow_lenia/checkpoints/optimization")))
    run_save_dir_rel = save_root_rel / f"run_{int(run_idx):03d}"
    run_save_dir_abs = resolve_path(run_save_dir_rel, repo_root())
    ensure_dir(run_save_dir_abs)

    base_cfg.meta.seed = int(run_seed)
    base_cfg.meta.save_dir = str(run_save_dir_rel)
    base_cfg.meta.resume = bool(stage_cfg.get("resume", True))

    wandb_project = paper_cfg.get("meta", {}).get("wandb_project", None)
    if wandb_project is not None:
        base_cfg.logging.wandb_project = str(wandb_project)
    wandb_mode = paper_cfg.get("meta", {}).get("wandb_mode", None)
    if wandb_mode is not None:
        base_cfg.logging.wandb_mode = str(wandb_mode)

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
    paper_section = paper_cfg.get("paper_check", {})
    explicit_runs = _explicit_run_indices(paper_section)
    if explicit_runs is None:
        total_runs = int(paper_section.get("num_optimizations", 1))
        if total_runs < 1:
            raise ValueError(f"paper_check.num_optimizations must be >= 1, got {total_runs}.")
        assigned = shard_indices(total_runs, machine_idx, num_machines)
    else:
        assigned = [idx for pos, idx in enumerate(explicit_runs) if pos % num_machines == machine_idx]
    print(
        f"[paper_check/optimization] machine_idx={machine_idx} num_machines={num_machines} "
        f"assigned_runs={assigned}"
    )

    repo = repo_root()
    stage_cfg = paper_cfg.get("optimization", {})
    for run_idx in assigned:
        _, resolved_config_path = _build_run_config(paper_cfg, config_path, run_idx)
        cmd = _entrypoint_command(stage_cfg, repo, resolved_config_path)
        print(
            f"[paper_check/optimization] starting run_idx={run_idx} "
            f"config={resolved_config_path} command={' '.join(cmd)}"
        )
        subprocess.run(
            cmd,
            cwd=str(repo),
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
