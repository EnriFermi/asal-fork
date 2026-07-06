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


def _values_from_index_list(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    return [int(v) for v in values]


def _explicit_run_indices_by_machine(paper_section, *, machine_idx: int, num_machines: int) -> list[int] | None:
    raw = paper_section.get("run_indices_by_machine", None)
    if raw is None:
        raw = paper_section.get("machine_run_indices", None)
    if raw is None:
        return None
    if paper_section.get("run_indices", None) is not None:
        raise ValueError("Use either paper_check.run_indices or paper_check.run_indices_by_machine, not both.")

    if isinstance(raw, str):
        raise ValueError(
            "paper_check.run_indices_by_machine must be a mapping/list, for example "
            "{0: [0,2,4], 1: [1,3]}."
        )

    by_machine: dict[int, list[int]] = {}
    if isinstance(raw, (list, tuple)) or OmegaConf.is_list(raw):
        for idx, values in enumerate(raw):
            by_machine[int(idx)] = _values_from_index_list(values)
    else:
        for key, values in raw.items():
            key_text = str(key).strip()
            if key_text.startswith("machine_"):
                key_text = key_text[len("machine_") :]
            by_machine[int(key_text)] = _values_from_index_list(values)

    expected_keys = set(range(int(num_machines)))
    got_keys = set(by_machine)
    if got_keys != expected_keys:
        raise ValueError(
            "paper_check.run_indices_by_machine keys must exactly match configured machines "
            f"{sorted(expected_keys)}, got {sorted(got_keys)}."
        )

    all_runs: list[int] = []
    for idx in sorted(by_machine):
        runs = by_machine[idx]
        if any(v < 0 for v in runs):
            raise ValueError(f"paper_check.run_indices_by_machine[{idx}] contains negative run ids: {runs}")
        if len(set(runs)) != len(runs):
            raise ValueError(f"paper_check.run_indices_by_machine[{idx}] contains duplicates: {runs}")
        all_runs.extend(runs)
    duplicates = sorted({v for v in all_runs if all_runs.count(v) > 1})
    if duplicates:
        raise ValueError(f"paper_check.run_indices_by_machine assigns runs to multiple machines: {duplicates}")
    return list(by_machine[int(machine_idx)])


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
    if str(base_cfg.get("substrate", {}).get("substrate", "")).strip().lower() == "lenia_flow":
        if base_cfg.substrate.get("flow_sigma", None) is None and base_cfg.substrate.get("sigma", None) is not None:
            base_cfg.substrate.flow_sigma = base_cfg.substrate.sigma

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

    fixed_eval_from_pair = bool(
        stage_cfg.get("fixed_eval_seed_base_from_pair_seed_base", False)
        or stage_cfg.get("eval_seed_base_from_pair_seed_base", False)
    )
    if fixed_eval_from_pair:
        if base_cfg.get("optimization") is None:
            base_cfg.optimization = OmegaConf.create()
        pair_seed_base = int(paper_section.get("pair_seed_base", 400_003))
        base_cfg.optimization.eval_seed_mode = str(base_cfg.optimization.get("eval_seed_mode", "fixed"))
        base_cfg.optimization.fixed_eval_seed_base = int(pair_seed_base + 2 * int(run_idx))

    resolved_config_path = run_save_dir_abs / "optimization_config.yaml"
    return base_cfg, write_resolved_yaml(resolved_config_path, base_cfg)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_paper_check_optimization.py <paper_check_config.yaml>")

    paper_cfg, config_path = load_paper_check_config(sys.argv[1])
    machine_idx, num_machines = validate_machine_config(paper_cfg)
    paper_section = paper_cfg.get("paper_check", {})
    explicit_by_machine = _explicit_run_indices_by_machine(
        paper_section,
        machine_idx=machine_idx,
        num_machines=num_machines,
    )
    assignment_mode = "run_indices_by_machine" if explicit_by_machine is not None else "modulo"
    if explicit_by_machine is not None:
        assigned = explicit_by_machine
    else:
        explicit_runs = _explicit_run_indices(paper_section)
        if explicit_runs is not None:
            assignment_mode = "run_indices_modulo"
    if explicit_by_machine is None and explicit_runs is None:
        total_runs = int(paper_section.get("num_optimizations", 1))
        if total_runs < 1:
            raise ValueError(f"paper_check.num_optimizations must be >= 1, got {total_runs}.")
        assigned = shard_indices(total_runs, machine_idx, num_machines)
    elif explicit_by_machine is None:
        assigned = [idx for pos, idx in enumerate(explicit_runs) if pos % num_machines == machine_idx]
    print(
        f"[paper_check/optimization] machine_idx={machine_idx} num_machines={num_machines} "
        f"assignment_mode={assignment_mode} assigned_runs={assigned}"
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
