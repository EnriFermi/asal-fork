from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from paper_suite_common import current_python, run_subprocess


def _call(script: str, config: str, *args: str) -> None:
    run_subprocess([current_python(), f"scripts/{script}", config, *args])


def _simulation(config: str, task: str, *, smoke: bool, force: bool, allow_heavy: bool, dry_run: bool) -> None:
    args = ["--task", _simulation_task(task)]
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    if allow_heavy:
        args.append("--allow-heavy")
    if dry_run:
        args.append("--dry-run")
    _call("paper_suite_simulation.py", config, *args)


def _metrics(config: str, task: str, *, smoke: bool, force: bool) -> None:
    if task in {"all", "synthetic"}:
        args = ["--layer", "metrics"]
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("paper_suite_synthetic.py", config, *args)
    if task in {"all", "c1", "c5", "c6"}:
        args = ["--task", "all" if task == "all" else task]
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("paper_suite_posthoc.py", config, *args)
    if task in {"all", "c2"}:
        args = []
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("paper_suite_c2_flowlenia_metrics.py", config, *args)
        args = []
        if smoke:
            args.append("--smoke")
        _call("paper_suite_c2_events.py", config, *args)
        args = ["--layer", "metrics"]
        if smoke:
            args.append("--smoke")
        _call("paper_suite_c2_branching.py", config, *args)


def _c2_branching_simulation(config: str, *, smoke: bool, force: bool, allow_heavy: bool, dry_run: bool) -> None:
    args = ["--layer", "simulation"]
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    if allow_heavy:
        args.append("--allow-heavy")
    if dry_run:
        args.append("--dry-run")
    _call("paper_suite_c2_branching.py", config, *args)


def _c2_branching_metrics(config: str, *, smoke: bool) -> None:
    args = ["--layer", "metrics"]
    if smoke:
        args.append("--smoke")
    _call("paper_suite_c2_branching.py", config, *args)


def _visualization(config: str, task: str, *, smoke: bool) -> None:
    vis_task = task if task in {"synthetic", "c1", "c2", "c5", "c6"} else "all"
    args = ["--task", vis_task]
    if smoke:
        args.append("--smoke")
    _call("paper_suite_visualize.py", config, *args)


def _simulation_task(task: str) -> str:
    if task == "synthetic":
        return "synthetic"
    if task == "c2":
        return "c2"
    if task in {"c1", "c5", "c6"}:
        return "paper_check"
    return "all"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-button MSPD paper experiment suite.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--layer", choices=["simulation", "metrics", "visualization", "all"], default="all")
    parser.add_argument("--task", choices=["all", "synthetic", "c1", "c2", "c5", "c6"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Use small CPU smoke settings and generated tiny posthoc fixtures.")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even when present.")
    parser.add_argument("--allow-heavy", action="store_true", help="Allow configured heavy real simulation commands.")
    parser.add_argument("--dry-run", action="store_true", help="Print heavy simulation commands without running them.")
    args = parser.parse_args(argv)

    if args.layer in {"simulation", "all"}:
        _simulation(args.config, args.task, smoke=args.smoke, force=args.force, allow_heavy=args.allow_heavy, dry_run=args.dry_run)
    if args.layer in {"metrics", "all"}:
        _metrics(args.config, args.task, smoke=args.smoke, force=args.force)
    if args.layer == "all" and args.task in {"all", "c2"}:
        _c2_branching_simulation(
            args.config,
            smoke=args.smoke,
            force=args.force,
            allow_heavy=args.allow_heavy,
            dry_run=args.dry_run,
        )
        _c2_branching_metrics(args.config, smoke=args.smoke)
    if args.layer in {"visualization", "all"}:
        _visualization(args.config, args.task, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
