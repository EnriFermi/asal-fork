from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from paper_suite_common import current_python, init_suite_logging, log_event, run_subprocess


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
    if task in {"all", "c2"}:
        args = []
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("paper_suite_c2_flowlenia_metrics.py", config, *args)
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
        _call("paper_suite_c2_events.py", config, *args)
        args = ["--layer", "metrics"]
        if smoke:
            args.append("--smoke")
        _call("paper_suite_c2_branching.py", config, *args)
        if not smoke:
            _call("paper_suite_c2_branching.py", config, "--layer", "metrics", "--branching-metric", "clip_chamfer")
        args = ["--layer", "metrics"]
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("paper_suite_c2_plife_plus.py", config, *args)
    if task in {"all", "c4"}:
        args = []
        if smoke:
            args.append("--smoke")
        if force:
            args.extend(["--force-metrics", "--force-clip"])
        _call("paper_suite_nnopt_vs_mspd_compare.py", config, *args)


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


def _c2_branching_metrics(config: str, *, smoke: bool, branching_metric: str | None = None) -> None:
    args = ["--layer", "metrics"]
    if smoke:
        args.append("--smoke")
    if branching_metric:
        args.extend(["--branching-metric", branching_metric])
    _call("paper_suite_c2_branching.py", config, *args)


def _c2_plife_plus_simulation(config: str, *, smoke: bool, force: bool, allow_heavy: bool, dry_run: bool) -> None:
    args = ["--layer", "simulation"]
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    if allow_heavy:
        args.append("--allow-heavy")
    if dry_run:
        args.append("--dry-run")
    _call("paper_suite_c2_plife_plus.py", config, *args)


def _c2_plife_plus_metrics(config: str, *, smoke: bool, force: bool = False) -> None:
    args = ["--layer", "metrics"]
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    _call("paper_suite_c2_plife_plus.py", config, *args)


def _plife_noise_sweep(config: str, *, smoke: bool, force: bool, allow_heavy: bool, dry_run: bool) -> None:
    args = []
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    if allow_heavy:
        args.append("--allow-heavy")
    if dry_run:
        args.append("--dry-run")
    _call("c2_plife_perturbation_strength_sweep.py", config, *args)


def _visualization(config: str, task: str, *, smoke: bool, force: bool) -> None:
    if task == "nnopt_apf":
        log_event("visualization skipped for nnopt_apf task", component="runner")
        return
    if task == "plife_videos":
        args = []
        if smoke:
            args.append("--smoke")
        if force:
            args.append("--force")
        _call("render_plife_videos.py", config, *args)
        return
    if task == "c4":
        args = ["--plot-only"]
        if smoke:
            args.append("--smoke")
        _call("paper_suite_nnopt_vs_mspd_compare.py", config, *args)
        return
    vis_task = task if task in {"synthetic", "c1", "c2", "c5", "c6"} else "all"
    args = ["--task", vis_task]
    if smoke:
        args.append("--smoke")
    if force:
        args.append("--force")
    _call("paper_suite_visualize.py", config, *args)


def _simulation_task(task: str) -> str:
    if task == "synthetic":
        return "synthetic"
    if task == "c2":
        return "c2"
    if task == "c5":
        return "paper_check_frustration"
    if task in {"c1", "c6"}:
        return "paper_check"
    if task == "c4":
        return "c4_apf"
    if task == "nnopt_apf":
        return "nnopt_apf"
    return "all"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-button MSPD paper experiment suite.")
    parser.add_argument("config", help="experiments/paper_suite/config.yaml")
    parser.add_argument("--layer", choices=["simulation", "metrics", "visualization", "all"], default="all")
    parser.add_argument(
        "--task",
        choices=["all", "synthetic", "c1", "c2", "c4", "c5", "c6", "nnopt_apf", "plife_videos", "plife_noise_sweep"],
        default="all",
    )
    parser.add_argument("--smoke", action="store_true", help="Use small CPU smoke settings and generated tiny posthoc fixtures.")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even when present.")
    parser.add_argument("--allow-heavy", action="store_true", help="Allow configured heavy real simulation commands.")
    parser.add_argument("--dry-run", action="store_true", help="Print heavy simulation commands without running them.")
    args = parser.parse_args(argv)

    master_log = init_suite_logging(args.config, smoke=args.smoke, layer=args.layer, task=args.task)
    log_event(f"master log: {master_log}", component="runner")
    if args.task == "plife_noise_sweep":
        if args.layer in {"simulation", "all"}:
            log_event("starting PLife++ C2 perturbation strength sweep", component="runner")
            _plife_noise_sweep(
                args.config,
                smoke=args.smoke,
                force=args.force,
                allow_heavy=args.allow_heavy,
                dry_run=args.dry_run,
            )
        else:
            log_event(f"PLife++ noise sweep has no standalone {args.layer} layer", component="runner")
        log_event("paper suite finished", component="runner")
        return 0
    if args.layer in {"simulation", "all"} and args.task != "plife_videos":
        log_event("starting simulation layer", component="runner")
        _simulation(args.config, args.task, smoke=args.smoke, force=args.force, allow_heavy=args.allow_heavy, dry_run=args.dry_run)
    if args.layer in {"metrics", "all"} and args.task != "plife_videos":
        log_event("starting metrics layer", component="runner")
        _metrics(args.config, args.task, smoke=args.smoke, force=args.force)
    if args.layer == "all" and args.task in {"all", "c2"}:
        log_event("starting C2 branch simulation after C2 metrics", component="runner")
        _c2_branching_simulation(
            args.config,
            smoke=args.smoke,
            force=args.force,
            allow_heavy=args.allow_heavy,
            dry_run=args.dry_run,
        )
        log_event("starting C2 branch metrics", component="runner")
        _c2_branching_metrics(args.config, smoke=args.smoke)
        if not args.smoke:
            log_event("starting C2 CLIP-Chamfer branch metrics", component="runner")
            _c2_branching_metrics(args.config, smoke=args.smoke, branching_metric="clip_chamfer")
        log_event("starting PLife++ C2 branch simulation after PLife++ C2 metrics", component="runner")
        _c2_plife_plus_simulation(
            args.config,
            smoke=args.smoke,
            force=args.force,
            allow_heavy=args.allow_heavy,
            dry_run=args.dry_run,
        )
        log_event("starting PLife++ C2 branch metrics", component="runner")
        _c2_plife_plus_metrics(args.config, smoke=args.smoke, force=False)
    if args.layer in {"visualization", "all"}:
        log_event("starting visualization layer", component="runner")
        _visualization(args.config, args.task, smoke=args.smoke, force=args.force)
    log_event("paper suite finished", component="runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
