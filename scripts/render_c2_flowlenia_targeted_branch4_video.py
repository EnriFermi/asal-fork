from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from omegaconf import OmegaConf

from paper_suite_common import load_config, resolve_path, write_json
from render_c2_flowlenia_branch_divergence_grid import (
    _Mp4Writer,
    _as_float,
    _as_int,
    _compose_grid,
    _default_output_root,
    _first_existing,
    _load_branch_frames,
    _load_source_frame,
    _matching_plan_rows,
    _parse_rewrite_prefix,
    _resolve_data_path,
    _resize_nearest,
    _write_mp4,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _safe_slug(value: Any) -> str:
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "item"


def _select_score_row(
    rows: list[dict[str, str]],
    *,
    selection_mode: str,
    min_delta_h_quantile: float,
    max_delta_h_quantile: float,
    traj_id: str | None,
    pair_id: int | None,
    condition: str | None,
) -> tuple[dict[str, str], float, int, str]:
    finite = [
        row
        for row in rows
        if _is_finite(_as_float(row.get("delta_h"))) and _is_finite(_as_float(row.get("branching_score")))
    ]
    if traj_id is not None:
        finite = [row for row in finite if str(row.get("traj_id")) == str(traj_id)]
    if pair_id is not None:
        finite = [row for row in finite if _as_int(row.get("pair_id")) == int(pair_id)]
    if condition is not None:
        finite = [row for row in finite if str(row.get("condition")) == str(condition)]
    if not finite:
        raise ValueError("No finite score rows remain after filters.")

    threshold = min(_as_float(row.get("delta_h")) for row in finite)
    candidates = finite
    mode = str(selection_mode).strip().lower()
    if traj_id is None and pair_id is None and condition is None:
        import numpy as np

        delta_h = np.asarray([_as_float(row.get("delta_h")) for row in finite], dtype=float)
        if mode == "high_delta_max_divergence":
            threshold = float(np.quantile(delta_h, float(min_delta_h_quantile)))
            candidates = [row for row in finite if _as_float(row.get("delta_h")) >= threshold]
        elif mode == "low_delta_min_divergence":
            threshold = float(np.quantile(delta_h, float(max_delta_h_quantile)))
            candidates = [row for row in finite if _as_float(row.get("delta_h")) <= threshold]
        else:
            raise ValueError(
                "--selection-mode must be one of "
                "['high_delta_max_divergence', 'low_delta_min_divergence'], "
                f"got {selection_mode!r}."
            )
        if not candidates:
            candidates = finite
    if mode == "high_delta_max_divergence":
        selected = max(candidates, key=lambda row: _as_float(row.get("branching_score")))
        rule = "max branching_score among rows with delta_h in top quantile"
    elif mode == "low_delta_min_divergence":
        selected = min(candidates, key=lambda row: _as_float(row.get("branching_score")))
        rule = "min branching_score among rows with delta_h in bottom quantile"
    else:
        raise ValueError(
            "--selection-mode must be one of "
            "['high_delta_max_divergence', 'low_delta_min_divergence'], "
            f"got {selection_mode!r}."
        )
    return selected, float(threshold), len(candidates), rule


def _select_current_top_row(
    rows: list[dict[str, str]],
    *,
    min_delta_h_quantile: float,
    traj_id: str | None,
    pair_id: int | None,
    condition: str | None,
) -> tuple[dict[str, str], float, int]:
    selected, threshold, n_candidates, _rule = _select_score_row(
        rows,
        selection_mode="high_delta_max_divergence",
        min_delta_h_quantile=min_delta_h_quantile,
        max_delta_h_quantile=0.15,
        traj_id=traj_id,
        pair_id=pair_id,
        condition=condition,
    )
    return selected, threshold, n_candidates


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def _cfg_branching(config: Path) -> Any:
    cfg, _ = load_config(config)
    return cfg.get("c2", {}).get("branching", {})


def _metric_suffix(metric: str) -> str:
    return "" if str(metric) == "apf" else "_clip_chamfer"


def _seed_for_extra_branch(selected: dict[str, str], existing: list[int], branch_id: int) -> int:
    if existing:
        return int(max(existing) + 131 * (branch_id + 1))
    payload = "|".join(
        [
            str(selected.get("traj_id", "")),
            str(selected.get("pair_id", "")),
            str(selected.get("condition", "")),
            str(selected.get("step", "")),
            str(branch_id),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(7000003 + int(digest[:8], 16) % 100000000)


def _target_branch_rows(
    *,
    selected: dict[str, str],
    template_rows: list[dict[str, str]],
    target_root: Path,
    rewrite_prefix: tuple[str, str] | None,
    n_branches: int,
    horizon_steps: int | None,
    snapshot_interval: int | None,
    perturb_a_std: float | None,
    perturb_p_std: float | None,
    perturb_lag_xy_std: float | None,
) -> list[dict[str, Any]]:
    if not template_rows:
        raise RuntimeError("Selected score row has no matching rows in current branch_plan.csv.")
    template = template_rows[0]
    step = _as_int(template.get("step"), _as_int(selected.get("step")))
    if step is None:
        raise ValueError("Could not determine branch start step from selected row/template branch plan.")
    source_traj_dir = _resolve_data_path(str(template["source_traj_dir"]), rewrite_prefix=rewrite_prefix)
    if not source_traj_dir.exists():
        raise FileNotFoundError(f"source_traj_dir not found: {source_traj_dir}")
    source_apf_raw = str(template.get("source_apf_dir", "")).strip()
    source_apf_dir = (
        _resolve_data_path(source_apf_raw, rewrite_prefix=rewrite_prefix)
        if source_apf_raw
        else source_traj_dir / "apf_logs"
    )

    horizon = horizon_steps
    if horizon is None:
        horizon = _as_int(template.get("horizon_steps"))
    if horizon is None:
        raise ValueError("Could not determine horizon_steps; pass --horizon-steps.")

    a_std = float(perturb_a_std if perturb_a_std is not None else _as_float(template.get("perturb_a_std"), 0.02))
    p_std = float(perturb_p_std if perturb_p_std is not None else _as_float(template.get("perturb_p_std"), 0.02))
    lag_std = float(
        perturb_lag_xy_std
        if perturb_lag_xy_std is not None
        else _as_float(template.get("perturb_lagrangian_xy_std"), 1.0)
    )

    by_branch = {_as_int(row.get("branch_id"), -1): row for row in template_rows}
    existing_seeds = [
        int(seed)
        for seed in (_as_int(row.get("branch_seed")) for row in template_rows)
        if seed is not None
    ]
    point_dir = (
        target_root
        / "branches"
        / _safe_slug(selected.get("traj_id", "traj"))
        / (
            f"pair_{_as_int(selected.get('pair_id'), 0):03d}_"
            f"{_safe_slug(selected.get('condition', 'condition'))}_"
            f"step_{int(step)}"
        )
    )
    out: list[dict[str, Any]] = []
    for branch_id in range(int(n_branches)):
        src = by_branch.get(branch_id)
        branch_seed = _as_int(src.get("branch_seed")) if src is not None else None
        if branch_seed is None:
            branch_seed = _seed_for_extra_branch(selected, existing_seeds, branch_id)
            existing_seeds.append(int(branch_seed))
        row = dict(template)
        row.update(
            {
                "traj_id": str(selected.get("traj_id")),
                "pair_id": int(_as_int(selected.get("pair_id"), 0) or 0),
                "condition": str(selected.get("condition")),
                "step": int(step),
                "delta_h": float(_as_float(selected.get("delta_h"))),
                "branching_score_selected": float(_as_float(selected.get("branching_score"))),
                "branch_id": int(branch_id),
                "branch_seed": int(branch_seed),
                "horizon_steps": int(horizon),
                "snapshot_interval": "" if snapshot_interval is None else int(snapshot_interval),
                "perturb_a_std": float(a_std),
                "perturb_p_std": float(p_std),
                "perturb_lagrangian_xy_std": float(lag_std),
                "source_traj_dir": str(source_traj_dir),
                "source_apf_dir": str(source_apf_dir),
                "branch_dir": str(point_dir / f"branch_{branch_id:02d}"),
            }
        )
        out.append(row)
    return out


def _jobs_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for row in rows:
        jobs.append(
            {
                "source_traj_dir": str(row["source_traj_dir"]),
                "step": int(row["step"]),
                "additional_steps": int(row["horizon_steps"]),
                "output_dir": str(row["branch_dir"]),
                "branch_seed": int(row["branch_seed"]),
                "perturb_a_std": float(row["perturb_a_std"]),
                "perturb_p_std": float(row["perturb_p_std"]),
                "perturb_lagrangian_xy_std": float(row["perturb_lagrangian_xy_std"]),
            }
        )
        if row.get("snapshot_interval") not in (None, ""):
            jobs[-1]["snapshot_interval"] = int(row["snapshot_interval"])
    return jobs


def _run_branch_jobs(jobs_path: Path, *, batch_size: int, force: bool, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/flowlenia_minibang_resume_batch.py",
        "--jobs-json",
        str(jobs_path),
        "--batch-size",
        str(int(batch_size)),
    ]
    if force:
        cmd.append("--overwrite")
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


def _frame_u8(frame: Any, *, panel_size: int | None) -> Any:
    import numpy as np

    out = np.asarray(frame)
    if out.dtype != np.uint8:
        out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    if panel_size is not None:
        out = _resize_nearest(out, int(panel_size))
    return out


def _render_state_grid(substrate: Any, state: Any, params: Any, *, panel_size: int | None, gap: int) -> Any:
    import jax
    import numpy as np

    def render_one(st, pr):
        return substrate.render_state(st, pr, img_size=int(panel_size or 256))

    frames = jax.device_get(jax.vmap(render_one)(state, params))
    panels = [_frame_u8(frames[i], panel_size=panel_size) for i in range(int(frames.shape[0]))]
    if len(panels) != 4:
        raise ValueError(f"Direct grid renderer expects exactly 4 panels, got {len(panels)}.")
    return _compose_grid(panels, gap=int(gap))


def _run_direct_video_jobs(
    jobs: list[dict[str, Any]],
    *,
    output: Path,
    metadata_path: Path,
    selected: dict[str, str],
    selection_rule: str,
    delta_h_threshold: float,
    n_delta_h_candidates: int,
    scores_path: Path,
    plan_path: Path,
    target_root: Path,
    fps: float,
    panel_size: int,
    gap: int,
    horizon_steps: int,
    render_interval: int,
    jit_microbatch: int | None,
    max_frames: int,
    loop_hold_frames: int,
    force: bool,
    no_prepend_source_frame: bool,
) -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tqdm import tqdm

    from flowlenia_minibang_resume import _apply_resume_perturbation, _get, _reconstruct_state
    from flowlenia_minibang_resume_batch import _make_batched_stepper, _prepare_job, _write_metadata
    from flowlenia_minibang_simulate import _make_substrate

    if len(jobs) != 4:
        raise ValueError(f"Direct video mode expects exactly 4 jobs, got {len(jobs)}.")
    if int(render_interval) < 1:
        raise ValueError(f"--direct-render-interval must be >= 1, got {render_interval}.")
    if int(horizon_steps) < 1:
        raise ValueError(f"--horizon-steps must be >= 1, got {horizon_steps}.")

    raw_jobs = [dict(job, additional_steps=int(horizon_steps)) for job in jobs]
    prepared = [_prepare_job(raw, overwrite=bool(force)) for raw in raw_jobs]
    group_keys = {job["group_key"] for job in prepared}
    if len(group_keys) != 1:
        raise ValueError("Selected direct-video jobs are not batch-compatible.")

    first = prepared[0]
    args = first["args"]
    substrate = _make_substrate(args)
    _ = substrate.seed_state(jax.random.PRNGKey(0), jnp.asarray(first["params"], dtype=jnp.float32))
    rt = substrate.RT

    common_state = _reconstruct_state(substrate, first["params"], first["snapshot"])
    common_params = jnp.asarray(first["params"], dtype=jnp.float32)
    common_frame = _frame_u8(
        jax.device_get(substrate.render_state(common_state, common_params, img_size=int(panel_size))),
        panel_size=int(panel_size),
    )
    common_grid = _compose_grid([common_frame, common_frame, common_frame, common_frame], gap=int(gap))

    states = []
    lag_xys = []
    lag_chs = []
    rngs = []
    params = []
    for job in prepared:
        state = _reconstruct_state(substrate, job["params"], job["snapshot"])
        lag_xy = jnp.asarray(np.asarray(job["snapshot"]["lagrangian_xy"], dtype=np.float32))
        rng = jnp.asarray(np.asarray(job["snapshot"]["resume_batch_rng_key"], dtype=np.uint32))
        state, lag_xy, rng = _apply_resume_perturbation(
            state=state,
            lag_xy=lag_xy,
            rng=rng,
            seed=int(job["raw"].get("branch_seed", -1)),
            a_std=float(job["raw"].get("perturb_a_std", 0.0)),
            p_std=float(job["raw"].get("perturb_p_std", 0.0)),
            lag_xy_std=float(job["raw"].get("perturb_lagrangian_xy_std", 0.0)),
            border=str(getattr(rt, "border", "wall")),
            sigma=float(getattr(rt, "sigma", 0.0)),
        )
        states.append(state)
        lag_xys.append(lag_xy)
        lag_chs.append(jnp.asarray(np.asarray(job["snapshot"]["lagrangian_c"], dtype=np.int32)))
        rngs.append(rng)
        params.append(jnp.asarray(job["params"], dtype=jnp.float32))
        _write_metadata(job)

    state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
    lag_xy = jnp.stack(lag_xys, axis=0)
    lag_ch = jnp.stack(lag_chs, axis=0)
    rng = jnp.stack(rngs, axis=0)
    params_j = jnp.stack(params, axis=0)
    original_batch_index = jnp.asarray([job["original_batch_index"] for job in prepared], dtype=jnp.int32)

    lag_flow_channel = int(_get(args, "lagrangian_flow_channel", _get(args, "metric_lagrangian_flow_channel", -1)))
    lag_flow_reduce = str(_get(args, "lagrangian_flow_reduce", _get(args, "metric_lagrangian_flow_reduce", "mass_weighted")))
    lag_channel_mode = str(_get(args, "lagrangian_channel_mode", _get(args, "metric_lagrangian_channel_mode", "resample")))
    lag_noise_model = str(_get(args, "lagrangian_noise_model", _get(args, "metric_lagrangian_noise_model", "rt_box")))
    lag_diffusion_scale = float(_get(args, "lagrangian_diffusion_scale", _get(args, "metric_lagrangian_diffusion_scale", 1.0)))
    stepper = _make_batched_stepper(
        substrate=substrate,
        rt=rt,
        original_batch_size=int(first["original_batch_size"]),
        lag_flow_channel=lag_flow_channel,
        lag_flow_reduce=lag_flow_reduce,
        lag_channel_mode=lag_channel_mode,
        lag_noise_model=lag_noise_model,
        lag_diffusion_scale=lag_diffusion_scale,
    )

    if jit_microbatch is None:
        step_chunk = int(first["jit_microbatch"])
    else:
        step_chunk = int(jit_microbatch)
    step_chunk = max(1, min(step_chunk, int(render_interval)))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = _Mp4Writer(output, fps=float(fps), first_frame=common_grid)
    frame_rows: list[dict[str, Any]] = []
    frames_written = 0

    def append_grid(grid, rel_step: int) -> None:
        nonlocal frames_written
        if int(max_frames) > 0 and frames_written >= int(max_frames):
            return
        writer.append_data(grid)
        frame_rows.append(
            {
                "frame_idx": int(frames_written),
                "relative_step": int(rel_step),
                "step": int(first["start_step"]) + int(rel_step),
                "video_sec": float(frames_written) / float(fps),
            }
        )
        frames_written += 1

    try:
        if not no_prepend_source_frame:
            append_grid(common_grid, 0)
        perturbed_grid = _render_state_grid(substrate, state, params_j, panel_size=int(panel_size), gap=int(gap))
        append_grid(perturbed_grid, 0)

        rel_done = 0
        pbar = tqdm(total=int(horizon_steps) * len(prepared), desc=f"direct branch4 {horizon_steps} steps")
        try:
            while rel_done < int(horizon_steps):
                target_rel = min(int(horizon_steps), ((rel_done // int(render_interval)) + 1) * int(render_interval))
                while rel_done < target_rel:
                    n = min(int(step_chunk), int(target_rel) - int(rel_done))
                    split = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
                    rng = split[:, 0, :]
                    subkey = split[:, 1, :]
                    state, lag_xy, lag_ch = stepper(n)(state, lag_xy, lag_ch, subkey, params_j, original_batch_index)
                    rel_done += int(n)
                    pbar.update(int(n) * len(prepared))
                grid = _render_state_grid(substrate, state, params_j, panel_size=int(panel_size), gap=int(gap))
                append_grid(grid, rel_done)
                if int(max_frames) > 0 and frames_written >= int(max_frames):
                    break
        finally:
            pbar.close()

        if loop_hold_frames > 0 and frame_rows:
            final_grid = _render_state_grid(substrate, state, params_j, panel_size=int(panel_size), gap=int(gap))
            for _ in range(int(loop_hold_frames)):
                writer.append_data(final_grid)
                frames_written += 1
    finally:
        writer.close()

    write_json(
        metadata_path,
        {
            "status": "ok",
            "mode": "direct_video",
            "output": str(output),
            "frames_written": int(frames_written),
            "frame_rows": frame_rows,
            "render_interval": int(render_interval),
            "horizon_steps": int(horizon_steps),
            "jit_microbatch": int(step_chunk),
            "selection_rule": str(selection_rule),
            "selection_scores_path": str(scores_path),
            "selection_branch_plan": str(plan_path),
            "target_root": str(target_root),
            "delta_h_threshold": float(delta_h_threshold),
            "n_delta_h_candidates": int(n_delta_h_candidates),
            "selected_score_row": selected,
            "jobs": raw_jobs,
        },
    )
    return int(frames_written)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Select a Flow-Lenia C2 point by Delta-H/divergence rule, "
            "simulate exactly four fresh branches for that point, and render a 2x2 video."
        )
    )
    ap.add_argument("--config", type=Path, default=Path("experiments/paper_suite/config.yaml"))
    ap.add_argument("--output-root", type=Path, default=None, help="Existing result root used for selection.")
    ap.add_argument("--scores", type=Path, default=None)
    ap.add_argument("--branch-plan", type=Path, default=None)
    ap.add_argument("--metric-suffix", choices=["clip_chamfer", "apf"], default="clip_chamfer")
    ap.add_argument("--target-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metadata-out", type=Path, default=None)
    ap.add_argument(
        "--selection-mode",
        choices=["high_delta_max_divergence", "low_delta_min_divergence"],
        default="high_delta_max_divergence",
        help=(
            "Point selection rule. Default keeps the old behavior: top Delta-H quantile, "
            "then max branching_score. low_delta_min_divergence uses bottom Delta-H "
            "quantile, then min branching_score."
        ),
    )
    ap.add_argument("--min-delta-h-quantile", type=float, default=0.75)
    ap.add_argument("--max-delta-h-quantile", type=float, default=0.15)
    ap.add_argument("--traj-id", default=None)
    ap.add_argument("--pair-id", type=int, default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--horizon-steps", type=int, default=None)
    ap.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Override APF snapshot cadence for the freshly simulated branches.",
    )
    ap.add_argument("--perturb-a-std", type=float, default=None)
    ap.add_argument("--perturb-p-std", type=float, default=None)
    ap.add_argument("--perturb-lagrangian-xy-std", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument(
        "--direct-video",
        action="store_true",
        help="Simulate the four selected branches and write the 2x2 mp4 directly without APF snapshot rendering.",
    )
    ap.add_argument(
        "--direct-render-interval",
        type=int,
        default=100,
        help="Simulation steps between frames in --direct-video mode.",
    )
    ap.add_argument(
        "--direct-jit-microbatch",
        type=int,
        default=None,
        help="Step chunk size for --direct-video mode; defaults to the source APF resume microbatch.",
    )
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--snapshot-stride", type=int, default=1)
    ap.add_argument("--panel-size", type=int, default=256)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument("--loop-hold-frames", type=int, default=12)
    ap.add_argument("--prefer-saved-rgb", action="store_true")
    ap.add_argument("--no-prepend-source-frame", action="store_true")
    ap.add_argument("--rewrite-prefix", default=None, help="Rewrite stored absolute paths, e.g. /old/repo=/home/coder/project.")
    args = ap.parse_args()

    if int(args.n_branches) != 4:
        raise ValueError("This targeted video renderer expects --n-branches 4.")

    output_root = args.output_root
    if output_root is None:
        output_root = _default_output_root(args.config)
    elif not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()
    c2_dir = output_root / "c2_branching"

    if args.scores is not None:
        scores_path = args.scores if args.scores.is_absolute() else (_REPO_ROOT / args.scores)
    else:
        suffix = _metric_suffix(args.metric_suffix)
        scores_path = _first_existing([c2_dir / f"branching_scores{suffix}.csv", c2_dir / "branching_scores.csv"])
    if args.branch_plan is not None:
        plan_path = args.branch_plan if args.branch_plan.is_absolute() else (_REPO_ROOT / args.branch_plan)
    else:
        plan_path = c2_dir / "branch_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing branch plan: {plan_path}")

    target_root = args.target_root
    if target_root is None:
        if args.selection_mode == "low_delta_min_divergence":
            target_root = output_root / "c2_branching_targeted_branch4_low_delta_min_divergence_video"
        else:
            target_root = output_root / "c2_branching_targeted_branch4_video"
    elif not target_root.is_absolute():
        target_root = (_REPO_ROOT / target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    out_path = args.out
    if out_path is None:
        out_path = target_root / "c2_flowlenia_selected_branch_divergence_2x2.mp4"
    elif not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    meta_path = args.metadata_out
    if meta_path is None:
        meta_path = out_path.with_suffix(".json")
    elif not meta_path.is_absolute():
        meta_path = (_REPO_ROOT / meta_path).resolve()

    score_rows = _read_csv(scores_path)
    selected, threshold, n_candidates, selection_rule = _select_score_row(
        score_rows,
        selection_mode=str(args.selection_mode),
        min_delta_h_quantile=float(args.min_delta_h_quantile),
        max_delta_h_quantile=float(args.max_delta_h_quantile),
        traj_id=args.traj_id,
        pair_id=args.pair_id,
        condition=args.condition,
    )
    plan_rows = _read_csv(plan_path)
    template_rows = _matching_plan_rows(plan_rows, selected)
    rewrite_prefix = _parse_rewrite_prefix(args.rewrite_prefix)
    selected_branch_rows = _target_branch_rows(
        selected=selected,
        template_rows=template_rows,
        target_root=target_root,
        rewrite_prefix=rewrite_prefix,
        n_branches=int(args.n_branches),
        horizon_steps=args.horizon_steps,
        snapshot_interval=args.snapshot_interval,
        perturb_a_std=args.perturb_a_std,
        perturb_p_std=args.perturb_p_std,
        perturb_lag_xy_std=args.perturb_lagrangian_xy_std,
    )
    jobs = _jobs_from_rows(selected_branch_rows)
    jobs_path = target_root / "selected_branch4_jobs.json"
    write_json(jobs_path, {"jobs": jobs})

    if args.direct_video and args.render_only:
        raise ValueError("--direct-video cannot be combined with --render-only.")

    if not args.render_only:
        if args.direct_video:
            if args.dry_run:
                cmd = [
                    sys.executable,
                    "scripts/render_c2_flowlenia_targeted_branch4_video.py",
                    "--config",
                    str(args.config),
                    "--direct-video",
                    "--direct-render-interval",
                    str(int(args.direct_render_interval)),
                    "--horizon-steps",
                    str(int(args.horizon_steps or selected_branch_rows[0]["horizon_steps"])),
                    "--selection-mode",
                    str(args.selection_mode),
                ]
                if args.selection_mode == "low_delta_min_divergence":
                    cmd.extend(["--max-delta-h-quantile", str(float(args.max_delta_h_quantile))])
                else:
                    cmd.extend(["--min-delta-h-quantile", str(float(args.min_delta_h_quantile))])
                if args.force:
                    cmd.append("--force")
                print(" ".join(cmd))
            else:
                frames_written = _run_direct_video_jobs(
                    jobs,
                    output=out_path,
                    metadata_path=meta_path,
                    selected=selected,
                    selection_rule=selection_rule,
                    delta_h_threshold=float(threshold),
                    n_delta_h_candidates=int(n_candidates),
                    scores_path=scores_path,
                    plan_path=plan_path,
                    target_root=target_root,
                    fps=float(args.fps),
                    panel_size=int(args.panel_size),
                    gap=int(args.gap),
                    horizon_steps=int(args.horizon_steps or selected_branch_rows[0]["horizon_steps"]),
                    render_interval=int(args.direct_render_interval),
                    jit_microbatch=args.direct_jit_microbatch,
                    max_frames=int(args.max_frames),
                    loop_hold_frames=int(args.loop_hold_frames),
                    force=bool(args.force),
                    no_prepend_source_frame=bool(args.no_prepend_source_frame),
                )
                print(
                    "Rendered direct targeted C2 branch4 video "
                    f"traj={selected.get('traj_id')} pair={selected.get('pair_id')} "
                    f"condition={selected.get('condition')} frames={frames_written} output={out_path}"
                )
                print(f"Metadata: {meta_path}")
                return 0
        else:
            _run_branch_jobs(jobs_path, batch_size=int(args.batch_size), force=bool(args.force), dry_run=bool(args.dry_run))
    if args.dry_run:
        write_json(
            meta_path,
            {
                "status": "dry_run",
                "selection_scores_path": str(scores_path),
                "selection_branch_plan": str(plan_path),
                "target_root": str(target_root),
                "jobs_path": str(jobs_path),
                "selection_mode": str(args.selection_mode),
                "selection_rule": str(selection_rule),
                "selected_score_row": selected,
                "selected_branch_rows": selected_branch_rows,
                "delta_h_threshold": float(threshold),
                "n_delta_h_candidates": int(n_candidates),
            },
        )
        print(f"Dry-run metadata: {meta_path}")
        return 0

    branch_dirs = [Path(str(row["branch_dir"])) for row in selected_branch_rows]
    branch_frames = [
        _load_branch_frames(
            branch_dir,
            max_frames=int(args.max_frames),
            snapshot_stride=int(args.snapshot_stride),
            panel_size=int(args.panel_size) if args.panel_size and args.panel_size > 0 else None,
            prefer_saved_rgb=bool(args.prefer_saved_rgb),
        )
        for branch_dir in branch_dirs
    ]
    source_frame = None
    if not args.no_prepend_source_frame:
        source_frame = _load_source_frame(
            selected_branch_rows[0],
            rewrite_prefix=None,
            panel_size=int(args.panel_size) if args.panel_size and args.panel_size > 0 else None,
            prefer_saved_rgb=bool(args.prefer_saved_rgb),
        )
        if source_frame is not None:
            branch_frames = [[source_frame] + frames for frames in branch_frames]

    written = _write_mp4(
        branch_frames,
        out_path,
        fps=float(args.fps),
        gap=int(args.gap),
        loop_hold_frames=int(args.loop_hold_frames),
    )
    payload = {
        "status": "ok",
        "output": str(out_path),
        "frames_written": int(written),
        "selection_mode": str(args.selection_mode),
        "selection_rule": str(selection_rule),
        "selection_scores_path": str(scores_path),
        "selection_branch_plan": str(plan_path),
        "target_root": str(target_root),
        "jobs_path": str(jobs_path),
        "delta_h_threshold": float(threshold),
        "n_delta_h_candidates": int(n_candidates),
        "selected_score_row": selected,
        "selected_branch_rows": selected_branch_rows,
        "branch_dirs": [str(path) for path in branch_dirs],
        "prepended_source_frame": source_frame is not None,
        "per_branch_frames": [len(frames) for frames in branch_frames],
    }
    write_json(meta_path, payload)
    print(
        "Rendered targeted C2 branch4 video "
        f"traj={selected.get('traj_id')} pair={selected.get('pair_id')} condition={selected.get('condition')} "
        f"delta_h={_as_float(selected.get('delta_h')):.6g} "
        f"branching_score={_as_float(selected.get('branching_score')):.6g} "
        f"output={out_path}"
    )
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
