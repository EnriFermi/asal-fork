from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flowlenia_minibang_common import list_apf_chunks
from flowlenia_minibang_simulate import compute_metrics_for_run, expected_delta_h_metric_metadata
from paper_suite_c2_flowlenia_metrics import _apf_status, _flat_metric_args, _iter_trajectories
from paper_suite_common import ensure_dir, load_config, log_event, read_csv, resolve_path, write_csv, write_json
from paper_suite_metric_cache import compare_metrics_npz_metadata, stable_json


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _simulation_section(cfg: Any, name: str) -> Any:
    return _get(cfg.get("simulation", {}), name, {})


def _root_from_section(cfg: Any, name: str, default: str) -> Path:
    section = _simulation_section(cfg, name)
    root = resolve_path(_get(section, "output_root", default))
    if root is None:
        raise ValueError(f"Cannot resolve output_root for {name}.")
    return root


def _rollout_config_from_section(cfg: Any, name: str) -> Path:
    section = _simulation_section(cfg, name)
    raw = _get(section, "rollout_config", "experiments/paper_suite/flowlenia_arun_apf_500k.yaml")
    path = resolve_path(raw)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Rollout config not found for {name}: {path}")
    return path


def _source_specs(cfg: Any) -> list[dict[str, Any]]:
    return [
        {
            "objective": "MSPD-opt",
            "objective_key": "mspd_opt",
            "root": _root_from_section(
                cfg,
                "flow_lenia_arun_lagrangian_apf",
                "experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k",
            ),
            "section": "flow_lenia_arun_lagrangian_apf",
        },
        {
            "objective": "NN-opt",
            "objective_key": "nn_opt",
            "root": _root_from_section(
                cfg,
                "flow_lenia_nnopt_lagrangian_apf",
                "experiments/paper_check_flow_lenia/checkpoints/nnopt_lagrangian_apf_500k",
            ),
            "section": "flow_lenia_nnopt_lagrangian_apf",
        },
    ]


def _c4_section(cfg: Any) -> Any:
    return _get(cfg, "c4", {})


def _c4_out_dir(cfg: Any, output_root: Path) -> Path:
    section = _c4_section(cfg)
    raw = _get(section, "output_dir", None)
    if raw is None or str(raw).strip() == "":
        return ensure_dir(output_root / "c4_nnopt_vs_mspd")
    path = Path(str(raw))
    if path.is_absolute():
        return ensure_dir(path)
    if len(path.parts) == 1:
        return ensure_dir(output_root / path)
    resolved = resolve_path(str(raw))
    return ensure_dir(resolved if resolved is not None else output_root / "c4_nnopt_vs_mspd")


def _condition_label(objective_key: str, candidate_kind: str) -> str:
    if objective_key == "mspd_opt" and candidate_kind == "random":
        return "Random"
    if objective_key == "mspd_opt":
        return "MSPD-opt"
    if objective_key == "nn_opt":
        return "NN-opt"
    return str(objective_key)


def _repair_item_paths(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    traj_id = str(out["traj_id"])
    default_traj_dir = root / traj_id
    default_apf_dir = default_traj_dir / "apf_logs"
    if not Path(out.get("traj_dir", "")).exists() and default_traj_dir.exists():
        out["traj_dir"] = default_traj_dir
    if not Path(out.get("apf_dir", "")).exists() and default_apf_dir.exists():
        out["apf_dir"] = default_apf_dir
    default_metrics = default_traj_dir / "metrics.npz"
    if not Path(out.get("metrics_path", "")).exists() and (default_traj_dir.exists() or default_apf_dir.exists()):
        out["metrics_path"] = default_metrics
    return out


def _objective_items(root: Path, *, include_random: bool) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for item in _iter_trajectories(root):
        item = _repair_item_paths(root, item)
        kind = str(item.get("candidate_kind", "optimized"))
        if kind != "optimized" and not include_random:
            continue
        rows.append(item)
    return rows


def _ensure_delta_h_metrics(
    *,
    item: dict[str, Any],
    flat_args: dict[str, Any],
    force_metrics: bool,
) -> tuple[Path, str]:
    apf_dir = Path(item["apf_dir"])
    metrics_path = Path(item["metrics_path"])
    apf_ready, apf_msg, _n_chunks = _apf_status(apf_dir)
    if not apf_ready:
        if metrics_path.exists() and not force_metrics:
            # Useful on local analysis machines that only received precomputed
            # metrics archives. Freshness cannot be revalidated without APF.
            return metrics_path, "exists_no_apf_unvalidated"
        raise FileNotFoundError(f"{item['traj_id']}: APF not ready: {apf_msg}")
    if metrics_path.exists() and not force_metrics:
        metric_cfg, input_identity, _metadata = expected_delta_h_metric_metadata(apf_dir, flat_args)
        fresh, reason, _expected = compare_metrics_npz_metadata(metrics_path, metric_cfg, input_identity)
        if not fresh:
            raise ValueError(
                f"{item['traj_id']}: stale metrics cache at {metrics_path}: {reason}. "
                "Re-run this script with --force-metrics."
            )
        return metrics_path, "exists"

    run_row = {
        "traj_id": str(item["traj_id"]),
        "traj_dir": Path(item["traj_dir"]),
        "apf_dir": apf_dir,
        "selection": {
            "selection_idx": int(item["selection_idx"]),
            "iter": int(item.get("run_idx", -1)),
            "saturation_T": np.nan,
        },
    }
    compute_metrics_for_run(run_row, flat_args)
    computed_path = Path(run_row["traj_dir"]) / "metrics.npz"
    if not computed_path.exists():
        raise FileNotFoundError(
            f"{item['traj_id']}: Delta-H/MSPD metrics computation finished but did not write {computed_path}"
        )
    return computed_path, "computed"


def _metric_scalar_rows(metrics_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(metrics_path, allow_pickle=False) as data:
        tau = np.asarray(data["delta_h_tau_steps"], dtype=np.float64).reshape(-1)
        score = np.asarray(data["delta_h_score_by_tau"], dtype=np.float64).reshape(-1)
        amp = np.asarray(data.get("delta_h_amp_by_tau", np.full_like(score, np.nan)), dtype=np.float64).reshape(-1)
        msc = np.asarray(data.get("delta_h_msc_by_tau", np.full_like(score, np.nan)), dtype=np.float64).reshape(-1)
        selected_tau = int(np.asarray(data.get("delta_h_selected_tau_steps", tau[int(np.nanargmax(score))])).item())
        selected_idx = int(np.where(tau == selected_tau)[0][0]) if np.any(tau == selected_tau) else int(np.nanargmax(score))
        finite = np.isfinite(tau) & np.isfinite(score) & (tau > 0)
        if int(np.sum(finite)) >= 2:
            x = np.log(tau[finite])
            y = score[finite]
            order = np.argsort(x)
            area = float(np.trapezoid(y[order], x[order])) if hasattr(np, "trapezoid") else float(
                np.sum(0.5 * (y[order][1:] + y[order][:-1]) * np.diff(x[order]))
            )
            log_tau_integral = float(area / max(1e-12, float(x[order][-1] - x[order][0])))
        elif int(np.sum(finite)) == 1:
            log_tau_integral = float(score[finite][0])
        else:
            log_tau_integral = float("nan")
        scalar = {
            "selected_tau_steps": selected_tau,
            "selected_tau_idx": selected_idx,
            "mspd_selected_score": float(score[selected_idx]) if 0 <= selected_idx < score.size else float("nan"),
            "mspd_best_score": float(np.nanmax(score)) if score.size else float("nan"),
            "mspd_log_tau_integral": log_tau_integral,
            "mspd_score_mean_tau": float(np.nanmean(score)) if score.size else float("nan"),
            "mspd_score_median_tau": float(np.nanmedian(score)) if score.size else float("nan"),
        }
    rows = []
    for i, tau_i in enumerate(tau.astype(int).tolist()):
        rows.append(
            {
                "tau_steps": int(tau_i),
                "score_by_tau": float(score[i]),
                "amp_by_tau": float(amp[i]) if i < amp.size else float("nan"),
                "msc_by_tau": float(msc[i]) if i < msc.size else float("nan"),
                "selected": int(i == selected_idx),
            }
        )
    return rows, scalar


def _render_apf_rgb(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float32)
    if p_arr.ndim != 3:
        raise ValueError(f"P frame must have shape (H,W,C), got {p_arr.shape}.")
    if p_arr.shape[-1] < 3:
        reps = int(math.ceil(3 / max(1, int(p_arr.shape[-1]))))
        p3 = np.tile(p_arr, (1, 1, reps))[..., :3]
    else:
        p3 = p_arr[..., :3]
    a_arr = np.asarray(a, dtype=np.float32)
    if a_arr.ndim != 3:
        raise ValueError(f"A frame must have shape (H,W,C), got {a_arr.shape}.")
    return np.clip(np.sum(a_arr, axis=-1, keepdims=True) * p3, 0.0, 1.0).astype(np.float32)


def _frame_records(apf_dir: Path, *, start_steps: int | None, end_steps: int | None) -> list[tuple[Path, int, int]]:
    records: list[tuple[Path, int, int]] = []
    for path, _start, _end, _idx in list_apf_chunks(apf_dir):
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64).reshape(-1)
        for local_idx, step in enumerate(steps.tolist()):
            step_i = int(step)
            if start_steps is not None and step_i < int(start_steps):
                continue
            if end_steps is not None and step_i > int(end_steps):
                continue
            records.append((path, int(local_idx), step_i))
    records.sort(key=lambda r: (r[2], str(r[0]), r[1]))
    return records


def _select_records(records: list[tuple[Path, int, int]], *, max_frames: int | None) -> list[tuple[Path, int, int]]:
    if max_frames is None or max_frames <= 0 or len(records) <= int(max_frames):
        return records
    idx = np.linspace(0, len(records) - 1, int(max_frames))
    selected_idx = sorted({int(round(float(i))) for i in idx.tolist()})
    return [records[i] for i in selected_idx]


def _clip_cache_path(
    *,
    cache_dir: Path,
    item: dict[str, Any],
    objective_key: str,
    foundation_model: str,
    selected_steps: list[int],
    max_frames: int | None,
) -> Path:
    payload = {
        "version": "nnopt_vs_mspd_clip_oe_v1",
        "objective_key": objective_key,
        "traj_id": str(item["traj_id"]),
        "apf_dir": str(Path(item["apf_dir"]).resolve()),
        "foundation_model": str(foundation_model),
        "selected_steps": [int(x) for x in selected_steps],
        "max_frames": None if max_frames is None else int(max_frames),
    }
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:20]
    safe = str(item["traj_id"]).replace("/", "__")
    return cache_dir / f"{objective_key}__{safe}__{digest}.npz"


def _compute_clip_oe_loss(
    *,
    item: dict[str, Any],
    objective_key: str,
    fm: Any,
    foundation_model: str,
    cache_dir: Path,
    max_frames: int | None,
    start_steps: int | None,
    end_steps: int | None,
    force_clip: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import asal_metrics

    records = _select_records(
        _frame_records(Path(item["apf_dir"]), start_steps=start_steps, end_steps=end_steps),
        max_frames=max_frames,
    )
    if not records:
        raise ValueError(f"{item['traj_id']}: no APF frames available for CLIP-OE.")
    selected_steps = [int(step) for _path, _local_idx, step in records]
    cache_path = _clip_cache_path(
        cache_dir=cache_dir,
        item=item,
        objective_key=objective_key,
        foundation_model=foundation_model,
        selected_steps=selected_steps,
        max_frames=max_frames,
    )
    if cache_path.exists() and not force_clip:
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "clip_oe_loss": float(np.asarray(data["clip_oe_loss"]).item()),
                "clip_n_frames": int(np.asarray(data["clip_n_frames"]).item()),
                "clip_first_step": int(np.asarray(data["clip_first_step"]).item()),
                "clip_last_step": int(np.asarray(data["clip_last_step"]).item()),
                "clip_cache_path": str(cache_path),
                "clip_status": "exists",
            }

    zs: list[np.ndarray] = []
    grouped: dict[Path, list[tuple[int, int]]] = {}
    for path, local_idx, step in records:
        grouped.setdefault(path, []).append((local_idx, step))
    done = 0
    total = len(records)
    for path, local_items in grouped.items():
        with np.load(path, allow_pickle=False) as data:
            a_chunk = np.asarray(data["A"], dtype=np.float32)
            p_chunk = np.asarray(data["P"], dtype=np.float32)
        for local_idx, step in local_items:
            frame = _render_apf_rgb(a_chunk[local_idx], p_chunk[local_idx])
            z = jax.device_get(fm.embed_img(frame))
            zs.append(np.asarray(z, dtype=np.float32).reshape(-1))
            done += 1
            if done == 1 or done == total or done % 25 == 0:
                log_event(
                    f"CLIP-OE embedding {objective_key}/{item['traj_id']} {done}/{total} step={step}",
                    component="nnopt-vs-mspd",
                )
    z_arr = np.stack(zs, axis=0).astype(np.float32)
    loss = float(np.asarray(jax.device_get(asal_metrics.calc_open_endedness_score(jnp.asarray(z_arr, dtype=jnp.float32)))))
    np.savez_compressed(
        cache_path,
        z=z_arr,
        selected_steps=np.asarray(selected_steps, dtype=np.int64),
        clip_oe_loss=np.asarray(loss, dtype=np.float32),
        clip_n_frames=np.asarray(len(selected_steps), dtype=np.int32),
        clip_first_step=np.asarray(selected_steps[0], dtype=np.int64),
        clip_last_step=np.asarray(selected_steps[-1], dtype=np.int64),
        foundation_model=np.asarray(str(foundation_model)),
    )
    return {
        "clip_oe_loss": loss,
        "clip_n_frames": len(selected_steps),
        "clip_first_step": selected_steps[0],
        "clip_last_step": selected_steps[-1],
        "clip_cache_path": str(cache_path),
        "clip_status": "computed",
    }


def _finite_metric(rows: list[dict[str, Any]], condition: str, metric: str) -> np.ndarray:
    vals = []
    for row in rows:
        if str(row.get("condition", "")) != condition:
            continue
        try:
            value = float(row.get(metric, np.nan))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def _as_float(value: Any) -> float:
    try:
        if value is None or str(value) == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / float(a.size * b.size))


def _one_sided_permutation_test(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    max_exact: int,
    reps: int,
) -> dict[str, Any]:
    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    treatment = treatment[np.isfinite(treatment)]
    control = control[np.isfinite(control)]
    observed = float(np.mean(treatment) - np.mean(control)) if treatment.size and control.size else float("nan")
    p_value = float("nan")
    method = "insufficient_data"
    n_permutations = 0
    if treatment.size and control.size:
        combined = np.concatenate([treatment, control], axis=0)
        n_treat = int(treatment.size)
        total_combinations = math.comb(int(combined.size), n_treat)
        if total_combinations <= int(max_exact):
            method = "exact_permutation_mean_difference"
            hits = 0
            n_permutations = 0
            for idx_tuple in itertools.combinations(range(int(combined.size)), n_treat):
                mask = np.zeros(int(combined.size), dtype=bool)
                mask[list(idx_tuple)] = True
                stat = float(np.mean(combined[mask]) - np.mean(combined[~mask]))
                hits += int(stat >= observed - 1e-15)
                n_permutations += 1
            p_value = float(hits / max(1, n_permutations))
        else:
            method = "monte_carlo_permutation_mean_difference"
            rng = np.random.default_rng(int(seed))
            hits = 1
            n_permutations = int(reps)
            for _ in range(n_permutations):
                perm = rng.permutation(combined)
                stat = float(np.mean(perm[:n_treat]) - np.mean(perm[n_treat:]))
                hits += int(stat >= observed - 1e-15)
            p_value = float(hits / float(n_permutations + 1))
    return {
        "n_treatment": int(treatment.size),
        "n_control": int(control.size),
        "treatment_mean": float(np.mean(treatment)) if treatment.size else float("nan"),
        "control_mean": float(np.mean(control)) if control.size else float("nan"),
        "mean_difference": observed,
        "treatment_median": float(np.median(treatment)) if treatment.size else float("nan"),
        "control_median": float(np.median(control)) if control.size else float("nan"),
        "median_difference": float(np.median(treatment) - np.median(control)) if treatment.size and control.size else float("nan"),
        "cliffs_delta": _cliffs_delta(treatment, control),
        "p_value_one_sided_greater": p_value,
        "test_method": method,
        "n_permutations": int(n_permutations),
    }


def _hypothesis_rows(
    run_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_exact: int,
    reps: int,
) -> list[dict[str, Any]]:
    specs = [
        {
            "hypothesis": "mspd_opt_nn_opt_score_gt_random",
            "family": "cross_transfer",
            "treatment_condition": "MSPD-opt",
            "control_condition": "Random",
            "metric": "nn_opt_score",
            "metric_direction": "higher_is_better",
        },
        {
            "hypothesis": "nn_opt_mspd_metric_gt_random",
            "family": "cross_transfer",
            "treatment_condition": "NN-opt",
            "control_condition": "Random",
            "metric": "mspd_metric",
            "metric_direction": "higher_is_better",
        },
        {
            "hypothesis": "mspd_opt_mspd_metric_gt_random",
            "family": "direct_sanity_check",
            "treatment_condition": "MSPD-opt",
            "control_condition": "Random",
            "metric": "mspd_metric",
            "metric_direction": "higher_is_better",
        },
        {
            "hypothesis": "nn_opt_nn_opt_score_gt_random",
            "family": "direct_sanity_check",
            "treatment_condition": "NN-opt",
            "control_condition": "Random",
            "metric": "nn_opt_score",
            "metric_direction": "higher_is_better",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        treatment = _finite_metric(run_rows, str(spec["treatment_condition"]), str(spec["metric"]))
        control = _finite_metric(run_rows, str(spec["control_condition"]), str(spec["metric"]))
        stats = _one_sided_permutation_test(
            treatment,
            control,
            seed=seed,
            max_exact=max_exact,
            reps=reps,
        )
        rows.append({**spec, "alternative": "treatment_greater_than_control", **stats})
    return rows


def _plot(out_path: Path, run_rows: list[dict[str, Any]], tau_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    conditions = ["Random", "MSPD-opt", "NN-opt"]
    x_by_condition = {name: idx for idx, name in enumerate(conditions)}
    condition_colors = {"Random": "#777777", "MSPD-opt": "#d62728", "NN-opt": "#1f77b4"}
    fig, ax_mspd = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax_nn = ax_mspd.twinx()

    for condition in conditions:
        rows = [row for row in run_rows if str(row.get("condition", "")) == condition]
        if not rows:
            continue
        x0 = float(x_by_condition[condition])
        jitter = np.linspace(-0.11, 0.11, len(rows)) if len(rows) > 1 else np.asarray([0.0])
        color = condition_colors[condition]
        mspd = np.asarray([_as_float(row.get("mspd_metric", np.nan)) for row in rows], dtype=np.float64)
        nn_score = np.asarray([_as_float(row.get("nn_opt_score", np.nan)) for row in rows], dtype=np.float64)
        finite_mspd = np.isfinite(mspd)
        finite_nn = np.isfinite(nn_score)
        if np.any(finite_mspd):
            ax_mspd.scatter(
                np.full(int(np.sum(finite_mspd)), x0) + jitter[finite_mspd] - 0.06,
                mspd[finite_mspd],
                marker="o",
                s=44,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
                label="MSPD metric" if condition == conditions[0] else None,
            )
            med = float(np.median(mspd[finite_mspd]))
            ax_mspd.plot([x0 - 0.22, x0 - 0.02], [med, med], color=color, linewidth=2.2)
        if np.any(finite_nn):
            ax_nn.scatter(
                np.full(int(np.sum(finite_nn)), x0) + jitter[finite_nn] + 0.06,
                nn_score[finite_nn],
                marker="s",
                s=40,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.75,
                label="NN-opt score" if condition == conditions[0] else None,
            )
            med = float(np.median(nn_score[finite_nn]))
            ax_nn.plot([x0 + 0.02, x0 + 0.22], [med, med], color=color, linewidth=2.2, linestyle="--")

    ax_mspd.set_xticks(range(len(conditions)), conditions)
    ax_mspd.set_ylabel("MSPD metric (log-tau integral; higher is better)")
    ax_nn.set_ylabel("NN-opt score (-CLIP-OE loss; higher is better)")
    ax_mspd.set_title("C4: MSPD-opt vs NN-opt vs random on both metrics")
    ax_mspd.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    handles_m, labels_m = ax_mspd.get_legend_handles_labels()
    handles_n, labels_n = ax_nn.get_legend_handles_labels()
    ax_mspd.legend(handles_m + handles_n, labels_m + labels_n, loc="best", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    alias = out_path.parent / "nnopt_vs_mspd_comparison.png"
    if alias != out_path:
        fig.savefig(alias, dpi=180)
    plt.close(fig)


def _plot_from_existing(config_path: str | Path) -> dict[str, Any]:
    cfg, _ = load_config(config_path)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    out_dir = _c4_out_dir(cfg, output_root)
    run_table = out_dir / "objective_run_scores.csv"
    tau_table = out_dir / "tau_profiles.csv"
    if not run_table.exists():
        raise FileNotFoundError(f"C4 run score table not found: {run_table}")
    run_rows = [dict(row) for row in read_csv(run_table)]
    tau_rows = [dict(row) for row in read_csv(tau_table)] if tau_table.exists() else []
    fig_path = out_dir / "c4_nnopt_vs_mspd_dual_axis.png"
    _plot(fig_path, run_rows, tau_rows)
    summary_path = out_dir / "c4_nnopt_vs_mspd_summary.json"
    summary = {"status": "ok", "figure": str(fig_path), "run_scores": str(run_table), "tau_profiles": str(tau_table)}
    write_json(summary_path, summary)
    return summary


def run(
    config_path: str | Path,
    *,
    smoke: bool = False,
    force_metrics: bool = False,
    force_clip: bool = False,
    skip_clip: bool = False,
    include_random: bool | None = None,
    max_clip_frames: int | None = None,
    clip_range_start_steps: int | None = None,
    clip_range_end_steps: int | None = None,
    foundation_model: str | None = None,
    plot_only: bool = False,
) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    out_dir = _c4_out_dir(cfg, output_root)
    c4_cfg = _c4_section(cfg)
    if smoke:
        summary = {"status": "smoke_skipped", "reason": "C4 uses precomputed 500k APF/NN-opt artifacts", "output_dir": str(out_dir)}
        write_json(out_dir / "c4_nnopt_vs_mspd_summary.json", summary)
        return summary
    if plot_only:
        return _plot_from_existing(config_path)
    if include_random is None:
        include_random = bool(_get(c4_cfg, "include_random", True))
    if max_clip_frames is None:
        raw_max = _get(c4_cfg, "max_clip_frames", 128)
        max_clip_frames = None if raw_max is None else int(raw_max)
    stat_seed = int(_get(c4_cfg, "stat_seed", 0))
    stat_max_exact = int(_get(c4_cfg, "stat_max_exact_permutations", 100_000))
    stat_reps = int(_get(c4_cfg, "stat_permutation_reps", 20_000))
    clip_cache_dir = ensure_dir(out_dir / "clip_cache")
    rollout_config = _rollout_config_from_section(cfg, "flow_lenia_arun_lagrangian_apf")
    flat_args = _flat_metric_args(rollout_config)
    flat_args["compute_clusters"] = False
    flat_args["compute_delta_h"] = True
    flat_args["metrics_strict"] = True

    if clip_range_start_steps is None:
        configured_start = _get(c4_cfg, "clip_range_start_steps", None)
        clip_range_start_steps = (
            int(configured_start)
            if configured_start is not None
            else int(flat_args.get("metric_range_start_steps", 0) or 0)
        )
    if clip_range_end_steps is None:
        configured_end = _get(c4_cfg, "clip_range_end_steps", None)
        clip_range_end_steps = (
            int(configured_end)
            if configured_end is not None
            else int(flat_args.get("metric_range_end_steps", flat_args.get("rollout_steps", 500_000)) or 500_000)
        )
    if foundation_model is None:
        foundation_model = str(
            _get(c4_cfg, "foundation_model", _get(_get(cfg.get("c2", {}), "branching", {}), "clip_foundation_model", "clip"))
        )

    fm = None
    if not skip_clip:
        import foundation_models

        fm = foundation_models.create_foundation_model(str(foundation_model))

    run_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    for spec in _source_specs(cfg):
        objective = str(spec["objective"])
        objective_key = str(spec["objective_key"])
        root = Path(spec["root"])
        items = _objective_items(root, include_random=include_random)
        log_event(f"{objective}: discovered n_items={len(items)} root={root}", component="nnopt-vs-mspd")
        for idx, item in enumerate(items, start=1):
            traj_id = str(item["traj_id"])
            log_event(f"{objective}: processing {idx}/{len(items)} traj={traj_id}", component="nnopt-vs-mspd")
            metrics_path, metrics_status = _ensure_delta_h_metrics(
                item=item,
                flat_args=flat_args,
                force_metrics=force_metrics,
            )
            per_tau, scalar = _metric_scalar_rows(metrics_path)
            for row in per_tau:
                condition = _condition_label(objective_key, str(item.get("candidate_kind", "optimized")))
                row.update(
                    {
                        "objective": objective,
                        "objective_key": objective_key,
                        "condition": condition,
                        "traj_id": traj_id,
                        "candidate_kind": str(item.get("candidate_kind", "optimized")),
                        "metrics_path": str(metrics_path),
                    }
                )
                tau_rows.append(row)
            clip_payload: dict[str, Any] = {
                "clip_oe_loss": float("nan"),
                "clip_n_frames": 0,
                "clip_first_step": "",
                "clip_last_step": "",
                "clip_cache_path": "",
                "clip_status": "skipped",
            }
            if fm is not None:
                clip_payload = _compute_clip_oe_loss(
                    item=item,
                    objective_key=objective_key,
                    fm=fm,
                    foundation_model=str(foundation_model),
                    cache_dir=clip_cache_dir,
                    max_frames=max_clip_frames,
                    start_steps=clip_range_start_steps,
                    end_steps=clip_range_end_steps,
                    force_clip=force_clip,
                )
            condition = _condition_label(objective_key, str(item.get("candidate_kind", "optimized")))
            nn_opt_loss = float(clip_payload.get("clip_oe_loss", float("nan")))
            nn_opt_score = -nn_opt_loss if np.isfinite(nn_opt_loss) else float("nan")
            mspd_metric = float(scalar.get("mspd_log_tau_integral", float("nan")))
            run_rows.append(
                {
                    "objective": objective,
                    "objective_key": objective_key,
                    "condition": condition,
                    "traj_id": traj_id,
                    "candidate_kind": str(item.get("candidate_kind", "optimized")),
                    "apf_dir": str(item["apf_dir"]),
                    "metrics_path": str(metrics_path),
                    "metrics_status": metrics_status,
                    "mspd_metric": mspd_metric,
                    "nn_opt_loss": nn_opt_loss,
                    "nn_opt_score": nn_opt_score,
                    **scalar,
                    **clip_payload,
                }
            )

    run_table = out_dir / "objective_run_scores.csv"
    tau_table = out_dir / "tau_profiles.csv"
    stats_table = out_dir / "hypothesis_tests.csv"
    write_csv(run_table, run_rows)
    write_csv(tau_table, tau_rows)
    stats_rows = _hypothesis_rows(run_rows, seed=stat_seed, max_exact=stat_max_exact, reps=stat_reps)
    write_csv(stats_table, stats_rows)
    fig_path = out_dir / "c4_nnopt_vs_mspd_dual_axis.png"
    _plot(fig_path, run_rows, tau_rows)

    summary_by_condition = {}
    for condition in sorted({row["condition"] for row in run_rows}):
        rows = [row for row in run_rows if row["condition"] == condition]
        summary_by_condition[condition] = {
            "n": len(rows),
            "median_mspd_metric": float(np.nanmedian([float(r["mspd_metric"]) for r in rows])) if rows else float("nan"),
            "median_mspd_best_score": float(np.nanmedian([float(r["mspd_best_score"]) for r in rows])) if rows else float("nan"),
            "median_nn_opt_loss": float(np.nanmedian([float(r["nn_opt_loss"]) for r in rows])) if rows and not skip_clip else float("nan"),
            "median_nn_opt_score": float(np.nanmedian([float(r["nn_opt_score"]) for r in rows])) if rows and not skip_clip else float("nan"),
        }
    summary = {
        "status": "ok",
        "claim": "C4",
        "rollout_config": str(rollout_config),
        "foundation_model": str(foundation_model),
        "clip_range_start_steps": None if clip_range_start_steps is None else int(clip_range_start_steps),
        "clip_range_end_steps": None if clip_range_end_steps is None else int(clip_range_end_steps),
        "max_clip_frames": None if max_clip_frames is None else int(max_clip_frames),
        "run_scores": str(run_table),
        "tau_profiles": str(tau_table),
        "hypothesis_tests": str(stats_table),
        "figure": str(fig_path),
        "summary_by_condition": summary_by_condition,
        "hypotheses": stats_rows,
    }
    write_json(out_dir / "c4_nnopt_vs_mspd_summary.json", summary)
    write_json(out_dir / "nnopt_vs_mspd_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C4: compare MSPD-opt, NN-opt, and random Flow-Lenia APF rollouts posthoc.")
    parser.add_argument("config")
    parser.add_argument("--smoke", action="store_true", help="Write a smoke skip summary.")
    parser.add_argument("--force-metrics", action="store_true", help="Recompute Delta-H/MSPD metrics from APF logs.")
    parser.add_argument("--force-clip", action="store_true", help="Recompute cached CLIP embeddings and CLIP-OE losses.")
    parser.add_argument("--skip-clip", action="store_true", help="Only compute MSPD tau profiles.")
    parser.add_argument("--include-random", dest="include_random", action="store_true", default=None, help="Include random MSPD A-run baselines.")
    parser.add_argument("--no-random", dest="include_random", action="store_false", help="Exclude random MSPD A-run baselines.")
    parser.add_argument("--max-clip-frames", type=int, default=None)
    parser.add_argument("--clip-range-start-steps", type=int, default=None)
    parser.add_argument("--clip-range-end-steps", type=int, default=None)
    parser.add_argument("--foundation-model", default=None)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the C4 plot from existing CSV outputs only.")
    args = parser.parse_args(argv)
    print(
        run(
            args.config,
            smoke=args.smoke,
            force_metrics=args.force_metrics,
            force_clip=args.force_clip,
            skip_clip=args.skip_clip,
            include_random=args.include_random,
            max_clip_frames=args.max_clip_frames,
            clip_range_start_steps=args.clip_range_start_steps,
            clip_range_end_steps=args.clip_range_end_steps,
            foundation_model=args.foundation_model,
            plot_only=args.plot_only,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
