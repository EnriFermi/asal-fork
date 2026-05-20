from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from flowlenia_minibang_common import list_apf_chunks
from paper_suite_common import ensure_dir, load_config, resolve_path, write_csv, write_json


DEFAULT_FIELD_WEIGHTS = {"A": 1.0, "P": 0.25, "F": 0.25}


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _parse_ints(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or list(default)


def _parse_field_weights(raw: str | None) -> dict[str, float]:
    if raw is None or str(raw).strip() == "":
        return dict(DEFAULT_FIELD_WEIGHTS)
    out: dict[str, float] = {}
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, value = part.split(":", 1)
            weight = float(value)
        else:
            key, weight = part, 1.0
        key = key.strip()
        if key and weight > 0.0:
            out[key] = float(weight)
    return out or dict(DEFAULT_FIELD_WEIGHTS)


def _trajectory_root(cfg: Any) -> Path:
    c2_cfg = cfg.get("c2", {})
    raw = _get(c2_cfg, "trajectory_root", "experiments/paper_check_flow_lenia/checkpoints/arun_lagrangian_apf_500k")
    path = resolve_path(raw)
    if path is None:
        raise ValueError("Could not resolve c2.trajectory_root.")
    return path


def _path_from_manifest(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw).strip() == "":
        return default
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return root / path


def _iter_trajectories(root: Path, *, include_random: bool) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    items: list[dict[str, Any]] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            kind = str(row.get("candidate_kind", "optimized")).strip().lower()
            if kind != "optimized" and not include_random:
                continue
            traj_id = str(row.get("traj_id", f"traj_{idx:05d}"))
            traj_dir = _path_from_manifest(root, row.get("traj_dir"), default=root / traj_id)
            apf_dir = _path_from_manifest(root, row.get("apf_dir"), default=traj_dir / "apf_logs")
            metrics_path = _path_from_manifest(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            items.append(
                {
                    "traj_id": traj_id,
                    "candidate_kind": kind,
                    "candidate_label": str(row.get("candidate_label", kind)),
                    "traj_dir": traj_dir,
                    "apf_dir": apf_dir,
                    "metrics_path": metrics_path,
                }
            )
    if items:
        return items

    for traj_dir in sorted(root.glob("flow_opt*")):
        if not traj_dir.is_dir():
            continue
        kind = "random" if "_random_" in traj_dir.name else "optimized"
        if kind != "optimized" and not include_random:
            continue
        items.append(
            {
                "traj_id": traj_dir.name,
                "candidate_kind": kind,
                "candidate_label": kind,
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
            }
        )
    return items


def _load_delta_h(metrics_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(metrics_path, allow_pickle=False) as data:
        if "delta_h_best" in data.files:
            delta_h = np.asarray(data["delta_h_best"], dtype=np.float64).reshape(-1)
        elif "delta_h_map" in data.files:
            dh_map = np.asarray(data["delta_h_map"], dtype=np.float64)
            selected = int(np.asarray(data.get("delta_h_selected_tau_idx", np.asarray(0))).reshape(-1)[0])
            delta_h = np.asarray(dh_map[selected], dtype=np.float64).reshape(-1)
        else:
            raise ValueError(f"{metrics_path} has neither delta_h_best nor delta_h_map.")
        if "delta_h_window_center_steps" in data.files:
            centers = np.asarray(data["delta_h_window_center_steps"], dtype=np.float64).reshape(-1)
        elif "delta_h_window_start_steps" in data.files and "delta_h_window_end_steps" in data.files:
            starts = np.asarray(data["delta_h_window_start_steps"], dtype=np.float64).reshape(-1)
            ends = np.asarray(data["delta_h_window_end_steps"], dtype=np.float64).reshape(-1)
            centers = 0.5 * (starts + ends)
        else:
            centers = np.arange(delta_h.size, dtype=np.float64)
        meta = {}
        for key in ("delta_h_selected_tau_steps", "delta_h_selected_tau_idx"):
            if key in data.files:
                arr = np.asarray(data[key]).reshape(-1)
                meta[key] = arr[0].item() if arr.size else None
    n = min(delta_h.size, centers.size)
    if n == 0:
        raise ValueError(f"{metrics_path} has empty Delta-H arrays.")
    return centers[:n], delta_h[:n], meta


def _steps_for_chunk(data: np.lib.npyio.NpzFile, *, start: int, end: int, n: int) -> np.ndarray:
    if "state_t" in data.files:
        state_t = np.asarray(data["state_t"], dtype=np.float64).reshape(-1)
        if state_t.size == n:
            return state_t
    if n <= 1:
        return np.asarray([float(start)], dtype=np.float64)
    return np.linspace(float(start), float(end), int(n), dtype=np.float64)


def _avg_pool_spatial(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    factor = int(factor)
    if factor <= 1 or x.ndim < 4:
        return x
    h = int(x.shape[1])
    w = int(x.shape[2])
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    if h2 < factor or w2 < factor:
        return x
    x = x[:, :h2, :w2, ...]
    rest = x.shape[3:]
    return x.reshape((x.shape[0], h2 // factor, factor, w2 // factor, factor) + rest).mean(axis=(2, 4))


def _load_apf_series(
    apf_dir: Path,
    *,
    field_weights: dict[str, float],
    spatial_downsample: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        raise FileNotFoundError(f"No APF chunks found in {apf_dir}")
    step_parts: list[np.ndarray] = []
    field_parts: dict[str, list[np.ndarray]] = {key: [] for key in field_weights}
    for path, start, end, _idx in chunks:
        with np.load(path, allow_pickle=False) as data:
            n = 0
            for key in field_weights:
                if key not in data.files:
                    continue
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim < 4 or arr.shape[0] < 1:
                    continue
                arr = _avg_pool_spatial(arr, spatial_downsample)
                field_parts[key].append(arr)
                n = max(n, int(arr.shape[0]))
            if n > 0:
                step_parts.append(_steps_for_chunk(data, start=start, end=end, n=n))
    fields = {key: np.concatenate(parts, axis=0) for key, parts in field_parts.items() if parts}
    if not step_parts or not fields:
        raise ValueError(f"No usable APF field arrays found in {apf_dir}.")
    steps = np.concatenate(step_parts).astype(np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    fields = {key: value[order] for key, value in fields.items()}
    keep = np.ones(steps.shape, dtype=bool)
    if steps.size > 1:
        keep[1:] = np.diff(steps) > 0
    steps = steps[keep]
    fields = {key: value[keep] for key, value in fields.items()}
    return steps, fields


def _nearest_indices(steps: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(steps, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    pos = np.searchsorted(s, t, side="left")
    pos = np.clip(pos, 0, s.size - 1)
    prev = np.clip(pos - 1, 0, s.size - 1)
    choose_prev = np.abs(s[prev] - t) <= np.abs(s[pos] - t)
    idx = np.where(choose_prev, prev, pos).astype(np.int64)
    err = np.abs(s[idx] - t)
    return idx, err


def _pool_spatial(arr: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    scale = int(scale)
    if scale <= 1 or x.ndim < 4:
        return x
    h = int(x.shape[1])
    w = int(x.shape[2])
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 < scale or w2 < scale:
        return x
    x = x[:, :h2, :w2, ...]
    rest = x.shape[3:]
    return x.reshape((x.shape[0], h2 // scale, scale, w2 // scale, scale) + rest).mean(axis=(2, 4))


def _field_l2(a: np.ndarray, b: np.ndarray, *, scales: list[int]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n = min(int(aa.shape[0]), int(bb.shape[0]))
    if n < 1:
        return float("nan")
    aa = aa[:n]
    bb = bb[:n]
    vals: list[float] = []
    for scale in scales:
        pa = _pool_spatial(aa, int(scale))
        pb = _pool_spatial(bb, int(scale))
        diff = pa - pb
        vals.append(float(np.sqrt(np.mean(diff * diff))))
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _future_distance(
    *,
    steps: np.ndarray,
    fields: dict[str, np.ndarray],
    t0: float,
    offset_steps: int,
    horizon_steps: int,
    max_future_frames: int,
    field_weights: dict[str, float],
    scales: list[int],
    max_step_error: float,
) -> float:
    if offset_steps == 0:
        return float("nan")
    t1 = float(t0) + float(offset_steps)
    if t0 < steps[0] or t1 < steps[0] or t0 + horizon_steps > steps[-1] or t1 + horizon_steps > steps[-1]:
        return float("nan")
    n_frames = max(2, int(max_future_frames))
    rel = np.linspace(0.0, float(horizon_steps), n_frames, dtype=np.float64)
    idx0, err0 = _nearest_indices(steps, float(t0) + rel)
    idx1, err1 = _nearest_indices(steps, float(t1) + rel)
    valid = (err0 <= max_step_error) & (err1 <= max_step_error)
    if int(np.sum(valid)) < 2:
        return float("nan")
    weighted: list[float] = []
    denom = 0.0
    for key, weight in field_weights.items():
        if key not in fields:
            continue
        d = _field_l2(fields[key][idx0[valid]], fields[key][idx1[valid]], scales=scales)
        if np.isfinite(d):
            weighted.append(float(weight) * d)
            denom += float(weight)
    if not weighted or denom <= 0.0:
        return float("nan")
    return float(sum(weighted) / max(denom, 1e-12))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(finite)) < 2:
        return float("nan")
    xx = xx[finite]
    yy = yy[finite]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _summary(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    x = np.asarray([float(row["delta_h"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["local_divergence"]) for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    return {
        "label": label,
        "n": int(x.size),
        "pearson_r": _corr(x, y),
        "spearman_r": _corr(_average_ranks(x), _average_ranks(y)) if x.size >= 2 else float("nan"),
        "delta_h_min": float(np.nanmin(x)) if x.size else float("nan"),
        "delta_h_max": float(np.nanmax(x)) if x.size else float("nan"),
        "local_divergence_min": float(np.nanmin(y)) if y.size else float("nan"),
        "local_divergence_max": float(np.nanmax(y)) if y.size else float("nan"),
    }


def _write_scatter(rows: list[dict[str, Any]], out_path: Path, title: str) -> str | None:
    if not rows:
        return None
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-cache-c2-local-divergence"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    x = np.asarray([float(row["delta_h"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["local_divergence"]) for row in rows], dtype=np.float64)
    labels = [str(row.get("traj_id", "")) for row in rows]
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    colors = np.asarray([unique[label] for label in labels], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(x[finite], y[finite], c=colors[finite], cmap="tab10", s=24, alpha=0.85)
    if int(np.sum(finite)) >= 2 and float(np.std(x[finite])) > 1e-12 and float(np.std(y[finite])) > 1e-12:
        coef = np.polyfit(x[finite], y[finite], 1)
        xs = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1)
        title = f"{title}; r={np.corrcoef(x[finite], y[finite])[0, 1]:.3g}"
    ax.set_xlabel("Delta-H")
    ax.set_ylabel("single-trajectory local future divergence")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _analyze_item(
    item: dict[str, Any],
    *,
    field_weights: dict[str, float],
    scales: list[int],
    horizon_steps: int,
    offset_steps: list[int],
    max_future_frames: int,
    spatial_downsample: int,
    max_step_error: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = Path(item["metrics_path"])
    apf_dir = Path(item["apf_dir"])
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not apf_dir.exists():
        raise FileNotFoundError(f"Missing APF dir: {apf_dir}")
    centers, delta_h, metric_meta = _load_delta_h(metrics_path)
    steps, fields = _load_apf_series(apf_dir, field_weights=field_weights, spatial_downsample=spatial_downsample)
    if steps.size < 2:
        raise ValueError(f"Too few APF snapshots in {apf_dir}.")
    sample_step = float(np.nanmedian(np.diff(steps)))
    tolerance = float(max_step_error) if max_step_error is not None else max(1.0, 0.55 * sample_step)

    rows: list[dict[str, Any]] = []
    for idx, (center, dh) in enumerate(zip(centers, delta_h)):
        vals = [
            _future_distance(
                steps=steps,
                fields=fields,
                t0=float(center),
                offset_steps=int(offset),
                horizon_steps=int(horizon_steps),
                max_future_frames=int(max_future_frames),
                field_weights=field_weights,
                scales=scales,
                max_step_error=tolerance,
            )
            for offset in offset_steps
        ]
        finite_vals = [float(v) for v in vals if np.isfinite(v)]
        if not finite_vals:
            continue
        row = {
            "traj_id": str(item["traj_id"]),
            "candidate_kind": str(item.get("candidate_kind", "")),
            "candidate_label": str(item.get("candidate_label", "")),
            "window_idx": int(idx),
            "step": int(round(float(center))),
            "delta_h": float(dh),
            "local_divergence": float(np.mean(finite_vals)),
            "local_divergence_median": float(np.median(finite_vals)),
            "n_offsets_used": int(len(finite_vals)),
            "offset_steps_used": ",".join(str(offset) for offset, value in zip(offset_steps, vals) if np.isfinite(value)),
            "metrics_path": str(metrics_path),
            "apf_dir": str(apf_dir),
        }
        for key, value in metric_meta.items():
            row[key] = value
        rows.append(row)
    summary = _summary(rows, label=str(item["traj_id"]))
    summary.update(
        {
            "traj_id": str(item["traj_id"]),
            "candidate_kind": str(item.get("candidate_kind", "")),
            "metrics_path": str(metrics_path),
            "apf_dir": str(apf_dir),
            "n_apf_steps": int(steps.size),
            "apf_step_min": float(steps[0]),
            "apf_step_max": float(steps[-1]),
            "apf_sample_step": sample_step,
            "field_keys_loaded": ",".join(sorted(fields)),
        }
    )
    return rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, _ = load_config(args.config, smoke=False)
    root = resolve_path(args.trajectory_root) if args.trajectory_root else _trajectory_root(cfg)
    if root is None:
        raise ValueError("Could not resolve trajectory root.")
    if not root.exists():
        raise FileNotFoundError(f"Trajectory root not found: {root}")
    output_dir = ensure_dir(
        resolve_path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "analysis" / "results" / "paper_suite" / "c2_local_divergence_probe"
    )
    field_weights = _parse_field_weights(args.field_weights)
    scales = _parse_ints(args.scales, [1, 2, 4])
    offsets = _parse_ints(args.offset_steps, [5000, 10000, 20000])
    items = _iter_trajectories(root, include_random=bool(args.include_random))
    if args.max_trajectories is not None:
        items = items[: max(0, int(args.max_trajectories))]
    if not items:
        raise ValueError(f"No trajectories discovered under {root}")

    all_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        print(f"[c2-local-divergence] {idx}/{len(items)} {item['traj_id']}", flush=True)
        try:
            rows, summary = _analyze_item(
                item,
                field_weights=field_weights,
                scales=scales,
                horizon_steps=int(args.horizon_steps),
                offset_steps=offsets,
                max_future_frames=int(args.max_future_frames),
                spatial_downsample=int(args.spatial_downsample),
                max_step_error=float(args.max_step_error) if args.max_step_error is not None else None,
            )
            all_rows.extend(rows)
            traj_rows.append(summary)
        except Exception as exc:
            failures.append({"traj_id": str(item.get("traj_id", "")), "error": f"{type(exc).__name__}: {exc}"})
            if args.strict:
                raise

    rows_path = output_dir / "local_divergence_rows.csv"
    per_traj_path = output_dir / "local_divergence_by_trajectory.csv"
    summary_path = output_dir / "local_divergence_summary.json"
    figure_path = output_dir / "local_divergence_vs_delta_h.png"
    write_csv(rows_path, all_rows)
    write_csv(per_traj_path, traj_rows)
    pooled = _summary(all_rows, label="pooled")
    figure = _write_scatter(all_rows, figure_path, "C2 local divergence probe")
    summary = {
        "status": "ok" if all_rows else "empty",
        "trajectory_root": str(root),
        "output_dir": str(output_dir),
        "n_trajectories_requested": len(items),
        "n_trajectories_scored": len(traj_rows),
        "n_rows": len(all_rows),
        "pooled": pooled,
        "field_weights": field_weights,
        "scales": scales,
        "horizon_steps": int(args.horizon_steps),
        "offset_steps": offsets,
        "max_future_frames": int(args.max_future_frames),
        "spatial_downsample": int(args.spatial_downsample),
        "rows_csv": str(rows_path),
        "per_trajectory_csv": str(per_traj_path),
        "figure": figure,
        "failures": failures,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="c2_local_divergence_probe_"))
    try:
        root = tmp / "root"
        traj = root / "flow_opt_test"
        apf = traj / "apf_logs"
        apf.mkdir(parents=True)
        steps = np.arange(0, 60000, 1000, dtype=np.int32)
        t = np.linspace(0, 1, steps.size, dtype=np.float32)
        yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
        base = ((yy + xx) / 30.0)[None, :, :, None]
        wave = np.sin(12.0 * t)[:, None, None, None].astype(np.float32)
        a = np.clip(base + 0.1 * wave, 0.0, 1.0).astype(np.float32)
        p = np.repeat(a, 3, axis=-1)
        f = np.concatenate([a, 1.0 - a], axis=-1)
        np.savez_compressed(
            apf / "P_steps_000000_059000__secs_0.000_59.000__idx_0000.npz",
            A=a,
            P=p,
            F=f,
            state_t=steps,
        )
        centers = np.arange(5000, 40000, 5000, dtype=np.int32)
        delta_h = np.linspace(0.0, 1.0, centers.size, dtype=np.float32)
        np.savez_compressed(traj / "metrics.npz", delta_h_best=delta_h, delta_h_window_center_steps=centers)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "trajectories": [
                        {
                            "traj_id": "flow_opt_test",
                            "candidate_kind": "optimized",
                            "traj_dir": "flow_opt_test",
                            "apf_dir": "flow_opt_test/apf_logs",
                            "metrics_path": "flow_opt_test/metrics.npz",
                        }
                    ]
                }
            )
        )
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text("meta:\n  output_root: analysis/results/paper_suite\nc2:\n  trajectory_root: unused\n")
        ns = argparse.Namespace(
            config=str(cfg_path),
            trajectory_root=str(root),
            output_dir=str(tmp / "out"),
            include_random=False,
            max_trajectories=None,
            field_weights="A:1,P:0.25,F:0.25",
            scales="1,2",
            horizon_steps=10000,
            offset_steps="1000,2000",
            max_future_frames=6,
            spatial_downsample=2,
            max_step_error=None,
            strict=True,
        )
        summary = run(ns)
        if summary["n_rows"] <= 0:
            raise AssertionError("self-test produced no rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate C2 future divergence from a single saved APF trajectory, "
            "then correlate it with Delta-H from metrics.npz. This never runs branching simulation."
        )
    )
    parser.add_argument("config", nargs="?", default="experiments/paper_suite/config.yaml")
    parser.add_argument("--trajectory-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--field-weights", default="A:1.0,P:0.25,F:0.25")
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--horizon-steps", type=int, default=30000)
    parser.add_argument("--offset-steps", default="5000,10000,20000")
    parser.add_argument("--max-future-frames", type=int, default=24)
    parser.add_argument("--spatial-downsample", type=int, default=4)
    parser.add_argument("--max-step-error", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        print("self-test ok")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
