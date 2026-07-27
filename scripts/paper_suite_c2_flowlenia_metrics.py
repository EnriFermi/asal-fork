from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from omegaconf import OmegaConf

from flowlenia_minibang_common import list_apf_chunks
from flowlenia_minibang_common import load_config as load_rollout_config
from flowlenia_minibang_simulate import compute_metrics_for_run, expected_delta_h_metric_metadata
from paper_suite_common import ensure_dir
from paper_suite_common import load_config as load_suite_config
from paper_suite_common import log_event
from paper_suite_common import resolve_path, write_csv, write_json
from paper_suite_metric_cache import compare_metrics_npz_metadata


REQUIRED_APF_KEYS = (
    "A",
    "P",
    "F",
    "lagrangian_xy",
    "lagrangian_c",
    "resume_batch_rng_key",
    "state_t",
    "state_mass_cycle_start",
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _trajectory_root(c2_cfg: Any) -> Path | None:
    raw = _get(c2_cfg, "trajectory_root", None)
    if raw is None:
        raw = _get(c2_cfg, "minibang_root", "experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set")
    return resolve_path(raw)


def _highres_section(c2_cfg: Any) -> Any:
    return _get(c2_cfg, "flow_lenia_highres", {})


def _rollout_config(c2_cfg: Any) -> Path:
    section = _highres_section(c2_cfg)
    raw = _get(section, "rollout_config", "experiments/paper_suite/c2_flowlenia_highres_rollout.yaml")
    path = resolve_path(raw)
    if path is None or not path.exists():
        raise FileNotFoundError(f"C2 high-res rollout config not found: {path}")
    return path


def _path_from_manifest(root: Path, raw: Any, *, default: Path) -> Path:
    if raw is None or str(raw) == "":
        return default
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return root / path


def _iter_trajectories(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    items: list[dict[str, Any]] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for idx, row in enumerate(payload.get("trajectories", [])):
            traj_id = str(row.get("traj_id", f"traj_{idx:05d}"))
            traj_dir = _path_from_manifest(root, row.get("traj_dir"), default=root / traj_id / "traj_00000")
            apf_dir = _path_from_manifest(root, row.get("apf_dir"), default=traj_dir / "apf_logs")
            metrics_path = _path_from_manifest(root, row.get("metrics_path"), default=traj_dir / "metrics.npz")
            items.append(
                {
                    "traj_id": traj_id,
                    "selection_idx": int(row.get("selection_idx", idx)),
                    "run_idx": int(row.get("suite_run_idx", row.get("run_idx", -1))),
                    "source_run_idx": int(row.get("source_run_idx", row.get("suite_run_idx", row.get("run_idx", -1)))),
                    "candidate_kind": str(row.get("candidate_kind", "optimized")),
                    "candidate_idx": int(row.get("candidate_idx", 0)),
                    "candidate_label": str(row.get("candidate_label", row.get("candidate_kind", "optimized"))),
                    "rollout_seed_idx": int(row.get("rollout_seed_idx", 0)),
                    "traj_dir": traj_dir,
                    "apf_dir": apf_dir,
                    "metrics_path": metrics_path,
                    "manifest_row": row,
                }
            )
    if items:
        return items

    for idx, traj_dir in enumerate(sorted(root.glob("flow_opt_*"))):
        if not traj_dir.is_dir() or not (traj_dir / "apf_logs").is_dir():
            continue
        items.append(
            {
                "traj_id": traj_dir.name,
                "selection_idx": idx,
                "run_idx": -1,
                "candidate_kind": "optimized",
                "candidate_idx": 0,
                "candidate_label": "optimized",
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
                "manifest_row": {},
            }
        )
    if items:
        return items

    for idx, traj_dir in enumerate(sorted(root.glob("flow_opt_*/traj_*"))):
        if not traj_dir.is_dir():
            continue
        items.append(
            {
                "traj_id": traj_dir.parent.name,
                "selection_idx": idx,
                "run_idx": -1,
                "candidate_kind": "optimized",
                "candidate_idx": 0,
                "candidate_label": "optimized",
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
                "manifest_row": {},
            }
        )
    if items:
        return items

    for idx, traj_dir in enumerate(sorted(root.glob("traj_*"))):
        if not traj_dir.is_dir():
            continue
        items.append(
            {
                "traj_id": traj_dir.name,
                "selection_idx": idx,
                "run_idx": -1,
                "candidate_kind": "optimized",
                "candidate_idx": 0,
                "candidate_label": "optimized",
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
                "manifest_row": {},
            }
        )
    return items


def _int_set(raw: Any) -> set[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, int)):
        raw = [raw]
    return {int(value) for value in raw}


def _str_set(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    return {str(value).strip().lower() for value in raw}


def _filter_trajectories(items: list[dict[str, Any]], c2_cfg: Any) -> list[dict[str, Any]]:
    source_filter = _get(c2_cfg, "source_filter", {})
    kinds = _str_set(_get(source_filter, "candidate_kinds", None))
    seed_indices = _int_set(_get(source_filter, "rollout_seed_indices", None))
    run_indices = _int_set(_get(source_filter, "source_run_indices", None))
    filtered = []
    for item in items:
        if kinds is not None and str(item.get("candidate_kind", "")).strip().lower() not in kinds:
            continue
        if seed_indices is not None and int(item.get("rollout_seed_idx", 0)) not in seed_indices:
            continue
        if run_indices is not None and int(item.get("source_run_idx", item.get("run_idx", -1))) not in run_indices:
            continue
        filtered.append(item)

    expected = _get(source_filter, "expected_trajectories", None)
    if expected is not None and len(filtered) != int(expected):
        raise ValueError(
            f"C2 source_filter selected {len(filtered)} trajectories, expected {int(expected)}."
        )
    if bool(_get(source_filter, "require_one_per_source_run", False)):
        counts: dict[int, int] = {}
        for item in filtered:
            run_idx = int(item.get("source_run_idx", item.get("run_idx", -1)))
            counts[run_idx] = counts.get(run_idx, 0) + 1
        bad = {run_idx: count for run_idx, count in counts.items() if count != 1}
        expected_runs = run_indices if run_indices is not None else set(counts)
        missing = sorted(expected_runs - set(counts))
        if bad or missing:
            raise ValueError(
                "C2 source_filter requires exactly one trajectory per source run; "
                f"bad_counts={bad}, missing_runs={missing}."
            )
    return filtered


def _apf_status(apf_dir: Path) -> tuple[bool, str, int]:
    if not apf_dir.exists():
        return False, f"missing APF dir {apf_dir}", 0
    chunks = list_apf_chunks(apf_dir)
    if not chunks:
        return False, f"missing APF chunks in {apf_dir}", 0
    first = chunks[0][0]
    try:
        with np.load(first, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_APF_KEYS if key not in data.files]
    except Exception as exc:
        return False, f"cannot read APF chunk {first}: {exc}", len(chunks)
    if missing:
        return False, f"{first} missing APF keys: {','.join(missing)}", len(chunks)
    return True, "", len(chunks)


def _flat_metric_args(rollout_config: Path, *, compute_clusters: bool = True) -> dict[str, Any]:
    _cfg, flat = load_rollout_config(rollout_config)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    if not isinstance(flat_args, dict):
        flat_args = dict(flat)
    flat_args["compute_delta_h"] = True
    flat_args["compute_clusters"] = bool(compute_clusters)
    flat_args["metrics_strict"] = True
    return flat_args


def _update_manifest(root: Path, rows: list[dict[str, Any]]) -> None:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return
    payload = json.loads(manifest.read_text())
    by_id = {str(row["traj_id"]): row for row in rows}
    for item in payload.get("trajectories", []):
        row = by_id.get(str(item.get("traj_id", "")))
        if row is None:
            continue
        item["apf_ready"] = bool(row["apf_ready"])
        item["metrics_ready"] = bool(row["metrics_ready"])
        item["metrics_status"] = str(row["status"])
        item["apf_status"] = str(row["message"])
    write_json(manifest, payload)


def _write_derived_manifest(cache_root: Path, items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Path:
    rows_by_id = {str(row["traj_id"]): row for row in rows}
    trajectories = []
    for item in items:
        traj_id = str(item["traj_id"])
        metric_row = rows_by_id[traj_id]
        source_row = dict(item.get("manifest_row", {}))
        source_row.update(
            {
                "traj_id": traj_id,
                "selection_idx": int(item["selection_idx"]),
                "suite_run_idx": int(item.get("run_idx", -1)),
                "source_run_idx": int(item.get("source_run_idx", item.get("run_idx", -1))),
                "candidate_kind": str(item.get("candidate_kind", "optimized")),
                "candidate_idx": int(item.get("candidate_idx", 0)),
                "candidate_label": str(item.get("candidate_label", item.get("candidate_kind", "optimized"))),
                "rollout_seed_idx": int(item.get("rollout_seed_idx", 0)),
                "traj_dir": str(Path(item["traj_dir"]).resolve()),
                "apf_dir": str(Path(item["apf_dir"]).resolve()),
                "metrics_path": str(Path(metric_row["metrics_path"]).resolve()),
                "apf_ready": bool(metric_row["apf_ready"]),
                "metrics_ready": bool(metric_row["metrics_ready"]),
                "metrics_status": str(metric_row["status"]),
            }
        )
        trajectories.append(source_row)
    manifest = cache_root / "manifest.json"
    write_json(
        manifest,
        {
            "source": "paper_suite_c2_fixed_training_trajectory_view",
            "n_trajectories": len(trajectories),
            "trajectories": trajectories,
        },
    )
    return manifest


def run(config_path: str | Path, *, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    cfg, _ = load_suite_config(config_path, smoke=smoke)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    out_dir = ensure_dir(output_root / "c2_highres_metrics")
    log_event(f"C2 highres metrics start smoke={smoke} force={force} output={out_dir}", component="c2-metrics")
    if smoke:
        summary = {"status": "smoke_skipped", "reason": "smoke C2 metrics use generated tiny metrics fixtures"}
        write_json(out_dir / "c2_highres_metrics_summary.json", summary)
        log_event("C2 highres metrics smoke skipped", component="c2-metrics")
        return summary

    c2_cfg = cfg.get("c2", {})
    root = _trajectory_root(c2_cfg)
    required = bool(_get(c2_cfg, "required", False))
    if root is None or not root.exists():
        if required:
            raise FileNotFoundError(f"C2 trajectory root not found: {root}")
        summary = {"status": "skipped", "reason": f"missing trajectory root {root}"}
        write_json(out_dir / "c2_highres_metrics_summary.json", summary)
        log_event(f"C2 highres metrics skipped missing root={root}", component="c2-metrics")
        return summary

    items = _filter_trajectories(_iter_trajectories(root), c2_cfg)
    if not items:
        if required:
            raise FileNotFoundError(f"No C2 trajectories found under {root}")
        summary = {"status": "skipped", "reason": f"no trajectories under {root}"}
        write_json(out_dir / "c2_highres_metrics_summary.json", summary)
        log_event(f"C2 highres metrics skipped no trajectories root={root}", component="c2-metrics")
        return summary

    rollout_config = _rollout_config(c2_cfg)
    flat_args = _flat_metric_args(
        rollout_config,
        compute_clusters=bool(_get(c2_cfg, "compute_source_clusters", True)),
    )
    cache_root_raw = _get(c2_cfg, "metrics_cache_root", None)
    cache_root = ensure_dir(resolve_path(cache_root_raw)) if cache_root_raw is not None else None
    rows: list[dict[str, Any]] = []
    log_event(f"C2 highres metrics found n_trajectories={len(items)} root={root}", component="c2-metrics")
    for idx, item in enumerate(items, start=1):
        metrics_path = (
            cache_root / str(item["traj_id"]) / "metrics.npz"
            if cache_root is not None
            else Path(item["metrics_path"])
        )
        apf_ready, apf_message, n_chunks = _apf_status(Path(item["apf_dir"]))
        if not apf_ready:
            status = "missing_apf"
            message = apf_message
            log_event(
                f"C2 highres metrics {idx}/{len(items)} traj={item['traj_id']} missing_apf message={message}",
                component="c2-metrics",
            )
            if required:
                raise FileNotFoundError(f"Cannot compute C2 metrics for {item['traj_id']}: {message}")
        elif metrics_path.exists() and not force:
            metric_cfg, input_identity, _metadata = expected_delta_h_metric_metadata(Path(item["apf_dir"]), flat_args)
            fresh, reason, _expected = compare_metrics_npz_metadata(metrics_path, metric_cfg, input_identity)
            if not fresh:
                raise ValueError(
                    f"C2 highres metrics stale cache for {item['traj_id']} at {metrics_path}: {reason}. "
                    "Run the metrics layer with --force to recompute from existing APF logs."
                )
            status = "exists"
            message = "fresh"
            log_event(
                f"C2 highres metrics {idx}/{len(items)} traj={item['traj_id']} exists fresh metrics={metrics_path}",
                component="c2-metrics",
            )
        else:
            log_event(
                f"C2 highres metrics {idx}/{len(items)} traj={item['traj_id']} computing from apf_chunks={n_chunks}",
                component="c2-metrics",
            )
            run_row = {
                "traj_id": str(item["traj_id"]),
                "traj_dir": Path(item["traj_dir"]),
                "apf_dir": Path(item["apf_dir"]),
                "metrics_path": metrics_path,
                "metrics_summary_path": metrics_path.with_name("metrics_summary.json"),
                "selection": {
                    "selection_idx": int(item["selection_idx"]),
                    "iter": int(item.get("run_idx", -1)),
                    "saturation_T": np.nan,
                },
            }
            compute_metrics_for_run(run_row, flat_args)
            metric_cfg, input_identity, _metadata = expected_delta_h_metric_metadata(Path(item["apf_dir"]), flat_args)
            fresh, reason, _expected = compare_metrics_npz_metadata(metrics_path, metric_cfg, input_identity)
            if not fresh:
                raise ValueError(f"C2 highres metrics wrote invalid cache for {item['traj_id']} at {metrics_path}: {reason}")
            status = "computed"
            message = ""
            log_event(
                f"C2 highres metrics {idx}/{len(items)} traj={item['traj_id']} computed metrics={metrics_path}",
                component="c2-metrics",
            )
        rows.append(
            {
                "traj_id": str(item["traj_id"]),
                "selection_idx": int(item["selection_idx"]),
                "run_idx": int(item.get("run_idx", -1)),
                "source_run_idx": int(item.get("source_run_idx", item.get("run_idx", -1))),
                "candidate_kind": str(item.get("candidate_kind", "optimized")),
                "candidate_idx": int(item.get("candidate_idx", 0)),
                "candidate_label": str(item.get("candidate_label", item.get("candidate_kind", "optimized"))),
                "rollout_seed_idx": int(item.get("rollout_seed_idx", 0)),
                "traj_dir": str(item["traj_dir"]),
                "apf_dir": str(item["apf_dir"]),
                "metrics_path": str(metrics_path),
                "apf_ready": bool(apf_ready),
                "metrics_ready": bool(metrics_path.exists()),
                "n_apf_chunks": int(n_chunks),
                "status": status,
                "message": message,
            }
        )

    derived_manifest = None
    if cache_root is None:
        _update_manifest(root, rows)
    else:
        derived_manifest = _write_derived_manifest(cache_root, items, rows)
    table = out_dir / "c2_highres_metrics_manifest.csv"
    write_csv(table, rows)
    n_apf_ready = sum(1 for row in rows if row["apf_ready"])
    n_metrics_ready = sum(1 for row in rows if row["metrics_ready"])
    n_optimized = sum(1 for row in rows if str(row.get("candidate_kind", "optimized")) == "optimized")
    n_random = sum(1 for row in rows if str(row.get("candidate_kind", "optimized")) == "random")
    summary = {
        "status": "ok" if n_apf_ready == len(rows) and n_metrics_ready == len(rows) else "incomplete",
        "trajectory_root": str(root),
        "rollout_config": str(rollout_config),
        "n_trajectories": len(rows),
        "n_optimized": int(n_optimized),
        "n_random": int(n_random),
        "n_apf_ready": int(n_apf_ready),
        "n_metrics_ready": int(n_metrics_ready),
        "table": str(table),
        "derived_manifest": None if derived_manifest is None else str(derived_manifest),
    }
    write_json(out_dir / "c2_highres_metrics_summary.json", summary)
    log_event(
        f"C2 highres metrics done status={summary['status']} n_metrics_ready={n_metrics_ready}/{len(rows)} table={table}",
        component="c2-metrics",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute C2 Flow-Lenia high-res metrics from saved APF logs.")
    parser.add_argument("config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, smoke=args.smoke, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
