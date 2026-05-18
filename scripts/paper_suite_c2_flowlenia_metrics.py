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
from flowlenia_minibang_simulate import compute_metrics_for_run
from paper_suite_common import ensure_dir
from paper_suite_common import load_config as load_suite_config
from paper_suite_common import log_event
from paper_suite_common import resolve_path, write_csv, write_json


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
                    "run_idx": int(row.get("run_idx", -1)),
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
                "traj_dir": traj_dir,
                "apf_dir": traj_dir / "apf_logs",
                "metrics_path": traj_dir / "metrics.npz",
                "manifest_row": {},
            }
        )
    return items


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


def _flat_metric_args(rollout_config: Path) -> dict[str, Any]:
    _cfg, flat = load_rollout_config(rollout_config)
    flat_args = OmegaConf.to_container(flat, resolve=True)
    if not isinstance(flat_args, dict):
        flat_args = dict(flat)
    flat_args["compute_delta_h"] = True
    flat_args["compute_clusters"] = True
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

    items = _iter_trajectories(root)
    if not items:
        if required:
            raise FileNotFoundError(f"No C2 trajectories found under {root}")
        summary = {"status": "skipped", "reason": f"no trajectories under {root}"}
        write_json(out_dir / "c2_highres_metrics_summary.json", summary)
        log_event(f"C2 highres metrics skipped no trajectories root={root}", component="c2-metrics")
        return summary

    rollout_config = _rollout_config(c2_cfg)
    flat_args = _flat_metric_args(rollout_config)
    rows: list[dict[str, Any]] = []
    log_event(f"C2 highres metrics found n_trajectories={len(items)} root={root}", component="c2-metrics")
    for idx, item in enumerate(items, start=1):
        metrics_path = Path(item["metrics_path"])
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
            status = "exists"
            message = ""
            log_event(
                f"C2 highres metrics {idx}/{len(items)} traj={item['traj_id']} exists metrics={metrics_path}",
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
                "selection": {
                    "selection_idx": int(item["selection_idx"]),
                    "iter": int(item.get("run_idx", -1)),
                    "saturation_T": np.nan,
                },
            }
            compute_metrics_for_run(run_row, flat_args)
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

    _update_manifest(root, rows)
    table = out_dir / "c2_highres_metrics_manifest.csv"
    write_csv(table, rows)
    n_apf_ready = sum(1 for row in rows if row["apf_ready"])
    n_metrics_ready = sum(1 for row in rows if row["metrics_ready"])
    summary = {
        "status": "ok" if n_apf_ready == len(rows) and n_metrics_ready == len(rows) else "incomplete",
        "trajectory_root": str(root),
        "rollout_config": str(rollout_config),
        "n_trajectories": len(rows),
        "n_apf_ready": int(n_apf_ready),
        "n_metrics_ready": int(n_metrics_ready),
        "table": str(table),
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
