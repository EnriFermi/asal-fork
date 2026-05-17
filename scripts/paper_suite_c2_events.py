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

from paper_suite_common import ensure_dir, load_config, resolve_path, write_csv, write_json


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _write_smoke_minibang(output_root: Path) -> Path:
    root = ensure_dir(output_root / "smoke_inputs" / "minibang")
    traj = ensure_dir(root / "traj_00000")
    steps = np.arange(20, 140, 20, dtype=np.int64)
    tau_steps = np.asarray([2, 4, 8], dtype=np.int32)
    dh_map = np.vstack(
        [
            np.sin(np.linspace(0, np.pi, steps.size)) * 0.1,
            np.sin(np.linspace(0, np.pi, steps.size)) * 0.4,
            np.sin(np.linspace(0, np.pi, steps.size)) * 0.2,
        ]
    ).astype(np.float32)
    selected = 1
    np.savez_compressed(
        traj / "metrics.npz",
        delta_h_map=dh_map,
        delta_h_best=dh_map[selected],
        delta_h_tau_steps=tau_steps,
        delta_h_selected_tau_idx=np.asarray(selected, dtype=np.int32),
        delta_h_selected_tau_steps=np.asarray(int(tau_steps[selected]), dtype=np.int32),
        delta_h_window_center_steps=steps,
        delta_h_window_start_steps=steps - 10,
        delta_h_window_end_steps=steps + 10,
        cluster_tv_steps=steps,
        cluster_tv=np.abs(np.gradient(dh_map[selected])),
    )
    manifest = {"trajectories": [{"traj_id": "traj_00000", "traj_dir": str(traj), "metrics_path": str(traj / "metrics.npz")}], "config_path": ""}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def _iter_metric_paths(root: Path) -> list[Path]:
    manifest = root / "manifest.json"
    paths: list[Path] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for row in payload.get("trajectories", []):
            raw = row.get("metrics_path")
            if raw:
                path = Path(str(raw))
                if not path.is_absolute():
                    path = root / Path(str(row.get("traj_id", ""))) / path.name
                if path.exists():
                    paths.append(path)
            traj_id = row.get("traj_id")
            if traj_id:
                candidate = root / str(traj_id) / "metrics.npz"
                if candidate.exists() and candidate not in paths:
                    paths.append(candidate)
    if not paths:
        paths = sorted(root.glob("traj_*/metrics.npz"))
    return paths


def _safe_arr(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key])


def run(config_path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    cfg, _ = load_config(config_path, smoke=smoke)
    output_root = ensure_dir(resolve_path(cfg.get("meta", {}).get("output_root", "analysis/results/paper_suite")) or Path("analysis/results/paper_suite"))
    c2_cfg = cfg.get("c2", {})
    root = _write_smoke_minibang(output_root) if smoke else resolve_path(_get(c2_cfg, "minibang_root", "experiments/flow_lenia_mspd/checkpoints/test_run_longrun_check/minibang_golden_set"))
    if root is None or not root.exists():
        required = bool(_get(c2_cfg, "required", False))
        if required:
            raise FileNotFoundError(f"C2 minibang root not found: {root}")
        summary = {"status": "skipped", "reason": f"missing minibang root {root}"}
        write_json(output_root / "c2_event_summary.json", summary)
        return summary
    metric_paths = _iter_metric_paths(root)
    rows = []
    for path in metric_paths:
        with np.load(path, allow_pickle=False) as data:
            dh = _safe_arr(data, "delta_h_best")
            if dh is None:
                dh_map = _safe_arr(data, "delta_h_map")
                selected = int(np.asarray(_safe_arr(data, "delta_h_selected_tau_idx", np.asarray(0))).item())
                dh = np.asarray(dh_map[selected], dtype=np.float64)
            dh = np.asarray(dh, dtype=np.float64).reshape(-1)
            centers = _safe_arr(data, "delta_h_window_center_steps")
            if centers is None:
                starts = _safe_arr(data, "delta_h_window_start_steps", np.arange(dh.size))
                centers = np.asarray(starts, dtype=np.float64)
            centers = np.asarray(centers, dtype=np.float64).reshape(-1)
            tau = _safe_arr(data, "delta_h_selected_tau_steps", np.asarray(np.nan))
            tau_val = float(np.asarray(tau).reshape(-1)[0]) if tau is not None else float("nan")
            peak_idx = int(np.nanargmax(dh)) if dh.size else -1
            tv = _safe_arr(data, "cluster_tv", None)
            tv_peak = float(np.nanmax(tv)) if tv is not None and np.asarray(tv).size else float("nan")
            rows.append(
                {
                    "traj_id": path.parent.name,
                    "metrics_path": str(path),
                    "selected_tau_steps": tau_val,
                    "delta_h_peak_step": float(centers[peak_idx]) if peak_idx >= 0 and peak_idx < centers.size else float("nan"),
                    "delta_h_peak": float(dh[peak_idx]) if peak_idx >= 0 else float("nan"),
                    "delta_h_mean": float(np.nanmean(dh)) if dh.size else float("nan"),
                    "cluster_tv_peak": tv_peak,
                }
            )
    out_dir = ensure_dir(output_root / "c2_events")
    write_csv(out_dir / "c2_event_summary.csv", rows)
    summary = {"status": "ok", "n_trajectories": len(rows), "minibang_root": str(root), "table": str(out_dir / "c2_event_summary.csv")}
    write_json(output_root / "c2_event_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline C2 event report from minibang/APF metrics.")
    parser.add_argument("config")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, smoke=args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

