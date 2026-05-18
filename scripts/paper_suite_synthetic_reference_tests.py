from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from paper_suite_synthetic import _unwrap_periodic_xy, metrics, simulate, visualize


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(root: Path) -> Path:
    path = root / "synthetic_reference_config.yaml"
    path.write_text(
        f"""
meta:
  output_root: {root / "out"}
synthetic:
  families: [S0, S1, S4, S6]
  seeds: 1
  seed_base: 700
  n_particles: 32
  time_steps: 96
  domain_size: 1.0
  tau_grid_steps: [1, 2, 4, 8]
  metric_window_size_steps: 24
  metric_window_step_steps: 12
  metric_range_start_steps: 0
  metric_m_samples: 8
  metric_m_min: 4
  metric_n_proj: 4
  metric_null_reps: 1
  metric_particle_samples: 24
  metric_dirs_seed: 123
  metric_seed: 12345
  metric_delta_h_floor: 0.0
  render:
    enabled: true
    fps: 8
    size_px: 256
    max_frames: 24
    trail_steps: 4
    particle_radius_px: 2
    codec: mp4v
  visualization:
    heatmaps_enabled: true
    heatmap_max_runs: null
""".lstrip()
    )
    return path


def run() -> dict[str, str]:
    wrapped = np.asarray([[[0.90, 0.10]], [[0.10, 0.10]], [[0.30, 0.10]]], dtype=np.float32)
    unwrapped = _unwrap_periodic_xy(wrapped, domain=1.0)
    if abs(float(unwrapped[-1, 0, 0]) - 1.30) > 1e-6:
        raise AssertionError("Synthetic torus unwrap failed across the periodic boundary.")

    with tempfile.TemporaryDirectory(prefix="paper_suite_synthetic_ref_") as td:
        root = Path(td)
        cfg = _write_config(root)
        simulate(cfg, force=True)
        metrics(cfg, force=True)
        visualize(cfg, force=True)
        out = root / "out" / "synthetic_calibration"
        scores_path = out / "per_family_scores.csv"
        tau_path = out / "tau_profiles.csv"
        role_path = out / "role_recovery.csv"
        event_path = out / "event_localization.csv"
        sim_manifest_path = out / "simulation_manifest.csv"
        heatmap_manifest_path = out / "delta_h_heatmap_manifest.csv"

        scores = pd.read_csv(scores_path)
        tau = pd.read_csv(tau_path)
        role = pd.read_csv(role_path)
        events = pd.read_csv(event_path)
        sim_manifest = pd.read_csv(sim_manifest_path)
        heatmap_manifest = pd.read_csv(heatmap_manifest_path)

        expected = {"S0", "S1", "S4", "S6"}
        if set(scores["family"]) != expected:
            raise AssertionError(f"Expected families {expected}, got {set(scores['family'])}")
        for col in ("score", "msc", "amp", "delta_h_mean", "tau_best_steps"):
            if scores[col].isna().any():
                raise AssertionError(f"Non-finite score column {col}")
        if not set(scores["tau_best_steps"].astype(int)).issubset({1, 2, 4, 8}):
            raise AssertionError("Selected tau outside configured grid.")
        required_tau_cols = {"score_by_tau", "amp_by_tau", "msc_by_tau", "delta_h_mean", "delta_h_median", "selected"}
        if not required_tau_cols.issubset(tau.columns):
            raise AssertionError(f"Missing tau profile columns: {required_tau_cols - set(tau.columns)}")
        for fam in expected:
            selected = tau[(tau["family"] == fam) & (tau["selected"].astype(bool))]
            if selected.shape[0] != 1:
                raise AssertionError(f"{fam} should have exactly one selected tau row, got {selected.shape[0]}")
        if role[role["family"].isin(["S4", "S6"])]["ari"].dropna().empty:
            raise AssertionError("S4/S6 role recovery ARI rows missing.")
        if events[events["family"] == "S6"].empty:
            raise AssertionError("S6 event localization row missing.")
        if (events["event_error_steps"].astype(float) < 0).any():
            raise AssertionError("Event localization error must be non-negative.")
        if sim_manifest["video_path"].isna().any() or sim_manifest["video_status"].isin(["written", "rewritten", "exists"]).sum() != len(expected):
            raise AssertionError("Synthetic simulation did not render expected trajectory videos.")
        for raw_path in sim_manifest["video_path"].astype(str):
            video_path = Path(raw_path)
            if not video_path.exists() or video_path.stat().st_size <= 1024:
                raise AssertionError(f"Synthetic video missing or empty: {video_path}")
        if heatmap_manifest.empty or heatmap_manifest["status"].isin(["written", "exists"]).sum() != len(expected):
            raise AssertionError("Synthetic Delta-H heatmap manifest is missing rendered runs.")
        for raw_path in heatmap_manifest["path"].astype(str):
            heatmap_path = Path(raw_path)
            if not heatmap_path.exists() or heatmap_path.stat().st_size <= 1024:
                raise AssertionError(f"Synthetic Delta-H heatmap missing or empty: {heatmap_path}")
        for figure in [
            root / "out" / "figures" / "synthetic_calibration_grid.png",
            root / "out" / "figures" / "synthetic_delta_h_heatmaps.png",
        ]:
            if not figure.exists() or figure.stat().st_size <= 1024:
                raise AssertionError(f"Synthetic visualization figure missing or empty: {figure}")

        before = (_sha(scores_path), _sha(tau_path), _sha(role_path), _sha(event_path))
        metrics(cfg, force=True)
        after = (_sha(scores_path), _sha(tau_path), _sha(role_path), _sha(event_path))
        if before != after:
            raise AssertionError("Synthetic force recompute is not deterministic.")
        return {"status": "ok", "output_root": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny deterministic synthetic calibration reference tests.")
    parser.parse_args()
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
