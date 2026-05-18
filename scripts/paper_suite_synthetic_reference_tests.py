from __future__ import annotations

import argparse
import hashlib
import json
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
  families: [S0, S1, S4, S6, S8]
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
  metric_delta_h_floor: 1.0e-6
  metric_msc_floor: 0.01
  metric_eps: 1.0e-12
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
    msc_r_tau_steps: null
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
        msc_scale_path = out / "msc_scale_profiles.csv"
        role_path = out / "role_recovery.csv"
        event_path = out / "event_localization.csv"
        sim_manifest_path = out / "simulation_manifest.csv"
        heatmap_manifest_path = out / "delta_h_heatmap_manifest.csv"

        scores = pd.read_csv(scores_path)
        tau = pd.read_csv(tau_path)
        msc_scale = pd.read_csv(msc_scale_path)
        role = pd.read_csv(role_path)
        events = pd.read_csv(event_path)
        sim_manifest = pd.read_csv(sim_manifest_path)
        heatmap_manifest = pd.read_csv(heatmap_manifest_path)

        expected = {"S0", "S1", "S4", "S6", "S8"}
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
        required_msc_cols = {
            "family",
            "seed",
            "tau_steps",
            "scale_r",
            "scale_weight",
            "msc_r_raw",
            "msc_r_weighted_unnormalized",
            "msc_r_weighted",
            "msc_by_tau",
            "selected",
        }
        if not required_msc_cols.issubset(msc_scale.columns):
            raise AssertionError(f"Missing MSC scale columns: {required_msc_cols - set(msc_scale.columns)}")
        if msc_scale.empty or msc_scale[["msc_r_raw", "msc_r_weighted_unnormalized", "msc_r_weighted"]].isna().any().any():
            raise AssertionError("MSC scale decomposition is empty or non-finite.")
        summed = msc_scale.groupby(["family", "seed", "tau_steps"])["msc_r_weighted"].sum().reset_index(name="msc_sum")
        joined = summed.merge(tau[["family", "seed", "tau_steps", "msc_by_tau"]], on=["family", "seed", "tau_steps"], how="inner")
        if joined.empty or np.max(np.abs(joined["msc_sum"].to_numpy() - joined["msc_by_tau"].to_numpy())) > 1e-5:
            raise AssertionError("Normalized MSC scale weighted terms do not sum to msc_by_tau.")
        with np.load(str(scores.iloc[0]["metrics_path"]), allow_pickle=False) as data:
            metric_cfg = json.loads(str(np.asarray(data["_metric_config_json"]).item()))
            if metric_cfg.get("msc_term") != "floor_reconstruction_error":
                raise AssertionError(f"Synthetic MSC term is not floor_reconstruction_error: {metric_cfg.get('msc_term')}")
            if metric_cfg.get("scale_normalization") != "sum_weight_r":
                raise AssertionError(f"Synthetic MSC is not normalized by weight sum: {metric_cfg.get('scale_normalization')}")
            h_map = np.asarray(data["delta_h_processed_map"], dtype=np.float64)
            raw_by_scale = np.asarray(data["msc_raw_by_scale_by_tau"], dtype=np.float64)
            weighted_by_scale = np.asarray(data["msc_by_scale_by_tau"], dtype=np.float64)
            scale_r = np.asarray(data["msc_scale_r"], dtype=np.int32)
            scale_w = np.asarray(data["msc_scale_weight"], dtype=np.float64)
            if abs(float(metric_cfg.get("delta_h_floor", 0.0)) - 1.0e-6) > 1e-12:
                raise AssertionError("Synthetic Delta-H cutoff floor did not come from metric_delta_h_floor.")
            if abs(float(metric_cfg.get("msc_floor", 0.0)) - 0.01) > 1e-12:
                raise AssertionError("Synthetic MSC denominator floor did not come from metric_msc_floor.")
            floor = float(metric_cfg.get("msc_floor", 0.0))
            eps = float(metric_cfg.get("eps", 1e-12))
            expected_raw = np.zeros_like(raw_by_scale)
            for tau_i in range(h_map.shape[0]):
                h = h_map[tau_i]
                W = int(h.shape[0])
                for scale_i, r_val in enumerate(scale_r):
                    r = int(r_val)
                    u_r = W // r
                    u_2r = W // (2 * r)
                    g = h[: u_r * r].reshape(u_r, r).mean(axis=1)
                    coarse = h[: u_2r * (2 * r)].reshape(u_2r, 2 * r).mean(axis=1)
                    u_cmp = min(u_r, 2 * u_2r)
                    g_cmp = g[:u_cmp]
                    up = np.repeat(coarse, 2)[:u_cmp]
                    expected_raw[tau_i, scale_i] = np.mean((g_cmp - up) ** 2) / (
                        np.mean(g_cmp * g_cmp) + floor * floor + eps
                    )
            expected_weighted = expected_raw * scale_w[None, :] / (np.sum(scale_w) + eps)
            if not np.allclose(raw_by_scale, expected_raw, rtol=1e-5, atol=1e-7):
                raise AssertionError("Synthetic MSC raw scale terms do not match floor-aware reconstruction error.")
            if not np.allclose(weighted_by_scale, expected_weighted, rtol=1e-5, atol=1e-7):
                raise AssertionError("Synthetic MSC weighted terms are not normalized by sum(weight_r).")
        for fam in expected:
            selected = tau[(tau["family"] == fam) & (tau["selected"].astype(bool))]
            if selected.shape[0] != 1:
                raise AssertionError(f"{fam} should have exactly one selected tau row, got {selected.shape[0]}")
        for fam in ["S4", "S6", "S8"]:
            if role[role["family"] == fam]["ari"].dropna().empty:
                raise AssertionError(f"{fam} role recovery ARI rows missing.")
        if events[events["family"] == "S6"].empty:
            raise AssertionError("S6 event localization row missing.")
        if events[events["family"] == "S8"].empty:
            raise AssertionError("S8 split event localization row missing.")
        s8_rows = sim_manifest[sim_manifest["family"] == "S8"]
        if s8_rows.empty:
            raise AssertionError("S8 simulation manifest row missing.")
        with np.load(str(s8_rows.iloc[0]["path"]), allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            if metadata.get("expected") != "split_transition_s3_to_s7":
                raise AssertionError(f"Unexpected S8 metadata: {metadata}")
            if "labels_t" not in data.files or np.unique(np.asarray(data["labels_t"])[-1]).size < 2:
                raise AssertionError("S8 must store dynamic labels that split into multiple final blobs.")
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
            root / "out" / "figures" / "synthetic_msc_by_scale.png",
            root / "out" / "figures" / "synthetic_delta_h_heatmaps.png",
        ]:
            if not figure.exists() or figure.stat().st_size <= 1024:
                raise AssertionError(f"Synthetic visualization figure missing or empty: {figure}")

        before = (_sha(scores_path), _sha(tau_path), _sha(msc_scale_path), _sha(role_path), _sha(event_path))
        metrics(cfg, force=True)
        after = (_sha(scores_path), _sha(tau_path), _sha(msc_scale_path), _sha(role_path), _sha(event_path))
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
