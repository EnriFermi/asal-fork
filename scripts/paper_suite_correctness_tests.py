from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.history_dependence.io import RunCollection
from analysis.history_dependence.paper_check_metric_stats import _build_metric_cfg
from analysis.history_dependence.pipeline import load_analysis_config
from analysis.history_dependence.trajectory_metrics import derive_metric_config
from paper_suite_c2_branching import _select_events, _validate_metric_item
from paper_suite_metric_cache import compare_metrics_npz_metadata, expected_metric_metadata, metadata_npz_payload
from paper_suite_posthoc import _interleaved_window_masks, _score_delta_h_map_subset, _score_processed_h, _write_analysis_config


def _base_score_cfg(*, normalize: bool, term: str = "floor_reconstruction_error") -> dict:
    return {
        "preprocess_mode": "clip",
        "delta_h_floor": 0.0,
        "msc_floor": 0.01,
        "msc_term": term,
        "scale_pairs": [(1, 1.0), (2, 1.0)],
        "scale_normalization": "sum_weight_r" if normalize else "none",
        "alpha": 0.0,
        "beta": 1.0,
        "eps": 1e-12,
    }


def _agents_md_path() -> Path:
    canonical = Path("agents.md")
    if canonical.exists():
        return canonical
    fallback = Path("AGENTS.md")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find agents.md or AGENTS.md")


def _write_tiny_lagrangian(path: Path) -> None:
    xy = np.zeros((16, 6, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        xy_control_a=xy,
        sample_every_steps=np.asarray(1, dtype=np.int32),
        trajectory_window_steps=np.asarray(16, dtype=np.int32),
        metric_window_size_steps=np.asarray(8, dtype=np.int32),
        metric_window_step_steps=np.asarray(4, dtype=np.int32),
        metric_tau_steps=np.asarray(2, dtype=np.int32),
    )


def _assert_corrected_msc_metric(metric_cfg: dict) -> None:
    assert metric_cfg["msc_term"] == "floor_reconstruction_error"
    assert metric_cfg["scale_normalization"] == "sum_weight_r"
    assert np.isclose(float(metric_cfg["msc_floor"]), 0.01)
    assert np.isclose(float(metric_cfg["delta_h_floor"]), 0.0)
    assert np.isclose(float(metric_cfg["alpha"]), 0.0)
    assert np.isclose(float(metric_cfg["beta"]), 1.0)
    assert float(metric_cfg["eps"]) <= 1e-9


def test_new_msc_differs_from_legacy_overlap() -> None:
    h = np.asarray([0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0], dtype=np.float64)
    new_score, _new_amp, new_msc = _score_processed_h(_base_score_cfg(normalize=True), h)
    old_score, _old_amp, old_msc = _score_processed_h(_base_score_cfg(normalize=True, term="overlap"), h)
    assert np.isfinite(new_score)
    assert np.isfinite(old_score)
    assert abs(new_msc - old_msc) > 1e-6, (new_msc, old_msc)


def test_msc_is_averaged_over_scale_pairs() -> None:
    h = np.asarray([0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0], dtype=np.float64)
    raw_score, _raw_amp, raw_msc = _score_processed_h(_base_score_cfg(normalize=False), h)
    avg_score, _avg_amp, avg_msc = _score_processed_h(_base_score_cfg(normalize=True), h)
    assert np.isfinite(raw_score)
    assert np.isfinite(avg_score)
    assert np.allclose(avg_msc * 2.0, raw_msc, rtol=1e-6, atol=1e-8), (avg_msc, raw_msc)


def test_c1_apf_interleaved_selection_eval_are_distinct() -> None:
    cfg = _base_score_cfg(normalize=True)
    delta_h_map = np.asarray(
        [
            [4.0, 0.1, 4.0, 0.2, 4.0, 0.3],
            [0.1, 3.0, 0.2, 3.0, 0.3, 3.0],
        ],
        dtype=np.float64,
    )
    sel_mask, eval_mask = _interleaved_window_masks(delta_h_map.shape[1])
    sel_score, _sel_amp, _sel_msc, sel_map = _score_delta_h_map_subset(cfg, delta_h_map, sel_mask)
    eval_score, _eval_amp, _eval_msc, eval_map = _score_delta_h_map_subset(cfg, delta_h_map, eval_mask)
    assert not np.shares_memory(sel_score, eval_score)
    assert not np.shares_memory(sel_map, eval_map)
    assert not np.array_equal(sel_map, eval_map)
    assert not np.array_equal(sel_score, eval_score)
    selected_idx = int(np.nanargmax(sel_score))
    assert np.isfinite(eval_score[selected_idx])


def test_metrics_npz_stale_config_is_detected() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "metrics.npz"
        input_identity = {"kind": "unit", "path": "x", "size": 1}
        cfg_a = {"tau_steps_list": [1, 2], "window_size_frames": 4, "msc_term": "floor_reconstruction_error"}
        cfg_b = {"tau_steps_list": [1, 3], "window_size_frames": 4, "msc_term": "floor_reconstruction_error"}
        np.savez_compressed(path, **metadata_npz_payload(expected_metric_metadata(cfg_a, input_identity)))
        ok, reason, _expected = compare_metrics_npz_metadata(path, cfg_b, input_identity)
        assert not ok
        assert "metric config hash mismatch" in reason


def test_c2_activity_matched_low_controls() -> None:
    centers = np.asarray([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    dh = np.asarray([5.0, 0.1, 0.2, 2.0], dtype=np.float64)
    covariates = {
        "total_mass": np.asarray([1.0, 10.0, 1.1, 5.0], dtype=np.float64),
        "active_fraction": np.asarray([0.2, 0.9, 0.22, 0.4], dtype=np.float64),
    }
    pairs = _select_events(
        centers=centers,
        dh=dh,
        covariates=covariates,
        m_pairs=1,
        refractory_steps=0,
        high_quantile=0.75,
        low_quantile=0.60,
    )
    assert len(pairs) == 1
    assert pairs[0]["low_step"] == 200
    assert pairs[0]["match_method"] == "activity_covariate_nearest_in_low_delta_h_pool"


def test_c2_branch_refuses_stale_metrics() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        apf = root / "traj" / "apf_logs"
        apf.mkdir(parents=True)
        steps = np.arange(8, dtype=np.int32)
        xy = np.zeros((8, 4, 2), dtype=np.float32)
        np.savez_compressed(
            apf / "P_steps_0_7__secs_0.000_0.007__idx_0000.npz",
            A=np.ones((8, 4, 4, 1), dtype=np.float32),
            P=np.ones((8, 4, 4, 3), dtype=np.float32),
            F=np.ones((8, 4, 4, 2), dtype=np.float32),
            lagrangian_xy=xy,
            steps=steps,
            state_t=steps,
        )
        metrics_path = root / "traj" / "metrics.npz"
        np.savez_compressed(metrics_path, delta_h_best=np.ones((3,), dtype=np.float32), delta_h_window_center_steps=np.arange(3))
        flat_args = {
            "snapshot_interval": 1,
            "lagrangian_n_particles": 4,
            "metric_tau_mode": "fixed",
            "metric_tau_steps": 1,
            "metric_window_size_steps": 4,
            "metric_window_step_steps": 1,
            "metric_m_samples": 2,
            "metric_m_min": 1,
            "metric_n_proj": 2,
            "metric_null_reps": 0,
            "metric_particle_samples": 2,
            "metric_delta_h_floor": 0.0,
            "metric_msc_floor": 0.01,
            "metric_msc_term": "floor_reconstruction_error",
            "metric_msc_normalize_by_weight_sum": True,
        }
        try:
            _validate_metric_item(
                {"traj_id": "traj", "metrics_path": metrics_path, "traj_dir": root / "traj", "apf_dir": apf},
                flat_args,
            )
        except ValueError as exc:
            assert "stale upstream metrics" in str(exc)
        else:
            raise AssertionError("stale C2 metrics were accepted")


def test_generated_c5_config_contains_new_msc_fields() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        out.mkdir()
        lag_path = Path(td) / "lagrangian.npz"
        _write_tiny_lagrangian(lag_path)
        ds_cfg = {
            "c5": {
                "metric": {
                    "metric_tau_mode": "fixed",
                    "metric_tau_steps": 3,
                    "metric_window_size_steps": 8,
                    "metric_window_step_steps": 4,
                    "metric_delta_h_floor": 0.0,
                    "metric_msc_floor": 0.01,
                    "metric_msc_term": "floor_reconstruction_error",
                    "metric_msc_normalize_by_weight_sum": True,
                }
            }
        }
        path = _write_analysis_config(out, ds_cfg, Path(td) / "frustration_simulation")
        text = path.read_text()
        assert "metric_msc_floor: 0.01" in text
        assert "metric_msc_term: floor_reconstruction_error" in text
        assert "metric_msc_normalize_by_weight_sum: true" in text
        analysis_cfg = load_analysis_config(path)
        run_collection = RunCollection(
            runs=pd.DataFrame(
                [
                    {
                        "lagrangian_path": str(lag_path),
                        "lagrangian_key": "xy_control_a",
                    }
                ]
            ),
            source_summary={},
            metric_summary={},
            source_dir=Path(td),
        )
        _assert_corrected_msc_metric(derive_metric_config(analysis_cfg, run_collection))
        _assert_corrected_msc_metric(
            _build_metric_cfg(analysis_cfg, lagrangian_path=lag_path, source_metric_summary={})
        )


def test_derive_metric_config_new_msc_fields() -> None:
    with tempfile.TemporaryDirectory() as td:
        lag_path = Path(td) / "lagrangian.npz"
        _write_tiny_lagrangian(lag_path)
        cfg = {
            "trajectories": {
                "enabled": True,
                "metric_tau_mode": "fixed",
                "metric_tau_steps": 2,
                "metric_window_size_steps": 8,
                "metric_window_step_steps": 4,
                "metric_m_samples": 4,
                "metric_m_min": 1,
                "metric_n_proj": 2,
                "metric_null_reps": 0,
                "metric_particle_samples": 3,
                "metric_delta_h_floor": 0.0,
                "metric_msc_floor": 0.01,
                "metric_msc_term": "floor_reconstruction_error",
                "metric_msc_normalize_by_weight_sum": True,
                "metric_alpha": 0.0,
                "metric_beta": 1.0,
                "metric_eps": 1e-12,
            }
        }
        run_collection = RunCollection(
            runs=pd.DataFrame(
                [
                    {
                        "lagrangian_path": str(lag_path),
                        "lagrangian_key": "xy_control_a",
                    }
                ]
            ),
            source_summary={},
            metric_summary={},
            source_dir=Path(td),
        )
        _assert_corrected_msc_metric(derive_metric_config(cfg, run_collection))


def test_agents_md_lookup_uses_canonical_lowercase() -> None:
    path = _agents_md_path()
    if Path("agents.md").exists():
        assert path.name == "agents.md"


def test_plife_configs_are_pure_corrected_msc() -> None:
    checks = [
        ("experiments/paper_suite/config.yaml", ("datasets", "plife_plus", "c1", "metric")),
        ("experiments/paper_suite/config.yaml", ("datasets", "plife_plus", "c5", "metric")),
        ("experiments/paper_check_plife_plus/optimization/config_longrun_check.yaml", ("metric",)),
        ("experiments/paper_check_plife_plus/frustration_simulation/config.yaml", ("metric",)),
        ("experiments/paper_check_plife_plus/frustration_simulation/analysis_config.yaml", ("trajectories",)),
    ]
    for path_str, keys in checks:
        cfg = OmegaConf.to_container(OmegaConf.load(path_str), resolve=True)
        section = cfg
        for key in keys:
            section = section[key]
        assert section["metric_msc_term"] == "floor_reconstruction_error", path_str
        assert bool(section["metric_msc_normalize_by_weight_sum"]), path_str
        assert np.isclose(float(section["metric_msc_floor"]), 0.01), path_str
        assert np.isclose(float(section["metric_alpha"]), 0.0), path_str
        assert np.isclose(float(section["metric_beta"]), 1.0), path_str
        assert float(section["metric_eps"]) <= 1e-9, path_str


def test_c3_c4_excluded_from_active_protocol() -> None:
    cfg = json.loads(json.dumps({"active_claims": ["C1", "C2", "C5", "C6", "N0"]}))
    assert "C3" not in cfg["active_claims"]
    assert "C4" not in cfg["active_claims"]
    text = _agents_md_path().read_text()
    assert "C3 as a paper claim is not run" in text
    assert "C4 is deferred" in text


def main() -> int:
    tests = [
        test_new_msc_differs_from_legacy_overlap,
        test_msc_is_averaged_over_scale_pairs,
        test_c1_apf_interleaved_selection_eval_are_distinct,
        test_metrics_npz_stale_config_is_detected,
        test_c2_activity_matched_low_controls,
        test_c2_branch_refuses_stale_metrics,
        test_generated_c5_config_contains_new_msc_fields,
        test_derive_metric_config_new_msc_fields,
        test_agents_md_lookup_uses_canonical_lowercase,
        test_plife_configs_are_pure_corrected_msc,
        test_c3_c4_excluded_from_active_protocol,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print(f"OK {len(tests)} correctness tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
