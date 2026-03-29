from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .embedding_metrics import compute_embedding_pairwise
from .io import build_run_collection, load_embeddings, load_lagrangian
from .report_main_observable import generate_main_observable_report, generate_trial_level_metric_tests
from .reporting import (
    compute_concordance,
    compute_effect_sizes,
    permutation_test_from_matrix,
    plot_delta_h_examples,
    plot_distance_matrix,
    plot_frame_panel,
    plot_pair_strip,
    plot_scatter,
    plot_table,
    summarize_pairwise_values,
)
from .trajectory_metrics import (
    compute_trajectory_observables,
    compute_trajectory_pairwise,
    derive_metric_config,
)
from .utils import REPO_ROOT, ensure_dir, progress_bar, save_dataframe, save_matrix, save_npz, write_json
from .utils import resolve_config_path


DEFAULT_CONFIG: dict[str, Any] = {
    "source": {
        "type": "history_dependence_eval",
        "path": "experiments/frustration/checkpoints/history_dependence_opt1_best",
    },
    "output": {
        "dir": "experiments/frustration/checkpoints/history_dependence_opt1_best/offline_analysis",
    },
    "embeddings": {
        "enabled": True,
        "normalize": True,
        "synced_metrics": ["cosine", "euclidean"],
        "cloud_metrics": ["cosine"],
        "cloud_method": "chamfer",
        "primary_metric": "embedding_cloud_chamfer_cosine",
    },
    "trajectories": {
        "enabled": True,
        "metric_tau_mode": "max_grid",
        "metric_tau_grid_steps": [1000, 3000, 5000, 7000, 9000],
        "metric_window_size_steps": 20000,
        "metric_window_step_steps": 5000,
        "metric_m_samples": 48,
        "metric_m_min": 4,
        "metric_n_proj": 16,
        "metric_null_reps": 6,
        "metric_particle_samples": 64,
        "metric_dirs_seed": 123,
        "metric_preprocess_mode": "clip",
        "metric_alpha": 0.0,
        "metric_beta": 1.0,
        "occupancy_bins": 64,
        "pairwise_map_metrics": ["l2", "mean_abs"],
        "fixed_tau_distribution_steps": 3000,
        "pairwise_distribution_metrics": ["wasserstein", "ks", "energy"],
        "primary_map_metric": "delta_h_l2",
    },
    "statistics": {
        "permutation_max_exact": 200000,
        "permutation_samples": 20000,
        "permutation_seed": 0,
    },
    "main_observable": {
        "enabled": True,
        "distance_name": "embedding_cloud_chamfer_cosine",
        "class_a": "free-wall",
        "class_b": "free-free",
        "tie_tolerance": 0.0,
        "bootstrap_reps": 2000,
        "bootstrap_seed": 0,
        "ci_level": 0.95,
        "permutation_max_exact": 200000,
        "permutation_samples": 20000,
        "permutation_seed": 0,
        "strip_center": "mean",
        "include_matrix_figure": True,
        "figure_dpi": 180,
        "trial_effect_mode": "mean_controls",
        "trial_anchor_variant": "control_a",
    },
    "trial_level_tests": {
        "enabled": True,
        "metrics": [
            "delta_h_dist_tau3000_ks",
            "delta_h_dist_tau3000_energy",
            "absdiff_mean_speed",
            "absdiff_speed_std",
            "absdiff_spatial_spread",
        ],
        "effect_mode": "anchor",
        "anchor_variant": "control_a",
        "bootstrap_reps": 2000,
        "bootstrap_seed": 0,
        "ci_level": 0.95,
        "permutation_max_exact": 200000,
        "permutation_samples": 20000,
        "permutation_seed": 0,
        "zero_tolerance": 1e-12,
        "figure_dpi": 180,
    },
    "progress": {
        "enabled": True,
        "show_inner": True,
    },
    "reporting": {
        "representative_runs_per_condition": 2,
        "primary_embedding_metric": "embedding_cloud_chamfer_cosine",
        "primary_delta_h_metric": "delta_h_l2",
        "primary_scalar_metric": "absdiff_msc_scalar",
        "figure_dpi": 180,
    },
}


def _strip_private(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if not str(key).startswith("_")}


def load_analysis_config(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULT_CONFIG), OmegaConf.load(path))
    data = OmegaConf.to_container(cfg, resolve=True)
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    data["_repo_root"] = str(REPO_ROOT)
    return data


def _merge_pair_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = non_empty[0].copy()
    join_cols = [
        "run_a",
        "run_b",
        "condition_a",
        "condition_b",
        "pair_type",
        "pair_group_a",
        "pair_group_b",
        "same_pair_group",
    ]
    for frame in non_empty[1:]:
        merged = merged.merge(frame, on=join_cols, how="outer")
    return merged


def _save_resolved_config(cfg: dict[str, Any], output_dir: Path) -> None:
    resolved = OmegaConf.create(_strip_private(cfg))
    (output_dir / "resolved_analysis_config.yaml").write_text(OmegaConf.to_yaml(resolved, resolve=True))


def run_analysis(config_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = load_analysis_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    progress_cfg = dict(cfg.get("progress", {}))
    show_progress = bool(progress_cfg.get("enabled", True))
    output_dir = resolve_config_path(cfg["output"]["dir"], Path(cfg["_config_dir"]))
    if output_dir is None:
        raise ValueError("output.dir must be set.")
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "matrices")
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "run_observables")
    _save_resolved_config(cfg, output_dir)

    stage_total = 9
    with progress_bar(total=stage_total, desc="Offline analysis", enabled=show_progress, leave=True) as stage_bar:
        run_collection = build_run_collection(cfg)
        runs = run_collection.runs.copy()
        save_dataframe(output_dir / "tables" / "run_catalog.csv", runs)
        write_json(output_dir / "source_summary.json", run_collection.source_summary)
        if run_collection.metric_summary is not None:
            write_json(output_dir / "source_metric_summary.json", run_collection.metric_summary)
        stage_bar.set_postfix(step="loaded runs")
        stage_bar.update(1)

        matrices: dict[str, pd.DataFrame] = {}
        pair_frames = []
        figures: dict[str, str] = {}

        if bool(cfg.get("embeddings", {}).get("enabled", True)):
            emb_pairs, emb_matrices_raw = compute_embedding_pairwise(runs, load_embeddings, cfg)
            emb_matrices = {f"embedding_{name}": matrix for name, matrix in emb_matrices_raw.items()}
            matrices.update(emb_matrices)
            pair_frames.append(emb_pairs)
            for name, matrix in emb_matrices.items():
                save_matrix(output_dir / "matrices" / f"{name}.csv", matrix)
        stage_bar.set_postfix(step="embeddings")
        stage_bar.update(1)

        metric_cfg = derive_metric_config(cfg, run_collection)
        run_metrics = pd.DataFrame()
        per_run = {}
        if metric_cfg is not None and bool(cfg.get("trajectories", {}).get("enabled", True)):
            write_json(output_dir / "trajectory_metric_config.json", metric_cfg)
            run_metrics, per_run = compute_trajectory_observables(runs, load_lagrangian, cfg, metric_cfg)
            for _, row in run_metrics.iterrows():
                run_id = row["run_id"]
                data = per_run[run_id]
                safe_id = run_id.replace("/", "__")
                map_path = output_dir / "run_observables" / f"{safe_id}_delta_h_map.npz"
                payload = dict(
                    delta_h_map=data["delta_h_map"],
                    delta_h_best=data["delta_h_best"],
                    tau_frames=data["tau_frames"],
                    tau_steps=data["tau_steps"],
                    window_start_frames=data["window_start_frames"],
                    window_start_steps=data["window_start_steps"],
                    score_by_tau=data["score_by_tau"],
                    amp_by_tau=data["amp_by_tau"],
                    msc_by_tau=data["msc_by_tau"],
                )
                if "delta_h_fixed_tau" in data:
                    payload["delta_h_fixed_tau"] = data["delta_h_fixed_tau"]
                    payload["delta_h_fixed_tau_idx"] = np.asarray(data["delta_h_fixed_tau_idx"], dtype=np.int32)
                    payload["delta_h_fixed_tau_steps"] = np.asarray(data["delta_h_fixed_tau_steps"], dtype=np.int32)
                    payload["delta_h_fixed_tau_frames"] = np.asarray(data["delta_h_fixed_tau_frames"], dtype=np.int32)
                save_npz(
                    map_path,
                    **payload,
                )
                run_metrics.loc[run_metrics["run_id"] == run_id, "delta_h_map_path"] = str(map_path)
            save_dataframe(output_dir / "tables" / "run_trajectory_observables.csv", run_metrics)
            traj_pairs, traj_matrices = compute_trajectory_pairwise(runs, run_metrics, per_run, cfg)
            matrices.update(traj_matrices)
            pair_frames.append(traj_pairs)
            for name, matrix in traj_matrices.items():
                save_matrix(output_dir / "matrices" / f"{name}.csv", matrix)
        stage_bar.set_postfix(step="trajectories")
        stage_bar.update(1)

        pairwise = _merge_pair_frames(pair_frames)
        if not pairwise.empty:
            save_dataframe(output_dir / "tables" / "pairwise_distances.csv", pairwise)

        numeric_pair_cols = [
            column
            for column in pairwise.columns
            if column not in {
                "run_a",
                "run_b",
                "condition_a",
                "condition_b",
                "pair_type",
                "pair_group_a",
                "pair_group_b",
                "same_pair_group",
            }
        ]
        pair_summary = summarize_pairwise_values(pairwise, numeric_pair_cols)
        effect_sizes = compute_effect_sizes(pairwise, numeric_pair_cols)
        save_dataframe(output_dir / "tables" / "pairwise_summary.csv", pair_summary)
        save_dataframe(output_dir / "tables" / "effect_sizes.csv", effect_sizes)
        stage_bar.set_postfix(step="summaries")
        stage_bar.update(1)

        stats_cfg = dict(cfg.get("statistics", {}))
        permutation_rows = []
        with progress_bar(total=len(matrices), desc="Permutation tests", enabled=show_progress, leave=False) as perm_bar:
            for name, matrix in matrices.items():
                permutation_rows.append(
                    permutation_test_from_matrix(
                        matrix,
                        runs,
                        observable=name,
                        max_exact=int(stats_cfg.get("permutation_max_exact", 200000)),
                        n_samples=int(stats_cfg.get("permutation_samples", 20000)),
                        seed=int(stats_cfg.get("permutation_seed", 0)),
                    )
                )
                perm_bar.update(1)
        permutation_df = pd.DataFrame(permutation_rows)
        save_dataframe(output_dir / "tables" / "permutation_tests.csv", permutation_df)
        stage_bar.set_postfix(step="permutations")
        stage_bar.update(1)

        main_observable_report = generate_main_observable_report(runs, matrices, output_dir, cfg)
        stage_bar.set_postfix(step="main observable")
        stage_bar.update(1)

        trial_level_tests = generate_trial_level_metric_tests(runs, matrices, output_dir, cfg)
        stage_bar.set_postfix(step="trial tests")
        stage_bar.update(1)

        reporting_cfg = dict(cfg.get("reporting", {}))
        concordance_cols = [
            reporting_cfg.get("primary_embedding_metric"),
            reporting_cfg.get("primary_delta_h_metric"),
            reporting_cfg.get("primary_scalar_metric"),
        ]
        concordance_cols = [col for col in concordance_cols if col]
        concordance = compute_concordance(pairwise, concordance_cols)
        save_dataframe(output_dir / "tables" / "concordance.csv", concordance)
        stage_bar.set_postfix(step="concordance")
        stage_bar.update(1)

        dpi = int(reporting_cfg.get("figure_dpi", 180))
        frame_panel_path = output_dir / "figures" / "representative_frames.png"
        if plot_frame_panel(
            runs,
            frame_panel_path,
            n_per_condition=int(reporting_cfg.get("representative_runs_per_condition", 2)),
            dpi=dpi,
        ):
            figures["representative_frames"] = str(frame_panel_path)

        primary_embedding = str(reporting_cfg.get("primary_embedding_metric", "embedding_cloud_chamfer_cosine"))
        if primary_embedding in matrices:
            matrix_path = output_dir / "figures" / f"{primary_embedding}_matrix.png"
            plot_distance_matrix(matrices[primary_embedding], runs, matrix_path, title=f"{primary_embedding} distance matrix", dpi=dpi)
            figures["embedding_distance_matrix"] = str(matrix_path)
            strip_path = output_dir / "figures" / f"{primary_embedding}_strip.png"
            plot_pair_strip(pairwise, primary_embedding, strip_path, title=f"{primary_embedding} by pair class", dpi=dpi)
            figures["embedding_strip"] = str(strip_path)

        primary_delta_h = str(reporting_cfg.get("primary_delta_h_metric", "delta_h_l2"))
        if primary_delta_h in matrices:
            dh_matrix_path = output_dir / "figures" / f"{primary_delta_h}_matrix.png"
            plot_distance_matrix(matrices[primary_delta_h], runs, dh_matrix_path, title=f"{primary_delta_h} distance matrix", dpi=dpi)
            figures["delta_h_distance_matrix"] = str(dh_matrix_path)
            dh_strip_path = output_dir / "figures" / f"{primary_delta_h}_strip.png"
            plot_pair_strip(pairwise, primary_delta_h, dh_strip_path, title=f"{primary_delta_h} by pair class", dpi=dpi)
            figures["delta_h_strip"] = str(dh_strip_path)

        if per_run:
            delta_h_examples_path = output_dir / "figures" / "delta_h_examples.png"
            plot_delta_h_examples(
                per_run,
                runs,
                delta_h_examples_path,
                n_per_condition=int(reporting_cfg.get("representative_runs_per_condition", 2)),
                dpi=dpi,
            )
            figures["delta_h_examples"] = str(delta_h_examples_path)

        if primary_embedding in pairwise.columns and primary_delta_h in pairwise.columns:
            scatter_path = output_dir / "figures" / "embedding_vs_delta_h.png"
            plot_scatter(
                pairwise,
                primary_embedding,
                primary_delta_h,
                scatter_path,
                title="Embedding distance vs delta-h distance",
                dpi=dpi,
            )
            figures["embedding_vs_delta_h"] = str(scatter_path)

        if not run_metrics.empty:
            msc_cols = [
                "run_id",
                "condition",
                "msc_scalar",
                "score_scalar",
                "amp_scalar",
                "tau_best_steps",
                "mean_speed",
                "speed_std",
                "occupied_area_fraction",
                "spatial_spread",
            ]
            msc_table = run_metrics[msc_cols].sort_values(["condition", "run_id"]).reset_index(drop=True)
            save_dataframe(output_dir / "tables" / "msc_run_values.csv", msc_table)
            pair_cols = ["run_a", "run_b", "pair_type", "absdiff_msc_scalar"]
            if "absdiff_msc_scalar" in pairwise.columns:
                save_dataframe(output_dir / "tables" / "msc_pairwise_differences.csv", pairwise[pair_cols])
            table_path = output_dir / "figures" / "msc_run_table.png"
            plot_table(msc_table, table_path, title="MSC_t and coarse trajectory observables", dpi=dpi)
            figures["msc_table"] = str(table_path)
        stage_bar.set_postfix(step="figures")
        stage_bar.update(1)

    if main_observable_report:
        figures.update(main_observable_report["paths"]["figure_paths"])
    if trial_level_tests:
        figures.update(trial_level_tests["paths"]["figure_paths"])

    overview = {
        "n_runs": int(runs.shape[0]),
        "n_free_runs": int((runs["condition"] == "free").sum()),
        "n_wall_runs": int((runs["condition"] == "wall").sum()),
        "output_dir": str(output_dir),
        "primary_embedding_metric": primary_embedding,
        "primary_delta_h_metric": primary_delta_h,
        "primary_scalar_metric": str(reporting_cfg.get("primary_scalar_metric", "absdiff_msc_scalar")),
        "figure_paths": figures,
        "main_observable_distance_name": None if not main_observable_report else main_observable_report["summary"]["distance_name"],
        "main_observable_report": None if not main_observable_report else main_observable_report["text_summary"],
        "trial_level_report": None if not trial_level_tests else trial_level_tests.get("report_text"),
    }
    write_json(output_dir / "overview.json", overview)

    return {
        "config": cfg,
        "output_dir": str(output_dir),
        "runs": runs,
        "run_metrics": run_metrics,
        "pairwise": pairwise,
        "pair_summary": pair_summary,
        "effect_sizes": effect_sizes,
        "permutation_tests": permutation_df,
        "concordance": concordance,
        "matrices": matrices,
        "figures": figures,
        "main_observable": main_observable_report,
        "trial_level_tests": trial_level_tests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline analysis of Flow-Lenia history dependence runs.")
    parser.add_argument("config", help="Path to analysis YAML config.")
    args = parser.parse_args(argv)
    result = run_analysis(args.config)
    print(f"Saved offline analysis to {result['output_dir']}")
    print(f"Runs: {result['runs'].shape[0]}")
    if result.get("main_observable"):
        print(result["main_observable"]["text_summary"])
    if result.get("trial_level_tests", {}).get("report_text"):
        print(result["trial_level_tests"]["report_text"])
    if not result["effect_sizes"].empty:
        top = result["effect_sizes"].sort_values("observable").head(8)
        print(top.to_string(index=False))
    return 0
