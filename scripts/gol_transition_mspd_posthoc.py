#!/usr/bin/env python3
"""Posthoc C1/C2/C5 checks for transition-law MSPD cellular automata runs.

This script is intentionally downstream-only: it reads an existing
`gol_transition_mspd_experiment.py` optimization/rule-sweep output directory and
does not rerun the optimizer.  The checks mirror the paper-suite semantics used
for Flow-Lenia/PLife++ as closely as the discrete CA setting allows:

* C1: MSPD-optimized CA candidate(s) versus matched random controls.
* C2: Delta-H predicts sensitivity to small future perturbations.
* C5: history-dependence/frustration assay with control_a, control_b, and
  walls variants, matching the Flow-Lenia/PLife++ contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gol_transition_mspd_experiment import (  # noqa: E402
    CONWAY_LIFE_RULE,
    ExperimentConfig,
    compute_transition_mspd_batch_auto,
    compute_transition_mspd_auto,
    lifelike_rule_label,
    simulate_lifelike_rule_batch,
)
from paper_suite_common import ensure_dir, sign_test_greater, write_csv, write_json  # noqa: E402


DEFAULT_POSTHOC_CONFIG: dict[str, Any] = {
    "input_dir": None,
    "results_root": "analysis/results/gol_transition_mspd",
    "output_dir": None,
    "task": "all",
    "backend": None,
    "seed": 0,
    "optimized_top_k": 1,
    "random_controls": 32,
    "c1_eval_mode": "fresh_holdout",
    "c1_holdout_n_initial_boards": 32,
    "c1_holdout_density_mode": "source",
    "c1_holdout_density": None,
    "c1_holdout_density_range": None,
    "c1_eval_batch_size": None,
    "c2_branch_windows": 24,
    "c2_branches_per_window": 4,
    "c2_horizon": 64,
    "c2_perturb_fraction": 0.01,
    "c5_random_controls": None,
    "c5_n_trials": 4,
    "c5_warmup_steps": None,
    "c5_late_window_start_steps": None,
    "c5_late_window_end_steps": None,
    "c5_wall_grid_split": 2,
}


@dataclass(frozen=True)
class LoadedCAResult:
    input_dir: Path
    mode: str
    config: ExperimentConfig
    best_rule_id: int
    best_rule_label: str
    best_initial_board: np.ndarray
    initial_boards: np.ndarray
    best_trajectory: np.ndarray
    best_delta_h: np.ndarray
    initial_probabilities: np.ndarray
    initial_density_range: tuple[float, float] | None
    per_rule_scores: dict[int, dict[int, float]]
    rule_labels: dict[int, str]
    random_initial_boards: dict[int, np.ndarray]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml_like(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        from omegaconf import OmegaConf

        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return dict(payload or {})
    except Exception:
        try:
            import yaml

            payload = yaml.safe_load(path.read_text())
            return dict(payload or {})
        except Exception as exc:
            raise RuntimeError(f"Could not parse config as YAML/OmegaConf: {path}") from exc


def _flatten_posthoc_config(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in {"c1", "c2", "c5"}:
            flat[key] = value

    c1 = payload.get("c1") or {}
    if isinstance(c1, dict):
        mapping = {
            "optimized_top_k": "optimized_top_k",
            "random_controls": "random_controls",
            "eval_mode": "c1_eval_mode",
            "holdout_n_initial_boards": "c1_holdout_n_initial_boards",
            "holdout_density_mode": "c1_holdout_density_mode",
            "holdout_density": "c1_holdout_density",
            "holdout_density_range": "c1_holdout_density_range",
            "eval_batch_size": "c1_eval_batch_size",
        }
        for src, dst in mapping.items():
            if src in c1:
                flat[dst] = c1[src]

    c2 = payload.get("c2") or {}
    if isinstance(c2, dict):
        mapping = {
            "branch_windows": "c2_branch_windows",
            "branches_per_window": "c2_branches_per_window",
            "horizon": "c2_horizon",
            "perturb_fraction": "c2_perturb_fraction",
        }
        for src, dst in mapping.items():
            if src in c2:
                flat[dst] = c2[src]

    c5 = payload.get("c5") or {}
    if isinstance(c5, dict):
        mapping = {
            "random_controls": "c5_random_controls",
            "n_trials": "c5_n_trials",
            "warmup_steps": "c5_warmup_steps",
            "late_window_start_steps": "c5_late_window_start_steps",
            "late_window_end_steps": "c5_late_window_end_steps",
            "wall_grid_split": "c5_wall_grid_split",
        }
        for src, dst in mapping.items():
            if src in c5:
                flat[dst] = c5[src]
    return flat


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_POSTHOC_CONFIG)
    if args.config:
        config_path = _resolve_path(args.config)
        config.update(_flatten_posthoc_config(_load_yaml_like(config_path)))
    for key in DEFAULT_POSTHOC_CONFIG:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def _experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    valid = set(ExperimentConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in payload.items() if key in valid}
    return ExperimentConfig(**kwargs)


def _load_config(input_dir: Path) -> tuple[str, ExperimentConfig]:
    if not input_dir.exists():
        raise FileNotFoundError(f"CA MSPD input_dir does not exist: {input_dir}")
    rule_cfg = input_dir / "rule_sweep_config.json"
    ga_cfg = input_dir / "config.json"
    if rule_cfg.exists():
        payload = json.loads(rule_cfg.read_text())
        return "rule_sweep", _experiment_config_from_dict(payload.get("base_config", {}))
    if ga_cfg.exists():
        return "ga_initial_board", _experiment_config_from_dict(json.loads(ga_cfg.read_text()))
    raise FileNotFoundError(
        f"No rule_sweep_config.json or config.json in {input_dir}. "
        "Point --input-dir at a completed GoL/CA MSPD result directory, or set input_dir: null "
        "in experiments/gol_transition_mspd/ca_posthoc_claims.yaml to auto-pick the latest completed run."
    )


def _rule_sweep_density_range(input_dir: Path) -> tuple[float, float] | None:
    path = input_dir / "rule_sweep_config.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    raw = payload.get("initial_density_range", None)
    if raw is None:
        return None
    try:
        low, high = list(raw)[:2]
        return float(low), float(high)
    except Exception:
        return None


def _completed_input_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.glob("*"):
        if not path.is_dir():
            continue
        has_result = (path / "best_rule_result.npz").exists() or (path / "best_result.npz").exists()
        has_config = (path / "rule_sweep_config.json").exists() or (path / "config.json").exists()
        if has_result and has_config:
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_completed_input_dir(root: Path) -> Path:
    candidates = _completed_input_dirs(root)
    if not candidates:
        partials = sorted(
            [p for p in root.glob("*") if p.is_dir() and ((p / "rule_sweep_config.json").exists() or (p / "config.json").exists())],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]
        partial_msg = "\n".join(f"  partial: {p}" for p in partials)
        raise FileNotFoundError(
            f"No completed GoL MSPD result directories found under {root}. "
            "Completed means best_rule_result.npz or best_result.npz plus rule_sweep_config.json/config.json.\n"
            + (partial_msg if partial_msg else "No partial config directories found either.")
        )
    return candidates[0]


def _load_rule_sweep_scores(per_init_path: Path) -> tuple[dict[int, dict[int, float]], dict[int, str]]:
    per_rule_scores: dict[int, dict[int, float]] = {}
    rule_labels: dict[int, str] = {}
    with per_init_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rule_id = _safe_int(row.get("rule_id"))
            init_id = _safe_int(row.get("init_id"))
            score = _safe_float(row.get("mspd_score"))
            if not np.isfinite(score):
                continue
            per_rule_scores.setdefault(rule_id, {})[init_id] = score
            label = row.get("rule_label") or lifelike_rule_label(rule_id)
            rule_labels[rule_id] = label
    if not per_rule_scores:
        raise ValueError(f"No usable rows in {per_init_path}")
    return per_rule_scores, rule_labels


def _load_ga_scores(
    input_dir: Path,
    best_score: float,
    config: ExperimentConfig,
) -> tuple[dict[int, dict[int, float]], dict[int, str], dict[int, np.ndarray]]:
    per_rule_scores: dict[int, dict[int, float]] = {CONWAY_LIFE_RULE: {0: best_score}}
    rule_labels: dict[int, str] = {CONWAY_LIFE_RULE: lifelike_rule_label(CONWAY_LIFE_RULE)}
    random_initial_boards: dict[int, np.ndarray] = {}
    controls = input_dir / "random_control_scores.csv"
    if controls.exists():
        with controls.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                control_id = _safe_int(row.get("control_id"), default=-1)
                score = _safe_float(row.get("fitness"))
                if control_id < 0 or not np.isfinite(score):
                    continue
                # Negative synthetic IDs keep GA random boards distinct from the
                # optimized Conway initial board while preserving matched syntax.
                synthetic_id = -1 - control_id
                per_rule_scores[synthetic_id] = {0: score}
                rule_labels[synthetic_id] = f"random_board_{control_id:05d}"
                seed = _safe_int(row.get("seed"), default=config.random_seed + 500000003 + control_id)
                rng = np.random.default_rng(seed)
                random_initial_boards[synthetic_id] = (
                    rng.random((config.L, config.L)) < float(config.initial_density)
                ).astype(np.uint8)
    return per_rule_scores, rule_labels, random_initial_boards


def load_ca_result(input_dir: Path) -> LoadedCAResult:
    mode, config = _load_config(input_dir)
    if (input_dir / "best_rule_result.npz").exists():
        with np.load(input_dir / "best_rule_result.npz", allow_pickle=False) as data:
            best_rule_id = int(np.asarray(data["rule_id"]).item())
            best_rule_label = str(np.asarray(data["rule_label"]).item())
            best_initial_board = np.asarray(data["initial_board"], dtype=np.uint8)
            initial_boards = np.asarray(data["initial_boards"], dtype=np.uint8)
            initial_probabilities = np.asarray(
                data["initial_board_probabilities"]
                if "initial_board_probabilities" in data
                else np.full((initial_boards.shape[0],), config.initial_density),
                dtype=np.float64,
            )
            best_trajectory = np.asarray(data["trajectory"], dtype=np.uint8)
            best_delta_h = np.asarray(data["delta_h"] if "delta_h" in data else data["DeltaH"], dtype=np.float64)
        per_rule_scores, rule_labels = _load_rule_sweep_scores(input_dir / "rule_sweep_per_init_scores.csv")
        rule_labels[best_rule_id] = best_rule_label
        return LoadedCAResult(
            input_dir=input_dir,
            mode="rule_sweep",
            config=config,
            best_rule_id=best_rule_id,
            best_rule_label=best_rule_label,
            best_initial_board=best_initial_board,
            initial_boards=initial_boards,
            best_trajectory=best_trajectory,
            best_delta_h=best_delta_h,
            initial_probabilities=initial_probabilities,
            initial_density_range=_rule_sweep_density_range(input_dir),
            per_rule_scores=per_rule_scores,
            rule_labels=rule_labels,
            random_initial_boards={},
        )

    if (input_dir / "best_result.npz").exists():
        with np.load(input_dir / "best_result.npz", allow_pickle=False) as data:
            best_initial_board = np.asarray(data["initial_board"], dtype=np.uint8)
            best_trajectory = np.asarray(data["trajectory"], dtype=np.uint8)
            best_delta_h = np.asarray(data["delta_h"] if "delta_h" in data else data["DeltaH"], dtype=np.float64)
            best_score = _npz_scalar(data, ["score", "mspd_score", "best_score"], default=float(np.nan))
        per_rule_scores, rule_labels, random_initial_boards = _load_ga_scores(input_dir, best_score, config)
        return LoadedCAResult(
            input_dir=input_dir,
            mode=mode,
            config=config,
            best_rule_id=CONWAY_LIFE_RULE,
            best_rule_label=lifelike_rule_label(CONWAY_LIFE_RULE),
            best_initial_board=best_initial_board,
            initial_boards=best_initial_board[None, ...],
            best_trajectory=best_trajectory,
            best_delta_h=best_delta_h,
            initial_probabilities=np.asarray([config.initial_density], dtype=np.float64),
            initial_density_range=None,
            per_rule_scores=per_rule_scores,
            rule_labels=rule_labels,
            random_initial_boards=random_initial_boards,
        )

    raise FileNotFoundError(f"No best_rule_result.npz or best_result.npz in {input_dir}")


def _npz_scalar(data: Any, keys: Sequence[str], default: float = float("nan")) -> float:
    for key in keys:
        if key in data:
            return float(np.asarray(data[key]).item())
    return default


def _rule_mean(scores_by_init: dict[int, float]) -> float:
    values = np.asarray(list(scores_by_init.values()), dtype=np.float64)
    return float(np.mean(values)) if values.size else float("nan")


def _rank_rules(per_rule_scores: dict[int, dict[int, float]]) -> list[int]:
    return sorted(
        per_rule_scores.keys(),
        key=lambda rule_id: (_rule_mean(per_rule_scores[rule_id]), rule_id),
        reverse=True,
    )


def _optimized_rule_ids(loaded: LoadedCAResult, optimized_top_k: int) -> list[int]:
    if loaded.mode == "ga_initial_board":
        return [CONWAY_LIFE_RULE]
    return _rank_rules(loaded.per_rule_scores)[: max(1, optimized_top_k)]


def _sample_random_rule_ids(
    all_rule_ids: Sequence[int],
    excluded: set[int],
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = np.asarray([rid for rid in all_rule_ids if rid not in excluded], dtype=np.int64)
    if candidates.size == 0:
        return []
    if candidates.size <= n:
        return [int(x) for x in candidates.tolist()]
    picked = rng.choice(candidates, size=n, replace=False)
    return [int(x) for x in picked.tolist()]


def _density_range_from_raw(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    try:
        low, high = list(raw)[:2]
        low_f = float(low)
        high_f = float(high)
        if not (0.0 <= low_f <= high_f <= 1.0):
            raise ValueError
        return low_f, high_f
    except Exception as exc:
        raise ValueError(f"Invalid holdout density range {raw!r}; expected [low, high] in [0,1].") from exc


def _make_c1_holdout_boards(
    loaded: LoadedCAResult,
    *,
    n_initial_boards: int,
    density_mode: str,
    density: float | None,
    density_range: tuple[float, float] | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if n_initial_boards <= 0:
        raise ValueError("c1.holdout_n_initial_boards must be positive.")
    rng = np.random.default_rng(int(seed))
    mode = str(density_mode or "source").strip().lower()
    if mode == "source":
        if density_range is not None:
            probs = rng.uniform(density_range[0], density_range[1], size=n_initial_boards)
        elif loaded.initial_density_range is not None:
            low, high = loaded.initial_density_range
            probs = rng.uniform(low, high, size=n_initial_boards)
        else:
            probs = np.full((n_initial_boards,), float(loaded.config.initial_density), dtype=np.float64)
    elif mode in {"fixed", "constant"}:
        p = float(loaded.config.initial_density if density is None else density)
        probs = np.full((n_initial_boards,), p, dtype=np.float64)
    elif mode in {"uniform", "uniform_range"}:
        if density_range is None:
            raise ValueError("c1.holdout_density_mode=uniform requires c1.holdout_density_range.")
        probs = rng.uniform(density_range[0], density_range[1], size=n_initial_boards)
    else:
        raise ValueError(f"Unknown c1.holdout_density_mode={density_mode!r}; use source, fixed, or uniform.")
    boards = (rng.random((n_initial_boards, loaded.config.L, loaded.config.L)) < probs[:, None, None]).astype(np.uint8)
    alive = boards.reshape(n_initial_boards, -1).mean(axis=1).astype(np.float64)
    seeds = [int(seed + 10_000_019 + 9176 * i) for i in range(n_initial_boards)]
    return boards, probs.astype(np.float64), alive, seeds


def _run_c1_fresh_holdout(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    optimized_top_k: int,
    random_controls: int,
    holdout_n_initial_boards: int,
    holdout_density_mode: str,
    holdout_density: float | None,
    holdout_density_range: tuple[float, float] | None,
    eval_batch_size: int | None,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir / "c1")
    rng = np.random.default_rng(seed)
    ranked = _rank_rules(loaded.per_rule_scores)
    optimized_rules = _optimized_rule_ids(loaded, optimized_top_k)
    random_rules = _sample_random_rule_ids(ranked, set(optimized_rules), random_controls, rng)
    selected_rules = optimized_rules + random_rules

    if loaded.mode != "rule_sweep":
        raise ValueError(
            "Strict C1 fresh holdout is defined for rule_sweep CA runs. "
            "For GA initial-board runs, use c1.eval_mode: saved_score or run a rule-sweep artifact."
        )

    holdout_boards, holdout_p, holdout_alive, holdout_board_seeds = _make_c1_holdout_boards(
        loaded,
        n_initial_boards=holdout_n_initial_boards,
        density_mode=holdout_density_mode,
        density=holdout_density,
        density_range=holdout_density_range,
        seed=seed + 1_000_003,
    )
    metric_cfg = replace(
        loaded.config,
        backend=backend,
        eval_batch_size=int(eval_batch_size or loaded.config.eval_batch_size),
    )

    score_rows: list[dict[str, Any]] = []
    eval_pairs = [(rule_id, init_id) for rule_id in selected_rules for init_id in range(holdout_boards.shape[0])]
    batch_size = max(1, int(eval_batch_size or metric_cfg.eval_batch_size))
    _log(
        "C1 fresh holdout evaluating "
        f"rules={len(selected_rules)} holdout_boards={holdout_boards.shape[0]} "
        f"total_trajectories={len(eval_pairs)} batch_size={batch_size}"
    )
    for start in range(0, len(eval_pairs), batch_size):
        stop = min(start + batch_size, len(eval_pairs))
        chunk = eval_pairs[start:stop]
        boards = np.stack([holdout_boards[init_id] for _rule_id, init_id in chunk], axis=0)
        rules = np.asarray([rule_id for rule_id, _init_id in chunk], dtype=np.uint32)
        trajectories = simulate_lifelike_rule_batch(boards, rules, metric_cfg.T, backend=backend)
        metric_seeds = [
            int((seed + 20_000_033 + 1009 * (abs(int(rule_id)) + 1) + 9176 * int(init_id)) % (2**31 - 1))
            for rule_id, init_id in chunk
        ]
        results = compute_transition_mspd_batch_auto(trajectories, metric_cfg, metric_seeds)
        for (rule_id, init_id), result, metric_seed in zip(chunk, results, metric_seeds):
            group = "optimized" if rule_id in set(optimized_rules) else "random"
            score_rows.append(
                {
                    "claim": "C1",
                    "eval_mode": "fresh_holdout_initial_boards",
                    "group": group,
                    "rule_id": int(rule_id),
                    "rule_label": loaded.rule_labels.get(int(rule_id), lifelike_rule_label(int(rule_id))),
                    "holdout_init_id": int(init_id),
                    "holdout_board_seed": int(holdout_board_seeds[init_id]),
                    "metric_seed": int(metric_seed),
                    "initial_p": float(holdout_p[init_id]),
                    "initial_alive_fraction": float(holdout_alive[init_id]),
                    "mspd_score": float(result.fitness_score),
                    "raw_mspd_score": float(result.mspd_score),
                    "delta_h_nonzero_frac": float(result.delta_h_nonzero_frac),
                    "passes_delta_h_filter": int(result.passes_delta_h_filter),
                }
            )

    rule_level_rows: list[dict[str, Any]] = []
    for group, rule_ids in [("optimized", optimized_rules), ("random", random_rules)]:
        for rule_id in rule_ids:
            rows = [row for row in score_rows if int(row["rule_id"]) == int(rule_id)]
            values = np.asarray([row["mspd_score"] for row in rows], dtype=np.float64)
            raw_values = np.asarray([row["raw_mspd_score"] for row in rows], dtype=np.float64)
            rule_level_rows.append(
                {
                    "group": group,
                    "rule_id": int(rule_id),
                    "rule_label": loaded.rule_labels.get(int(rule_id), lifelike_rule_label(int(rule_id))),
                    "training_mean_mspd": _rule_mean(loaded.per_rule_scores[int(rule_id)]),
                    "holdout_mean_mspd": float(np.mean(values)) if values.size else float("nan"),
                    "holdout_median_mspd": float(np.median(values)) if values.size else float("nan"),
                    "holdout_std_mspd": float(np.std(values, ddof=0)) if values.size else float("nan"),
                    "holdout_mean_raw_mspd": float(np.mean(raw_values)) if raw_values.size else float("nan"),
                    "n_holdout_initial_boards": int(values.size),
                }
            )

    random_by_init: dict[int, list[float]] = {}
    for row in score_rows:
        if row["group"] == "random":
            random_by_init.setdefault(int(row["holdout_init_id"]), []).append(float(row["mspd_score"]))
    contrast_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for row in score_rows:
        if row["group"] != "optimized":
            continue
        init_id = int(row["holdout_init_id"])
        baseline = np.asarray(random_by_init.get(init_id, []), dtype=np.float64)
        if baseline.size == 0:
            continue
        delta = float(row["mspd_score"]) - float(np.median(baseline))
        deltas.append(delta)
        contrast_rows.append(
            {
                "claim": "C1",
                "eval_mode": "fresh_holdout_initial_boards",
                "opt_rule_id": int(row["rule_id"]),
                "opt_rule_label": row["rule_label"],
                "holdout_init_id": init_id,
                "optimized_mspd": float(row["mspd_score"]),
                "random_median_mspd": float(np.median(baseline)),
                "random_mean_mspd": float(np.mean(baseline)),
                "delta_vs_random_median": delta,
                "n_random_controls_for_init": int(baseline.size),
            }
        )

    opt_values = np.asarray([row["holdout_mean_mspd"] for row in rule_level_rows if row["group"] == "optimized"], dtype=np.float64)
    random_values = np.asarray([row["holdout_mean_mspd"] for row in rule_level_rows if row["group"] == "random"], dtype=np.float64)
    summary = {
        "claim": "C1",
        "mode": loaded.mode,
        "input_dir": str(loaded.input_dir),
        "c1_eval_mode": "fresh_holdout_initial_boards",
        "optimized_top_k": int(len(optimized_rules)),
        "random_controls": int(len(random_rules)),
        "optimized_rule_ids": optimized_rules,
        "random_rule_ids": random_rules,
        "holdout_n_initial_boards": int(holdout_boards.shape[0]),
        "holdout_density_mode": str(holdout_density_mode),
        "holdout_density_range": None if holdout_density_range is None else list(holdout_density_range),
        "holdout_initial_p_min": float(np.min(holdout_p)),
        "holdout_initial_p_max": float(np.max(holdout_p)),
        "optimized_holdout_mean_mspd": float(np.mean(opt_values)) if opt_values.size else float("nan"),
        "random_holdout_mean_mspd": float(np.mean(random_values)) if random_values.size else float("nan"),
        "random_holdout_median_mspd": float(np.median(random_values)) if random_values.size else float("nan"),
        "delta_vs_random_median_sign_test": sign_test_greater(deltas),
    }

    write_csv(out / "c1_checkpoint_scores.csv", score_rows)
    write_csv(out / "c1_rule_level_scores.csv", rule_level_rows)
    write_csv(out / "c1_group_contrasts.csv", contrast_rows)
    write_json(out / "c1_summary.json", summary)
    np.savez_compressed(
        out / "c1_holdout_initial_boards.npz",
        boards=holdout_boards.astype(np.uint8),
        initial_p=holdout_p.astype(np.float64),
        initial_alive_fraction=holdout_alive.astype(np.float64),
        board_seeds=np.asarray(holdout_board_seeds, dtype=np.int64),
        optimized_rule_ids=np.asarray(optimized_rules, dtype=np.int64),
        random_rule_ids=np.asarray(random_rules, dtype=np.int64),
    )
    _plot_c1(out, rule_level_rows, contrast_rows)
    return summary


def _run_c1_saved_score(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    optimized_top_k: int,
    random_controls: int,
    seed: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir / "c1")
    rng = np.random.default_rng(seed)
    ranked = _rank_rules(loaded.per_rule_scores)
    optimized_rules = _optimized_rule_ids(loaded, optimized_top_k)
    random_rules = _sample_random_rule_ids(ranked, set(optimized_rules), random_controls, rng)

    score_rows: list[dict[str, Any]] = []
    rule_level_rows: list[dict[str, Any]] = []
    for group, rule_ids in [("optimized", optimized_rules), ("random", random_rules)]:
        for rule_id in rule_ids:
            scores_by_init = loaded.per_rule_scores[rule_id]
            values = np.asarray(list(scores_by_init.values()), dtype=np.float64)
            rule_level_rows.append(
                {
                    "group": group,
                    "rule_id": rule_id,
                    "rule_label": loaded.rule_labels.get(rule_id, lifelike_rule_label(rule_id) if rule_id >= 0 else str(rule_id)),
                    "mean_mspd": float(np.mean(values)) if values.size else float("nan"),
                    "median_mspd": float(np.median(values)) if values.size else float("nan"),
                    "std_mspd": float(np.std(values, ddof=0)) if values.size else float("nan"),
                    "n_initial_boards": int(values.size),
                }
            )
            for init_id, score in sorted(scores_by_init.items()):
                score_rows.append(
                    {
                        "claim": "C1",
                        "group": group,
                        "rule_id": rule_id,
                        "rule_label": loaded.rule_labels.get(rule_id, lifelike_rule_label(rule_id) if rule_id >= 0 else str(rule_id)),
                        "init_id": init_id,
                        "mspd_score": float(score),
                    }
                )

    deltas: list[float] = []
    contrast_rows: list[dict[str, Any]] = []
    random_by_init: dict[int, list[float]] = {}
    for rule_id in random_rules:
        for init_id, score in loaded.per_rule_scores[rule_id].items():
            random_by_init.setdefault(init_id, []).append(float(score))

    for opt_rule_id in optimized_rules:
        for init_id, opt_score in loaded.per_rule_scores[opt_rule_id].items():
            baseline = np.asarray(random_by_init.get(init_id, []), dtype=np.float64)
            if baseline.size == 0:
                continue
            delta = float(opt_score) - float(np.median(baseline))
            deltas.append(delta)
            contrast_rows.append(
                {
                    "claim": "C1",
                    "opt_rule_id": opt_rule_id,
                    "opt_rule_label": loaded.rule_labels.get(opt_rule_id, lifelike_rule_label(opt_rule_id)),
                    "init_id": init_id,
                    "optimized_mspd": float(opt_score),
                    "random_median_mspd": float(np.median(baseline)),
                    "random_mean_mspd": float(np.mean(baseline)),
                    "delta_vs_random_median": delta,
                    "n_random_controls_for_init": int(baseline.size),
                }
            )

    opt_values = np.asarray([row["mean_mspd"] for row in rule_level_rows if row["group"] == "optimized"], dtype=np.float64)
    random_values = np.asarray([row["mean_mspd"] for row in rule_level_rows if row["group"] == "random"], dtype=np.float64)
    stat = sign_test_greater(deltas)
    summary = {
        "claim": "C1",
        "mode": loaded.mode,
        "input_dir": str(loaded.input_dir),
        "optimized_top_k": int(len(optimized_rules)),
        "random_controls": int(len(random_rules)),
        "optimized_rule_ids": optimized_rules,
        "random_rule_ids": random_rules,
        "optimized_mean_mspd": float(np.mean(opt_values)) if opt_values.size else float("nan"),
        "random_mean_mspd": float(np.mean(random_values)) if random_values.size else float("nan"),
        "random_median_mspd": float(np.median(random_values)) if random_values.size else float("nan"),
        "delta_vs_random_median_sign_test": stat,
        "c1_eval_mode": "matched_initial_boards_from_saved_ca_scores",
        "selection_holdout_note": "No tau-selection holdout is applied here; the CA transition-law MSPD score has no tau grid. For rule-sweep runs this is still not an independent new-initial-state holdout.",
    }

    write_csv(out / "c1_checkpoint_scores.csv", score_rows)
    write_csv(out / "c1_rule_level_scores.csv", rule_level_rows)
    write_csv(out / "c1_group_contrasts.csv", contrast_rows)
    write_json(out / "c1_summary.json", summary)
    _plot_c1(out, rule_level_rows, contrast_rows)
    return summary


def run_c1(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    optimized_top_k: int,
    random_controls: int,
    eval_mode: str,
    holdout_n_initial_boards: int,
    holdout_density_mode: str,
    holdout_density: float | None,
    holdout_density_range: tuple[float, float] | None,
    eval_batch_size: int | None,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    mode = str(eval_mode or "fresh_holdout").strip().lower()
    if mode in {"fresh", "fresh_holdout", "fresh_holdout_initial_boards", "holdout"}:
        return _run_c1_fresh_holdout(
            loaded,
            output_dir,
            optimized_top_k=optimized_top_k,
            random_controls=random_controls,
            holdout_n_initial_boards=holdout_n_initial_boards,
            holdout_density_mode=holdout_density_mode,
            holdout_density=holdout_density,
            holdout_density_range=holdout_density_range,
            eval_batch_size=eval_batch_size,
            backend=backend,
            seed=seed,
        )
    if mode in {"saved", "saved_score", "matched_saved_scores", "legacy"}:
        return _run_c1_saved_score(
            loaded,
            output_dir,
            optimized_top_k=optimized_top_k,
            random_controls=random_controls,
            seed=seed,
        )
    raise ValueError(f"Unknown c1.eval_mode={eval_mode!r}; use fresh_holdout or saved_score.")


def _window_branch_times(config: ExperimentConfig, delta_h: np.ndarray, trajectory_len: int, horizon: int) -> np.ndarray:
    centers = (
        int(config.burn_in)
        + np.arange(delta_h.size, dtype=np.int64) * int(config.window_step)
        + int(config.window_size) // 2
    )
    max_start = int(trajectory_len) - 1 - int(horizon)
    return centers[centers <= max_start]


def _quantile_window_indices(delta_h: np.ndarray, starts: np.ndarray, n_windows: int) -> np.ndarray:
    valid_n = int(min(delta_h.size, starts.size))
    if valid_n == 0:
        return np.zeros((0,), dtype=np.int64)
    valid_idx = np.arange(valid_n, dtype=np.int64)
    order = valid_idx[np.argsort(delta_h[valid_idx])]
    if order.size <= n_windows:
        return order
    positions = np.linspace(0, order.size - 1, n_windows)
    picked = np.unique(np.rint(positions).astype(np.int64))
    return order[picked]


def _small_bit_flip(state: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(state, dtype=np.uint8).copy()
    flat = out.reshape(-1)
    n_flip = max(1, int(round(flat.size * float(fraction))))
    n_flip = min(n_flip, flat.size)
    idx = rng.choice(flat.size, size=n_flip, replace=False)
    flat[idx] ^= np.uint8(1)
    return out


def _simulate_branches(
    initial_boards: list[np.ndarray],
    rule_id: int,
    horizon: int,
    backend: str,
) -> np.ndarray:
    if not initial_boards:
        return np.zeros((0, horizon + 1, 0, 0), dtype=np.uint8)
    boards = np.stack(initial_boards, axis=0).astype(np.uint8)
    rules = np.full((boards.shape[0],), int(rule_id), dtype=np.uint32)
    return simulate_lifelike_rule_batch(boards, rules, int(horizon), backend=backend)


def _bootstrap_mean_ci(values: list[float] | np.ndarray, *, n_boot: int = 2000, seed: int = 8123) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    stats = np.mean(arr[idx], axis=1)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def run_c2(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    n_windows: int,
    branches_per_window: int,
    horizon: int,
    perturb_fraction: float,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir / "c2")
    rng = np.random.default_rng(seed)
    branch_times = _window_branch_times(loaded.config, loaded.best_delta_h, loaded.best_trajectory.shape[0], horizon)
    picked = _quantile_window_indices(loaded.best_delta_h, branch_times, n_windows)
    if picked.size < 2:
        raise ValueError("C2 requires at least two valid Delta-H windows before trajectory end.")
    if int(branches_per_window) < 2:
        raise ValueError("C2 requires branches_per_window >= 2 for pairwise branch divergence.")

    branch_initials: list[np.ndarray] = []
    branch_meta: list[dict[str, Any]] = []
    for window_idx in picked:
        window_start = int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step))
        window_center = int(window_start + int(loaded.config.window_size) // 2)
        t0 = window_center
        base_state = loaded.best_trajectory[t0]
        for rep in range(branches_per_window):
            branch_initials.append(_small_bit_flip(base_state, perturb_fraction, rng))
            branch_meta.append(
                {
                    "window_idx": int(window_idx),
                    "t0": t0,
                    "window_start_t": window_start,
                    "window_center_t": window_center,
                    "branch_rep": rep,
                    "delta_h": float(loaded.best_delta_h[int(window_idx)]),
                }
            )

    branches = _simulate_branches(branch_initials, loaded.best_rule_id, horizon, backend)
    branch_detail_rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(branch_meta):
        t0 = int(meta["t0"])
        base_future = loaded.best_trajectory[t0 : t0 + horizon + 1]
        hamming_t = np.mean(branches[idx, 1:] != base_future[1:], axis=(1, 2))
        branch_detail_rows.append(
            {
                "claim": "C2",
                "rule_id": loaded.best_rule_id,
                "rule_label": loaded.best_rule_label,
                **meta,
                "perturbation_kind": "small_bit_flip",
                "perturb_fraction": float(perturb_fraction),
                "horizon_steps": int(horizon),
                "mean_future_hamming": float(np.mean(hamming_t)),
                "final_future_hamming": float(hamming_t[-1]),
                "max_future_hamming": float(np.max(hamming_t)),
            }
        )

    window_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for window_idx in picked:
        branch_indices = [
            idx for idx, meta in enumerate(branch_meta)
            if int(meta["window_idx"]) == int(window_idx)
        ]
        original_vals = np.asarray(
            [row["mean_future_hamming"] for row in branch_detail_rows if int(row["window_idx"]) == int(window_idx)],
            dtype=np.float64,
        )
        pair_vals = []
        for local_i, idx_i in enumerate(branch_indices):
            for idx_j in branch_indices[local_i + 1 :]:
                pair_hamming = float(np.mean(branches[idx_i, 1:] != branches[idx_j, 1:]))
                pair_vals.append(pair_hamming)
                pair_rows.append(
                    {
                        "claim": "C2",
                        "rule_id": loaded.best_rule_id,
                        "rule_label": loaded.best_rule_label,
                        "window_idx": int(window_idx),
                        "t0": int(branch_meta[idx_i]["t0"]),
                        "window_start_t": int(branch_meta[idx_i]["window_start_t"]),
                        "window_center_t": int(branch_meta[idx_i]["window_center_t"]),
                        "delta_h": float(loaded.best_delta_h[int(window_idx)]),
                        "branch_rep_i": int(branch_meta[idx_i]["branch_rep"]),
                        "branch_rep_j": int(branch_meta[idx_j]["branch_rep"]),
                        "pairwise_future_hamming": pair_hamming,
                    }
                )
        pair_arr = np.asarray(pair_vals, dtype=np.float64)
        ci_low, ci_high = _bootstrap_mean_ci(pair_arr)
        window_rows.append(
            {
                "claim": "C2",
                "rule_id": loaded.best_rule_id,
                "rule_label": loaded.best_rule_label,
                "window_idx": int(window_idx),
                "t0": int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step) + int(loaded.config.window_size) // 2),
                "window_start_t": int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step)),
                "window_center_t": int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step) + int(loaded.config.window_size) // 2),
                "delta_h": float(loaded.best_delta_h[int(window_idx)]),
                "branching_score": float(np.mean(pair_arr)) if pair_arr.size else float("nan"),
                "branching_score_ci_low": ci_low,
                "branching_score_ci_high": ci_high,
                "branching_score_pair_median": float(np.nanmedian(pair_arr)) if pair_arr.size else float("nan"),
                "branching_score_pair_std": float(np.nanstd(pair_arr, ddof=1)) if pair_arr.size >= 2 else float("nan"),
                "branching_metric": "pairwise_future_hamming",
                "mean_branch_vs_original_hamming": float(np.mean(original_vals)) if original_vals.size else float("nan"),
                "std_branch_vs_original_hamming": float(np.std(original_vals, ddof=0)) if original_vals.size else float("nan"),
                "n_branches": int(len(branch_indices)),
                "n_branch_pairs": int(pair_arr.size),
            }
        )

    x = np.asarray([row["delta_h"] for row in window_rows], dtype=np.float64)
    y = np.asarray([row["branching_score"] for row in window_rows], dtype=np.float64)
    corr = _correlation_summary(x, y)
    high_low = _high_low_delta_summary(window_rows)
    summary = {
        "claim": "C2",
        "mode": loaded.mode,
        "input_dir": str(loaded.input_dir),
        "rule_id": loaded.best_rule_id,
        "rule_label": loaded.best_rule_label,
        "n_windows": int(len(window_rows)),
        "branches_per_window": int(branches_per_window),
        "horizon_steps": int(horizon),
        "perturb_fraction": float(perturb_fraction),
        "branching_metric": "pairwise_future_hamming",
        "branch_time_alignment": "delta_h_window_center",
        "correlation": corr,
        "high_low_delta": high_low,
    }
    write_csv(out / "c2_branching_scores.csv", window_rows)
    write_csv(out / "c2_branching_window_scores.csv", window_rows)
    write_csv(out / "c2_branch_detail_rows.csv", branch_detail_rows)
    write_csv(out / "c2_branch_pair_scores.csv", pair_rows)
    write_csv(out / "c2_delta_h_correlation.csv", [corr])
    write_json(out / "c2_branching_metrics_summary.json", summary)
    _plot_c2(out, window_rows, pair_rows)
    return summary


def _make_initial_board(config: ExperimentConfig, seed: int, density: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return (rng.random((config.L, config.L)) < float(density)).astype(np.uint8)


def _lifelike_step_dead_boundary_batch(boards: np.ndarray, rules: np.ndarray) -> np.ndarray:
    boards_u8 = boards.astype(np.uint8, copy=False)
    rules_u32 = rules.astype(np.uint32, copy=False)
    padded = np.pad(boards_u8.astype(np.int16), ((0, 0), (1, 1), (1, 1)), mode="constant")
    neighbors = np.zeros_like(boards_u8, dtype=np.int16)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            neighbors += padded[:, di : di + boards_u8.shape[1], dj : dj + boards_u8.shape[2]]
    update_idx = boards_u8.astype(np.uint32) * np.uint32(9) + neighbors.astype(np.uint32)
    return ((rules_u32[:, None, None] >> update_idx) & np.uint32(1)).astype(np.uint8)


def _simulate_walled_warmup(initial_board: np.ndarray, rule_id: int, warmup_steps: int, grid_split: int) -> np.ndarray:
    board = np.asarray(initial_board, dtype=np.uint8)
    if warmup_steps <= 0:
        return board.copy()
    if grid_split <= 1:
        return _simulate_full_trajectory(board, rule_id, warmup_steps, backend="numpy")[-1]
    h, w = board.shape
    if h % grid_split != 0 or w % grid_split != 0:
        raise ValueError(f"Board shape {board.shape} is not divisible by C5 wall_grid_split={grid_split}")
    bh, bw = h // grid_split, w // grid_split
    blocks = []
    for i in range(grid_split):
        for j in range(grid_split):
            blocks.append(board[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw].copy())
    block_batch = np.stack(blocks, axis=0)
    rules = np.full((block_batch.shape[0],), int(rule_id), dtype=np.uint32)
    for _ in range(int(warmup_steps)):
        block_batch = _lifelike_step_dead_boundary_batch(block_batch, rules)
    out = np.empty_like(board)
    k = 0
    for i in range(grid_split):
        for j in range(grid_split):
            out[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw] = block_batch[k]
            k += 1
    return out


def _simulate_late_control(initial_board: np.ndarray, rule_id: int, late_start: int, late_end: int, backend: str) -> np.ndarray:
    prefix = _simulate_full_trajectory(initial_board, rule_id, late_start, backend)
    late = _simulate_full_trajectory(prefix[-1], rule_id, late_end - late_start, backend)
    return late


def _simulate_late_walls(
    initial_board: np.ndarray,
    rule_id: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    wall_grid_split: int,
    backend: str,
) -> np.ndarray:
    if warmup_steps > late_start:
        raise ValueError(f"C5 warmup_steps={warmup_steps} must be <= late_window_start_steps={late_start}.")
    merged = _simulate_walled_warmup(initial_board, rule_id, warmup_steps, wall_grid_split)
    if late_start > warmup_steps:
        post = _simulate_full_trajectory(merged, rule_id, late_start - warmup_steps, backend)
        start = post[-1]
    else:
        start = merged
    return _simulate_full_trajectory(start, rule_id, late_end - late_start, backend)


def _delta_h_for_late_trajectory(
    trajectory: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    metric_backend: str,
) -> np.ndarray:
    metric_cfg = replace(config, burn_in=0, backend=metric_backend)
    result = compute_transition_mspd_auto(trajectory, metric_cfg, metric_seed=int(seed))
    return np.asarray(result.delta_h, dtype=np.float64)


def _aligned_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(a.size), int(b.size))
    if n == 0:
        return float("nan")
    diff = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _aligned_mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(a.size), int(b.size))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64))))


def _trajectory_hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] != b[:n]))


def _c5_trial(
    loaded: LoadedCAResult,
    *,
    group: str,
    rule_id: int,
    trial_idx: int,
    density: float,
    seed_x: int,
    seed_x1: int,
    warmup_steps: int,
    late_start: int,
    late_end: int,
    wall_grid_split: int,
    backend: str,
) -> dict[str, Any]:
    simulation_rule_id = rule_id if rule_id >= 0 else CONWAY_LIFE_RULE
    label = loaded.rule_labels.get(rule_id, lifelike_rule_label(simulation_rule_id))
    x0 = _make_initial_board(loaded.config, seed_x, density)
    x1 = _make_initial_board(loaded.config, seed_x1, density)

    control_a = _simulate_late_control(x0, simulation_rule_id, late_start, late_end, backend)
    control_b = _simulate_late_control(x1, simulation_rule_id, late_start, late_end, backend)
    walls = _simulate_late_walls(
        x0,
        simulation_rule_id,
        warmup_steps,
        late_start,
        late_end,
        wall_grid_split,
        backend,
    )

    dh_a = _delta_h_for_late_trajectory(control_a, loaded.config, seed_x + 17, backend)
    dh_b = _delta_h_for_late_trajectory(control_b, loaded.config, seed_x1 + 17, backend)
    dh_w = _delta_h_for_late_trajectory(walls, loaded.config, seed_x + 31, backend)

    baseline_l2 = _aligned_l2(dh_a, dh_b)
    walls_a_l2 = _aligned_l2(dh_a, dh_w)
    walls_b_l2 = _aligned_l2(dh_b, dh_w)
    walls_effect_l2 = 0.5 * (walls_a_l2 + walls_b_l2)

    baseline_abs = _aligned_mean_abs(dh_a, dh_b)
    walls_a_abs = _aligned_mean_abs(dh_a, dh_w)
    walls_b_abs = _aligned_mean_abs(dh_b, dh_w)
    walls_effect_abs = 0.5 * (walls_a_abs + walls_b_abs)

    baseline_hamming = _trajectory_hamming_distance(control_a, control_b)
    walls_a_hamming = _trajectory_hamming_distance(control_a, walls)
    walls_b_hamming = _trajectory_hamming_distance(control_b, walls)
    walls_effect_hamming = 0.5 * (walls_a_hamming + walls_b_hamming)

    return {
        "claim": "C5",
        "group": group,
        "trial_idx": int(trial_idx),
        "rule_id": int(rule_id),
        "rule_label": label,
        "seed_x": int(seed_x),
        "seed_x1": int(seed_x1),
        "initial_density": float(density),
        "warmup_steps": int(warmup_steps),
        "late_window_start_steps": int(late_start),
        "late_window_end_steps": int(late_end),
        "late_window_steps": int(late_end - late_start),
        "wall_grid_split": int(wall_grid_split),
        "delta_h_l2__baseline_distance": float(baseline_l2),
        "delta_h_l2__walls_effect_distance": float(walls_effect_l2),
        "delta_h_l2__walls_effect_distance_ctrl_a": float(walls_a_l2),
        "delta_h_l2__walls_effect_distance_ctrl_b": float(walls_b_l2),
        "delta_h_l2__anchor_effect_minus_baseline": float(walls_effect_l2 - baseline_l2),
        "delta_h_mean_abs__baseline_distance": float(baseline_abs),
        "delta_h_mean_abs__walls_effect_distance": float(walls_effect_abs),
        "delta_h_mean_abs__walls_effect_distance_ctrl_a": float(walls_a_abs),
        "delta_h_mean_abs__walls_effect_distance_ctrl_b": float(walls_b_abs),
        "delta_h_mean_abs__anchor_effect_minus_baseline": float(walls_effect_abs - baseline_abs),
        "state_hamming__baseline_distance": float(baseline_hamming),
        "state_hamming__walls_effect_distance": float(walls_effect_hamming),
        "state_hamming__walls_effect_distance_ctrl_a": float(walls_a_hamming),
        "state_hamming__walls_effect_distance_ctrl_b": float(walls_b_hamming),
        "state_hamming__anchor_effect_minus_baseline": float(walls_effect_hamming - baseline_hamming),
    }


def run_c5(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    optimized_top_k: int,
    random_controls: int,
    n_trials: int,
    warmup_steps: int | None,
    late_window_start_steps: int | None,
    late_window_end_steps: int | None,
    wall_grid_split: int,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir / "c5")
    rng = np.random.default_rng(seed)
    late_start = int(loaded.config.burn_in if late_window_start_steps is None else late_window_start_steps)
    late_end = int(loaded.config.T if late_window_end_steps is None else late_window_end_steps)
    warmup = int(max(0, late_start // 2) if warmup_steps is None else warmup_steps)
    if not (0 <= warmup <= late_start < late_end):
        raise ValueError(
            "C5 requires 0 <= warmup_steps <= late_window_start_steps < late_window_end_steps; "
            f"got warmup={warmup}, late_start={late_start}, late_end={late_end}."
        )

    ranked = _rank_rules(loaded.per_rule_scores)
    optimized_rules = _optimized_rule_ids(loaded, optimized_top_k)
    random_rules = _sample_random_rule_ids(ranked, set(optimized_rules), random_controls, rng)

    trial_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    density_values = loaded.initial_probabilities
    if density_values.size == 0:
        density_values = np.asarray([loaded.config.initial_density], dtype=np.float64)

    def evaluate_candidate(group: str, rule_id: int) -> None:
        label = loaded.rule_labels.get(rule_id, lifelike_rule_label(rule_id if rule_id >= 0 else CONWAY_LIFE_RULE))
        candidate_rows: list[dict[str, Any]] = []
        for trial_idx in range(int(n_trials)):
            density = float(density_values[trial_idx % density_values.size])
            seed_x = int(seed + 1000003 * (abs(rule_id) + 1) + 9176 * trial_idx + (0 if group == "optimized" else 50000019))
            seed_x1 = int(seed_x + 104729)
            row = _c5_trial(
                loaded,
                group=group,
                rule_id=rule_id,
                trial_idx=trial_idx,
                density=density,
                seed_x=seed_x,
                seed_x1=seed_x1,
                warmup_steps=warmup,
                late_start=late_start,
                late_end=late_end,
                wall_grid_split=wall_grid_split,
                backend=backend,
            )
            candidate_rows.append(row)
            trial_rows.append(row)
        run_row = {
            "claim": "C5",
            "group": group,
            "rule_id": int(rule_id),
            "rule_label": label,
            "n_trials": int(len(candidate_rows)),
        }
        metric_cols = [
            "delta_h_l2__anchor_effect_minus_baseline",
            "delta_h_mean_abs__anchor_effect_minus_baseline",
            "state_hamming__anchor_effect_minus_baseline",
        ]
        for col in metric_cols:
            vals = np.asarray([row[col] for row in candidate_rows], dtype=np.float64)
            run_row[col] = float(np.mean(vals))
        run_rows.append(run_row)

    for rule_id in optimized_rules:
        evaluate_candidate("optimized", rule_id)
    for rule_id in random_rules:
        evaluate_candidate("random", rule_id)

    metric_cols = [
        "delta_h_l2__anchor_effect_minus_baseline",
        "delta_h_mean_abs__anchor_effect_minus_baseline",
        "state_hamming__anchor_effect_minus_baseline",
    ]
    random_rows = [row for row in run_rows if row["group"] == "random"]
    opt_rows = [row for row in run_rows if row["group"] == "optimized"]
    for col in metric_cols:
        random_vals = np.asarray([row[col] for row in random_rows], dtype=np.float64)
        random_median = float(np.median(random_vals)) if random_vals.size else float("nan")
        for row in opt_rows:
            row[f"{col}__random_median"] = random_median
            row[f"{col}__delta_vs_random_median"] = float(row[col] - random_median)

    primary_col = "delta_h_l2__anchor_effect_minus_baseline"
    opt_effects = np.asarray([row[primary_col] for row in opt_rows], dtype=np.float64)
    random_effects = np.asarray([row[primary_col] for row in random_rows], dtype=np.float64)
    random_median = float(np.median(random_effects)) if random_effects.size else float("nan")
    deltas = [float(row[f"{primary_col}__delta_vs_random_median"]) for row in opt_rows if np.isfinite(row.get(f"{primary_col}__delta_vs_random_median", np.nan))]
    summary_rows = [
        {"claim": "C5", "metric": col, **sign_test_greater([row.get(f"{col}__delta_vs_random_median", np.nan) for row in opt_rows])}
        for col in metric_cols
    ]
    summary = {
        "claim": "C5",
        "mode": loaded.mode,
        "input_dir": str(loaded.input_dir),
        "optimized_top_k": int(len(optimized_rules)),
        "random_controls": int(len(random_rules)),
        "n_trials": int(n_trials),
        "warmup_steps": int(warmup),
        "late_window_start_steps": int(late_start),
        "late_window_end_steps": int(late_end),
        "wall_grid_split": int(wall_grid_split),
        "primary_metric": primary_col,
        "optimized_mean_frustration_effect": float(np.mean(opt_effects)) if opt_effects.size else float("nan"),
        "random_mean_frustration_effect": float(np.mean(random_effects)) if random_effects.size else float("nan"),
        "random_median_frustration_effect": random_median,
        "optimized_gt_random_median_sign_test": sign_test_greater(deltas),
    }
    write_csv(out / "c5_frustration_trial_metrics.csv", trial_rows)
    write_csv(out / "c5_frustration_run_level.csv", run_rows)
    write_csv(out / "c5_frustration_metric_summary.csv", summary_rows)
    write_json(out / "c5_frustration_summary.json", summary)
    _plot_c5(out, run_rows)
    return summary


def _simulate_full_trajectory(initial_board: np.ndarray, rule_id: int, steps: int, backend: str) -> np.ndarray:
    trajectories = simulate_lifelike_rule_batch(
        np.asarray(initial_board, dtype=np.uint8)[None, ...],
        np.asarray([int(rule_id)], dtype=np.uint32),
        int(steps),
        backend=backend,
    )
    return np.asarray(trajectories[0], dtype=np.uint8)


def _correlation_summary(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out: dict[str, Any] = {"n": int(x.size)}
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        out.update(
            {
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
            }
        )
        return out
    try:
        from scipy import stats as scipy_stats

        pearson = scipy_stats.pearsonr(x, y)
        spearman = scipy_stats.spearmanr(x, y)
        out.update(
            {
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    except Exception:
        out.update(
            {
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "pearson_p": float("nan"),
                "spearman_r": float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1]),
                "spearman_p": float("nan"),
            }
        )
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _high_low_delta_summary(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(window_rows) < 2:
        return {}
    rows = sorted(window_rows, key=lambda row: float(row["delta_h"]))
    half = len(rows) // 2
    score_key = "branching_score" if "branching_score" in rows[0] else "mean_future_hamming"
    low = np.asarray([row[score_key] for row in rows[:half]], dtype=np.float64)
    high = np.asarray([row[score_key] for row in rows[-half:]], dtype=np.float64)
    delta = float(np.mean(high) - np.mean(low)) if low.size and high.size else float("nan")
    return {
        "n_low": int(low.size),
        "n_high": int(high.size),
        "low_mean_divergence": float(np.mean(low)) if low.size else float("nan"),
        "high_mean_divergence": float(np.mean(high)) if high.size else float("nan"),
        "high_minus_low_mean_divergence": delta,
    }


def _matplotlib():
    mpl_config = REPO_ROOT / "analysis" / "results" / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache = REPO_ROOT / "analysis" / "results" / ".cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _boxplot(ax: Any, values: list[list[float]], labels: list[str]) -> None:
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)


def _plot_c1(out: Path, rule_level_rows: list[dict[str, Any]], contrast_rows: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    score_key = "holdout_mean_mspd" if rule_level_rows and "holdout_mean_mspd" in rule_level_rows[0] else "mean_mspd"
    opt = [float(row[score_key]) for row in rule_level_rows if row["group"] == "optimized"]
    rnd = [float(row[score_key]) for row in rule_level_rows if row["group"] == "random"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=150)
    _boxplot(ax, [rnd, opt], ["random", "optimized"])
    ax.scatter(np.ones(len(rnd)), rnd, s=16, alpha=0.55, color="#6b7280")
    ax.scatter(np.full(len(opt), 2), opt, s=30, alpha=0.9, color="#2563eb")
    ax.set_ylabel("transition-law MSPD")
    ax.set_title("C1 CA MSPD optimized vs random")
    fig.tight_layout()
    fig.savefig(out / "c1_ca_optimized_vs_random.png")
    plt.close(fig)

    if contrast_rows:
        deltas = [float(row["delta_vs_random_median"]) for row in contrast_rows]
        fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=150)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.plot(np.arange(len(deltas)), deltas, marker="o", linewidth=1.2, markersize=4)
        ax.set_xlabel("matched optimized init")
        ax.set_ylabel("MSPD - random median")
        ax.set_title("C1 matched contrasts")
        fig.tight_layout()
        fig.savefig(out / "c1_ca_matched_contrasts.png")
        plt.close(fig)


def _plot_c2(out: Path, window_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]] | None = None) -> None:
    plt = _matplotlib()
    x = np.asarray([row["delta_h"] for row in window_rows], dtype=np.float64)
    score_key = "branching_score" if window_rows and "branching_score" in window_rows[0] else "mean_future_hamming"
    y = np.asarray([row[score_key] for row in window_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=150)
    ax.scatter(x, y, s=32, color="#7c3aed", alpha=0.8)
    if x.size >= 2 and np.std(x) > 0.0:
        coeff = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="#111827", linewidth=1.2)
    ax.set_xlabel("Delta-H")
    ax.set_ylabel("pairwise future Hamming divergence")
    ax.set_title("C2 CA Delta-H vs perturbation sensitivity")
    fig.tight_layout()
    fig.savefig(out / "c2_ca_delta_h_branching_sensitivity.png")
    plt.close(fig)

    pair_rows = pair_rows or []
    by_window: dict[int, list[float]] = {}
    for row in pair_rows:
        try:
            window_idx = int(row["window_idx"])
            value = float(row.get("pairwise_future_hamming", "nan"))
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(value):
            by_window.setdefault(window_idx, []).append(value)
    if by_window:
        lows: list[float] = []
        highs: list[float] = []
        n_pairs: list[int] = []
        values_for_rows: list[list[float]] = []
        for row in window_rows:
            vals = by_window.get(int(row["window_idx"]), [])
            values_for_rows.append(vals)
            lo, hi = _bootstrap_mean_ci(vals)
            center = float(row[score_key])
            lows.append(center if not np.isfinite(lo) else lo)
            highs.append(center if not np.isfinite(hi) else hi)
            n_pairs.append(len(vals))
        lo_arr = np.asarray(lows, dtype=np.float64)
        hi_arr = np.asarray(highs, dtype=np.float64)
        yerr = np.vstack([np.maximum(0.0, y - lo_arr), np.maximum(0.0, hi_arr - y)])
        fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.5,
            color="#7c3aed",
            ecolor="#7c3aed",
            elinewidth=1.2,
            capsize=3,
            alpha=0.92,
            label="window mean +/- bootstrap 95% CI",
        )
        if x.size >= 2 and np.std(x) > 0.0:
            coeff = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#111827", linewidth=1.2)
        span = max(float(np.ptp(x)), 1.0)
        for xi, vals in zip(x, values_for_rows):
            if not vals:
                continue
            offsets = np.linspace(-0.004, 0.004, len(vals)) * span
            ax.scatter(np.full(len(vals), xi) + offsets, vals, s=12, color="#a78bfa", alpha=0.35, linewidth=0)
        ax.set_xlabel("Delta-H")
        ax.set_ylabel("pairwise future Hamming divergence")
        ax.set_title("C2 CA Delta-H vs perturbation sensitivity\nwith branch-pair bootstrap intervals")
        ax.text(
            0.02,
            0.02,
            f"error bars use raw branch pairs per window; median n_pairs={np.median(n_pairs):.0f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 3},
        )
        ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "c2_ca_delta_h_branching_sensitivity_ci.png")
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    ordered = sorted(window_rows, key=lambda row: int(row["window_idx"]))
    ax1.plot([row["window_idx"] for row in ordered], [row["delta_h"] for row in ordered], color="#2563eb", label="Delta-H")
    ax1.set_xlabel("window")
    ax1.set_ylabel("Delta-H", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(
        [row["window_idx"] for row in ordered],
        [row[score_key] for row in ordered],
        color="#dc2626",
        label="divergence",
    )
    ax2.set_ylabel("pairwise future Hamming divergence", color="#dc2626")
    ax1.set_title("C2 sampled windows")
    fig.tight_layout()
    fig.savefig(out / "c2_ca_delta_h_trace_sampled_windows.png")
    plt.close(fig)


def _plot_c5(out: Path, run_rows: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    primary_col = "delta_h_l2__anchor_effect_minus_baseline"
    opt = [float(row[primary_col]) for row in run_rows if row["group"] == "optimized"]
    rnd = [float(row[primary_col]) for row in run_rows if row["group"] == "random"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=150)
    ax.axhline(0.0, color="black", linewidth=0.8)
    _boxplot(ax, [rnd, opt], ["random", "optimized"])
    ax.scatter(np.ones(len(rnd)), rnd, s=14, alpha=0.5, color="#6b7280")
    ax.scatter(np.full(len(opt), 2), opt, s=24, alpha=0.85, color="#059669")
    ax.set_ylabel("Delta-H L2: walls effect - control baseline")
    ax.set_title("C5 CA frustration contrast")
    fig.tight_layout()
    fig.savefig(out / "c5_ca_frustration_contrast.png")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/gol_transition_mspd/ca_posthoc_claims.yaml",
        help="YAML config. CLI arguments override values from this file.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Existing GoL MSPD result dir. Defaults to latest completed analysis/results/gol_transition_mspd/*.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root used only when --input-dir is omitted.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <input-dir>/posthoc_claims.")
    parser.add_argument("--task", choices=["all", "c1", "c2", "c5"], default=None)
    parser.add_argument("--backend", choices=["numpy", "jax"], default=None, help="Simulation backend for branch checks.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optimized-top-k", type=int, default=None)
    parser.add_argument("--random-controls", type=int, default=None)
    parser.add_argument("--c1-eval-mode", choices=["fresh_holdout", "saved_score"], default=None)
    parser.add_argument("--c1-holdout-n-initial-boards", type=int, default=None)
    parser.add_argument("--c1-holdout-density-mode", choices=["source", "fixed", "uniform"], default=None)
    parser.add_argument("--c1-holdout-density", type=float, default=None)
    parser.add_argument("--c1-holdout-density-range", type=float, nargs=2, default=None)
    parser.add_argument("--c1-eval-batch-size", type=int, default=None)

    parser.add_argument("--c2-branch-windows", type=int, default=None)
    parser.add_argument("--c2-branches-per-window", type=int, default=None)
    parser.add_argument("--c2-horizon", type=int, default=None)
    parser.add_argument("--c2-perturb-fraction", type=float, default=None)

    parser.add_argument("--c5-random-controls", type=int, default=None)
    parser.add_argument("--c5-n-trials", type=int, default=None)
    parser.add_argument("--c5-warmup-steps", type=int, default=None)
    parser.add_argument("--c5-late-window-start-steps", type=int, default=None)
    parser.add_argument("--c5-late-window-end-steps", type=int, default=None)
    parser.add_argument("--c5-wall-grid-split", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _effective_config(args)
    input_dir = (
        _resolve_path(cfg["input_dir"])
        if cfg.get("input_dir")
        else _latest_completed_input_dir(_resolve_path(cfg["results_root"]))
    )
    output_dir = _resolve_path(cfg["output_dir"]) if cfg.get("output_dir") else input_dir / "posthoc_claims"
    ensure_dir(output_dir)

    loaded = load_ca_result(input_dir)
    backend = cfg.get("backend") or loaded.config.backend
    completed_dirs = _completed_input_dirs(_resolve_path(cfg["results_root"]))[:10]
    _log(
        "CA MSPD posthoc start: "
        f"mode={loaded.mode} input_dir={input_dir} output_dir={output_dir} "
        f"rule={loaded.best_rule_id} {loaded.best_rule_label} backend={backend}"
    )

    summaries: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": loaded.mode,
        "best_rule_id": loaded.best_rule_id,
        "best_rule_label": loaded.best_rule_label,
        "config": cfg,
        "available_completed_input_dirs": [str(path) for path in completed_dirs],
    }
    if cfg["task"] in {"all", "c1"}:
        _log(f"C1: optimized-vs-random MSPD contrast mode={cfg['c1_eval_mode']}")
        summaries["c1"] = run_c1(
            loaded,
            output_dir,
            optimized_top_k=int(cfg["optimized_top_k"]),
            random_controls=int(cfg["random_controls"]),
            eval_mode=str(cfg["c1_eval_mode"]),
            holdout_n_initial_boards=int(cfg["c1_holdout_n_initial_boards"]),
            holdout_density_mode=str(cfg["c1_holdout_density_mode"]),
            holdout_density=None if cfg["c1_holdout_density"] is None else float(cfg["c1_holdout_density"]),
            holdout_density_range=_density_range_from_raw(cfg["c1_holdout_density_range"]),
            eval_batch_size=None if cfg["c1_eval_batch_size"] is None else int(cfg["c1_eval_batch_size"]),
            backend=backend,
            seed=int(cfg["seed"]) + 101,
        )
    if cfg["task"] in {"all", "c2"}:
        _log("C2: Delta-H vs branch perturbation sensitivity")
        summaries["c2"] = run_c2(
            loaded,
            output_dir,
            n_windows=int(cfg["c2_branch_windows"]),
            branches_per_window=int(cfg["c2_branches_per_window"]),
            horizon=int(cfg["c2_horizon"]),
            perturb_fraction=float(cfg["c2_perturb_fraction"]),
            backend=backend,
            seed=int(cfg["seed"]) + 202,
        )
    if cfg["task"] in {"all", "c5"}:
        _log("C5: control_a/control_b/walls history-dependence contrast")
        summaries["c5"] = run_c5(
            loaded,
            output_dir,
            optimized_top_k=int(cfg["optimized_top_k"]),
            random_controls=int(cfg["c5_random_controls"] if cfg["c5_random_controls"] is not None else cfg["random_controls"]),
            n_trials=int(cfg["c5_n_trials"]),
            warmup_steps=None if cfg["c5_warmup_steps"] is None else int(cfg["c5_warmup_steps"]),
            late_window_start_steps=None
            if cfg["c5_late_window_start_steps"] is None
            else int(cfg["c5_late_window_start_steps"]),
            late_window_end_steps=None
            if cfg["c5_late_window_end_steps"] is None
            else int(cfg["c5_late_window_end_steps"]),
            wall_grid_split=int(cfg["c5_wall_grid_split"]),
            backend=backend,
            seed=int(cfg["seed"]) + 303,
        )

    write_json(output_dir / "ca_posthoc_claims_summary.json", summaries)
    _log(f"CA MSPD posthoc done: {output_dir}")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
