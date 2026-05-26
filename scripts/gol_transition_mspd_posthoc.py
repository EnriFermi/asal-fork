#!/usr/bin/env python3
"""Posthoc C1/C2/C5 checks for transition-law MSPD cellular automata runs.

This script is intentionally downstream-only: it reads an existing
`gol_transition_mspd_experiment.py` optimization/rule-sweep output directory and
does not rerun the optimizer.  The checks mirror the paper-suite semantics used
for Flow-Lenia/PLife++ as closely as the discrete CA setting allows:

* C1: MSPD-optimized CA candidate(s) versus matched random controls.
* C2: Delta-H predicts sensitivity to small future perturbations.
* C5: structured "walls"/cell-shuffle perturbations create more future
  divergence than matched random-control perturbations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gol_transition_mspd_experiment import (  # noqa: E402
    CONWAY_LIFE_RULE,
    ExperimentConfig,
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
    "c2_branch_windows": 24,
    "c2_branches_per_window": 4,
    "c2_horizon": 64,
    "c2_perturb_fraction": 0.01,
    "c5_random_controls": None,
    "c5_anchors": 12,
    "c5_branches_per_anchor": 4,
    "c5_horizon": 64,
    "c5_control_flip_fraction": 0.01,
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
        if "optimized_top_k" in c1:
            flat["optimized_top_k"] = c1["optimized_top_k"]
        if "random_controls" in c1:
            flat["random_controls"] = c1["random_controls"]

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
            "anchors": "c5_anchors",
            "branches_per_anchor": "c5_branches_per_anchor",
            "horizon": "c5_horizon",
            "control_flip_fraction": "c5_control_flip_fraction",
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


def run_c1(
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
    }

    write_csv(out / "c1_checkpoint_scores.csv", score_rows)
    write_csv(out / "c1_rule_level_scores.csv", rule_level_rows)
    write_csv(out / "c1_group_contrasts.csv", contrast_rows)
    write_json(out / "c1_summary.json", summary)
    _plot_c1(out, rule_level_rows, contrast_rows)
    return summary


def _window_starts(config: ExperimentConfig, delta_h: np.ndarray, trajectory_len: int, horizon: int) -> np.ndarray:
    starts = config.burn_in + np.arange(delta_h.size, dtype=np.int64) * int(config.window_step)
    max_start = int(trajectory_len) - 1 - int(horizon)
    return starts[starts <= max_start]


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


def _block_shuffle(state: np.ndarray, grid_split: int, rng: np.random.Generator) -> np.ndarray:
    board = np.asarray(state, dtype=np.uint8)
    if grid_split <= 1:
        return _small_bit_flip(board, 0.5, rng)
    h, w = board.shape
    if h % grid_split != 0 or w % grid_split != 0:
        raise ValueError(f"Board shape {board.shape} is not divisible by grid_split={grid_split}")
    bh, bw = h // grid_split, w // grid_split
    blocks = []
    for i in range(grid_split):
        for j in range(grid_split):
            blocks.append(board[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw].copy())
    perm = rng.permutation(len(blocks))
    out = np.empty_like(board)
    k = 0
    for i in range(grid_split):
        for j in range(grid_split):
            out[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw] = blocks[int(perm[k])]
            k += 1
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
    starts = _window_starts(loaded.config, loaded.best_delta_h, loaded.best_trajectory.shape[0], horizon)
    picked = _quantile_window_indices(loaded.best_delta_h, starts, n_windows)
    if picked.size < 2:
        raise ValueError("C2 requires at least two valid Delta-H windows before trajectory end.")

    branch_initials: list[np.ndarray] = []
    branch_meta: list[dict[str, Any]] = []
    for window_idx in picked:
        t0 = int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step))
        base_state = loaded.best_trajectory[t0]
        for rep in range(branches_per_window):
            branch_initials.append(_small_bit_flip(base_state, perturb_fraction, rng))
            branch_meta.append(
                {
                    "window_idx": int(window_idx),
                    "t0": t0,
                    "branch_rep": rep,
                    "delta_h": float(loaded.best_delta_h[int(window_idx)]),
                }
            )

    branches = _simulate_branches(branch_initials, loaded.best_rule_id, horizon, backend)
    branch_rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(branch_meta):
        t0 = int(meta["t0"])
        base_future = loaded.best_trajectory[t0 : t0 + horizon + 1]
        hamming_t = np.mean(branches[idx, 1:] != base_future[1:], axis=(1, 2))
        branch_rows.append(
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
    for window_idx in picked:
        rows = [row for row in branch_rows if int(row["window_idx"]) == int(window_idx)]
        vals = np.asarray([row["mean_future_hamming"] for row in rows], dtype=np.float64)
        window_rows.append(
            {
                "claim": "C2",
                "rule_id": loaded.best_rule_id,
                "rule_label": loaded.best_rule_label,
                "window_idx": int(window_idx),
                "t0": int(loaded.config.burn_in + int(window_idx) * int(loaded.config.window_step)),
                "delta_h": float(loaded.best_delta_h[int(window_idx)]),
                "mean_future_hamming": float(np.mean(vals)),
                "std_future_hamming": float(np.std(vals, ddof=0)),
                "n_branches": int(vals.size),
            }
        )

    x = np.asarray([row["delta_h"] for row in window_rows], dtype=np.float64)
    y = np.asarray([row["mean_future_hamming"] for row in window_rows], dtype=np.float64)
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
        "correlation": corr,
        "high_low_delta": high_low,
    }
    write_csv(out / "c2_branching_scores.csv", branch_rows)
    write_csv(out / "c2_branching_window_scores.csv", window_rows)
    write_csv(out / "c2_delta_h_correlation.csv", [corr])
    write_json(out / "c2_branching_metrics_summary.json", summary)
    _plot_c2(out, window_rows)
    return summary


def _trajectory_frustration_scores(
    trajectory: np.ndarray,
    rule_id: int,
    label: str,
    group: str,
    item_id: str,
    config: ExperimentConfig,
    *,
    horizon: int,
    anchors: Sequence[int],
    branches_per_anchor: int,
    control_flip_fraction: float,
    wall_grid_split: int,
    backend: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    branch_initials: list[np.ndarray] = []
    branch_meta: list[dict[str, Any]] = []
    for anchor_idx, t0 in enumerate(anchors):
        base_state = trajectory[int(t0)]
        for rep in range(branches_per_anchor):
            branch_initials.append(_small_bit_flip(base_state, control_flip_fraction, rng))
            branch_meta.append({"anchor_idx": anchor_idx, "t0": int(t0), "branch_rep": rep, "condition": "control_b"})
            branch_initials.append(_block_shuffle(base_state, wall_grid_split, rng))
            branch_meta.append({"anchor_idx": anchor_idx, "t0": int(t0), "branch_rep": rep, "condition": "walls"})

    branches = _simulate_branches(branch_initials, rule_id, horizon, backend)
    raw_rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(branch_meta):
        t0 = int(meta["t0"])
        base_future = trajectory[t0 : t0 + horizon + 1]
        hamming_t = np.mean(branches[idx, 1:] != base_future[1:], axis=(1, 2))
        raw_rows.append(
            {
                "claim": "C5",
                "group": group,
                "item_id": item_id,
                "rule_id": int(rule_id),
                "rule_label": label,
                **meta,
                "horizon_steps": int(horizon),
                "mean_future_hamming": float(np.mean(hamming_t)),
                "final_future_hamming": float(hamming_t[-1]),
                "max_future_hamming": float(np.max(hamming_t)),
            }
        )
    return raw_rows


def run_c5(
    loaded: LoadedCAResult,
    output_dir: Path,
    *,
    optimized_top_k: int,
    random_controls: int,
    n_anchors: int,
    branches_per_anchor: int,
    horizon: int,
    control_flip_fraction: float,
    wall_grid_split: int,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir / "c5")
    rng = np.random.default_rng(seed)
    max_anchor = loaded.config.T - int(horizon)
    if max_anchor <= loaded.config.burn_in:
        raise ValueError("C5 horizon leaves no valid post-burn-in anchor times.")
    anchors = np.unique(np.rint(np.linspace(loaded.config.burn_in, max_anchor, n_anchors)).astype(np.int64))

    ranked = _rank_rules(loaded.per_rule_scores)
    optimized_rules = _optimized_rule_ids(loaded, optimized_top_k)
    random_rules = _sample_random_rule_ids(ranked, set(optimized_rules), random_controls, rng)

    init_boards = loaded.initial_boards
    if init_boards.ndim != 3 or init_boards.shape[0] == 0:
        init_boards = loaded.best_initial_board[None, ...]
    init_ids = list(range(int(init_boards.shape[0])))

    raw_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    def add_item(group: str, rule_id: int, init_id: int, trajectory: np.ndarray) -> None:
        label = loaded.rule_labels.get(rule_id, lifelike_rule_label(rule_id) if rule_id >= 0 else str(rule_id))
        item_id = f"{group}_rule{rule_id}_init{init_id}"
        simulation_rule_id = rule_id if rule_id >= 0 else CONWAY_LIFE_RULE
        rows = _trajectory_frustration_scores(
            trajectory,
            simulation_rule_id,
            label,
            group,
            item_id,
            loaded.config,
            horizon=horizon,
            anchors=anchors,
            branches_per_anchor=branches_per_anchor,
            control_flip_fraction=control_flip_fraction,
            wall_grid_split=wall_grid_split,
            backend=backend,
            rng=rng,
        )
        raw_rows.extend(rows)
        control_vals = np.asarray([row["mean_future_hamming"] for row in rows if row["condition"] == "control_b"], dtype=np.float64)
        wall_vals = np.asarray([row["mean_future_hamming"] for row in rows if row["condition"] == "walls"], dtype=np.float64)
        run_rows.append(
            {
                "claim": "C5",
                "group": group,
                "item_id": item_id,
                "rule_id": int(rule_id),
                "rule_label": label,
                "init_id": int(init_id),
                "d_control_a_control_b": float(np.mean(control_vals)),
                "d_control_a_walls": float(np.mean(wall_vals)),
                "frustration_effect": float(np.mean(wall_vals) - np.mean(control_vals)),
                "n_anchor_reps": int(min(control_vals.size, wall_vals.size)),
            }
        )

    for rule_id in optimized_rules:
        for init_id in init_ids:
            if (
                rule_id == loaded.best_rule_id
                and loaded.best_trajectory.shape[0] == loaded.config.T + 1
                and np.array_equal(init_boards[init_id], loaded.best_initial_board)
            ):
                trajectory = loaded.best_trajectory
            else:
                trajectory = _simulate_full_trajectory(init_boards[init_id], rule_id, loaded.config.T, backend)
            add_item("optimized", rule_id, init_id, trajectory)

    for rule_id in random_rules:
        if rule_id < 0:
            board = loaded.random_initial_boards.get(rule_id)
            if board is None:
                continue
            trajectory = _simulate_full_trajectory(board, CONWAY_LIFE_RULE, loaded.config.T, backend)
            add_item("random", rule_id, 0, trajectory)
        else:
            for init_id in init_ids:
                trajectory = _simulate_full_trajectory(init_boards[init_id], rule_id, loaded.config.T, backend)
                add_item("random", rule_id, init_id, trajectory)

    opt_effects = np.asarray([row["frustration_effect"] for row in run_rows if row["group"] == "optimized"], dtype=np.float64)
    random_effects = np.asarray([row["frustration_effect"] for row in run_rows if row["group"] == "random"], dtype=np.float64)
    random_median = float(np.median(random_effects)) if random_effects.size else float("nan")
    deltas = [float(v - random_median) for v in opt_effects if np.isfinite(random_median)]
    summary = {
        "claim": "C5",
        "mode": loaded.mode,
        "input_dir": str(loaded.input_dir),
        "optimized_top_k": int(len(optimized_rules)),
        "random_controls": int(len(random_rules)),
        "n_initial_boards": int(len(init_ids)),
        "n_anchors": int(len(anchors)),
        "branches_per_anchor": int(branches_per_anchor),
        "horizon_steps": int(horizon),
        "control_flip_fraction": float(control_flip_fraction),
        "wall_grid_split": int(wall_grid_split),
        "optimized_mean_frustration_effect": float(np.mean(opt_effects)) if opt_effects.size else float("nan"),
        "random_mean_frustration_effect": float(np.mean(random_effects)) if random_effects.size else float("nan"),
        "random_median_frustration_effect": random_median,
        "optimized_gt_random_median_sign_test": sign_test_greater(deltas),
    }
    write_csv(out / "c5_frustration_branch_rows.csv", raw_rows)
    write_csv(out / "c5_frustration_run_level.csv", run_rows)
    write_csv(out / "c5_frustration_metric_summary.csv", [summary])
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
    low = np.asarray([row["mean_future_hamming"] for row in rows[:half]], dtype=np.float64)
    high = np.asarray([row["mean_future_hamming"] for row in rows[-half:]], dtype=np.float64)
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
    opt = [float(row["mean_mspd"]) for row in rule_level_rows if row["group"] == "optimized"]
    rnd = [float(row["mean_mspd"]) for row in rule_level_rows if row["group"] == "random"]
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


def _plot_c2(out: Path, window_rows: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    x = np.asarray([row["delta_h"] for row in window_rows], dtype=np.float64)
    y = np.asarray([row["mean_future_hamming"] for row in window_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=150)
    ax.scatter(x, y, s=32, color="#7c3aed", alpha=0.8)
    if x.size >= 2 and np.std(x) > 0.0:
        coeff = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="#111827", linewidth=1.2)
    ax.set_xlabel("Delta-H")
    ax.set_ylabel("future Hamming divergence")
    ax.set_title("C2 CA Delta-H vs perturbation sensitivity")
    fig.tight_layout()
    fig.savefig(out / "c2_ca_delta_h_branching_sensitivity.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    ordered = sorted(window_rows, key=lambda row: int(row["window_idx"]))
    ax1.plot([row["window_idx"] for row in ordered], [row["delta_h"] for row in ordered], color="#2563eb", label="Delta-H")
    ax1.set_xlabel("window")
    ax1.set_ylabel("Delta-H", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(
        [row["window_idx"] for row in ordered],
        [row["mean_future_hamming"] for row in ordered],
        color="#dc2626",
        label="divergence",
    )
    ax2.set_ylabel("future Hamming divergence", color="#dc2626")
    ax1.set_title("C2 sampled windows")
    fig.tight_layout()
    fig.savefig(out / "c2_ca_delta_h_trace_sampled_windows.png")
    plt.close(fig)


def _plot_c5(out: Path, run_rows: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    opt = [float(row["frustration_effect"]) for row in run_rows if row["group"] == "optimized"]
    rnd = [float(row["frustration_effect"]) for row in run_rows if row["group"] == "random"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=150)
    ax.axhline(0.0, color="black", linewidth=0.8)
    _boxplot(ax, [rnd, opt], ["random", "optimized"])
    ax.scatter(np.ones(len(rnd)), rnd, s=14, alpha=0.5, color="#6b7280")
    ax.scatter(np.full(len(opt), 2), opt, s=24, alpha=0.85, color="#059669")
    ax.set_ylabel("d(control_a,walls) - d(control_a,control_b)")
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

    parser.add_argument("--c2-branch-windows", type=int, default=None)
    parser.add_argument("--c2-branches-per-window", type=int, default=None)
    parser.add_argument("--c2-horizon", type=int, default=None)
    parser.add_argument("--c2-perturb-fraction", type=float, default=None)

    parser.add_argument("--c5-random-controls", type=int, default=None)
    parser.add_argument("--c5-anchors", type=int, default=None)
    parser.add_argument("--c5-branches-per-anchor", type=int, default=None)
    parser.add_argument("--c5-horizon", type=int, default=None)
    parser.add_argument("--c5-control-flip-fraction", type=float, default=None)
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
        _log("C1: optimized-vs-random MSPD contrast")
        summaries["c1"] = run_c1(
            loaded,
            output_dir,
            optimized_top_k=int(cfg["optimized_top_k"]),
            random_controls=int(cfg["random_controls"]),
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
        _log("C5: structured cell-shuffle frustration contrast")
        summaries["c5"] = run_c5(
            loaded,
            output_dir,
            optimized_top_k=int(cfg["optimized_top_k"]),
            random_controls=int(cfg["c5_random_controls"] if cfg["c5_random_controls"] is not None else cfg["random_controls"]),
            n_anchors=int(cfg["c5_anchors"]),
            branches_per_anchor=int(cfg["c5_branches_per_anchor"]),
            horizon=int(cfg["c5_horizon"]),
            control_flip_fraction=float(cfg["c5_control_flip_fraction"]),
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
