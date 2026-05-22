#!/usr/bin/env python3
"""Transition-law MSPD optimization for Conway's Game of Life.

This experiment optimizes only the initial binary board. The score is computed
from local empirical transition laws T_hat_i over Game-of-Life transition
symbols, not from velocity or displacement features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial.distance import pdist
except Exception:  # pragma: no cover - exercised only when scipy is missing.
    pdist = None


N_TRANSITION_SYMBOLS = 1024
NEIGHBORHOOD_MASK = (1 << 9) - 1
CONWAY_LIFE_RULE = 6152
N_LIFELIKE_RULES = 2**18


@dataclass
class ExperimentConfig:
    L: int = 64
    T: int = 512
    burn_in: int = 64
    window_size: int = 32
    window_step: int = 8
    n_cell_sample: int = 256
    null_reps: int = 4
    population_size: int = 32
    generations: int = 50
    elite_frac: float = 0.1
    mutation_rate: float = 0.01
    initial_density: float = 0.25
    random_seed: int = 0

    backend: str = "jax"
    eval_batch_size: int = 8
    jax_metric_batch_size: int = 0
    distance: str = "js"
    pooled_null: bool = True
    pair_sample: Optional[int] = 512
    delta_h_floor: float = 0.0
    min_delta_h_nonzero_frac: float = 0.5
    delta_h_nonzero_eps: float = 0.0
    mspd_floor: float = 1e-6
    eps: float = 1e-12
    tournament_size: int = 3
    random_controls: Optional[int] = None

    output_dir: str = field(
        default_factory=lambda: "analysis/results/gol_transition_mspd/"
        + time.strftime("run_%Y%m%d_%H%M%S")
    )
    save_videos: bool = True
    video_top_k: int = 3
    video_fps: int = 12
    video_scale: int = 8
    video_stride: int = 1
    montage_frames: int = 16
    verbose: bool = True
    progress_every: int = 1


@dataclass
class EvaluationResult:
    mspd_score: float
    fitness_score: float
    delta_h_nonzero_frac: float
    passes_delta_h_filter: bool
    delta_h: np.ndarray
    h_trace: np.ndarray
    h_raw: np.ndarray
    h_null: np.ndarray
    scales: np.ndarray
    scale_scores: np.ndarray
    active_counts: np.ndarray
    active_available_counts: np.ndarray
    trajectory: Optional[np.ndarray] = None


def validate_config(config: ExperimentConfig) -> None:
    if config.L <= 0:
        raise ValueError("L must be positive.")
    if config.T <= 0:
        raise ValueError("T must be positive.")
    if config.burn_in < 0 or config.burn_in >= config.T:
        raise ValueError("burn_in must satisfy 0 <= burn_in < T.")
    if config.window_size <= 0 or config.window_step <= 0:
        raise ValueError("window_size and window_step must be positive.")
    if config.T - config.burn_in < config.window_size:
        raise ValueError("T - burn_in must be at least window_size.")
    if config.n_cell_sample <= 0:
        raise ValueError("n_cell_sample must be positive.")
    if config.null_reps < 0:
        raise ValueError("null_reps must be non-negative.")
    if config.population_size <= 0 or config.generations <= 0:
        raise ValueError("population_size and generations must be positive.")
    if config.backend not in {"jax", "numpy"}:
        raise ValueError("backend must be either 'jax' or 'numpy'.")
    if config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if config.jax_metric_batch_size < 0:
        raise ValueError("jax_metric_batch_size must be non-negative; 0 means full rollout batch.")
    if not (0.0 <= config.elite_frac <= 1.0):
        raise ValueError("elite_frac must be in [0, 1].")
    if not (0.0 <= config.mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")
    if not (0.0 <= config.initial_density <= 1.0):
        raise ValueError("initial_density must be in [0, 1].")
    if config.distance not in {"js", "tv"}:
        raise ValueError("distance must be either 'js' or 'tv'.")
    if config.pair_sample is not None and config.pair_sample < 0:
        raise ValueError("pair_sample must be non-negative or None.")
    if not (0.0 <= config.min_delta_h_nonzero_frac < 1.0):
        raise ValueError("min_delta_h_nonzero_frac must be in [0, 1).")
    if config.delta_h_nonzero_eps < 0.0:
        raise ValueError("delta_h_nonzero_eps must be non-negative.")
    if config.tournament_size <= 0:
        raise ValueError("tournament_size must be positive.")
    if config.random_controls is not None and config.random_controls < 0:
        raise ValueError("random_controls must be non-negative.")
    if config.video_top_k < 0:
        raise ValueError("video_top_k must be non-negative.")
    if config.video_fps <= 0 or config.video_scale <= 0 or config.video_stride <= 0:
        raise ValueError("video_fps, video_scale, and video_stride must be positive.")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be positive.")


def log_verbose(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def life_step(board: np.ndarray) -> np.ndarray:
    """Apply one standard Conway Life step with toroidal boundaries."""
    board_u8 = board.astype(np.uint8, copy=False)
    neighbors = np.zeros_like(board_u8, dtype=np.uint8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += np.roll(board_u8, shift=(-di, -dj), axis=(0, 1))
    survives = (board_u8 == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (board_u8 == 0) & (neighbors == 3)
    return (survives | born).astype(np.uint8)


def life_step_batch_numpy(boards: np.ndarray) -> np.ndarray:
    """Apply one Life step to a batch of boards on CPU/NumPy."""
    boards_u8 = boards.astype(np.uint8, copy=False)
    boards_i16 = boards_u8.astype(np.int16, copy=False)
    neighbors = np.zeros_like(boards_i16, dtype=np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += np.roll(boards_i16, shift=(-di, -dj), axis=(-2, -1))
    survives = (boards_u8 == 1) & ((neighbors == 2) | (neighbors == 3))
    born = (boards_u8 == 0) & (neighbors == 3)
    return (survives | born).astype(np.uint8)


def simulate_life(initial_board: np.ndarray, steps: int) -> np.ndarray:
    """Return a trajectory of shape (steps + 1, L, L)."""
    if initial_board.ndim != 2:
        raise ValueError("initial_board must be a 2-D array.")
    trajectory = np.empty((steps + 1, *initial_board.shape), dtype=np.uint8)
    trajectory[0] = (initial_board > 0).astype(np.uint8)
    for t in range(steps):
        trajectory[t + 1] = life_step(trajectory[t])
    return trajectory


def simulate_life_batch_numpy(initial_boards: np.ndarray, steps: int) -> np.ndarray:
    """Return trajectories with shape (batch, steps + 1, L, L)."""
    if initial_boards.ndim != 3:
        raise ValueError("initial_boards must have shape (batch, L, L).")
    batch = int(initial_boards.shape[0])
    trajectory = np.empty((batch, steps + 1, *initial_boards.shape[1:]), dtype=np.uint8)
    trajectory[:, 0] = (initial_boards > 0).astype(np.uint8)
    state = trajectory[:, 0]
    for t in range(steps):
        state = life_step_batch_numpy(state)
        trajectory[:, t + 1] = state
    return trajectory


_JAX_LIFE_BATCH_SIMULATOR = None


def get_jax_life_batch_simulator():
    """Create/cache a jitted batched Conway Life rollout."""
    global _JAX_LIFE_BATCH_SIMULATOR
    if _JAX_LIFE_BATCH_SIMULATOR is not None:
        return _JAX_LIFE_BATCH_SIMULATOR

    try:
        import jax
        import jax.numpy as jnp
        from functools import partial
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "JAX backend requested, but jax/jaxlib could not be imported. "
            "Set backend='numpy' or install JAX."
        ) from exc

    def step_batch(state):
        state_u8 = state.astype(jnp.uint8)
        state_i16 = state_u8.astype(jnp.int16)
        neighbors = jnp.zeros_like(state_i16)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbors = neighbors + jnp.roll(state_i16, shift=(-di, -dj), axis=(-2, -1))
        survives = (state_u8 == 1) & ((neighbors == 2) | (neighbors == 3))
        born = (state_u8 == 0) & (neighbors == 3)
        return (survives | born).astype(jnp.uint8)

    @partial(jax.jit, static_argnames=("steps",))
    def simulate(initial_boards, steps: int):
        initial_boards = initial_boards.astype(jnp.uint8)

        def scan_step(carry, _unused):
            next_state = step_batch(carry)
            return next_state, next_state

        _final_state, frames = jax.lax.scan(scan_step, initial_boards, xs=None, length=steps)
        trajectory_t_first = jnp.concatenate([initial_boards[None, ...], frames], axis=0)
        return jnp.swapaxes(trajectory_t_first, 0, 1)

    _JAX_LIFE_BATCH_SIMULATOR = simulate
    return simulate


def simulate_life_batch(initial_boards: np.ndarray, steps: int, backend: str) -> np.ndarray:
    """Return batched Life trajectories using the requested backend."""
    boards = (initial_boards > 0).astype(np.uint8, copy=False)
    if backend == "numpy":
        return simulate_life_batch_numpy(boards, steps)
    if backend == "jax":
        import jax.numpy as jnp

        simulator = get_jax_life_batch_simulator()
        trajectories = simulator(jnp.asarray(boards), steps)
        return np.asarray(trajectories.block_until_ready(), dtype=np.uint8)
    raise ValueError(f"Unknown backend: {backend}")


def lifelike_rule_label(rule_id: int) -> str:
    """Return B/S notation for the 18-bit totalistic Life-like rule."""
    rule = int(rule_id)
    births = "".join(str(n) for n in range(9) if ((rule >> n) & 1))
    survives = "".join(str(n) for n in range(9) if ((rule >> (9 + n)) & 1))
    return f"B{births}/S{survives}"


def life_step_lifelike_rule_batch_numpy(boards: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """Apply one totalistic Life-like rule step to each board/rule pair."""
    boards_u8 = boards.astype(np.uint8, copy=False)
    rules_u32 = rules.astype(np.uint32, copy=False)
    if boards_u8.ndim != 3:
        raise ValueError("boards must have shape (batch, L, L).")
    if rules_u32.shape != (boards_u8.shape[0],):
        raise ValueError("rules must have shape (batch,).")

    boards_i16 = boards_u8.astype(np.int16, copy=False)
    neighbors = np.zeros_like(boards_i16, dtype=np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += np.roll(boards_i16, shift=(-di, -dj), axis=(-2, -1))
    update_idx = boards_u8.astype(np.uint32) * 9 + neighbors.astype(np.uint32)
    return ((rules_u32[:, None, None] >> update_idx) & 1).astype(np.uint8)


def simulate_lifelike_rule_batch_numpy(
    initial_boards: np.ndarray,
    rules: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return trajectories for board/rule pairs with shape (batch, steps + 1, L, L)."""
    if initial_boards.ndim != 3:
        raise ValueError("initial_boards must have shape (batch, L, L).")
    if rules.shape != (initial_boards.shape[0],):
        raise ValueError("rules must have shape (batch,).")
    batch = int(initial_boards.shape[0])
    trajectory = np.empty((batch, steps + 1, *initial_boards.shape[1:]), dtype=np.uint8)
    trajectory[:, 0] = (initial_boards > 0).astype(np.uint8)
    state = trajectory[:, 0]
    for t in range(steps):
        state = life_step_lifelike_rule_batch_numpy(state, rules)
        trajectory[:, t + 1] = state
    return trajectory


_JAX_LIFELIKE_RULE_BATCH_SIMULATOR = None


def get_jax_lifelike_rule_batch_simulator():
    """Create/cache a jitted batched totalistic Life-like rollout."""
    global _JAX_LIFELIKE_RULE_BATCH_SIMULATOR
    if _JAX_LIFELIKE_RULE_BATCH_SIMULATOR is not None:
        return _JAX_LIFELIKE_RULE_BATCH_SIMULATOR

    try:
        import jax
        import jax.numpy as jnp
        from functools import partial
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "JAX backend requested, but jax/jaxlib could not be imported. "
            "Set backend='numpy' or install JAX."
        ) from exc

    def step_batch(state, rules):
        state_u8 = state.astype(jnp.uint8)
        state_i16 = state_u8.astype(jnp.int16)
        neighbors = jnp.zeros_like(state_i16)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbors = neighbors + jnp.roll(state_i16, shift=(-di, -dj), axis=(-2, -1))
        update_idx = state_u8.astype(jnp.uint32) * jnp.uint32(9) + neighbors.astype(jnp.uint32)
        return jnp.bitwise_and(jnp.right_shift(rules[:, None, None], update_idx), jnp.uint32(1)).astype(jnp.uint8)

    @partial(jax.jit, static_argnames=("steps",))
    def simulate(initial_boards, rules, steps: int):
        initial_boards = initial_boards.astype(jnp.uint8)
        rules = rules.astype(jnp.uint32)

        def scan_step(carry, _unused):
            next_state = step_batch(carry, rules)
            return next_state, next_state

        _final_state, frames = jax.lax.scan(scan_step, initial_boards, xs=None, length=steps)
        trajectory_t_first = jnp.concatenate([initial_boards[None, ...], frames], axis=0)
        return jnp.swapaxes(trajectory_t_first, 0, 1)

    _JAX_LIFELIKE_RULE_BATCH_SIMULATOR = simulate
    return simulate


def simulate_lifelike_rule_batch(
    initial_boards: np.ndarray,
    rules: np.ndarray,
    steps: int,
    backend: str,
) -> np.ndarray:
    """Return batched trajectories for arbitrary totalistic Life-like rules."""
    boards = (initial_boards > 0).astype(np.uint8, copy=False)
    rules_u32 = np.asarray(rules, dtype=np.uint32)
    if backend == "numpy":
        return simulate_lifelike_rule_batch_numpy(boards, rules_u32, steps)
    if backend == "jax":
        import jax.numpy as jnp

        simulator = get_jax_lifelike_rule_batch_simulator()
        trajectories = simulator(jnp.asarray(boards), jnp.asarray(rules_u32), steps)
        return np.asarray(trajectories.block_until_ready(), dtype=np.uint8)
    raise ValueError(f"Unknown backend: {backend}")


def transition_symbols(trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode A_t(i) = (3x3 neighborhood at t, x_{t+1}(i)) as 10-bit ints."""
    if trajectory.ndim != 3 or trajectory.shape[0] < 2:
        raise ValueError("trajectory must have shape (time >= 2, L, L).")
    states_t = trajectory[:-1].astype(np.uint16, copy=False)
    outputs = trajectory[1:].astype(np.uint16, copy=False)
    symbols = np.zeros_like(states_t, dtype=np.uint16)

    bit = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            neigh = np.roll(states_t, shift=(-di, -dj), axis=(1, 2))
            symbols |= neigh << bit
            bit += 1
    symbols |= outputs << 9
    neighborhood_nonzero = (symbols & NEIGHBORHOOD_MASK) != 0
    return symbols, neighborhood_nonzero


def histograms_from_codes(codes: np.ndarray) -> np.ndarray:
    """Build per-row categorical histograms over the 1024 transition symbols."""
    if codes.ndim != 2:
        raise ValueError("codes must have shape (n_cells, window_size).")
    n_cells, window_size = codes.shape
    hist = np.zeros((n_cells, N_TRANSITION_SYMBOLS), dtype=np.float32)
    if n_cells == 0 or window_size == 0:
        return hist
    rows = np.repeat(np.arange(n_cells), window_size)
    np.add.at(hist, (rows, codes.reshape(-1).astype(np.int64)), 1.0)
    hist /= float(window_size)
    return hist


def entropy_from_codes(codes: np.ndarray) -> np.ndarray:
    """Per-row entropy for empirical categorical samples without dense 1024 bins."""
    if codes.ndim != 2:
        raise ValueError("codes must have shape (n_rows, n_samples).")
    n_rows, n_samples = codes.shape
    if n_rows == 0 or n_samples == 0:
        return np.zeros(n_rows, dtype=np.float64)

    sorted_codes = np.sort(codes, axis=1)
    starts = np.empty_like(sorted_codes, dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = sorted_codes[:, 1:] != sorted_codes[:, :-1]
    group_ids = np.cumsum(starts, axis=1) - 1
    counts = np.zeros((n_rows, n_samples), dtype=np.float32)
    row_idx = np.repeat(np.arange(n_rows), n_samples)
    np.add.at(counts, (row_idx, group_ids.reshape(-1)), 1.0)
    probs = counts / float(n_samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0.0, probs * np.log(probs), 0.0)
    return -terms.sum(axis=1).astype(np.float64, copy=False)


def choose_pair_indices(
    n_items: int,
    rng: np.random.Generator,
    pair_sample: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    n_pairs = n_items * (n_items - 1) // 2
    if n_pairs <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    all_i, all_j = np.triu_indices(n_items, k=1)
    if pair_sample is None or pair_sample == 0 or pair_sample >= n_pairs:
        return all_i.astype(np.int64, copy=False), all_j.astype(np.int64, copy=False)

    chosen = rng.choice(n_pairs, size=int(pair_sample), replace=False)
    return all_i[chosen].astype(np.int64, copy=False), all_j[chosen].astype(np.int64, copy=False)


def mean_pairwise_js_from_codes(
    codes: np.ndarray,
    rng: np.random.Generator,
    pair_sample: Optional[int],
) -> float:
    """Mean Jensen-Shannon distance using sparse empirical transition samples."""
    n_cells, _window_size = codes.shape
    if n_cells < 2:
        return 0.0
    pair_i, pair_j = choose_pair_indices(n_cells, rng, pair_sample)
    if pair_i.size == 0:
        return 0.0

    row_entropy = entropy_from_codes(codes)
    mixed_codes = np.concatenate([codes[pair_i], codes[pair_j]], axis=1)
    mixed_entropy = entropy_from_codes(mixed_codes)
    js_div = mixed_entropy - 0.5 * (row_entropy[pair_i] + row_entropy[pair_j])
    return float(np.sqrt(np.maximum(js_div, 0.0)).mean())


def mean_pairwise_tv_from_codes(
    codes: np.ndarray,
    rng: np.random.Generator,
    pair_sample: Optional[int],
) -> float:
    """Mean total-variation distance for sampled pairs."""
    n_cells, _window_size = codes.shape
    if n_cells < 2:
        return 0.0
    pair_i, pair_j = choose_pair_indices(n_cells, rng, pair_sample)
    if pair_i.size == 0:
        return 0.0
    hist = histograms_from_codes(codes)
    values = 0.5 * np.abs(hist[pair_i] - hist[pair_j]).sum(axis=1)
    return float(values.mean())


def mean_pairwise_distance_from_codes(
    codes: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    pair_sample: Optional[int],
) -> float:
    """Mean pairwise law distance from per-cell transition symbols."""
    if metric == "js":
        return mean_pairwise_js_from_codes(codes, rng, pair_sample)
    if metric == "tv":
        return mean_pairwise_tv_from_codes(codes, rng, pair_sample)
    raise ValueError(f"Unknown distance metric: {metric}")


def mean_pairwise_distance(hist: np.ndarray, metric: str) -> float:
    """Mean pairwise Jensen-Shannon or total-variation distance."""
    n = hist.shape[0]
    if n < 2:
        return 0.0
    hist64 = hist.astype(np.float64, copy=False)
    if pdist is not None:
        if metric == "js":
            values = pdist(hist64, metric="jensenshannon")
        elif metric == "tv":
            values = 0.5 * pdist(hist64, metric="cityblock")
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        return float(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).mean())

    total = 0.0
    count = 0
    tiny = np.finfo(np.float64).tiny
    for i in range(n - 1):
        p = hist64[i : i + 1]
        q = hist64[i + 1 :]
        if metric == "tv":
            dist = 0.5 * np.abs(q - p).sum(axis=1)
        elif metric == "js":
            m = 0.5 * (p + q)
            kl_p = np.sum(np.where(p > 0, p * np.log(p / np.maximum(m, tiny)), 0.0), axis=1)
            kl_q = np.sum(np.where(q > 0, q * np.log(q / np.maximum(m, tiny)), 0.0), axis=1)
            dist = np.sqrt(np.maximum(0.0, 0.5 * (kl_p + kl_q)))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        total += float(dist.sum())
        count += int(dist.size)
    return total / max(count, 1)


def mspd_by_scale(
    h_trace: np.ndarray,
    mspd_floor: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute dyadic MSPD scale contrasts from a 1-D heterogeneity trace."""
    h = np.asarray(h_trace, dtype=np.float64)
    n_windows = int(h.size)
    scales: List[int] = []
    scores: List[float] = []
    r = 1
    while 2 * r <= n_windows:
        n_blocks_2r = n_windows // (2 * r)
        if n_blocks_2r <= 0:
            break
        usable = n_blocks_2r * 2 * r
        trimmed = h[:usable]
        g_r = trimmed.reshape(-1, r).mean(axis=1)
        g_2r = trimmed.reshape(-1, 2 * r).mean(axis=1)
        up_g_2r = np.repeat(g_2r, 2)
        denom = float(np.mean(g_r**2) + mspd_floor**2 + eps)
        d_r = float(np.mean((g_r - up_g_2r) ** 2) / denom)
        scales.append(r)
        scores.append(d_r)
        r *= 2
    scores_arr = np.asarray(scores, dtype=np.float64)
    mspd = float(scores_arr.mean()) if scores_arr.size else 0.0
    return np.asarray(scales, dtype=np.int64), scores_arr, mspd


def compute_transition_mspd(
    trajectory: np.ndarray,
    config: ExperimentConfig,
    metric_seed: int,
) -> EvaluationResult:
    """Compute transition-law heterogeneity and MSPD from saved states."""
    symbols, neighborhood_nonzero = transition_symbols(trajectory)
    symbols = symbols[config.burn_in :]
    neighborhood_nonzero = neighborhood_nonzero[config.burn_in :]
    n_transition_steps = int(symbols.shape[0])
    if n_transition_steps < config.window_size:
        raise ValueError("Not enough post-burn-in transition steps for one window.")

    rng = np.random.default_rng(metric_seed)
    starts = range(0, n_transition_steps - config.window_size + 1, config.window_step)
    h_raw: List[float] = []
    h_null: List[float] = []
    active_counts: List[int] = []
    active_available_counts: List[int] = []

    flat_symbols = symbols.reshape(n_transition_steps, -1)
    flat_active = neighborhood_nonzero.reshape(n_transition_steps, -1)

    for start in starts:
        stop = start + config.window_size
        window_active = flat_active[start:stop].any(axis=0)
        active_idx = np.flatnonzero(window_active)
        active_available_counts.append(int(active_idx.size))

        if active_idx.size > config.n_cell_sample:
            selected_idx = rng.choice(active_idx, size=config.n_cell_sample, replace=False)
        else:
            selected_idx = active_idx

        n_selected = int(selected_idx.size)
        active_counts.append(n_selected)
        if n_selected < 2:
            h_raw.append(0.0)
            h_null.append(0.0)
            continue

        cell_codes = flat_symbols[start:stop, selected_idx].T
        raw = mean_pairwise_distance_from_codes(
            cell_codes,
            metric=config.distance,
            rng=rng,
            pair_sample=config.pair_sample,
        )

        if config.pooled_null and config.null_reps > 0:
            pooled_codes = cell_codes.reshape(-1)
            null_values = []
            for _ in range(config.null_reps):
                pseudo_codes = rng.choice(
                    pooled_codes,
                    size=(n_selected, config.window_size),
                    replace=True,
                )
                null_values.append(
                    mean_pairwise_distance_from_codes(
                        pseudo_codes,
                        metric=config.distance,
                        rng=rng,
                        pair_sample=config.pair_sample,
                    )
                )
            null = float(np.median(null_values))
        else:
            null = 0.0

        h_raw.append(raw)
        h_null.append(null)

    h_raw_arr = np.asarray(h_raw, dtype=np.float64)
    h_null_arr = np.asarray(h_null, dtype=np.float64)
    delta_h = np.maximum(h_raw_arr - h_null_arr, 0.0)
    h_trace = np.maximum(delta_h - config.delta_h_floor, 0.0)
    scales, scale_scores, mspd = mspd_by_scale(h_trace, config.mspd_floor, config.eps)
    delta_h_nonzero_frac = float(np.mean(delta_h > config.delta_h_nonzero_eps)) if delta_h.size else 0.0
    passes_delta_h_filter = delta_h_nonzero_frac > config.min_delta_h_nonzero_frac
    fitness_score = mspd if passes_delta_h_filter else 0.0
    return EvaluationResult(
        mspd_score=mspd,
        fitness_score=float(fitness_score),
        delta_h_nonzero_frac=delta_h_nonzero_frac,
        passes_delta_h_filter=bool(passes_delta_h_filter),
        delta_h=delta_h,
        h_trace=h_trace,
        h_raw=h_raw_arr,
        h_null=h_null_arr,
        scales=scales,
        scale_scores=scale_scores,
        active_counts=np.asarray(active_counts, dtype=np.int64),
        active_available_counts=np.asarray(active_available_counts, dtype=np.int64),
        trajectory=trajectory,
    )


_JAX_TRANSITION_SCORER_CACHE: Dict[Tuple[Any, ...], Any] = {}
_JAX_TRANSITION_BATCH_SCORER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def supports_jax_transition_metric(config: ExperimentConfig) -> bool:
    return (
        config.backend == "jax"
        and config.distance == "js"
        and config.pair_sample is not None
        and config.pair_sample > 0
    )


def get_jax_transition_scorer(config: ExperimentConfig, trajectory_shape: Tuple[int, int, int]):
    """Create/cache a jitted JS transition-law heterogeneity scorer.

    This scorer accelerates the expensive window/pair/null part of the metric.
    It intentionally supports the practical sampled-pair JS path used for large
    sweeps; exact all-pairs and TV use the NumPy implementation.
    """
    if config.distance != "js" or config.pair_sample is None or config.pair_sample <= 0:
        raise ValueError("JAX transition scorer requires distance='js' and pair_sample > 0.")

    key = (
        tuple(int(x) for x in trajectory_shape),
        int(config.burn_in),
        int(config.window_size),
        int(config.window_step),
        int(config.n_cell_sample),
        int(config.null_reps),
        int(config.pair_sample),
    )
    if key in _JAX_TRANSITION_SCORER_CACHE:
        return _JAX_TRANSITION_SCORER_CACHE[key]

    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError("JAX transition scorer requested, but jax/jaxlib could not be imported.") from exc

    traj_len, grid_h, grid_w = (int(x) for x in trajectory_shape)
    n_cells = grid_h * grid_w
    n_transition_steps = traj_len - 1 - int(config.burn_in)
    n_windows = (n_transition_steps - int(config.window_size)) // int(config.window_step) + 1
    n_sample = min(int(config.n_cell_sample), n_cells)
    pair_sample = int(config.pair_sample)
    window_size = int(config.window_size)
    null_reps = int(config.null_reps)

    def entropy_from_code_rows(code_rows, sample_len):
        code_rows_i32 = code_rows.astype(jnp.int32)

        def row_counts(row):
            return jnp.bincount(row, length=N_TRANSITION_SYMBOLS)

        counts = jax.vmap(row_counts)(code_rows_i32).astype(jnp.float32)
        probs = counts / jnp.float32(sample_len)
        terms = jnp.where(probs > 0.0, probs * jnp.log(probs), 0.0)
        return -jnp.sum(terms, axis=-1)

    def js_mean_from_codes(codes, valid, pair_i, pair_j):
        pair_valid = valid[pair_i] & valid[pair_j]
        p = codes[pair_i]
        q = codes[pair_j]
        h_p = entropy_from_code_rows(p, window_size)
        h_q = entropy_from_code_rows(q, window_size)
        h_m = entropy_from_code_rows(jnp.concatenate([p, q], axis=1), 2 * window_size)
        js = jnp.sqrt(jnp.maximum(h_m - 0.5 * (h_p + h_q), 0.0))
        denom = jnp.maximum(jnp.sum(pair_valid), 1)
        return jnp.sum(jnp.where(pair_valid, js, 0.0)) / denom

    def transition_symbols_jax(trajectory):
        states_t = trajectory[:-1].astype(jnp.uint16)
        outputs = trajectory[1:].astype(jnp.uint16)
        symbols = jnp.zeros_like(states_t, dtype=jnp.uint16)
        bit = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neigh = jnp.roll(states_t, shift=(-di, -dj), axis=(1, 2))
                symbols = jnp.bitwise_or(symbols, jnp.left_shift(neigh, bit))
                bit += 1
        symbols = jnp.bitwise_or(symbols, jnp.left_shift(outputs, 9))
        neighborhood_nonzero = jnp.bitwise_and(symbols, NEIGHBORHOOD_MASK) != 0
        return symbols, neighborhood_nonzero

    @jax.jit
    def score_trajectory(trajectory, rng_key):
        symbols, neighborhood_nonzero = transition_symbols_jax(trajectory)
        symbols = symbols[int(config.burn_in) :]
        neighborhood_nonzero = neighborhood_nonzero[int(config.burn_in) :]
        flat_symbols = symbols.reshape((symbols.shape[0], n_cells))
        flat_active = neighborhood_nonzero.reshape((neighborhood_nonzero.shape[0], n_cells))

        key_order, key_pairs, key_windows = jax.random.split(rng_key, 3)
        cell_order = jax.random.permutation(key_order, n_cells)
        pair_i = jax.random.randint(key_pairs, (pair_sample,), minval=0, maxval=n_sample)
        pair_j_raw = jax.random.randint(key_pairs, (pair_sample,), minval=0, maxval=max(n_sample - 1, 1))
        pair_j = jnp.where(pair_j_raw >= pair_i, pair_j_raw + 1, pair_j_raw)
        pair_j = jnp.minimum(pair_j, n_sample - 1)
        starts = jnp.arange(n_windows, dtype=jnp.int32) * int(config.window_step)
        window_keys = jax.random.split(key_windows, n_windows)

        def one_window(start, window_key):
            symbols_win = jax.lax.dynamic_slice(flat_symbols, (start, 0), (window_size, n_cells))
            active_win = jax.lax.dynamic_slice(flat_active, (start, 0), (window_size, n_cells))
            active_mask = jnp.any(active_win, axis=0)
            active_available = jnp.sum(active_mask)

            active_ordered = active_mask[cell_order]
            order_rank = jnp.arange(n_cells)
            sort_keys = jnp.where(active_ordered, order_rank, n_cells + order_rank)
            selected = cell_order[jnp.argsort(sort_keys)[:n_sample]]
            valid = active_mask[selected]
            n_selected = jnp.sum(valid)

            codes = jnp.take(symbols_win, selected, axis=1).T
            raw = jnp.where(n_selected >= 2, js_mean_from_codes(codes, valid, pair_i, pair_j), 0.0)

            def one_null(rep_key):
                pool_count = jnp.maximum(n_selected * window_size, 1)
                uniform = jax.random.uniform(rep_key, (n_sample, window_size))
                sample_idx = jnp.floor(uniform * pool_count).astype(jnp.int32)
                pseudo_codes = codes.reshape((-1,))[sample_idx]
                pseudo_valid = jnp.arange(n_sample) < n_selected
                return js_mean_from_codes(pseudo_codes, pseudo_valid, pair_i, pair_j)

            if null_reps > 0:
                null_keys = jax.random.split(window_key, null_reps)
                null_values = jax.vmap(one_null)(null_keys)
                null = jnp.median(null_values)
            else:
                null = jnp.float32(0.0)
            return raw, null, n_selected, active_available

        raw, null, active_counts, active_available_counts = jax.vmap(one_window)(starts, window_keys)
        delta_h = jnp.maximum(raw - null, 0.0)
        return delta_h, raw, null, active_counts, active_available_counts

    _JAX_TRANSITION_SCORER_CACHE[key] = score_trajectory
    return score_trajectory


def get_jax_transition_batch_scorer(config: ExperimentConfig, trajectory_shape: Tuple[int, int, int]):
    """Create/cache jit(vmap(single_trajectory_transition_scorer))."""
    key = ("batch", tuple(int(x) for x in trajectory_shape), int(config.burn_in), int(config.window_size),
           int(config.window_step), int(config.n_cell_sample), int(config.null_reps), int(config.pair_sample or 0))
    if key in _JAX_TRANSITION_BATCH_SCORER_CACHE:
        return _JAX_TRANSITION_BATCH_SCORER_CACHE[key]

    import jax

    single_scorer = get_jax_transition_scorer(config, trajectory_shape)

    @jax.jit
    def score_batch(trajectories, rng_keys):
        return jax.vmap(single_scorer, in_axes=(0, 0))(trajectories, rng_keys)

    _JAX_TRANSITION_BATCH_SCORER_CACHE[key] = score_batch
    return score_batch


def evaluation_from_metric_arrays(
    trajectory: Optional[np.ndarray],
    config: ExperimentConfig,
    delta_h: np.ndarray,
    h_raw: np.ndarray,
    h_null: np.ndarray,
    active_counts: np.ndarray,
    active_available_counts: np.ndarray,
) -> EvaluationResult:
    delta_h_np = np.asarray(delta_h, dtype=np.float64)
    h_raw_np = np.asarray(h_raw, dtype=np.float64)
    h_null_np = np.asarray(h_null, dtype=np.float64)
    h_trace = np.maximum(delta_h_np - config.delta_h_floor, 0.0)
    scales, scale_scores, mspd = mspd_by_scale(h_trace, config.mspd_floor, config.eps)
    delta_h_nonzero_frac = float(np.mean(delta_h_np > config.delta_h_nonzero_eps)) if delta_h_np.size else 0.0
    passes_delta_h_filter = delta_h_nonzero_frac > config.min_delta_h_nonzero_frac
    fitness_score = mspd if passes_delta_h_filter else 0.0
    return EvaluationResult(
        mspd_score=mspd,
        fitness_score=float(fitness_score),
        delta_h_nonzero_frac=delta_h_nonzero_frac,
        passes_delta_h_filter=bool(passes_delta_h_filter),
        delta_h=delta_h_np,
        h_trace=h_trace,
        h_raw=h_raw_np,
        h_null=h_null_np,
        scales=scales,
        scale_scores=scale_scores,
        active_counts=np.asarray(active_counts, dtype=np.int64),
        active_available_counts=np.asarray(active_available_counts, dtype=np.int64),
        trajectory=trajectory,
    )


def compute_transition_mspd_jax(
    trajectory: np.ndarray,
    config: ExperimentConfig,
    metric_seed: int,
) -> EvaluationResult:
    import jax
    import jax.numpy as jnp

    scorer = get_jax_transition_scorer(config, tuple(int(x) for x in trajectory.shape))
    key = jax.random.PRNGKey(int(metric_seed) % (2**31 - 1))
    delta_h, h_raw, h_null, active_counts, active_available_counts = scorer(jnp.asarray(trajectory), key)
    return evaluation_from_metric_arrays(
        trajectory,
        config,
        np.asarray(delta_h.block_until_ready()),
        np.asarray(h_raw),
        np.asarray(h_null),
        np.asarray(active_counts),
        np.asarray(active_available_counts),
    )


def compute_transition_mspd_batch_auto(
    trajectories: np.ndarray,
    config: ExperimentConfig,
    metric_seeds: Sequence[int],
) -> List[EvaluationResult]:
    """Compute transition MSPD for a trajectory batch, using jit(vmap) when available."""
    if trajectories.ndim != 4:
        raise ValueError("trajectories must have shape (batch, time, L, L).")
    if len(metric_seeds) != int(trajectories.shape[0]):
        raise ValueError("metric_seeds length must match trajectory batch size.")

    if not supports_jax_transition_metric(config):
        return [
            compute_transition_mspd(trajectories[i], config, metric_seed=int(metric_seeds[i]))
            for i in range(int(trajectories.shape[0]))
        ]

    import jax
    import jax.numpy as jnp

    batch_size = int(trajectories.shape[0])
    metric_batch_size = batch_size if config.jax_metric_batch_size == 0 else min(config.jax_metric_batch_size, batch_size)
    results: List[EvaluationResult] = []
    scorer = get_jax_transition_batch_scorer(config, tuple(int(x) for x in trajectories.shape[1:]))
    for start in range(0, batch_size, metric_batch_size):
        stop = min(start + metric_batch_size, batch_size)
        keys = jnp.stack([jax.random.PRNGKey(int(seed) % (2**31 - 1)) for seed in metric_seeds[start:stop]])
        delta_h, h_raw, h_null, active_counts, active_available_counts = scorer(
            jnp.asarray(trajectories[start:stop]),
            keys,
        )
        delta_h_np = np.asarray(delta_h.block_until_ready())
        h_raw_np = np.asarray(h_raw)
        h_null_np = np.asarray(h_null)
        active_counts_np = np.asarray(active_counts)
        active_available_counts_np = np.asarray(active_available_counts)
        for local_idx in range(stop - start):
            results.append(
                evaluation_from_metric_arrays(
                    None,
                    config,
                    delta_h_np[local_idx],
                    h_raw_np[local_idx],
                    h_null_np[local_idx],
                    active_counts_np[local_idx],
                    active_available_counts_np[local_idx],
                )
            )
    return results


def compute_transition_mspd_auto(
    trajectory: np.ndarray,
    config: ExperimentConfig,
    metric_seed: int,
) -> EvaluationResult:
    if supports_jax_transition_metric(config):
        return compute_transition_mspd_jax(trajectory, config, metric_seed)
    return compute_transition_mspd(trajectory, config, metric_seed)


def evaluate_board(
    initial_board: np.ndarray,
    config: ExperimentConfig,
    metric_seed: int,
    keep_trajectory: bool = False,
) -> EvaluationResult:
    trajectory = simulate_life(initial_board, config.T)
    result = compute_transition_mspd_auto(trajectory, config, metric_seed=metric_seed)
    if not keep_trajectory:
        result.trajectory = None
    return result


def iter_evaluated_boards(
    boards: np.ndarray,
    config: ExperimentConfig,
    metric_seeds: Sequence[int],
) -> Iterable[Tuple[int, EvaluationResult, np.ndarray]]:
    """Evaluate boards in trajectory batches and yield per-board metric results."""
    if boards.ndim != 3:
        raise ValueError("boards must have shape (n_boards, L, L).")
    if len(metric_seeds) != int(boards.shape[0]):
        raise ValueError("metric_seeds length must match number of boards.")

    n_boards = int(boards.shape[0])
    if n_boards == 0:
        return
    batch_size = min(config.eval_batch_size, n_boards)
    for start in range(0, n_boards, batch_size):
        stop = min(start + batch_size, n_boards)
        trajectories = simulate_life_batch(boards[start:stop], config.T, backend=config.backend)
        batch_results = compute_transition_mspd_batch_auto(
            trajectories,
            config,
            metric_seeds[start:stop],
        )
        for offset, result in enumerate(batch_results):
            board_idx = start + offset
            result.trajectory = None
            yield board_idx, result, trajectories[offset]


def tournament_select(fitness: np.ndarray, rng: np.random.Generator, tournament_size: int) -> int:
    n = int(fitness.size)
    size = min(max(1, tournament_size), n)
    idx = rng.choice(n, size=size, replace=False)
    return int(idx[np.argmax(fitness[idx])])


def next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size = int(population.shape[0])
    elite_count = max(1, int(round(pop_size * config.elite_frac))) if config.elite_frac > 0 else 0
    elite_count = min(elite_count, pop_size)
    elite_idx = np.argsort(fitness)[-elite_count:][::-1] if elite_count else np.asarray([], dtype=int)

    children: List[np.ndarray] = [population[i].copy() for i in elite_idx]
    while len(children) < pop_size:
        p1 = tournament_select(fitness, rng, config.tournament_size)
        p2 = tournament_select(fitness, rng, config.tournament_size)
        mask = rng.random(population.shape[1:]) < 0.5
        child = np.where(mask, population[p1], population[p2]).astype(np.uint8)
        mutation_mask = rng.random(population.shape[1:]) < config.mutation_rate
        child ^= mutation_mask.astype(np.uint8)
        children.append(child)
    return np.stack(children, axis=0).astype(np.uint8)


def candidate_metric_seed(base_seed: int, generation: int, candidate_id: int) -> int:
    return int((base_seed + 1_000_003 * generation + 9_176 * candidate_id + 17) % (2**32 - 1))


def control_seed(base_seed: int, control_id: int) -> int:
    return int((base_seed + 500_000_003 + 104_729 * control_id) % (2**32 - 1))


def rule_metric_seed(base_seed: int, rule_position: int, init_id: int) -> int:
    return int((base_seed + 700_000_001 + 1_000_003 * rule_position + 9_176 * init_id) % (2**32 - 1))


def make_lifelike_rule_candidates(
    mode: str,
    n_rules: Optional[int],
    random_seed: int,
    include_conway: bool = True,
) -> np.ndarray:
    """Create totalistic Life-like rule IDs for ASAL-style rule sweeps."""
    if mode not in {"random", "all", "linspace"}:
        raise ValueError("rule candidate mode must be 'random', 'all', or 'linspace'.")

    if mode == "all":
        rules = np.arange(N_LIFELIKE_RULES, dtype=np.uint32)
    else:
        if n_rules is None or n_rules <= 0:
            raise ValueError("n_rules must be positive for random/linspace rule candidate modes.")
        n_rules = min(int(n_rules), N_LIFELIKE_RULES)
        if mode == "random":
            rng = np.random.default_rng(random_seed)
            rules = rng.choice(N_LIFELIKE_RULES, size=n_rules, replace=False).astype(np.uint32)
        else:
            rules = np.linspace(0, N_LIFELIKE_RULES - 1, n_rules, dtype=np.uint32)

    if include_conway and not bool(np.any(rules == CONWAY_LIFE_RULE)):
        rules = np.concatenate([np.asarray([CONWAY_LIFE_RULE], dtype=np.uint32), rules])
    return np.unique(rules).astype(np.uint32)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def ensure_matplotlib():
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "gol_transition_mspd_cache"))
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "gol_transition_mspd_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_outputs(
    output_dir: Path,
    generation_rows: Sequence[Dict[str, Any]],
    best_rows: Sequence[Dict[str, Any]],
    best_result: EvaluationResult,
    random_rows: Sequence[Dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    plt = ensure_matplotlib()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generations = np.asarray([int(row["generation"]) for row in best_rows], dtype=np.int64)
    best_scores = np.asarray([float(row["fitness"]) for row in best_rows], dtype=np.float64)
    mean_scores = []
    for gen in generations:
        scores = [float(row["fitness"]) for row in generation_rows if int(row["generation"]) == int(gen)]
        mean_scores.append(float(np.mean(scores)) if scores else np.nan)
    mean_scores_arr = np.asarray(mean_scores, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(generations, best_scores, label="best", linewidth=2)
    ax.plot(generations, mean_scores_arr, label="population mean", linewidth=1.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("MSPD score")
    ax.set_title("Best fitness over generations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "best_fitness_over_generations.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(best_result.delta_h.size)
    ax.plot(x, best_result.delta_h, label="DeltaH", linewidth=2)
    ax.plot(x, best_result.h_raw, label="H_raw", linewidth=1.2, alpha=0.8)
    ax.plot(x, best_result.h_null, label="H_null", linewidth=1.2, alpha=0.8)
    if config.delta_h_floor > 0:
        ax.plot(x, best_result.h_trace, label="h trace", linewidth=1.2)
    ax.set_xlabel("window")
    ax.set_ylabel("transition-law heterogeneity")
    ax.set_title("Best DeltaH trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "best_DeltaH_trace.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(int(scale)) for scale in best_result.scales]
    ax.bar(labels, best_result.scale_scores)
    ax.set_xlabel("dyadic scale r")
    ax.set_ylabel("d_r")
    ax.set_title("Best MSPD by scale")
    fig.tight_layout()
    fig.savefig(plots_dir / "best_mspd_by_scale.png", dpi=160)
    plt.close(fig)

    if best_result.trajectory is not None:
        n_frames = min(config.montage_frames, int(best_result.trajectory.shape[0]))
        frame_idx = np.linspace(0, best_result.trajectory.shape[0] - 1, n_frames, dtype=int)
        n_cols = min(4, n_frames)
        n_rows = int(math.ceil(n_frames / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, idx in zip(axes_arr, frame_idx):
            ax.imshow(best_result.trajectory[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"t={int(idx)}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes_arr[len(frame_idx) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "best_life_montage.png", dpi=160)
        plt.close(fig)

    random_scores = [float(row["fitness"]) for row in random_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [random_scores, [best_result.fitness_score]],
        tick_labels=["random controls", "optimized best"],
    )
    ax.scatter(np.ones(len(random_scores)), random_scores, s=16, alpha=0.7)
    ax.scatter([2], [best_result.fitness_score], s=48, color="tab:red", zorder=3)
    ax.set_ylabel("filtered MSPD score")
    ax.set_title("Optimized best vs random controls")
    fig.tight_layout()
    fig.savefig(plots_dir / "optimized_vs_random_score_boxplot.png", dpi=160)
    plt.close(fig)


def plot_rule_sweep_outputs(
    output_dir: Path,
    rule_rows: Sequence[Dict[str, Any]],
    best_result: EvaluationResult,
    best_trajectory: np.ndarray,
    top_n: int = 20,
) -> None:
    plt = ensure_matplotlib()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rule_rows, key=lambda row: float(row["mean_mspd"]), reverse=True)
    top_rows = sorted_rows[: min(top_n, len(sorted_rows))]
    labels = [f"{row['rule_id']}\n{row['rule_label']}" for row in top_rows]
    scores = [float(row["mean_mspd"]) for row in top_rows]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.5))
    ax.bar(np.arange(len(labels)), scores)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("mean MSPD")
    ax.set_title("Top Life-like rule candidates")
    fig.tight_layout()
    fig.savefig(plots_dir / "rule_sweep_top_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(best_result.delta_h.size)
    ax.plot(x, best_result.delta_h, label="DeltaH", linewidth=2)
    ax.plot(x, best_result.h_raw, label="H_raw", linewidth=1.2, alpha=0.8)
    ax.plot(x, best_result.h_null, label="H_null", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("window")
    ax.set_ylabel("transition-law heterogeneity")
    ax.set_title("Best rule DeltaH trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "rule_sweep_best_DeltaH_trace.png", dpi=160)
    plt.close(fig)

    n_frames = min(16, int(best_trajectory.shape[0]))
    frame_idx = np.linspace(0, best_trajectory.shape[0] - 1, n_frames, dtype=int)
    n_cols = min(4, n_frames)
    n_rows = int(math.ceil(n_frames / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes_arr, frame_idx):
        ax.imshow(best_trajectory[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"t={int(idx)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(frame_idx) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "rule_sweep_best_life_montage.png", dpi=160)
    plt.close(fig)


def render_life_rgb(frame: np.ndarray, scale: int) -> np.ndarray:
    img = (frame.astype(np.uint8) * 255)
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def write_life_video(
    trajectory: np.ndarray,
    path: Path,
    fps: int,
    scale: int,
    stride: int,
) -> Path:
    """Write an MP4 video, falling back to GIF if OpenCV is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = trajectory[::stride]
    rgb_frames = [render_life_rgb(frame, scale) for frame in frames]
    if not rgb_frames:
        raise ValueError("No frames available for video.")

    try:
        import cv2

        h, w, _ = rgb_frames[0].shape
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if writer.isOpened():
            for frame in rgb_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            if path.exists() and path.stat().st_size > 0:
                return path
        writer.release()
    except Exception:
        pass

    gif_path = path.with_suffix(".gif")
    from PIL import Image

    pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
    )
    return gif_path


def make_videos(
    output_dir: Path,
    config: ExperimentConfig,
    best_result: EvaluationResult,
    final_population: np.ndarray,
    final_fitness: np.ndarray,
    random_boards: Sequence[np.ndarray],
    random_scores: Sequence[float],
) -> None:
    if not config.save_videos:
        return
    videos_dir = output_dir / "videos"
    rows: List[Dict[str, Any]] = []

    def add_video(label: str, trajectory: np.ndarray, score: float, source: str) -> None:
        video_path = write_life_video(
            trajectory,
            videos_dir / f"{label}.mp4",
            fps=config.video_fps,
            scale=config.video_scale,
            stride=config.video_stride,
        )
        rows.append({"label": label, "source": source, "score": score, "path": str(video_path)})

    if best_result.trajectory is not None:
        add_video("best_complex_life", best_result.trajectory, best_result.fitness_score, "global_best")

    top_k = min(config.video_top_k, int(final_population.shape[0]))
    if top_k > 0:
        top_idx = np.argsort(final_fitness)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            traj = simulate_life(final_population[int(idx)], config.T)
            add_video(
                f"final_population_complex_{rank:02d}",
                traj,
                float(final_fitness[int(idx)]),
                "final_population_top",
            )

    if final_population.shape[0] > 0:
        median_score = float(np.median(final_fitness))
        typical_idx = int(np.argmin(np.abs(final_fitness - median_score)))
        traj = simulate_life(final_population[typical_idx], config.T)
        add_video(
            "final_population_typical_life",
            traj,
            float(final_fitness[typical_idx]),
            "final_population_median",
        )

    if random_boards:
        scores = np.asarray(random_scores, dtype=np.float64)
        typical_idx = int(np.argmin(np.abs(scores - np.median(scores))))
        traj = simulate_life(random_boards[typical_idx], config.T)
        add_video(
            "random_typical_life",
            traj,
            float(scores[typical_idx]),
            "random_control_median",
        )

    write_csv(videos_dir / "video_manifest.csv", rows, ["label", "source", "score", "path"])


def save_best_npz(output_dir: Path, best_board: np.ndarray, best_result: EvaluationResult) -> None:
    if best_result.trajectory is None:
        raise ValueError("best_result.trajectory is required when saving best_result.npz.")
    else:
        trajectory = best_result.trajectory
    np.savez_compressed(
        output_dir / "best_result.npz",
        initial_board=best_board.astype(np.uint8),
        trajectory=trajectory.astype(np.uint8),
        DeltaH=best_result.delta_h,
        delta_h=best_result.delta_h,
        h_trace=best_result.h_trace,
        H_raw=best_result.h_raw,
        H_null=best_result.h_null,
        active_counts=best_result.active_counts,
        active_available_counts=best_result.active_available_counts,
        mspd_scales=best_result.scales,
        mspd_scale_scores=best_result.scale_scores,
        mspd_score=np.asarray(best_result.fitness_score, dtype=np.float64),
        fitness_score=np.asarray(best_result.fitness_score, dtype=np.float64),
        raw_mspd_score=np.asarray(best_result.mspd_score, dtype=np.float64),
        delta_h_nonzero_frac=np.asarray(best_result.delta_h_nonzero_frac, dtype=np.float64),
        passes_delta_h_filter=np.asarray(best_result.passes_delta_h_filter, dtype=bool),
    )


def run_rule_sweep_experiment(
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
    rule_candidates: Optional[Sequence[int]] = None,
    rule_candidate_mode: str = "random",
    n_rule_candidates: Optional[int] = 512,
    n_initial_boards: int = 4,
    initial_density: Optional[float] = None,
    initial_density_range: Optional[Tuple[float, float]] = None,
    include_conway: bool = True,
    stream_per_init_csv: bool = True,
    progress_interval_rules: int = 64,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate totalistic Life-like rules, matching the ASAL rule-search setup.

    Each rule candidate is evaluated on the same set of Bernoulli initial boards.
    The score is the mean transition-law MSPD over those starts.
    """
    validate_config(config)
    if n_initial_boards <= 0:
        raise ValueError("n_initial_boards must be positive.")
    if progress_interval_rules <= 0:
        raise ValueError("progress_interval_rules must be positive.")
    verbose_flag = config.verbose if verbose is None else bool(verbose)
    density = config.initial_density if initial_density is None else float(initial_density)
    if not (0.0 <= density <= 1.0):
        raise ValueError("initial_density must be in [0, 1].")
    density_low: Optional[float] = None
    density_high: Optional[float] = None
    if initial_density_range is not None:
        if len(initial_density_range) != 2:
            raise ValueError("initial_density_range must be a pair: (low, high).")
        density_low = float(initial_density_range[0])
        density_high = float(initial_density_range[1])
        if not (0.0 <= density_low <= density_high <= 1.0):
            raise ValueError("initial_density_range must satisfy 0 <= low <= high <= 1.")

    if rule_candidates is None:
        rules = make_lifelike_rule_candidates(
            rule_candidate_mode,
            n_rule_candidates,
            random_seed=config.random_seed + 991,
            include_conway=include_conway,
        )
    else:
        rules = np.asarray(rule_candidates, dtype=np.uint32)
        if include_conway and not bool(np.any(rules == CONWAY_LIFE_RULE)):
            rules = np.concatenate([np.asarray([CONWAY_LIFE_RULE], dtype=np.uint32), rules])
        rules = np.unique(rules).astype(np.uint32)

    if rules.size == 0:
        raise ValueError("No rule candidates to evaluate.")
    if np.any(rules >= N_LIFELIKE_RULES):
        raise ValueError(f"Life-like rule IDs must be in [0, {N_LIFELIKE_RULES}).")

    rng = np.random.default_rng(config.random_seed + 883)
    if density_low is None or density_high is None:
        initial_probabilities = np.full(n_initial_boards, density, dtype=np.float64)
        density_mode = "fixed"
    else:
        initial_probabilities = rng.uniform(density_low, density_high, size=n_initial_boards).astype(np.float64)
        density_mode = "uniform_per_board"
    initial_boards = (
        rng.random((n_initial_boards, config.L, config.L)) < initial_probabilities[:, None, None]
    ).astype(np.uint8)
    realized_densities = initial_boards.reshape(n_initial_boards, -1).mean(axis=1).astype(np.float64)

    rule_output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir) / "rule_sweep"
    rule_output_dir.mkdir(parents=True, exist_ok=True)
    with (rule_output_dir / "rule_sweep_config.json").open("w") as f:
        json.dump(
            {
                "base_config": asdict(config),
                "rule_candidate_mode": rule_candidate_mode,
                "n_rule_candidates": None if n_rule_candidates is None else int(n_rule_candidates),
                "n_initial_boards": int(n_initial_boards),
                "initial_density": density,
                "initial_density_mode": density_mode,
                "initial_density_range": None
                if density_low is None or density_high is None
                else [density_low, density_high],
                "initial_board_probabilities": initial_probabilities.tolist(),
                "initial_board_realized_densities": realized_densities.tolist(),
                "include_conway": include_conway,
                "stream_per_init_csv": stream_per_init_csv,
                "progress_interval_rules": int(progress_interval_rules),
                "n_rules_evaluated": int(rules.size),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    log_verbose(
        "Starting ASAL-style Life-like rule sweep: "
        f"rules={rules.size}, n_initial_boards={n_initial_boards}, "
        f"initialization={density_mode}, p_min={initial_probabilities.min():.4f}, "
        f"p_max={initial_probabilities.max():.4f}, "
        f"L={config.L}, T={config.T}, backend={config.backend}, "
        f"eval_batch_size={config.eval_batch_size}, pair_sample={config.pair_sample}, "
        f"min_delta_h_nonzero_frac>{config.min_delta_h_nonzero_frac}, "
        f"output_dir={rule_output_dir}",
        verbose_flag,
    )

    rules_per_batch = max(1, config.eval_batch_size // n_initial_boards)
    rule_rows: List[Dict[str, Any]] = []
    per_init_rows: List[Dict[str, Any]] = []
    per_init_fieldnames = [
        "rule_position",
        "rule_id",
        "rule_label",
        "init_id",
        "seed",
        "initial_p",
        "initial_alive_fraction",
        "mspd_score",
        "raw_mspd_score",
        "delta_h_nonzero_frac",
        "passes_delta_h_filter",
    ]
    per_init_file = None
    per_init_writer = None
    if stream_per_init_csv:
        per_init_file = (rule_output_dir / "rule_sweep_per_init_scores.csv").open("w", newline="")
        per_init_writer = csv.DictWriter(per_init_file, fieldnames=per_init_fieldnames)
        per_init_writer.writeheader()
    best_mean_score = -np.inf
    best_rule_id: Optional[int] = None
    best_result: Optional[EvaluationResult] = None
    best_trajectory: Optional[np.ndarray] = None
    best_initial_board: Optional[np.ndarray] = None
    progress_interval = int(progress_interval_rules)
    next_progress = progress_interval
    sweep_t0 = time.time()

    try:
        for batch_start in range(0, int(rules.size), rules_per_batch):
            batch_stop = min(batch_start + rules_per_batch, int(rules.size))
            rule_chunk = rules[batch_start:batch_stop]
            boards_batch = np.concatenate([initial_boards for _ in rule_chunk], axis=0)
            rules_batch = np.repeat(rule_chunk, n_initial_boards).astype(np.uint32)
            trajectories = simulate_lifelike_rule_batch(
                boards_batch,
                rules_batch,
                config.T,
                backend=config.backend,
            )
            metric_seeds = [
                rule_metric_seed(config.random_seed, batch_start + flat_idx // n_initial_boards, flat_idx % n_initial_boards)
                for flat_idx in range(int(trajectories.shape[0]))
            ]
            batch_results = compute_transition_mspd_batch_auto(trajectories, config, metric_seeds)

            for local_rule_idx, rule_id_u32 in enumerate(rule_chunk):
                rule_position = batch_start + local_rule_idx
                rule_id = int(rule_id_u32)
                scores: List[float] = []
                raw_scores: List[float] = []
                init_results: List[Tuple[int, EvaluationResult, np.ndarray]] = []
                for init_id in range(n_initial_boards):
                    flat_idx = local_rule_idx * n_initial_boards + init_id
                    seed = rule_metric_seed(config.random_seed, rule_position, init_id)
                    result = batch_results[flat_idx]
                    result.trajectory = None
                    scores.append(float(result.fitness_score))
                    raw_scores.append(float(result.mspd_score))
                    init_results.append((init_id, result, trajectories[flat_idx]))
                    per_init_row = {
                        "rule_position": rule_position,
                        "rule_id": rule_id,
                        "rule_label": lifelike_rule_label(rule_id),
                        "init_id": init_id,
                        "seed": seed,
                        "initial_p": float(initial_probabilities[init_id]),
                        "initial_alive_fraction": float(realized_densities[init_id]),
                        "mspd_score": float(result.fitness_score),
                        "raw_mspd_score": float(result.mspd_score),
                        "delta_h_nonzero_frac": float(result.delta_h_nonzero_frac),
                        "passes_delta_h_filter": int(result.passes_delta_h_filter),
                    }
                    if per_init_writer is not None:
                        per_init_writer.writerow(per_init_row)
                    else:
                        per_init_rows.append(per_init_row)

                scores_arr = np.asarray(scores, dtype=np.float64)
                raw_scores_arr = np.asarray(raw_scores, dtype=np.float64)
                mean_score = float(scores_arr.mean())
                row = {
                    "rule_position": rule_position,
                    "rule_id": rule_id,
                    "rule_label": lifelike_rule_label(rule_id),
                    "mean_mspd": mean_score,
                    "std_mspd": float(scores_arr.std(ddof=0)),
                    "min_mspd": float(scores_arr.min()),
                    "max_mspd": float(scores_arr.max()),
                    "mean_raw_mspd": float(raw_scores_arr.mean()),
                    "std_raw_mspd": float(raw_scores_arr.std(ddof=0)),
                    "pass_fraction": float(np.mean(scores_arr > 0.0)),
                    "n_initial_boards": n_initial_boards,
                }
                rule_rows.append(row)

                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_rule_id = rule_id
                    best_init_id, best_init_result, best_init_trajectory = max(
                        init_results,
                        key=lambda item: (float(item[1].fitness_score), float(item[1].mspd_score)),
                    )
                    best_init_result.trajectory = best_init_trajectory.copy()
                    best_result = best_init_result
                    best_trajectory = best_init_trajectory.copy()
                    best_initial_board = initial_boards[best_init_id].copy()
                    log_verbose(
                        f"  new best rule: position={rule_position} rule={rule_id} "
                        f"{lifelike_rule_label(rule_id)} mean_filtered_MSPD={best_mean_score:.6f}",
                        verbose_flag,
                    )

                rules_done = rule_position + 1
                if verbose_flag and (rules_done >= next_progress or rules_done == int(rules.size)):
                    elapsed = time.time() - sweep_t0
                    rate = rules_done / max(elapsed, 1e-9)
                    remaining = (int(rules.size) - rules_done) / max(rate, 1e-9)
                    log_verbose(
                        f"evaluated {rules_done}/{rules.size} rules "
                        f"({rules_done / int(rules.size):.1%}) | "
                        f"{rate:.2f} rules/s | eta {remaining / 3600:.2f}h | "
                        f"best={best_mean_score:.6f} rule={best_rule_id}",
                        verbose_flag,
                    )
                    while next_progress <= rules_done:
                        next_progress += progress_interval

    finally:
        if per_init_file is not None:
            per_init_file.close()

    if best_rule_id is None or best_result is None or best_trajectory is None or best_initial_board is None:
        raise RuntimeError("Rule sweep produced no evaluated rule.")

    sorted_rows = sorted(rule_rows, key=lambda row: float(row["mean_mspd"]), reverse=True)
    for rank, row in enumerate(sorted_rows, start=1):
        row["rank"] = rank

    write_csv(
        rule_output_dir / "rule_sweep_scores.csv",
        sorted_rows,
        [
            "rank",
            "rule_position",
            "rule_id",
            "rule_label",
            "mean_mspd",
            "std_mspd",
            "min_mspd",
            "max_mspd",
            "mean_raw_mspd",
            "std_raw_mspd",
            "pass_fraction",
            "n_initial_boards",
        ],
    )
    if not stream_per_init_csv:
        write_csv(
            rule_output_dir / "rule_sweep_per_init_scores.csv",
            per_init_rows,
            per_init_fieldnames,
        )
    np.savez_compressed(
        rule_output_dir / "best_rule_result.npz",
        rule_id=np.asarray(best_rule_id, dtype=np.uint32),
        rule_label=np.asarray(lifelike_rule_label(best_rule_id)),
        initial_board=best_initial_board.astype(np.uint8),
        initial_boards=initial_boards.astype(np.uint8),
        initial_board_probabilities=initial_probabilities.astype(np.float64),
        initial_board_realized_densities=realized_densities.astype(np.float64),
        trajectory=best_trajectory.astype(np.uint8),
        DeltaH=best_result.delta_h,
        delta_h=best_result.delta_h,
        h_trace=best_result.h_trace,
        H_raw=best_result.h_raw,
        H_null=best_result.h_null,
        active_counts=best_result.active_counts,
        active_available_counts=best_result.active_available_counts,
        mspd_scales=best_result.scales,
        mspd_scale_scores=best_result.scale_scores,
        best_init_mspd_score=np.asarray(best_result.fitness_score, dtype=np.float64),
        best_init_raw_mspd_score=np.asarray(best_result.mspd_score, dtype=np.float64),
        best_rule_mean_mspd_score=np.asarray(best_mean_score, dtype=np.float64),
        delta_h_nonzero_frac=np.asarray(best_result.delta_h_nonzero_frac, dtype=np.float64),
        passes_delta_h_filter=np.asarray(best_result.passes_delta_h_filter, dtype=bool),
    )
    plot_rule_sweep_outputs(rule_output_dir, sorted_rows, best_result, best_trajectory)

    summary = {
        "output_dir": str(rule_output_dir),
        "n_rules_evaluated": int(rules.size),
        "n_initial_boards": int(n_initial_boards),
        "initial_density_mode": density_mode,
        "initial_density_range": None
        if density_low is None or density_high is None
        else [density_low, density_high],
        "best_rule_id": int(best_rule_id),
        "best_rule_label": lifelike_rule_label(best_rule_id),
        "best_rule_mean_mspd_score": float(best_mean_score),
        "best_init_mspd_score": float(best_result.fitness_score),
        "best_init_raw_mspd_score": float(best_result.mspd_score),
        "best_init_delta_h_nonzero_frac": float(best_result.delta_h_nonzero_frac),
    }
    with (rule_output_dir / "rule_sweep_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log_verbose(
        f"Done rule sweep. Best rule={summary['best_rule_id']} "
        f"{summary['best_rule_label']} mean_filtered_MSPD={best_mean_score:.6f}",
        verbose_flag,
    )
    return {
        "config": config,
        "summary": summary,
        "rule_rows": sorted_rows,
        "per_init_rows": per_init_rows,
        "rules": rules,
        "initial_boards": initial_boards,
        "initial_board_probabilities": initial_probabilities,
        "initial_board_realized_densities": realized_densities,
        "best_result": best_result,
        "best_rule_id": best_rule_id,
        "best_rule_label": lifelike_rule_label(best_rule_id),
        "best_trajectory": best_trajectory,
    }


def run_experiment(config: ExperimentConfig, verbose: Optional[bool] = None) -> Dict[str, Any]:
    validate_config(config)
    verbose_flag = config.verbose if verbose is None else bool(verbose)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    log_verbose(
        "Starting GoL transition-law MSPD experiment: "
        f"L={config.L}, T={config.T}, burn_in={config.burn_in}, "
        f"windows={config.window_size}/{config.window_step}, "
        f"population={config.population_size}, generations={config.generations}, "
        f"backend={config.backend}, eval_batch_size={config.eval_batch_size}, "
        f"distance={config.distance}, pooled_null={config.pooled_null}, "
        f"pair_sample={config.pair_sample}, "
        f"min_delta_h_nonzero_frac>{config.min_delta_h_nonzero_frac}, "
        f"output_dir={output_dir}",
        verbose_flag,
    )

    rng = np.random.default_rng(config.random_seed)
    population = (rng.random((config.population_size, config.L, config.L)) < config.initial_density).astype(
        np.uint8
    )

    generation_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    global_best_score = -np.inf
    global_best_board: Optional[np.ndarray] = None
    global_best_result: Optional[EvaluationResult] = None
    final_fitness = np.zeros(config.population_size, dtype=np.float64)

    for generation in range(config.generations):
        log_verbose(f"Evaluating generation {generation + 1}/{config.generations}...", verbose_flag)
        fitness = np.zeros(config.population_size, dtype=np.float64)
        metric_seeds = [
            candidate_metric_seed(config.random_seed, generation, candidate_id)
            for candidate_id in range(config.population_size)
        ]
        for candidate_id, result, trajectory in iter_evaluated_boards(population, config, metric_seeds):
            seed = metric_seeds[candidate_id]
            fitness[candidate_id] = result.fitness_score
            generation_rows.append(
                {
                    "generation": generation,
                    "candidate_id": candidate_id,
                    "fitness": result.fitness_score,
                    "raw_mspd_score": result.mspd_score,
                    "delta_h_nonzero_frac": result.delta_h_nonzero_frac,
                    "passes_delta_h_filter": int(result.passes_delta_h_filter),
                    "seed": seed,
                }
            )
            if result.fitness_score > global_best_score:
                global_best_score = float(result.fitness_score)
                global_best_board = population[candidate_id].copy()
                result.trajectory = trajectory.copy()
                global_best_result = result
                log_verbose(
                    f"  new global best: generation={generation} candidate={candidate_id} "
                    f"filtered_MSPD={global_best_score:.6f} raw_MSPD={result.mspd_score:.6f} "
                    f"DeltaH_nonzero_frac={result.delta_h_nonzero_frac:.3f}",
                    verbose_flag,
                )

        best_idx = int(np.argmax(fitness))
        best_seed = candidate_metric_seed(config.random_seed, generation, best_idx)
        best_rows.append(
            {
                "generation": generation,
                "candidate_id": best_idx,
                "fitness": float(fitness[best_idx]),
                "raw_mspd_score": generation_rows[-config.population_size + best_idx]["raw_mspd_score"],
                "delta_h_nonzero_frac": generation_rows[-config.population_size + best_idx]["delta_h_nonzero_frac"],
                "passes_delta_h_filter": generation_rows[-config.population_size + best_idx]["passes_delta_h_filter"],
                "seed": best_seed,
            }
        )
        final_fitness = fitness.copy()
        should_log_generation = (
            (generation + 1) % config.progress_every == 0
            or generation == 0
            or generation == config.generations - 1
        )
        if should_log_generation:
            log_verbose(
                f"generation {generation + 1}/{config.generations}: "
                f"best={fitness[best_idx]:.6f} mean={fitness.mean():.6f} "
                f"median={np.median(fitness):.6f} global_best={global_best_score:.6f}",
                verbose_flag,
            )
        if generation < config.generations - 1:
            population = next_generation(population, fitness, config, rng)

    if global_best_board is None or global_best_result is None:
        raise RuntimeError("GA produced no evaluated candidate.")

    n_controls = config.random_controls if config.random_controls is not None else config.population_size
    log_verbose(f"Evaluating {n_controls} random controls...", verbose_flag)
    random_rows: List[Dict[str, Any]] = []
    random_boards: List[np.ndarray] = []
    random_scores: List[float] = []
    random_metric_seeds: List[int] = []
    for control_id in range(n_controls):
        seed = control_seed(config.random_seed, control_id)
        control_rng = np.random.default_rng(seed)
        board = (control_rng.random((config.L, config.L)) < config.initial_density).astype(np.uint8)
        random_boards.append(board)
        random_metric_seeds.append(seed + 17)

    random_boards_arr = np.stack(random_boards, axis=0) if random_boards else np.empty((0, config.L, config.L), dtype=np.uint8)
    for control_id, result, _trajectory in iter_evaluated_boards(random_boards_arr, config, random_metric_seeds):
        seed = control_seed(config.random_seed, control_id)
        random_scores.append(float(result.fitness_score))
        random_rows.append(
            {
                "control_id": control_id,
                "fitness": result.fitness_score,
                "raw_mspd_score": result.mspd_score,
                "delta_h_nonzero_frac": result.delta_h_nonzero_frac,
                "passes_delta_h_filter": int(result.passes_delta_h_filter),
                "seed": seed,
            }
        )

    log_verbose("Writing CSV/NPZ artifacts...", verbose_flag)
    write_csv(
        output_dir / "generation_scores.csv",
        generation_rows,
        [
            "generation",
            "candidate_id",
            "fitness",
            "raw_mspd_score",
            "delta_h_nonzero_frac",
            "passes_delta_h_filter",
            "seed",
        ],
    )
    write_csv(
        output_dir / "best_per_generation.csv",
        best_rows,
        [
            "generation",
            "candidate_id",
            "fitness",
            "raw_mspd_score",
            "delta_h_nonzero_frac",
            "passes_delta_h_filter",
            "seed",
        ],
    )
    write_csv(
        output_dir / "random_control_scores.csv",
        random_rows,
        [
            "control_id",
            "fitness",
            "raw_mspd_score",
            "delta_h_nonzero_frac",
            "passes_delta_h_filter",
            "seed",
        ],
    )
    save_best_npz(output_dir, global_best_board, global_best_result)
    log_verbose("Rendering plots...", verbose_flag)
    plot_outputs(output_dir, generation_rows, best_rows, global_best_result, random_rows, config)
    log_verbose("Rendering videos..." if config.save_videos else "Skipping videos.", verbose_flag)
    make_videos(output_dir, config, global_best_result, population, final_fitness, random_boards, random_scores)

    summary = {
        "output_dir": str(output_dir),
        "best_mspd_score": float(global_best_result.fitness_score),
        "best_raw_mspd_score": float(global_best_result.mspd_score),
        "best_delta_h_nonzero_frac": float(global_best_result.delta_h_nonzero_frac),
        "best_passes_delta_h_filter": bool(global_best_result.passes_delta_h_filter),
        "best_generation": int(max(best_rows, key=lambda row: float(row["fitness"]))["generation"]),
        "final_generation_pass_fraction": float(np.mean([row["passes_delta_h_filter"] for row in generation_rows[-config.population_size:]]))
        if generation_rows
        else None,
        "random_control_pass_fraction": float(np.mean([row["passes_delta_h_filter"] for row in random_rows]))
        if random_rows
        else None,
        "random_control_mean": float(np.mean(random_scores)) if random_scores else None,
        "random_control_median": float(np.median(random_scores)) if random_scores else None,
        "random_control_max": float(np.max(random_scores)) if random_scores else None,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log_verbose(
        f"Done. Best filtered_MSPD={global_best_result.fitness_score:.6f} "
        f"raw_MSPD={global_best_result.mspd_score:.6f} "
        f"DeltaH_nonzero_frac={global_best_result.delta_h_nonzero_frac:.3f}",
        verbose_flag,
    )
    return {
        "config": config,
        "summary": summary,
        "best_result": global_best_result,
        "best_board": global_best_board,
        "final_population": population,
        "final_fitness": final_fitness,
        "random_boards": np.asarray(random_boards, dtype=np.uint8),
        "random_scores": np.asarray(random_scores, dtype=np.float64),
    }


def load_config_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object.")
    return data


def configure_jax_platform(platform_name: Optional[str]) -> None:
    """Set JAX platform before any lazy JAX import happens."""
    if platform_name:
        os.environ.setdefault("JAX_PLATFORM_NAME", platform_name)


def jax_device_summary(require_accelerator: bool = False) -> Dict[str, Any]:
    """Return JAX backend/device info and optionally fail if only CPU is visible."""
    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    device_rows = [
        {
            "id": idx,
            "platform": getattr(device, "platform", ""),
            "device_kind": getattr(device, "device_kind", str(device)),
            "repr": str(device),
        }
        for idx, device in enumerate(devices)
    ]
    has_accelerator = any(str(row["platform"]).lower() not in {"cpu"} for row in device_rows)
    if require_accelerator and not has_accelerator:
        raise RuntimeError(
            "JAX sees only CPU devices. Install/configure CUDA or jax-metal, or rerun without "
            "--require-accelerator. Visible devices: "
            + json.dumps(device_rows)
        )
    return {"backend": backend, "devices": device_rows, "has_accelerator": has_accelerator}


def parser_from_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["ga", "rule-sweep"], default="ga")
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny local sanity-check configuration.")
    parser.add_argument("--jax-platform", type=str, default=None, help="Set JAX_PLATFORM_NAME before importing JAX.")
    parser.add_argument("--require-accelerator", action="store_true", help="Fail if JAX sees only CPU devices.")
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--n-cell-sample", type=int, default=None)
    parser.add_argument("--null-reps", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--elite-frac", type=float, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--initial-density", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--backend", choices=["jax", "numpy"], default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--jax-metric-batch-size", type=int, default=None, help="0 means full rollout batch.")
    parser.add_argument("--distance", choices=["js", "tv"], default=None)
    parser.add_argument("--pair-sample", type=int, default=None, help="0 means exact all-pairs.")
    parser.add_argument("--pooled-null", dest="pooled_null", action="store_true", default=None)
    parser.add_argument("--no-pooled-null", dest="pooled_null", action="store_false")
    parser.add_argument("--delta-h-floor", type=float, default=None)
    parser.add_argument("--min-delta-h-nonzero-frac", type=float, default=None)
    parser.add_argument("--delta-h-nonzero-eps", type=float, default=None)
    parser.add_argument("--mspd-floor", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--tournament-size", type=int, default=None)
    parser.add_argument("--random-controls", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-videos", dest="save_videos", action="store_true", default=None)
    parser.add_argument("--no-videos", dest="save_videos", action="store_false")
    parser.add_argument("--video-top-k", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--video-scale", type=int, default=None)
    parser.add_argument("--video-stride", type=int, default=None)
    parser.add_argument("--montage-frames", type=int, default=None)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.add_argument("--progress-every", type=int, default=None)
    rule_group = parser.add_argument_group("rule sweep")
    rule_group.add_argument("--rule-candidate-mode", choices=["random", "linspace", "all"], default="random")
    rule_group.add_argument("--all-rules", action="store_true", help="Evaluate all 262144 totalistic Life-like rules.")
    rule_group.add_argument("--n-rule-candidates", type=int, default=512)
    rule_group.add_argument("--n-rule-initial-boards", type=int, default=4)
    rule_group.add_argument(
        "--rule-initial-density-min",
        type=float,
        default=None,
        help="If set with --rule-initial-density-max, sample each rule-sweep start p ~ Uniform(min, max).",
    )
    rule_group.add_argument(
        "--rule-initial-density-max",
        type=float,
        default=None,
        help="If set with --rule-initial-density-min, sample each rule-sweep start p ~ Uniform(min, max).",
    )
    rule_group.add_argument("--include-conway-rule", dest="include_conway_rule", action="store_true", default=True)
    rule_group.add_argument("--no-include-conway-rule", dest="include_conway_rule", action="store_false")
    rule_group.add_argument("--stream-per-init-csv", dest="stream_per_init_csv", action="store_true", default=True)
    rule_group.add_argument("--no-stream-per-init-csv", dest="stream_per_init_csv", action="store_false")
    rule_group.add_argument("--progress-interval-rules", type=int, default=64)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.config_json is not None:
        data = load_config_json(Path(args.config_json))
        valid = set(ExperimentConfig.__dataclass_fields__.keys())
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown config fields in JSON: {unknown}")
        cfg = replace(cfg, **data)
    if args.smoke:
        cfg = replace(
            cfg,
            L=24,
            T=64,
            burn_in=8,
            window_size=8,
            window_step=4,
            n_cell_sample=32,
            null_reps=1,
            population_size=6,
            generations=3,
            random_controls=6,
            video_top_k=1,
            video_scale=8,
        )

    overrides = {}
    for field_name in ExperimentConfig.__dataclass_fields__.keys():
        arg_name = field_name
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                overrides[field_name] = value
    cli_aliases = {
        "burn_in": args.burn_in,
        "window_size": args.window_size,
        "window_step": args.window_step,
        "n_cell_sample": args.n_cell_sample,
        "null_reps": args.null_reps,
        "population_size": args.population_size,
        "elite_frac": args.elite_frac,
        "mutation_rate": args.mutation_rate,
        "initial_density": args.initial_density,
        "random_seed": args.random_seed,
        "backend": args.backend,
        "eval_batch_size": args.eval_batch_size,
        "jax_metric_batch_size": args.jax_metric_batch_size,
        "pair_sample": args.pair_sample,
        "delta_h_floor": args.delta_h_floor,
        "min_delta_h_nonzero_frac": args.min_delta_h_nonzero_frac,
        "delta_h_nonzero_eps": args.delta_h_nonzero_eps,
        "mspd_floor": args.mspd_floor,
        "tournament_size": args.tournament_size,
        "random_controls": args.random_controls,
        "output_dir": args.output_dir,
        "save_videos": args.save_videos,
        "video_top_k": args.video_top_k,
        "video_fps": args.video_fps,
        "video_scale": args.video_scale,
        "video_stride": args.video_stride,
        "montage_frames": args.montage_frames,
        "verbose": args.verbose,
        "progress_every": args.progress_every,
    }
    for key, value in cli_aliases.items():
        if value is not None:
            overrides[key] = value
    return replace(cfg, **overrides)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = parser_from_config()
    args = parser.parse_args(argv)
    configure_jax_platform(args.jax_platform)
    config = config_from_args(args)
    if config.backend == "jax" or args.require_accelerator:
        device_info = jax_device_summary(require_accelerator=args.require_accelerator)
        log_verbose(
            "JAX devices: "
            + ", ".join(f"{row['platform']}:{row['device_kind']}" for row in device_info["devices"]),
            config.verbose,
        )
    if args.experiment == "rule-sweep":
        mode = "all" if args.all_rules else args.rule_candidate_mode
        n_rule_candidates = None if mode == "all" else args.n_rule_candidates
        if (args.rule_initial_density_min is None) != (args.rule_initial_density_max is None):
            raise ValueError(
                "Set both --rule-initial-density-min and --rule-initial-density-max, or neither."
            )
        rule_initial_density_range = None
        if args.rule_initial_density_min is not None:
            rule_initial_density_range = (
                float(args.rule_initial_density_min),
                float(args.rule_initial_density_max),
            )
        result = run_rule_sweep_experiment(
            config,
            output_dir=config.output_dir,
            rule_candidate_mode=mode,
            n_rule_candidates=n_rule_candidates,
            n_initial_boards=args.n_rule_initial_boards,
            initial_density_range=rule_initial_density_range,
            include_conway=args.include_conway_rule,
            stream_per_init_csv=args.stream_per_init_csv,
            progress_interval_rules=args.progress_interval_rules,
            verbose=config.verbose,
        )
    else:
        result = run_experiment(config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
