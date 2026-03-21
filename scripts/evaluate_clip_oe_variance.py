import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import asal_metrics
import foundation_models
import substrates
import util
from clip_deltah_msc_metric import make_metric_loss_fn, metric_summary, resolve_metric_config
from rollout import rollout_simulation


def _patch_wandb_pandas_check() -> None:
    try:
        import wandb.util as wandb_util
    except Exception:
        return
    orig = getattr(wandb_util, "is_pandas_data_frame", None)
    if orig is None:
        return

    def _safe_is_pandas_data_frame(val):
        try:
            return orig(val)
        except Exception:
            return False

    wandb_util.is_pandas_data_frame = _safe_is_pandas_data_frame


_patch_wandb_pandas_check()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_like: str | None, root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return root / path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "run"


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/evaluate_clip_oe_variance.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("source", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("metric", {}),
        cfg.get("experiment", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def _validate_param_dim(substrate, params: np.ndarray) -> None:
    expected_len = int(np.asarray(substrate.default_params(jax.random.PRNGKey(0))).size)
    if int(params.shape[-1]) != expected_len:
        raise ValueError(
            f"Loaded parameter length {int(params.shape[-1])} does not match substrate expectation {expected_len}. "
            "Check that checkpoint and substrate config match."
        )


def _select_generation_indices(n_iters: int, fractions, indices) -> list[int]:
    chosen = []
    if fractions is not None:
        for frac in fractions:
            f = float(frac)
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"generation_fractions must be in [0,1], got {f}.")
            chosen.append(int(round(f * max(0, n_iters - 1))))
    if indices is not None:
        for idx in indices:
            i = int(idx)
            if i < 0:
                i = n_iters + i
            chosen.append(i)
    if not chosen:
        raise ValueError("Specify source.generation_fractions and/or source.generation_indices.")
    chosen = sorted({i for i in chosen if 0 <= i < n_iters})
    if not chosen:
        raise ValueError(f"No valid generation indices for n_iters={n_iters}.")
    return chosen


def _take_evenly_from_ranked(order: np.ndarray, n_take: int) -> list[int]:
    order = np.asarray(order, dtype=np.int32)
    if n_take <= 0 or order.size == 0:
        return []
    if n_take >= order.size:
        return [int(x) for x in order.tolist()]
    pos = np.linspace(0, order.size - 1, n_take)
    chosen = []
    seen = set()
    for p in pos:
        idx = int(order[int(round(float(p)))])
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    if len(chosen) < n_take:
        for idx in order.tolist():
            ii = int(idx)
            if ii not in seen:
                chosen.append(ii)
                seen.add(ii)
                if len(chosen) >= n_take:
                    break
    return chosen[:n_take]


def _select_candidate_indices(
    *,
    training_losses: np.ndarray | None,
    max_candidates: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if training_losses is None:
        raise ValueError("training_losses must be provided for candidate subsampling.")
    losses = np.asarray(training_losses, dtype=np.float64)
    pop_size = int(losses.size)
    if max_candidates is None or max_candidates >= pop_size:
        idx = np.arange(pop_size, dtype=np.int32)
        labels = np.asarray(["all"] * pop_size, dtype="<U8")
        return idx, labels
    if max_candidates < 2:
        order = np.argsort(losses, kind="mergesort")
        return np.asarray([int(order[0])], dtype=np.int32), np.asarray(["top"], dtype="<U8")

    rank_order = np.argsort(losses, kind="mergesort")
    k = int(max_candidates)
    top_n = max(1, int(np.ceil(k / 3.0)))
    mid_n = max(1, int(np.floor(k / 3.0)))
    rem_n = max(0, k - top_n - mid_n)

    selected: list[int] = [int(x) for x in rank_order[:top_n].tolist()]
    selected_labels: list[str] = ["top"] * len(selected)
    selected_set = set(selected)

    mid_center = pop_size // 2
    mid_half_width = max(mid_n, pop_size // 6)
    mid_lo = max(0, mid_center - mid_half_width)
    mid_hi = min(pop_size, mid_center + mid_half_width + 1)
    mid_band = np.asarray([int(x) for x in rank_order[mid_lo:mid_hi].tolist() if int(x) not in selected_set], dtype=np.int32)
    mid_take = _take_evenly_from_ranked(mid_band, mid_n)
    selected.extend(mid_take)
    selected_labels.extend(["mid"] * len(mid_take))
    selected_set.update(mid_take)

    remaining_ranked = np.asarray([int(x) for x in rank_order.tolist() if int(x) not in selected_set], dtype=np.int32)
    rem_take = _take_evenly_from_ranked(remaining_ranked, rem_n)
    selected.extend(rem_take)
    selected_labels.extend(["spread"] * len(rem_take))
    selected_set.update(rem_take)

    if len(selected) < k:
        rng = np.random.default_rng(seed)
        remaining = np.asarray([int(x) for x in rank_order.tolist() if int(x) not in selected_set], dtype=np.int32)
        if remaining.size > 0:
            extra = remaining[rng.permutation(remaining.size)[: k - len(selected)]]
            selected.extend([int(x) for x in extra.tolist()])
            selected_labels.extend(["extra"] * int(extra.size))

    selected = selected[:k]
    selected_labels = selected_labels[:k]
    paired = sorted(zip(selected, selected_labels), key=lambda x: losses[x[0]])
    idx = np.asarray([int(i) for i, _ in paired], dtype=np.int32)
    labels = np.asarray([str(lbl) for _, lbl in paired], dtype="<U8")
    return idx, labels


def _rank_ordinal(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt(np.sum(da * da) * np.sum(db * db))
    if denom <= 0:
        return float("nan")
    return float(np.sum(da * db) / denom)


def _pairwise_order_agreement(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    n = int(ref.size)
    total = 0
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_ref = float(ref[i] - ref[j])
            if abs(d_ref) <= eps:
                continue
            d_est = float(est[i] - est[j])
            total += 1
            if d_ref * d_est > 0:
                agree += 1
    if total == 0:
        return float("nan")
    return float(agree / total)


def _topk_overlap_curve(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    n = int(ref.size)
    ref_order = np.argsort(ref, kind="mergesort")
    est_order = np.argsort(est, kind="mergesort")
    in_ref = np.zeros((n,), dtype=bool)
    in_est = np.zeros((n,), dtype=bool)
    overlap_curve = np.empty((n,), dtype=np.float64)
    inter = 0
    for k in range(n):
        a = int(ref_order[k])
        b = int(est_order[k])
        if a == b:
            if not in_ref[a]:
                inter += 1
        else:
            if in_est[a]:
                inter += 1
            if in_ref[b]:
                inter += 1
        in_ref[a] = True
        in_est[b] = True
        overlap_curve[k] = inter / float(k + 1)
    return overlap_curve


def _ranking_metrics(ref: np.ndarray, est: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    ref_ranks = _rank_ordinal(ref)
    est_ranks = _rank_ordinal(est)
    topk_curve = _topk_overlap_curve(ref, est)
    return (
        {
            "spearman_rho": _pearson_corr(ref_ranks, est_ranks),
            "top1_match": float(int(np.argmin(ref) == np.argmin(est))),
            "mean_topk_overlap_allk": float(np.mean(topk_curve)),
            "pairwise_agreement": _pairwise_order_agreement(ref, est),
        },
        topk_curve,
    )


def _plot_metric_summary(rows: list[dict], metric_name: str, out_path: Path):
    rows = [r for r in rows if r["metric"] == metric_name]
    bs = np.array([int(r["bs"]) for r in rows], dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("mean_spearman_rho", "Spearman rho"),
        ("mean_top1_match", "Top-1 match rate"),
        ("mean_topk_overlap_allk", "Mean top-k overlap (all k)"),
        ("mean_pairwise_agreement", "Pairwise agreement"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        y = np.array([float(r[key]) for r in rows], dtype=np.float64)
        p10 = np.array([float(r[f"p10_{key[5:]}"]) for r in rows], dtype=np.float64)
        ax.plot(bs, y, marker="o")
        ax.plot(bs, p10, marker="s", linestyle="--", alpha=0.8, label="p10")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bs")
        ax.set_ylabel(title)
        ax.set_title(f"{metric_name}: {title}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def _plot_metric_comparison(rows: list[dict], out_path: Path):
    metrics = [
        ("mean_spearman_rho", "Spearman rho"),
        ("mean_top1_match", "Top-1 match rate"),
        ("mean_topk_overlap_allk", "Mean top-k overlap (all k)"),
        ("mean_pairwise_agreement", "Pairwise agreement"),
    ]
    metric_names = sorted({r["metric"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for metric_name in metric_names:
            sub = [r for r in rows if r["metric"] == metric_name]
            bs = np.array([int(r["bs"]) for r in sub], dtype=np.int32)
            y = np.array([float(r[key]) for r in sub], dtype=np.float64)
            ax.plot(bs, y, marker="o", label=metric_name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bs")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def _plot_topk_heatmap(rows: list[dict], metric_name: str, out_path: Path):
    rows = [r for r in rows if r["metric"] == metric_name]
    bs_vals = sorted({int(r["bs"]) for r in rows})
    k_vals = sorted({int(r["k"]) for r in rows})
    mean_grid = np.full((len(bs_vals), len(k_vals)), np.nan, dtype=np.float64)
    p10_grid = np.full((len(bs_vals), len(k_vals)), np.nan, dtype=np.float64)
    bs_to_i = {bs: i for i, bs in enumerate(bs_vals)}
    k_to_j = {k: j for j, k in enumerate(k_vals)}
    for row in rows:
        mean_grid[bs_to_i[int(row["bs"])], k_to_j[int(row["k"])]] = float(row["mean_topk_overlap"])
        p10_grid[bs_to_i[int(row["bs"])], k_to_j[int(row["k"])]] = float(row["p10_topk_overlap"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, grid, title in zip(
        axes,
        [mean_grid, p10_grid],
        ["Mean top-k overlap", "P10 top-k overlap"],
    ):
        im = ax.imshow(grid, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{metric_name}: {title}")
        ax.set_xlabel("k")
        ax.set_ylabel("bs")
        ax.set_xticks(np.arange(len(k_vals)))
        ax.set_xticklabels(k_vals)
        ax.set_yticks(np.arange(len(bs_vals)))
        ax.set_yticklabels(bs_vals)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def _plot_reference_spread(reference_rows: list[dict], metric_name: str, out_path: Path):
    rows = [r for r in reference_rows if r["metric"] == metric_name]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(rows))
    means = [float(r["ref_mean_mean"]) for r in rows]
    spreads = [float(r["ref_score_std"]) for r in rows]
    labels = [f"{r['run_label']}\niter={r['generation_idx']}" for r in rows]
    ax.bar(xs, means, yerr=spreads, color="#4C78A8", alpha=0.9)
    ax.set_title(f"{metric_name}: reference candidate scores")
    ax.set_ylabel("reference loss")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return fig


def _recommend_bs(summary_rows: list[dict], metric_name: str, cfg_exp) -> int | None:
    rows = [r for r in summary_rows if r["metric"] == metric_name]
    thr = cfg_exp.get("recommendation", {})
    if thr is None:
        return None
    thr = OmegaConf.to_container(thr, resolve=True) if OmegaConf.is_config(thr) else dict(thr)
    default_thr = {
        "min_mean_spearman_rho": 0.9,
        "min_mean_top1_match": 0.75,
        "min_mean_topk_overlap_allk": 0.8,
        "min_mean_pairwise_agreement": 0.9,
        "min_p10_spearman_rho": 0.6,
        "min_p10_topk_overlap_allk": 0.65,
        "min_p10_pairwise_agreement": 0.75,
    }
    default_thr.update({k: v for k, v in thr.items() if v is not None})

    for row in rows:
        if float(row["mean_spearman_rho"]) < float(default_thr["min_mean_spearman_rho"]):
            continue
        if float(row["mean_top1_match"]) < float(default_thr["min_mean_top1_match"]):
            continue
        if float(row["mean_topk_overlap_allk"]) < float(default_thr["min_mean_topk_overlap_allk"]):
            continue
        if float(row["mean_pairwise_agreement"]) < float(default_thr["min_mean_pairwise_agreement"]):
            continue
        if float(row["p10_spearman_rho"]) < float(default_thr["min_p10_spearman_rho"]):
            continue
        if float(row["p10_topk_overlap_allk"]) < float(default_thr["min_p10_topk_overlap_allk"]):
            continue
        if float(row["p10_pairwise_agreement"]) < float(default_thr["min_p10_pairwise_agreement"]):
            continue
        return int(row["bs"])
    return None


def _init_lagrangian_points_jax(
    A0: jax.Array,
    *,
    n_particles: int,
    init_mode: str,
    border: str,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    sx = int(A0.shape[0])
    sy = int(A0.shape[1])
    init_mode = str(init_mode).strip().lower()
    if init_mode == "uniform":
        k0, k1 = jax.random.split(key)
        y = jax.random.uniform(k0, (n_particles,), minval=0.5, maxval=sx - 0.5)
        x = jax.random.uniform(k1, (n_particles,), minval=0.5, maxval=sy - 0.5)
        pts = jnp.stack((y, x), axis=-1)
    elif init_mode == "mass":
        mass = jnp.clip(jnp.asarray(A0, dtype=jnp.float32).sum(axis=-1), 0.0, jnp.inf)
        flat = mass.reshape(-1)
        total = jnp.sum(flat)
        probs = jnp.where(total > 0.0, flat / jnp.maximum(total, 1e-12), jnp.ones_like(flat) / flat.size)
        k_idx, k_jit = jax.random.split(key)
        idx = jax.random.choice(k_idx, flat.size, shape=(n_particles,), replace=True, p=probs)
        iy = idx // sy
        ix = idx % sy
        jitter = jax.random.uniform(k_jit, (n_particles, 2), minval=-0.49, maxval=0.49)
        pts = jnp.stack((iy.astype(jnp.float32) + 0.5, ix.astype(jnp.float32) + 0.5), axis=-1) + jitter
    else:
        raise ValueError(f"Unknown metric_lagrangian_init_mode={init_mode!r}. Use 'mass' or 'uniform'.")

    if border == "torus":
        y = jnp.mod(pts[:, 0] - 0.5, sx) + 0.5
        x = jnp.mod(pts[:, 1] - 0.5, sy) + 0.5
        pts = jnp.stack((y, x), axis=-1)
    else:
        lo = float(sigma)
        hi_y = float(sx - sigma)
        hi_x = float(sy - sigma)
        y = jnp.clip(pts[:, 0], lo, hi_y)
        x = jnp.clip(pts[:, 1], lo, hi_x)
        pts = jnp.stack((y, x), axis=-1)
    return pts.astype(jnp.float32)


def _save_raw_population_scores(
    out_path: Path,
    *,
    score_matrices: dict[str, np.ndarray],
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
    candidate_indices: np.ndarray,
    candidate_labels: np.ndarray,
    training_losses: np.ndarray | None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "meta__bs_values": np.asarray(bs_values, dtype=np.int32),
        "meta__bs_ref": np.asarray(bs_ref, dtype=np.int32),
        "meta__n_repeats": np.asarray(n_repeats, dtype=np.int32),
        "meta__candidate_indices": np.asarray(candidate_indices, dtype=np.int32),
        "meta__candidate_labels": np.asarray(candidate_labels),
        "meta__metric_names": np.asarray(sorted(score_matrices.keys())),
    }
    if training_losses is not None:
        payload["meta__training_losses_selected"] = np.asarray(training_losses, dtype=np.float32)
    for metric_name, score_matrix in score_matrices.items():
        ref_block = np.asarray(score_matrix[:, :bs_ref], dtype=np.float32)
        ref_scores = ref_block.mean(axis=1).astype(np.float32)
        payload[f"{metric_name}__score_matrix_single"] = np.asarray(score_matrix, dtype=np.float32)
        payload[f"{metric_name}__ref_single_scores"] = ref_block
        payload[f"{metric_name}__ref_scores"] = ref_scores
        offset = bs_ref
        for bs in bs_values:
            block = np.asarray(score_matrix[:, offset:offset + n_repeats * bs], dtype=np.float32)
            block = block.reshape(score_matrix.shape[0], n_repeats, bs)
            payload[f"{metric_name}__bs_{bs}_single_scores"] = block
            payload[f"{metric_name}__bs_{bs}_mean_scores"] = block.mean(axis=2).T.astype(np.float32)
            offset += n_repeats * bs
    _save_npz_atomic(out_path, **payload)


def _load_raw_population_scores(
    in_path: Path,
    *,
    metric_names: list[str],
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(in_path, allow_pickle=False) as data:
        stored_bs_values = np.asarray(data["meta__bs_values"], dtype=np.int32).tolist()
        stored_bs_ref = int(np.asarray(data["meta__bs_ref"]).reshape(()))
        stored_n_repeats = int(np.asarray(data["meta__n_repeats"]).reshape(()))
        if stored_bs_values != [int(x) for x in bs_values] or stored_bs_ref != int(bs_ref) or stored_n_repeats != int(n_repeats):
            raise ValueError(
                f"Raw cache mismatch in {in_path}: expected bs_ref={bs_ref}, bs_values={bs_values}, n_repeats={n_repeats}; "
                f"got bs_ref={stored_bs_ref}, bs_values={stored_bs_values}, n_repeats={stored_n_repeats}."
            )
        stored_metric_names = None
        if "meta__metric_names" in data.files:
            stored_metric_names = sorted(str(x) for x in np.asarray(data["meta__metric_names"]).tolist())
            if stored_metric_names != sorted(metric_names):
                raise ValueError(
                    f"Raw cache mismatch in {in_path}: expected metrics {sorted(metric_names)}, got {stored_metric_names}."
                )
        candidate_indices = np.asarray(data["meta__candidate_indices"], dtype=np.int32)
        candidate_labels = np.asarray(data["meta__candidate_labels"]).astype(str)
        training_losses = None
        if "meta__training_losses_selected" in data.files:
            training_losses = np.asarray(data["meta__training_losses_selected"], dtype=np.float32)
        score_matrices = {}
        for metric_name in metric_names:
            key = f"{metric_name}__score_matrix_single"
            if key not in data.files:
                raise ValueError(f"Raw cache {in_path} missing required array {key}.")
            score_matrices[metric_name] = np.asarray(data[key], dtype=np.float32)
    return score_matrices, candidate_indices, candidate_labels, training_losses


def _save_partial_population_scores(
    out_path: Path,
    *,
    scores_flat_by_metric: dict[str, np.ndarray],
    completed: int,
    total_single_scores: int,
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
    candidate_indices: np.ndarray,
    candidate_labels: np.ndarray,
    training_losses: np.ndarray | None,
    known_mask: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "meta__bs_values": np.asarray(bs_values, dtype=np.int32),
        "meta__bs_ref": np.asarray(bs_ref, dtype=np.int32),
        "meta__n_repeats": np.asarray(n_repeats, dtype=np.int32),
        "meta__completed": np.asarray(int(completed), dtype=np.int32),
        "meta__total_single_scores": np.asarray(int(total_single_scores), dtype=np.int32),
        "meta__candidate_indices": np.asarray(candidate_indices, dtype=np.int32),
        "meta__candidate_labels": np.asarray(candidate_labels),
        "meta__metric_names": np.asarray(sorted(scores_flat_by_metric.keys())),
    }
    if training_losses is not None:
        payload["meta__training_losses_selected"] = np.asarray(training_losses, dtype=np.float32)
    if known_mask is not None:
        payload["meta__known_mask"] = np.asarray(known_mask, dtype=bool)
    for metric_name, flat_scores in scores_flat_by_metric.items():
        payload[f"{metric_name}__scores_flat"] = np.asarray(flat_scores, dtype=np.float32)
    _save_npz_atomic(out_path, **payload)


def _load_partial_population_scores(
    in_path: Path,
    *,
    metric_names: list[str],
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
    total_size: int,
    total_single_scores: int,
) -> tuple[dict[str, np.ndarray], int, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(in_path, allow_pickle=False) as data:
        stored_bs_values = np.asarray(data["meta__bs_values"], dtype=np.int32).tolist()
        stored_bs_ref = int(np.asarray(data["meta__bs_ref"]).reshape(()))
        stored_n_repeats = int(np.asarray(data["meta__n_repeats"]).reshape(()))
        if stored_bs_values != [int(x) for x in bs_values] or stored_bs_ref != int(bs_ref) or stored_n_repeats != int(n_repeats):
            raise ValueError(
                f"Partial cache mismatch in {in_path}: expected bs_ref={bs_ref}, bs_values={bs_values}, n_repeats={n_repeats}; "
                f"got bs_ref={stored_bs_ref}, bs_values={stored_bs_values}, n_repeats={stored_n_repeats}."
            )
        stored_metric_names = sorted(str(x) for x in np.asarray(data["meta__metric_names"]).tolist())
        if stored_metric_names != sorted(metric_names):
            raise ValueError(
                f"Partial cache mismatch in {in_path}: expected metrics {sorted(metric_names)}, got {stored_metric_names}."
            )
        candidate_indices = np.asarray(data["meta__candidate_indices"], dtype=np.int32)
        candidate_labels = np.asarray(data["meta__candidate_labels"]).astype(str)
        training_losses = None
        if "meta__training_losses_selected" in data.files:
            training_losses = np.asarray(data["meta__training_losses_selected"], dtype=np.float32)

        # Backward compatibility:
        # some interrupted runs may leave a full raw cache under the partial filename.
        # Those archives do not have meta__completed / meta__total_single_scores, but they
        # do have <metric>__score_matrix_single arrays, which we can flatten and treat as complete.
        if "meta__completed" not in data.files:
            raw_score_matrices = {}
            for metric_name in metric_names:
                raw_key = f"{metric_name}__score_matrix_single"
                if raw_key not in data.files:
                    raise ValueError(
                        f"Partial cache {in_path} is missing meta__completed and does not look like a raw cache; "
                        f"expected either meta__completed or {raw_key}."
                    )
                matrix = np.asarray(data[raw_key], dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape != (candidate_indices.size, total_single_scores):
                    raise ValueError(
                        f"Partial cache {in_path} raw matrix {raw_key} has shape {matrix.shape}, "
                        f"expected {(candidate_indices.size, total_single_scores)}."
                    )
                raw_score_matrices[metric_name] = matrix.reshape(-1)
            return raw_score_matrices, total_size, candidate_indices, candidate_labels, training_losses

        stored_completed = int(np.asarray(data["meta__completed"]).reshape(()))
        stored_total_single_scores = int(np.asarray(data["meta__total_single_scores"]).reshape(()))
        if stored_total_single_scores != int(total_single_scores):
            raise ValueError(
                f"Partial cache mismatch in {in_path}: expected total_single_scores={total_single_scores}, "
                f"got {stored_total_single_scores}."
            )
        if not (0 <= stored_completed <= total_size):
            raise ValueError(
                f"Partial cache mismatch in {in_path}: completed={stored_completed} outside [0, {total_size}]."
            )

        scores_flat_by_metric = {}
        for metric_name in metric_names:
            key = f"{metric_name}__scores_flat"
            if key not in data.files:
                raise ValueError(f"Partial cache {in_path} missing required array {key}.")
            flat = np.asarray(data[key], dtype=np.float32)
            if flat.size != total_size:
                raise ValueError(
                    f"Partial cache {in_path} array {key} has size {flat.size}, expected {total_size}."
                )
            scores_flat_by_metric[metric_name] = flat
    return scores_flat_by_metric, stored_completed, candidate_indices, candidate_labels, training_losses


def _compute_total_single_scores(bs_ref: int, bs_values: list[int], n_repeats: int) -> int:
    return int(bs_ref) + int(n_repeats) * sum(int(bs) for bs in bs_values)


def _score_layout(bs_ref: int, bs_values: list[int], n_repeats: int) -> dict[str | int, tuple[int, int]]:
    layout: dict[str | int, tuple[int, int]] = {"ref": (0, int(bs_ref))}
    offset = int(bs_ref)
    for bs in bs_values:
        width = int(n_repeats) * int(bs)
        layout[int(bs)] = (offset, offset + width)
        offset += width
    return layout


def _iter_resume_cache_paths(
    *,
    expected_raw_path: Path,
    expected_partial_path: Path,
    save_dir: Path,
    run_path: Path,
    project_root: Path,
    gen_idx: int,
) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        path = Path(path)
        if path in seen:
            return
        seen.add(path)
        out.append(path)

    add(expected_raw_path)
    add(expected_partial_path)
    patterns = [
        f"*__iter_{int(gen_idx):05d}.npz",
        f"*__iter_{int(gen_idx):05d}.partial*.npz",
    ]
    direct_bases = (
        expected_raw_path.parent,
        save_dir,
        run_path / "raw_scores",
        run_path,
        project_root,
    )
    for base in direct_bases:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in sorted(base.glob(pattern)):
                add(path)
    sibling_root = run_path.parent
    if sibling_root.exists():
        for pattern in patterns:
            for path in sorted(sibling_root.glob(f"*/raw_scores/{pattern}")):
                add(path)
            for path in sorted(sibling_root.glob(pattern)):
                add(path)
    return out


def _seed_scores_from_cache_file(
    in_path: Path,
    *,
    metric_names: list[str],
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
    candidate_indices_current: np.ndarray,
    total_single_scores: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]] | None:
    if not in_path.exists():
        return None

    with np.load(in_path, allow_pickle=False) as data:
        required_meta = {"meta__bs_values", "meta__bs_ref", "meta__n_repeats", "meta__candidate_indices"}
        if not required_meta.issubset(set(data.files)):
            return None

        stored_bs_values = np.asarray(data["meta__bs_values"], dtype=np.int32).tolist()
        stored_bs_ref = int(np.asarray(data["meta__bs_ref"]).reshape(()))
        stored_n_repeats = int(np.asarray(data["meta__n_repeats"]).reshape(()))
        if stored_bs_values != [int(x) for x in bs_values]:
            return None
        if stored_bs_ref != int(bs_ref):
            return None
        if stored_n_repeats < 1 or stored_n_repeats > int(n_repeats):
            return None

        if "meta__metric_names" in data.files:
            stored_metric_names = sorted(str(x) for x in np.asarray(data["meta__metric_names"]).tolist())
            if stored_metric_names != sorted(metric_names):
                return None

        candidate_indices_stored = np.asarray(data["meta__candidate_indices"], dtype=np.int32)
        candidate_labels_stored = (
            np.asarray(data["meta__candidate_labels"]).astype(str)
            if "meta__candidate_labels" in data.files
            else np.asarray(["cached"] * int(candidate_indices_stored.size), dtype="<U8")
        )

        current_row_by_idx = {int(idx): row for row, idx in enumerate(np.asarray(candidate_indices_current, dtype=np.int32).tolist())}
        if any(int(idx) not in current_row_by_idx for idx in candidate_indices_stored.tolist()):
            return None

        stored_total_single_scores = int(
            np.asarray(data["meta__total_single_scores"]).reshape(())
        ) if "meta__total_single_scores" in data.files else _compute_total_single_scores(
            stored_bs_ref,
            [int(x) for x in stored_bs_values],
            stored_n_repeats,
        )

        cached_score_matrices: dict[str, np.ndarray] = {}
        cache_kind = None
        if all(f"{metric_name}__score_matrix_single" in data.files for metric_name in metric_names):
            cache_kind = "raw"
            for metric_name in metric_names:
                matrix = np.asarray(data[f"{metric_name}__score_matrix_single"], dtype=np.float32)
                if matrix.shape != (int(candidate_indices_stored.size), stored_total_single_scores):
                    return None
                cached_score_matrices[metric_name] = matrix
            stored_known_mask = np.ones(
                (int(candidate_indices_stored.size), stored_total_single_scores),
                dtype=bool,
            )
        elif all(f"{metric_name}__scores_flat" in data.files for metric_name in metric_names):
            cache_kind = "partial"
            flat_size = int(candidate_indices_stored.size) * stored_total_single_scores
            if "meta__known_mask" in data.files:
                known_flat = np.asarray(data["meta__known_mask"], dtype=bool).reshape(-1)
                if known_flat.size != flat_size:
                    return None
            elif "meta__completed" in data.files:
                completed = int(np.asarray(data["meta__completed"]).reshape(()))
                if not (0 <= completed <= flat_size):
                    return None
                known_flat = np.zeros((flat_size,), dtype=bool)
                known_flat[:completed] = True
            else:
                return None
            stored_known_mask = known_flat.reshape(int(candidate_indices_stored.size), stored_total_single_scores)
            for metric_name in metric_names:
                flat = np.asarray(data[f"{metric_name}__scores_flat"], dtype=np.float32).reshape(-1)
                if flat.size != flat_size:
                    return None
                cached_score_matrices[metric_name] = flat.reshape(int(candidate_indices_stored.size), stored_total_single_scores)
        else:
            return None

    current_candidate_count = int(candidate_indices_current.size)
    seeded_matrices = {
        metric_name: np.full((current_candidate_count, total_single_scores), np.nan, dtype=np.float32)
        for metric_name in metric_names
    }
    seeded_known_mask = np.zeros((current_candidate_count, total_single_scores), dtype=bool)

    stored_layout = _score_layout(int(bs_ref), [int(x) for x in bs_values], stored_n_repeats)
    requested_layout = _score_layout(int(bs_ref), [int(x) for x in bs_values], int(n_repeats))

    for stored_row, cand_idx in enumerate(candidate_indices_stored.tolist()):
        current_row = current_row_by_idx[int(cand_idx)]
        block_specs: list[tuple[slice, slice]] = [
            (slice(*stored_layout["ref"]), slice(*requested_layout["ref"])),
        ]
        for bs in bs_values:
            src_start, src_end = stored_layout[int(bs)]
            dst_start, _ = requested_layout[int(bs)]
            block_specs.append((slice(src_start, src_end), slice(dst_start, dst_start + (src_end - src_start))))

        for src_slice, dst_slice in block_specs:
            known_block = stored_known_mask[stored_row, src_slice]
            if not np.any(known_block):
                continue
            seeded_known_mask[current_row, dst_slice][known_block] = True
            for metric_name in metric_names:
                seeded_matrices[metric_name][current_row, dst_slice][known_block] = cached_score_matrices[metric_name][stored_row, src_slice][known_block]

    seeded_count = int(seeded_known_mask.sum())
    if seeded_count == 0:
        return None

    info = {
        "path": str(in_path),
        "kind": str(cache_kind),
        "seeded_count": int(seeded_count),
        "stored_n_repeats": int(stored_n_repeats),
        "stored_candidate_count": int(candidate_indices_stored.size),
        "stored_candidate_labels": [str(x) for x in candidate_labels_stored.tolist()],
    }
    return (
        {metric_name: matrix.reshape(-1) for metric_name, matrix in seeded_matrices.items()},
        seeded_known_mask.reshape(-1),
        info,
    )


def _accumulate_population_rows(
    *,
    metric_names: list[str],
    score_matrices: dict[str, np.ndarray],
    bs_values: list[int],
    n_repeats: int,
    bs_ref: int,
    run_label: str,
    run_path: Path,
    gen_idx: int,
    pop_size: int,
    candidate_indices: np.ndarray,
    candidate_labels: np.ndarray,
    raw_scores_path: Path,
    reference_rows: list[dict],
    all_repeat_rows: list[dict],
    topk_repeat_rows: list[dict],
    summary_by_metric: dict,
    topk_summary_by_metric: dict,
) -> None:
    selected_pop_size = int(candidate_indices.size)
    for metric_name in metric_names:
        score_matrix = np.asarray(score_matrices[metric_name], dtype=np.float32)
        if score_matrix.shape[0] != selected_pop_size:
            raise ValueError(
                f"Score matrix for {metric_name} has first dim {score_matrix.shape[0]}, expected {selected_pop_size}."
            )
        ref_block = score_matrix[:, :bs_ref]
        ref_scores = ref_block.mean(axis=1)
        ref_rank = np.argsort(ref_scores)
        reference_rows.append(
            {
                "metric": metric_name,
                "run_label": run_label,
                "run_save_dir": str(run_path),
                "generation_idx": int(gen_idx),
                "pop_size": int(pop_size),
                "selected_pop_size": int(selected_pop_size),
                "candidate_indices": [int(x) for x in candidate_indices.tolist()],
                "candidate_labels": [str(x) for x in candidate_labels.tolist()],
                "bs_ref": int(bs_ref),
                "ref_mean_mean": float(ref_scores.mean()),
                "ref_score_std": float(ref_scores.std(ddof=1) if selected_pop_size > 1 else 0.0),
                "ref_best_candidate": int(ref_rank[0]),
                "ref_best_candidate_global": int(candidate_indices[ref_rank[0]]),
                "ref_best_candidate_label": str(candidate_labels[ref_rank[0]]),
                "ref_rank_order_local": [int(x) for x in ref_rank.tolist()],
                "ref_rank_order_global": [int(candidate_indices[x]) for x in ref_rank.tolist()],
                "ref_rank_order_labels": [str(candidate_labels[x]) for x in ref_rank.tolist()],
                "raw_scores_path": str(raw_scores_path),
            }
        )

        offset = bs_ref
        for bs in bs_values:
            block = score_matrix[:, offset:offset + n_repeats * bs]
            offset += n_repeats * bs
            est_scores = block.reshape(selected_pop_size, n_repeats, bs).mean(axis=2).T
            for rep_idx in range(n_repeats):
                metrics, topk_curve = _ranking_metrics(ref_scores, est_scores[rep_idx])
                all_repeat_rows.append(
                    {
                        "metric": metric_name,
                        "run_label": run_label,
                        "run_save_dir": str(run_path),
                        "generation_idx": int(gen_idx),
                        "pop_size": int(pop_size),
                        "selected_pop_size": int(selected_pop_size),
                        "bs_ref": int(bs_ref),
                        "bs": int(bs),
                        "repeat_idx": int(rep_idx),
                        **{k: float(v) for k, v in metrics.items()},
                    }
                )
                summary_by_metric[metric_name][bs].append(metrics)
                for k_idx, overlap in enumerate(topk_curve, start=1):
                    topk_repeat_rows.append(
                        {
                            "metric": metric_name,
                            "run_label": run_label,
                            "run_save_dir": str(run_path),
                            "generation_idx": int(gen_idx),
                            "pop_size": int(pop_size),
                            "selected_pop_size": int(selected_pop_size),
                            "bs_ref": int(bs_ref),
                            "bs": int(bs),
                            "repeat_idx": int(rep_idx),
                            "k": int(k_idx),
                            "k_frac": float(k_idx / pop_size),
                            "topk_overlap": float(overlap),
                        }
                    )
                    topk_summary_by_metric[metric_name][bs].setdefault(k_idx, []).append(float(overlap))


def _create_base_substrate(args, enable_msc: bool):
    if args.substrate == "lenia_flow":
        base_substrate = substrates.create_substrate(
            args.substrate,
            **util.flow_lenia_kwargs_from_args(args),
        )
    else:
        base_substrate = substrates.create_substrate(args.substrate)
    if enable_msc and hasattr(base_substrate, "debug_return_F"):
        base_substrate.debug_return_F = True
    return base_substrate


def main(cfg, args):
    project_root = _repo_root()
    save_dir = _resolve_path(getattr(args, "save_dir", None), project_root)
    if save_dir is None:
        raise ValueError("meta.save_dir must be set.")
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_mode = str(getattr(args, "wandb_mode", "online"))
    run = wandb.init(
        project=str(getattr(args, "wandb_project", "asal")),
        mode=wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        enable_clip = bool(getattr(args, "enable_clip_loss", True))
        enable_msc = bool(getattr(args, "enable_msc_loss", True))
        resume = bool(getattr(args, "resume", True))
        if not enable_clip and not enable_msc:
            raise ValueError("At least one of enable_clip_loss / enable_msc_loss must be true.")

        base_substrate = _create_base_substrate(args, enable_msc)
        substrate = substrates.FlattenSubstrateParameters(base_substrate)

        rollout_steps = substrate.rollout_steps if getattr(args, "rollout_steps", None) is None else int(args.rollout_steps)
        metric_names: list[str] = []
        eval_fns: dict[str, callable] = {}
        eval_summaries: dict[str, dict] = {}

        if enable_clip:
            clip_time_sampling = int(getattr(args, "time_sampling"))
            if rollout_steps % clip_time_sampling != 0:
                raise ValueError(
                    f"rollout_steps ({rollout_steps}) must be divisible by time_sampling ({clip_time_sampling}) for clip."
                )
            fm = foundation_models.create_foundation_model(str(getattr(args, "foundation_model", "clip")))
            clip_img_size = int(getattr(args, "clip_img_size", 224))

            def _one_clip_rollout(rng_key, params_in):
                rollout_data = rollout_simulation(
                    rng_key,
                    params_in,
                    s0=None,
                    substrate=substrate,
                    fm=fm,
                    rollout_steps=rollout_steps,
                    time_sampling=(clip_time_sampling, True),
                    img_size=clip_img_size,
                    return_state=False,
                )
                return asal_metrics.calc_open_endedness_score(rollout_data["z"])

            eval_fns["clip"] = jax.jit(jax.vmap(_one_clip_rollout, in_axes=(0, 0)))
            eval_summaries["clip"] = {
                "time_sampling": clip_time_sampling,
                "img_size": clip_img_size,
                "foundation_model": str(getattr(args, "foundation_model", "clip")),
            }
            metric_names.append("clip")

        if enable_msc:
            metric_node = OmegaConf.merge(cfg.get("substrate", {}), cfg.get("metric", {}))
            metric_dict = OmegaConf.to_container(metric_node, resolve=True)
            metric_args = SimpleNamespace(**metric_dict)
            metric_args.rollout_steps = rollout_steps

            if getattr(metric_args, "metric_periodic", None) is None:
                metric_args.metric_periodic = str(getattr(substrate, "border", "wall")) == "torus"
            if getattr(metric_args, "metric_domain_y", None) is None:
                metric_args.metric_domain_y = float(
                    getattr(getattr(substrate, "cfg", None), "X", getattr(substrate, "grid_size", 0))
                )
            if getattr(metric_args, "metric_domain_x", None) is None:
                metric_args.metric_domain_x = float(
                    getattr(getattr(substrate, "cfg", None), "Y", getattr(substrate, "grid_size", 0))
                )

            metric_cfg = resolve_metric_config(metric_args)
            metric_loss_fn = make_metric_loss_fn(metric_cfg)
            metric_info = metric_summary(metric_cfg)

            lag_n_particles = int(getattr(metric_args, "metric_lagrangian_n_particles", 256))
            lag_init_mode = str(getattr(metric_args, "metric_lagrangian_init_mode", "mass"))
            lag_flow_channel = int(getattr(metric_args, "metric_lagrangian_flow_channel", -1))
            lag_flow_reduce = str(getattr(metric_args, "metric_lagrangian_flow_reduce", "mass_weighted"))
            lag_channel_mode = str(getattr(metric_args, "metric_lagrangian_channel_mode", "mix"))
            lag_noise_model = str(getattr(metric_args, "metric_lagrangian_noise_model", "none"))
            lag_diffusion_scale = float(getattr(metric_args, "metric_lagrangian_diffusion_scale", 1.0))
            chunk_steps = int(metric_cfg["sample_every_steps"])
            time_sampling_msc = int(metric_cfg["time_sampling"])

            def rollout_lagrangian_xy(rng, params):
                k_state, k_pts, k_ch, k_scan = jax.random.split(rng, 4)
                s0 = substrate.init_state(k_state, params)
                if "F" not in s0:
                    raise ValueError(
                        "State does not contain flow field F. For FlowLenia set debug_return_F=true."
                    )
                if not hasattr(substrate, "RT"):
                    raise ValueError("Substrate does not provide RT for lagrangian advection.")
                rt = substrate.RT
                pts0 = _init_lagrangian_points_jax(
                    s0["A"],
                    n_particles=lag_n_particles,
                    init_mode=lag_init_mode,
                    border=str(getattr(rt, "border", "wall")),
                    sigma=float(getattr(rt, "sigma", 0.0)),
                    key=k_pts,
                )
                if lag_channel_mode in ("fixed", "resample"):
                    ch0 = rt.sample_point_channels(pts0, s0["A"], k_ch)
                else:
                    ch0 = jnp.zeros((lag_n_particles,), dtype=jnp.int32)

                def step_fn(state, key_step):
                    st, pts, ch = state
                    st = substrate.step_state(key_step, st, params)
                    lag_key = jax.random.fold_in(key_step, jnp.uint32(0x4C4147))
                    pts, ch = rt.advect_particles(
                        points=pts,
                        F=st["F"],
                        A=st["A"],
                        channel=lag_flow_channel,
                        reduce=lag_flow_reduce,
                        point_channels=ch,
                        channel_mode=lag_channel_mode,
                        key=lag_key,
                        noise_model=lag_noise_model,
                        diffusion_scale=lag_diffusion_scale,
                    )
                    return (st, pts, ch), None

                def chunk_fn(state, key_chunk):
                    state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(key_chunk, chunk_steps))
                    return state_next, state_next[1]

                (_, _, _), xy_seq = jax.lax.scan(
                    chunk_fn,
                    (s0, pts0, jnp.asarray(ch0)),
                    jax.random.split(k_scan, time_sampling_msc),
                )
                return xy_seq

            def _one_msc_rollout(rng_key, params_in):
                rng_roll, rng_metric = jax.random.split(rng_key)
                xy_seq = rollout_lagrangian_xy(rng_roll, params_in)
                loss, _ = metric_loss_fn(rng_metric, xy_seq)
                return loss

            eval_fns["msc"] = jax.jit(jax.vmap(_one_msc_rollout, in_axes=(0, 0)))
            eval_summaries["msc"] = metric_info
            metric_names.append("msc")

        for metric_name, info in eval_summaries.items():
            run.summary[f"eval/{metric_name}"] = str(info)

        run_dirs_cfg = getattr(cfg, "source", {}).get("run_save_dirs", None)
        if run_dirs_cfg is None:
            raise ValueError("source.run_save_dirs must be set.")
        run_dirs = OmegaConf.to_container(run_dirs_cfg, resolve=True)
        run_labels_cfg = getattr(cfg, "source", {}).get("run_labels", None)
        run_labels = None if run_labels_cfg is None else OmegaConf.to_container(run_labels_cfg, resolve=True)
        fractions_cfg = getattr(cfg, "source", {}).get("generation_fractions", None)
        generation_fractions = None if fractions_cfg is None else OmegaConf.to_container(fractions_cfg, resolve=True)
        indices_cfg = getattr(cfg, "source", {}).get("generation_indices", None)
        generation_indices = None if indices_cfg is None else OmegaConf.to_container(indices_cfg, resolve=True)

        bs_ref = int(getattr(args, "bs_ref"))
        if bs_ref < 1:
            raise ValueError("experiment.bs_ref must be >= 1.")
        bs_values_cfg = OmegaConf.to_container(getattr(cfg, "experiment", {}).get("bs_values", []), resolve=True)
        if not bs_values_cfg:
            raise ValueError("experiment.bs_values must be non-empty.")
        bs_values = sorted({int(x) for x in bs_values_cfg})
        if bs_values[0] < 1:
            raise ValueError(f"Invalid bs_values={bs_values}. All bs must be >= 1.")
        n_repeats = int(getattr(args, "n_repeats"))
        if n_repeats < 1:
            raise ValueError("experiment.n_repeats must be >= 1.")
        eval_batch_size = int(getattr(args, "eval_batch_size", 1))
        if eval_batch_size < 1:
            raise ValueError("experiment.eval_batch_size must be >= 1.")
        max_candidates_per_generation = getattr(args, "max_candidates_per_generation", None)
        max_candidates_per_generation = None if max_candidates_per_generation is None else int(max_candidates_per_generation)
        if max_candidates_per_generation is not None and max_candidates_per_generation < 1:
            raise ValueError("experiment.max_candidates_per_generation must be >= 1 or null.")

        total_single_scores = bs_ref + n_repeats * sum(bs_values)
        all_repeat_rows = []
        topk_repeat_rows = []
        reference_rows = []
        generation_meta = []
        summary_by_metric = {metric_name: {bs: [] for bs in bs_values} for metric_name in metric_names}
        topk_summary_by_metric = {
            metric_name: {bs: {} for bs in bs_values} for metric_name in metric_names
        }
        raw_scores_dir = save_dir / "raw_scores"
        rng = jax.random.PRNGKey(int(getattr(args, "seed", 0)))
        resumed_generations = 0
        resumed_partial_generations = 0
        computed_generations = 0

        run_label_list = []
        for idx, run_dir_raw in enumerate(run_dirs):
            run_path = _resolve_path(str(run_dir_raw), project_root)
            if run_path is None:
                raise ValueError("Invalid run_save_dirs entry.")
            run_label = str(run_labels[idx]) if run_labels is not None and idx < len(run_labels) else run_path.name
            run_label_list.append(run_label)
            pop_path = run_path / "pop_traj.pkl"
            if not pop_path.exists():
                raise FileNotFoundError(f"pop_traj.pkl not found in {run_path}.")
            pop_traj = util.load_pkl(str(run_path), "pop_traj")
            params_traj = np.asarray(pop_traj["params"], dtype=np.float32)
            loss_traj = np.asarray(pop_traj["loss"], dtype=np.float32) if "loss" in pop_traj else None
            if params_traj.ndim != 3:
                raise ValueError(f"Expected pop_traj['params'] to have shape (T, pop, D), got {params_traj.shape}.")
            if loss_traj is None or loss_traj.shape[:2] != params_traj.shape[:2]:
                raise ValueError(
                    f"Expected pop_traj['loss'] to have shape {params_traj.shape[:2]}, got "
                    f"{None if loss_traj is None else loss_traj.shape}."
                )
            _validate_param_dim(substrate, params_traj[0])
            n_iters, pop_size, n_params = params_traj.shape
            gen_indices = _select_generation_indices(n_iters, generation_fractions, generation_indices)

            generation_meta.append(
                {
                    "run_label": run_label,
                    "run_save_dir": str(run_path),
                    "n_iters_available": int(n_iters),
                    "pop_size": int(pop_size),
                    "n_params": int(n_params),
                    "max_candidates_per_generation": None if max_candidates_per_generation is None else int(max_candidates_per_generation),
                    "selected_generation_indices": [int(x) for x in gen_indices],
                }
            )

            for gen_idx in gen_indices:
                params_gen_full = np.asarray(params_traj[gen_idx], dtype=np.float32)
                train_loss_gen_full = np.asarray(loss_traj[gen_idx], dtype=np.float32)
                candidate_indices, candidate_labels = _select_candidate_indices(
                    training_losses=train_loss_gen_full,
                    max_candidates=max_candidates_per_generation,
                    seed=int(getattr(args, "seed", 0)) + 10007 * idx + 997 * int(gen_idx),
                )
                params_gen = np.asarray(params_gen_full[candidate_indices], dtype=np.float32)
                train_loss_gen = np.asarray(train_loss_gen_full[candidate_indices], dtype=np.float32)
                selected_pop_size = int(params_gen.shape[0])
                raw_scores_path = raw_scores_dir / f"{_slugify(run_label)}__iter_{int(gen_idx):05d}.npz"
                partial_scores_path = raw_scores_dir / f"{_slugify(run_label)}__iter_{int(gen_idx):05d}.partial.npz"

                candidate_ids = np.repeat(np.arange(selected_pop_size, dtype=np.int32), total_single_scores)
                total_size = int(candidate_ids.size)
                scores_flat_by_metric = {
                    metric_name: np.full((total_size,), np.nan, dtype=np.float32) for metric_name in metric_names
                }
                known_mask = np.zeros((total_size,), dtype=bool)
                seed_info = None
                if resume:
                    best_seed = None
                    for cache_path in _iter_resume_cache_paths(
                        expected_raw_path=raw_scores_path,
                        expected_partial_path=partial_scores_path,
                        save_dir=save_dir,
                        run_path=run_path,
                        project_root=project_root,
                        gen_idx=int(gen_idx),
                    ):
                        seed_result = _seed_scores_from_cache_file(
                            cache_path,
                            metric_names=metric_names,
                            bs_values=bs_values,
                            n_repeats=n_repeats,
                            bs_ref=bs_ref,
                            candidate_indices_current=candidate_indices,
                            total_single_scores=total_single_scores,
                        )
                        if seed_result is None:
                            continue
                        seeded_scores, seeded_mask, seeded_info = seed_result
                        if best_seed is None or int(seeded_info["seeded_count"]) > int(best_seed[2]["seeded_count"]):
                            best_seed = (seeded_scores, seeded_mask, seeded_info)
                    if best_seed is not None:
                        scores_flat_by_metric, known_mask, seed_info = best_seed

                loaded_candidate_indices = candidate_indices
                loaded_candidate_labels = candidate_labels
                loaded_train_loss_gen = train_loss_gen

                score_matrices = None
                known_count = int(known_mask.sum())
                if known_count == total_size:
                    resumed_generations += 1
                    print(
                        f"[resume] using cached scores: {run_label} iter={int(gen_idx)} "
                        f"from {known_count}/{total_size} singles via {seed_info['path'] if seed_info else raw_scores_path} "
                        f"({selected_pop_size}/{pop_size} candidates)"
                    )
                    score_matrices = {
                        metric_name: scores_flat_by_metric[metric_name].reshape(selected_pop_size, total_single_scores)
                        for metric_name in metric_names
                    }
                    if seed_info is not None and Path(str(seed_info["path"])) != raw_scores_path:
                        _save_raw_population_scores(
                            raw_scores_path,
                            score_matrices=score_matrices,
                            bs_values=bs_values,
                            n_repeats=n_repeats,
                            bs_ref=bs_ref,
                            candidate_indices=candidate_indices,
                            candidate_labels=candidate_labels,
                            training_losses=train_loss_gen,
                        )
                else:
                    if known_count > 0:
                        resumed_partial_generations += 1
                        print(
                            f"[resume] continuing from cached scores: {run_label} iter={int(gen_idx)} "
                            f"at {known_count}/{total_size} singles via {seed_info['path'] if seed_info else partial_scores_path} "
                            f"({selected_pop_size}/{pop_size} candidates)"
                        )
                    pending_flat_idx = np.flatnonzero(~known_mask)
                    remaining_size = int(pending_flat_idx.size)
                    desc = f"{run_label} iter={gen_idx} cand={selected_pop_size}/{pop_size}"
                    if known_count > 0:
                        desc += f" resume={known_count}/{total_size}"
                    pbar = tqdm(
                        total=remaining_size,
                        desc=desc,
                        leave=False,
                    )
                    cursor = 0
                    while cursor < remaining_size:
                        batch = min(eval_batch_size, remaining_size - cursor)
                        flat_idx_chunk = pending_flat_idx[cursor:cursor + batch]
                        ids_chunk = candidate_ids[flat_idx_chunk]
                        params_chunk = jnp.asarray(params_gen[ids_chunk])
                        rng, rng_batch = jax.random.split(rng)
                        keys = jax.random.split(rng_batch, batch)
                        for metric_name in metric_names:
                            chunk_scores = np.asarray(jax.device_get(eval_fns[metric_name](keys, params_chunk)), dtype=np.float32)
                            scores_flat_by_metric[metric_name][flat_idx_chunk] = chunk_scores
                        known_mask[flat_idx_chunk] = True
                        cursor += batch
                        completed = int(known_mask.sum())
                        if resume:
                            _save_partial_population_scores(
                                partial_scores_path,
                                scores_flat_by_metric=scores_flat_by_metric,
                                completed=completed,
                                total_single_scores=total_single_scores,
                                bs_values=bs_values,
                                n_repeats=n_repeats,
                                bs_ref=bs_ref,
                                candidate_indices=candidate_indices,
                                candidate_labels=candidate_labels,
                                training_losses=train_loss_gen,
                                known_mask=known_mask,
                            )
                        pbar.update(batch)
                    pbar.close()

                    score_matrices = {
                        metric_name: scores_flat_by_metric[metric_name].reshape(selected_pop_size, total_single_scores)
                        for metric_name in metric_names
                    }
                    _save_raw_population_scores(
                        raw_scores_path,
                        score_matrices=score_matrices,
                        bs_values=bs_values,
                        n_repeats=n_repeats,
                        bs_ref=bs_ref,
                        candidate_indices=candidate_indices,
                        candidate_labels=candidate_labels,
                        training_losses=train_loss_gen,
                    )
                    if partial_scores_path.exists():
                        partial_scores_path.unlink()
                    computed_generations += 1

                _accumulate_population_rows(
                    metric_names=metric_names,
                    score_matrices=score_matrices,
                    bs_values=bs_values,
                    n_repeats=n_repeats,
                    bs_ref=bs_ref,
                    run_label=run_label,
                    run_path=run_path,
                    gen_idx=int(gen_idx),
                    pop_size=int(pop_size),
                    candidate_indices=np.asarray(loaded_candidate_indices, dtype=np.int32),
                    candidate_labels=np.asarray(loaded_candidate_labels).astype(str),
                    raw_scores_path=raw_scores_path,
                    reference_rows=reference_rows,
                    all_repeat_rows=all_repeat_rows,
                    topk_repeat_rows=topk_repeat_rows,
                    summary_by_metric=summary_by_metric,
                    topk_summary_by_metric=topk_summary_by_metric,
                )

        summary_rows = []
        topk_summary_rows = []
        recommended_bs = {}
        for metric_name in metric_names:
            for bs in bs_values:
                metrics_list = summary_by_metric[metric_name][bs]
                if not metrics_list:
                    continue

                def arr(key):
                    return np.asarray([float(m[key]) for m in metrics_list], dtype=np.float64)

                spearman = arr("spearman_rho")
                top1 = arr("top1_match")
                topk_allk = arr("mean_topk_overlap_allk")
                pair = arr("pairwise_agreement")
                summary_rows.append(
                    {
                        "metric": metric_name,
                        "bs": int(bs),
                        "n_population_repeats": int(len(metrics_list)),
                        "mean_spearman_rho": float(np.nanmean(spearman)),
                        "std_spearman_rho": float(np.nanstd(spearman, ddof=1) if spearman.size > 1 else 0.0),
                        "p10_spearman_rho": float(np.nanquantile(spearman, 0.10)),
                        "mean_top1_match": float(np.nanmean(top1)),
                        "std_top1_match": float(np.nanstd(top1, ddof=1) if top1.size > 1 else 0.0),
                        "p10_top1_match": float(np.nanquantile(top1, 0.10)),
                        "mean_topk_overlap_allk": float(np.nanmean(topk_allk)),
                        "std_topk_overlap_allk": float(np.nanstd(topk_allk, ddof=1) if topk_allk.size > 1 else 0.0),
                        "p10_topk_overlap_allk": float(np.nanquantile(topk_allk, 0.10)),
                        "mean_pairwise_agreement": float(np.nanmean(pair)),
                        "std_pairwise_agreement": float(np.nanstd(pair, ddof=1) if pair.size > 1 else 0.0),
                        "p10_pairwise_agreement": float(np.nanquantile(pair, 0.10)),
                        "mean_pairwise_flip_rate": float(1.0 - np.nanmean(pair)),
                    }
                )
            for bs in bs_values:
                for k_idx in sorted(topk_summary_by_metric[metric_name][bs].keys()):
                    vals = np.asarray(topk_summary_by_metric[metric_name][bs][k_idx], dtype=np.float64)
                    topk_summary_rows.append(
                        {
                            "metric": metric_name,
                            "bs": int(bs),
                            "k": int(k_idx),
                            "n_population_repeats": int(vals.size),
                            "mean_topk_overlap": float(np.nanmean(vals)),
                            "std_topk_overlap": float(np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0),
                            "p10_topk_overlap": float(np.nanquantile(vals, 0.10)),
                            "median_topk_overlap": float(np.nanmedian(vals)),
                        }
                    )
            recommended_bs[metric_name] = _recommend_bs(summary_rows, metric_name, getattr(cfg, "experiment", {}))

        _save_csv(save_dir / "ranking_repeats.csv", all_repeat_rows)
        _save_csv(save_dir / "topk_repeats.csv", topk_repeat_rows)
        _save_csv(save_dir / "bs_summary.csv", summary_rows)
        _save_csv(save_dir / "topk_summary.csv", topk_summary_rows)
        _save_csv(save_dir / "reference_populations.csv", reference_rows)
        _write_json(save_dir / "selected_generations.json", generation_meta)
        _write_json(
            save_dir / "summary.json",
            {
                "metrics": metric_names,
                "run_labels": run_label_list,
                "n_runs": int(len(run_label_list)),
                "n_selected_populations": int(len({(r['run_label'], r['generation_idx']) for r in reference_rows})),
                "bs_ref": int(bs_ref),
                "bs_values": [int(x) for x in bs_values],
                "n_repeats": int(n_repeats),
                "rollout_steps": int(rollout_steps),
                "recommended_bs": {k: v for k, v in recommended_bs.items()},
                "eval_summaries": eval_summaries,
                "resume_enabled": bool(resume),
                "resumed_generations": int(resumed_generations),
                "resumed_partial_generations": int(resumed_partial_generations),
                "computed_generations": int(computed_generations),
            },
        )

        comparison_fig = _plot_metric_comparison(summary_rows, save_dir / "ranking_stability_comparison.png") if summary_rows else None
        if comparison_fig is not None:
            run.log({"stability/ranking_stability_comparison": wandb.Image(comparison_fig)})
            plt.close(comparison_fig)

        for metric_name in metric_names:
            metric_summary_fig = _plot_metric_summary(
                summary_rows,
                metric_name,
                save_dir / f"ranking_stability_vs_bs__{metric_name}.png",
            ) if summary_rows else None
            if metric_summary_fig is not None:
                run.log({f"stability/{metric_name}_ranking_stability": wandb.Image(metric_summary_fig)})
                plt.close(metric_summary_fig)

            metric_topk_fig = _plot_topk_heatmap(
                topk_summary_rows,
                metric_name,
                save_dir / f"topk_overlap_heatmap__{metric_name}.png",
            ) if topk_summary_rows else None
            if metric_topk_fig is not None:
                run.log({f"stability/{metric_name}_topk_overlap_heatmap": wandb.Image(metric_topk_fig)})
                plt.close(metric_topk_fig)

            metric_ref_fig = _plot_reference_spread(
                reference_rows,
                metric_name,
                save_dir / f"reference_population_scores__{metric_name}.png",
            ) if reference_rows else None
            if metric_ref_fig is not None:
                run.log({f"stability/{metric_name}_reference_scores": wandb.Image(metric_ref_fig)})
                plt.close(metric_ref_fig)

        if summary_rows:
            table = wandb.Table(
                columns=list(summary_rows[0].keys()),
                data=[[row[col] for col in summary_rows[0].keys()] for row in summary_rows],
            )
            run.log({"stability/bs_summary": table})
        if topk_summary_rows:
            topk_table = wandb.Table(
                columns=list(topk_summary_rows[0].keys()),
                data=[[row[col] for col in topk_summary_rows[0].keys()] for row in topk_summary_rows],
            )
            run.log({"stability/topk_summary": topk_table})

        run.summary["stability/n_runs"] = int(len(run_label_list))
        run.summary["stability/n_selected_populations"] = int(len({(r['run_label'], r['generation_idx']) for r in reference_rows}))
        run.summary["stability/bs_ref"] = int(bs_ref)
        run.summary["stability/resumed_generations"] = int(resumed_generations)
        run.summary["stability/resumed_partial_generations"] = int(resumed_partial_generations)
        run.summary["stability/computed_generations"] = int(computed_generations)
        for metric_name, rec in recommended_bs.items():
            if rec is not None:
                run.summary[f"stability/recommended_bs_{metric_name}"] = int(rec)

        print(f"Selected populations: {len({(r['run_label'], r['generation_idx']) for r in reference_rows})} across {len(run_label_list)} runs")
        print(f"Reference bs_ref={bs_ref}, repeated evals per bs={n_repeats}")
        print(
            f"Resume: enabled={resume}, resumed_generations={resumed_generations}, "
            f"resumed_partial_generations={resumed_partial_generations}, computed_generations={computed_generations}"
        )
        for metric_name in metric_names:
            print(f"Metric: {metric_name}")
            print(f"{'bs':>4} {'spearman':>10} {'top1':>8} {'all-k':>8} {'pairwise':>10} {'flip':>8}")
            for row in [r for r in summary_rows if r['metric'] == metric_name]:
                print(
                    f"{int(row['bs']):4d} "
                    f"{float(row['mean_spearman_rho']):10.4f} "
                    f"{float(row['mean_top1_match']):8.4f} "
                    f"{float(row['mean_topk_overlap_allk']):8.4f} "
                    f"{float(row['mean_pairwise_agreement']):10.4f} "
                    f"{float(row['mean_pairwise_flip_rate']):8.4f}"
                )
            if recommended_bs[metric_name] is not None:
                print(f"Recommended bs ({metric_name}): {recommended_bs[metric_name]}")
            else:
                print(f"Recommended bs ({metric_name}): none satisfied configured thresholds.")
    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
