from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import pair_type


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _prepare_embeddings(z: np.ndarray, normalize: bool = True) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must have shape (T, D), got {arr.shape}.")
    return _normalize_rows(arr) if normalize else arr


def synchronized_distance(z_a: np.ndarray, z_b: np.ndarray, metric: str = "cosine") -> float:
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    t = min(a.shape[0], b.shape[0])
    if t < 1:
        raise ValueError("Embedding sequences must contain at least one frame.")
    a = a[:t]
    b = b[:t]
    metric = str(metric).strip().lower()
    if metric == "cosine":
        return float(np.mean(1.0 - np.sum(a * b, axis=-1)))
    if metric == "euclidean":
        return float(np.mean(np.linalg.norm(a - b, axis=-1)))
    if metric == "sqeuclidean":
        diff = a - b
        return float(np.mean(np.sum(diff * diff, axis=-1)))
    raise ValueError(f"Unsupported synchronized embedding metric={metric!r}.")


def _cross_distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return 1.0 - (a @ b.T)
    if metric == "euclidean":
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        sq = np.clip(aa + bb - 2.0 * (a @ b.T), 0.0, None)
        return np.sqrt(sq)
    raise ValueError(f"Unsupported cloud embedding metric={metric!r}.")


def cloud_distance(
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    metric: str = "cosine",
    method: str = "chamfer",
) -> float:
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    d = _cross_distances(a, b, metric=str(metric).strip().lower())
    method = str(method).strip().lower()
    if method == "chamfer":
        return float(0.5 * (np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))))
    if method == "mean_cross":
        return float(np.mean(d))
    raise ValueError(f"Unsupported embedding cloud method={method!r}.")


def compute_embedding_pairwise(
    runs: pd.DataFrame,
    load_embeddings_fn,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    emb_cfg = dict(cfg.get("embeddings", {}))
    normalize = bool(emb_cfg.get("normalize", True))
    synced_metrics = [str(x) for x in emb_cfg.get("synced_metrics", ["cosine"])]
    cloud_metrics = [str(x) for x in emb_cfg.get("cloud_metrics", ["cosine"])]
    cloud_method = str(emb_cfg.get("cloud_method", "chamfer"))

    available = runs[runs["has_embeddings"]].copy().reset_index(drop=True)
    if available.empty:
        return pd.DataFrame(), {}

    prepared = {
        row["run_id"]: _prepare_embeddings(load_embeddings_fn(row), normalize=normalize)
        for _, row in available.iterrows()
    }
    n_runs = int(available.shape[0])
    run_ids = available["run_id"].tolist()

    matrix_names = [f"synced_{metric}" for metric in synced_metrics]
    matrix_names.extend(f"cloud_{cloud_method}_{metric}" for metric in cloud_metrics)
    matrices = {
        name: np.zeros((n_runs, n_runs), dtype=np.float64)
        for name in matrix_names
    }

    pair_rows: list[dict[str, Any]] = []
    for i in range(n_runs):
        row_i = available.iloc[i]
        zi = prepared[row_i["run_id"]]
        for j in range(i + 1, n_runs):
            row_j = available.iloc[j]
            zj = prepared[row_j["run_id"]]
            record = {
                "run_a": row_i["run_id"],
                "run_b": row_j["run_id"],
                "condition_a": row_i["condition"],
                "condition_b": row_j["condition"],
                "pair_type": pair_type(row_i["condition"], row_j["condition"]),
                "pair_group_a": row_i["pair_group_id"],
                "pair_group_b": row_j["pair_group_id"],
                "same_pair_group": bool(row_i["pair_group_id"] == row_j["pair_group_id"]),
            }
            for metric in synced_metrics:
                value = synchronized_distance(zi, zj, metric=metric)
                name = f"synced_{metric}"
                matrices[name][i, j] = value
                matrices[name][j, i] = value
                record[f"embedding_{name}"] = value
            for metric in cloud_metrics:
                value = cloud_distance(zi, zj, metric=metric, method=cloud_method)
                name = f"cloud_{cloud_method}_{metric}"
                matrices[name][i, j] = value
                matrices[name][j, i] = value
                record[f"embedding_{name}"] = value
            pair_rows.append(record)

    matrix_frames = {
        name: pd.DataFrame(value, index=run_ids, columns=run_ids)
        for name, value in matrices.items()
    }
    pair_frame = pd.DataFrame(pair_rows)
    return pair_frame, matrix_frames
