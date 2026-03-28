import argparse
import csv
from pathlib import Path
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_group_name(group_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(group_name)).strip("_") or "group"


def _candidate_groups(candidate_labels: np.ndarray) -> list[tuple[str, np.ndarray]]:
    labels = np.asarray(candidate_labels).astype(str)
    groups = [("all", np.ones(labels.shape, dtype=bool))]
    seen = set()
    for label in labels.tolist():
        if label in seen:
            continue
        seen.add(label)
        mask = labels == label
        if np.any(mask):
            groups.append((label, mask))
    return groups


def _plot_metric_summary(rows: list[dict], metric_name: str, group_name: str, out_path: Path):
    if plt is None:
        return
    rows = [r for r in rows if r["metric"] == metric_name and r["group"] == group_name]
    if not rows:
        return
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
        ax.set_title(f"{metric_name} [{group_name}]: {title}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_comparison(rows: list[dict], group_name: str, out_path: Path):
    if plt is None:
        return
    rows = [r for r in rows if r["group"] == group_name]
    if not rows:
        return
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
        ax.set_title(f"{title} [{group_name}]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_topk_heatmap(rows: list[dict], metric_name: str, group_name: str, out_path: Path):
    if plt is None:
        return
    rows = [r for r in rows if r["metric"] == metric_name and r["group"] == group_name]
    if not rows:
        return
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
        ax.set_title(f"{metric_name} [{group_name}]: {title}")
        ax.set_xlabel("k")
        ax.set_ylabel("bs")
        ax.set_xticks(np.arange(len(k_vals)))
        ax.set_xticklabels(k_vals)
        ax.set_yticks(np.arange(len(bs_vals)))
        ax.set_yticklabels(bs_vals)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reference_scores(
    ref_scores: np.ndarray,
    candidate_labels: np.ndarray,
    metric_name: str,
    group_name: str,
    out_path: Path,
):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(ref_scores.size)
    ax.bar(xs, ref_scores, color="#4C78A8", alpha=0.9)
    ax.set_title(f"{metric_name} [{group_name}]: reference scores")
    ax.set_ylabel("reference loss")
    ax.set_xticks(xs)
    ax.set_xticklabels(candidate_labels.tolist(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_metric_names(data: np.lib.npyio.NpzFile) -> list[str]:
    if "meta__metric_names" in data.files:
        return sorted(str(x) for x in np.asarray(data["meta__metric_names"]).tolist())
    metric_names = []
    for key in data.files:
        if key.endswith("__score_matrix_single"):
            metric_names.append(key.split("__", 1)[0])
    return sorted(set(metric_names))


def main():
    ap = argparse.ArgumentParser(description="Plot ranking-stability graphs from a raw cache NPZ.")
    ap.add_argument("npz_path", type=str, help="Path to raw cache .npz file")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to alongside input)")
    ap.add_argument("--prefix", type=str, default=None, help="Filename prefix (defaults to input stem)")
    args = ap.parse_args()

    npz_path = Path(args.npz_path).resolve()
    out_dir = npz_path.parent if args.out_dir is None else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = npz_path.stem if args.prefix is None else str(args.prefix)

    with np.load(npz_path, allow_pickle=False) as data:
        metric_names = _load_metric_names(data)
        if not metric_names:
            raise ValueError(f"No metrics found in {npz_path}.")

        bs_ref = int(np.asarray(data["meta__bs_ref"]).reshape(()))
        bs_values = sorted(int(x) for x in np.asarray(data["meta__bs_values"], dtype=np.int32).tolist())
        n_repeats = int(np.asarray(data["meta__n_repeats"]).reshape(()))
        candidate_indices = np.asarray(data["meta__candidate_indices"], dtype=np.int32)
        if "meta__candidate_labels" in data.files:
            candidate_labels = np.asarray(data["meta__candidate_labels"]).astype(str)
        else:
            candidate_labels = np.asarray([str(int(x)) for x in candidate_indices.tolist()], dtype="<U16")
        candidate_groups = _candidate_groups(candidate_labels)

        summary_rows: list[dict] = []
        topk_rows: list[dict] = []
        reference_rows: list[dict] = []

        for metric_name in metric_names:
            ref_key = f"{metric_name}__ref_scores"
            if ref_key not in data.files:
                raise ValueError(f"{npz_path} missing {ref_key}.")
            ref_scores = np.asarray(data[ref_key], dtype=np.float32)
            reference_rows.extend(
                {
                    "metric": metric_name,
                    "candidate_rank": int(i),
                    "candidate_index": int(candidate_indices[i]),
                    "candidate_label": str(candidate_labels[i]),
                    "ref_score": float(ref_scores[i]),
                }
                for i in range(ref_scores.size)
            )

            for group_name, group_mask in candidate_groups:
                group_ref_scores = ref_scores[group_mask]
                group_candidate_labels = candidate_labels[group_mask]
                group_suffix = "" if group_name == "all" else f"__group_{_sanitize_group_name(group_name)}"

                _plot_reference_scores(
                    ref_scores=group_ref_scores,
                    candidate_labels=group_candidate_labels,
                    metric_name=metric_name,
                    group_name=group_name,
                    out_path=out_dir / f"{prefix}__reference_scores__{metric_name}{group_suffix}.png",
                )

                for bs in bs_values:
                    key = f"{metric_name}__bs_{bs}_mean_scores"
                    if key not in data.files:
                        raise ValueError(f"{npz_path} missing {key}.")
                    est_scores = np.asarray(data[key], dtype=np.float32)
                    if est_scores.ndim != 2:
                        raise ValueError(f"{key} must have shape (n_repeats, n_candidates), got {est_scores.shape}.")

                    group_est_scores = est_scores[:, group_mask]
                    metrics_list = []
                    topk_curves = []
                    for rep_idx in range(group_est_scores.shape[0]):
                        metrics, topk_curve = _ranking_metrics(group_ref_scores, group_est_scores[rep_idx])
                        metrics_list.append(metrics)
                        topk_curves.append(topk_curve)

                    def arr(name: str) -> np.ndarray:
                        return np.asarray([float(m[name]) for m in metrics_list], dtype=np.float64)

                    spearman = arr("spearman_rho")
                    top1 = arr("top1_match")
                    topk_allk = arr("mean_topk_overlap_allk")
                    pair = arr("pairwise_agreement")
                    summary_rows.append(
                        {
                            "metric": metric_name,
                            "group": group_name,
                            "bs": int(bs),
                            "bs_ref": int(bs_ref),
                            "n_repeats": int(group_est_scores.shape[0]),
                            "n_candidates": int(group_ref_scores.size),
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
                        }
                    )

                    topk_curves_arr = np.asarray(topk_curves, dtype=np.float64)
                    for k_idx in range(topk_curves_arr.shape[1]):
                        vals = topk_curves_arr[:, k_idx]
                        topk_rows.append(
                            {
                                "metric": metric_name,
                                "group": group_name,
                                "bs": int(bs),
                                "k": int(k_idx + 1),
                                "mean_topk_overlap": float(np.nanmean(vals)),
                                "std_topk_overlap": float(np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0),
                                "p10_topk_overlap": float(np.nanquantile(vals, 0.10)),
                                "median_topk_overlap": float(np.nanmedian(vals)),
                            }
                        )

        _save_csv(out_dir / f"{prefix}__bs_summary.csv", summary_rows)
        _save_csv(out_dir / f"{prefix}__topk_summary.csv", topk_rows)
        _save_csv(out_dir / f"{prefix}__reference_scores.csv", reference_rows)

        if plt is not None:
            for group_name, _ in candidate_groups:
                group_suffix = "" if group_name == "all" else f"__group_{_sanitize_group_name(group_name)}"
                for metric_name in metric_names:
                    _plot_metric_summary(
                        summary_rows,
                        metric_name,
                        group_name,
                        out_dir / f"{prefix}__ranking_stability_vs_bs__{metric_name}{group_suffix}.png",
                    )
                    _plot_topk_heatmap(
                        topk_rows,
                        metric_name,
                        group_name,
                        out_dir / f"{prefix}__topk_overlap_heatmap__{metric_name}{group_suffix}.png",
                    )

                if len(metric_names) > 1:
                    _plot_metric_comparison(
                        summary_rows,
                        group_name,
                        out_dir / f"{prefix}__ranking_stability_comparison{group_suffix}.png",
                    )

        print(f"Saved outputs to {out_dir}")
        print(f"metrics={metric_names}, bs_ref={bs_ref}, bs_values={bs_values}, n_repeats={n_repeats}, n_candidates={candidate_indices.size}")
        print(f"groups={[name for name, _ in candidate_groups]}")
        if plt is None:
            print("matplotlib is not installed; wrote CSV outputs only.")


if __name__ == "__main__":
    main()
