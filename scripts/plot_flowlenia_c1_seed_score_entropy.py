#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _candidate_key(row: dict[str, str]) -> tuple[int, str, int]:
    return (
        _int(row, "optimized_run_idx"),
        row["candidate_kind"],
        _int(row, "candidate_idx"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metric", default="eval_score_mspd")
    args = parser.parse_args()

    with args.scores_csv.open(newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    if not input_rows or args.metric not in input_rows[0]:
        raise ValueError(f"Missing metric {args.metric!r} in {args.scores_csv}")

    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = {}
    for row in input_rows:
        grouped.setdefault(_candidate_key(row), []).append(row)

    entropy_rows: list[dict[str, object]] = []
    for (group_idx, candidate_kind, candidate_idx), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: _int(row, "rollout_seed_idx"))
        seed_indices = [_int(row, "rollout_seed_idx") for row in rows]
        scores = np.asarray([float(row[args.metric]) for row in rows], dtype=np.float64)
        if len(seed_indices) != 4 or seed_indices != [0, 1, 2, 3]:
            raise ValueError(
                f"Expected seed_idx 0..3 for group={group_idx}, kind={candidate_kind}, "
                f"candidate={candidate_idx}; got {seed_indices}"
            )
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or float(scores.sum()) <= 0.0:
            raise ValueError(
                f"Entropy requires finite nonnegative scores with positive sum: "
                f"group={group_idx}, kind={candidate_kind}, candidate={candidate_idx}, scores={scores}"
            )
        probabilities = scores / float(scores.sum())
        positive = probabilities > 0.0
        entropy_nats = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        normalized_entropy = float(entropy_nats / math.log(len(probabilities)))
        entropy_rows.append(
            {
                "optimized_run_idx": group_idx,
                "candidate_kind": candidate_kind,
                "candidate_idx": candidate_idx,
                "candidate_label": "optimized" if candidate_kind == "optimized" else f"random_{candidate_idx}",
                "metric": args.metric,
                "n_seeds": len(seed_indices),
                "seed_indices": json.dumps(seed_indices),
                "scores": json.dumps(scores.tolist()),
                "probabilities": json.dumps(probabilities.tolist()),
                "entropy_nats": entropy_nats,
                "normalized_entropy": normalized_entropy,
                "effective_seed_count": float(math.exp(entropy_nats)),
            }
        )

    expected_count = len({_int(row, "optimized_run_idx") for row in input_rows}) * 4
    if len(entropy_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} candidate entropy rows, got {len(entropy_rows)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entropy_rows[0]))
        writer.writeheader()
        writer.writerows(entropy_rows)

    groups = sorted({int(row["optimized_run_idx"]) for row in entropy_rows})
    series = (
        ("optimized", "optimized", 0, "#d62728", "o", "-"),
        ("random 0", "random", 0, "#1f77b4", "s", "--"),
        ("random 1", "random", 1, "#2a9d8f", "^", "--"),
        ("random 2", "random", 2, "#9467bd", "D", "--"),
    )
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    all_values: list[float] = []
    for label, kind, candidate_idx, color, marker, linestyle in series:
        values = []
        for group_idx in groups:
            matches = [
                row
                for row in entropy_rows
                if int(row["optimized_run_idx"]) == group_idx
                and row["candidate_kind"] == kind
                and int(row["candidate_idx"]) == candidate_idx
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one entropy row for group={group_idx}, kind={kind}, candidate={candidate_idx}"
                )
            values.append(float(matches[0]["normalized_entropy"]))
        all_values.extend(values)
        ax.plot(
            np.arange(len(groups)),
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6,
        )

    lower = max(0.0, min(all_values) - 0.05)
    ax.set_ylim(lower, 1.015)
    ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_xticks(np.arange(len(groups)), [f"opt_{group:03d}" for group in groups])
    ax.set_xlabel("matched optimization group")
    ax.set_ylabel("normalized seed-score entropy")
    ax.set_title("Flow-Lenia C1 eval MSPD entropy across seeds")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.text(
        0.995,
        0.03,
        "H / log(4),  p(seed) = MSPD(seed) / sum MSPD",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "output_figure": str(args.output_figure.resolve()),
                "output_csv": str(args.output_csv.resolve()),
                "n_candidates": len(entropy_rows),
                "metric": args.metric,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
