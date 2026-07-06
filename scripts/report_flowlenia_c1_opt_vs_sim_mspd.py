from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SELECTED_ROOT = (
    "experiments/paper_check_flow_lenia/"
    "checkpoints_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/optimization"
)
DEFAULT_SCORES_CSV = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
    "flow_lenia/checkpoint_scores.csv"
)
DEFAULT_OUTPUT_CSV = (
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_9opt_completed_robust_c1_3random/"
    "flow_lenia/optimized_opt_vs_sim_mspd.csv"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_idx_from_name(path: Path) -> int | None:
    match = re.fullmatch(r"run_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _to_float(raw: Any) -> float:
    if raw is None:
        return float("nan")
    try:
        if isinstance(raw, str) and raw.strip() == "":
            return float("nan")
        return float(raw)
    except Exception:
        return float("nan")


def _to_int(raw: Any) -> int | None:
    val = _to_float(raw)
    if not math.isfinite(val):
        return None
    return int(val)


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _stats(prefix: str, values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _format_float(raw: Any) -> str:
    val = _to_float(raw)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.6g}"


def _format_list(values: list[float] | np.ndarray) -> str:
    arr = _finite(values)
    return ";".join(f"{float(v):.9g}" for v in arr)


def _tau_steps(meta: dict[str, Any]) -> int | None:
    tau = meta.get("tau")
    if isinstance(tau, dict):
        value = _to_int(tau.get("tau_steps"))
        if value is not None:
            return value
    return _to_int(meta.get("tau_steps"))


def _load_selected_runs(selected_root: Path) -> dict[int, dict[str, Any]]:
    if not selected_root.exists():
        raise FileNotFoundError(f"Selected optimization root not found: {selected_root}")
    out: dict[int, dict[str, Any]] = {}
    for run_dir in sorted(selected_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        run_idx = _run_idx_from_name(run_dir)
        if run_idx is None:
            continue
        selected_path = run_dir / "selected_candidate.json"
        if not selected_path.exists():
            raise FileNotFoundError(f"Missing selected_candidate.json for {run_dir}: {selected_path}")
        meta = _load_json(selected_path)
        out[run_idx] = {"run_dir": run_dir, "selected_path": selected_path, "meta": meta}
    if not out:
        raise ValueError(f"No run_*/selected_candidate.json files found under {selected_root}")
    return out


def _group_optimized_scores(score_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in score_rows:
        if str(row.get("candidate_kind", "")).strip() != "optimized":
            continue
        # selected_root/run_XXX is keyed by the original optimization run id.
        # In completed-run subsets, optimized_run_idx is a dense suite index and
        # can refer to a different original run.
        run_idx = _to_int(row.get("source_optimized_run_idx"))
        if run_idx is None:
            run_idx = _to_int(row.get("optimized_run_idx"))
        if run_idx is None:
            continue
        grouped.setdefault(run_idx, []).append(row)
    return grouped


def _summarize_one(run_idx: int, selected: dict[str, Any], sim_rows: list[dict[str, str]]) -> dict[str, Any]:
    meta = selected["meta"]
    opt_seed_scores = [_to_float(v) for v in meta.get("seed_scores_mspd", [])]
    opt_score = _to_float(meta.get("score_mspd"))
    if not math.isfinite(opt_score):
        opt_score = float(np.mean(_finite(opt_seed_scores))) if _finite(opt_seed_scores).size else float("nan")

    eval_values = [_to_float(row.get("eval_score_mspd")) for row in sim_rows]
    full_train_values = [_to_float(row.get("full_score_train_tau_mspd")) for row in sim_rows]
    full_max_values = [_to_float(row.get("full_score_max_mspd")) for row in sim_rows]
    eval_max_values = [_to_float(row.get("eval_score_max_mspd")) for row in sim_rows]
    rollout_seed_idx = [_to_int(row.get("rollout_seed_idx")) for row in sim_rows]
    rollout_seed_idx = [x for x in rollout_seed_idx if x is not None]
    run_seed_values = [_to_int(row.get("run_seed")) for row in sim_rows]
    run_seed_values = [x for x in run_seed_values if x is not None]
    suite_run_values = [_to_int(row.get("optimized_run_idx")) for row in sim_rows]
    suite_run_values = [x for x in suite_run_values if x is not None]
    source_run_values = [_to_int(row.get("source_optimized_run_idx")) for row in sim_rows]
    source_run_values = [x for x in source_run_values if x is not None]

    sim_by_seed = {}
    for sim_row in sim_rows:
        seed_idx = _to_int(sim_row.get("rollout_seed_idx"))
        if seed_idx is None:
            continue
        sim_by_seed[int(seed_idx)] = _to_float(sim_row.get("full_score_train_tau_mspd"))
    common_seed_diffs = []
    for seed_idx, opt_val in enumerate(opt_seed_scores):
        sim_val = sim_by_seed.get(seed_idx, float("nan"))
        if math.isfinite(opt_val) and math.isfinite(sim_val):
            common_seed_diffs.append(float(opt_val) - float(sim_val))

    row: dict[str, Any] = {
        "run": f"run_{run_idx:03d}",
        "run_idx": run_idx,
        "selected_iter": meta.get("iter", ""),
        "selected_pop_idx": meta.get("pop_idx", ""),
        "selected_tau_steps": _tau_steps(meta),
        "optimizer_score_mspd": opt_score,
        "optimizer_seed_scores_mspd": _format_list(opt_seed_scores),
        "optimizer_seed_lcb_mspd": meta.get("seed_lcb_mspd", np.nan),
        "optimizer_seed_std_mspd": meta.get("seed_std_mspd", np.nan),
        "optimizer_seed_min_mspd": meta.get("seed_min_mspd", np.nan),
        "optimizer_seed_max_mspd": meta.get("seed_max_mspd", np.nan),
        "simulation_n_rows": len(sim_rows),
        "simulation_suite_optimized_run_idx": ";".join(str(x) for x in sorted(set(suite_run_values))),
        "simulation_source_optimized_run_idx": ";".join(str(x) for x in sorted(set(source_run_values))),
        "simulation_rollout_seed_idx": ";".join(str(x) for x in sorted(set(rollout_seed_idx))),
        "simulation_run_seeds": ";".join(str(x) for x in sorted(set(run_seed_values))),
        "simulation_eval_score_mspd_values": _format_list(eval_values),
        "simulation_full_score_train_tau_mspd_values": _format_list(full_train_values),
        "per_seed_optimizer_minus_simulation_full_train": _format_list(common_seed_diffs),
        "selected_candidate_json": str(selected["selected_path"]),
    }
    row.update(_stats("simulation_eval_score_mspd", eval_values))
    row.update(_stats("simulation_full_score_train_tau_mspd", full_train_values))
    row.update(_stats("simulation_full_score_max_mspd", full_max_values))
    row.update(_stats("simulation_eval_score_max_mspd", eval_max_values))

    sim_train_mean = _to_float(row.get("simulation_full_score_train_tau_mspd_mean"))
    sim_eval_mean = _to_float(row.get("simulation_eval_score_mspd_mean"))
    row["optimizer_minus_simulation_full_train_mean"] = (
        opt_score - sim_train_mean if math.isfinite(opt_score) and math.isfinite(sim_train_mean) else np.nan
    )
    row["optimizer_minus_simulation_eval_mean"] = (
        opt_score - sim_eval_mean if math.isfinite(opt_score) and math.isfinite(sim_eval_mean) else np.nan
    )
    if not sim_rows:
        status = "missing_simulation_rows"
    elif len(_finite(opt_seed_scores)) != len(_finite(full_train_values)):
        status = "seed_count_mismatch"
    elif len(set(source_run_values)) != 1 or int(source_run_values[0]) != int(run_idx):
        status = "source_run_mismatch"
    else:
        status = "ok"
    row["simulation_status"] = status
    return row


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "selected_iter",
        "selected_pop_idx",
        "selected_tau_steps",
        "optimizer_score_mspd",
        "simulation_full_score_train_tau_mspd_mean",
        "simulation_eval_score_mspd_mean",
        "optimizer_minus_simulation_full_train_mean",
        "optimizer_minus_simulation_eval_mean",
        "simulation_n_rows",
        "simulation_status",
    ]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            value = row.get(col, "")
            text = _format_float(value) if "mspd" in col or "minus" in col else str(value)
            widths[col] = max(widths[col], len(text))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            text = _format_float(value) if "mspd" in col or "minus" in col else str(value)
            cells.append(text.ljust(widths[col]))
        print("  ".join(cells))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = _load_selected_runs(Path(args.selected_root))
    scores_path = Path(args.scores_csv)
    if not scores_path.exists():
        raise FileNotFoundError(f"C1 checkpoint scores CSV not found: {scores_path}")
    sim_grouped = _group_optimized_scores(_read_csv(scores_path))

    rows = [_summarize_one(run_idx, selected[run_idx], sim_grouped.get(run_idx, [])) for run_idx in sorted(selected)]
    if args.strict and any(row["simulation_status"] != "ok" for row in rows):
        missing = ", ".join(row["run"] for row in rows if row["simulation_status"] != "ok")
        raise ValueError(f"Missing optimized simulation rows for: {missing}")
    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Flow-Lenia robust selected optimization MSPD against C1 posthoc simulation MSPD "
            "for completed fixed-init runs."
        )
    )
    parser.add_argument("--selected-root", default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--scores-csv", default=DEFAULT_SCORES_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--strict", action="store_true", help="Fail if any selected run has no optimized C1 rows.")
    parser.add_argument("--no-table", action="store_true", help="Only write CSV; do not print the stdout table.")
    args = parser.parse_args()

    rows = run(args)
    if not args.no_table:
        _print_table(rows)
        if args.output_csv:
            print(f"\nWrote: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
