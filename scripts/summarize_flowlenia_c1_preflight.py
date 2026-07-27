from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            return default
    return cur


def _pair_status(summary: dict[str, Any], path: str) -> str:
    item = _get(summary, path, None)
    if not isinstance(item, dict):
        return f"{path}: <missing>"
    parts = [f"{path}: status={item.get('status')}"]
    for key in (
        "first_failed_step",
        "first_failed_internal_step",
        "max_abs_xy_diff",
        "mean_abs_xy_diff",
        "final_max_abs_xy_diff",
        "final_mean_abs_xy_diff",
    ):
        if key in item:
            parts.append(f"{key}={item.get(key)}")
    return " ".join(parts)


def _print_heading(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _iter_closest_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for anchor, anchor_rows in value.items():
            if not isinstance(anchor_rows, list):
                continue
            for row in anchor_rows:
                if isinstance(row, dict):
                    rows.append({"anchor": anchor, **row})
        rows.sort(
            key=lambda row: (
                float(row.get("mean_abs_xy_diff", float("inf"))),
                float(row.get("max_abs_xy_diff", float("inf"))),
                str(row.get("anchor", "")),
                str(row.get("name", "")),
            )
        )
        return rows
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Flow-Lenia C1 replay preflight diagnostics.")
    parser.add_argument("json_path", help="Path to *_preflight_smoke_before_apf.json.")
    args = parser.parse_args()

    path = Path(args.json_path)
    data = json.loads(path.read_text())
    smoke = data.get("replay_smoke", {})

    print(f"file: {path}")
    print(f"summary_status: {data.get('status')}")
    print(f"summary_errors: {data.get('errors')}")
    if data.get("warnings"):
        print(f"summary_warnings: {data.get('warnings')}")
    print(f"smoke_status: {smoke.get('status')} strict_status={smoke.get('strict_status')}")
    print(f"known_execution_divergence: {smoke.get('known_execution_divergence')}")
    print(f"run_seed: {smoke.get('run_seed')} run_idx={smoke.get('run_idx')} seed_idx={smoke.get('seed_idx')}")
    print(f"first_failed_step: {smoke.get('first_failed_step')}")
    print(f"initial_max_abs_xy_diff: {smoke.get('initial_max_abs_xy_diff')}")
    print(f"max_abs_xy_diff: {smoke.get('max_abs_xy_diff')}")
    print(f"mean_abs_xy_diff: {smoke.get('mean_abs_xy_diff')}")

    _print_heading("Config Diffs")
    print(f"protocol_diffs: {smoke.get('rollout_vs_optimization_config_diffs')}")
    print(f"ignored_diffs: {smoke.get('rollout_vs_optimization_ignored_config_diffs')}")

    _print_heading("One Step")
    one = smoke.get("one_step_diagnostic", {})
    print(f"initial_diff: {one.get('initial_diff')}")
    print(f"after_one_step_diff: {one.get('after_one_step_diff')}")

    _print_heading("First Chunk Trace")
    first_chunk = _get(smoke, "diagnostic_pack.first_chunk_trace", {})
    print(f"trace_status: {first_chunk.get('status')}")
    for name, item in sorted((first_chunk.get("pairwise_trace") or {}).items()):
        print(_pair_status({"x": item}, "x").replace("x:", f"{name}:"))

    _print_heading("Full Rollout Closest Pairs")
    for row in _iter_closest_rows(_get(smoke, "diagnostic_pack.full_rollout_pairwise_xy.closest_by_mean_abs", []))[:12]:
        print(row)

    _print_heading("First Chunk Closest Pairs")
    for row in _iter_closest_rows(_get(smoke, "diagnostic_pack.first_chunk_pairwise_xy.closest_by_mean_abs", []))[:12]:
        print(row)

    _print_heading("APF Style Chunk")
    for key in (
        "no_capture_vs_single_selected_apf",
        "with_capture_vs_single_selected_apf",
        "no_capture_vs_eager_reference",
        "with_capture_vs_eager_reference",
        "no_capture_vs_flat_jit_reference",
        "with_capture_vs_flat_jit_reference",
    ):
        print(_pair_status(smoke, f"apf_style_chunk_reference.{key}"))

    _print_heading("JIT References")
    for key in (
        "nested_jit_vs_nested_eager",
        "nested_jit_vs_single_selected_apf",
        "flat_jit_vs_flat_eager",
        "flat_jit_vs_single_selected_apf",
        "flat_jit_vs_nested_jit",
    ):
        print(_pair_status(smoke, f"jit_reference.{key}"))

    _print_heading("Optimizer Context APF")
    context = smoke.get("optimizer_context_apf", {})
    print(f"context_status: {context.get('status')} selected_seed_int={context.get('selected_seed_int')}")
    for prefix in ("same_seed_pop_batch", "all_seeds_pop_grid"):
        print(_pair_status(context, f"{prefix}.initial_snapshot_diff_vs_optimizer_reference"))
        print(_pair_status(context, f"{prefix}.diff_vs_optimizer_reference"))
        print(_pair_status(context, f"{prefix}.diff_vs_single_selected_apf"))
        print(_pair_status(context, f"{prefix}.diff_vs_scalar_reference"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
