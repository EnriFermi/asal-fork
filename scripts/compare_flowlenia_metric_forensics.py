from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATAFLOW_ORDER = (
    "dir_key",
    "dirs_raw",
    "dirs",
    "key_k",
    "key_p",
    "key_null",
    "k_idx",
    "p_idx",
    "X0",
    "X1",
    "dx",
    "dt",
    "v_s",
    "proj",
    "proj_sorted",
    "sig",
    "pairwise_real",
    "h_real",
    "null_keys",
    "null_idx",
    "null_proj",
    "null_proj_sorted",
    "null_sig",
    "null_pairwise",
    "h0",
    "h_null",
    "delta_h",
)


def _compare_array(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
    }
    if a.shape != b.shape or a.dtype != b.dtype:
        result["equal"] = False
        result["structural_mismatch"] = True
        return result
    different = a != b
    count = int(np.count_nonzero(different))
    result["equal"] = count == 0
    result["different_values"] = count
    if count:
        index = tuple(int(i) for i in np.argwhere(different)[0]) if a.ndim else ()
        result["first_index"] = list(index)
        result["first_a"] = a[index].item()
        result["first_b"] = b[index].item()
        if np.issubdtype(a.dtype, np.number):
            delta = b.astype(np.float64) - a.astype(np.float64)
            result["first_delta"] = float(delta[index])
            result["max_abs_diff"] = float(np.max(np.abs(delta)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_a")
    parser.add_argument("capture_b")
    parser.add_argument("--output")
    args = parser.parse_args()

    dir_a = Path(args.capture_a).resolve()
    dir_b = Path(args.capture_b).resolve()
    with np.load(dir_a / "staged_window.npz", allow_pickle=False) as a, np.load(
        dir_b / "staged_window.npz", allow_pickle=False
    ) as b:
        fields: dict[str, object] = {}
        first_divergence = None
        for key in DATAFLOW_ORDER:
            comparison = _compare_array(a[key], b[key])
            fields[key] = comparison
            if first_divergence is None and not comparison["equal"]:
                first_divergence = key

    report = {
        "capture_a": str(dir_a),
        "capture_b": str(dir_b),
        "stablehlo_equal": (dir_a / "exact_metric.stablehlo.mlir").read_bytes()
        == (dir_b / "exact_metric.stablehlo.mlir").read_bytes(),
        "hlo_equal": (dir_a / "exact_metric.hlo.txt").read_bytes()
        == (dir_b / "exact_metric.hlo.txt").read_bytes(),
        "first_dataflow_divergence": first_divergence,
        "fields": fields,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
