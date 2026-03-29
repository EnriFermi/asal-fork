#!/usr/bin/env python3

import argparse
import math
import pickle
import sys
from pathlib import Path

try:
    import numpy  # noqa: F401
except Exception as exc:
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Find all best.pkl files under a repo and print parameter counts."
    )
    ap.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="Root directory to scan. Defaults to the repo root.",
    )
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="Print absolute paths instead of paths relative to root.",
    )
    return ap.parse_args()


def iter_best_paths(root: Path):
    for path in sorted(root.rglob("best.pkl")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def load_param_count(path: Path) -> int:
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        params = obj[0]
    elif isinstance(obj, dict) and "params" in obj:
        params = obj["params"]
    else:
        raise ValueError(f"Unsupported best.pkl payload type: {type(obj).__name__}")

    return infer_param_count(params)


def infer_param_count(params) -> int:
    size = getattr(params, "size", None)
    if size is not None:
        try:
            return int(size)
        except Exception:
            pass

    shape = getattr(params, "shape", None)
    if shape is not None:
        try:
            return int(math.prod(int(dim) for dim in shape))
        except Exception:
            pass

    if isinstance(params, dict):
        return sum(infer_param_count(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(infer_param_count(v) for v in params)
    return 1


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    if NUMPY_IMPORT_ERROR is not None:
        raise SystemExit(
            "This script needs numpy installed because best.pkl checkpoints were pickled with numpy arrays. "
            f"Original import error: {NUMPY_IMPORT_ERROR!r}"
        )

    rows = []
    failures = []
    for path in iter_best_paths(root):
        try:
            n_params = load_param_count(path)
            rows.append((path, n_params))
        except Exception as exc:
            failures.append((path, repr(exc)))

    if not rows and not failures:
        print(f"No best.pkl files found under {root}")
        return 0

    width = max(
        [len("n_params")]
        + [len(str(n_params)) for _, n_params in rows]
    )
    print(f"{'n_params'.rjust(width)}  path")
    for path, n_params in rows:
        shown = str(path if args.absolute else path.relative_to(root))
        print(f"{str(n_params).rjust(width)}  {shown}")

    if failures:
        print("\nFailed to read:")
        for path, message in failures:
            shown = str(path if args.absolute else path.relative_to(root))
            print(f"- {shown}: {message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
