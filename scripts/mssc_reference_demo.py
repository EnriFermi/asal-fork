from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "mssc_reference_square_spiral_demo"


@dataclass(frozen=True)
class MSSCTerm:
    scale_index: int
    fine_shape: tuple[int, int]
    coarse_shape: tuple[int, int]
    fine_self_overlap: float
    coarse_self_overlap: float
    cross_overlap: float
    complexity_term: float


@dataclass(frozen=True)
class MSSCResult:
    complexity: float
    block_size: int
    n_terms: int
    input_shape: tuple[int, ...]
    terms: tuple[MSSCTerm, ...]


def _as_channel_last_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D image or a channel-last 3D image, got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"Image is too small for MSSC: shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("MSSC input contains NaN or infinite values")
    return arr


def block_average(image: np.ndarray, *, block_size: int = 2) -> np.ndarray:
    """Dyadic Kadanoff-style block average used by the reference MSSC demo."""
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    arr = _as_channel_last_image(image)
    h, w, c = arr.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    if h_crop == 0 or w_crop == 0:
        raise ValueError(f"Image shape {arr.shape} is too small for block_size={block_size}")
    arr = arr[:h_crop, :w_crop, :]
    return arr.reshape(
        h_crop // block_size,
        block_size,
        w_crop // block_size,
        block_size,
        c,
    ).mean(axis=(1, 3))


def self_overlap(image: np.ndarray) -> float:
    arr = _as_channel_last_image(image)
    return float(np.mean(np.sum(arr * arr, axis=-1)))


def cross_scale_overlap(fine: np.ndarray, coarse: np.ndarray, *, block_size: int = 2) -> float:
    fine_arr = _as_channel_last_image(fine)
    coarse_arr = _as_channel_last_image(coarse)
    h = coarse_arr.shape[0] * block_size
    w = coarse_arr.shape[1] * block_size
    fine_arr = fine_arr[:h, :w, :]
    if fine_arr.shape[-1] != coarse_arr.shape[-1]:
        raise ValueError(
            f"Fine/coarse channel mismatch: {fine_arr.shape[-1]} vs {coarse_arr.shape[-1]}"
        )
    coarse_up = np.repeat(np.repeat(coarse_arr, block_size, axis=0), block_size, axis=1)
    return float(np.mean(np.sum(fine_arr * coarse_up, axis=-1)))


def mssc_complexity(
    image: np.ndarray,
    *,
    block_size: int = 2,
    max_terms: int | None = None,
) -> MSSCResult:
    """Compute the Bagrov-Iakovlev-Iliasov-Katsnelson-Mazurenko MSSC score.

    The implementation follows the reference overlap form:

        C_k = |O_{k+1,k} - 0.5 * (O_{k,k} + O_{k+1,k+1})|
        C   = sum_k C_k

    where scale k+1 is obtained from scale k by block averaging. Inputs are not
    normalized internally; scores are only comparable when images use the same
    value scale.
    """
    current = _as_channel_last_image(image)
    terms: list[MSSCTerm] = []
    scale_index = 0

    while current.shape[0] >= block_size and current.shape[1] >= block_size:
        if max_terms is not None and len(terms) >= max_terms:
            break
        coarse = block_average(current, block_size=block_size)
        fine_self = self_overlap(current[: coarse.shape[0] * block_size, : coarse.shape[1] * block_size, :])
        coarse_self = self_overlap(coarse)
        cross = cross_scale_overlap(current, coarse, block_size=block_size)
        term = abs(cross - 0.5 * (fine_self + coarse_self))
        terms.append(
            MSSCTerm(
                scale_index=scale_index,
                fine_shape=(int(current.shape[0]), int(current.shape[1])),
                coarse_shape=(int(coarse.shape[0]), int(coarse.shape[1])),
                fine_self_overlap=fine_self,
                coarse_self_overlap=coarse_self,
                cross_overlap=cross,
                complexity_term=float(term),
            )
        )
        current = coarse
        scale_index += 1

    return MSSCResult(
        complexity=float(sum(term.complexity_term for term in terms)),
        block_size=block_size,
        n_terms=len(terms),
        input_shape=tuple(int(x) for x in np.asarray(image).shape),
        terms=tuple(terms),
    )


def gaussian_noise(size: int, *, seed: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(loc=0.5, scale=sigma, size=(size, size)), 0.0, 1.0)


def uniform_image(size: int, *, value: float) -> np.ndarray:
    return np.full((size, size), float(value), dtype=np.float64)


def square_spiral_seed(base_size: int = 8) -> np.ndarray:
    """One-cell-thick square spiral mask used as the recursive fractal generator."""
    if base_size < 4:
        raise ValueError(f"base_size must be >= 4, got {base_size}")
    grid = np.zeros((base_size, base_size), dtype=np.float64)
    y = 0
    x = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_index = 0

    for segment_index, length in enumerate(range(base_size, 0, -1)):
        dy, dx = directions[direction_index % len(directions)]
        for step in range(length):
            if 0 <= y < base_size and 0 <= x < base_size:
                grid[y, x] = 1.0
            if step != length - 1:
                y += dy
                x += dx
        direction_index += 1
        if segment_index != base_size - 1:
            dy, dx = directions[direction_index % len(directions)]
            y += dy
            x += dx
    return grid


def square_spiral_fractal(size: int, *, base_size: int, levels: int) -> np.ndarray:
    """Recursive square spiral bitmap fractal, returned in [0, 1]."""
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    seed = square_spiral_seed(base_size)
    image = seed.copy()
    for _ in range(levels - 1):
        image = np.kron(image, seed)

    if image.shape[0] == size and image.shape[1] == size:
        return image
    indices_y = np.rint(np.linspace(0, image.shape[0] - 1, size)).astype(np.int64)
    indices_x = np.rint(np.linspace(0, image.shape[1] - 1, size)).astype(np.int64)
    return image[np.ix_(indices_y, indices_x)]


def _import_matplotlib(output_dir: Path):
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_single_panel(
    image: np.ndarray,
    *,
    label: str,
    complexity: float,
    output_path: Path,
    cmap: str,
) -> None:
    plt = _import_matplotlib(output_path.parent)
    fig, ax = plt.subplots(figsize=(4.0, 4.55), dpi=220)
    fig.patch.set_facecolor("white")
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.18)
    fig.text(
        0.5,
        0.085,
        f"{label}\nMSSC complexity = {complexity:.6f}",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def mspd_scale_profile(
    signal: np.ndarray,
    *,
    msc_floor: float = 0.01,
    eps: float = 1.0e-12,
) -> tuple[float, list[dict[str, float | int | str]]]:
    """Exact numpy version of the paper MSPD dyadic scale terms.

    This mirrors scripts/clip_deltah_msc_metric.py for
    metric_objective='mspd': h is clipped at zero, each adjacent dyadic pair
    r -> 2r is scored by floor-normalized reconstruction error, and equal
    scale weights are normalized by their sum.
    """
    h_pos = np.maximum(np.asarray(signal, dtype=np.float64).reshape(-1), 0.0)
    if h_pos.size < 4:
        raise ValueError(f"MSPD profile needs at least 4 samples, got {h_pos.size}")

    scales: list[int] = []
    r = 1
    while r <= h_pos.size // 2:
        scales.append(r)
        r *= 2
    scales_set = set(scales)
    scale_pairs = [r for r in scales if (2 * r) in scales_set]
    weight_denom = float(len(scale_pairs)) + float(eps)

    rows: list[dict[str, float | int | str]] = []
    total = 0.0
    cumulative = 0.0
    for scale_index, r in enumerate(scale_pairs):
        u_r = h_pos.size // r
        u_2r = h_pos.size // (2 * r)
        g_r = h_pos[: u_r * r].reshape(u_r, r).mean(axis=1)
        g_2r = h_pos[: u_2r * (2 * r)].reshape(u_2r, 2 * r).mean(axis=1)
        u_cmp = min(u_r, 2 * u_2r)
        g_r_cmp = g_r[:u_cmp]
        up = np.repeat(g_2r, 2)[:u_cmp]
        numerator = float(np.mean((g_r_cmp - up) ** 2))
        denominator = float(np.mean(g_r_cmp * g_r_cmp) + msc_floor * msc_floor + eps)
        raw = numerator / denominator
        weighted = raw / weight_denom
        total += weighted
        cumulative += weighted
        rows.append(
            {
                "scale_index": scale_index,
                "scale_r": int(r),
                "scale_2r": int(2 * r),
                "scale_label": f"{r}->{2 * r}",
                "u_r": int(u_r),
                "u_2r": int(u_2r),
                "mspd_r_raw": raw,
                "mspd_r_weighted": weighted,
                "numerator_mse": numerator,
                "denominator_floor_normalized": denominator,
                "cumulative_mspd": cumulative,
            }
        )
    return float(total), rows


def image_to_mspd_signal(image: np.ndarray) -> np.ndarray:
    """Demo extraction: a 1D observation profile from a static image."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image for profile extraction, got {arr.shape}")
    return arr.mean(axis=1)


def save_mspd_profile_plot(profile_rows: list[dict[str, object]], *, output_path: Path) -> None:
    plt = _import_matplotlib(output_path.parent)

    def _fmt_total(value: float) -> str:
        if abs(value) < 1.0e-3 and value != 0.0:
            return f"{value:.2e}"
        return f"{value:.4f}"

    patterns = [
        ("gaussian_noise", "Gaussian noise", "#4C78A8", "o"),
        ("spiral_fractal", "Square spiral fractal", "#F58518", "s"),
        ("uniform", "Uniform field", "#222222", "D"),
    ]
    scale_indices = sorted({int(row["scale_index"]) for row in profile_rows})
    scale_labels = {
        int(row["scale_index"]): str(row["scale_label"])
        for row in profile_rows
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    fig.patch.set_facecolor("white")
    for pattern, label, color, marker in patterns:
        rows = [row for row in profile_rows if row["pattern"] == pattern]
        values_by_scale = {
            int(row["scale_index"]): float(row["mspd_r_weighted"])
            for row in rows
        }
        total = sum(values_by_scale.values())
        y = [values_by_scale.get(scale_index, 0.0) for scale_index in scale_indices]
        ax.plot(
            scale_indices,
            y,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            label=f"{label} (total={_fmt_total(total)})",
        )

    ax.set_title("MSPD dyadic scale profile", fontsize=13)
    ax.set_xlabel("dyadic scale pair r -> 2r", fontsize=11)
    ax.set_ylabel(r"weighted MSPD contribution (symlog)", fontsize=11)
    ax.set_yscale("symlog", linthresh=1.0e-7, linscale=0.8)
    ax.set_xticks(scale_indices)
    ax.set_xticklabels([scale_labels[idx] for idx in scale_indices], rotation=35, ha="right")
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_results(
    output_dir: Path,
    rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
    details: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mssc_reference_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "complexity",
                "block_size",
                "n_terms",
                "image_size",
                "output_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "mspd_reference_profile.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "scale_index",
                "scale_r",
                "scale_2r",
                "scale_label",
                "profile_source",
                "u_r",
                "u_2r",
                "mspd_r_raw",
                "mspd_r_weighted",
                "numerator_mse",
                "denominator_floor_normalized",
                "cumulative_mspd",
            ],
        )
        writer.writeheader()
        writer.writerows(profile_rows)
    (output_dir / "mssc_reference_details.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n"
    )


def run_demo(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    patterns = {
        "gaussian_noise": {
            "label": "Gaussian noise",
            "image": gaussian_noise(args.size, seed=args.seed, sigma=args.noise_sigma),
            "cmap": "gray",
            "filename": "mssc_gaussian_noise.png",
        },
        "spiral_fractal": {
            "label": "Square spiral fractal",
            "image": square_spiral_fractal(
                args.size,
                base_size=args.spiral_base_size,
                levels=args.spiral_levels,
            ),
            "cmap": "magma",
            "filename": "mssc_square_spiral_fractal.png",
        },
        "uniform": {
            "label": "Uniform field",
            "image": uniform_image(args.size, value=args.uniform_value),
            "cmap": "gray",
            "filename": "mssc_uniform.png",
        },
    }

    rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "size": args.size,
        "seed": args.seed,
        "noise_sigma": args.noise_sigma,
        "uniform_value": args.uniform_value,
        "spiral_base_size": args.spiral_base_size,
        "spiral_levels": args.spiral_levels,
        "block_size": args.block_size,
        "mspd_profile_source": "row_mean_intensity",
        "mspd_msc_floor": args.mspd_msc_floor,
        "mspd_scale_normalization": "sum_equal_weights",
        "mspd_term": "floor_reconstruction_error",
        "patterns": {},
    }

    for name, item in patterns.items():
        image = np.asarray(item["image"], dtype=np.float64)
        result = mssc_complexity(image, block_size=args.block_size)
        png_path = output_dir / str(item["filename"])
        save_single_panel(
            image,
            label=str(item["label"]),
            complexity=result.complexity,
            output_path=png_path,
            cmap=str(item["cmap"]),
        )
        rows.append(
            {
                "pattern": name,
                "complexity": f"{result.complexity:.12g}",
                "block_size": result.block_size,
                "n_terms": result.n_terms,
                "image_size": args.size,
                "output_png": str(png_path.relative_to(REPO_ROOT)),
            }
        )
        details["patterns"][name] = {
            "label": item["label"],
            "output_png": str(png_path.relative_to(REPO_ROOT)),
            "result": {
                "complexity": result.complexity,
                "block_size": result.block_size,
                "n_terms": result.n_terms,
                "input_shape": result.input_shape,
                "terms": [asdict(term) for term in result.terms],
            },
        }
        signal = image_to_mspd_signal(image)
        mspd_total, mspd_rows = mspd_scale_profile(signal, msc_floor=args.mspd_msc_floor)
        details["patterns"][name]["mspd_profile"] = {
            "source": "row_mean_intensity",
            "signal_length": int(signal.size),
            "mspd_total": mspd_total,
            "mspd_msc_floor": args.mspd_msc_floor,
        }
        for term in mspd_rows:
            profile_rows.append(
                {
                    "pattern": name,
                    "scale_index": int(term["scale_index"]),
                    "scale_r": int(term["scale_r"]),
                    "scale_2r": int(term["scale_2r"]),
                    "scale_label": str(term["scale_label"]),
                    "profile_source": "row_mean_intensity",
                    "u_r": int(term["u_r"]),
                    "u_2r": int(term["u_2r"]),
                    "mspd_r_raw": f"{float(term['mspd_r_raw']):.12g}",
                    "mspd_r_weighted": f"{float(term['mspd_r_weighted']):.12g}",
                    "numerator_mse": f"{float(term['numerator_mse']):.12g}",
                    "denominator_floor_normalized": f"{float(term['denominator_floor_normalized']):.12g}",
                    "cumulative_mspd": f"{float(term['cumulative_mspd']):.12g}",
                }
            )

    profile_png = output_dir / "mspd_dyadic_profile.png"
    save_mspd_profile_plot(profile_rows, output_path=profile_png)
    save_results(output_dir, rows, profile_rows, details)
    return {
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "scores_csv": str((output_dir / "mssc_reference_scores.csv").relative_to(REPO_ROOT)),
        "profile_csv": str((output_dir / "mspd_reference_profile.csv").relative_to(REPO_ROOT)),
        "profile_png": str(profile_png.relative_to(REPO_ROOT)),
        "details_json": str((output_dir / "mssc_reference_details.json").relative_to(REPO_ROOT)),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reference MSSC implementation demo.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--noise-sigma", type=float, default=0.12)
    parser.add_argument("--uniform-value", type=float, default=0.0)
    parser.add_argument("--spiral-base-size", type=int, default=8)
    parser.add_argument("--spiral-levels", type=int, default=3)
    parser.add_argument("--mspd-msc-floor", type=float, default=0.01)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_demo(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
