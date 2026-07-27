#!/usr/bin/env python3
"""Correct RNG-only replay of the selected run_003 C5 wall example."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import imageio.v2 as imageio
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import flowlenia_c5_branch_frustration as c5
import flowlenia_c5_mass_preserving_wall_probe as mass_probe
import flowlenia_c5_mass_probe_reference_video as reference_video
from flowlenia_c5_branch_analysis import (
    _font,
    _rgb_u8,
    _wall_lines,
    _with_field_pyramid,
)
from paper_suite_c2_branching import _render_apf_rgb


PROBE_VERSION = "flowlenia-c5-selected-rng-only-wall-probe-v1"
OUTPUT_ROOT = (
    c5.DEFAULT_OUTPUT_ROOT
    / "selected_examples"
    / "rng_only_wall_probe_run_003_optimized_point_00"
)
SWEEP_PLAN = Path(
    "analysis/results/"
    "paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/"
    "c2_noise_horizon_sweep/full/sweep_plan.csv"
)


def _load_rows() -> tuple[list[dict[str, str]], dict[str, Any]]:
    base_rows, protocol = mass_probe._load_selected_rows()
    with c5._resolve(SWEEP_PLAN).open(newline="") as stream:
        sweep_rows = list(csv.DictReader(stream))
    rng_only = [
        row
        for row in sweep_rows
        if float(row["strength"]) == 0.0
        and int(row["run_idx"]) == mass_probe.RUN_IDX
        and int(row["step"]) == c5._as_int(base_rows[0]["step"])
        and row["condition"] == base_rows[0]["condition"]
        and int(row["pair_id"]) == c5._as_int(base_rows[0]["pair_id"])
    ]
    rng_only.sort(key=lambda row: int(row["branch_id"]))
    if [int(row["branch_id"]) for row in rng_only] != [0, 1, 2]:
        raise RuntimeError("RNG-only sweep rows are incomplete")
    by_branch = {int(row["branch_id"]): row for row in rng_only}
    result = []
    for base in base_rows:
        branch_id = c5._as_int(base["branch_id"])
        sweep = by_branch[branch_id]
        checks = {
            "branch_seed": (
                c5._as_int(base["branch_seed"]),
                int(sweep["branch_seed"]),
            ),
        }
        mismatches = {
            key: {"c5": left, "rng_only": right}
            for key, (left, right) in checks.items()
            if left != right
        }
        delta_h_c5 = c5._as_float(base["delta_h"])
        delta_h_rng_only = float(sweep["delta_h"])
        if not np.isclose(
            delta_h_c5,
            delta_h_rng_only,
            rtol=0.0,
            atol=1.0e-15,
        ):
            mismatches["delta_h"] = {
                "c5": delta_h_c5,
                "rng_only": delta_h_rng_only,
            }
        if mismatches:
            raise RuntimeError(f"RNG-only linkage mismatch: {mismatches}")
        row = dict(base)
        row.update(
            {
                "free_branch_dir": sweep["branch_dir"],
                "free_provenance": "reused_rng_only_sweep",
                "perturb_a_std": "0.0",
                "perturb_p_std": "0.0",
                "perturb_lagrangian_xy_std": "0.0",
            }
        )
        result.append(row)
    return result, protocol


def _save_variant(
    root: Path,
    variant: str,
    item: dict[str, Any],
    captures: list[dict[str, Any]],
    relative_steps: np.ndarray,
    *,
    roundtrip: dict[str, Any],
    batch_context: dict[str, Any],
) -> Path:
    row = item["row"]
    branch_id = c5._as_int(row["branch_id"])
    path = root / variant / f"branch_{branch_id:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    states = {
        key: np.stack(
            [np.asarray(capture["state"][key]) for capture in captures],
            axis=0,
        )
        for key in ("A", "P", "F")
    }
    mass_by_channel = np.sum(
        states["A"].astype(np.float64),
        axis=(1, 2),
    )
    np.savez_compressed(
        path,
        steps=relative_steps + c5._as_int(row["step"]),
        relative_steps=relative_steps,
        A=states["A"].astype(np.float16),
        P=states["P"].astype(np.float16),
        F=states["F"].astype(np.float16),
        mass_by_channel_float32=mass_by_channel,
        mass_total_float32=np.sum(mass_by_channel, axis=-1),
        resume_batch_rng_key=np.stack(
            [
                np.asarray(capture["rng"], dtype=np.uint32)
                for capture in captures
            ],
            axis=0,
        ),
        row_id=np.asarray(c5._as_int(row["row_id"]), dtype=np.int32),
        branch_id=np.asarray(branch_id, dtype=np.int32),
        branch_seed=np.asarray(
            c5._as_int(row["branch_seed"]),
            dtype=np.int64,
        ),
    )
    mass_probe._write_json(
        path.with_suffix(".metadata.json"),
        {
            "probe_version": PROBE_VERSION,
            "variant": variant,
            "external_state_perturbation": {
                "A_std": 0.0,
                "P_std": 0.0,
                "lagrangian_xy_std": 0.0,
            },
            "branch_seed": c5._as_int(row["branch_seed"]),
            "canonical_row_id_for_selection_only": c5._as_int(row["row_id"]),
            "rng_only_free_branch_dir": row["free_branch_dir"],
            "capture_relative_steps": relative_steps,
            "split_merge_roundtrip": roundtrip,
            "outer_batch": batch_context,
            "artifact_sha256": mass_probe._sha256_file(path),
        },
    )
    return path


def _run_variant(
    *,
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    root: Path,
    engine: dict[str, Any],
    variant: str,
    stepper: Callable[[int], Callable[..., Any]],
) -> tuple[dict[str, Any], dict[int, Path]]:
    written: dict[int, Path] = {}

    def writer(
        item: dict[str, Any],
        captures: list[dict[str, Any]],
        relative_steps: np.ndarray,
        *,
        protocol: dict[str, Any],
        roundtrip: dict[str, Any],
        geometry: c5.SimulationGeometry,
        batch_context: dict[str, Any],
    ) -> None:
        del protocol, geometry
        branch_id = c5._as_int(item["row"]["branch_id"])
        written[branch_id] = _save_variant(
            root,
            variant,
            item,
            captures,
            relative_steps,
            roundtrip=roundtrip,
            batch_context=batch_context,
        )

    old_writer = c5._write_wall_branch
    old_capture = c5._capture_steps_from_free
    c5._write_wall_branch = writer
    c5._capture_steps_from_free = (
        lambda _row: mass_probe.CAPTURE_RELATIVE_STEPS.copy()
    )
    engine["block_state_stepper"] = stepper
    try:
        result = c5._run_simulation_batch(
            rows,
            protocol=protocol,
            batch_size=c5.SIMULATION_BATCH_SIZE,
            mode="walls",
            output_root=root,
            shared_engine=engine,
        )
    finally:
        c5._write_wall_branch = old_writer
        c5._capture_steps_from_free = old_capture
    if tuple(sorted(written)) != (0, 1, 2):
        raise RuntimeError(f"{variant} outputs are incomplete: {written}")
    return result, written


def _load_free(
    row: dict[str, str],
) -> dict[str, np.ndarray]:
    chunks = list((c5._resolve(row["free_branch_dir"]) / "apf_logs").glob("*.npz"))
    if len(chunks) != 1:
        raise RuntimeError(f"Expected one RNG-only APF chunk, found {chunks}")
    with np.load(chunks[0], allow_pickle=False) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        relative = steps - c5._as_int(row["step"])
        indices = mass_probe._common_indices(relative)
        return {
            "steps": steps[indices],
            "relative_steps": relative[indices],
            "A": np.asarray(data["A"][indices]),
            "P": np.asarray(data["P"][indices]),
        }


def _embed_arms(
    arms: dict[str, Sequence[dict[str, np.ndarray]]],
    root: Path,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    import foundation_models

    model = foundation_models.create_foundation_model("clip")
    cache: dict[str, np.ndarray] = {}
    embedded: dict[str, list[np.ndarray]] = {}
    calls = 0
    for arm, branches in arms.items():
        embedded[arm] = []
        for branch in branches:
            rgb = _render_apf_rgb({"A": branch["A"], "P": branch["P"]})
            values = []
            for frame in rgb:
                contiguous = np.ascontiguousarray(frame, dtype=np.float32)
                key = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()
                value = cache.get(key)
                if value is None:
                    value = np.asarray(
                        jax.device_get(model.embed_img(jnp.asarray(contiguous))),
                        dtype=np.float32,
                    ).reshape(-1)
                    value = value / max(
                        float(np.linalg.norm(value)),
                        1.0e-12,
                    )
                    cache[key] = value
                    calls += 1
                values.append(value)
            embedded[arm].append(np.stack(values, axis=0))
        print(f"[clip] {arm} complete", flush=True)
    path = root / "clip_embeddings.npz"
    np.savez_compressed(
        path,
        reference=np.stack(embedded["reference"], axis=0),
        free=np.stack(embedded["free"], axis=0),
        absorbing=np.stack(embedded["absorbing"], axis=0),
        projected=np.stack(embedded["projected"], axis=0),
        relative_steps=mass_probe.COMMON_RELATIVE_STEPS,
        inference_mode=np.asarray(
            "authoritative_c2_unjitted_single_frame"
        ),
    )
    return embedded, {
        "model_id": "openai/clip-vit-base-patch32",
        "inference_mode": "authoritative_c2_unjitted_single_frame",
        "unique_model_calls": calls,
        "exact_render_cache_entries": len(cache),
        "artifact": path,
        "artifact_sha256": mass_probe._sha256_file(path),
    }


def _mass(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "mass_total_float32" in arrays:
        return np.asarray(arrays["mass_total_float32"], dtype=np.float64)
    return np.sum(
        np.asarray(arrays["A"], dtype=np.float64),
        axis=(1, 2, 3),
    )


def _video_frame(
    *,
    frame_idx: int,
    relative_step: int,
    rgb: dict[str, Sequence[np.ndarray]],
    mass: dict[str, Sequence[np.ndarray]],
) -> np.ndarray:
    tile_size = 200
    header = 56
    left = 210
    gap = 4
    arms = (
        (
            "reference",
            "Optimization reference",
            "same source in every column",
            (75, 75, 75),
        ),
        (
            "free",
            "RNG-only free branches",
            "identical state; different RNG",
            (0, 110, 80),
        ),
        (
            "absorbing",
            "RNG-only absorbing walls",
            "",
            (165, 45, 115),
        ),
        (
            "projected",
            "RNG-only mass-projected walls",
            "",
            (35, 95, 170),
        ),
    )
    canvas = Image.new(
        "RGB",
        (
            left + 3 * tile_size + 2 * gap,
            header + 4 * tile_size + 3 * gap,
        ),
        color=(245, 247, 249),
    )
    draw = ImageDraw.Draw(canvas)
    phase = (
        "walls active"
        if relative_step <= c5.WALL_STEPS
        else "walls removed"
    )
    draw.text(
        (10, 7),
        (
            f"{mass_probe.CANDIDATE_ID} | point {mass_probe.POINT_ID:02d} | "
            f"+{relative_step:,} steps | {phase}"
        ),
        fill=(20, 25, 30),
        font=_font(17, bold=True),
    )
    for branch_id in range(3):
        x = left + branch_id * (tile_size + gap)
        draw.text(
            (x + 63, 34),
            f"Branch {branch_id}",
            fill=(45, 50, 56),
            font=_font(13),
        )
    for arm_idx, (key, label, detail, color) in enumerate(arms):
        y = header + arm_idx * (tile_size + gap)
        draw.text(
            (8, y + 62),
            label,
            fill=color,
            font=_font(14, bold=True),
        )
        if detail:
            draw.text(
                (8, y + 86),
                detail,
                fill=(75, 80, 86),
                font=_font(10),
            )
        draw.text(
            (8, y + 115),
            (
                "median mass "
                f"{np.median([mass[key][b][frame_idx] for b in range(3)]):,.1f}"
            ),
            fill=(65, 70, 76),
            font=_font(11),
        )
        for branch_id in range(3):
            x = left + branch_id * (tile_size + gap)
            image = Image.fromarray(
                rgb[key][branch_id][frame_idx]
            ).resize(
                (tile_size, tile_size),
                resample=Image.Resampling.NEAREST,
            )
            if (
                key in {"absorbing", "projected"}
                and relative_step <= c5.WALL_STEPS
            ):
                _wall_lines(image)
            canvas.paste(image, (x, y))
    return np.asarray(canvas)


def _make_visuals(
    *,
    root: Path,
    arms: dict[str, Sequence[dict[str, np.ndarray]]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rgb = {
        arm: [
            _rgb_u8({"A": branch["A"], "P": branch["P"]})
            for branch in branches
        ]
        for arm, branches in arms.items()
    }
    mass = {
        arm: [_mass(branch) for branch in branches]
        for arm, branches in arms.items()
    }
    video_path = root / "rng_only_reference_free_walls_mass_projected.mp4"
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio.get_writer(
        video_path,
        fps=24,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        pixelformat="yuv420p",
    )
    selected = {}
    try:
        for frame_idx, relative_step in enumerate(
            mass_probe.COMMON_RELATIVE_STEPS
        ):
            frame = _video_frame(
                frame_idx=frame_idx,
                relative_step=int(relative_step),
                rgb=rgb,
                mass=mass,
            )
            if frame_idx in {0, 3, 4, 7}:
                selected[frame_idx] = frame
            for _ in range(12):
                writer.append_data(frame)
    finally:
        writer.close()

    reader = imageio.get_reader(video_path)
    decoded = 0
    shape = None
    try:
        for frame in reader:
            decoded += 1
            shape = tuple(int(value) for value in frame.shape)
    finally:
        reader.close()
    if decoded != 96:
        raise RuntimeError(f"Video decoded {decoded} frames, expected 96")

    tiles = [
        Image.fromarray(selected[idx]).resize(
            (
                selected[idx].shape[1] // 2,
                selected[idx].shape[0] // 2,
            ),
            resample=Image.Resampling.LANCZOS,
        )
        for idx in (0, 3, 4, 7)
    ]
    contact_path = video_path.with_suffix(".contact_sheet.png")
    contact = Image.new(
        "RGB",
        (tiles[0].width * 2, tiles[0].height * 2),
        color=(255, 255, 255),
    )
    for idx, tile in enumerate(tiles):
        contact.paste(
            tile,
            ((idx % 2) * tile.width, (idx // 2) * tile.height),
        )
    contact.save(contact_path)

    figure_path = root / "rng_only_mass_and_divergence.png"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        constrained_layout=True,
    )
    colors = {
        "reference": "#666666",
        "free": "#009E73",
        "absorbing": "#CC79A7",
        "projected": "#0072B2",
    }
    labels = {
        "reference": "Optimization reference",
        "free": "RNG-only free",
        "absorbing": "Absorbing walls",
        "projected": "Mass-projected walls",
    }
    for arm in ("reference", "free", "absorbing", "projected"):
        values = np.stack(mass[arm], axis=0)
        axes[0].plot(
            mass_probe.COMMON_RELATIVE_STEPS,
            np.median(values, axis=0),
            marker="o",
            linewidth=2,
            color=colors[arm],
            label=labels[arm],
        )
    axes[0].axvline(
        c5.WALL_STEPS,
        color="#555555",
        linestyle="--",
        linewidth=1,
    )
    axes[0].set(
        xlabel="Relative simulation step",
        ylabel="Total A mass",
        title="RNG-only selected replay",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    within = metrics["post_release"]["within_branch"]
    names = ("free", "absorbing", "projected")
    axes[1].bar(
        np.arange(3),
        [within[name]["clip_chamfer"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[1].set_xticks(
        np.arange(3),
        ["RNG-only\nfree", "Absorbing\nwalls", "Mass-projected\nwalls"],
    )
    axes[1].set(
        ylabel="Within-arm CLIP Chamfer",
        title="Post-release branch divergence",
    )
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "video": video_path,
        "video_sha256": mass_probe._sha256_file(video_path),
        "video_frames": decoded,
        "video_shape": shape,
        "contact_sheet": contact_path,
        "contact_sheet_sha256": mass_probe._sha256_file(contact_path),
        "figure": figure_path,
        "figure_sha256": mass_probe._sha256_file(figure_path),
    }


def _analyze(
    *,
    rows: list[dict[str, str]],
    root: Path,
    absorbing_paths: dict[int, Path],
    projected_paths: dict[int, Path],
) -> dict[str, Any]:
    free = [_load_free(row) for row in rows]
    absorbing_full = [
        mass_probe._load_npz(absorbing_paths[idx]) for idx in range(3)
    ]
    projected_full = [
        mass_probe._load_npz(projected_paths[idx]) for idx in range(3)
    ]
    common_indices = mass_probe._common_indices(
        absorbing_full[0]["relative_steps"]
    )
    absorbing = [
        {
            key: (
                value[common_indices]
                if key in {"A", "P", "F", "mass_total_float32"}
                else value
            )
            for key, value in arrays.items()
        }
        for arrays in absorbing_full
    ]
    projected = [
        {
            key: (
                value[common_indices]
                if key in {"A", "P", "F", "mass_total_float32"}
                else value
            )
            for key, value in arrays.items()
        }
        for arrays in projected_full
    ]
    source_dir = c5._resolve(rows[0]["source_traj_dir"])
    absolute_steps = (
        c5._as_int(rows[0]["step"])
        + mass_probe.COMMON_RELATIVE_STEPS
    )
    reference, source_files = reference_video._load_source_reference(
        source_dir,
        absolute_steps,
    )
    reference_branches = [reference, reference, reference]
    arms = {
        "reference": reference_branches,
        "free": free,
        "absorbing": absorbing,
        "projected": projected,
    }
    start_exact = {
        arm: {
            str(branch_id): {
                key: bool(
                    np.array_equal(
                        branches[branch_id][key][0],
                        reference[key][0],
                    )
                )
                for key in ("A", "P")
            }
            for branch_id in range(3)
        }
        for arm, branches in arms.items()
    }
    if not all(
        all(all(fields.values()) for fields in branches.values())
        for branches in start_exact.values()
    ):
        raise RuntimeError(f"RNG-only initial equality failed: {start_exact}")
    rng_exact_between_wall_variants = {
        str(branch_id): bool(
            np.array_equal(
                absorbing_full[branch_id]["resume_batch_rng_key"],
                projected_full[branch_id]["resume_batch_rng_key"],
            )
        )
        for branch_id in range(3)
    }
    if not all(rng_exact_between_wall_variants.values()):
        raise RuntimeError("Wall variant RNG streams differ")

    embedded, embedding_info = _embed_arms(arms, root)
    fields = {
        arm: [
            _with_field_pyramid(
                {"A": branch["A"], "P": branch["P"]}
            )
            for branch in branches
        ]
        for arm, branches in arms.items()
    }
    post = np.flatnonzero(
        mass_probe.COMMON_RELATIVE_STEPS > c5.WALL_STEPS
    )
    within = {
        arm: mass_probe._within_metrics(
            embedded[arm],
            fields[arm],
            post,
        )
        for arm in ("free", "absorbing", "projected")
    }
    paired = {
        "free_vs_absorbing": mass_probe._paired_metrics(
            embedded["free"],
            embedded["absorbing"],
            fields["free"],
            fields["absorbing"],
            post,
        ),
        "free_vs_projected": mass_probe._paired_metrics(
            embedded["free"],
            embedded["projected"],
            fields["free"],
            fields["projected"],
            post,
        ),
    }
    mass_summary = {}
    for arm, branches in arms.items():
        values = np.stack([_mass(branch) for branch in branches], axis=0)
        mass_summary[arm] = {
            "median_by_step": np.median(values, axis=0),
            "branch_fraction_lost_at_20000": (
                1.0 - values[:, -1] / values[:, 0]
            ),
        }
    metrics = {
        "probe_version": PROBE_VERSION,
        "protocol": {
            "external_state_perturbation": {
                "A_std": 0.0,
                "P_std": 0.0,
                "lagrangian_xy_std": 0.0,
            },
            "branch_semantics": (
                "identical source A/P/lagrangian state; only branch_seed is "
                "folded into the continuation RNG"
            ),
            "branch_seeds": [
                c5._as_int(row["branch_seed"]) for row in rows
            ],
        },
        "initial_A_P_exact_vs_reference": start_exact,
        "wall_variant_rng_exact": rng_exact_between_wall_variants,
        "source_files": source_files,
        "embedding": embedding_info,
        "post_release": {
            "relative_steps": mass_probe.COMMON_RELATIVE_STEPS[post],
            "within_branch": within,
            "paired_same_seed": paired,
            "free_minus_absorbing_within_clip": (
                within["free"]["clip_chamfer"]
                - within["absorbing"]["clip_chamfer"]
            ),
            "free_minus_projected_within_clip": (
                within["free"]["clip_chamfer"]
                - within["projected"]["clip_chamfer"]
            ),
        },
        "mass": mass_summary,
    }
    visuals = _make_visuals(root=root, arms=arms, metrics=metrics)
    metrics["visuals"] = visuals
    mass_probe._write_json(root / "metrics.json", metrics)
    return metrics


def main() -> int:
    started = time.monotonic()
    root = c5._resolve(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    rows, protocol = _load_rows()
    engine = c5._create_wall_engine(rows[0])
    runtime = engine["runtime"]
    canonical_stepper = engine["block_state_stepper"]
    clone_stepper = mass_probe._make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=engine["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
        project_mass=False,
    )
    projected_stepper = mass_probe._make_block_state_stepper(
        runtime["block_substrate"],
        n_blocks=runtime["geometry"].n_blocks,
        original_batch_size=engine["original_batch_size"],
        valid_mask=runtime["valid_mask"],
        geometry=runtime["geometry"],
        mutation_spec=runtime["mutation_spec"],
        block_rt_gumbel=runtime["block_rt_gumbel"],
        project_mass=True,
    )
    print("[rng-only] auditing stepper clone and mass projection", flush=True)
    stepper_audit = mass_probe._stepper_audit(
        rows,
        engine,
        clone_stepper,
        projected_stepper,
    )
    mass_probe._write_json(root / "stepper_audit.json", stepper_audit)
    def reusable_variant(
        variant: str,
    ) -> dict[int, Path] | None:
        paths = {
            branch_id: root / variant / f"branch_{branch_id:02d}.npz"
            for branch_id in range(3)
        }
        for branch_id, path in paths.items():
            metadata_path = path.with_suffix(".metadata.json")
            if not path.exists() or not metadata_path.exists():
                return None
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("probe_version") != PROBE_VERSION
                or metadata.get("variant") != variant
                or int(metadata.get("branch_seed", -1))
                != c5._as_int(rows[branch_id]["branch_seed"])
                or metadata.get("artifact_sha256")
                != mass_probe._sha256_file(path)
            ):
                return None
            arrays = mass_probe._load_npz(path)
            if (
                not np.array_equal(
                    arrays["relative_steps"],
                    mass_probe.CAPTURE_RELATIVE_STEPS,
                )
                or not all(
                    np.all(np.isfinite(arrays[key]))
                    for key in ("A", "P", "F")
                )
            ):
                return None
        return paths

    absorbing_paths = reusable_variant("absorbing_walls")
    if absorbing_paths is None:
        print("[rng-only] simulating absorbing walls", flush=True)
        absorbing_result, absorbing_paths = _run_variant(
            rows=rows,
            protocol=protocol,
            root=root,
            engine=engine,
            variant="absorbing_walls",
            stepper=canonical_stepper,
        )
    else:
        print("[rng-only] reusing audited absorbing walls", flush=True)
        absorbing_result = {
            "status": "reused",
            "rows": list(mass_probe.EXPECTED_ROW_IDS),
        }
    projected_paths = reusable_variant("mass_projected_walls")
    if projected_paths is None:
        print("[rng-only] simulating mass-projected walls", flush=True)
        projected_result, projected_paths = _run_variant(
            rows=rows,
            protocol=protocol,
            root=root,
            engine=engine,
            variant="mass_projected_walls",
            stepper=projected_stepper,
        )
    else:
        print("[rng-only] reusing audited mass-projected walls", flush=True)
        projected_result = {
            "status": "reused",
            "rows": list(mass_probe.EXPECTED_ROW_IDS),
        }
    print("[rng-only] computing metrics and video", flush=True)
    metrics = _analyze(
        rows=rows,
        root=root,
        absorbing_paths=absorbing_paths,
        projected_paths=projected_paths,
    )
    summary = {
        "status": "complete",
        "probe_version": PROBE_VERSION,
        "canonical_c5_untouched": True,
        "old_noisy_c5_not_used": True,
        "output_root": root,
        "stepper_audit": stepper_audit,
        "absorbing_simulation": absorbing_result,
        "projected_simulation": projected_result,
        "key_results": metrics["post_release"],
        "mass": metrics["mass"],
        "visuals": metrics["visuals"],
        "elapsed_seconds": time.monotonic() - started,
        "script": Path(__file__).resolve(),
        "script_sha256": mass_probe._sha256_file(Path(__file__).resolve()),
    }
    mass_probe._write_json(root / "summary.json", summary)
    print(
        json.dumps(
            mass_probe._json_value(summary),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
