#!/usr/bin/env python3
"""Add the unperturbed optimization continuation to the selected C5 probe video."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

import flowlenia_c5_branch_frustration as c5
import flowlenia_c5_mass_preserving_wall_probe as probe
from flowlenia_c5_branch_analysis import _font, _rgb_u8, _wall_lines
from flowlenia_minibang_common import list_apf_chunks


VIDEO_VERSION = "flowlenia-c5-mass-probe-with-optimization-reference-v1"


def _load_source_reference(
    source_traj_dir: Path,
    absolute_steps: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    wanted = {int(step) for step in absolute_steps}
    captured: dict[int, dict[str, np.ndarray]] = {}
    source_files = []
    for path, step_min, step_max, _idx in list_apf_chunks(
        source_traj_dir / "apf_logs"
    ):
        if step_max < min(wanted) or step_min > max(wanted):
            continue
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            indices = np.flatnonzero(np.isin(steps, absolute_steps))
            for idx in indices:
                step = int(steps[idx])
                if step in captured:
                    raise RuntimeError(f"Duplicate source snapshot at {step}")
                captured[step] = {
                    key: np.asarray(data[key][idx])
                    for key in ("A", "P", "F")
                }
        source_files.append(
            {
                "path": path,
                "sha256": probe._sha256_file(path),
                "selected_steps": [
                    int(step)
                    for step in absolute_steps
                    if step_min <= int(step) <= step_max
                ],
            }
        )
    missing = sorted(wanted - set(captured))
    if missing:
        raise RuntimeError(f"Source trajectory lacks snapshots: {missing}")
    return (
        {
            key: np.stack(
                [captured[int(step)][key] for step in absolute_steps],
                axis=0,
            )
            for key in ("A", "P", "F")
        },
        source_files,
    )


def _mass(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(
        np.asarray(arrays["A"], dtype=np.float64),
        axis=(1, 2, 3),
    )


def _frame(
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
            "same unperturbed source in every column",
            (75, 75, 75),
        ),
        (
            "free",
            "Perturbed free branches",
            "",
            (0, 110, 80),
        ),
        (
            "absorbing",
            "Absorbing walls",
            "",
            (165, 45, 115),
        ),
        (
            "projected",
            "Mass-projected walls",
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
            f"{probe.CANDIDATE_ID} | point {probe.POINT_ID:02d} | "
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
            (8, y + 64),
            label,
            fill=color,
            font=_font(15, bold=True),
        )
        if detail:
            draw.text(
                (8, y + 88),
                detail,
                fill=(75, 80, 86),
                font=_font(10),
            )
        median_mass = float(
            np.median(
                [
                    mass[key][branch_id][frame_idx]
                    for branch_id in range(3)
                ]
            )
        )
        draw.text(
            (8, y + 115),
            f"median mass {median_mass:,.1f}",
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


def _initial_perturbation_audit(
    reference: dict[str, np.ndarray],
    free: Sequence[dict[str, np.ndarray]],
) -> dict[str, Any]:
    branches = {}
    for branch_id, arrays in enumerate(free):
        fields = {}
        for key in ("A", "P"):
            delta = (
                np.asarray(arrays[key][0], dtype=np.float32)
                - np.asarray(reference[key][0], dtype=np.float32)
            )
            fields[key] = {
                "exact": bool(np.array_equal(arrays[key][0], reference[key][0])),
                "max_abs": float(np.max(np.abs(delta))),
                "rms": float(np.sqrt(np.mean(delta * delta))),
            }
        branches[str(branch_id)] = fields
    return {
        "semantics": (
            "Reference is the unperturbed source continuation. Each free/wall "
            "branch starts after its configured branch-seed A/P perturbation."
        ),
        "branches": branches,
    }


def main() -> int:
    rows, protocol = probe._load_selected_rows()
    root = c5._resolve(probe.DEFAULT_ROOT)
    corrected = [
        probe._load_npz(root / "branches" / f"branch_{idx:02d}.npz")
        for idx in range(3)
    ]
    corrected_indices = probe._common_indices(
        corrected[0]["relative_steps"]
    )
    free = [
        c5._branch_arrays(
            c5._resolve(row["free_branch_dir"]),
            keys={"A", "P", "F"},
        )
        for row in rows
    ]
    absorbing = [
        c5._branch_arrays(
            c5._resolve(row["walls_branch_dir"]),
            keys={"A", "P", "F"},
        )
        for row in rows
    ]
    start_step = c5._as_int(rows[0]["step"])
    absolute_steps = start_step + probe.COMMON_RELATIVE_STEPS
    source_traj_dir = c5._resolve(rows[0]["source_traj_dir"])
    reference, source_files = _load_source_reference(
        source_traj_dir,
        absolute_steps,
    )

    reference_rgb = _rgb_u8({"A": reference["A"], "P": reference["P"]})
    free_rgb = [
        _rgb_u8({"A": arrays["A"], "P": arrays["P"]})
        for arrays in free
    ]
    absorbing_rgb = [
        _rgb_u8({"A": arrays["A"], "P": arrays["P"]})
        for arrays in absorbing
    ]
    projected_rgb = [
        _rgb_u8(
            {
                "A": arrays["A"][corrected_indices],
                "P": arrays["P"][corrected_indices],
            }
        )
        for arrays in corrected
    ]
    reference_mass = _mass(reference)
    rgb = {
        "reference": [reference_rgb, reference_rgb, reference_rgb],
        "free": free_rgb,
        "absorbing": absorbing_rgb,
        "projected": projected_rgb,
    }
    mass = {
        "reference": [reference_mass, reference_mass, reference_mass],
        "free": [_mass(arrays) for arrays in free],
        "absorbing": [_mass(arrays) for arrays in absorbing],
        "projected": [
            arrays["mass_total_float32"][corrected_indices]
            for arrays in corrected
        ],
    }

    video_path = (
        root
        / "free_absorbing_mass_projected_with_optimization_reference.mp4"
    )
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio.get_writer(
        video_path,
        fps=24,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        pixelformat="yuv420p",
    )
    selected_frames: dict[int, np.ndarray] = {}
    try:
        for frame_idx, relative_step in enumerate(
            probe.COMMON_RELATIVE_STEPS
        ):
            current = _frame(
                frame_idx=frame_idx,
                relative_step=int(relative_step),
                rgb=rgb,
                mass=mass,
            )
            if frame_idx in {0, 3, 4, 7}:
                selected_frames[frame_idx] = current
            for _ in range(12):
                writer.append_data(current)
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
    expected_frames = len(probe.COMMON_RELATIVE_STEPS) * 12
    if decoded != expected_frames:
        raise RuntimeError(
            f"Decoded {decoded} frames, expected {expected_frames}"
        )

    contact_tiles = []
    for frame_idx in (0, 3, 4, 7):
        image = Image.fromarray(selected_frames[frame_idx])
        contact_tiles.append(
            image.resize(
                (image.width // 2, image.height // 2),
                resample=Image.Resampling.LANCZOS,
            )
        )
    contact_path = video_path.with_suffix(".contact_sheet.png")
    contact = Image.new(
        "RGB",
        (
            contact_tiles[0].width * 2,
            contact_tiles[0].height * 2,
        ),
        color=(255, 255, 255),
    )
    for idx, tile in enumerate(contact_tiles):
        contact.paste(
            tile,
            (
                (idx % 2) * tile.width,
                (idx // 2) * tile.height,
            ),
        )
    contact.save(contact_path)

    provenance = {
        "status": "complete",
        "video_version": VIDEO_VERSION,
        "simulation_protocol_version": protocol["protocol_version"],
        "plan_sha256": protocol["plan_sha256"],
        "candidate_id": probe.CANDIDATE_ID,
        "point_id": probe.POINT_ID,
        "row_ids": list(probe.EXPECTED_ROW_IDS),
        "source_traj_id": rows[0]["source_traj_id"],
        "source_traj_dir": source_traj_dir,
        "reference_semantics": (
            "Unperturbed continuation recorded in the exact fixed-seed APF "
            "replay of the optimized trajectory used as the branch source."
        ),
        "reference_repeated_across_columns": True,
        "absolute_steps": absolute_steps,
        "relative_steps": probe.COMMON_RELATIVE_STEPS,
        "source_files": source_files,
        "initial_reference_vs_perturbed_free": (
            _initial_perturbation_audit(reference, free)
        ),
        "video": video_path,
        "video_sha256": probe._sha256_file(video_path),
        "video_bytes": video_path.stat().st_size,
        "fps": 24,
        "real_snapshots": len(probe.COMMON_RELATIVE_STEPS),
        "display_frames_per_real_snapshot": 12,
        "decoded_frames": decoded,
        "decoded_shape": shape,
        "contact_sheet": contact_path,
        "contact_sheet_sha256": probe._sha256_file(contact_path),
        "script": Path(__file__).resolve(),
        "script_sha256": probe._sha256_file(Path(__file__).resolve()),
    }
    provenance_path = video_path.with_suffix(".provenance.json")
    probe._write_json(provenance_path, provenance)
    print(
        json.dumps(
            probe._json_value(provenance),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
