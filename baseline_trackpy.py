#!/usr/bin/env python3
"""
Centroid-based baseline using trackpy.
- Reuses Flow-Lenia HSV detector (MaskGenerator).
- Links detections with trackpy.link_df.
- Reconstructs masks per track for overlays and JSON output.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from run_flowlenia_classic_tracker import (
    Config,
    MaskGenerator,
    VideoLoader,
    Visualizer,
    compute_tracks_from_segments,
    _parse_resize,
)


def run_trackpy(
    video_path: str,
    out_dir: str,
    cfg: Config,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    draw_ids_largest_k: int = 0,
    search_range_override: Optional[float] = None,
):
    try:
        import pandas as pd
        import trackpy as tp
    except Exception:
        print("[trackpy] trackpy or pandas not installed, skipping.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    fps, _, _ = VideoLoader.get_video_info(video_path)
    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, max_frames=max_frames, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    out_fps = fps / float(stride)

    mask_gen = MaskGenerator(cfg)
    records: List[dict] = []
    mask_store: List[np.ndarray] = []
    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    v_per_frame: Dict[int, np.ndarray] = {}

    for idx, pil_frame in enumerate(frames):
        frame_key = frame_indices[idx]
        rgb = np.array(pil_frame)
        v_per_frame[frame_key] = (
            np.array(pil_frame.convert("HSV"))[:, :, 2] if pil_frame.mode != "HSV" else np.array(pil_frame)[:, :, 2]
        )
        detections, _ = mask_gen.generate_detections(
            rgb,
            cfg.det_v_thr_hi,
            seeds=None,
            use_marker_split=cfg.marker_split,
            seed_radius=cfg.seed_radius,
        )
        for det in detections:
            mask_store.append(det.mask_u8.astype(np.uint8))
            records.append(
                {
                    "frame": frame_key,
                    "x": det.cx,
                    "y": det.cy,
                    "mass": det.mass,
                    "area": det.area,
                    "hue_bin": det.hue_bin,
                    "mask_idx": len(mask_store) - 1,
                }
            )

    if not records:
        raise RuntimeError("No detections for trackpy baseline.")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("frame")
    # trackpy can explode if search_range is too large; adaptively shrink on failure
    sr_try = search_range_override if search_range_override is not None else cfg.max_dist
    linked = None
    while sr_try >= 2.0:
        try:
            linked = tp.link_df(
                df,
                search_range=sr_try,
                memory=cfg.max_missed,
                t_column="frame",
                pos_columns=["x", "y"],
            )
            break
        except Exception as exc:  # narrow down to SubnetOversizeException if available
            print(f"[trackpy] link_df failed at search_range={sr_try}: {exc}")
            sr_try *= 0.5
    if linked is None:
        raise RuntimeError("trackpy linking failed even after reducing search_range")

    for _, row in linked.iterrows():
        frame_key = int(row["frame"])
        track_id = int(row["particle"])
        mask = mask_store[int(row["mask_idx"])]
        part_segments.setdefault(frame_key, {})
        if track_id in part_segments[frame_key]:
            part_segments[frame_key][track_id] = np.clip(
                part_segments[frame_key][track_id].astype(np.uint8) + mask.astype(np.uint8), 0, 1
            )
        else:
            part_segments[frame_key][track_id] = mask.astype(np.uint8)

    part_tracks = compute_tracks_from_segments(part_segments, v_per_frame)

    overlay_parts = os.path.join(out_dir, "overlay_parts.mp4")
    Visualizer.write_overlay_video(
        frames,
        part_segments,
        overlay_parts,
        out_fps,
        alpha=0.45,
        draw_contours=True,
        draw_ids=True,
        frame_indices=frame_indices,
        draw_ids_largest_k=draw_ids_largest_k,
    )

    # simple organism alias = tracks themselves
    overlay_org = os.path.join(out_dir, "overlay_organisms.mp4")
    Visualizer.write_overlay_video(
        frames,
        part_segments,
        overlay_org,
        out_fps,
        alpha=0.45,
        draw_contours=True,
        draw_ids=True,
        frame_indices=frame_indices,
        draw_ids_largest_k=draw_ids_largest_k,
    )

    tracks_parts_json = {
        "tracks": {
            str(pid): [
                {
                    "t": int(t),
                    "cx": float(cx),
                    "cy": float(cy),
                    "mass": float(mass),
                    "area": float(area),
                    "rg": float(rg),
                }
                for t, cx, cy, mass, area, rg in seq
            ]
            for pid, seq in part_tracks.items()
        }
    }
    with open(os.path.join(out_dir, "tracks_parts.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_parts_json, f, indent=2)

    tracks_org_json = {
        "tracks": tracks_parts_json["tracks"],
        "debug": {
            "backend": "trackpy",
            "search_range": cfg.max_dist,
            "memory": cfg.max_missed,
            "det_v_thr_hi": cfg.det_v_thr_hi,
        },
    }
    with open(os.path.join(out_dir, "tracks_organisms.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_org_json, f, indent=2)

    print(f"[trackpy] saved to {out_dir}")
    return {
        "part_segments": part_segments,
        "organism_segments": part_segments,
        "part_tracks": part_tracks,
        "organism_tracks": part_tracks,
        "overlay_org": overlay_org,
        "overlay_parts": overlay_parts,
    }


def main():
    parser = argparse.ArgumentParser(description="Centroid baseline with trackpy")
    parser.add_argument("--video", required=True, help="Path to input mp4")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--det_v_thr_hi", type=int, default=Config.det_v_thr_hi)
    parser.add_argument("--h_bins", type=int, default=Config.h_bins)
    parser.add_argument("--min_area", type=int, default=Config.min_area)
    parser.add_argument("--min_mass", type=float, default=Config.min_mass)
    parser.add_argument("--use_hue_bins", dest="use_hue_bins", action="store_true", default=True)
    parser.add_argument("--no_use_hue_bins", dest="use_hue_bins", action="store_false")
    parser.add_argument("--marker_split", dest="marker_split", action="store_true", default=Config.marker_split)
    parser.add_argument("--no_marker_split", dest="marker_split", action="store_false")
    parser.add_argument("--seed_radius", type=int, default=Config.seed_radius)
    parser.add_argument("--max_dist", type=float, default=Config.max_dist)
    parser.add_argument("--max_missed", type=int, default=Config.max_missed)
    parser.add_argument("--w_col", type=float, default=Config.w_col)
    parser.add_argument("--strict_color", action="store_true", default=Config.strict_color)
    parser.add_argument("--draw_ids_largest_k", type=int, default=Config.draw_ids_largest_k)
    parser.add_argument("--trackpy_search_range", type=float, default=None, help="Override search_range for trackpy")
    args = parser.parse_args()

    cfg = Config(
        det_v_thr_hi=args.det_v_thr_hi,
        h_bins=args.h_bins,
        min_area=args.min_area,
        min_mass=args.min_mass,
        use_hue_bins=args.use_hue_bins,
        marker_split=args.marker_split,
        seed_radius=args.seed_radius,
        max_dist=args.max_dist,
        max_missed=args.max_missed,
        w_col=args.w_col,
        strict_color=args.strict_color,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )
    resize = _parse_resize(args.resize)
    run_trackpy(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
        draw_ids_largest_k=args.draw_ids_largest_k,
        search_range_override=args.trackpy_search_range,
    )


if __name__ == "__main__":
    main()
