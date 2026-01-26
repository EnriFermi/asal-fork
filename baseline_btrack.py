#!/usr/bin/env python3
"""
btrack baseline wrapper.
- Builds label stack from Flow-Lenia detector.
- Runs BayesianTracker if btrack is installed and a config is available.
"""
import argparse
import json
import os
from pathlib import Path
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


def _label_stack(video_path, stride, max_frames, resize, cfg: Config):
    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, max_frames=max_frames, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    mask_gen = MaskGenerator(cfg)
    labels = []
    frame_label_maps: Dict[int, Dict[int, np.ndarray]] = {}
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
        label_img = np.zeros(rgb.shape[:2], dtype=np.int32)
        label_map: Dict[int, np.ndarray] = {}
        for lab, det in enumerate(detections, start=1):
            label_img[det.mask_u8.astype(bool)] = lab
            label_map[lab] = det.mask_u8.astype(np.uint8)
        labels.append(label_img)
        frame_label_maps[frame_key] = label_map
    return frames, frame_indices, np.stack(labels, axis=0), frame_label_maps, v_per_frame


def _find_default_config():
    try:
        import btrack  # type: ignore
    except Exception:
        return None
    pkg_dir = Path(btrack.__file__).resolve().parent
    for candidate in pkg_dir.glob("**/cell_config.json"):
        return str(candidate)
    for candidate in pkg_dir.glob("**/bayesian.json"):
        return str(candidate)
    return None


def run_btrack(
    video_path: str,
    out_dir: str,
    cfg: Config,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    draw_ids_largest_k: int = 0,
    config_path: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import btrack  # type: ignore
        from btrack import utils as bt_utils  # type: ignore
    except Exception:
        print("[btrack] btrack not installed; exported labels for manual run.")
        return {"label_stack": None}

    frames, frame_indices, labels, frame_label_maps, v_per_frame = _label_stack(
        video_path, stride, max_frames, resize, cfg
    )
    label_path = os.path.join(out_dir, "labels.npy")
    np.save(label_path, labels.astype(np.int32))

    cfg_path = config_path or _find_default_config()
    if not cfg_path or not Path(cfg_path).exists():
        print(f"[btrack] config not found (passed={config_path}), saved labels at {label_path}.")
        return {"label_stack": label_path}

    objects = bt_utils.segmentation_to_objects(labels)

    with btrack.BayesianTracker() as tracker:  # type: ignore
        tracker.configure(cfg_path)
        tracker.append(objects)
        h, w = labels.shape[1:]
        tracker.volume = ((0, w), (0, h))
        tracker.track()
        tracker.optimize()
        tracks = tracker.tracks

    if hasattr(bt_utils, "tracks_to_dataframe"):
        tracks_df = bt_utils.tracks_to_dataframe(tracks)
    else:
        # fallback: build minimal dataframe
        import pandas as pd

        rows = []
        for tr in tracks:
            for obs in tr:
                rows.append({"ID": tr.ID, "t": obs.t, "x": obs.x, "y": obs.y})
        tracks_df = pd.DataFrame(rows)

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for _, row in tracks_df.iterrows():
        t_idx = int(row.get("t") if "t" in row else row.get("frame", 0))
        if t_idx < 0 or t_idx >= len(frame_indices):
            continue
        frame_key = frame_indices[t_idx]
        x = int(round(row.get("x", row.get("X", 0))))
        y = int(round(row.get("y", row.get("Y", 0))))
        h, w = labels.shape[1:]
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        lbl = int(labels[t_idx, y, x])
        if lbl <= 0:
            continue
        mask = frame_label_maps[frame_key].get(lbl)
        if mask is None:
            continue
        tid = int(row.get("ID", row.get("track_id", 0)))
        part_segments.setdefault(frame_key, {})
        part_segments[frame_key][tid] = mask.astype(np.uint8)

    part_tracks = compute_tracks_from_segments(part_segments, v_per_frame)

    fps, _, _ = VideoLoader.get_video_info(video_path)
    out_fps = fps / float(max(1, stride))
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
        "debug": {"backend": "btrack", "config_path": cfg_path, "labels_file": label_path},
    }
    with open(os.path.join(out_dir, "tracks_organisms.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_org_json, f, indent=2)

    print(f"[btrack] saved to {out_dir}")
    return {
        "part_segments": part_segments,
        "organism_segments": part_segments,
        "part_tracks": part_tracks,
        "organism_tracks": part_tracks,
        "overlay_org": overlay_org,
        "overlay_parts": overlay_parts,
    }


def main():
    parser = argparse.ArgumentParser(description="btrack baseline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
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
    parser.add_argument("--draw_ids_largest_k", type=int, default=Config.draw_ids_largest_k)
    parser.add_argument("--btrack_config", type=str, default=None, help="Path to btrack config JSON")
    args = parser.parse_args()

    cfg = Config(
        det_v_thr_hi=args.det_v_thr_hi,
        h_bins=args.h_bins,
        min_area=args.min_area,
        min_mass=args.min_mass,
        use_hue_bins=args.use_hue_bins,
        marker_split=args.marker_split,
        seed_radius=args.seed_radius,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )
    resize = _parse_resize(args.resize)
    run_btrack(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
        draw_ids_largest_k=args.draw_ids_largest_k,
        config_path=args.btrack_config,
    )


if __name__ == "__main__":
    main()
