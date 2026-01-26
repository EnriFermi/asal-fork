#!/usr/bin/env python3
"""
Convert TrackMate CSV to Flow-Lenia JSON + overlays.
"""
import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from run_flowlenia_classic_tracker import Visualizer, compute_tracks_from_segments, _parse_resize, VideoLoader


def _load_labels(export_dir: str) -> Dict[int, np.ndarray]:
    labels_dir = os.path.join(export_dir, "labels")
    label_files = [f for f in os.listdir(labels_dir) if f.startswith("label_") and f.endswith(".png")]
    label_map: Dict[int, np.ndarray] = {}
    for fname in sorted(label_files):
        frame = int(fname.split("_")[1].split(".")[0])
        lab = np.array(Image.open(os.path.join(labels_dir, fname)))
        label_map[frame] = lab
    return label_map


def import_trackmate_csv(
    csv_path: str,
    export_dir: str,
    video_path: str,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    draw_ids_largest_k: int = 0,
):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("TrackMate CSV is empty.")

    labels_by_frame = _load_labels(export_dir) if os.path.exists(os.path.join(export_dir, "labels")) else {}

    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    fps, _, _ = VideoLoader.get_video_info(video_path)
    out_fps = fps / float(stride)

    v_per_frame: Dict[int, np.ndarray] = {}
    for idx, fr in enumerate(frames):
        frame_key = frame_indices[idx]
        v_per_frame[frame_key] = (
            np.array(fr.convert("HSV"))[:, :, 2] if fr.mode != "HSV" else np.array(fr)[:, :, 2]
        )

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for _, row in df.iterrows():
        frame = int(row.get("FRAME", row.get("frame", 0)))
        frame_key = frame
        if frame_key not in frame_indices:
            continue
        track_id = int(row.get("TRACK_ID", row.get("track_id", 0)))
        x = float(row.get("POSITION_X", row.get("X", 0.0)))
        y = float(row.get("POSITION_Y", row.get("Y", 0.0)))
        label_img = labels_by_frame.get(frame_key)
        if label_img is not None:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < label_img.shape[0] and 0 <= xi < label_img.shape[1]:
                lbl = int(label_img[yi, xi])
            else:
                lbl = 0
            if lbl > 0:
                mask = (label_img == lbl).astype(np.uint8)
            else:
                mask = np.zeros_like(label_img, dtype=np.uint8)
                rr = 3
                x0 = max(0, xi - rr)
                x1 = min(label_img.shape[1], xi + rr + 1)
                y0 = max(0, yi - rr)
                y1 = min(label_img.shape[0], yi + rr + 1)
                mask[y0:y1, x0:x1] = 1
        else:
            # no labels, draw small disk
            h, w = frames[0].size[1], frames[0].size[0]
            xi = int(round(x))
            yi = int(round(y))
            mask = np.zeros((h, w), dtype=np.uint8)
            rr = 3
            x0 = max(0, xi - rr)
            x1 = min(w, xi + rr + 1)
            y0 = max(0, yi - rr)
            y1 = min(h, yi + rr + 1)
            mask[y0:y1, x0:x1] = 1

        part_segments.setdefault(frame_key, {})
        part_segments[frame_key][track_id] = mask

    part_tracks = compute_tracks_from_segments(part_segments, v_per_frame)

    overlay_parts = os.path.join(export_dir, "overlay_parts.mp4")
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
    with open(os.path.join(export_dir, "tracks_parts.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_parts_json, f, indent=2)
    print(f"[trackmate] Imported CSV to {export_dir}")


def main():
    parser = argparse.ArgumentParser(description="Import TrackMate CSV to Flow-Lenia format.")
    parser.add_argument("--csv", required=True, help="Path to TrackMate CSV (with TRACK_ID, FRAME, POSITION_X/Y).")
    parser.add_argument("--export_dir", required=True, help="TrackMate export directory containing labels/")
    parser.add_argument("--video", required=True, help="Original video for overlays.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--draw_ids_largest_k", type=int, default=0)
    args = parser.parse_args()

    resize = _parse_resize(args.resize)
    import_trackmate_csv(
        csv_path=args.csv,
        export_dir=args.export_dir,
        video_path=args.video,
        stride=max(1, args.stride),
        resize=resize,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )


if __name__ == "__main__":
    main()
