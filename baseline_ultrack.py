#!/usr/bin/env python3
"""
Ultrack baseline wrapper.
- Builds label stack from Flow-Lenia detector.
- If ultrack is installed, runs Tracker(MainConfig) on the label stack.
- Falls back gracefully when ultrack missing.
"""
import argparse
import json
import os
import shutil
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


def _build_labels_and_v(
    frames,
    frame_indices: List[int],
    cfg: Config,
    stride: int,
    max_frames: Optional[int],
    resize: Optional[Tuple[int, int]],
    mask_gen: MaskGenerator,
) -> Tuple[np.ndarray, Dict[int, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    label_stack: List[np.ndarray] = []
    frame_labels: Dict[int, Dict[int, np.ndarray]] = {}
    v_per_frame: Dict[int, np.ndarray] = {}
    next_label = 1
    for idx, pil_frame in enumerate(frames):
        frame_key = frame_indices[idx]
        rgb = np.array(pil_frame)
        hsv_v = (
            np.array(pil_frame.convert("HSV"))[:, :, 2] if pil_frame.mode != "HSV" else np.array(pil_frame)[:, :, 2]
        )
        v_per_frame[frame_key] = hsv_v
        detections, _ = mask_gen.generate_detections(
            rgb,
            cfg.det_v_thr_hi,
            seeds=None,
            use_marker_split=cfg.marker_split,
            seed_radius=cfg.seed_radius,
        )
        label_img = np.zeros(rgb.shape[:2], dtype=np.int32)
        label_map: Dict[int, np.ndarray] = {}
        for det in detections:
            lab = next_label
            next_label += 1
            label_img[det.mask_u8.astype(bool)] = lab
            label_map[lab] = det.mask_u8.astype(np.uint8)
        label_stack.append(label_img)
        frame_labels[frame_key] = label_map
    stack = np.stack(label_stack, axis=0)

    # global relabel to guarantee uniqueness & compact ids (avoids DB UNIQUE errors)
    uniq = np.unique(stack)
    uniq = uniq[uniq > 0]
    if uniq.size:
        max_id = int(uniq.max())
        lut = np.zeros(max_id + 1, dtype=np.int32)
        lut[uniq] = np.arange(1, uniq.size + 1, dtype=np.int32)
        stack = lut[stack]

    # rebuild frame label maps using relabeled stack
    frame_labels_relabeled: Dict[int, Dict[int, np.ndarray]] = {}
    for idx, frame_key in enumerate(frame_indices):
        lab_img = stack[idx]
        ids = np.unique(lab_img)
        ids = ids[ids > 0]
        fmap: Dict[int, np.ndarray] = {}
        for lid in ids:
            fmap[int(lid)] = (lab_img == lid).astype(np.uint8)
        frame_labels_relabeled[frame_key] = fmap

    return stack.astype(np.int32), frame_labels_relabeled, v_per_frame


def run_ultrack(
    video_path: str,
    out_dir: str,
    cfg: Config,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    draw_ids_largest_k: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from ultrack import MainConfig, Tracker
    except Exception:
        print("[ultrack] ultrack not installed; exported label stack for manual run.")
        need_ultrack = True
    else:
        need_ultrack = False

    fps, _, _ = VideoLoader.get_video_info(video_path)
    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, max_frames=max_frames, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    out_fps = fps / float(stride)

    mask_gen = MaskGenerator(cfg)
    label_stack, frame_label_maps, v_per_frame = _build_labels_and_v(
        frames, frame_indices, cfg, stride, max_frames, resize, mask_gen
    )
    label_path = os.path.join(out_dir, "labels.npy")
    np.save(label_path, label_stack.astype(np.int32))

    if need_ultrack:
        return {"label_stack": label_path}

    # run ultrack
    ucfg = MainConfig()
    # handle versions without data_path/sqlite fields; use environment-based path overrides
    db_dir = os.path.join(out_dir, "ultrack_db")
    os.makedirs(db_dir, exist_ok=True)
    sqlite_path = os.path.join(db_dir, "tracks.sqlite")
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    mm_dir = os.path.join(db_dir, "memmaps")
    if os.path.isdir(mm_dir):
        shutil.rmtree(mm_dir, ignore_errors=True)

    # some versions look at env vars for paths
    os.environ["ULTRACK_DATA_PATH"] = db_dir
    os.environ["ULTRACK_SQLITE_FILENAME"] = sqlite_path
    try:
        ucfg.sqlite_filename = sqlite_path  # type: ignore[attr-defined]
    except Exception:
        pass

    tracker = Tracker(ucfg)
    try:
        tracker.track(labels=label_stack)
    except Exception as e:
        print(f"[ultrack] tracking failed: {e}")
        return {"label_stack": label_path, "error": str(e)}

    tracks_df = None
    if hasattr(tracker, "to_tracks_layer"):
        tracks_df, _ = tracker.to_tracks_layer()
    elif hasattr(tracker, "tracks"):
        try:
            import pandas as pd
        except Exception:
            pd = None
        if pd is not None:
            try:
                tracks_df = pd.DataFrame(tracker.tracks)
            except Exception:
                tracks_df = None
    if tracks_df is None:
        print("[ultrack] Could not extract tracks dataframe from Tracker; skipping.")
        return {"label_stack": label_path}

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for _, row in tracks_df.iterrows():
        t_idx = int(row["t"])
        if t_idx < 0 or t_idx >= len(frame_indices):
            continue
        frame_key = frame_indices[t_idx]
        y = int(round(row["y"]))
        x = int(round(row["x"]))
        h, w = label_stack.shape[1:]
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        lbl = int(label_stack[t_idx, y, x])
        if lbl <= 0:
            continue
        mask = frame_label_maps[frame_key].get(lbl)
        if mask is None:
            continue
        track_id = int(row["track_id"]) if "track_id" in row else int(row["track"])
        part_segments.setdefault(frame_key, {})
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
            "backend": "ultrack",
            "det_v_thr_hi": cfg.det_v_thr_hi,
            "labels_file": label_path,
        },
    }
    with open(os.path.join(out_dir, "tracks_organisms.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_org_json, f, indent=2)

    print(f"[ultrack] saved to {out_dir}")
    return {
        "part_segments": part_segments,
        "organism_segments": part_segments,
        "part_tracks": part_tracks,
        "organism_tracks": part_tracks,
        "overlay_org": overlay_org,
        "overlay_parts": overlay_parts,
    }


def main():
    parser = argparse.ArgumentParser(description="Ultrack baseline")
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
    run_ultrack(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
        draw_ids_largest_k=args.draw_ids_largest_k,
    )


if __name__ == "__main__":
    main()
