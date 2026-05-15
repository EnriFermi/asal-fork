#!/usr/bin/env python3
"""
TrackMate (Fiji) baseline helper.
- Exports frames and label masks to out_dir/trackmate_export.
- Writes a TrackMate macro skeleton (trackmate_macro.ijm).
- If a TrackMate CSV exists in that folder, converts it to overlays/JSON via import_trackmate_csv.py.
"""
import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.tracking.flowlenia_classic_tracker import Config, MaskGenerator, VideoLoader, _parse_resize


def export_for_trackmate(video_path: str, out_dir: str, cfg: Config, stride: int, max_frames: Optional[int], resize):
    export_dir = os.path.join(out_dir, "trackmate_export")
    frames_dir = os.path.join(export_dir, "frames")
    labels_dir = os.path.join(export_dir, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    frames = VideoLoader.load_mp4_frames(video_path, stride=stride, max_frames=max_frames, resize=resize)
    if not frames:
        raise RuntimeError("No frames loaded.")
    frame_indices = list(range(0, len(frames) * stride, stride))
    mask_gen = MaskGenerator(cfg)

    part_segments: Dict[int, Dict[int, np.ndarray]] = {}
    v_per_frame: Dict[int, np.ndarray] = {}
    next_id = 1

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
        label_img = np.zeros(rgb.shape[:2], dtype=np.uint16)
        for lab, det in enumerate(detections, start=1):
            label_img[det.mask_u8.astype(bool)] = lab
            pid = next_id
            next_id += 1
            part_segments.setdefault(frame_key, {})
            part_segments[frame_key][pid] = det.mask_u8.astype(np.uint8)
        # write RGB frame
        Image.fromarray(np.array(pil_frame)).save(os.path.join(frames_dir, f"frame_{frame_key:05d}.png"))
        # write label stack scaled for visibility and a binary mask helper
        if label_img.max() > 0:
            scale = 65535.0 / float(label_img.max())
            label_vis = (label_img.astype(np.float32) * scale).astype(np.uint16)
        else:
            label_vis = label_img
        Image.fromarray(label_vis).save(os.path.join(labels_dir, f"label_{frame_key:05d}.png"))
        mask_vis = (label_img > 0).astype(np.uint8) * 255
        Image.fromarray(mask_vis).save(os.path.join(labels_dir, f"mask_{frame_key:05d}.png"))

    macro_path = os.path.join(export_dir, "trackmate_macro.ijm")
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write(
            f"""// Auto-generated skeleton. Open in Fiji and run (requires TrackMate).
inputDir = "{frames_dir.replace(os.sep, '/')}";
labelDir = "{labels_dir.replace(os.sep, '/')}";
outCsv = "{os.path.join(export_dir, 'tracks.csv').replace(os.sep, '/')}";
// Load RGB frames as stack
run("Image Sequence...", "open=" + inputDir + " sort");
// Optional: load labels or masks to a second stack if you want to visualize segmentation
// run("Image Sequence...", "open=" + labelDir + " sort");
// Launch TrackMate with LAP tracker, gap closing and merge/split enabled.
// Please adjust detector radius/threshold if needed.
run("TrackMate", "open=[] detector=[Downsampled LoG detector] radius=3 threshold=0.0 do_subpixel=true "
    + "tracker=[LAP tracker] linking_max_distance={cfg.max_dist} gap_closing_max_distance={cfg.max_dist} "
    + "gap_closing_max_frame_gap={cfg.max_missed} allow_track_merging=true allow_track_splitting=true");
// When the TrackMate GUI finishes, export tracks to CSV:
// run('Export tracks to CSV', 'save=' + outCsv);
"""
        )
    print(f"[trackmate] Exported frames/labels to {export_dir}")

    # immediate fallback overlay using per-frame IDs (before CSV exists)
    part_tracks = compute_tracks_from_segments(part_segments, v_per_frame)
    fps, _, _ = VideoLoader.get_video_info(video_path)
    out_fps = fps / float(stride)
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
        draw_ids_largest_k=0,
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
        draw_ids_largest_k=0,
    )
    # duplicate fallbacks into export_dir for convenience
    try:
        shutil.copyfile(overlay_parts, os.path.join(export_dir, "overlay_parts_fallback.mp4"))
        shutil.copyfile(overlay_org, os.path.join(export_dir, "overlay_organisms_fallback.mp4"))
    except Exception:
        pass
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
            "backend": "trackmate_fallback",
            "det_v_thr_hi": cfg.det_v_thr_hi,
            "note": "Fallback per-frame IDs; run Fiji macro + import_trackmate_csv.py for real tracking.",
        },
    }
    with open(os.path.join(out_dir, "tracks_organisms.json"), "w", encoding="utf-8") as f:
        json.dump(tracks_org_json, f, indent=2)

    return export_dir


def maybe_import(export_dir: str, video_path: str, stride: int, resize) -> None:
    csv_path = os.path.join(export_dir, "tracks.csv")
    if not os.path.exists(csv_path):
        print("[trackmate] tracks.csv not found yet; run the macro in Fiji then rerun import_trackmate_csv.py.")
        return
    script = os.path.join(os.path.dirname(__file__), "import_trackmate_csv.py")
    cmd = ["python3", script, "--csv", csv_path, "--export_dir", export_dir, "--video", video_path, "--stride", str(stride)]
    if resize:
        cmd.extend(["--resize", f"{resize[0]}x{resize[1]}"])
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare TrackMate export bundle.")
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
    parser.add_argument("--run_import", action="store_true", help="If tracks.csv already present, convert it.")
    args = parser.parse_args()

    cfg = Config(
        det_v_thr_hi=args.det_v_thr_hi,
        h_bins=args.h_bins,
        min_area=args.min_area,
        min_mass=args.min_mass,
        use_hue_bins=args.use_hue_bins,
        marker_split=args.marker_split,
        seed_radius=args.seed_radius,
    )
    resize = _parse_resize(args.resize)
    export_dir = export_for_trackmate(
        video_path=args.video,
        out_dir=args.out_dir,
        cfg=cfg,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        resize=resize,
    )
    if args.run_import:
        maybe_import(export_dir, args.video, max(1, args.stride), resize)


if __name__ == "__main__":
    main()
