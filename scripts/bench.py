#!/usr/bin/env python3
"""
Run multiple tracking baselines on one video and collect overlays/JSON.
"""
import argparse
import json
import os
from typing import List, Optional, Tuple

from baseline_btrack import run_btrack
from baseline_trackmate import export_for_trackmate
from baseline_trackpy import run_trackpy
from baseline_ultrack import run_ultrack
from run_flowlenia_classic_tracker import Config, run_pipeline, _parse_resize


def build_cfg(args) -> Config:
    return Config(
        det_v_thr_hi=args.det_v_thr_hi,
        h_bins=args.h_bins,
        min_area=args.min_area,
        min_mass=args.min_mass,
        use_hue_bins=args.use_hue_bins,
        marker_split=args.marker_split,
        seed_radius=args.seed_radius,
        max_dist=args.max_dist,
        max_missed=args.max_missed,
        w_dist=args.w_dist,
        w_iou=args.w_iou,
        w_area=args.w_area,
        w_col=args.w_col,
        strict_color=args.strict_color,
        iou_dilate_r=args.iou_dilate_r,
        bbox_pad=args.bbox_pad,
        merge_iou_min=args.merge_iou_min,
        merge_area_ratio=args.merge_area_ratio,
        split_reacquire_dist=args.split_reacquire_dist,
        group_window=args.group_window,
        tau_sigma=args.tau_sigma,
        tau_dist=args.tau_dist,
        eta_eat=args.eta_eat,
        close_r=args.close_r,
        eat_confirm_frames=args.eat_confirm_frames,
        draw_ids_largest_k=args.draw_ids_largest_k,
        enable_stitching=args.enable_stitching,
        stitch_max_gap=args.stitch_max_gap,
        stitch_max_dist=args.stitch_max_dist,
        stitch_hue_max=args.stitch_hue_max,
        teleport_thr_mult=args.teleport_thr_mult,
        fragmentation_len_thr=args.fragmentation_len_thr,
        min_track_len_for_stitch=args.min_track_len_for_stitch,
    )


def summarize(method_dir: str):
    stats_path = os.path.join(method_dir, "tracks_organisms.json")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    debug = data.get("debug", {})
    stats = debug.get("stats", {})
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flow-Lenia tracking baselines.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--methods", type=str, default="self,trackpy,ultrack,btrack,trackmate")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resize", type=str, default=None)
    # core config toggles
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
    parser.add_argument("--w_dist", type=float, default=Config.w_dist)
    parser.add_argument("--w_iou", type=float, default=Config.w_iou)
    parser.add_argument("--w_area", type=float, default=Config.w_area)
    parser.add_argument("--w_col", type=float, default=Config.w_col)
    parser.add_argument("--strict_color", action="store_true", default=Config.strict_color)
    parser.add_argument("--iou_dilate_r", type=int, default=Config.iou_dilate_r)
    parser.add_argument("--bbox_pad", type=int, default=Config.bbox_pad)
    parser.add_argument("--merge_iou_min", type=float, default=Config.merge_iou_min)
    parser.add_argument("--merge_area_ratio", type=float, default=Config.merge_area_ratio)
    parser.add_argument("--split_reacquire_dist", type=float, default=Config.split_reacquire_dist)
    parser.add_argument("--group_window", type=int, default=Config.group_window)
    parser.add_argument("--tau_sigma", type=float, default=Config.tau_sigma)
    parser.add_argument("--tau_dist", type=float, default=Config.tau_dist)
    parser.add_argument("--eta_eat", type=float, default=Config.eta_eat)
    parser.add_argument("--close_r", type=int, default=Config.close_r)
    parser.add_argument("--eat_confirm_frames", type=int, default=Config.eat_confirm_frames)
    parser.add_argument("--draw_ids_largest_k", type=int, default=Config.draw_ids_largest_k)
    parser.add_argument("--enable_stitching", dest="enable_stitching", action="store_true", default=Config.enable_stitching)
    parser.add_argument("--no_enable_stitching", dest="enable_stitching", action="store_false")
    parser.add_argument("--stitch_max_gap", type=int, default=Config.stitch_max_gap)
    parser.add_argument("--stitch_max_dist", type=float, default=Config.stitch_max_dist)
    parser.add_argument("--stitch_hue_max", type=int, default=Config.stitch_hue_max)
    parser.add_argument("--teleport_thr_mult", type=float, default=Config.teleport_thr_mult)
    parser.add_argument("--fragmentation_len_thr", type=int, default=Config.fragmentation_len_thr)
    parser.add_argument("--min_track_len_for_stitch", type=int, default=Config.min_track_len_for_stitch)
    parser.add_argument("--btrack_config", type=str, default=None)
    args = parser.parse_args()

    cfg = build_cfg(args)
    resize = _parse_resize(args.resize)
    os.makedirs(args.out_dir, exist_ok=True)
    methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]

    for method in methods:
        mdir = os.path.join(args.out_dir, method)
        os.makedirs(mdir, exist_ok=True)
        if method == "self":
            run_pipeline(
                video_path=args.video,
                out_dir=mdir,
                cfg=cfg,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                resize=resize,
                draw_ids_largest_k=args.draw_ids_largest_k,
            )
        elif method == "trackpy":
            run_trackpy(
                video_path=args.video,
                out_dir=mdir,
                cfg=cfg,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                resize=resize,
                draw_ids_largest_k=args.draw_ids_largest_k,
            )
        elif method == "ultrack":
            run_ultrack(
                video_path=args.video,
                out_dir=mdir,
                cfg=cfg,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                resize=resize,
                draw_ids_largest_k=args.draw_ids_largest_k,
            )
        elif method == "btrack":
            run_btrack(
                video_path=args.video,
                out_dir=mdir,
                cfg=cfg,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                resize=resize,
                draw_ids_largest_k=args.draw_ids_largest_k,
                config_path=args.btrack_config,
            )
        elif method == "trackmate":
            export_for_trackmate(
                video_path=args.video,
                out_dir=mdir,
                cfg=cfg,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                resize=resize,
            )
        else:
            print(f"[bench] Unknown method '{method}', skipping.")

    print("\n--- summary ---")
    for method in methods:
        stats = summarize(os.path.join(args.out_dir, method))
        if stats:
            print(f"{method}: {stats}")
        else:
            print(f"{method}: (no stats JSON yet)")


if __name__ == "__main__":
    main()
