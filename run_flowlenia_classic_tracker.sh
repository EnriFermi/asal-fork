#!/usr/bin/env bash
python3 run_flowlenia_classic_tracker.py \
  --video output.mp4 --out_dir segmentation \
  --resize 224x224 \
  --no_enable_stitching \
  --merge_iou_min 2.0 \
  --stitch_max_gap 0 \
  --min_track_len_for_stitch 999999