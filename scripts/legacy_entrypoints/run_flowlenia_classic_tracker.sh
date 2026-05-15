#!/usr/bin/env bash
python3 tools/tracking/flowlenia_classic_tracker.py \
  --video artifacts/videos/output.mp4 --out_dir artifacts/segmentation \
  --resize 224x224 \
  --no_enable_stitching \
  --merge_iou_min 2.0 \
  --stitch_max_gap 0 \
  --min_track_len_for_stitch 999999
