#!/usr/bin/env bash
# Minimal wrapper around scripts/bench.py
# Usage: ./scripts/legacy_entrypoints/run_tracking_bench.sh <video.mp4> [out_dir] [methods] [max_dist] [resize]
# Defaults: video=artifacts/videos/try.mp4, out_dir=artifacts/tracking_bench, methods=self,trackpy,ultrack,btrack,trackmate, max_dist=60, resize=224x224

python3 scripts/bench.py \
  --video "${1:-artifacts/videos/try.mp4}" \
  --out_dir "${2:-artifacts/tracking_bench}" \
  --methods "${3:-self,trackpy,ultrack,btrack,trackmate}" \
  --max_dist "${4:-60}" \
  --resize "${5:-224x224}"
