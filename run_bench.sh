#!/usr/bin/env bash
# Minimal wrapper around scripts/bench.py
# Usage: ./run_bench.sh <video.mp4> [out_dir] [methods] [max_dist]
# Defaults: video=try.mp4, out_dir=results, methods=self,trackpy,ultrack,btrack,trackmate, max_dist=60

python3 scripts/bench.py \
  --video "${1:-try.mp4}" \
  --out_dir "${2:-results}" \
  --methods "${3:-self,trackpy,ultrack,btrack,trackmate}" \
  --max_dist "${4:-60}"
