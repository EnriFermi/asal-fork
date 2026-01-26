#!/usr/bin/env bash
# Minimal wrapper around scripts/bench.py
# Usage: ./run_bench.sh <video.mp4> [out_dir] [methods]
# methods default: self,trackpy,ultrack,btrack,trackmate

python3 scripts/bench.py \
  --video "try.mp4" \
  --out_dir "segmentation_bench" \
  --methods "self,trackpy,ultrack,btrack,trackmate"
