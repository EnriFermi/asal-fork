#!/usr/bin/env bash
set -euo pipefail

# Plot FlowLenia kernel evolution across a sequence of checkpoints.
# Usage:
#   ./run_plot_kernel_evolution.sh /path/to/ckpt1/best.pkl /path/to/ckpt2/best.pkl ...
# Additional args can be appended, e.g. --output custom.png

python3 scripts/plot_kernel_evolution.py "$@"
