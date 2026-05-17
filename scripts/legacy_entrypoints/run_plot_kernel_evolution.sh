

# Plot FlowLenia kernel evolution across a sequence of checkpoints.
# Usage:
#   ./run_plot_kernel_evolution.sh /path/to/ckpt1/best.pkl /path/to/ckpt2/best.pkl ...
# Additional args can be appended, e.g. --output custom.png

python3 scripts/plot_kernel_evolution.py \
    data/interp_supervised_0_1/interp_1/best.pkl \
    data/interp_supervised_0_1/interp_2/best.pkl \
    data/interp_supervised_0_1/interp_3/best.pkl \
    data/interp_supervised_0_1/interp_4/best.pkl \
    data/interp_supervised_0_1/interp_5/best.pkl \
    data/interp_supervised_0_1/interp_6/best.pkl \
    --output kernel_evo.png \
    --crop_size 100
