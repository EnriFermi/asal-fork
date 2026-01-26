python run_sam2_pipeline.py --video try.mp4 --out_dir segmentation \
  --model facebook/sam2.1-hiera-large \
  --group_window 50 --eta_eat 0.6 --close_r 4 --eat_confirm_frames 3 \
  --tau_sigma 3.0 --tau_dist 40 --save_parts_debug
