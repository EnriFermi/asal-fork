python scripts/simulate_after_training.py --seed=0 \
  --save_dir="./data/interp_supervised_0_1/interp_6" --substrate="lenia_flow" \
  --rollout_steps=1000 \
  --mutations --mutation_p=0.2 --mutation_sz=40 \
  --volcano --volcano_p=0.00003 --volcano_sz=400 --volcano_delta=10.0 \
  --n_seeds=40 --seed_mode="random_patches" \
  --img_size=200 --max_steps=1500000 \
  --output "volcano.mp4"
