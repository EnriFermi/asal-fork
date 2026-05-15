python scripts/simulate_after_training.py --seed=618 \
  --save_dir="./data/lockheed_1" --substrate="lenia_flow" \
  --rollout_steps=3000000 \
  --mutations --mutation_p=0.2 --mutation_sz=40 --mutation_scale=3.0 \
  --n_seeds=24 --seed_mode="random_patches" \
  --img_size=300 --max_steps=3000000 \
  --output "volcano__lockheed_1__1_5m.mp4"