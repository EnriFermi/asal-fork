
python3 scripts/main_opt_parallel.py --seed=0 --save_dir="./data/lockheed_1" --substrate="lenia_flow" \
  --time_sampling=16 --coef_prompt=0. --coef_softmax=0. \
  --coef_oe=1. --bs=1 --rollout_steps=8192 --pop_size=12 \
  --n_iters=100 --sigma=0.2 --mutations --mutation_p=0.2 \
  --mutation_sz=40 --seed_n_patches=16 --seed_mode="random_patches" \
  
