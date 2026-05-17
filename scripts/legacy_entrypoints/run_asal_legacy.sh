python scripts/main_opt.py --seed=0 --save_dir="./data/supervised_pca_track_next_iteration" --substrate="lenia_flow" \
--time_sampling=16 --coef_prompt=0. --coef_softmax=0. \
--coef_oe=1. --bs=1 --rollout_steps=4096 --pop_size=8 \
--n_iters=100 --sigma=0.2 --mutations --mutation_p=0.05 \
--mutation_sz=40 --seed_n_patches=16  --seed_mode="random_patches"
