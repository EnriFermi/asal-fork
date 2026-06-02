python scripts/main_opt.py --seed=0 --save_dir="./data/supervised_2_food_notauto_clipped" --substrate="lenia_flow" \
--time_sampling=16 --coef_prompt=0. --coef_softmax=0. \
--coef_oe=1. --bs=1 --rollout_steps=8192 --pop_size=8 \
--n_iters=100 --sigma=0.2 --mutations --mutation_p=0.15 \
--mutation_sz=40 --seed_n_patches=8  --seed_mode="random_patches" \
--food --food_interval=600 --food_n=30 --food_sz=5 \
--food_consume_rate=0.01 --food_bonus=16.0 --mass_decay=0.003 \
--food_conv_mode='scalar'  --food_amount=1.0 --food_diffusion_alpha=0.05 \
--mass_clip_eps 0.01
