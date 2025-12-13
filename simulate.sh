python simulate_after_training.py --seed=0 \
--save_dir="./data/interp_supervised_0_1/interp_6" --substrate='lenia_flow' \
--rollout_steps=1000 --mutations --mutation_p=0.2 \
--mutation_sz=40 --n_seeds=80  --seed_mode="random_patches" \
--img_size=256 --max_steps=3000000 --output "interp_out_heavy_mass.mp4"