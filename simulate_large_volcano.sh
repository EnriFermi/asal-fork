# python scripts/simulate_after_training.py --seed=0 \
#   --save_dir="./data/tmp_random_best" --substrate="lenia_flow" \
#   --rollout_steps=500000 \
#   --mutations --mutation_p=0.05 --mutation_sz=40 \
#   --volcano --volcano_p=0.0005 --volcano_sz=330 --volcano_delta=4.0 \
#   --n_seeds=16 --seed_mode="random_patches" \
#   --img_size=154 --max_steps=500000 \
#   --output "volcano_porcupine_02.mp4"


# python scripts/simulate_after_training.py --seed=0 \
#   --save_dir="./data/tmp_random_best" --substrate="lenia_flow" \
#   --rollout_steps=4000000 \
#   --mutations --mutation_p=0.05 --mutation_sz=40 \
#   --volcano --volcano_p=0.0005 --volcano_sz=340 --volcano_delta=4.0 \
#   --n_seeds=16 --seed_mode="random_patches" \
#   --img_size=154 --max_steps=4000000 \
#   --output "volcano__porcupine__food__1_5m.mp4" \
#   --food --food_auto_size --food_auto_scale=1.25 \
#   --food_interval=10000 \
#   --food_n=3 --food_sz=32 --food_amount=1.0 \
#   --food_consume_rate=0.01 --food_bonus=1.0 \
#   --food_channel=1 --food_conv_mode=scalar \
#   --mass_decay=6.93e-5 --food_diffusion_alpha=0.1 \
# #   --mass_clip_eps=1e-4

# python scripts/simulate_after_training.py --seed=20 \
#   --save_dir="./data/tmp_random_best" --substrate="lenia_flow" \
#   --rollout_steps=4000000 \
#   --mutations --mutation_p=0.2 --mutation_sz=10 \
#   --volcano --volcano_p=0.00005 --volcano_sz=350 --volcano_delta=4.0 \
#   --n_seeds=16 --seed_mode="random_patches" \
#   --img_size=154 --max_steps=4000000 \
#   --output "volcano__porcupine__food__4m.mp4" \
#   --food --food_auto_size --food_auto_scale=1.25 \
#   --food_interval=10000 \
#   --food_n=3 --food_sz=32 --food_amount=1.0 \
#   --food_consume_rate=0.01 --food_bonus=1.0 \
#   --food_channel=1 --food_conv_mode=scalar \
#   --mass_decay=6.93e-5 --food_diffusion_alpha=0.1 \
# #   --mass_clip_eps=1e-4


python scripts/simulate_after_training.py --seed=20 \
  --save_dir="./data/tmp_random_best" --substrate="lenia_flow" \
  --rollout_steps=500000 \
  --mutations --mutation_p=0.05 --mutation_sz=40 --mutation_scale=1.0 \
  --n_seeds=16 --seed_mode="random_patches" \
  --img_size=154 --max_steps=500000 \
  --output "volcano__porcupine__food__0_5m.mp4" \
  # --volcano --volcano_p=0.00001 --volcano_sz=350 --volcano_delta=4.0 \
  # --food --food_auto_size --food_auto_scale=2.0 \
  # --food_interval=10000 \
  # --food_n=6 --food_sz=6 --food_amount=0.2 \
  # --food_consume_rate=0.005 --food_bonus=1.0 \
  # --food_channel=1 --food_conv_mode=scalar \
  # --mass_decay=6.93e-5 --food_diffusion_alpha=0.005 \
  # --mass_clip_eps=1e-3 \
