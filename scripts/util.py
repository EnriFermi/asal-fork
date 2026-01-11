import os
import json
import pickle
from typing import Any, Dict, Tuple

import numpy as np


def save_json(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.json", "w") as f:
            json.dump(item, f)
            
def load_json(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.json", "r") as f:
            return json.load(f)
    else:
        return None

def save_pkl(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", "wb") as f:
            pickle.dump(item, f)


def load_pkl(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None


def parse_matrix_str(s: str) -> np.ndarray:
    rows = [r.strip() for r in s.split(";") if r.strip()]
    data = []
    for row in rows:
        cols = [c.strip() for c in row.split(",") if c.strip()]
        data.append([int(float(c)) for c in cols])
    return np.array(data, dtype=int)


def parse_color_str(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def flow_lenia_kwargs_from_args(args: Any) -> Dict[str, Any]:
    seed_n = getattr(args, "n_seeds", getattr(args, "seed_n_patches"))
    return dict(
        grid_size=int(args.grid_size),
        C=int(args.C),
        k=int(args.k),
        kernel_components=int(args.kernel_components),
        M=parse_matrix_str(args.M),
        dd=int(args.dd),
        dt=float(args.dt),
        sigma=float(args.sigma),
        border=str(args.border),
        mix_rule=str(args.mix_rule),
        seed_patch_size=int(args.seed_patch_size),
        seed_n_patches=int(seed_n),
        seed_mode=str(args.seed_mode),
        p_constant_per_patch=bool(int(args.p_constant_per_patch)),
        render_mode=str(args.render_mode),
        clip1=float(args.clip1),
        clip2=float(args.clip2),
        mutation=bool(args.mutations),
        mutation_patch_size=int(args.mutation_sz),
        mutation_prob=float(args.mutation_p),
        volcano=bool(args.volcano),
        volcano_patch_size=int(args.volcano_sz),
        volcano_prob=float(args.volcano_p),
        volcano_delta_scale=float(args.volcano_delta),
        food_enabled=bool(args.food),
        food_spawn_interval=int(args.food_interval),
        food_n_patches=int(args.food_n),
        food_patch_size=int(args.food_sz),
        food_amount=float(args.food_amount),
        food_consume_rate=float(args.food_consume_rate),
        food_bonus=float(args.food_bonus),
        mass_decay=float(args.mass_decay),
        food_green_channel=int(args.food_channel),
        food_auto_size=bool(args.food_auto_size),
        food_auto_scale=float(args.food_auto_scale),
        food_conv_mode=str(args.food_conv_mode),
        food_vis_scale=float(args.food_vis_scale),
        food_vis_color=parse_color_str(args.food_vis_color),
        food_diffusion_alpha=float(args.food_diffusion_alpha),
        mass_clip_eps=float(args.mass_clip_eps),
    )
