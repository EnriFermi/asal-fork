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
    mutation_scale = getattr(args, "mutation_scale", 1.0)
    optimize_mutation_scale = getattr(args, "optimize_mutation_scale", False)
    flow_sigma = getattr(args, "flow_sigma", None)
    if flow_sigma is None:
        flow_sigma = getattr(args, "sigma")
    return dict(
        grid_size=int(args.grid_size),
        C=int(args.C),
        k=int(args.k),
        kernel_components=int(args.kernel_components),
        M=parse_matrix_str(args.M),
        dd=int(args.dd),
        dt=float(args.dt),
        sigma=float(flow_sigma),
        border=str(args.border),
        mix_rule=str(args.mix_rule),
        base_seed=int(getattr(args, "base_seed", 0)),
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
        mutation_scale=float(mutation_scale),
        optimize_mutation_scale=bool(optimize_mutation_scale),
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


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def flow_lenia_palette_stats(params: np.ndarray, substrate: Any) -> Dict[str, Any] | None:
    base = substrate.substrate if hasattr(substrate, "substrate") else substrate
    if not hasattr(base, "base_dyn_raw"):
        return None
    if getattr(base, "render_mode", None) != "PcolorMix":
        return None
    n_dyn = int(base.base_dyn_raw.size)
    k = int(base.k)
    size = 3 * k
    if params is None or params.size < n_dyn + size:
        return None
    w_raw = np.asarray(params[n_dyn:n_dyn + size]).reshape(3, k)
    w_soft = _softmax_np(w_raw, axis=1)
    eps = 1e-8
    entropy = -np.sum(w_soft * np.log(w_soft + eps), axis=1)
    return dict(w_raw=w_raw, w_soft=w_soft, entropy=entropy)
