import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


META_FLAGS = {
    "seed": "seed",
    "save_dir": "save_dir",
}

SUBSTRATE_FLAGS = {
    "substrate": "substrate",
    "rollout_steps": "rollout_steps",
    "grid_size": "grid_size",
    "C": "C",
    "k": "k",
    "kernel_components": "kernel_components",
    "M": "M",
    "dd": "dd",
    "dt": "dt",
    "sigma": "flow_sigma",
    "border": "border",
    "mix_rule": "mix_rule",
    "base_seed": "base_seed",
    "seed_patch_size": "seed_patch_size",
    "seed_n_patches": "seed_n_patches",
    "mutations": "mutations",
    "mutation_sz": "mutation_sz",
    "mutation_p": "mutation_p",
    "mutation_scale": "mutation_scale",
    "optimize_mutation_scale": "optimize_mutation_scale",
    "volcano": "volcano",
    "volcano_sz": "volcano_sz",
    "volcano_p": "volcano_p",
    "volcano_delta": "volcano_delta",
    "seed_mode": "seed_mode",
    "p_constant_per_patch": "p_constant_per_patch",
    "render_mode": "render_mode",
    "clip1": "clip1",
    "clip2": "clip2",
    "food": "food",
    "food_interval": "food_interval",
    "food_n": "food_n",
    "food_sz": "food_sz",
    "food_amount": "food_amount",
    "food_consume_rate": "food_consume_rate",
    "food_bonus": "food_bonus",
    "mass_decay": "mass_decay",
    "food_channel": "food_channel",
    "food_auto_size": "food_auto_size",
    "food_auto_scale": "food_auto_scale",
    "food_conv_mode": "food_conv_mode",
    "food_vis_scale": "food_vis_scale",
    "food_vis_color": "food_vis_color",
    "food_diffusion_alpha": "food_diffusion_alpha",
    "mass_clip_eps": "mass_clip_eps",
}

EVALUATION_FLAGS = {
    "foundation_model": "foundation_model",
    "time_sampling": "time_sampling",
    "prompts": "prompts",
    "coef_prompt": "coef_prompt",
    "coef_softmax": "coef_softmax",
    "coef_oe": "coef_oe",
    "coef_smooth": "coef_smooth",
}

OPTIMIZATION_FLAGS = {
    "bs": "bs",
    "pop_size": "pop_size",
    "n_iters": "n_iters",
    "sigma": "sigma",
    "eval_splits": "eval_splits",
}

LOGGING_FLAGS = {
    "wandb_project": "wandb_project",
    "pca_every": "pca_every",
    "pca_history": "pca_history",
    "full_video_interval": "full_video_interval",
    "full_video_rollout_steps": "full_video_rollout_steps",
    "full_video_img_size": "full_video_img_size",
}

BOOL_SOURCE_KEYS = {
    "mutations",
    "optimize_mutation_scale",
    "volcano",
    "food",
    "food_auto_size",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_cfg(config_path: Path):
    return OmegaConf.load(str(config_path))


def _append_section_args(argv: list[str], section_cfg, key_map: dict[str, str]) -> None:
    if section_cfg is None:
        return
    section = OmegaConf.to_container(section_cfg, resolve=True)
    for src_key, cli_key in key_map.items():
        if src_key not in section:
            continue
        val = section[src_key]
        if val is None:
            continue
        flag = f"--{cli_key}"
        if src_key in BOOL_SOURCE_KEYS:
            if bool(val):
                argv.append(flag)
            continue
        argv.extend([flag, str(val)])


def _build_argv(cfg) -> list[str]:
    argv: list[str] = []
    _append_section_args(argv, cfg.get("meta", {}), META_FLAGS)
    _append_section_args(argv, cfg.get("substrate", {}), SUBSTRATE_FLAGS)
    _append_section_args(argv, cfg.get("evaluation", {}), EVALUATION_FLAGS)
    _append_section_args(argv, cfg.get("optimization", {}), OPTIMIZATION_FLAGS)
    _append_section_args(argv, cfg.get("logging", {}), LOGGING_FLAGS)
    return argv


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_main_opt_from_yaml.py <config.yaml>")

    repo_root = _repo_root()
    config_path = Path(sys.argv[1]).resolve()
    cfg = _load_cfg(config_path)
    cmd = [sys.executable, str(repo_root / "scripts" / "main_opt.py"), *_build_argv(cfg)]
    env = os.environ.copy()
    return subprocess.call(cmd, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
