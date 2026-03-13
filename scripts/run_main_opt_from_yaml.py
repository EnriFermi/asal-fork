import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


SUPPORTED_FLAGS = {
    "seed",
    "save_dir",
    "substrate",
    "rollout_steps",
    "grid_size",
    "C",
    "k",
    "kernel_components",
    "M",
    "dd",
    "dt",
    "sigma",
    "border",
    "mix_rule",
    "base_seed",
    "seed_patch_size",
    "seed_n_patches",
    "mutations",
    "mutation_sz",
    "mutation_p",
    "mutation_scale",
    "optimize_mutation_scale",
    "volcano",
    "volcano_sz",
    "volcano_p",
    "volcano_delta",
    "seed_mode",
    "p_constant_per_patch",
    "render_mode",
    "clip1",
    "clip2",
    "food",
    "food_interval",
    "food_n",
    "food_sz",
    "food_amount",
    "food_consume_rate",
    "food_bonus",
    "mass_decay",
    "food_channel",
    "food_auto_size",
    "food_auto_scale",
    "food_conv_mode",
    "food_vis_scale",
    "food_vis_color",
    "food_diffusion_alpha",
    "mass_clip_eps",
    "foundation_model",
    "time_sampling",
    "prompts",
    "coef_prompt",
    "coef_softmax",
    "coef_oe",
    "coef_smooth",
    "bs",
    "pop_size",
    "n_iters",
    "sigma",
    "eval_splits",
}

BOOL_FLAGS = {
    "mutations",
    "optimize_mutation_scale",
    "volcano",
    "food",
    "food_auto_size",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_flat_config(config_path: Path) -> dict:
    cfg = OmegaConf.load(str(config_path))
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
    )
    return OmegaConf.to_container(flat, resolve=True)


def _build_argv(flat_cfg: dict) -> list[str]:
    argv: list[str] = []
    for key in SUPPORTED_FLAGS:
        if key not in flat_cfg:
            continue
        val = flat_cfg[key]
        if val is None:
            continue
        flag = f"--{key}"
        if key in BOOL_FLAGS:
            if bool(val):
                argv.append(flag)
            continue
        argv.extend([flag, str(val)])
    return argv


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_main_opt_from_yaml.py <config.yaml>")

    repo_root = _repo_root()
    config_path = Path(sys.argv[1]).resolve()
    flat_cfg = _load_flat_config(config_path)
    cmd = [sys.executable, str(repo_root / "scripts" / "main_opt.py"), *_build_argv(flat_cfg)]
    env = os.environ.copy()
    return subprocess.call(cmd, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
