import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Avoid large upfront GPU preallocation in the parent process.
# This script spawns child JAX processes; without this, parent may hold most VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from omegaconf import OmegaConf

import substrates
import util


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(str(path))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_cfg(config_path: Path):
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(str(config_path))
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("random_batch", {}),
    )
    return cfg, flat


def _make_substrate(flat_args):
    if str(flat_args.substrate) != "lenia_flow":
        raise ValueError(
            "This script supports only substrate='lenia_flow'. "
            f"Got substrate={flat_args.substrate!r}."
        )
    substrate = substrates.create_substrate(
        flat_args.substrate,
        **util.flow_lenia_kwargs_from_args(flat_args),
    )
    return substrates.FlattenSubstrateParameters(substrate)


def _ensure_video_defaults(cfg_i, run_root: Path):
    if "logging" not in cfg_i:
        cfg_i.logging = {}
    if "simulation" not in cfg_i:
        cfg_i.simulation = {}

    if not hasattr(cfg_i.logging, "wandb_project"):
        cfg_i.logging.wandb_project = "asal"

    sim = cfg_i.simulation
    sim.time_sampling = getattr(sim, "time_sampling", "video")
    sim.img_size = int(getattr(sim, "img_size", 224))
    sim.n_seeds = int(getattr(sim, "n_seeds", 1))
    sim.seed_mode = getattr(sim, "seed_mode", "random_patches")
    sim.p_constant_per_patch = int(getattr(sim, "p_constant_per_patch", 1))
    sim.render_mode = getattr(sim, "render_mode", "Pcolor")
    sim.fps = int(getattr(sim, "fps", 120))
    sim.codec = getattr(sim, "codec", "libx264")
    sim.macro_block_size = getattr(sim, "macro_block_size", None)
    sim.batch_steps = int(getattr(sim, "batch_steps", 256))
    sim.jit_microbatch = int(getattr(sim, "jit_microbatch", 64))
    sim.log_mass_every = int(getattr(sim, "log_mass_every", 1000))
    sim.traj_iter = getattr(sim, "traj_iter", None)
    sim.compute_oe = bool(getattr(sim, "compute_oe", False))
    sim.oe_every = int(getattr(sim, "oe_every", 100))
    sim.output = str(run_root / "simulation.mp4")
    sim.mass_plot = str(run_root / "mass.png")
    sim.oe_plot = str(run_root / "oe_loss.png")


def _run_one(
    *,
    project_root: Path,
    python_bin: str,
    base_cfg,
    run_root: Path,
    rollout_steps: int,
    param_seed: int,
    sim_seed: int,
    wandb_mode: str,
):
    run_root.mkdir(parents=True, exist_ok=True)
    save_dir = run_root / "checkpoint"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg_i = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if "meta" not in cfg_i:
        cfg_i.meta = {}
    if "substrate" not in cfg_i:
        cfg_i.substrate = {}
    if "simulation" not in cfg_i:
        cfg_i.simulation = {}
    if "logging" not in cfg_i:
        cfg_i.logging = {}

    cfg_i.meta.save_dir = str(save_dir)
    cfg_i.substrate.rollout_steps = int(rollout_steps)
    cfg_i.simulation.rollout_steps = int(rollout_steps)
    cfg_i.simulation.max_steps = int(rollout_steps)
    cfg_i.simulation.seed = int(sim_seed)

    _ensure_video_defaults(cfg_i, run_root)

    cfg_path = run_root / "config_video.yaml"
    OmegaConf.save(config=cfg_i, f=str(cfg_path))

    flat_i = OmegaConf.merge(
        cfg_i.get("meta", {}),
        cfg_i.get("substrate", {}),
        cfg_i.get("simulation", {}),
        cfg_i.get("logging", {}),
    )
    flat_i = OmegaConf.to_container(flat_i, resolve=True)
    args_i = argparse.Namespace(**flat_i)
    substrate = _make_substrate(args_i)
    params = np.asarray(substrate.default_params(jax.random.PRNGKey(int(param_seed))))
    util.save_pkl(str(save_dir), "best", (params, 0.0))
    util.save_json(
        str(run_root),
        "random_init_info",
        dict(
            param_seed=int(param_seed),
            sim_seed=int(sim_seed),
            n_params=int(params.size),
            rollout_steps=int(rollout_steps),
            video_output=str(cfg_i.simulation.output),
        ),
    )

    cmd = [python_bin, str(project_root / "scripts" / "simulate_after_training.py"), str(cfg_path)]
    env = os.environ.copy()
    env["WANDB_MODE"] = str(wandb_mode)
    # Keep child JAX process from grabbing ~75% VRAM by default.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sample random FlowLenia initializations and render video for each run."
    )
    parser.add_argument(
        "config",
        help="Base config YAML compatible with scripts/simulate_after_training.py",
    )
    parser.add_argument("--n-inits", type=int, default=10, help="Number of random initializations.")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200_000,
        help="Simulation steps per initialization.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Where to store init_XX folders. Default: <config.meta.save_dir>/random_video_batch_200k.",
    )
    parser.add_argument(
        "--param-seed-start",
        type=int,
        default=0,
        help="First seed for parameter sampling (incremented by init index).",
    )
    parser.add_argument(
        "--sim-seed-start",
        type=int,
        default=0,
        help="First simulation seed (incremented by init index).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing init_XX folder before re-running that init.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable for subprocess runs.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        help="WANDB_MODE for each run (default: disabled).",
    )
    args = parser.parse_args()

    project_root = _project_root()
    cfg_path = _resolve_path(args.config, project_root)
    cfg, flat = _load_cfg(cfg_path)

    meta_save_dir = flat.get("save_dir", None)
    if meta_save_dir is None and args.output_root is None:
        raise ValueError(
            "Could not infer output root because config has no meta.save_dir. "
            "Provide --output-root explicitly."
        )
    if args.output_root is None:
        output_root = _resolve_path(str(meta_save_dir), project_root) / "random_video_batch_200k"
    else:
        output_root = _resolve_path(args.output_root, project_root)
    output_root.mkdir(parents=True, exist_ok=True)

    n_inits = int(args.n_inits)
    if n_inits < 1:
        raise ValueError(f"--n-inits must be >= 1, got {n_inits}.")

    print(f"Base config: {cfg_path}")
    print(f"Output root: {output_root}")
    print(f"n_inits={n_inits}, rollout_steps={int(args.rollout_steps)}")

    for i in range(n_inits):
        run_root = output_root / f"init_{i:02d}"
        if run_root.exists() and args.overwrite:
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)

        pseed = int(args.param_seed_start) + i
        sseed = int(args.sim_seed_start) + i
        print(
            f"[{i+1}/{n_inits}] run_root={run_root} | "
            f"param_seed={pseed}, sim_seed={sseed}"
        )

        _run_one(
            project_root=project_root,
            python_bin=str(args.python_bin),
            base_cfg=cfg,
            run_root=run_root,
            rollout_steps=int(args.rollout_steps),
            param_seed=pseed,
            sim_seed=sseed,
            wandb_mode=str(args.wandb_mode),
        )

    print("Done.")


if __name__ == "__main__":
    main()
