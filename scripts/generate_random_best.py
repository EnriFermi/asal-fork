import os
import sys

import jax
import numpy as np
from omegaconf import OmegaConf

import substrates
import util


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/generate_random_best.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("random_best", {}),
    )
    return cfg, flat


def main(cfg, args):
    proj_root = _project_root()
    save_dir = _resolve_path(str(getattr(args, "save_dir")), proj_root)

    seed = int(getattr(args, "seed", 0))
    fitness = float(getattr(args, "fitness", 0.0))

    if args.substrate == "lenia_flow":
        substrate = substrates.create_substrate(
            args.substrate,
            **util.flow_lenia_kwargs_from_args(args),
        )
    else:
        substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    params = substrate.default_params(jax.random.PRNGKey(seed))
    params_np = np.asarray(params)
    util.save_pkl(save_dir, "best", (params_np, fitness))

    print(
        f"Saved random best.pkl to {save_dir} "
        f"(substrate={args.substrate}, seed={seed}, n_params={params_np.size}, fitness={fitness})."
    )


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
