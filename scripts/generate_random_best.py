import os
import sys

import evosax
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


def _sample_params_substrate_default(substrate, *, seed: int):
    return substrate.default_params(jax.random.PRNGKey(seed))


def _sample_params_sep_cma_es_ask(
    substrate,
    *,
    seed: int,
    sigma_init: float,
    pop_size: int,
    member_idx: int,
):
    if pop_size < 1:
        raise ValueError(f"cma_pop_size must be >= 1, got {pop_size}.")
    if member_idx < 0 or member_idx >= pop_size:
        raise ValueError(
            f"cma_member_idx must be in [0, {pop_size - 1}], got {member_idx}."
        )

    rng = jax.random.PRNGKey(seed)
    strategy = evosax.Sep_CMA_ES(
        popsize=pop_size,
        num_dims=substrate.n_params,
        sigma_init=sigma_init,
    )
    es_params = strategy.default_params
    rng, rng_init, rng_ask = jax.random.split(rng, 3)
    es_state = strategy.initialize(rng_init, es_params)
    params_pop, _ = strategy.ask(rng_ask, es_state, es_params)
    return params_pop[member_idx]


def main(cfg, args):
    proj_root = _project_root()
    save_dir = _resolve_path(str(getattr(args, "save_dir")), proj_root)

    seed = int(getattr(args, "seed", 0))
    fitness = float(getattr(args, "fitness", 0.0))
    init_mode = str(getattr(args, "init_mode", "substrate_default")).strip().lower()

    if args.substrate == "lenia_flow":
        substrate = substrates.create_substrate(
            args.substrate,
            **util.flow_lenia_kwargs_from_args(args),
        )
    else:
        substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    if init_mode == "substrate_default":
        params = _sample_params_substrate_default(substrate, seed=seed)
    elif init_mode == "sep_cma_es_ask":
        sigma_init_raw = getattr(args, "cma_sigma_init", None)
        if sigma_init_raw is None:
            raise ValueError(
                "random_best.init_mode='sep_cma_es_ask' requires random_best.cma_sigma_init."
            )
        params = _sample_params_sep_cma_es_ask(
            substrate,
            seed=seed,
            sigma_init=float(sigma_init_raw),
            pop_size=int(getattr(args, "cma_pop_size", 1)),
            member_idx=int(getattr(args, "cma_member_idx", 0)),
        )
    else:
        raise ValueError(
            f"Unknown random_best.init_mode={init_mode!r}. "
            "Use 'substrate_default' or 'sep_cma_es_ask'."
        )

    params_np = np.asarray(params)
    util.save_pkl(save_dir, "best", (params_np, fitness))

    print(
        f"Saved random best.pkl to {save_dir} "
        f"(substrate={args.substrate}, seed={seed}, init_mode={init_mode}, "
        f"n_params={params_np.size}, fitness={fitness})."
    )


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
