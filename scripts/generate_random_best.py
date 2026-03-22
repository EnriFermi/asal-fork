import os
import argparse
import sys

import evosax
import jax
import numpy as np
from omegaconf import OmegaConf

import substrates
import simulate_save_apf
import util


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Generate random best and immediately run APF simulation for all random_batch.n_runs entries.",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def load_config(config_path: str, overrides=None):
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _flatten_generate_args(cfg):
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
        cfg.get("random_best", {}),
    )


def _flatten_simulation_args(cfg):
    return OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("simulation", {}),
        cfg.get("logging", {}),
    )


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _build_run_cfg(cfg, run_idx: int, n_runs: int):
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if run_cfg.get("meta") is None:
        run_cfg.meta = OmegaConf.create()
    if run_cfg.get("simulation") is None:
        run_cfg.simulation = OmegaConf.create()
    if run_cfg.get("logging") is None:
        run_cfg.logging = OmegaConf.create()
    if run_cfg.get("random_best") is None:
        run_cfg.random_best = OmegaConf.create()

    save_root = str(run_cfg.meta.get("save_dir"))
    single_output_dir = run_cfg.meta.get("output_dir", None)
    sim_seed_start = _as_int(run_cfg.simulation.get("seed", 0), 0)
    lagrangian_seed_start = _as_int(run_cfg.logging.get("lagrangian_seed", sim_seed_start), sim_seed_start)
    param_seed_start = _as_int(run_cfg.random_best.get("seed", 0), 0)

    init_mode = str(run_cfg.random_best.get("init_mode", "substrate_default")).strip().lower()
    cma_pop_size = _as_int(run_cfg.random_best.get("cma_pop_size", 1), 1)
    if init_mode == "sep_cma_es_ask" and cma_pop_size < 1:
        raise ValueError(f"random_best.cma_pop_size must be >= 1, got {cma_pop_size}.")

    if n_runs == 1:
        save_dir = save_root
        output_dir = (
            str(single_output_dir)
            if single_output_dir not in (None, "")
            else os.path.join(save_root, "apf_logs")
        )
    else:
        save_dir = os.path.join(save_root, f"run_{run_idx:03d}")
        output_dir = os.path.join(save_dir, "apf_logs")

    sim_seed = sim_seed_start + run_idx
    lagrangian_seed = lagrangian_seed_start + run_idx

    if init_mode == "sep_cma_es_ask":
        pop_round = run_idx // cma_pop_size
        member_idx = run_idx % cma_pop_size
        param_seed = param_seed_start + pop_round
    else:
        member_idx = _as_int(run_cfg.random_best.get("cma_member_idx", 0), 0)
        param_seed = param_seed_start + run_idx

    run_cfg.meta.save_dir = save_dir
    run_cfg.meta.output_dir = output_dir
    run_cfg.simulation.seed = sim_seed
    run_cfg.logging.lagrangian_seed = lagrangian_seed
    run_cfg.random_best.seed = param_seed
    run_cfg.random_best.init_mode = init_mode
    run_cfg.random_best.cma_member_idx = member_idx

    info = dict(
        init_mode=init_mode,
        param_seed=param_seed,
        member_idx=member_idx,
        sim_seed=sim_seed,
        save_dir=save_dir,
    )
    return run_cfg, info


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


def run_batch(cfg):
    n_runs = _as_int(cfg.get("random_batch", {}).get("n_runs", 1), 1)
    if n_runs < 1:
        raise ValueError(f"random_batch.n_runs must be >= 1, got {n_runs}.")

    for run_idx in range(n_runs):
        run_cfg, info = _build_run_cfg(cfg, run_idx, n_runs)
        print(
            f"[{run_idx + 1}/{n_runs}] "
            f"init_mode={info['init_mode']} "
            f"param_seed={info['param_seed']} "
            f"member_idx={info['member_idx']} "
            f"sim_seed={info['sim_seed']} "
            f"save_dir={info['save_dir']}"
        )
        main(run_cfg, _flatten_generate_args(run_cfg))
        simulate_save_apf.main(run_cfg, _flatten_simulation_args(run_cfg))


if __name__ == "__main__":
    cli = parse_cli()
    cfg = load_config(cli.config, cli.overrides)
    if cli.run_batch:
        run_batch(cfg)
    else:
        main(cfg, _flatten_generate_args(cfg))
