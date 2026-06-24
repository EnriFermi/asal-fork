from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from clip_deltah_msc_metric import resolve_metric_config, tau_selection_from_latent


class _DummyPickleClass:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except Exception:
            return _DummyPickleClass


def _load_pickle(path: Path, *, safe: bool = False) -> Any:
    with path.open("rb") as f:
        if safe:
            return _SafeUnpickler(f).load()
        return pickle.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _flat_optimization_config(path: Path) -> tuple[Any, SimpleNamespace]:
    cfg = OmegaConf.load(path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    return cfg, SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))


def _metric_config(args: SimpleNamespace) -> dict[str, Any]:
    # The audit must not instantiate Flow-Lenia or run any simulation.  For the
    # saved lockheed_1 configs these defaults match the resolved optimization
    # metric and are only used for metadata/tau decoding.
    if getattr(args, "metric_periodic", None) is None:
        args.metric_periodic = False
    if getattr(args, "metric_domain_y", None) is None:
        args.metric_domain_y = float(getattr(args, "grid_size", 0.0) or 0.0)
    if getattr(args, "metric_domain_x", None) is None:
        args.metric_domain_x = float(getattr(args, "grid_size", 0.0) or 0.0)
    return resolve_metric_config(args)


def _init_strategy(args: SimpleNamespace, candidate_dims: int):
    try:
        import evosax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "audit_flowlenia_optimization_run.py needs evosax to reconstruct "
            "the optimizer population. Run it in the same torchjax-style env "
            "used for Flow-Lenia optimization."
        ) from exc

    strategy = evosax.Sep_CMA_ES(
        popsize=int(args.pop_size),
        num_dims=candidate_dims,
        sigma_init=float(args.sigma),
    )
    es_params = strategy.default_params
    params_init = str(getattr(args, "params_init", "strategy_default")).strip().lower().replace("-", "_")
    rng = jax.random.PRNGKey(int(args.seed))
    if params_init in {"strategy_default", "optimizer_default", "default"}:
        rng, rng_init = jax.random.split(rng)
        es_state = strategy.initialize(rng_init, es_params)
        params_init = "strategy_default"
    else:
        raise ValueError(
            "This audit currently supports params_init='strategy_default' only; "
            f"got {params_init!r}."
        )
    return strategy, es_params, es_state, rng, params_init


def _load_resume_rng(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "resume_state.pkl"
    if not path.exists():
        return None
    try:
        state = _load_pickle(path, safe=True)
    except Exception:
        return None
    if not isinstance(state, dict) or "rng" not in state:
        return None
    return np.asarray(state["rng"], dtype=np.uint32)


def audit_run(
    run_dir: Path,
    *,
    output_dir: Path,
    atol: float,
) -> dict[str, Any]:
    cfg_path = run_dir / "optimization_config.yaml"
    pop_path = run_dir / "pop_traj.pkl"
    data_path = run_dir / "data.pkl"
    best_path = run_dir / "best.pkl"
    for path in (cfg_path, pop_path, data_path, best_path):
        if not path.exists():
            raise FileNotFoundError(path)

    _cfg, flat = _flat_optimization_config(cfg_path)
    pop = _load_pickle(pop_path)
    data = _load_pickle(data_path)
    best_params, best_loss = _load_pickle(best_path)
    pop_params = np.asarray(pop["params"], dtype=np.float32)
    pop_loss = np.asarray(pop["loss"], dtype=np.float32)
    pop_tau_raw = np.asarray(pop.get("tau_selector_raw", np.zeros(pop_loss.shape, dtype=np.float32)), dtype=np.float32)
    if pop_loss.ndim != 2:
        raise ValueError(f"pop loss must have shape (n_iters, pop_size), got {pop_loss.shape}")
    n_iters, pop_size = pop_loss.shape
    if int(flat.pop_size) != int(pop_size):
        raise ValueError(f"Config pop_size={flat.pop_size} but pop_traj has pop_size={pop_size}.")

    metric_cfg = _metric_config(flat)
    optimize_tau = str(metric_cfg.get("tau_mode", "fixed")) == "trainable_grid"
    tau_extra_dims = 1 if optimize_tau and "tau_selector_raw" in pop else 0
    substrate_param_dims = int(pop_params.shape[-1])
    candidate_dims = substrate_param_dims + tau_extra_dims
    strategy, es_params, es_state, rng, params_init = _init_strategy(flat, candidate_dims)

    best_flat = int(np.nanargmin(pop_loss))
    best_iter, best_pop_idx = np.unravel_index(best_flat, pop_loss.shape)
    best_loss_saved = float(np.asarray(best_loss).reshape(-1)[0])
    best_param_diff = float(np.max(np.abs(np.asarray(best_params, dtype=np.float32) - pop_params[best_iter, best_pop_idx])))

    data_loss = np.asarray(data.get("loss"), dtype=np.float32) if isinstance(data, dict) and "loss" in data else None
    data_pop_loss_max_abs_diff = (
        float(np.max(np.abs(data_loss - pop_loss)))
        if data_loss is not None and data_loss.shape == pop_loss.shape
        else None
    )

    rows: list[dict[str, Any]] = []
    max_param_diff = 0.0
    max_tau_diff = 0.0
    first_param_mismatch_iter: int | None = None
    first_tau_mismatch_iter: int | None = None
    predicted_eval_keys_best: list[list[int]] | None = None
    pop_batch = int(getattr(flat, "pop_batch", pop_size))
    bs = int(getattr(flat, "bs", 1))

    for i_iter in range(n_iters):
        rng, rng_ask = jax.random.split(rng)
        params_full, es_state = strategy.ask(rng_ask, es_state, es_params)
        params_full_np = np.asarray(jax.device_get(params_full), dtype=np.float32)
        params_np = params_full_np[:, :substrate_param_dims]
        param_diff = float(np.max(np.abs(params_np - pop_params[i_iter])))
        tau_diff = 0.0
        if tau_extra_dims:
            tau_raw_np = params_full_np[:, substrate_param_dims]
            tau_diff = float(np.max(np.abs(tau_raw_np - pop_tau_raw[i_iter])))
        if param_diff > max_param_diff:
            max_param_diff = param_diff
        if tau_diff > max_tau_diff:
            max_tau_diff = tau_diff
        if first_param_mismatch_iter is None and param_diff > atol:
            first_param_mismatch_iter = int(i_iter)
        if first_tau_mismatch_iter is None and tau_diff > atol:
            first_tau_mismatch_iter = int(i_iter)

        rng_eval = rng
        for start in range(0, pop_size, pop_batch):
            end = min(pop_size, start + pop_batch)
            rng_next, rng_metric_parent = jax.random.split(rng_eval)
            if i_iter == best_iter and start <= best_pop_idx < end:
                predicted_eval_keys_best = np.asarray(
                    jax.random.split(rng_metric_parent, bs),
                    dtype=np.uint32,
                ).tolist()
            rng_eval = rng_next
        rng = rng_eval

        loss_all = jnp.asarray(pop_loss[i_iter])
        es_state = strategy.tell(params_full, loss_all, es_state, es_params)
        rows.append(
            {
                "iter": int(i_iter),
                "param_max_abs_diff": param_diff,
                "tau_max_abs_diff": tau_diff,
                "pop_loss_min": float(np.min(pop_loss[i_iter])),
                "pop_loss_max": float(np.max(pop_loss[i_iter])),
                "pop_score_max": float(-np.min(pop_loss[i_iter])),
            }
        )

    final_rng_pred = np.asarray(jax.device_get(rng), dtype=np.uint32)
    final_rng_saved = _load_resume_rng(run_dir)
    final_rng_match = (
        bool(np.array_equal(final_rng_pred, final_rng_saved))
        if final_rng_saved is not None
        else None
    )

    score = -pop_loss.astype(np.float64).reshape(-1)
    score_quantiles = {
        str(q): float(np.percentile(score, q))
        for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)
    }
    tau_info = tau_selection_from_latent(metric_cfg, float(pop_tau_raw[best_iter, best_pop_idx]))
    summary = {
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "seed": int(flat.seed),
        "n_iters": int(n_iters),
        "pop_size": int(pop_size),
        "pop_batch": int(pop_batch),
        "bs": int(bs),
        "params_init": params_init,
        "substrate_param_dims": int(substrate_param_dims),
        "tau_extra_dims": int(tau_extra_dims),
        "metric_summary": {
            "objective": str(metric_cfg.get("objective")),
            "msc_term": str(metric_cfg.get("msc_term")),
            "scale_normalization": str(metric_cfg.get("scale_normalization")),
            "msc_floor": float(metric_cfg.get("msc_floor", 0.0)),
            "tau_mode": str(metric_cfg.get("tau_mode")),
            "tau_steps_list": [int(x) for x in metric_cfg.get("tau_steps_list", [])],
        },
        "best_iter": int(best_iter),
        "best_pop_idx": int(best_pop_idx),
        "best_loss_saved": best_loss_saved,
        "best_score_saved": float(-best_loss_saved),
        "best_loss_from_pop": float(pop_loss[best_iter, best_pop_idx]),
        "best_score_from_pop": float(-pop_loss[best_iter, best_pop_idx]),
        "best_param_diff_vs_pop": best_param_diff,
        "best_tau_info": tau_info,
        "data_pop_loss_max_abs_diff": data_pop_loss_max_abs_diff,
        "pop_reconstruction_max_param_abs_diff": max_param_diff,
        "pop_reconstruction_max_tau_abs_diff": max_tau_diff,
        "pop_reconstruction_match": bool(max(max_param_diff, max_tau_diff) <= atol),
        "first_param_mismatch_iter": first_param_mismatch_iter,
        "first_tau_mismatch_iter": first_tau_mismatch_iter,
        "predicted_final_rng": final_rng_pred.tolist(),
        "saved_final_rng": None if final_rng_saved is None else final_rng_saved.tolist(),
        "final_rng_match": final_rng_match,
        "predicted_eval_keys_best": predicted_eval_keys_best,
        "score_quantiles": score_quantiles,
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_count_ge_0_0005": int(np.sum(score >= 5e-4)),
        "score_count_ge_0_0006": int(np.sum(score >= 6e-4)),
        "score_count_ge_0_0007": int(np.sum(score >= 7e-4)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "optimization_run_audit_summary.json", summary)
    _write_csv(output_dir / "optimization_run_audit_iters.csv", rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a saved Flow-Lenia optimization run without rerunning simulations.")
    parser.add_argument("run_dir", help="Path to run_XXX directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write audit JSON/CSV. Default: analysis/results/flowlenia_optimization_audit/<run_name>",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if args.output_dir is None:
        output_dir = Path("analysis/results/flowlenia_optimization_audit") / run_dir.name
    else:
        output_dir = Path(args.output_dir)
    summary = audit_run(run_dir, output_dir=output_dir, atol=float(args.atol))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
