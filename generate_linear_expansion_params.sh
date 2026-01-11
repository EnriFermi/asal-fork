#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="./data/linear_expansion"
SEED=0

python - << 'PY'
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import substrates


BASE_DIR = "./data/linear_expansion"
SEED = 0
LAMBDAS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def logit(x: np.ndarray) -> np.ndarray:
    return np.log(x) - np.log1p(-x)


def decode_R(params_flat, fl) -> float:
    n_dyn = fl.base_dyn_raw.size
    dyn_delta = jnp.asarray(params_flat[:n_dyn])
    raw_dyn = fl.base_dyn_raw + jnp.clip(dyn_delta, -fl.clip1, fl.clip1)
    norm_dyn = jax.nn.sigmoid(raw_dyn)
    dyn_vals = fl._dyn_lo + norm_dyn * (fl._dyn_hi - fl._dyn_lo)
    return float(dyn_vals[0])


def set_R(params_flat, fl, R_new: float) -> np.ndarray:
    lo, hi = fl.bounds["R"]
    R_new = float(np.clip(R_new, lo, hi))
    norm = (R_new - lo) / (hi - lo)
    norm = np.clip(norm, 1e-6, 1 - 1e-6)
    raw = float(logit(norm))
    delta0 = raw - float(fl.base_dyn_raw[0])
    if np.isfinite(fl.clip1):
        delta0 = float(np.clip(delta0, -fl.clip1, fl.clip1))
    params_new = np.array(params_flat, dtype=np.float32).copy()
    params_new[0] = delta0
    return params_new


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    fl = substrates.create_substrate("lenia_flow")

    rng = jax.random.PRNGKey(SEED)
    params = np.array(fl.default_params(rng)).reshape(-1)
    R0 = decode_R(params, fl)

    print(f"Base R: {R0:.6f}")
    for lam in LAMBDAS:
        R_new = lam * (R0 + 15.0) - 15.0
        params_scaled = set_R(params, fl, R_new)
        save_dir = os.path.join(BASE_DIR, f"scale_{lam:.2f}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "best.pkl"), "wb") as f:
            pickle.dump((params_scaled, None), f)
        print(f"Saved {save_dir}/best.pkl with R'={(lam * (R0 + 15.0) - 15.0):.6f}")


if __name__ == "__main__":
    main()
PY
