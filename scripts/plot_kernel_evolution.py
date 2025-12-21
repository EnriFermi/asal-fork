import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import substrates
from substrates.lenia_flow.utils import get_kernels


def load_params(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0]  # (params, fitness)
    return obj


def decode_kernels(params_flat, fl):
    """
    Reconstruct spatial kernels (X,Y,k) from flattened FlowLenia params.
    """
    n_dyn = fl.base_dyn_raw.size
    dyn_delta = jnp.asarray(params_flat[:n_dyn])
    raw_dyn = fl.base_dyn_raw + jnp.clip(dyn_delta, -fl.clip1, fl.clip1)
    norm_dyn = jax.nn.sigmoid(raw_dyn)
    dyn_vals = fl._dyn_lo + norm_dyn * (fl._dyn_hi - fl._dyn_lo)

    idx = 0
    R = dyn_vals[idx]; idx += 1
    r = dyn_vals[idx:idx + fl.k]; idx += fl.k
    m = dyn_vals[idx:idx + fl.k]; idx += fl.k
    s = dyn_vals[idx:idx + fl.k]; idx += fl.k
    a = dyn_vals[idx:idx + fl.k * 3].reshape((fl.k, 3)); idx += fl.k * 3
    b = dyn_vals[idx:idx + fl.k * 3].reshape((fl.k, 3)); idx += fl.k * 3
    w = dyn_vals[idx:idx + fl.k * 3].reshape((fl.k, 3)); idx += fl.k * 3
    # fcr = dyn_vals[idx]  # unused for kernels

    nK = get_kernels(fl.cfg.X, fl.cfg.Y, fl.k, dict(R=R, r=r, a=a, w=w, b=b))
    nK = np.array(nK)  # (X,Y,k)
    return nK


def plot_grid(kernels_over_time, output):
    """
    kernels_over_time: list of (kH,kW,k) arrays, length T
    """
    T = len(kernels_over_time)
    kH, kW, k = kernels_over_time[0].shape
    fig, axs = plt.subplots(k, T, figsize=(2 * T, 2 * k), squeeze=False)

    vmin = min(k.min() for k in kernels_over_time)
    vmax = max(k.max() for k in kernels_over_time)

    for t, K in enumerate(kernels_over_time):
        for ki in range(k):
            ax = axs[ki, t]
            ax.imshow(K[:, :, ki], cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if ki == 0:
                ax.set_title(f"step {t}", fontsize=10)
            if t == 0:
                ax.set_ylabel(f"K{ki}", fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved kernel evolution to {output}")


def main():
    ap = argparse.ArgumentParser(description="Plot FlowLenia kernels over checkpoints.")
    ap.add_argument("checkpoints", nargs="+", help="Paths to best.pkl files (ordered over time).")
    ap.add_argument("--substrate", type=str, default="lenia_flow", help="Substrate name (must be lenia_flow).")
    ap.add_argument("--output", type=str, default="kernel_evolution.png", help="Output image path.")
    args = ap.parse_args()

    if args.substrate != "lenia_flow":
        raise ValueError("This script currently supports lenia_flow only.")

    fl = substrates.create_substrate("lenia_flow")
    fl = substrates.FlattenSubstrateParameters(fl)
    fl_base = fl.substrate  # unwrap for internal fields

    kernels_over_time = []
    for path in args.checkpoints:
        params = load_params(path)
        params = np.asarray(params).reshape(-1)
        K = decode_kernels(params, fl_base)
        kernels_over_time.append(K)

    plot_grid(kernels_over_time, args.output)


if __name__ == "__main__":
    main()
