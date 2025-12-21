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


def channel_maps(fl):
    """Return source/target channel for each kernel index."""
    src = list(fl.cfg.c0)
    tgt = [None] * len(src)
    for c, ks in enumerate(fl.cfg.c1):
        for kidx in ks:
            tgt[kidx] = c
    return src, tgt


def color_for_ch(ch):
    palette = {
        0: (1.0, 0.2, 0.2),  # red-ish
        1: (0.2, 0.9, 0.2),  # green-ish
        2: (0.2, 0.4, 1.0),  # blue-ish
    }
    return palette.get(int(ch), (0.7, 0.7, 0.7))


def plot_grid(kernels_over_time, output, crop_size, src_ch, tgt_ch):
    """
    kernels_over_time: list of (kH,kW,k) arrays, length T
    """
    T = len(kernels_over_time)
    kH, kW, k = kernels_over_time[0].shape
    if crop_size is not None:
        cs = int(crop_size)
        i0 = max(0, kH // 2 - cs // 2)
        j0 = max(0, kW // 2 - cs // 2)
        i1 = min(kH, i0 + cs)
        j1 = min(kW, j0 + cs)
        kernels_over_time = [K[i0:i1, j0:j1, :] for K in kernels_over_time]
        kH, kW, _ = kernels_over_time[0].shape
    fig, axs = plt.subplots(k, T, figsize=(2 * T, 2 * k), squeeze=False)

    vmin = min(k.min() for k in kernels_over_time)
    vmax = max(k.max() for k in kernels_over_time)

    for t, K in enumerate(kernels_over_time):
        for ki in range(k):
            ax = axs[ki, t]
            ax.imshow(K[:, :, ki], cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.axis("off")
            # Color-code spines: source channel on left/bottom, target channel on top/right
            scol = color_for_ch(src_ch[ki])
            tcol = color_for_ch(tgt_ch[ki])
            for spine in ("left", "bottom"):
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color(scol)
                ax.spines[spine].set_linewidth(3.0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color(tcol)
                ax.spines[spine].set_linewidth(3.0)
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
    ap.add_argument("--crop_size", type=int, default=None, help="If set, center-crop kernels to this size before plotting.")
    args = ap.parse_args()

    if args.substrate != "lenia_flow":
        raise ValueError("This script currently supports lenia_flow only.")

    fl = substrates.create_substrate("lenia_flow")
    fl = substrates.FlattenSubstrateParameters(fl)
    fl_base = fl.substrate  # unwrap for internal fields
    src_ch, tgt_ch = channel_maps(fl_base)

    kernels_over_time = []
    for path in args.checkpoints:
        params = load_params(path)
        params = np.asarray(params).reshape(-1)
        K = decode_kernels(params, fl_base)
        kernels_over_time.append(K)

    plot_grid(kernels_over_time, args.output, args.crop_size, src_ch, tgt_ch)


if __name__ == "__main__":
    main()
