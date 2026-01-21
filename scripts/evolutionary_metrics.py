import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt


def build_weighted_rgb_bins(frames_rgb_u8, frames_mass, q=4, min_bin_mass=0.0):
    """
    frames_rgb_u8: (T,H,W,3) uint8
    frames_mass:   (T,H,W)   float
    q: quantization step in RGB units (smaller = finer, slower)
    min_bin_mass: drop color-bins whose total mass across all frames is below this
    Returns:
      pts: (Nbins,3) float32, weighted mean RGB per occupied bin
      w:   (Nbins,)  float32, total mass per bin
    """
    T, H, W, _ = frames_rgb_u8.shape
    B = (256 + q - 1) // q
    M = B * B * B

    mass_sum = np.zeros(M, dtype=np.float64)
    rgb_sum = np.zeros((M, 3), dtype=np.float64)

    for t in range(T):
        m = frames_mass[t].reshape(-1)
        ok = m > 0
        if not ok.any():
            continue
        pix = frames_rgb_u8[t].reshape(-1, 3)[ok]
        w = m[ok].astype(np.float64)

        rb = (pix[:, 0] // q).astype(np.int32)
        gb = (pix[:, 1] // q).astype(np.int32)
        bb = (pix[:, 2] // q).astype(np.int32)
        key = (rb * B + gb) * B + bb

        mass_sum += np.bincount(key, weights=w, minlength=M)
        rgb_sum[:, 0] += np.bincount(key, weights=w * pix[:, 0], minlength=M)
        rgb_sum[:, 1] += np.bincount(key, weights=w * pix[:, 1], minlength=M)
        rgb_sum[:, 2] += np.bincount(key, weights=w * pix[:, 2], minlength=M)

    keys = np.nonzero(mass_sum > 0)[0]
    if min_bin_mass > 0:
        keys = keys[mass_sum[keys] >= min_bin_mass]

    w = mass_sum[keys].astype(np.float32)
    pts = (rgb_sum[keys] / mass_sum[keys][:, None]).astype(np.float32)
    return pts, w


def dpmeans_weighted(points, weights, lam=20.0, iters=8):
    """
    Weighted DP-means: creates new cluster if nearest center farther than lam (RGB L2 units).
    points:  (N,3) float32
    weights: (N,)  float32
    Returns:
      centers: (K,3) float32, deterministically ordered => stable label IDs across time
    """
    order = np.argsort(-weights)
    X = points[order]
    w = weights[order]

    centers = [X[0].copy()]
    lam2 = float(lam) * float(lam)

    for _ in range(iters):
        C = np.stack(centers, axis=0)  # (K,3)
        assign = np.empty(X.shape[0], dtype=np.int32)

        for i in range(X.shape[0]):
            d2 = np.sum((C - X[i]) ** 2, axis=1)
            k = int(np.argmin(d2))
            if d2[k] > lam2:
                centers.append(X[i].copy())
                C = np.vstack([C, X[i][None, :]])
                assign[i] = C.shape[0] - 1
            else:
                assign[i] = k

        K = int(assign.max()) + 1
        sums = np.zeros((K, 3), dtype=np.float64)
        sw = np.zeros(K, dtype=np.float64)
        np.add.at(sums, assign, (X * w[:, None]).astype(np.float64))
        np.add.at(sw, assign, w.astype(np.float64))
        Cnew = (sums / sw[:, None]).astype(np.float32)

        centers = [Cnew[k].copy() for k in range(K)]

    C = np.stack(centers, axis=0)
    order = np.lexsort((C[:, 2], C[:, 1], C[:, 0]))  # stable IDs by centroid RGB
    return C[order].astype(np.float32)


def make_jax_label_mass_batch(centers_rgb):
    centers = jnp.asarray(centers_rgb, dtype=jnp.float32)  # (K,3)
    K = int(centers.shape[0])

    palette = jnp.concatenate(
        [jnp.zeros((1, 3), jnp.uint8),
         jnp.clip(jnp.rint(centers), 0, 255).astype(jnp.uint8)],
        axis=0
    )  # (K+1,3)

    def _argmin_sqdist_and_dmin(X):
        N = X.shape[0]
        best_d = jnp.full((N,), jnp.inf, dtype=X.dtype)
        best_k = jnp.zeros((N,), dtype=jnp.int32)

        def body(k, state):
            bd, bk = state
            d = jnp.sum((X - centers[k]) ** 2, axis=1)
            take = d < bd
            return (jnp.where(take, d, bd), jnp.where(take, jnp.int32(k), bk))

        bd, bk = lax.fori_loop(0, K, body, (best_d, best_k))
        return bk + jnp.int32(1), bd

    def _label_one(rgb_u8, mass, mass_eps, rgb_void, max_rgb_dist):
        H, W, _ = rgb_u8.shape
        rgb_u8 = rgb_u8.astype(jnp.uint8)
        X = rgb_u8.reshape(-1, 3).astype(jnp.float32)

        lab, dmin = _argmin_sqdist_and_dmin(X)
        lab = lab.reshape(H, W)
        dmin = dmin.reshape(H, W)

        void = (mass <= mass_eps) | (jnp.max(rgb_u8, axis=-1) <= jnp.uint8(rgb_void))

        gate2 = jnp.where(max_rgb_dist > 0, max_rgb_dist * max_rgb_dist, jnp.inf)
        lab = jnp.where(dmin <= gate2, lab, jnp.int32(0))

        return jnp.where(void, jnp.int32(0), lab)

    @jax.jit
    def labels_frames(frames_rgb_u8, frames_mass, mass_eps, rgb_void=0, max_rgb_dist=0.0):
        return jax.vmap(lambda rgb, m: _label_one(rgb, m, mass_eps, rgb_void, max_rgb_dist))(
            frames_rgb_u8, frames_mass
        )

    @jax.jit
    def masses_frames(frames_rgb_u8, frames_mass, mass_eps, rgb_void=0, max_rgb_dist=0.0):
        def one(rgb, m):
            lab = _label_one(rgb, m, mass_eps, rgb_void, max_rgb_dist).reshape(-1)
            w = m.reshape(-1).astype(jnp.float32)
            return jnp.bincount(lab, weights=w, length=K + 1)
        return jax.vmap(one)(frames_rgb_u8, frames_mass)

    @jax.jit
    def render_labels_frames(labels):
        return palette[labels]  # (T,H,W,3) uint8

    return labels_frames, masses_frames, render_labels_frames, palette


def compute_mass_trajectories(frames_rgb_u8, frames_mass,
                             q=4, min_bin_mass=0.0,
                             lam=18.0, dp_iters=8,
                             mass_eps_rel=1e-6, rgb_void=3, max_rgb_dist=0.0,
                             backend=None, return_labels=False):
    """
    End-to-end:
      1) build global species centers via weighted DP-means in RGB space
      2) label every frame (0=void) and compute per-label total mass per time
    Returns:
      masses_T: (T, K+1) float32   masses_T[t, label] = total mass of label at time t
      centers:  (K,3) float32      representative RGB for each label 1..K
      palette:  (K+1,3) uint8      palette[0]=black, palette[i]=centers[i-1]
      labels_T: (T,H,W) int32      only if return_labels=True
    """
    pts, w = build_weighted_rgb_bins(frames_rgb_u8, frames_mass, q=q, min_bin_mass=min_bin_mass)
    centers = dpmeans_weighted(pts, w, lam=lam, iters=dp_iters)

    labels_fn, masses_fn, render_fn, palette = make_jax_label_mass_batch(centers)

    dev = None
    if backend is not None:
        dev = jax.devices(backend)[0]

    rgb_j = jax.device_put(frames_rgb_u8, device=dev) if dev is not None else jax.device_put(frames_rgb_u8)
    mass_j = jax.device_put(frames_mass, device=dev) if dev is not None else jax.device_put(frames_mass)

    mass_eps = float(mass_eps_rel) * float(np.max(frames_mass))
    masses_T = masses_fn(rgb_j, mass_j, mass_eps, rgb_void=rgb_void, max_rgb_dist=float(max_rgb_dist))
    masses_T = np.array(masses_T)

    if not return_labels:
        return masses_T, centers, np.array(palette)

    labels_T = labels_fn(rgb_j, mass_j, mass_eps, rgb_void=rgb_void, max_rgb_dist=float(max_rgb_dist))
    labels_T = np.array(labels_T)
    return masses_T, centers, np.array(palette), labels_T


def plot_mass_trajectories(masses_T, palette=None, include_void=False, top_k=12,
                           mass_floor=0.0, logy=False, figsize=(12, 6), title=None):
    """
    masses_T: (T, K+1)
    palette:  (K+1,3) uint8 (optional). If provided, line color matches label color.
    include_void: include label 0 trajectory
    top_k: plot only labels with largest total mass over time (keeps plot readable)
    mass_floor: ignore labels with total mass below this
    """
    M = np.asarray(masses_T, dtype=np.float64)
    T, L = M.shape
    totals = M.sum(axis=0)

    start = 0 if include_void else 1
    idx = np.arange(start, L, dtype=int)
    idx = idx[totals[idx] >= mass_floor]
    idx = idx[np.argsort(-totals[idx])][:top_k]

    x = (np.arange(T) * 256) / (250 * 60)

    plt.figure(figsize=figsize)
    for lab in idx:
        y = M[:, lab]
        if palette is not None:
            c = np.asarray(palette[lab], dtype=np.float64) / 255.0
            plt.plot(x, y, label=str(lab), color=c)
        else:
            plt.plot(x, y, label=str(lab))

    if logy:
        plt.yscale("log")

    plt.xlabel("time t")
    plt.ylabel("total mass")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Mass trajectories per label")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()
