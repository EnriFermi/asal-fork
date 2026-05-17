# dist_pipeline.py
# Pipeline: local distribution clustering by distance between P_{i,w,tau} (objects = (particle, window)).
# Distance: sliced 1D Wasserstein-1 over random projections (fast, bandwidth-free).
# Prototypes: global medoids per tau, learned on selected windows (uniform + hard by DeltaH).
# Outputs:
#   PASS A: Valid(w,tau), DeltaH(w,tau)
#   PASS B: prototypes per tau + stability curves + NEW: per-tau cluster separation metrics (geometry-based)
#   PASS C: pi(w,tau,k), pi_fast(w,tau), sep_ratio(w,tau), TV(w,tau), offdiag(w,tau)
#          + NEW: entropy/complexity of regime mixture: H_pi(w,tau), H_norm(w,tau), K_eff(w,tau)
#          + NEW: per-tau summary plots for separation metrics vs tau (optional)
#
# Requirements: numpy, matplotlib, tqdm, and project imports:
#   from flow_drift_metrics import infer_log_format, iter_npz_snapshots, load_state
#
# NOTE: "i,j" in D_ij are indices of objects a=(particle,window) in the training collection for a fixed tau.

import os
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_drift_metrics import infer_log_format, iter_npz_snapshots, load_state


# =========================
# USER CONFIG (edit here)
# =========================
save_pth = "experiments/flow_lenia_apf_rollouts/checkpoints/2602231617/interp_2"
log_format = "auto"                  # "auto" / "npz" / "pickle"

t1, t2 = 0, 1_000_000                # physical sim steps to analyze

window_size = 20_000                 # physical steps per window
window_step = 20_000                 # physical step between window starts

tau_list = [1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 56, 60, 70, 80, 100, 120, 140, 160, 180, 200]  # lag in FRAME units

# cell validity: trust (w,tau) if enough increments per particle for this lag
m_min = 100

# scan pass (cheap heterogeneity for hard-window selection)
S_scan = 256
m_scan = 48
null_reps_scan = 6

# training pass (global prototypes per tau)
n_train_uniform = 20
n_train_hard = 20
particles_per_train_window = 96
m_train = 64
n_proj = 16

# k selection (global)
n_min_cluster = 30
K_cap = 12
B_stab = 24
subsample_frac = 0.85
swap_trials = 250
seed = 0

# application pass
m_apply = 64
chunk_particles = 512

# periodic domain (set to None if not periodic)  -- wall mode => keep None
Lx = None
Ly = None

# outputs
out_dir = "metrics/dist_pipeline_A_interp2"
make_plots = True

# extra: per-tau cluster geometry/separation metrics
compute_tau_separation_metrics = True
save_small_tau_artifacts = True      # saves only small arrays (KxK medoid distances etc.)
save_full_train_labels = False       # if True: save labels for all training objects (can be big but ok)
save_full_train_D = False            # if True: saves D_train.npy (can be huge across taus)


# =========================
# Utilities: IO and window iterator
# =========================
def iter_lagrangian_xy(save_pth, t1, t2, log_format="auto"):
    fmt = log_format
    if fmt == "auto":
        fmt = infer_log_format(save_pth)
    if fmt == "npz":
        for t, s in iter_npz_snapshots(save_pth, t1, t2, fields=("lagrangian_xy",)):
            yield int(t), s["lagrangian_xy"]
    elif fmt == "pickle":
        for t in range(int(t1), int(t2) + 1):
            st = load_state(save_pth, t)
            xy = st.get("lagrangian_xy", None) if isinstance(st, dict) else getattr(st, "lagrangian_xy", None)
            if xy is None:
                raise ValueError(f"lagrangian_xy missing at t={t}")
            yield int(t), xy
    else:
        raise ValueError(f"Unknown log_format={log_format!r}")


def iter_windows(save_pth, t1, t2, window_size, window_step, log_format="auto"):
    """
    Yields one window at a time:
      (w_idx, ws, we, tw, Xw)
    tw: (Tw,) int64 physical times
    Xw: (Tw,N,2) float32 positions
    """
    ws = None
    we = None
    tw_buf = []
    X_buf = []
    for t, xy in iter_lagrangian_xy(save_pth, t1, t2, log_format=log_format):
        if ws is None:
            ws = (t // window_step) * window_step
            we = ws + window_size

        while t >= we:
            w_idx = int(ws // window_step)
            if len(tw_buf) > 0:
                tw = np.asarray(tw_buf, np.int64)
                Xw = np.stack(X_buf, axis=0).astype(np.float32)
            else:
                tw = np.asarray([], np.int64)
                Xw = None
            yield w_idx, ws, we, tw, Xw
            ws += window_step
            we = ws + window_size
            tw_buf = []
            X_buf = []

        if ws <= t < we:
            tw_buf.append(t)
            X_buf.append(np.asarray(xy, np.float32))

    if ws is not None:
        w_idx = int(ws // window_step)
        if len(tw_buf) > 0:
            tw = np.asarray(tw_buf, np.int64)
            Xw = np.stack(X_buf, axis=0).astype(np.float32)
        else:
            tw = np.asarray([], np.int64)
            Xw = None
        yield w_idx, ws, ws + window_size, tw, Xw


# =========================
# Geometry: periodic deltas
# =========================
def delta_periodic(dx, Lx=None, Ly=None):
    if Lx is not None:
        dx[..., 0] = (dx[..., 0] + 0.5 * Lx) % Lx - 0.5 * Lx
    if Ly is not None:
        dx[..., 1] = (dx[..., 1] + 0.5 * Ly) % Ly - 0.5 * Ly
    return dx


# =========================
# Core: signatures for distributions P_{i,w,tau}
# =========================
def make_dirs(n_proj, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, 2)).astype(np.float32)
    nrm = np.sqrt((dirs * dirs).sum(axis=1, keepdims=True))
    dirs = dirs / np.maximum(nrm, 1e-12)
    return dirs  # (L,2)

DIRS = make_dirs(n_proj, seed=seed + 123)


def sample_velocities_window(Xw, tw, tau, k_idx, particle_idx=None, Lx=None, Ly=None):
    """
    Xw: (Tw,N,2), tw: (Tw,)
    tau: lag in frames
    k_idx: (m,) indices into [0, Tw-tau)
    particle_idx: array of particle ids (None => all)
    Returns v_s: (m, Ns, 2) float32
    """
    Tw = Xw.shape[0]
    if tau >= Tw:
        return None
    k_idx = np.asarray(k_idx, np.int64)
    if k_idx.size == 0:
        return None
    if k_idx.max(initial=0) >= (Tw - tau):
        return None

    if particle_idx is None:
        X0 = Xw[k_idx, :, :]
        X1 = Xw[k_idx + tau, :, :]
    else:
        particle_idx = np.asarray(particle_idx, np.int64)
        X0 = Xw[k_idx][:, particle_idx, :]
        X1 = Xw[k_idx + tau][:, particle_idx, :]

    dx = (X1 - X0).astype(np.float32)
    if (Lx is not None) or (Ly is not None):
        dx = delta_periodic(dx, Lx=Lx, Ly=Ly)

    dt = (tw[k_idx + tau] - tw[k_idx]).astype(np.float32)  # (m,)
    v = dx / np.maximum(dt[:, None, None], 1e-12)
    return v


def signature_from_v(v_s, dirs):
    """
    v_s: (m, Ns, 2)
    dirs: (L,2)
    Signature per particle = concatenation of sorted projected samples for each dir.
    Returns Sig: (Ns, L*m) float32
    """
    m, Ns, _ = v_s.shape
    proj = v_s[..., 0:1] * dirs[:, 0][None, None, :] + v_s[..., 1:2] * dirs[:, 1][None, None, :]
    proj = np.sort(proj, axis=0)  # (m,Ns,L)
    Sig = np.transpose(proj, (1, 2, 0)).reshape(Ns, -1).astype(np.float32)  # (Ns, L*m)
    return Sig


def mean_pairwise_l1(Sig):
    """
    Sig: (S,D)
    returns mean_{i<j} mean_abs(Sig[i]-Sig[j])
    """
    S = Sig.shape[0]
    if S <= 1:
        return 0.0
    D = np.mean(np.abs(Sig[:, None, :] - Sig[None, :, :]), axis=2)  # (S,S)
    triu = np.triu_indices(S, k=1)
    return float(D[triu].mean())


# =========================
# Distance matrix on signatures (training)
# =========================
def pairwise_dist_matrix(Sig, chunk=16):
    """
    Sig: (M,D)
    returns D: (M,M) float32 where D[a,b] = mean_abs(Sig[a]-Sig[b])
    """
    M, Ddim = Sig.shape
    D = np.empty((M, M), np.float32)
    for i0 in tqdm(range(0, M, chunk), desc="pairwise D rows", leave=False):
        i1 = min(M, i0 + chunk)
        block = Sig[i0:i1, None, :]  # (c,1,D)
        d = np.mean(np.abs(block - Sig[None, :, :]), axis=2).astype(np.float32)
        D[i0:i1, :] = d
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# =========================
# k-medoids on precomputed distance matrix
# =========================
def kmedoids_build(D, K, rng):
    M = D.shape[0]
    m0 = int(np.argmin(D.sum(axis=1)))
    medoids = [m0]
    best = D[:, m0].copy()
    while len(medoids) < K:
        costs = np.empty(M, np.float64)
        for j in range(M):
            if j in medoids:
                costs[j] = np.inf
            else:
                costs[j] = np.minimum(best, D[:, j]).sum()
        mj = int(np.argmin(costs))
        medoids.append(mj)
        best = np.minimum(best, D[:, mj])
    return np.array(medoids, np.int64)


def assign_to_medoids(D, medoids):
    dm = D[:, medoids]  # (M,K)
    order = np.argsort(dm, axis=1)
    labels = order[:, 0]
    d1 = dm[np.arange(dm.shape[0]), order[:, 0]]
    d2 = dm[np.arange(dm.shape[0]), order[:, 1]] if dm.shape[1] > 1 else np.full_like(d1, np.inf)
    return labels, d1, d2


def kmedoids_refine(D, medoids, swap_trials, rng):
    M = D.shape[0]
    medoids = medoids.copy()
    is_med = np.zeros(M, dtype=bool)
    is_med[medoids] = True

    def cost(meds):
        dm = D[:, meds]
        return float(np.min(dm, axis=1).sum())

    best_cost = cost(medoids)

    for _ in range(swap_trials):
        out_idx = int(rng.integers(0, len(medoids)))
        out_med = int(medoids[out_idx])
        in_cand = int(rng.integers(0, M))
        if is_med[in_cand]:
            continue
        new_medoids = medoids.copy()
        new_medoids[out_idx] = in_cand
        c = cost(new_medoids)
        if c < best_cost:
            is_med[out_med] = False
            is_med[in_cand] = True
            medoids = new_medoids
            best_cost = c
    return medoids, best_cost


# =========================
# ARI for stability (no sklearn)
# =========================
def adjusted_rand_index(labels_true, labels_pred):
    labels_true = np.asarray(labels_true, np.int64)
    labels_pred = np.asarray(labels_pred, np.int64)
    n = int(labels_true.size)
    if n == 0:
        return 0.0

    _, lt = np.unique(labels_true, return_inverse=True)
    _, lp = np.unique(labels_pred, return_inverse=True)

    k1 = int(lt.max() + 1)
    k2 = int(lp.max() + 1)
    cont = np.zeros((k1, k2), dtype=np.int64)
    for a, b in zip(lt, lp):
        cont[a, b] += 1

    def comb2(x):
        x = np.asarray(x)
        return (x * (x - 1)) // 2

    nij = cont
    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)

    index = int(comb2(nij).sum())
    sum_ai = int(comb2(ai).sum())
    sum_bj = int(comb2(bj).sum())

    total = n * (n - 1) // 2
    if total == 0:
        return 0.0

    expected = (sum_ai * sum_bj) / float(total)
    max_index = 0.5 * (sum_ai + sum_bj)

    denom = max_index - expected
    if denom == 0.0:
        return 0.0
    return float((index - expected) / denom)


def stability_score(D, K, B, subsample_frac, swap_trials, rng):
    M = D.shape[0]
    m_sub = max(int(math.floor(M * subsample_frac)), K + 2)
    runs = []
    runs_null = []

    for b in range(B):
        idx = rng.choice(M, size=m_sub, replace=False)
        idx.sort()
        Dsub = D[np.ix_(idx, idx)]
        meds = kmedoids_build(Dsub, K, rng)
        meds, _ = kmedoids_refine(Dsub, meds, swap_trials, rng)
        lab, _, _ = assign_to_medoids(Dsub, meds)
        runs.append((idx, lab))

        lab_null = lab.copy()
        rng.shuffle(lab_null)
        runs_null.append((idx, lab_null))

    pairs = []
    for _ in range(min(60, B * (B - 1) // 2)):
        a = int(rng.integers(0, B))
        b = int(rng.integers(0, B - 1))
        if b >= a:
            b += 1
        pairs.append((a, b))

    aris = []
    aris_null = []
    for a, b in pairs:
        idx_a, lab_a = runs[a]
        idx_b, lab_b = runs[b]
        inter, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True)
        if inter.size < K + 2:
            continue
        aris.append(adjusted_rand_index(lab_a[ia], lab_b[ib]))

        idx_a, lab_a = runs_null[a]
        idx_b, lab_b = runs_null[b]
        inter, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True)
        if inter.size < K + 2:
            continue
        aris_null.append(adjusted_rand_index(lab_a[ia], lab_b[ib]))

    if len(aris) == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(aris))
    med0 = float(np.median(aris_null)) if len(aris_null) else 0.0
    return float(med - med0), med, med0


# =========================
# NEW: per-tau cluster separation metrics (computed from training D + final labels)
# =========================
def tau_separation_metrics(D_train, medoids, labels, eps=1e-12):
    """
    D_train: (M,M)
    medoids: (K,) indices in [0,M)
    labels: (M,) in [0,K)
    Returns dict of scalar metrics + small arrays (K sizes/radii/medoid distances)
    """
    M = D_train.shape[0]
    K = int(medoids.size)

    sizes = np.bincount(labels, minlength=K).astype(np.int64)

    radii_mean = np.zeros(K, np.float32)
    radii_q90 = np.zeros(K, np.float32)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            radii_mean[k] = np.nan
            radii_q90[k] = np.nan
            continue
        d = D_train[idx, int(medoids[k])]
        radii_mean[k] = float(np.mean(d))
        radii_q90[k] = float(np.quantile(d, 0.9))

    w = sizes.astype(np.float64) / max(float(M), 1.0)
    intra_mean = float(np.nansum(w * radii_mean))
    intra_q90 = float(np.nanmax(radii_q90)) if np.isfinite(radii_q90).any() else np.nan

    # medoid distance matrix (K,K)
    Dm = D_train[np.ix_(medoids, medoids)].astype(np.float32)
    Dm = 0.5 * (Dm + Dm.T)
    np.fill_diagonal(Dm, 0.0)

    if K <= 1:
        inter_min = np.nan
        inter_mean = np.nan
        dunn = np.nan
        db = np.nan
    else:
        triu = np.triu_indices(K, k=1)
        inter_min = float(np.min(Dm[triu]))
        inter_mean = float(np.mean(Dm[triu]))
        dunn = float(inter_min / (intra_mean + eps))

        # Davies–Bouldin with medoids: DB = mean_k max_{l!=k} ( (r_k + r_l) / d(m_k,m_l) )
        # use radii_mean as r_k
        db_terms = []
        for k in range(K):
            rk = float(radii_mean[k])
            if not np.isfinite(rk):
                continue
            worst = -np.inf
            for l in range(K):
                if l == k:
                    continue
                rl = float(radii_mean[l])
                dkl = float(Dm[k, l])
                if not (np.isfinite(rl) and np.isfinite(dkl) and dkl > 0):
                    continue
                worst = max(worst, (rk + rl) / dkl)
            if np.isfinite(worst):
                db_terms.append(worst)
        db = float(np.mean(db_terms)) if len(db_terms) else np.nan

    return dict(
        M=int(M),
        K=int(K),
        sizes=sizes,
        radii_mean=radii_mean,
        radii_q90=radii_q90,
        intra_mean=float(intra_mean),
        intra_q90=float(intra_q90) if np.isfinite(intra_q90) else np.nan,
        inter_min=float(inter_min) if np.isfinite(inter_min) else np.nan,
        inter_mean=float(inter_mean) if np.isfinite(inter_mean) else np.nan,
        dunn=float(dunn) if np.isfinite(dunn) else np.nan,
        db=float(db) if np.isfinite(db) else np.nan,
        Dm=Dm,
    )


# =========================
# Plot helpers
# =========================
def save_heatmap(mat, x_labels, y_labels, title, out_png, vmin=None, vmax=None, cmap=None):
    plt.figure(figsize=(10.5, 4.8))
    plt.imshow(mat, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_line(x, y, title, xlabel, ylabel, out_png, ylim=None):
    plt.figure(figsize=(8.0, 4.0))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def entropy_from_pi(pik, eps=1e-12):
    """
    pik: (K,) nonnegative, sum~1
    Returns H = -sum p log p
    """
    p = np.asarray(pik, np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + eps)))


# =========================
# MAIN PIPELINE
# =========================
def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    tau_list_sorted = list(map(int, tau_list))
    tau_labels = [str(t) for t in tau_list_sorted]
    tau_to_scan_ti = {int(t): i for i, t in enumerate(tau_list_sorted)}

    # ---------- PASS A: scan windows; compute validity + DeltaH ----------
    scan_meta = []  # list of (w_idx, ws, we, Tw)
    DeltaH = {}     # w_idx -> (Ttau,)
    Valid = {}      # w_idx -> (Ttau,) bool

    print("PASS A: scan windows (validity + DeltaH) ...")
    for w_idx, ws, we, tw, Xw in tqdm(
        iter_windows(save_pth, t1, t2, window_size, window_step, log_format=log_format),
        desc="scan windows"
    ):
        Tw = int(tw.size)
        scan_meta.append((w_idx, ws, we, Tw))
        if Tw == 0:
            DeltaH[w_idx] = np.full(len(tau_list_sorted), np.nan, np.float32)
            Valid[w_idx] = np.zeros(len(tau_list_sorted), bool)
            continue

        N = Xw.shape[1]
        p_scan = rng.choice(N, size=min(S_scan, N), replace=False)

        dh = np.full(len(tau_list_sorted), np.nan, np.float32)
        vd = np.zeros(len(tau_list_sorted), bool)

        for ti, tau in enumerate(tau_list_sorted):
            if tau >= Tw:
                continue
            Tseg = Tw - tau
            if Tseg < m_min:
                continue
            vd[ti] = True

            m0 = min(m_scan, Tseg)
            k_idx = rng.choice(Tseg, size=m0, replace=False)
            k_idx.sort()

            v_s = sample_velocities_window(Xw, tw, tau, k_idx, particle_idx=p_scan, Lx=Lx, Ly=Ly)
            if v_s is None:
                vd[ti] = False
                continue

            Sig = signature_from_v(v_s, DIRS)
            H_real = mean_pairwise_l1(Sig)

            v_pool = v_s.reshape(-1, 2)
            H_nulls = []
            for _ in range(null_reps_scan):
                idx = rng.integers(0, v_pool.shape[0], size=(m0, Sig.shape[0]))
                v_pseudo = v_pool[idx]  # (m0, S_scan, 2)
                Sig0 = signature_from_v(v_pseudo, DIRS)
                H_nulls.append(mean_pairwise_l1(Sig0))
            H0 = float(np.median(H_nulls))
            dh[ti] = float(H_real - H0)

        DeltaH[w_idx] = dh
        Valid[w_idx] = vd

    scan_meta.sort(key=lambda x: x[0])
    w_indices = [m[0] for m in scan_meta]
    W = len(w_indices)
    Ttau = len(tau_list_sorted)

    DeltaH_mat = np.vstack([DeltaH[w] for w in w_indices]).astype(np.float32)  # (W,Ttau)
    Valid_mat = np.vstack([Valid[w] for w in w_indices]).astype(bool)          # (W,Ttau)

    np.save(os.path.join(out_dir, "w_indices.npy"), np.asarray(w_indices, np.int64))
    np.save(os.path.join(out_dir, "window_meta.npy"), np.asarray(scan_meta, dtype=object))
    np.save(os.path.join(out_dir, "tau_list.npy"), np.asarray(tau_list_sorted, np.int64))
    np.save(os.path.join(out_dir, "DeltaH.npy"), DeltaH_mat)
    np.save(os.path.join(out_dir, "Valid.npy"), Valid_mat.astype(np.uint8))

    if make_plots:
        save_heatmap(
            Valid_mat.astype(float), tau_labels, [str(w) for w in w_indices],
            "Validity mask (1=valid cell)", os.path.join(out_dir, "valid_mask.png"),
            vmin=0, vmax=1
        )
        vmax = np.nanpercentile(np.abs(DeltaH_mat), 98) if np.isfinite(DeltaH_mat).any() else 1.0
        save_heatmap(
            DeltaH_mat, tau_labels, [str(w) for w in w_indices],
            "DeltaH(w,tau) = H_real - median(H_null) (scan)", os.path.join(out_dir, "DeltaH.png"),
            vmin=-vmax, vmax=vmax, cmap="coolwarm"
        )

    # ---------- PASS B: train global prototypes per tau ----------
    print("PASS B: train global prototypes per tau ...")
    proto = {}  # tau -> dict
    w_to_pos = {w: i for i, w in enumerate(w_indices)}

    def select_train_windows_for_tau(ti):
        base = np.unique(np.rint(np.linspace(0, W - 1, n_train_uniform)).astype(int)).tolist()
        base_w = [w_indices[i] for i in base]

        d = DeltaH_mat[:, ti].copy()
        v = Valid_mat[:, ti].copy()
        d[~v] = -np.inf
        hard_idx = np.argsort(-d)[:n_train_hard]
        hard_w = [w_indices[i] for i in hard_idx if np.isfinite(d[i])]

        ws = list(dict.fromkeys(base_w + hard_w))
        return ws

    tau_sep_summary = {}  # tau -> dict from tau_separation_metrics

    for ti, tau in enumerate(tqdm(tau_list_sorted, desc="train tau")):
        if int(Valid_mat[:, ti].sum()) < 3:
            continue

        train_w = select_train_windows_for_tau(ti)
        needed = set(train_w)
        rng_tau = np.random.default_rng(seed + 1000 + int(tau))

        obj_sigs = []
        obj_meta = []  # (w_idx, particle_id)

        # per-object speed/velocity stats (so medoid characterization does not require re-reading)
        obj_sp_q10 = []
        obj_sp_q50 = []
        obj_sp_q90 = []
        obj_sp_q99 = []
        obj_sp_mean = []
        obj_sp_std = []
        obj_tail = []     # q90/(q50+eps)
        obj_aniso = []    # anisotropy of velocity covariance eigenvalues: lam_max/(lam_min+eps)

        for w_idx, ws, we, tw, Xw in tqdm(
            iter_windows(save_pth, t1, t2, window_size, window_step, log_format=log_format),
            desc=f"collect train objects tau={tau}", leave=False
        ):
            if w_idx not in needed:
                continue
            Tw = int(tw.size)
            if Tw == 0 or tau >= Tw:
                continue
            Tseg = Tw - tau
            if Tseg < max(m_train, m_min):
                continue

            N = Xw.shape[1]
            p = rng_tau.choice(N, size=min(particles_per_train_window, N), replace=False)
            k_idx = rng_tau.choice(Tseg, size=m_train, replace=False)
            k_idx.sort()

            v_s = sample_velocities_window(Xw, tw, tau, k_idx, particle_idx=p, Lx=Lx, Ly=Ly)  # (m_train,P,2)
            if v_s is None:
                continue
            Sig = signature_from_v(v_s, DIRS)  # (P, Dsig)
            obj_sigs.append(Sig)
            obj_meta.extend([(w_idx, int(pid)) for pid in p.tolist()])

            sp = np.sqrt((v_s * v_s).sum(axis=2)).astype(np.float32)  # (m_train,P)
            q10 = np.quantile(sp, 0.10, axis=0).astype(np.float32)
            q50 = np.quantile(sp, 0.50, axis=0).astype(np.float32)
            q90 = np.quantile(sp, 0.90, axis=0).astype(np.float32)
            q99 = np.quantile(sp, 0.99, axis=0).astype(np.float32)
            mu = np.mean(sp, axis=0).astype(np.float32)
            sd = np.std(sp, axis=0).astype(np.float32)
            tail = (q90 / np.maximum(q50, 1e-12)).astype(np.float32)

            # anisotropy of velocity covariance per particle
            # v_s: (m_train,P,2) -> for each p: cov(2x2) eigenvalues
            v0 = v_s.astype(np.float32)
            vmean = np.mean(v0, axis=0, keepdims=True)  # (1,P,2)
            vc = v0 - vmean
            # cov entries: (P,)
            c00 = np.mean(vc[:, :, 0] * vc[:, :, 0], axis=0)
            c11 = np.mean(vc[:, :, 1] * vc[:, :, 1], axis=0)
            c01 = np.mean(vc[:, :, 0] * vc[:, :, 1], axis=0)
            tr = c00 + c11
            det = c00 * c11 - c01 * c01
            disc = np.maximum(tr * tr - 4.0 * det, 0.0)
            sdisc = np.sqrt(disc)
            lam1 = 0.5 * (tr + sdisc)
            lam2 = 0.5 * (tr - sdisc)
            aniso = (lam1 / np.maximum(lam2, 1e-12)).astype(np.float32)

            obj_sp_q10.append(q10)
            obj_sp_q50.append(q50)
            obj_sp_q90.append(q90)
            obj_sp_q99.append(q99)
            obj_sp_mean.append(mu)
            obj_sp_std.append(sd)
            obj_tail.append(tail)
            obj_aniso.append(aniso)

        if not obj_sigs:
            continue

        Sig_all = np.vstack(obj_sigs).astype(np.float32)  # (M,D)
        M = int(Sig_all.shape[0])

        sp_q10 = np.concatenate(obj_sp_q10).astype(np.float32)
        sp_q50 = np.concatenate(obj_sp_q50).astype(np.float32)
        sp_q90 = np.concatenate(obj_sp_q90).astype(np.float32)
        sp_q99 = np.concatenate(obj_sp_q99).astype(np.float32)
        sp_mean = np.concatenate(obj_sp_mean).astype(np.float32)
        sp_std = np.concatenate(obj_sp_std).astype(np.float32)
        tail = np.concatenate(obj_tail).astype(np.float32)
        aniso = np.concatenate(obj_aniso).astype(np.float32)

        # distance matrix among training objects
        Dmat = pairwise_dist_matrix(Sig_all, chunk=16)

        # choose K by stability
        Kmax = min(K_cap, M // n_min_cluster) if M >= n_min_cluster else 1
        if Kmax < 1:
            Kmax = 1

        K_scores = []
        for K in range(1, Kmax + 1):
            Sk, ari_med, ari0_med = stability_score(Dmat, K, B_stab, subsample_frac, swap_trials, rng_tau)
            K_scores.append((K, Sk, ari_med, ari0_med))

        Ks = np.array([k for k, _, _, _ in K_scores], np.int64)
        Ss = np.array([s for _, s, _, _ in K_scores], np.float64)
        if Ss.size == 0:
            continue
        Smax = float(Ss.max())
        if Smax <= 1e-6:
            K_global = 1
        else:
            target = 0.95 * Smax
            K_global = int(Ks[np.where(Ss >= target)[0].min()])

        meds = kmedoids_build(Dmat, K_global, rng_tau)
        meds, best_cost = kmedoids_refine(Dmat, meds, swap_trials * 3, rng_tau)
        labels, d1, d2 = assign_to_medoids(Dmat, meds)

        # prototypes: medoid signatures and their stats
        med_sig = Sig_all[meds].copy()
        med_sp_q10 = sp_q10[meds].copy()
        med_sp_q50 = sp_q50[meds].copy()
        med_sp_q90 = sp_q90[meds].copy()
        med_sp_q99 = sp_q99[meds].copy()
        med_sp_mean = sp_mean[meds].copy()
        med_sp_std = sp_std[meds].copy()
        med_tail = tail[meds].copy()
        med_aniso = aniso[meds].copy()

        fast_k = int(np.argmax(med_sp_q90)) if K_global >= 1 else 0

        proto[int(tau)] = dict(
            tau=int(tau),
            K=int(K_global),
            medoids=meds.astype(np.int64),
            med_sig=med_sig.astype(np.float32),
            fast_k=int(fast_k),
            K_scores=np.asarray(K_scores, dtype=np.float64),
            Smax=float(Smax),
            train_windows=np.asarray(train_w, np.int64),
            Sig_dim=int(Sig_all.shape[1]),
            med_sp_q10=med_sp_q10.astype(np.float32),
            med_sp_q50=med_sp_q50.astype(np.float32),
            med_sp_q90=med_sp_q90.astype(np.float32),
            med_sp_q99=med_sp_q99.astype(np.float32),
            med_sp_mean=med_sp_mean.astype(np.float32),
            med_sp_std=med_sp_std.astype(np.float32),
            med_tail=med_tail.astype(np.float32),
            med_aniso=med_aniso.astype(np.float32),
        )

        tau_dir = os.path.join(out_dir, f"tau_{int(tau):04d}")
        os.makedirs(tau_dir, exist_ok=True)

        np.save(os.path.join(tau_dir, "med_sig.npy"), med_sig.astype(np.float32))
        np.save(os.path.join(tau_dir, "medoids.npy"), meds.astype(np.int64))
        np.save(os.path.join(tau_dir, "fast_k.npy"), np.asarray([fast_k], np.int64))

        np.save(os.path.join(tau_dir, "med_sp_q10.npy"), med_sp_q10.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_sp_q50.npy"), med_sp_q50.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_sp_q90.npy"), med_sp_q90.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_sp_q99.npy"), med_sp_q99.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_sp_mean.npy"), med_sp_mean.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_sp_std.npy"), med_sp_std.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_tail.npy"), med_tail.astype(np.float32))
        np.save(os.path.join(tau_dir, "med_aniso.npy"), med_aniso.astype(np.float32))

        np.save(os.path.join(tau_dir, "K_scores.npy"), np.asarray(K_scores, np.float64))
        np.save(os.path.join(tau_dir, "train_windows.npy"), np.asarray(train_w, np.int64))
        np.save(os.path.join(tau_dir, "Smax.npy"), np.asarray([Smax], np.float64))

        if save_full_train_labels:
            np.save(os.path.join(tau_dir, "train_labels.npy"), labels.astype(np.int64))
            np.save(os.path.join(tau_dir, "train_obj_meta.npy"), np.asarray(obj_meta, dtype=np.int64))
        if save_full_train_D:
            np.save(os.path.join(tau_dir, "D_train.npy"), Dmat.astype(np.float32))

        if compute_tau_separation_metrics:
            sep = tau_separation_metrics(Dmat, meds, labels)
            tau_sep_summary[int(tau)] = sep
            if save_small_tau_artifacts:
                np.save(os.path.join(tau_dir, "sep_sizes.npy"), sep["sizes"].astype(np.int64))
                np.save(os.path.join(tau_dir, "sep_radii_mean.npy"), sep["radii_mean"].astype(np.float32))
                np.save(os.path.join(tau_dir, "sep_radii_q90.npy"), sep["radii_q90"].astype(np.float32))
                np.save(os.path.join(tau_dir, "sep_Dm.npy"), sep["Dm"].astype(np.float32))
                np.save(os.path.join(tau_dir, "sep_scalars.npy"), np.asarray([
                    sep["M"], sep["K"], sep["intra_mean"], sep["intra_q90"], sep["inter_min"], sep["inter_mean"], sep["dunn"], sep["db"]
                ], np.float64))

    # ---------- PASS C: apply prototypes to all windows ----------
    print("PASS C: apply prototypes; compute pi, separability, entropy, TV, transitions ...")

    tau_used = sorted(proto.keys())
    if not tau_used:
        raise RuntimeError("No tau produced prototypes. Check validity / training selection.")

    tau_used = [int(t) for t in tau_used]
    Tuse = len(tau_used)
    Kmax_global = max(int(proto[t]["K"]) for t in tau_used)

    pi = np.full((W, Tuse, Kmax_global), np.nan, np.float32)
    sep_ratio = np.full((W, Tuse), np.nan, np.float32)
    pi_fast = np.full((W, Tuse), np.nan, np.float32)
    tv = np.full((W, Tuse), np.nan, np.float32)
    offdiag = np.full((W, Tuse), np.nan, np.float32)

    # NEW: mixture entropy/complexity
    H_pi = np.full((W, Tuse), np.nan, np.float32)
    H_norm = np.full((W, Tuse), np.nan, np.float32)
    K_eff = np.full((W, Tuse), np.nan, np.float32)

    prev_labels = {t: None for t in tau_used}
    wpos = {w: i for i, w in enumerate(w_indices)}
    rng_apply = np.random.default_rng(seed + 9999)

    for w_idx, ws, we, tw, Xw in tqdm(
        iter_windows(save_pth, t1, t2, window_size, window_step, log_format=log_format),
        desc="apply windows"
    ):
        if w_idx not in wpos:
            continue
        wi = wpos[w_idx]
        Tw = int(tw.size)
        if Tw == 0:
            continue
        N = int(Xw.shape[1])

        for ti, tau in enumerate(tau_used):
            scan_ti = tau_to_scan_ti.get(int(tau), None)
            if scan_ti is None:
                continue
            if not bool(Valid_mat[wi, scan_ti]):
                continue
            if tau >= Tw:
                continue
            Tseg = Tw - tau
            if Tseg < max(m_apply, m_min):
                continue

            P = proto[int(tau)]
            K = int(P["K"])
            med_sig = P["med_sig"]  # (K, Dsig)

            k_idx = rng_apply.choice(Tseg, size=m_apply, replace=False)
            k_idx.sort()

            v_s = sample_velocities_window(Xw, tw, tau, k_idx, particle_idx=None, Lx=Lx, Ly=Ly)  # (m_apply,N,2)
            if v_s is None:
                continue
            Sig_all = signature_from_v(v_s, DIRS)  # (N, Dsig)

            labels = np.empty(N, np.int64)
            d1_all = np.empty(N, np.float32)
            d2_all = np.empty(N, np.float32)

            for i0 in range(0, N, chunk_particles):
                i1 = min(N, i0 + chunk_particles)
                Sg = Sig_all[i0:i1].astype(np.float32)  # (c,D)
                dist = np.mean(np.abs(Sg[:, None, :] - med_sig[None, :, :]), axis=2).astype(np.float32)  # (c,K)
                ordk = np.argsort(dist, axis=1)
                labels[i0:i1] = ordk[:, 0]
                d1_all[i0:i1] = dist[np.arange(i1 - i0), ordk[:, 0]]
                if K > 1:
                    d2_all[i0:i1] = dist[np.arange(i1 - i0), ordk[:, 1]]
                else:
                    d2_all[i0:i1] = np.inf

            counts = np.bincount(labels, minlength=K).astype(np.float32)
            pik = counts / max(float(N), 1.0)
            pi[wi, ti, :K] = pik

            if K > 1:
                ratio = np.median((d2_all - d1_all) / np.maximum(d1_all, 1e-12))
                sep_ratio[wi, ti] = float(ratio)
            else:
                sep_ratio[wi, ti] = 0.0

            fk = int(P["fast_k"])
            pi_fast[wi, ti] = float(pik[fk])

            # NEW: entropy / normalized entropy / effective number of regimes
            H = entropy_from_pi(pik)
            H_pi[wi, ti] = float(H)
            if K > 1:
                H_norm[wi, ti] = float(H / max(math.log(K), 1e-12))
            else:
                H_norm[wi, ti] = 0.0
            K_eff[wi, ti] = float(math.exp(H))

            prev = prev_labels[int(tau)]
            if prev is None:
                offdiag[wi, ti] = 0.0
            else:
                Tmat = np.zeros((K, K), np.float32)
                for a, b in zip(prev, labels):
                    Tmat[a, b] += 1.0
                Tmat /= max(float(N), 1.0)
                offdiag[wi, ti] = float(1.0 - np.trace(Tmat))
            prev_labels[int(tau)] = labels

    # TV
    for ti, tau in enumerate(tau_used):
        K = int(proto[int(tau)]["K"])
        for wi in range(1, W):
            a = pi[wi - 1, ti, :K]
            b = pi[wi, ti, :K]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            tv[wi, ti] = 0.5 * float(np.abs(a - b).sum())

    # save core arrays
    np.save(os.path.join(out_dir, "tau_used.npy"), np.asarray(tau_used, np.int64))
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    np.save(os.path.join(out_dir, "pi_fast.npy"), pi_fast)
    np.save(os.path.join(out_dir, "sep_ratio.npy"), sep_ratio)
    np.save(os.path.join(out_dir, "tv.npy"), tv)
    np.save(os.path.join(out_dir, "offdiag.npy"), offdiag)

    # NEW: entropy outputs
    np.save(os.path.join(out_dir, "H_pi.npy"), H_pi)
    np.save(os.path.join(out_dir, "H_norm.npy"), H_norm)
    np.save(os.path.join(out_dir, "K_eff.npy"), K_eff)

    # per-tau summary arrays
    K_of_tau = np.asarray([int(proto[t]["K"]) for t in tau_used], np.int64)
    Smax_of_tau = np.asarray([float(proto[t]["Smax"]) for t in tau_used], np.float64)
    np.save(os.path.join(out_dir, "K_of_tau.npy"), K_of_tau)
    np.save(os.path.join(out_dir, "Smax_of_tau.npy"), Smax_of_tau)

    if compute_tau_separation_metrics and len(tau_sep_summary) > 0:
        # align with tau_used
        dunn = np.full(Tuse, np.nan, np.float32)
        db = np.full(Tuse, np.nan, np.float32)
        intra = np.full(Tuse, np.nan, np.float32)
        inter_min = np.full(Tuse, np.nan, np.float32)
        for i, t in enumerate(tau_used):
            if int(t) not in tau_sep_summary:
                continue
            s = tau_sep_summary[int(t)]
            dunn[i] = float(s["dunn"]) if np.isfinite(s["dunn"]) else np.nan
            db[i] = float(s["db"]) if np.isfinite(s["db"]) else np.nan
            intra[i] = float(s["intra_mean"]) if np.isfinite(s["intra_mean"]) else np.nan
            inter_min[i] = float(s["inter_min"]) if np.isfinite(s["inter_min"]) else np.nan

        np.save(os.path.join(out_dir, "sep_dunn_of_tau.npy"), dunn)
        np.save(os.path.join(out_dir, "sep_db_of_tau.npy"), db)
        np.save(os.path.join(out_dir, "sep_intra_of_tau.npy"), intra)
        np.save(os.path.join(out_dir, "sep_inter_min_of_tau.npy"), inter_min)

    # plots
    if make_plots:
        x_use = [str(t) for t in tau_used]
        y_lab = [str(w) for w in w_indices]

        save_heatmap(
            sep_ratio, x_use, y_lab,
            "Separability: median((d2-d1)/d1) (higher=more confident)",
            os.path.join(out_dir, "sep_ratio.png")
        )
        save_heatmap(
            pi_fast, x_use, y_lab,
            "pi_fast(w,tau): weight of fastest prototype",
            os.path.join(out_dir, "pi_fast.png"),
            vmin=0, vmax=1
        )
        vmax_tv = np.nanpercentile(tv[np.isfinite(tv)], 98) if np.isfinite(tv).any() else 1.0
        save_heatmap(
            tv, x_use, y_lab,
            "TV(w,tau) between consecutive windows",
            os.path.join(out_dir, "tv.png"),
            vmin=0, vmax=float(vmax_tv)
        )
        vmax_off = np.nanpercentile(offdiag[np.isfinite(offdiag)], 98) if np.isfinite(offdiag).any() else 1.0
        save_heatmap(
            offdiag, x_use, y_lab,
            "offdiag mass: 1 - trace(T_ab) (fraction switching)",
            os.path.join(out_dir, "offdiag.png"),
            vmin=0, vmax=float(vmax_off)
        )

        # NEW: entropy heatmaps
        save_heatmap(
            H_pi, x_use, y_lab,
            "H_pi(w,tau) = -sum_k pi_k log(pi_k) (entropy of regime mixture)",
            os.path.join(out_dir, "H_pi.png")
        )
        save_heatmap(
            H_norm, x_use, y_lab,
            "H_norm(w,tau) = H_pi / log(K(tau)) (normalized entropy in [0,1])",
            os.path.join(out_dir, "H_norm.png"),
            vmin=0, vmax=1
        )
        save_heatmap(
            K_eff, x_use, y_lab,
            "K_eff(w,tau) = exp(H_pi) (effective number of regimes)",
            os.path.join(out_dir, "K_eff.png")
        )

        # NEW: per-tau stability/separation summary plots (x-axis is tau in frames)
        save_line(
            tau_used, K_of_tau,
            "Chosen K(tau) from trained prototypes",
            "tau (frames)", "K(tau)",
            os.path.join(out_dir, "K_of_tau.png")
        )
        save_line(
            tau_used, Smax_of_tau,
            "Smax(tau) = max_K [median(ARI) - median(ARI_null)]",
            "tau (frames)", "Smax(tau)",
            os.path.join(out_dir, "Smax_of_tau.png")
        )
        if compute_tau_separation_metrics and len(tau_sep_summary) > 0:
            dunn = np.load(os.path.join(out_dir, "sep_dunn_of_tau.npy"))
            db = np.load(os.path.join(out_dir, "sep_db_of_tau.npy"))
            save_line(
                tau_used, dunn,
                "Cluster separability vs tau: Dunn-like = inter_min / intra_mean",
                "tau (frames)", "dunn(tau)",
                os.path.join(out_dir, "sep_dunn_of_tau.png")
            )
            save_line(
                tau_used, db,
                "Davies–Bouldin vs tau (lower is better separated)",
                "tau (frames)", "DB(tau)",
                os.path.join(out_dir, "sep_db_of_tau.png")
            )

    print("DONE. Outputs in:", out_dir)
    print("Trained taus:", tau_used)
    for tau in tau_used:
        P = proto[int(tau)]
        msg = f"tau={tau}: K={P['K']}, fast_k={P['fast_k']}, Smax={P['Smax']:.3f}, train_windows={len(P['train_windows'])}"
        if compute_tau_separation_metrics and int(tau) in tau_sep_summary:
            s = tau_sep_summary[int(tau)]
            msg += f", dunn={s['dunn']:.3f}, db={s['db']:.3f}"
        print(msg)


if __name__ == "__main__":
    main()
