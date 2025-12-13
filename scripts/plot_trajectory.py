import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection

import util  # from this repo

# --- Load trajectory ---
save_dir = "data/supervised_pca_track"  # adjust to your run dir
traj = util.load_pkl(save_dir, "best_traj")

X = np.asarray(traj["params"])  # shape (T, D), best params per iteration
T, D = X.shape
steps = np.arange(T)

# --- 2D PCA on parameters ---
X_centered = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
pcs = X_centered @ Vt[:2].T  # shape (T, 2); columns = PC1, PC2

# --- Colors from start to end of trajectory ---
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.0, 1.0, T))

# --- 3D plot: x,y = PCA, z = step index ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Scatter with color changing along trajectory
sc = ax.scatter(pcs[:, 0], pcs[:, 1], steps, c=steps, cmap="viridis", s=10)

# Optional: thin line connecting points
ax.plot(pcs[:, 0], pcs[:, 1], steps, color="k", linewidth=0.5, alpha=0.4)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Iteration (step)")
ax.set_title("Best-parameter trajectory in 2D PCA space")

cb = fig.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("Iteration (start → end)")

plt.tight_layout()
plt.show()
plt.savefig('pca.png')
