import os
import sys
import subprocess
import numpy as np

import util

# --- config: adjust paths/substrate if needed ---
save_dir_0 = "data/supervised_0"   # checkpoint WITHOUT trajectory (best.pkl only)
save_dir_1 = "data/supervised_1"   # checkpoint WITH best_traj.pkl
substrate_name = "lenia"           # must match what you used in main_opt
out_root = "data/interp_supervised_0_1"
os.makedirs(out_root, exist_ok=True)

# --- load endpoints ---
params0, best_loss0 = util.load_pkl(save_dir_0, "best")        # best params from supervised_0
traj1 = util.load_pkl(save_dir_1, "best_traj")                 # trajectory from supervised_1
params1_start = traj1["params"][0]                             # first-iteration params

params0 = np.asarray(params0)
params1_start = np.asarray(params1_start)


def write_best_pkl(save_dir, params, fitness=0.0):
    """Write a best.pkl compatible with simulate_after_training."""
    os.makedirs(save_dir, exist_ok=True)
    util.save_pkl(save_dir, "best", (params, float(fitness)))


def run_sim_with_simulate_after_training(save_dir, output_path):
    """Call simulate_after_training.py for a given save_dir and output video path."""
    cmd = [
        sys.executable,
        "simulate_after_training.py",
        "--save_dir", save_dir,
        "--substrate", substrate_name,
        "--time_sampling", "video",
        "--output", output_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # 6 interpolations along the line:
    # 1st: params1_start, 6th: params0
    alphas = np.linspace(0.0, 1.0, 6)  # 0, 0.2, ..., 1.0

    for i, a in enumerate(alphas):
        params_interp = (1.0 - a) * params1_start + a * params0
        interp_name = f"interp_{i+1}"
        interp_save_dir = os.path.join(out_root, interp_name)
        output_path = os.path.join(out_root, f"{interp_name}.mp4")

        # Create a temporary best.pkl for this interpolated parameter set
        write_best_pkl(interp_save_dir, params_interp, fitness=0.0)

        # Run simulation via simulate_after_training.py
        run_sim_with_simulate_after_training(interp_save_dir, output_path)

        print(f"Saved simulation for {interp_name} to {output_path}")


if __name__ == "__main__":
    main()
