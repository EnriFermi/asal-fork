from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

def _activate_source_root(source_root: Path | None) -> None:
    if source_root is None:
        return
    source_root = source_root.resolve()
    source_scripts = source_root / "scripts"
    this_scripts = _REPO_ROOT / "scripts"
    remove = {str(_REPO_ROOT.resolve()), str(this_scripts.resolve())}
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) not in remove]
    for path in (str(source_scripts), str(source_root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _flat_config(path: Path) -> SimpleNamespace:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("substrate", {}),
        cfg.get("evaluation", {}),
        cfg.get("optimization", {}),
        cfg.get("logging", {}),
        cfg.get("metric", {}),
    )
    return SimpleNamespace(**OmegaConf.to_container(flat, resolve=True))


def _resolve_run_and_best(path_like: str) -> tuple[Path, Path]:
    path = Path(path_like)
    if path.name == "best.pkl":
        return path.parent, path
    return path, path / "best.pkl"


def _build_substrate(flat: SimpleNamespace):
    import substrates
    import util

    base_substrate = substrates.create_substrate(
        str(flat.substrate),
        **util.substrate_kwargs_from_args(flat),
    )
    if hasattr(base_substrate, "render_mode") and getattr(flat, "render_mode", None) is not None:
        base_substrate.render_mode = str(flat.render_mode)
    return substrates.FlattenSubstrateParameters(base_substrate)


def _coerce_best_params(best_obj: Any) -> tuple[np.ndarray, float | None]:
    import numpy as np

    if isinstance(best_obj, tuple) and len(best_obj) >= 1:
        params = best_obj[0]
        fitness = float(np.asarray(best_obj[1]).reshape(-1)[0]) if len(best_obj) >= 2 else None
        return np.asarray(params, dtype=np.float32), fitness
    if isinstance(best_obj, dict) and "params" in best_obj:
        fitness = best_obj.get("loss", best_obj.get("fitness", None))
        fitness_f = float(np.asarray(fitness).reshape(-1)[0]) if fitness is not None else None
        return np.asarray(best_obj["params"], dtype=np.float32), fitness_f
    return np.asarray(best_obj, dtype=np.float32), None


def _render_frame(substrate: Any, state: Any, params: jax.Array, img_size: int, resize_method: str):
    import jax
    import jax.numpy as jnp

    frame = substrate.render_state(state, params, img_size=None)
    if img_size > 0 and int(frame.shape[0]) != int(img_size):
        frame = jax.image.resize(frame, (int(img_size), int(img_size), 3), method=str(resize_method))
    return jnp.clip(frame, 0.0, 1.0)


def _build_frame_batcher(
    *,
    substrate: Any,
    params: jax.Array,
    stride_steps: int,
    img_size: int,
    resize_method: str,
):
    import jax
    import jax.numpy as jnp

    def one_frame(state, keys_for_frame):
        def one_step(st, key_step):
            return substrate.step_state(key_step, st, params), None

        state_next, _ = jax.lax.scan(one_step, state, keys_for_frame)
        frame = _render_frame(substrate, state_next, params, img_size, resize_method)
        mass = jnp.sum(state_next["A"], axis=(0, 1))
        p_max = (
            jnp.mean(jnp.max(state_next["P"], axis=-1))
            if isinstance(state_next, dict) and "P" in state_next
            else jnp.asarray(jnp.nan, dtype=jnp.float32)
        )
        return state_next, (frame, mass, p_max)

    def run_batch(state, keys_by_frame):
        return jax.lax.scan(one_frame, state, keys_by_frame)

    return jax.jit(run_batch)


def _write_mass_outputs(path: Path, rows: list[dict[str, float]]) -> tuple[str | None, str | None]:
    import matplotlib.pyplot as plt

    if not rows:
        return None, None
    csv_path = path.with_suffix(".mass.csv")
    png_path = path.with_suffix(".mass.png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    steps = [r["step"] for r in rows]
    channel_keys = [k for k in keys if k.startswith("mass_ch")]
    plt.figure(figsize=(8, 4))
    for key in channel_keys:
        plt.plot(steps, [r[key] for r in rows], label=key)
    plt.plot(steps, [r["mass_total"] for r in rows], color="black", linewidth=2, alpha=0.7, label="total")
    plt.xlabel("step")
    plt.ylabel("mass")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return str(csv_path), str(png_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import imageio
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tqdm import tqdm

    _activate_source_root(Path(args.source_root) if args.source_root else None)

    run_dir, best_path = _resolve_run_and_best(args.run_or_best)
    config_path = Path(args.config) if args.config else run_dir / "optimization_config.yaml"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pkl not found: {best_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"optimization config not found: {config_path}")

    flat = _flat_config(config_path)
    if args.grid_size is not None:
        flat.grid_size = int(args.grid_size)
    if args.render_mode is not None:
        flat.render_mode = str(args.render_mode)

    substrate = _build_substrate(flat)
    params_np, best_fitness = _coerce_best_params(_load_pickle(best_path))
    expected = int(substrate.n_params)
    if int(params_np.size) == expected + 1:
        print(f"Loaded {params_np.size} params; slicing first {expected} substrate params and ignoring trailing tau latent.")
        params_np = params_np[:expected]
    if int(params_np.size) != expected:
        raise ValueError(
            f"best parameter length {params_np.size} does not match substrate expectation {expected}. "
            f"Check config/grid_size/border/kernel settings."
        )

    max_steps = int(args.max_steps if args.max_steps is not None else getattr(flat, "rollout_steps"))
    stride_steps = int(args.video_stride_steps)
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
    if stride_steps < 1:
        raise ValueError(f"video_stride_steps must be >= 1, got {stride_steps}.")
    n_frames = max_steps // stride_steps
    if n_frames < 1:
        raise ValueError(f"max_steps={max_steps} gives zero frames with video_stride_steps={stride_steps}.")

    output = Path(args.output) if args.output else run_dir / "videos" / f"best_grid{int(flat.grid_size)}_{max_steps}_stride{stride_steps}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    params = jnp.asarray(params_np, dtype=jnp.float32)
    rng = jax.random.PRNGKey(int(args.seed))
    key_init, key_scan = jax.random.split(rng)
    state = substrate.init_state(key_init, params)
    frame_batcher = _build_frame_batcher(
        substrate=substrate,
        params=params,
        stride_steps=stride_steps,
        img_size=int(args.img_size),
        resize_method=str(args.resize_method),
    )

    rows: list[dict[str, float]] = []
    frames_written = 0
    with imageio.get_writer(str(output), fps=int(args.fps), codec=str(args.codec), macro_block_size=args.macro_block_size) as writer:
        for start in tqdm(range(0, n_frames, int(args.frame_batch_size)), desc="render video"):
            batch_n = min(int(args.frame_batch_size), n_frames - start)
            key_scan, key_batch = jax.random.split(key_scan)
            keys = jax.random.split(key_batch, batch_n * stride_steps).reshape((batch_n, stride_steps, 2))
            state, (frames, masses, p_max) = frame_batcher(state, keys)
            frames_np = np.asarray(jax.device_get(frames))
            masses_np = np.asarray(jax.device_get(masses))
            p_max_np = np.asarray(jax.device_get(p_max))
            frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
            for i in range(batch_n):
                writer.append_data(frames_u8[i])
                step = int((start + i + 1) * stride_steps)
                mass_row = {f"mass_ch{c}": float(masses_np[i, c]) for c in range(masses_np.shape[1])}
                mass_row.update(
                    step=float(step),
                    mass_total=float(np.sum(masses_np[i])),
                    p_max_mean=float(p_max_np[i]),
                )
                rows.append(mass_row)
                frames_written += 1

    mass_csv, mass_png = _write_mass_outputs(output, rows)
    summary = {
        "run_dir": str(run_dir),
        "best_path": str(best_path),
        "config_path": str(config_path),
        "output": str(output),
        "best_fitness": best_fitness,
        "substrate": str(flat.substrate),
        "grid_size": int(flat.grid_size),
        "border": str(getattr(flat, "border", "")),
        "render_mode": str(getattr(flat, "render_mode", "")),
        "seed": int(args.seed),
        "max_steps": int(max_steps),
        "video_stride_steps": int(stride_steps),
        "frames_written": int(frames_written),
        "fps": int(args.fps),
        "img_size": int(args.img_size),
        "resize_method": str(args.resize_method),
        "mass_csv": mass_csv,
        "mass_plot": mass_png,
    }
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Flow-Lenia best.pkl checkpoint to an mp4 video.")
    parser.add_argument(
        "run_or_best",
        help="Optimization run directory or direct path to best.pkl.",
    )
    parser.add_argument("--config", default=None, help="Override optimization_config.yaml path.")
    parser.add_argument("--source-root", default=None, help="Optional source checkout to import code from.")
    parser.add_argument("--output", default=None, help="Output mp4 path. Defaults to <run_dir>/videos/...")
    parser.add_argument("--seed", type=int, default=0, help="Simulation seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Simulation horizon. Defaults to config rollout_steps.")
    parser.add_argument("--video-stride-steps", type=int, default=500, help="Simulation steps between rendered frames.")
    parser.add_argument("--frame-batch-size", type=int, default=16, help="Rendered frames per JIT batch.")
    parser.add_argument("--img-size", type=int, default=384, help="Rendered square video size.")
    parser.add_argument("--fps", type=int, default=60, help="Video fps.")
    parser.add_argument("--codec", default="libx264", help="ImageIO/ffmpeg codec.")
    parser.add_argument("--macro-block-size", type=int, default=None, help="ImageIO macro_block_size.")
    parser.add_argument(
        "--resize-method",
        default="linear",
        choices=["nearest", "linear", "cubic", "lanczos3", "lanczos5"],
        help="Resize method from native grid to img-size.",
    )
    parser.add_argument("--grid-size", type=int, default=None, help="Optional grid_size override.")
    parser.add_argument("--render-mode", default=None, help="Optional render_mode override, e.g. Pcolor.")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
