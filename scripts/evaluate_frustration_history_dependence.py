import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

import foundation_models
import substrates
import util


def _patch_wandb_pandas_check() -> None:
    try:
        import wandb.util as wandb_util
    except Exception:
        return
    orig = getattr(wandb_util, "is_pandas_data_frame", None)
    if orig is None:
        return

    def _safe_is_pandas_data_frame(val):
        try:
            return orig(val)
        except Exception:
            return False

    wandb_util.is_pandas_data_frame = _safe_is_pandas_data_frame


_patch_wandb_pandas_check()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_like: str | None, root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return root / path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("._-")
    return slug or "run"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/evaluate_frustration_history_dependence.py <config.yaml>")
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda k, default=None: os.getenv(k, default))
    cfg = OmegaConf.load(sys.argv[1])
    flat = OmegaConf.merge(
        cfg.get("meta", {}),
        cfg.get("source", {}),
        cfg.get("substrate", {}),
        cfg.get("protocol", {}),
        cfg.get("evaluation", {}),
        cfg.get("logging", {}),
    )
    return cfg, flat


def _load_params(args, project_root: Path) -> jax.Array:
    params_path = _resolve_path(getattr(args, "params_path", None), project_root)
    if params_path is not None:
        if not params_path.exists():
            raise FileNotFoundError(f"params_path does not exist: {params_path}")
        if params_path.suffix == ".npy":
            params_obj = np.load(params_path)
        elif params_path.suffix == ".npz":
            with np.load(params_path, allow_pickle=False) as data:
                if "params" not in data.files:
                    raise ValueError(f"{params_path} does not contain array 'params'.")
                params_obj = data["params"]
        elif params_path.suffix == ".pkl":
            params_obj = util.load_pkl(str(params_path.parent), params_path.stem)
        else:
            raise ValueError(f"Unsupported params_path suffix for {params_path}. Use .pkl, .npy, or .npz.")
    else:
        checkpoint_dir = _resolve_path(
            getattr(args, "checkpoint_dir", getattr(args, "source_save_dir", None)),
            project_root,
        )
        if checkpoint_dir is None:
            raise ValueError("Set source.checkpoint_dir or source.params_path.")
        params_name = str(getattr(args, "params_name", "best"))
        params_obj = util.load_pkl(str(checkpoint_dir), params_name)
        if params_obj is None:
            raise FileNotFoundError(f"{params_name}.pkl not found in {checkpoint_dir}")
    params = params_obj[0] if isinstance(params_obj, tuple) else params_obj
    return jnp.asarray(np.asarray(params, dtype=np.float32))


def _create_substrate(args):
    if args.substrate == "lenia_flow":
        base = substrates.create_substrate(
            args.substrate,
            **util.flow_lenia_kwargs_from_args(args),
        )
    else:
        base = substrates.create_substrate(args.substrate)
    return substrates.FlattenSubstrateParameters(base)


def _resolve_window(args) -> tuple[int, int]:
    total_steps = int(getattr(args, "total_steps"))
    warmup_steps = int(getattr(args, "warmup_steps"))
    end = getattr(args, "late_window_end_steps", None)
    end = total_steps if end is None else int(end)
    start = getattr(args, "late_window_start_steps", None)
    if start is None:
        size = getattr(args, "late_window_size_steps", None)
        if size is None:
            raise ValueError("Specify evaluation.late_window_start_steps or evaluation.late_window_size_steps.")
        start = end - int(size)
    else:
        start = int(start)
    if total_steps < 1:
        raise ValueError("protocol.total_steps must be >= 1.")
    if warmup_steps < 0:
        raise ValueError("protocol.warmup_steps must be >= 0.")
    if not (0 <= start < end <= total_steps):
        raise ValueError(
            f"Late window must satisfy 0 <= start < end <= total_steps. "
            f"Got start={start}, end={end}, total_steps={total_steps}."
        )
    if start < warmup_steps:
        raise ValueError(
            f"Late window must start after walls are removed. "
            f"Got late_window_start_steps={start}, warmup_steps={warmup_steps}."
        )
    return start, end


def _prepare_block_template_state(
    *,
    initial_state,
    block_template,
    split_n: int,
    block_size: int,
    pad: int,
    C: int,
    k: int,
):
    block_sim_size = block_size + 2 * pad
    A0 = np.asarray(initial_state["A"], dtype=np.float32)
    P0 = np.asarray(initial_state["P"], dtype=np.float32)
    Food0 = np.asarray(initial_state.get("Food", np.zeros(A0.shape[:2], dtype=np.float32)), dtype=np.float32)

    B = split_n * split_n
    A_blocks = np.zeros((B, block_sim_size, block_sim_size, C), dtype=np.float32)
    P_blocks = np.zeros((B, block_sim_size, block_sim_size, k), dtype=np.float32)
    Food_blocks = np.zeros((B, block_sim_size, block_sim_size), dtype=np.float32)

    for bi in range(B):
        i = bi // split_n
        j = bi % split_n
        i0 = i * block_size
        j0 = j * block_size
        A_blocks[bi, pad:pad + block_size, pad:pad + block_size] = A0[i0:i0 + block_size, j0:j0 + block_size]
        P_blocks[bi, pad:pad + block_size, pad:pad + block_size] = P0[i0:i0 + block_size, j0:j0 + block_size]
        Food_blocks[bi, pad:pad + block_size, pad:pad + block_size] = Food0[i0:i0 + block_size, j0:j0 + block_size]

    blocks_state = {}
    for key, value in block_template.items():
        if key == "A":
            blocks_state[key] = jnp.asarray(A_blocks)
        elif key == "P":
            blocks_state[key] = jnp.asarray(P_blocks)
        elif key == "Food":
            blocks_state[key] = jnp.asarray(Food_blocks)
        elif key == "mass_cycle_start":
            blocks_state[key] = jnp.sum(jnp.asarray(A_blocks), axis=(1, 2, 3))
        elif key == "t":
            t0 = initial_state.get("t", jnp.array(0, dtype=jnp.int32))
            blocks_state[key] = jnp.broadcast_to(t0, (B,))
        else:
            blocks_state[key] = jnp.broadcast_to(value, (B,) + value.shape)
    return blocks_state


def _assemble_blocks_jax(blocks_state, *, split_n: int, block_size: int, pad: int):
    A_blocks = blocks_state["A"]
    P_blocks = blocks_state["P"]
    Food_blocks = blocks_state.get("Food", None)

    if pad > 0:
        A_blocks = A_blocks[:, pad:pad + block_size, pad:pad + block_size, :]
        P_blocks = P_blocks[:, pad:pad + block_size, pad:pad + block_size, :]
        if Food_blocks is not None:
            Food_blocks = Food_blocks[:, pad:pad + block_size, pad:pad + block_size]

    C = int(A_blocks.shape[-1])
    K = int(P_blocks.shape[-1])
    H = block_size * split_n

    A_grid = A_blocks.reshape((split_n, split_n, block_size, block_size, C))
    A_grid = jnp.transpose(A_grid, (0, 2, 1, 3, 4))
    A_full = A_grid.reshape((H, H, C))

    P_grid = P_blocks.reshape((split_n, split_n, block_size, block_size, K))
    P_grid = jnp.transpose(P_grid, (0, 2, 1, 3, 4))
    P_full = P_grid.reshape((H, H, K))

    if Food_blocks is None:
        Food_full = jnp.zeros((H, H), dtype=A_full.dtype)
    else:
        F_grid = Food_blocks.reshape((split_n, split_n, block_size, block_size))
        F_grid = jnp.transpose(F_grid, (0, 2, 1, 3))
        Food_full = F_grid.reshape((H, H))

    return A_full, P_full, Food_full


def _merge_blocks_into_global_state(initial_state, blocks_state, *, split_n: int, block_size: int, pad: int):
    A_full, P_full, Food_full = _assemble_blocks_jax(
        blocks_state,
        split_n=split_n,
        block_size=block_size,
        pad=pad,
    )
    merged = dict(initial_state)
    merged["A"] = A_full
    merged["P"] = P_full
    merged["Food"] = Food_full
    if "t" in blocks_state:
        merged["t"] = jnp.asarray(blocks_state["t"][0])
    if "mass_cycle_start" in initial_state or "mass_cycle_start" in blocks_state:
        merged["mass_cycle_start"] = jnp.sum(A_full)
    return merged


def _build_state_advancer(substrate, steps: int):
    if steps <= 0:
        return lambda rng_key, state0, params_in: state0

    def advance_fn(rng_key, state0, params_in):
        def body_fn(state, step_key):
            return substrate.step_state(step_key, state, params_in), None

        state_final, _ = jax.lax.scan(body_fn, state0, jax.random.split(rng_key, steps))
        return state_final

    return jax.jit(advance_fn)


def _build_block_warmupper(block_substrate, n_blocks: int, steps: int):
    if steps <= 0:
        return lambda rng_key, state0, params_in: state0

    def block_step(state, key, params_in):
        return block_substrate.step_state(key, state, params_in)

    vmapped_step = jax.vmap(block_step, in_axes=(0, 0, None))

    def warmup_fn(rng_key, state0, params_in):
        def body_fn(state, step_key):
            block_keys = jax.random.split(step_key, n_blocks)
            return vmapped_step(state, block_keys, params_in), None

        state_final, _ = jax.lax.scan(body_fn, state0, jax.random.split(rng_key, steps))
        return state_final

    return jax.jit(warmup_fn)


def _build_embedding_rollout(substrate, fm, *, window_steps: int, time_sampling: int, img_size: int):
    if window_steps <= 0:
        raise ValueError("Late window must have at least one step.")
    if time_sampling < 1:
        raise ValueError("evaluation.time_sampling must be >= 1.")
    if window_steps % time_sampling != 0:
        raise ValueError(
            f"Late window length ({window_steps}) must be divisible by time_sampling ({time_sampling})."
        )
    chunk_steps = window_steps // time_sampling

    def rollout_fn(rng_key, state0, params_in):
        def step_fn(state, step_key):
            return substrate.step_state(step_key, state, params_in), None

        def chunk_fn(state, chunk_key):
            state_next, _ = jax.lax.scan(step_fn, state, jax.random.split(chunk_key, chunk_steps))
            img = substrate.render_state(state_next, params=params_in, img_size=img_size)
            z = fm.embed_img(img)
            return state_next, z

        _, z_seq = jax.lax.scan(chunk_fn, state0, jax.random.split(rng_key, time_sampling))
        return z_seq

    return jax.jit(rollout_fn)


def _sequence_distance(z_a: np.ndarray, z_b: np.ndarray, metric: str) -> tuple[float, np.ndarray]:
    a = np.asarray(z_a, dtype=np.float32)
    b = np.asarray(z_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Embedding sequence shapes must match, got {a.shape} vs {b.shape}.")

    metric = str(metric).strip().lower()
    if metric == "cosine_mean":
        per_t = 1.0 - np.sum(a * b, axis=-1)
    elif metric == "euclidean_mean":
        per_t = np.linalg.norm(a - b, axis=-1)
    elif metric == "sqeuclidean_mean":
        diff = a - b
        per_t = np.sum(diff * diff, axis=-1)
    elif metric == "cosine_last":
        per_t = np.array([1.0 - float(np.sum(a[-1] * b[-1]))], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported evaluation.distance_metric={metric!r}. "
            "Use one of ['cosine_mean', 'euclidean_mean', 'sqeuclidean_mean', 'cosine_last']."
        )
    return float(np.mean(per_t)), np.asarray(per_t, dtype=np.float32)


def _summarize_trials(rows: list[dict]) -> dict:
    if not rows:
        return {}

    baseline = np.asarray([float(r["baseline_distance"]) for r in rows], dtype=np.float64)
    effect = np.asarray([float(r["walls_effect_distance"]) for r in rows], dtype=np.float64)
    diff = effect - baseline
    ratio = effect / np.maximum(baseline, 1e-12)
    n_trials = int(len(rows))

    wilcoxon_greater = None
    wilcoxon_two_sided = None
    if n_trials > 0 and np.any(np.abs(diff) > 0):
        try:
            wilcoxon_greater = float(scipy_stats.wilcoxon(effect, baseline, alternative="greater").pvalue)
        except Exception:
            wilcoxon_greater = None
        try:
            wilcoxon_two_sided = float(scipy_stats.wilcoxon(effect, baseline, alternative="two-sided").pvalue)
        except Exception:
            wilcoxon_two_sided = None

    gt_count = int(np.sum(diff > 0))
    ge_count = int(np.sum(diff >= 0))
    sign_test_greater = None
    if n_trials > 0:
        try:
            sign_test_greater = float(scipy_stats.binomtest(gt_count, n_trials, 0.5, alternative="greater").pvalue)
        except Exception:
            sign_test_greater = None

    return {
        "n_trials": n_trials,
        "mean_baseline_distance": float(np.mean(baseline)),
        "std_baseline_distance": float(np.std(baseline, ddof=1) if n_trials > 1 else 0.0),
        "mean_walls_effect_distance": float(np.mean(effect)),
        "std_walls_effect_distance": float(np.std(effect, ddof=1) if n_trials > 1 else 0.0),
        "mean_effect_minus_baseline": float(np.mean(diff)),
        "median_effect_minus_baseline": float(np.median(diff)),
        "std_effect_minus_baseline": float(np.std(diff, ddof=1) if n_trials > 1 else 0.0),
        "mean_effect_over_baseline_ratio": float(np.mean(ratio)),
        "median_effect_over_baseline_ratio": float(np.median(ratio)),
        "fraction_effect_gt_baseline": float(np.mean(diff > 0)),
        "fraction_effect_ge_baseline": float(np.mean(diff >= 0)),
        "gt_count": gt_count,
        "ge_count": ge_count,
        "wilcoxon_greater_pvalue": wilcoxon_greater,
        "wilcoxon_two_sided_pvalue": wilcoxon_two_sided,
        "sign_test_greater_pvalue": sign_test_greater,
    }


def main(cfg, args):
    project_root = _repo_root()
    save_dir = _resolve_path(getattr(args, "save_dir", None), project_root)
    if save_dir is None:
        raise ValueError("meta.save_dir must be set.")
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_text(save_dir / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
    resolved_checkpoint_dir = _resolve_path(
        getattr(args, "checkpoint_dir", getattr(args, "source_save_dir", None)),
        project_root,
    )
    resolved_params_path = _resolve_path(getattr(args, "params_path", None), project_root)

    run = wandb.init(
        project=str(getattr(args, "wandb_project", "asal")),
        mode=str(getattr(args, "wandb_mode", "online")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        params = _load_params(args, project_root)
        substrate = _create_substrate(args)
        expected_len = int(np.asarray(substrate.default_params(jax.random.PRNGKey(0))).size)
        if int(params.shape[-1]) != expected_len:
            raise ValueError(
                f"Loaded parameter length {int(params.shape[-1])} does not match substrate expectation {expected_len}."
            )

        if str(args.substrate) != "lenia_flow":
            raise ValueError("This evaluator currently supports substrate='lenia_flow' only.")

        total_steps = int(getattr(args, "total_steps"))
        warmup_steps = int(getattr(args, "warmup_steps"))
        late_start, late_end = _resolve_window(args)
        late_window_steps = int(late_end - late_start)
        time_sampling = int(getattr(args, "time_sampling"))
        n_initial_states = int(getattr(args, "n_initial_states"))
        if n_initial_states < 1:
            raise ValueError("evaluation.n_initial_states must be >= 1.")

        grid_size = int(getattr(args, "grid_size"))
        split_n = int(getattr(args, "grid_split"))
        if split_n < 1:
            raise ValueError("protocol.grid_split must be >= 1.")
        if grid_size % split_n != 0:
            raise ValueError(f"grid_size {grid_size} must be divisible by grid_split {split_n}.")
        block_size = grid_size // split_n
        pad = int(getattr(args, "wall_pad", int(args.dd)))
        block_sim_size = block_size + 2 * pad
        n_blocks = split_n * split_n

        block_kwargs = util.flow_lenia_kwargs_from_args(args)
        block_kwargs["grid_size"] = block_sim_size
        block_substrate = substrates.FlattenSubstrateParameters(
            substrates.create_substrate("lenia_flow", **block_kwargs)
        )

        fm_name = str(getattr(args, "foundation_model", "clip"))
        fm = foundation_models.create_foundation_model(fm_name)
        clip_img_size = int(getattr(args, "clip_img_size", 224))
        distance_metric = str(getattr(args, "distance_metric", "cosine_mean"))
        resume = bool(getattr(args, "resume", True))
        save_embeddings = bool(getattr(args, "save_embeddings", True))

        block_template = block_substrate.init_state(jax.random.PRNGKey(0), params)

        control_prefix_advancer = _build_state_advancer(substrate, late_start)
        walls_warmupper = _build_block_warmupper(block_substrate, n_blocks, warmup_steps)
        walls_post_advancer = _build_state_advancer(substrate, late_start - warmup_steps)
        embed_rollout = _build_embedding_rollout(
            substrate,
            fm,
            window_steps=late_window_steps,
            time_sampling=time_sampling,
            img_size=clip_img_size,
        )

        trial_dir = save_dir / "trial_data"
        trial_dir.mkdir(parents=True, exist_ok=True)

        master_key = jax.random.PRNGKey(int(getattr(args, "seed", 0)))
        rows: list[dict] = []

        pbar = tqdm(range(n_initial_states), desc="history_dependence", leave=True)
        for trial_idx in pbar:
            trial_key = jax.random.fold_in(master_key, int(trial_idx))
            (
                k_init,
                k_ctrl_a_prefix,
                k_ctrl_a_window,
                k_ctrl_b_prefix,
                k_ctrl_b_window,
                k_walls_warm,
                k_walls_post,
                k_walls_window,
            ) = jax.random.split(trial_key, 8)

            trial_json = trial_dir / f"trial_{trial_idx:05d}.json"
            trial_npz = trial_dir / f"trial_{trial_idx:05d}_embeddings.npz"
            trial_complete = trial_json.exists() and (not save_embeddings or trial_npz.exists())
            if resume and trial_complete:
                with trial_json.open("r") as f:
                    row = json.load(f)
                rows.append(row)
                continue

            initial_state = substrate.init_state(k_init, params)

            control_a_start = control_prefix_advancer(k_ctrl_a_prefix, initial_state, params)
            z_control_a = np.asarray(jax.device_get(embed_rollout(k_ctrl_a_window, control_a_start, params)), dtype=np.float32)

            control_b_start = control_prefix_advancer(k_ctrl_b_prefix, initial_state, params)
            z_control_b = np.asarray(jax.device_get(embed_rollout(k_ctrl_b_window, control_b_start, params)), dtype=np.float32)

            block_state0 = _prepare_block_template_state(
                initial_state=initial_state,
                block_template=block_template,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
                C=int(getattr(args, "C")),
                k=int(getattr(args, "k")),
            )
            block_state_warm = walls_warmupper(k_walls_warm, block_state0, params)
            merged_state = _merge_blocks_into_global_state(
                initial_state,
                block_state_warm,
                split_n=split_n,
                block_size=block_size,
                pad=pad,
            )
            walls_start = walls_post_advancer(k_walls_post, merged_state, params)
            z_walls = np.asarray(jax.device_get(embed_rollout(k_walls_window, walls_start, params)), dtype=np.float32)

            baseline_distance, baseline_per_t = _sequence_distance(z_control_a, z_control_b, distance_metric)
            walls_a_distance, walls_a_per_t = _sequence_distance(z_control_a, z_walls, distance_metric)
            walls_b_distance, walls_b_per_t = _sequence_distance(z_control_b, z_walls, distance_metric)
            walls_effect_distance = float(0.5 * (walls_a_distance + walls_b_distance))

            row = {
                "trial_idx": int(trial_idx),
                "baseline_distance": float(baseline_distance),
                "walls_effect_distance": float(walls_effect_distance),
                "walls_effect_distance_ctrl_a": float(walls_a_distance),
                "walls_effect_distance_ctrl_b": float(walls_b_distance),
                "effect_minus_baseline": float(walls_effect_distance - baseline_distance),
                "effect_over_baseline_ratio": float(walls_effect_distance / max(baseline_distance, 1e-12)),
                "embeddings_path": None if not save_embeddings else str(trial_npz),
                "late_window_start_steps": int(late_start),
                "late_window_end_steps": int(late_end),
                "late_window_steps": int(late_window_steps),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(total_steps),
                "time_sampling": int(time_sampling),
                "distance_metric": distance_metric,
                "foundation_model": fm_name,
            }
            _write_json(trial_json, row)
            if save_embeddings:
                _save_npz_atomic(
                    trial_npz,
                    z_control_a=z_control_a,
                    z_control_b=z_control_b,
                    z_walls=z_walls,
                    baseline_per_t=baseline_per_t,
                    walls_ctrl_a_per_t=walls_a_per_t,
                    walls_ctrl_b_per_t=walls_b_per_t,
                )
            rows.append(row)

            pbar.set_postfix(
                baseline=f"{baseline_distance:.4f}",
                effect=f"{walls_effect_distance:.4f}",
                delta=f"{(walls_effect_distance - baseline_distance):.4f}",
            )

        rows = sorted(rows, key=lambda r: int(r["trial_idx"]))
        summary = _summarize_trials(rows)
        summary.update(
            {
                "save_dir": str(save_dir),
                "checkpoint_dir": None if resolved_checkpoint_dir is None else str(resolved_checkpoint_dir),
                "params_path": None if resolved_params_path is None else str(resolved_params_path),
                "params_name": str(getattr(args, "params_name", "best")),
                "foundation_model": fm_name,
                "distance_metric": distance_metric,
                "grid_split": int(split_n),
                "wall_pad": int(pad),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(total_steps),
                "late_window_start_steps": int(late_start),
                "late_window_end_steps": int(late_end),
                "late_window_steps": int(late_window_steps),
                "time_sampling": int(time_sampling),
                "resume_enabled": bool(resume),
                "save_embeddings": bool(save_embeddings),
            }
        )

        _save_csv(save_dir / "trial_results.csv", rows)
        _write_json(save_dir / "summary.json", summary)

        if rows:
            table = wandb.Table(
                columns=list(rows[0].keys()),
                data=[[row[k] for k in rows[0].keys()] for row in rows],
            )
            run.log({"history_dependence/trials": table})
        for key, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                run.summary[f"history_dependence/{key}"] = value

        print(f"Completed trials: {summary.get('n_trials', 0)}")
        print(f"Mean baseline distance: {summary.get('mean_baseline_distance', float('nan')):.6f}")
        print(f"Mean walls-effect distance: {summary.get('mean_walls_effect_distance', float('nan')):.6f}")
        print(f"Mean effect-baseline delta: {summary.get('mean_effect_minus_baseline', float('nan')):.6f}")
        print(f"Fraction effect > baseline: {summary.get('fraction_effect_gt_baseline', float('nan')):.6f}")
        pval = summary.get("wilcoxon_greater_pvalue", None)
        if pval is not None:
            print(f"Wilcoxon p(effect > baseline): {pval:.6g}")
    finally:
        run.finish()


if __name__ == "__main__":
    cfg, flat = load_config()
    main(cfg, flat)
