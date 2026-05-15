python3 - << 'PY'
import numpy as np
import jax
from jax.random import PRNGKey
from substrates.gol import GameOfLife
from rollout import rollout_simulation
import imageio.v3 as iio


def rule_to_params(rule: str) -> int:
    """
    Convert a Life-like rule string (e.g. 'B3/S23', 'B012345/S0235')
    into the integer params used by substrates.gol.GameOfLife.
    Bits 0–8  : dead cell -> alive (B)
    Bits 9–17 : live cell stays alive (S)
    """
    rule = rule.strip().upper()
    if '/' not in rule or not rule.startswith('B'):
        raise ValueError(f"Invalid rule format: {rule}")

    b_part, s_part = rule.split('/')
    if not s_part.startswith('S'):
        raise ValueError(f"Invalid rule format: {rule}")

    B = [int(ch) for ch in b_part[1:] if ch.isdigit()]
    S = [int(ch) for ch in s_part[1:] if ch.isdigit()]

    # sanity check
    if any(n < 0 or n > 8 for n in B + S):
        raise ValueError(f"Neighbor counts must be between 0 and 8: {B}, {S}")

    params = sum(1 << n for n in B) + sum(1 << (9 + n) for n in S)
    return params


# Life-like rule B012345/S0235 encoded for this repo's GameOfLife:
# params = sum(2**n for n in B) + sum(2**(9+n) for n in S)
params = rule_to_params("B0145/S01234") 


sim = GameOfLife(grid_size=512)
rng = PRNGKey(0)

data = rollout_simulation(
    rng,
    params,
    s0=None,
    substrate=sim,
    fm=None,              # no CLIP needed
    rollout_steps=1000000,
    time_sampling='video',
    img_size=224,
    return_state=False,
)

frames = np.asarray(data["rgb"])
frames_u8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
iio.imwrite("gol_B0145_S01234.mp4", frames_u8, fps=200)
PY
