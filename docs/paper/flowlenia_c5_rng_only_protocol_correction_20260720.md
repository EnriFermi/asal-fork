# Flow-Lenia C5 RNG-only protocol correction

Date: 2026-07-20

Status: the existing C5 result root is not valid for the final RNG-only C2/C5
protocol.

## Error

The final C2 decision defines future uncertainty using identical branch-point
states and independent continuation RNG only:

- `perturb_a_std = 0`;
- `perturb_p_std = 0`;
- `perturb_lagrangian_xy_std = 0`;
- each branch differs only by its folded `branch_seed`.

However, `scripts/flowlenia_c5_branch_frustration.py` froze and reused the
superseded external-noise C2 condition:

- `perturb_a_std = 0.02`;
- `perturb_p_std = 0.02`;
- `perturb_lagrangian_xy_std = 1.0`.

Therefore all simulations, metrics, figures, and videos under
`flow_lenia/c5_c2_paired_walls_half` describe the superseded external-state
intervention protocol and must not be used as final C5 evidence.

## Existing reusable work

The 450 optimized RNG-only free branches already exist under
`c2_noise_horizon_sweep/full/branches/noise_0`. They use the same selected
states and branch seeds as the old C2 plan. For the audited selected point
`run_003/point_00`, all three branch-point `A` and `P` arrays equal the source
trajectory bitwise.

## Selected-point diagnostic

A separate RNG-only diagnostic re-simulated both absorbing and
mass-projected wall arms for `run_003/point_00`. Its outputs are under:

`flow_lenia/c5_c2_paired_walls_half/selected_examples/rng_only_wall_probe_run_003_optimized_point_00`

For this one point, post-release within-arm CLIP Chamfer was:

- RNG-only free: `0.005758292430132608`;
- absorbing walls: `0.005758292430132608`;
- mass-projected walls: `0.005756373881405472`.

This single-point result does not establish the full C5 outcome. It does show
that the dramatic behavior in the earlier selected video depended on the
superseded dense state perturbation and cannot be used to motivate the final
C5 claim.

## Required full correction

1. Create a new versioned C5 root and protocol identity; do not overwrite the
   old root.
2. Set all external branch perturbations to zero and require bitwise
   branch-point equality against the source state.
3. Reuse the 450 optimized RNG-only free branches.
4. Generate RNG-only free branches for matched random candidates.
5. Recompute all wall arms, metrics, inference, figures, and videos.
6. Resolve and freeze wall mass semantics before the production calculation.
