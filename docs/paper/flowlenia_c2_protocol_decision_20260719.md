# Flow-Lenia C2 Protocol Decision

Date: 2026-07-19

Status: agreed scientific interpretation; final paper-ready rescoring is still
pending.

## Claim Being Tested

C2 is a claim about conditional uncertainty of the future:

```text
States with higher Delta-H have a broader distribution of possible futures.
```

For a fixed saved state `X_t`, independent continuations are sampled from the
intrinsic stochastic transition process:

```text
future_b ~ P(future | X_t),  b = 1, ..., R
U_T(X_t) = median_{a < b} distance(future_a, future_b)
```

The primary test is whether `Delta-H(t)` is positively associated with
`U_T(X_t)`.

External state perturbation is not part of this claim.

## Correct Primary Condition

The `noise_scale = 0` condition is the correct primary condition, but it must
be described as `intrinsic future uncertainty` or `RNG-only branching`, not as
generic "zero noise".

For the three branches of one selected state:

- the source checkpoint, parameters, simulation configuration, `A`, `P`, and
  Lagrangian coordinates are identical at the branch point;
- no Gaussian state perturbation is applied;
- each branch folds a different `branch_seed` into the saved continuation RNG;
- Flow-Lenia's enabled stochastic mutation process therefore samples distinct
  possible futures.

The CLIP divergence is computed from rendered `A/P`, so randomness in the
Lagrangian tracker does not itself create the measured visual divergence.

A spot audit of `opt_000 / low / pair_0` found exact `A` and `P` equality
across all three branches at step 215000, followed by divergence by step
215700. A final analysis must extend this exact branch-point equality audit to
all 150 states and all 450 RNG-only branches.

## Cached Preliminary Result

The existing trajectory-level CLIP-Chamfer score includes the identical
post-branch frame at relative step zero. This frame should be excluded because
it is not part of the uncertain future.

Cached post-hoc rescoring at the 20k horizon, without running new simulations,
gave:

- pooled Pearson `r = +0.252`;
- median within-run Spearman `rho = +0.266`;
- positive within-run Spearman sign in 9 of 10 optimization runs.

These values are preliminary until the calculation is moved into a
reproducible script and accompanied by run-cluster bootstrap intervals,
matched-pair inference, and multiplicity handling.

## Interpretation Of The External-Noise Sweep

The external-noise sweep tests a different question:

```text
How robust is X_t to an exogenous state intervention?
```

It must be retained as a supplementary intervention-robustness analysis and
must not be used as the primary test of C2 future uncertainty.

The current absolute perturbation simultaneously applies:

```text
A += Normal(0, 0.02 * scale), followed by clipping A >= 0
P += Normal(0, 0.02 * scale)
lagrangian_xy += Normal(0, 1.0 * scale)
```

It also changes the future rollout RNG through the same `branch_seed`.
Consequently, nonzero-scale branches mix state intervention and intrinsic RNG
variation.

The negative association at nonzero scales is not a simulation or parity bug.
It begins partly at the intervention frame and is strongly affected by state
occupancy:

- median active-mass area is about 18.2% for selected high-Delta-H states and
  9.0% for selected low-Delta-H states;
- median visibly active area is about 14.7% versus 7.3%;
- total mean mass is nearly equal.

Absolute additive noise therefore changes sparse low-Delta-H states more
strongly in relative visual terms, including by creating positive mass in
previously empty background after clipping. This intervention answers a
robustness question, not the C2 uncertainty question.

Removing only the intervention frame does not repair that supplementary
condition. At `scale = 1`, horizon 20k, pooled Pearson changes from `-0.378`
to only `-0.361` after dropping relative step zero, and remains `-0.334` when
using frames from 5.7k onward.

## Final Paper-Ready Work

Do not rerun the already completed RNG-only branch simulations.

1. Build a dedicated cached rescoring command for `noise_scale = 0`.
2. Exclude relative step zero from every horizon's future embedding cloud.
3. Audit exact branch-point equality for all 150 states and 450 branches.
4. Recompute pooled Pearson/Spearman, per-run correlations, matched high-low
   contrasts, run-cluster bootstrap intervals, sign tests, and FDR results.
5. Produce the primary C2 table, pooled and per-run plots, matched-pair plot,
   Delta-H selection maps, and representative branch videos.
6. Label all primary outputs `intrinsic future uncertainty` or
   `RNG-only branching`.
7. Move the external-noise and horizon sweep to a clearly labelled
   supplementary robustness section.
8. Preserve all existing outputs and write final results to a separate,
   versioned paper-ready directory.

## Current Artifacts

The completed sensitivity sweep and its current diagnostics are under:

```text
analysis/results/
paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/
c2_noise_horizon_sweep/
```

Published sweep figures and tables are under:

```text
analysis/results/
paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/
figures/c2_noise_horizon/
```
