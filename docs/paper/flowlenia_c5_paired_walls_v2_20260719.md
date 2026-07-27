# FlowLenia C5 paired walls protocol v2

## Question

Do optimized FlowLenia trajectories have greater frustration potential than
matched random controls?

At the same C2-selected divergence states, compare:

- a free 20,000-step C2 branch;
- a paired branch with hard compartment walls for steps 0--9,999 and the
  walls removed for steps 10,000--19,999.

The primary estimand is post-release divergence caused by the intervention,
after subtracting the matched free/free stochastic divergence floor.

## Pairing

- 10 optimization runs.
- One optimized and three matched random candidates per run.
- The optimized C2 trajectory selects five high-, five mid-, and five
  low-DeltaH states.
- The same 15 absolute steps and three branch seeds are reused for all four
  candidates in a run.
- Every pair shares the reconstructed source state, Gaussian perturbation,
  folded branch key, parameters, optimizer-native batch size/index, 50-step
  key schedule, and retained frame offsets.
- Horizon: 20,000 steps.
- Retained offsets: 0, 2,850, 5,700, 8,550, 11,400, 14,250, 17,100, 20,000.

Total: 10 x 4 x 15 x 3 = 1,800 paired branch rows.

## Wall intervention v2

The original v1 implementation is excluded from analysis. It called
`step_state` independently in nine blocks while mutation probability was
0.05, producing nine mutation opportunities instead of one. Its stochastic
reintegration also restarted a shape-dependent fixed-key categorical draw in
each 54x54 block.

Version 2 isolates compartment geometry:

1. Native block-local mutation is disabled.
2. For each optimizer-equivalent lane key, the exact single global 128x128
   mutation event is reconstructed with native key splits, probability,
   location, 40x40 patch, channel vector, and scale.
3. The single global mutation delta is center-padded to 132x132, partitioned
   into 3x3 44x44 cores, and added to corresponding block cores.
4. Native stochastic reintegration Gumbels from `PRNGKey(42)` are generated at
   global 128x128 shape. Corresponding spatial values are partitioned into
   block cores; blocks do not generate independent categorical noise.
5. Five padding cells around every core are hard-zeroed after every confined
   transition.
6. At relative step 10,000, cores are merged once and all later transitions
   use native global `step_state` with the unchanged next lane key.

The frozen configuration requires mutation size 40, probability 0.05, scale
1.0, `mix_rule=stoch`, volcano off, food off, mass decay zero, and mass
renormalization off.

## Required preflight

Mass wall simulation cannot start unless all checks pass:

- the native full-tracker sham exactly reproduces authoritative C2 A/P/F,
  lagrangian state, and RNG at all retained frames;
- native global mutation/RT equals mutation-disabled global dynamics plus the
  reconstructed mutation and externally supplied global Gumbels;
- that controlled no-wall replay is exact for 10,000 controlled steps plus
  10,000 native steps;
- global mutation and RT fields survive partition/merge bitwise;
- hard-wall padding is exactly zero;
- one-lane and frozen 30-lane wall steps are bitwise identical per lane;
- source parameters/configs and simulation code bundle match recorded hashes.

## Metrics

For each selected point:

- primary: median of three same-seed free/wall post-release CLIP Chamfer
  distances minus median of three free/free branch-pair distances;
- secondary: synchronized CLIP distance, full-horizon effect, wall/free
  ensemble spread, multiscale A/P RMS distances, separate A and P distances,
  and signed/absolute mass diagnostics.

Post-release uses the four retained frames after step 10,000. Candidate values
are medians over 15 points. A run effect is the optimized value minus the
median of its three random candidates. The statistical unit is the optimization
run (n=10); reported inference uses bootstrap median confidence intervals,
one-sided sign tests, and one-sided Wilcoxon tests.

The single pre-specified primary metric is the confirmatory test. The remaining
twelve reported metrics are secondary diagnostics; their p-values are explicitly
exploratory and unadjusted rather than treated as thirteen confirmatory tests.

Candidate-label permutation p-values are not reported because optimized and
random labels were neither randomized nor exchangeable.

## Locked identities and audits

The canonical v2 plan contains 1,800 rows and has SHA-256
`0843d44d7db4adb3eeb9a611bf929a6709be0f0482ec983eb1c41f206404b6c6`.
The frozen simulation-code bundle has SHA-256
`10480e6bd51446117f02c3687e3faa4632e0305bc6ef83359d66fde588dc9fbb`.

Before the full wall calculation, the following gates passed:

- all 1,800 free branches passed state, config, parameter, frame, and
  optimizer-native batch/JIT provenance checks without recomputation;
- 30 pre-cache random branches were replayed by the current batched runner and
  matched bitwise for every serialized field and the complete APF SHA-256;
- native and controlled 30-lane shams matched all authoritative captures with
  maximum absolute error zero and exact RNG;
- mutation injection, global reintegration Gumbels, partition roundtrips,
  frozen 30-lane topology, hard-wall padding, and 30 preflight wall outputs all
  passed their exact checks.

The analysis cache key includes the plan and source APF identities, exact
analysis/rendering code bundle, inference batch size, complete CLIP model and
image-processor configs, and a SHA-256 over every model parameter. Metric
tables, figures, and videos each have downstream manifests that bind them to
those inputs. Completion also decodes every video and verifies all artifact
hashes rather than accepting files by name or size.

## Outputs

Canonical root:

`analysis/results/paper_suite_flowlenia_lockheed_1_openai_es_fixed_init_10opt_c2_c5_paper/flow_lenia/c5_c2_paired_walls_half`

Version 2 wall branches are stored under
`walls_to_free_rng_matched`. Legacy v1 branches remain under
`walls_to_free` and cannot satisfy the v2 protocol/hash audit.

Final outputs include point/candidate/run tables, a LaTeX statistical table,
paper figures (PNG and PDF), 40 candidate videos, the full simulation protocol
audit, and the analysis completion audit.

## Final result

All production calculations and audits completed successfully.

- Simulation protocol audit: 1,800/1,800 free branches and 1,800/1,800
  walls-then-free branches passed, with no provenance failures.
- CLIP inference used the authoritative C2 unjitted single-frame path.
  All 450 optimized free branches matched the C2 reference caches
  (`max_abs=1.1920928955078125e-7`, tolerance `1e-5`).
- All 1,800 paired free/wall initial-frame embeddings were bitwise identical.
- The primary run-level effect was `0.0007481660671931414` (optimized minus
  matched-random median), with bootstrap 95% CI
  `[-0.0024892917611703397, 0.002593242756083136]`.
- Six of ten run effects were positive. The one-sided sign-test p-value was
  `0.376953125`; the one-sided Wilcoxon p-value was `0.384765625`.
  The pre-specified CLIP confirmatory analysis therefore does not reject its
  null.
- The secondary multiscale field effect was positive in eight of ten runs:
  median `0.3958832522233333`, bootstrap 95% CI
  `[0.09242183963457684, 0.6802272752547807]`, one-sided Wilcoxon
  `p=0.0322265625`. This remains exploratory and unadjusted.
- Nine required figure stems were emitted as both PNG and PDF, and all 40
  candidate videos passed full-frame decode and provenance checks.

Final identities:

- Analysis code identity:
  `8578a905f61ecf723deee69193de9bb09df7169d50e938a445a4fd5d64651cdd`
- CLIP model identity:
  `d1695538e471779612ea2f2e736d6ef3a5f43158a0e1588de9e962e3b39b2122`
- Simulation protocol audit SHA-256:
  `31d10fe9dc3989b6ffb92c709365089e9e3ed9d4970fa74d8052336aac4df295`
- Embedding protocol SHA-256:
  `73051005d0ca1cffa1c4a700b151c5d8525ef62d070e62640781ef11ef3f9252`
- Statistical summary SHA-256:
  `42cc1fe8cf52c0ee8df6888dbb4f909af3c5d4e999d8ebe9a2fb0c9d813b1f16`
- Completion audit SHA-256:
  `0fba5c840a08613b51f6f73a9aa2def0ac62fd3f7e72771705bec8afda7a79e8`
