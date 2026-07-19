# Parameter sweeps: branch tracking, per-value analysis, IC ensembles

Read this only when the paper's sweep is more than a **grid of independent points** —
a hysteresis / partial-synchronization branch, a continuation, a bifurcation diagram you
must *track*, or a per-value analysis like λ₁(K). A plain product grid over independent
cells needs none of this: an `Exploration` with `explored_values` / a `GridAxis` and the
vmapped grid path (see **running-simulations**) is enough.

## Track a multistable branch — don't cold-start each value

- **The trap.** At each swept value a cold start relaxes to the *dominant* attractor, so a
  partial-sync / hysteresis branch never appears — you get the async (or fully-sync) state
  everywhere and wrongly conclude the branch is absent.
- **The fix.** `Exploration.sweep_seeding: from_previous` carries each value's settled state
  into the next (the slow, adiabatic ramp). Add `sweep_direction: bidirectional` to ramp up
  then back down — the up/down mismatch *is* the hysteresis. `record: [...]` the settled
  observations at every value (and the full settled state if you will restart analysis on it).
- Warm-started points settle fast (you start near the branch), so only the first is expensive.
- **Caveat (delays).** The delay-history buffer is not carried across values, so warm-start is
  exact only for *instantaneous* coupling. For delayed coupling, pre-roll a τ-second transient
  at each value (`transient_time: τ`) instead of relying on the carried snapshot.

## Restart a per-value analysis over the recorded branch

- λ₁(K), an eigen-analysis, or any per-point measurement must be seeded from *that point's*
  settled branch state, not a cold start (a cold start measures the wrong attractor).
- `initial_state: {method: from_experiment, source_experiment: <sweep-id>, source_point: branch}`
  restarts the analysis at every recorded branch point.
- These points are **independent** (unlike the sequential scan that produced them) → shardable:
  `--shard i/N` or `workflow.slurm.array_chunk` splits the branch across tasks; results
  reassemble by branch *position*, so the up- and down-branches stay separable even where the
  swept value repeats.

## IC ensembles — which seed?

- **Two independent seeds, not interchangeable.** A state variable's `distribution.seed` sets
  the random **initial condition**; `execution.random_seed` sets the **noise** realization.
- To vary initial conditions across ensemble members, vary `distribution.seed` (per state
  variable). Varying `execution.random_seed` reseeds only noise — and in a deterministic
  (noiseless) run it changes *nothing*: a silent no-op that wastes the entire ensemble.
- Isolate each member under its own `--results-root` so a `from_experiment` branch restart
  reads *that* member's branch, not a sibling's (the `*exp-<id>_*.h5` source glob otherwise
  collides across members).
- Make each member's runner idempotent (skip a stage whose result already exists) so an
  overnight ensemble is restart-safe.

## Realization honesty — what averaging will and won't fix

- **Coarse structure reproduces** and should match: whether a branch exists, roughly where a
  transition sits, the qualitative bifurcation.
- **Seed-set fine structure does not** — the exact parameter where a chaotic λ₁ peaks, an exact
  solitary-node count — because it is fixed by the paper's unpublished seed. More ensemble
  members or a longer Benettin average will NOT move a chaotic peak onto the paper's sampled
  points; the offset is a realization width, not under-convergence.
- So report the branch/transition faithfully, count and compare **median-relative**, and state
  a seed-width offset as an accepted realization difference — don't burn compute chasing it.
  (A *magnitude* mismatch may instead be a unit/rescaling convention — confirm from the methods
  and Phase 7, per the main skill, before calling it physics.)
