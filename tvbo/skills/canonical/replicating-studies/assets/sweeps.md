# Parameter sweeps: branch tracking, per-value analysis, IC ensembles

Read this only when the paper's sweep is more than a **grid of independent points** — a hysteresis / partial-synchronization branch, a continuation, a bifurcation diagram you must *track*, or a per-value analysis like λ₁(K). A plain product grid over independent cells needs none of this: an `Exploration` with `explored_values` / a `GridAxis` and the vmapped grid path (see **running-simulations**) is enough.

## Track a multistable branch — don't cold-start each value

- **The trap.** At each swept value a cold start relaxes to the *dominant* attractor, so a partial-sync / hysteresis branch never appears — you get the async (or fully-sync) state everywhere and wrongly conclude the branch is absent.
- **The fix.** `Exploration.sweep_seeding: from_previous` carries each value's settled state into the next (the slow, adiabatic ramp). Add `sweep_direction: bidirectional` to ramp up then back down — the up/down mismatch *is* the hysteresis. `record: [...]` the settled observations at every value (and the full settled state if you will restart analysis on it).
- Warm-started points settle fast (you start near the branch), so only the first is expensive.
- **Caveat (delays).** The delay-history buffer is not carried across values, so warm-start is exact only for *instantaneous* coupling. For delayed coupling, pre-roll a τ-second transient at each value (`transient_time: τ`) instead of relying on the carried snapshot.

## Restart a per-value analysis over the recorded branch

- λ₁(K), an eigen-analysis, or any per-point measurement must be seeded from *that point's* settled branch state, not a cold start (a cold start measures the wrong attractor).
- `initial_state: {method: from_experiment, source_experiment: <sweep-id>, source_point: branch}` restarts the analysis at every recorded branch point.
- These points are **independent** (unlike the sequential scan that produced them) → shardable: `--shard i/N` or `workflow.slurm.array_chunk` splits the branch across tasks; results reassemble by branch *position*, so the up- and down-branches stay separable even where the swept value repeats.

## IC ensembles — deterministic sweep or random seed?

- **First ask: does the paper vary ICs on a grid, or draw them at random?** A figure that shows an *evenly-spaced fan* of trajectories from initial conditions on a linspace is a deterministic grid, not a random draw — reproduce it with a deterministic **`initial_conditions.<state_var>`** exploration axis (`parameter: initial_conditions.E`, `domain: {lo, hi, n}` or `explored_values`), which sweeps one state variable's initial value and yields one trajectory per value, keyed as a first-class `initial_conditions.E` result dim. This matches the paper's exact construction (`linspace(lo, hi, n)`), is reproducible run to run, and needs no seed. Prefer it whenever the paper's ICs are deterministic. The swept SV must NOT also carry a `distribution` — that would resample and overwrite the swept value, so codegen rejects the combination. Use the random-seed path below only for a genuinely *stochastic* IC ensemble.

- **Two independent seeds, not interchangeable.** A state variable's `distribution.seed` sets the random **initial condition**; `execution.random_seed` sets the **noise** realization.
- To vary initial conditions across ensemble members, vary `distribution.seed` (per state variable). Varying `execution.random_seed` reseeds only noise — and in a deterministic (noiseless) run it changes *nothing*: a silent no-op that wastes the entire ensemble.
- Isolate each member under its own `--results-root` so a `from_experiment` branch restart reads *that* member's branch, not a sibling's (the `*exp-<id>_*.h5` source glob otherwise collides across members).
- Make each member's runner idempotent (skip a stage whose result already exists) so an overnight ensemble is restart-safe.

## A landscape whose every point is ONE draw states a realization as a fit

A grid search that runs one simulation per grid point gives a curve whose HEIGHT is a noise draw. Every cell normally shares one `execution.random_seed`, so the curve comes out *smooth*, which reads as a converged landscape and hides the problem completely. Before you believe any level read off such a curve — or call a disagreement with the paper's version of it a finding — measure the realization width: re-run ONE grid point under N independent seeds.

The width differs enormously between metrics, and that alone decides which of them may be compared across studies at all. Pang2023 at the fitted `r_s`, ten seeds: edge FC r 0.390 ± 0.020, FCD KS 0.069 ± 0.021, node FC r **0.451 ± 0.165**. Node FC's spread is nearly the entire vertical range of the published curve, because 180 node strengths do not average a draw away where ~16,000 edges do. Two single-draw node-FC numbers can therefore differ by 0.2 and agree perfectly — which is exactly what looked like a 0.25 disagreement with the published landscape and was not one: all three of that paper's headline values sit within one realization sd of our ensemble mean.

Two consequences for the recipe:

- **Sweep the seed as a second exploration axis** and plot the ensemble mean with a ±1 sd `mark: band` (see `figures.md`), not one line per metric. When the metrics differ in width the band IS the result, and three clean lines assert a precision none of them has.
- **Take the optimum on the MEAN curve.** The argmin of a single noisy curve is a *selected* minimum, biased toward the tail that selected it: Pang2023's single-draw argmin returns KS 0.029 where the ten-seed mean at the same `r_s` is 0.069 — and the published values are 0.06–0.08, i.e. the ensemble mean agrees and the single draw looks spuriously better.

**Then show it, don't just say it.** A published number quoted beside yours reads as agreement or as disagreement depending only on which draw yours was. Locate it *inside* your ensemble instead: one analysis per metric returning the density plus `mean`, `sd`, `q05`/`q95`, the paper's `reference` (declared in the recipe as an argument, quoted and never recomputed) and its distance from your mean in sd. The figure draws the density with `mark: area` and the two lines with `mark: rule`; the report table reads the same container, so the marked line and the quoted z cannot disagree. Declare the histogram RANGE rather than deriving it, so the two arms of a metric share an x-axis, and raise if a value falls outside it — `np.histogram` silently drops what its `range` excludes, which would narrow the very distribution the panel exists to show.

## Realization honesty — what averaging will and won't fix

- **Coarse structure reproduces** and should match: whether a branch exists, roughly where a transition sits, the qualitative bifurcation.
- **Seed-set fine structure does not** — the exact parameter where a chaotic λ₁ peaks, an exact solitary-node count — because it is fixed by the paper's unpublished seed. More ensemble members or a longer Benettin average will NOT move a chaotic peak onto the paper's sampled points; the offset is a realization width, not under-convergence.
- So report the branch/transition faithfully, count and compare **median-relative**, and state a seed-width offset as an accepted realization difference — don't burn compute chasing it. (A *magnitude* mismatch may instead be a unit/rescaling convention — confirm from the methods and Phase 7, per the main skill, before calling it physics.)
