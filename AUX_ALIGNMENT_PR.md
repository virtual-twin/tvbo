# Align recorded auxiliaries with the recorded state

## Problem

The native (tvboptim) solver records each row's **state** as the post-step value but its
**auxiliaries** as evaluated at the *step-start* state (Heun/Euler both return the
first-stage aux). A recorded derived variable therefore lags the recorded state by one
integration step: `x[t] == L·sin(θ[t-1])` instead of `L·sin(θ[t])`. This affects every
recorded derived variable — firing rates, sigmoids, `x²+y²`, input currents — whenever
they are read at per-timestep resolution.

Ground truth (`tests/test_auxiliary_state_alignment.py`, stock tvboptim): pendulum
state-only alignment failed for Heun and Euler; the coupling-dependent per_stage case
failed; the default per_step (frozen coupling) correctly lagged.

## Fix — split by where the information lives

**State-only derived variables → tvbo codegen (this PR).** A derived variable that is a
pure function of the state is recomputed from the recorded post-step state after the
solve, removing the lag. Backend-independent, works against the pinned tvboptim.

- `tvbo/templates/tvboptim/utils.py::state_only_recorded_aux` — classifies recorded
  derived variables as state-only (expression, expanded through derived
  variables/parameters, references no coupling/external input). Conservative: anything
  unresolved is left to the solver.
- `tvbo/templates/tvboptim/tvbo-tvboptim-experiment.py.mako` — emits
  `_realign_state_auxiliaries()` and applies it to the main-sim result. Single-mode only.

**Coupling-dependent derived variables → tvboptim solver (companion branch).** Their
value at the recorded state needs the coupling *at that instant*, which only exists inside
the solver scan (per_stage re-evaluation, delay history). Fixed in the solver by
re-evaluating auxiliaries at the post-step state; carries the coupling, so a
coupling-dependent aux aligns under per_stage and lags under per_step.

- tvboptim branch `feat/auxiliary-state-alignment`, `network_dynamics/solve.py`. Gated on
  `record_auxiliaries` (runs recording none are byte-identical).

## Tests

`tests/test_auxiliary_state_alignment.py`: pendulum (Heun/Euler) and per_step pass against
pinned tvboptim. `test_coupling_dependent_aux_aligns_under_per_stage` is `xfail`
(`strict=False`) pending the tvboptim branch — it xpasses once that lands and the pin is
bumped, at which point the marker (and, optionally, the now-redundant codegen recompute)
can be removed.

## Notes

- State channels are never touched → simulation trajectories stay byte-identical.
- Multi-mode and non-integration paths (exploration/algorithm/optimization) keep the
  solver's behavior; the tvboptim branch is the single home once released.
