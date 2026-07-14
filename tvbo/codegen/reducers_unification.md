# Unifying the sliding-window FC reducer with `Observation.dynamics` / `resolve_reduction`

## What the two engines actually are (read 2026-07-14)

**`resolve_reduction` + `render_reduction`** (utils.py + observation.mako:619, your WIP) —
a **co-integrated observer**: an `Observation.dynamics` whose state vars carry an `init` +
an `update` recurrence (`equation_type: recurrence`). `render_reduction` emits it as an
`(init, update, finalize)` triple that **folds over integration steps** — `_update` runs a
`jax.lax.scan` over a block, gating accumulators on `_gstep > skip` (transient). Consumed by
`prepare(reduce=…)` (grid, no trajectory held) and the host monitor. Handles mean (running
sum / count) and median (per-node histogram). **No eviction, no window, no resync.**

**Streaming FC reducer** (`streaming_reducers.py` + `reducers.py`, my work) — a
**sliding-window** reducer `(add, evict, emit, resync)` **driven manually in the algorithm
tuning loop**: each tuning iteration adds the arriving BOLD sample, evicts the leaving one,
and reads FC. It slides over the **BOLD ring buffer per tuning iteration**, not over
integration steps.

**They fold over different axes.** One reduces the integration scan (`(init,update,finalize)`
into `prepare(reduce=…)`); the other slides a window across tuning iterations. So this is NOT
"add `evict` and they're the same" — the *use sites* and *emission shapes* genuinely differ.

## Where they SHOULD unify (and where they shouldn't)

Shared and worth merging:
1. **The spec form.** Both are declarative recurrence specs. `Observation.dynamics`
   (state_variables + recurrence + output) is the schema home; my YAML
   (`state`/`add`/`evict`/`resync`/`emit`) is the same idea outside the schema. → The FC
   reducer should be an `Observation.dynamics`, authored once.
2. **The symbolic resolution.** `resolve_reduction` and `resolve_streaming_reducer` both
   sympify recurrence RHS against a vocabulary. `resolve_reduction` is the more complete one
   (unknown-symbol check, user functions, accumulator classification). → One resolver.

Distinct and should stay two emission modes of ONE partial:
3. **Emission.** `render_reduction` emits `(init, update, finalize)` (cumulative, for
   `prepare(reduce=…)`). A windowed observer additionally needs `(add, evict, emit, resync)`
   for the algorithm loop. → `render_reduction` gains a **windowed mode** selected by the
   observer declaring an `evict` recurrence (and a window/`resync`), reusing the same parsed
   state/output.

## Concrete changes

1. **Schema** (`schema/*.yaml`, then `make gen-linkml`): on the reduction `StateVariable`,
   add an optional second recurrence — `evict_equation` (or a role tag on a second
   `equation`) — the reverse update for the leaving sample. Optionally a `window`/`resync`
   marker on the observer (else `resync` is derived as the two-pass rebuild). The FC state
   `comoment`'s update is the Welford `add`; its `evict_equation` is the reverse-Welford
   downdate. Follows the linkml-schema skill; `tvbo/datamodel/**` is generated.
2. **`resolve_reduction`**: when a state carries `evict_equation`, parse it too (same vocab)
   and set `red['windowed'] = True`; carry `evict`/`resync`/`emit`(=output) alongside
   `update`. Cumulative path unchanged (no `evict` ⇒ `windowed=False`).
3. **`render_reduction`**: add a `% if red['windowed']:` branch emitting the
   `(add, evict, emit, resync)` protocol object (the generic scaffolding now in
   `algorithm.mako`) from the resolved exprs. The cumulative branch is untouched.
4. **`algorithm.mako`**: `streaming_map` resolves the source observation's reduction via
   `resolve_reduction` and emits via `render_reduction`'s windowed branch — deleting the
   `next(iter(streaming_map))['spec']` path.
5. **Author FC as `Observation.dynamics`** in `EI_Tuning_FIC_EIB_Optimization.yaml` (and the
   other online-tuning experiments), replacing the `pipeline: [compute_fc]` derived obs.
6. **Retire** `tvbo/codegen/streaming_reducers.py`, `tvbo/codegen/reducers.py`, and
   `tvbo/database/reducers/windowed_fc.yaml` — subsumed. Keep the array-op printer
   primitives (`code.py`) and parse vocab (`expression.py`) — they're general.

## Validation
Byte-identity on **both** paths, independently: the grid/`prepare(reduce=…)` path (existing
`Observation.dynamics` observers must be unchanged) AND the algorithm streaming path
(`test_experiment_runs[EI_Tuning_FIC_EIB_Optimization]`), each vs `TVBO_STREAMING_REDUCERS=0`
recompute.

## Recommendation
This is a schema + engine change spanning your active `resolve_reduction`/`render_reduction`
WIP. Do it on a **stable base** (once that WIP is committed) as one focused effort with the
two-path byte-identity gate above — not interleaved with the WIP. The array-op primitives and
the declarative-spec + general-resolver + general-template shape already landed are the
foundation this builds on; nothing here is blocked on new infrastructure, only on a clean base.
