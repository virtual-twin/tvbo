---
name: replicating-studies
description: "How to replicate a published study in TVBO \u2014 turn a paper into\
  \ ONE declarative, fully tvbo-native recipe (all or selected experiments) + simple\
  \ plotting + an honest, fully-computed report. Encodes the hard-won rules so the\
  \ replication is fast and trustworthy. Composes the atomic skills (writing-models,\
  \ running-simulations)."
---

# Replicating a study in TVBO

You are reproducing a published paper as a **single declarative TVBO recipe** plus
minimal plotting, with a report whose every number is computed from the run — never
typed by hand. This skill owns the *replication-specific* layer; for the atomic
how-to it defers to **writing-models** (Dynamics YAML), **running-simulations**
(sourcing / CLI / backends), and **codegen-templates** (render internals).

Work the phases in order. Each has a REQUIRED output. Do not skip ahead — the
scorecard in Phase 6 maps 1:1 to the criteria you write in Phase 1, and Phase 7
verification is what stops you trusting figures that silently integrate the wrong
attractor.

## The non-negotiables (MUST)

1. **ONE declarative recipe.** All targeted experiments live in one `<Study>.yaml`
   as metadata (anchors + `<<:` inheritance, `from_experiment` seeding). **No Python
   drivers.** `tvbo run <Study>.yaml` runs everything in dependency order; add
   `--experiment 2,3` to run a subset.
2. **Nothing hardcoded in the report.** Every reported value is computed inline from
   a result container (`output/nc/exp*/…h5`) or the recipe metadata — solitary counts,
   ⟨Δω⟩, decay times, N_c, topology counts, K/α/τ. If you typed a number into prose,
   it is a bug. (Papers are not ground truth; your own asserted numbers are not either.)
3. **Backend-independent metadata, backend chosen by fit.** The YAML states *intent*,
   never one backend's mechanism. The execution backend is picked in Phase 1.5 from the
   targets' feature needs, not defaulted.
4. **FAIR layout** (copy `assets/skeleton/`): `original_study/` (paper + analysis),
   `input/` (open data + `DATA.md`), `code/` (one prep script, analysis callables,
   one reference integration, `figures/plot.py`), `figures/`, `report/`. `output/` is
   gitignored (regenerable).
5. **Replication, stated honestly.** Frame it as *replication* (independent code +
   independently-sourced data → same conclusions), not bit-exact *reproduction*. Ship a
   **scorecard** (met / partial / out-of-scope) and name the **accepted limitations**
   (e.g. unpublished-seed realization dependence) up front.
6. **One plotting script**, `code/figures/plot.py`, one `main()` (topology → sweeps →
   control → compose). Simple matplotlib next to the recipe.
7. **Verify against an independent reference** (Phase 7) before trusting any figure.

---

## Phase 1 — Analyze the paper → `targets.md` + `figures.md`

Read the version of record (put it in `original_study/`, figures as `img/fig*.png`).
Produce two artifacts under `original_study/analysis/`:

- **`targets.md`** — a numbered table of replication targets `T1..Tn`. Each row:
  target · figure(s) · **key verbatim params** (copy them exactly — K, α, τ, seeds,
  transient/window times, step sizes as *printed*) · a **pass/fail validation
  criterion** · a feasibility/tier tag (`core` / `extended`).
- **`figures.md`** — per-figure panel map: panels, axes + ranges, colour convention,
  line styles, any quirks to reproduce as-is (mislabelled axes, unit conventions).

Watch for the trap that the *printed* equation is not the one the figures use (Taher's
Eq. 9 has a √N normalization typo; the figures use the plain std). Record the quantity
the *figures* actually show, with the discrepancy noted.

## Phase 1.5 — Scope, then backend-fit + gaps → `backend-fit.md`

**Scope.** Pick which targets to replicate: **all** (default) or a **selected subset**
(`{T1,T2,T7}`). Only selected targets become experiments in Phase 3. Use `/grill-me`
if the scope is contested.

**Backend-fit + gaps** (`original_study/analysis/backend-fit.md`). For the selected
targets, build a feature matrix (delays? Lyapunov/Benettin? adiabatic sweep? noise?
multi-mode? time-gated events? sparse coupling?) and pick the execution backend that
supports them — **with rationale**. tvboptim (JAX) is common because delays, Lyapunov
and adiabatic `lax.scan` sweeps are tvboptim-gated today; plain forward sims and
operating points run on any backend. **Surface feature gaps now**: a need not yet
supported (e.g. the Lyapunov exponent of a *delayed* closed loop under `vmap`) BLOCKS
its target — flag it as a framework/schema enhancement before you build, and mark the
target `partial`/`out` in the eventual scorecard. This early gap-finding is what sets
honest expectations instead of surprising you mid-YAML.

## Phase 2 — Source the data → `input/` + `DATA.md`

Prefer **open data**; carry it self-contained in `input/` (a zip is fine) and write
`input/DATA.md` from `assets/DATA.md.tmpl`: exact source (author, year, DOI, licence),
the sheet/column → paper-quantity map, checksums, and **which quantities are synthesised
vs sourced**. Name the true upstream source, never a derived intermediate. `code/`'s
prep script reads the raw open data and emits the tvbo-ready `Network` (+ small
CSV/npz for inspection); do NOT hand-edit derived artifacts.

## Phase 3 — The recipe: one tvbo-native `<Study>.yaml`

See **writing-models** for the Dynamics form and **running-simulations** for sourcing
(inline vs YAML vs `iri`) and the CLI. Replication-specific rules:

- One file; shared `&dynamics` / `&params` / `&network` anchors; per-experiment `<<:`
  overrides. Order experiments so a `from_experiment` source precedes its dependents
  (operating point before its control runs) — then bare `tvbo run <Study>.yaml`
  resolves the seeds in one pass.
- Non-obvious params get a one-line comment tying them to the paper (equation/figure).
- Overriding a param replaces it wholesale (YAML merge is shallow) — restate `unit`/
  `description`, or don't override when the anchor default already matches.
- Encode the *intent* declaratively (gates via a Piecewise/`autonomous:false` RHS,
  adiabatic branch via `Exploration.sweep_seeding: from_previous`, delayed self-terms
  via the coupling graph) — not a backend mechanism.

## Phase 4 — Analysis callables (only for non-closed-form pipelines)

Order parameters, solitary-node ordering, control masks, custom reductions → pure,
backend-agnostic, independently-testable functions in `code/<study>_analysis.py`,
referenced from the recipe via `callable: {name, module: <study>_analysis}`. Keep them
NumPy; carry data as **labelled xarrays**, never positional reshapes. When aligning a
paper's connectome/observable to your node order, match **by label**, never by position
(guards silent hemisphere/order swaps). Note the host/grid split: *declared* observations
run on the host (plain NumPy is fine); only what you put under `record:` runs inside the
jitted/vmapped grid and must be backend-traceable (a non-traceable recording raises).

## Phase 5 — Plotting: one `code/figures/plot.py`

Copy `assets/plot.py.tmpl` (one `main()`: topology → sweeps → control → compose) and
`assets/compose_ab.py` (pairs each reproduction with the paper original into
`ab_fig{N}.png`). Read the native result containers directly. Always draw the paper's
full multi-panel layout; a not-yet-run panel renders a labelled placeholder.

## Phase 6 — Report: `report/report.qmd` (every number computed)

Copy `assets/report.qmd.tmpl`. It carries the three things that took us the longest:

- a **metrics cell** that opens each result container and computes every quantity once
  into a dict `M`, referenced in prose/captions via inline `` `{python} …` `` (works in
  figure captions too). **No hand-typed numbers.**
- the **reproduction-vs-replication** section (NASEM framing) + a **scorecard** table
  (criterion → met / partial / out) mapping 1:1 to `targets.md`.
- model/coupling equations rendered from metadata via `EXP.dynamics.render("markdown")`
  and `coupling.render("markdown")` — generated, not transcribed. Render the controlled
  variant `relative to` the base (`dynamics.render(baseline=…)`) so only the delta shows.

PDF-targeted `.qmd`: write math as LaTeX, never Unicode (xelatex drops it). Avoid a
closing `$` immediately followed by a digit (breaks pandoc math). Run the finished prose
through the **anti-ai-slop** skill before you call it done.

## Phase 7 — Verify against an independent reference

Before trusting figures, validate the recipe's core dynamics against a **standalone
reference integration** of the paper's governing equation in `code/<study>_reference.py`
(plain NumPy, or another backend) — recipe output must match it (byte-exact, or within a
stated tolerance). Where feasible, also cross-check via `render_code('tvb')` vs
`render_code('tvboptim')`. This is what catches modelling bugs a PDF can't: e.g.
per-step vs per-stage coupling evaluation converging to a *different attractor*.

---

## Dynamical & numerical traps (these cost us the most time)

- **The integrator, not the physics, can move the attractor.** An explicit scheme
  (Heun/RK2) at too large a `dt` sustains lightly-damped fast librations at high
  coupling: the *time-averaged* spread climbs and reads like desynchronization, but it is
  numerical. Halve `dt` (or switch to RK4) and confirm the transition / operating point
  are unaffected. A paper's quoted "Δt" is often a Lyapunov / rescaling unit, **not** the
  solver step — do not copy it into `step_size`. Phase 7's reference integration is how
  you tell numerical drift from real dynamics.
- **Seeding a *delayed* system needs the delay HISTORY, not just a snapshot.** A
  `from_experiment` seed carries the state but not the τ seconds of history a delayed term
  reads; feedback engaging against an unfilled buffer spikes or fails to converge. Fill it
  with a τ-second **transient pre-roll** (`transient_time: τ`) that replays the operating
  point; the recorded onset is then `gate.t_on − transient_time` (derive it, never hardcode).
- **Big graphs: make the coupling sparse/vectorized before reaching for HPC.** For an
  N-node grid the per-step dense N×N coupling matmul dominates; `network.graph_representation:
  sparse` (with a factored/angle-addition coupling) turned a multi-hour sweep into minutes
  locally, numerically identical (~1e-16). Assess this first — often no cluster is needed.

## Pitfalls we hit (so you don't)

- **A metric's *definition* is part of the claim.** t_c (1/e vs exponential-fit),
  ⟨Δω⟩ (std about the mean vs the median), λ₁ units — pick a documented definition, state
  it, and compute it. A magnitude that differs from the paper may be a unit/rescaling
  convention rather than a physics gap — but **confirm that from the methods**, don't
  assume it (we labelled it "likely" and it stayed unverified).

- **Coupling evaluated once per step** silently integrates a different, multistable
  attractor. Use `Integrator.coupling_evaluation: per_stage` for chaotic/multistable
  systems and verify against the reference (Phase 7).
- **Hardcoded fidelity numbers** creep into captions ("t_c ≈ 2.6 s") and read as
  matches when they aren't. Compute them (Phase 6). A recomputed value that *differs*
  from the paper is honest; a typed one that matches is not.
- **Realization dependence.** Exact solitary counts / magnitudes depend on unpublished
  seeds — count median-relative, state the difference as an accepted limitation, don't
  chase the integer.
- **Redundant scripts.** One prep script (emits the tvbo Network directly), one plot
  script. Don't split what one `main()` can do.
- **Cross-references.** The report must stand alone — no "as in the sibling X study".
- **Framework gaps surface late** if you skip Phase 1.5. Find them before the YAML.
